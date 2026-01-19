import json
import asyncio
import os
import sglang as sgl
import logging
import time
import subprocess
from tqdm.asyncio import tqdm
from typing import Optional, Tuple, Any, Dict, Set, List

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("BatchInference")

class BatchInferenceEngine:
    """
    A robust, async wrapper around SGLang for high-throughput offline inference.
    Supports resume, crash recovery, and detailed monitoring.
    """

    def __init__(
        self,
        model_path: str,
        dp_size: int = 1,
        tp_size: int = 1,
        mem_fraction_static: float = 0.90,
        max_inflight: int = 512,
        # Speculative Decoding Config
        speculative_algorithm: Optional[str] = None,
        speculative_draft_model_path: Optional[str] = None,
        speculative_num_steps: Optional[int] = None,
        speculative_eagle_topk: Optional[int] = None,
        speculative_num_draft_tokens: Optional[int] = None,
    ):

        # ------------------------- Configuration ------------------------

        self.model_path = model_path
        self.dp_size = dp_size
        self.tp_size = tp_size
        self.mem_fraction_static = mem_fraction_static
        self.max_inflight = max_inflight
        
        # -------------------------  Speculative params ------------------------- 

        self.spec_args = {
            "speculative_algorithm": speculative_algorithm,
            "speculative_draft_model_path": speculative_draft_model_path,
            "speculative_num_steps": speculative_num_steps,
            "speculative_eagle_topk": speculative_eagle_topk,
            "speculative_num_draft_tokens": speculative_num_draft_tokens,
        }

        # ------------------------- Runtime State ------------------------

        self.llm: Optional[sgl.Engine] = None
        self.input_queue: Optional[asyncio.Queue] = None
        self.output_queue: Optional[asyncio.Queue] = None
        
    # ------------------------- Lifecycle Management (Context Manager) ------------------------

    async def __aenter__(self):
        """Initializes the SGLang Engine when entering 'async with' block."""

        os.environ["SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN"] = "1"
        logger.info(f"Initializing Engine (DP={self.dp_size}, TP={self.tp_size})...")
        
        self.llm = sgl.Engine(
            model_path=self.model_path,
            dp_size=self.dp_size,
            tp_size=self.tp_size,
            mem_fraction_static=self.mem_fraction_static,
            log_level="error", # Keep SGLang quiet to not mess up tqdm
            disable_radix_cache=True,
            trust_remote_code=True,
            **self.spec_args
        )
        
        # Initialize queues based on concurrency limit
        self.input_queue = asyncio.Queue(maxsize=self.max_inflight * 2)
        self.output_queue = asyncio.Queue(maxsize=self.max_inflight * 2)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Ensures the Engine shuts down gracefully on exit or error."""
        if self.llm:
            logger.info("Shutting down Engine...")
            self.llm.shutdown()
        if exc_val:
            logger.error(f"Engine exited with error: {exc_val}")

    # ------------------------- Core Helpers ------------------------

    @staticmethod
    def _count_lines_fast(file_path: str) -> int:
        """
        Counts lines using system 'wc -l' for speed, with Python fallback.
        """
        if not os.path.exists(file_path):
            return 0
        try:
            # Uses 'wc -l' which is extremely fast on Linux/Mac
            result = subprocess.run(['wc', '-l', file_path], capture_output=True, text=True)
            return int(result.stdout.split()[0])
        except Exception:
            # Fallback for Windows or if wc fails
            count = 0
            with open(file_path, 'r', encoding='utf-8') as f:
                for _ in f: 
                    if _.strip(): count += 1
            return count

    def _extract_stats(self, output: Dict[str, Any]) -> Dict[str, int]:
        """Extracts token counts safely."""
        meta = output.get("meta_info") or output.get("usage") or {}
        pt = meta.get("prompt_tokens") or meta.get("input_tokens") or 0
        ct = meta.get("completion_tokens") or meta.get("output_tokens") or 0
        tt = meta.get("total_tokens") or (pt + ct)
        return {"prompt": int(pt), "completion": int(ct), "total": int(tt)}

    async def _generate_safe(self, prompt: str, params: dict) -> Dict[str, Any]:
        """Wraps generation with a fallback mechanism for incompatible params."""
        try:
            return await self.llm.async_generate(prompt, params)
        except Exception as e:
            # Fallback: Drop keys that might cause issues (e.g. stop sequences on some backends)
            drop_keys = {
                "stop", "stop_token_ids", "repetition_penalty", 
                "frequency_penalty", "presence_penalty", "min_new_tokens"
            }
            filtered = {k: v for k, v in params.items() if k not in drop_keys}
            
            if filtered == params:
                raise e # If filtering didn't change anything, it's a hard error
            
            logger.warning(f"Retrying without optional params due to: {e}")
            return await self.llm.async_generate(prompt, filtered)

    # ------------------------- Async Tasks (Producer / Worker / Writer) ------------------------

    async def _producer(self, input_file: str, existing_ids: Set[str]):
        """Reads input file and populates input_queue."""
        logger.info("Producer: Started reading file...")
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                try:
                    data = json.loads(line)
                    # Resume logic
                    if existing_ids and data.get("id") in existing_ids:
                        continue
                    await self.input_queue.put(data)
                except json.JSONDecodeError:
                    pass # Skip bad lines
        
        # Send poison pills to stop workers
        for _ in range(self.max_inflight):
            await self.input_queue.put(None)
        logger.info("Producer: Finished.")

    async def _worker(self, sampling_params: dict):
        """Consumes inputs, runs inference, pushes to output."""
        while True:
            item = await self.input_queue.get()
            if item is None:
                self.input_queue.task_done()
                break
            
            try:
                start_t = time.perf_counter()
                output = await self._generate_safe(item["prompt"], sampling_params)
                duration = time.perf_counter() - start_t
                
                # Attach results
                item["response"] = output["text"]
                item["_stats"] = self._extract_stats(output)
                item["_latency"] = duration
                
                await self.output_queue.put(item)
            except Exception as e:
                logger.error(f"Worker Error: {e}")
            finally:
                self.input_queue.task_done()

    async def _writer(self, output_file: str, total_items: int, resume: bool):
        """Consumes results and writes to disk."""
        logger.info("Writer: Started...")
        
        mode = "a" if resume else "w"
        pbar = tqdm(total=total_items, desc="Inference")
        
        acc_tokens = 0
        start_global = time.perf_counter()
        
        with open(output_file, mode, encoding='utf-8') as f_out:
            while True:
                result = await self.output_queue.get()
                if result is None:
                    self.output_queue.task_done()
                    break
                
                # Update Stats
                stats = result.pop("_stats", {})
                result.pop("_latency", None)
                acc_tokens += stats.get("total", 0)
                
                # Write
                f_out.write(json.dumps(result, ensure_ascii=False) + "\n")
                
                # Update UI
                pbar.update(1)
                elapsed = max(1e-9, time.perf_counter() - start_global)
                pbar.set_postfix({
                    "tok/s": f"{acc_tokens / elapsed:.1f}",
                    "it/s": f"{pbar.n / elapsed:.1f}"
                })
                
                self.output_queue.task_done()
        
        pbar.close()

    # ------------------------- Main Entry Point ------------------------

    async def run(self, input_file: str, output_file: str, sampling_params: dict, resume: bool = False):
        """
        Orchestrates the pipeline: Producer -> Workers -> Writer.
        """
        # ------------- 1. Pre-check: Load existing IDs for resume -------------
        existing_ids = set()
        if resume and os.path.exists(output_file):
            logger.info("Scanning output file for resume...")
            with open(output_file, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        obj = json.loads(line)
                        if "id" in obj: existing_ids.add(obj["id"])
                    except: pass
            logger.info(f"Resuming: Skipped {len(existing_ids)} items.")

        # ------------- 2. Pre-check: Count workload -------------
        total_lines = self._count_lines_fast(input_file)
        remaining_lines = max(0, total_lines - len(existing_ids))
        
        if remaining_lines == 0 and total_lines > 0:
            logger.info("Nothing to do.")
            return

        # ------------- 3. Create Tasks -------------
        producer_task = asyncio.create_task(
            self._producer(input_file, existing_ids)
        )
        
        worker_tasks = [
            asyncio.create_task(self._worker(sampling_params))
            for _ in range(self.max_inflight)
        ]
        
        writer_task = asyncio.create_task(
            self._writer(output_file, remaining_lines, resume)
        )

        # ------------- 4. Wait for completion sequence -------------
        await producer_task
        await self.input_queue.join()  # Wait for workers to finish processing inputs
        
        # Signal writer to stop
        await self.output_queue.put(None)
        await writer_task
        
        logger.info(f"Done. Output saved to {output_file}")


if __name__ == "__main__":
    
    # Define params
    sampling_params = {
        "temperature": 0.6,
        "top_p": 0.9,
        "max_new_tokens": 2048
    }

    INPUT_FILE = "data.jsonl"
    OUTPUT_FILE = "results.jsonl"
    MODEL = "/path/to/model"

    async def main():
        # Initialize the Engine Context
        engine_config = dict(
            model_path=MODEL,
            dp_size=8,
            max_inflight=512, # Concurrency
            speculative_algorithm="EAGLE3", # Optional: Example of spec decoding
            speculative_draft_model_path="path/to/eagle",
        )

        async with BatchInferenceEngine(**engine_config) as engine:
            await engine.run(
                input_file=INPUT_FILE,
                output_file=OUTPUT_FILE,
                sampling_params=sampling_params,
                resume=True
            )

    # Run the event loop
    asyncio.run(main())