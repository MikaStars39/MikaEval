import logging
import os

import src.data.tasks as tasks
from src.data.tasks import DATASETS
from src.data.template import apply_template_to_jsonl

def prepare_pass_at_k_jsonl(
    config_str: str, 
    output_file: str, 
    cache_dir: str = None
):
    """
    Parses config_str (e.g., 'aime2024@32,math500@4') and generates a JSONL file 
    where each question is repeated k times for Pass@k sampling.
    """
    dataset_configs = []
    for item in config_str.split(","):
        name, k_val = item.split("@")
        dataset_configs.append((name.strip(), int(k_val.strip())))

    out_dir = os.path.dirname(output_file)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f_out:
        for ds_name, k in dataset_configs:

            logging.info(f"Processing {ds_name} (repeat {k} times)...")
            
            loader_name = f"load_{ds_name.replace('-', '_')}"
            loader = getattr(tasks, loader_name, None)
            if loader is None:
                raise ValueError(
                    f"Could not find loader '{loader_name}' for dataset '{ds_name}'. "
                    f"Please implement '{loader_name}(dataset_name, cache_dir, k, f_out)'."
                )
            loader(ds_name, cache_dir, k, f_out)

    logging.info(f"Successfully generated {output_file}.")

if __name__ == "__main__":
    # The input configuration string
    config_str = "aime2024@32,aime2025@32,amc2023@32,math500@4,minerva@4,hmmt2025@32"
    output_file = "outputs/debug/prepared_inference_data.jsonl"
    cache_dir = "/mnt/llm-train/users/explore-train/qingyu/.cache"
    
    prepare_pass_at_k_jsonl(
        config_str=config_str, 
        output_file=output_file, 
        cache_dir=cache_dir,
    )