from pathlib import Path
from datasets import load_dataset
import logging

DATASETS = {
    "aime2024": ("HuggingFaceH4/aime_2024", "train"),
    "aime2025": ("yentinglin/aime_2025", "train"),
    "amc2023": ("zwhe99/amc23", "test"),
    "math500": ("HuggingFaceH4/MATH-500", "test"),
    "minerva": ("math-ai/minervamath", "test"),
    "hmmt2025": ("FlagEval/HMMT_2025", "train"),
}


def load_dataset_from_hf(dataset_name: str, cache_dir: str = None):
    if dataset_name in DATASETS:
        hf_name, split = DATASETS[dataset_name]
        if cache_dir is not None:
            # Use the HuggingFace dataset name to construct cache path
            cache_dataset_name = hf_name.split("/")[-1]  # Extract dataset name from HF path
            cache_path = Path(cache_dir) / cache_dataset_name
            if cache_path.exists():
                # Load from local cache directory
                try:
                    return load_dataset(str(cache_path), split=split)
                except AttributeError as e:
                    if "'tqdm' has no attribute '_lock'" in str(e):
                        # Handle tqdm threading issue by falling back to HF loading
                        print(f"Warning: Cache loading failed due to tqdm threading issue, falling back to HF: {e}")
                        return load_dataset(hf_name, split=split)
                    else:
                        raise
        # Fall back to HuggingFace
        return load_dataset(hf_name, split=split)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")