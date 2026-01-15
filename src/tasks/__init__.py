

"""
`src.data.tasks` is a convenience namespace that:
- exposes the shared dataset registry / helpers
- imports all `load_*` dataset loaders so callers can `getattr(tasks, "load_xxx")`

Important: loader modules must import shared helpers from `src.data.tasks.base`
to avoid circular imports during package initialization.
"""

from src.data.tasks.base import DATASETS, get_answer_text, get_question_text, load_dataset_from_hf

# Import loaders (kept at module scope so they're available via `src.data.tasks.load_xxx`)
from src.data.tasks.aime2024 import load_aime2024  # noqa: F401
from src.data.tasks.aime2025 import load_aime2025  # noqa: F401
from src.data.tasks.amc2023 import load_amc2023  # noqa: F401
from src.data.tasks.hmmt2025 import load_hmmt2025  # noqa: F401
from src.data.tasks.math500 import load_math500  # noqa: F401
from src.data.tasks.minerva import load_minerva  # noqa: F401
from src.data.tasks.mmlu_pro import load_mmlu_pro  # noqa: F401

__all__ = [
    "DATASETS",
    "get_question_text",
    "get_answer_text",
    "load_dataset_from_hf",
    "load_aime2024",
    "load_aime2025",
    "load_amc23",
    "load_amc2023",
    "load_math500",
    "load_minerva",
    "load_hmmt25",
    "load_hmmt2025",
    "load_mmlu_pro",
]