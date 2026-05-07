from typing import Any, Optional
from pathlib import Path
import sys
import joblib

# Import custom transformers so they are available during unpickling.
# The trained pipeline referenced `BinaryMapper` defined in the training
# environment; ensure the same name is present in `__main__` so pickle can
# find it when joblib loads the file.
try:
    from src.preprocess import BinaryMapper  # type: ignore
except Exception:
    BinaryMapper = None

# If necessary, inject the BinaryMapper into the __main__ module so
# unpickling can resolve the reference.
if BinaryMapper is not None:
    try:
        import __main__ as _main

        if not hasattr(_main, "BinaryMapper"):
            setattr(_main, "BinaryMapper", BinaryMapper)
    except Exception:
        # Not fatal; joblib.load may still succeed if the pickle references
        # a fully-qualified module path. We'll attempt to load regardless.
        pass

# Global container for loaded model
MODEL: Optional[Any] = None

# Default model path (relative to repo root). Adjust if you use a different filename/path.
DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[1] / "model" / "social_model_v1.joblib"
    

def load_model(path: Optional[Path] = None) -> Any:
    """Load the sklearn pipeline from disk and cache it in the module global.

    Args:
        path: Optional Path to the joblib file. If None, DEFAULT_MODEL_PATH is used.

    Returns:
        The loaded model object.
    """
    global MODEL
    p = path or DEFAULT_MODEL_PATH
    MODEL = joblib.load(p)
    return MODEL


def get_model() -> Any:
    """Return loaded model, loading it if necessary."""
    global MODEL
    if MODEL is None:
        return load_model()
    return MODEL
