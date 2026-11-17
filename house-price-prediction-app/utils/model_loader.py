from pathlib import Path
import joblib
import sklearn

APP_DIR = Path(__file__).resolve().parent.parent
PROJECT_DIR = APP_DIR.parent
MODEL_FILENAME = "house_price_prediction_model.pkl"

MODEL_SEARCH_PATHS = [
    APP_DIR / MODEL_FILENAME,
    APP_DIR / "models" / MODEL_FILENAME,
    PROJECT_DIR / MODEL_FILENAME,
    PROJECT_DIR / "models" / MODEL_FILENAME,
]
MODEL_PATH = next((p for p in MODEL_SEARCH_PATHS if p.is_file() and p.stat().st_size > 0), None)

def load_model():
    if MODEL_PATH is None:
        checked = "\n".join(f" - {p}" for p in MODEL_SEARCH_PATHS)
        raise FileNotFoundError(
            f"A non-empty {MODEL_FILENAME} was not found. Checked:\n{checked}"
        )
    try:
        model = joblib.load(MODEL_PATH)
    except Exception as exc:
        raise RuntimeError(
            f"Found model at {MODEL_PATH}, but it could not be loaded. "
            f"Installed scikit-learn: {sklearn.__version__}. "
            f"Original error: {type(exc).__name__}: {exc}"
        ) from exc
    if not hasattr(model, "predict"):
        raise TypeError(f"Loaded object {type(model).__name__} has no predict() method.")
    return model
