"""AI-only inference extracted from the private project's prediction path.

The trained model is *not* distributed in this public repository. Supply a
locally held, trusted artifact using THROMBOLYSIS_MODEL_PATH.
"""
from functools import lru_cache
from pathlib import Path
import os

from .features import to_model_frame


@lru_cache(maxsize=1)
def load_model():
    path_value = os.environ.get("THROMBOLYSIS_MODEL_PATH")
    if not path_value:
        raise RuntimeError(
            "Set THROMBOLYSIS_MODEL_PATH to the authorized, local trained model"
        )
    model_path = Path(path_value).expanduser()
    if not model_path.is_file():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    # Joblib loads Python pickle objects and can execute code. Trust source.
    import joblib
    return joblib.load(model_path)


def predict_thrombolysis(features: dict):
    """Return the trained model's label and positive-class probability.

    Preserves the inference behavior in the original project's
    fastapi/services/predictor.py without its auth, logging or mobile backend.
    """
    model = load_model()
    input_df = to_model_frame(features)
    prediction = model.predict(input_df)[0]
    probability = None
    if hasattr(model, "predict_proba"):
        probability = float(model.predict_proba(input_df)[0][1])
    return prediction, probability
