"""Exact feature order from the private project's prediction service.

This module contains a schema only, NOT patient records or clinical advice.
"""
import pandas as pd

FEATURE_ORDER = [
    "Onset to door time",
    "Hypodensity signs",
    "Sedentary lifestyle",
    "normal CT scan",
    "Early ischemic changes",
    "Diastolic blood pressure",
    "Previous stroke",
    "Initial NIHSS",
    "Systolic blood pressure",
    "INR",
    "Age",
    "Gender",
]


def to_model_frame(features: dict) -> pd.DataFrame:
    """Construct the single-row input exactly as in the original inference service.

    Reject missing or unexpected fields instead of silently reordering or
    accepting incomplete inputs. Does not perform clinical plausibility checks.
    """
    missing = [name for name in FEATURE_ORDER if name not in features]
    extra = sorted(set(features) - set(FEATURE_ORDER))
    if missing or extra:
        raise ValueError(f"Feature mismatch: missing={missing}, extra={extra}")
    return pd.DataFrame([features], columns=FEATURE_ORDER)
