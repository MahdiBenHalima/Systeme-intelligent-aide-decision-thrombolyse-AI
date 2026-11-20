"""Inference entry point requiring a locally stored, trusted model artifact.

Usage:
  export THROMBOLYSIS_MODEL_PATH="/private/path/to/modele.joblib"
  python -m clinical_ml.predict --input /private/path/to/features.json

The input JSON must contain all 12 model features. Do not commit real records.
The output is a research model prediction, not a clinical recommendation.
"""
import argparse
import json
from .inference import predict_thrombolysis


def main():
    parser = argparse.ArgumentParser(description="Local thrombolysis ML inference")
    parser.add_argument("--input", required=True, help="Path to private JSON features")
    args = parser.parse_args()

    with open(args.input, encoding="utf-8") as stream:
        data = json.load(stream)
    if not isinstance(data, dict):
        parser.error("Input must be a JSON object of named features")
    label, probability = predict_thrombolysis(data)
    print(json.dumps({
        "predicted_label": bool(label),
        "positive_class_probability": probability,
        "disclaimer": "Research only. Not validated for clinical use.",
    }))


if __name__ == "__main__":
    main()
