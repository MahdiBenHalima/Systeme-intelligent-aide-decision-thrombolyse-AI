"""Reproducible SOFTWARE DEMO using synthetic, NON-CLINICAL data.

This does NOT reconstruct the original patient cohort or published research
results. The genuine trained model is used only through clinical_ml/inference.py.
"""
from __future__ import annotations

import argparse
import json

from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_estimators(seed: int = 42):
    """Keep preprocessing inside the CV pipeline to prevent train-fold leakage."""
    return {
        "logistic_regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=500, random_state=seed)),
        ]),
        "random_forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", RandomForestClassifier(
                n_estimators=80, min_samples_leaf=2, random_state=seed, n_jobs=1
            )),
        ]),
    }


def run_demo(n_samples: int = 240, seed: int = 42):
    """Train-only cross-validation, untouched test split, no stored patient records."""
    if n_samples < 100:
        raise ValueError("At least 100 synthetic samples required")
    X, y = make_classification(
        n_samples=n_samples,
        n_features=12,
        n_informative=7,
        n_redundant=2,
        n_classes=2,
        flip_y=0.06,
        class_sep=0.8,
        random_state=seed,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=seed
    )
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    report = {
        "dataset": "sklearn.make_classification (synthetic, non-clinical)",
        "clinical_validation": False,
        "original_study_reproduction": False,
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "seed": seed,
        "models": {},
    }
    for name, estimator in build_estimators(seed).items():
        scores = cross_val_score(estimator, X_train, y_train, cv=cv, scoring="roc_auc")
        estimator.fit(X_train, y_train)
        predictions = estimator.predict(X_test)
        probabilities = estimator.predict_proba(X_test)[:, 1]
        report["models"][name] = {
            "train_cv_auc_mean": round(float(scores.mean()), 4),
            "train_cv_auc_std": round(float(scores.std()), 4),
            "synthetic_holdout_auc": round(float(roc_auc_score(y_test, probabilities)), 4),
            "synthetic_holdout_accuracy": round(float(accuracy_score(y_test, predictions)), 4),
            "synthetic_holdout_recall": round(float(recall_score(y_test, predictions)), 4),
            "synthetic_holdout_f1": round(float(f1_score(y_test, predictions)), 4),
        }
    return report


def main():
    parser = argparse.ArgumentParser(description="Synthetic software-only ML evaluation demo")
    parser.add_argument("--samples", type=int, default=240)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(run_demo(args.samples, args.seed), indent=2))


if __name__ == "__main__":
    main()
