"""Software reproducibility checks; absolutely no patient data or trained artifacts."""
from experimental_ml.demo_pipeline import run_demo, build_estimators
import pytest


def test_reproducible_synthetic_demo():
    one = run_demo(n_samples=120, seed=13)
    two = run_demo(n_samples=120, seed=13)
    assert one == two
    assert one["clinical_validation"] is False
    assert one["original_study_reproduction"] is False
    assert one["n_train"] == 96
    assert one["n_test"] == 24


def test_model_training_uses_fold_local_preprocessing():
    for estimator in build_estimators().values():
        assert estimator.steps[0][0] == "imputer"
        assert estimator.steps[-1][0] == "classifier"


def test_metrics_are_software_demo_only():
    report = run_demo(n_samples=120)
    assert "synthetic, non-clinical" in report["dataset"]
    for metrics in report["models"].values():
        for metric in ("synthetic_holdout_auc", "synthetic_holdout_recall", "synthetic_holdout_f1"):
            assert 0 <= metrics[metric] <= 1


def test_reject_insufficient_demo_samples():
    with pytest.raises(ValueError, match="At least"):
        run_demo(n_samples=20)
