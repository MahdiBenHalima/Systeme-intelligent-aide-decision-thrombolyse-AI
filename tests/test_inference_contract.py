"""Contract tests for the actual inference code, without disclosing patient data."""
import pytest
from clinical_ml.features import FEATURE_ORDER, to_model_frame
from clinical_ml import inference


def sample_contract_values():
    # Controlled type-only fixture; NOT a patient record or clinical scenario.
    return {name: 0 for name in FEATURE_ORDER}


def test_original_feature_contract_has_12_ordered_fields():
    assert len(FEATURE_ORDER) == 12
    frame = to_model_frame(sample_contract_values())
    assert frame.shape == (1, 12)
    assert frame.columns.tolist() == FEATURE_ORDER


def test_missing_and_unexpected_fields_rejected():
    data = sample_contract_values()
    data.pop("Age")
    with pytest.raises(ValueError, match="missing"):
        to_model_frame(data)
    data = sample_contract_values()
    data["patient_id"] = 123
    with pytest.raises(ValueError, match="extra"):
        to_model_frame(data)


def test_inference_calls_loaded_model_without_fake_predictions(monkeypatch):
    class ContractModel:
        def predict(self, frame):
            assert list(frame.columns) == FEATURE_ORDER
            return [1]

        def predict_proba(self, frame):
            assert list(frame.columns) == FEATURE_ORDER
            return [[0.2, 0.8]]

    monkeypatch.setattr(inference, "load_model", lambda: ContractModel())
    label, prob = inference.predict_thrombolysis(sample_contract_values())
    assert label == 1
    assert prob == pytest.approx(0.8)


def test_no_model_configured_fails_explicitly(monkeypatch):
    monkeypatch.delenv("THROMBOLYSIS_MODEL_PATH", raising=False)
    inference.load_model.cache_clear()
    with pytest.raises(RuntimeError, match="THROMBOLYSIS_MODEL_PATH"):
        inference.load_model()
