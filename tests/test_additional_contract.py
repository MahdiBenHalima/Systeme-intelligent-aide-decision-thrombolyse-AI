"""Additional privacy-safe contract tests: no trained model or patient data required."""
import pytest

from clinical_ml.features import FEATURE_ORDER, to_model_frame
from clinical_ml import inference


def contract_values():
    # Deliberately invalid as a clinical record; schema-order fixture only.
    return {key: 0 for key in FEATURE_ORDER}


def test_columns_are_preserved_even_when_input_mapping_is_reversed():
    reverse_input = dict(reversed(list(contract_values().items())))
    df = to_model_frame(reverse_input)
    assert df.columns.tolist() == FEATURE_ORDER
    assert df.shape == (1, len(FEATURE_ORDER))


def test_extra_patient_identifier_is_not_accepted():
    features = contract_values()
    features["patient_id"] = "NOT_A_PATIENT"
    with pytest.raises(ValueError, match="extra"):
        to_model_frame(features)


def test_missing_model_artifact_is_a_clear_error(monkeypatch, tmp_path):
    monkeypatch.setenv("THROMBOLYSIS_MODEL_PATH", str(tmp_path / "absent.joblib"))
    inference.load_model.cache_clear()
    with pytest.raises(FileNotFoundError, match="Model file not found"):
        inference.load_model()
    inference.load_model.cache_clear()


def test_model_without_predict_proba_is_supported(monkeypatch):
    class PredictOnlyStub:
        def predict(self, frame):
            assert frame.columns.tolist() == FEATURE_ORDER
            return [0]

    monkeypatch.setattr(inference, "load_model", lambda: PredictOnlyStub())
    label, probability = inference.predict_thrombolysis(contract_values())
    assert label == 0
    assert probability is None


def test_second_probability_column_follows_original_contract(monkeypatch):
    class BinaryContractStub:
        def predict(self, frame):
            return [1]

        def predict_proba(self, frame):
            return [[0.65, 0.35]]

    monkeypatch.setattr(inference, "load_model", lambda: BinaryContractStub())
    _, probability = inference.predict_thrombolysis(contract_values())
    assert probability == pytest.approx(0.35)
