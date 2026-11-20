# Thrombolysis AI — Machine Learning Decision-Support Research

**Clinical machine learning · Structured-data classification · Model evaluation**

This repository documents the AI component of a research prototype investigating **machine-learning predictions associated with thrombolysis decisions**. It is a technical portfolio artifact, **not a medical device or clinical decision rule**. Predictions must never be used to determine whether a patient should receive thrombolysis.

The source project includes preprocessing research, experimental model comparisons, a trained model and an application. This public repository contains **AI-only code** and a faithful extraction of the original inference feature contract. The full application, training data and trained model remain private.

## Key components

| Component | Public implementation |
|---|---|
| Original deployed-model input contract | `clinical_ml/features.py` — same 12 named, ordered fields as the private inference service |
| Original inference logic | `clinical_ml/inference.py` — loads a local trained model, calls `predict` and optional `predict_proba` |
| Local execution interface | `clinical_ml/predict.py` — accepts a JSON file supplied locally; no bundled patient records |
| Separate reproducible non-clinical ML demo | `experimental_ml/demo_pipeline.py` — synthetic holdout, fold-safe preprocessing, CV, model comparison; not patient research |
| Experimental research utilities | `preprocessing/`, `feature_selection/`, `models/`, `evaluation/` |
| Automated validation | `tests/` and GitHub Actions; checks schema, input validation and model interface |

### Real inference data flow

```text
Authorized local feature input (12 clinical variables)
                    |
                    v
    Validate field names and original order
                    |
                    v
     Construct one-row pandas DataFrame
                    |
                    v
  Load locally supplied, trusted model artifact
                    |
                    v
    Model.predict / Model.predict_proba
                    |
                    v
 Research prediction output (not clinical advice)
```

## Exact deployed-model feature schema

The ordered feature names were taken from the private project's `fastapi/services/predictor.py`:

```text
1.  Onset to door time
2.  Hypodensity signs
3.  Sedentary lifestyle
4.  normal CT scan
5.  Early ischemic changes
6.  Diastolic blood pressure
7.  Previous stroke
8.  Initial NIHSS
9.  Systolic blood pressure
10. INR
11. Age
12. Gender
```

**Important:** These fields describe the existing research model's interface. They do not constitute a clinically validated thrombolysis eligibility checklist. The model's historical feature engineering and categorical coding conventions must be confirmed against the original trained artifact before interpreting outputs.

## Training research vs deployed inference

The public `preprocessing/`, `feature_selection/`, `models/` and `evaluation/` modules provide reusable experimental ML utilities, including imputation/scaling, mutual-information feature selection, Logistic Regression, Random Forest, XGBoost and metrics/cross-validation.

**These public utilities are not asserted to reproduce the exact original training notebook.** The private research notebook includes analyses, data-quality work and data that are not safe to publish without further review. This repository prioritizes a faithful public **inference implementation** and documented research methods rather than falsely claiming full reproducibility of the historical training run.

## Reproducible synthetic software demonstration

The portfolio also includes a **separate training and evaluation demonstration**. This is not the original clinical cohort and does not recreate its metrics. It uses only `sklearn.make_classification` synthetic tabular data to demonstrate the engineering mechanics of:

- Stratified 80/20 holdout split before any fitting
- Train-only 3-fold stratified cross-validation
- Fold-local imputation and optional scaling via `sklearn.pipeline.Pipeline`
- Logistic Regression, Random Forest and optional XGBoost comparisons (XGBoost installed in CI)
- ROC AUC, precision, recall, specificity, F1, accuracy and confusion matrix on the untouched synthetic holdout
- Deterministic seed and machine-readable JSON output

Run:

```bash
python -m pip install "numpy>=1.26" "pandas>=2.0" "scikit-learn>=1.3" "xgboost>=2.0"
python -m experimental_ml.demo_pipeline --samples 240 --seed 42
```

**All reported scores from this command are synthetic software-test scores, never estimates of clinical effectiveness.** A separate source-traceability audit and the validation limitations are in [TRACEABILITY.md](TRACEABILITY.md).

## Running the inference module

Python 3.10+ is recommended.

```bash
pip install pandas joblib scikit-learn
```

Set the environment variable to your **own authorized and trusted model file**; it is not included in the repo. Then supply a private JSON file whose keys match all twelve feature names:

```bash
export THROMBOLYSIS_MODEL_PATH="/path/to/private/modele.joblib"
python -m clinical_ml.predict --input /path/to/private/features.json
```

Do not commit either file. Joblib/pickle artifacts can execute arbitrary code when loaded; use only artifacts you trust and control.

The experimental ML modules have additional dependencies listed in `requirements.txt`.

## Tests and validation

```bash
python -m pip install "pandas>=2.0" "pytest>=8"
python -m pytest tests -q
```

CI validates the feature contract, function-level interface and non-clinical training demonstration without a trained artifact or patient data. **This does not validate model accuracy, clinical safety, real-patient generalization or the end-to-end private application.** Local evaluation with authorized data and the original training pipeline requires a separate controlled environment.

No patient-level performance metrics are published here because source-specific provenance, data splits and independent validation must be checked before claiming them publicly.

## Data privacy and clinical safety

- No real or synthetic patient spreadsheets, case-level notebook outputs, duplicates listings or train/test data are included in this public version.
- The `modele.joblib` trained artifact is intentionally **not published** pending privacy, provenance and redistribution review.
- No patient identifiers, clinical prediction logs, authentication or Flutter application code are distributed.
- The code is **for engineering research and demonstration of the original implementation**, not medical decision-making.

---

**Focus:** Clinical ML Research · Classification · Feature Engineering · Model Integration
