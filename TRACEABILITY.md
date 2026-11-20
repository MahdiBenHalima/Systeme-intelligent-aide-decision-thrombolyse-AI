# Source-to-public AI traceability

This file documents what was **verified from source code** and what remains **unverified**. It is not a report of clinical performance.

| Public component | Private source | Status |
| --- | --- | --- |
| `clinical_ml/features.py` | `fastapi/services/predictor.py` | All 12 field names and their order verified against original code |
| `clinical_ml/inference.py` | `fastapi/services/predictor.py` | Preserves the original `predict` call and optional second-column `predict_proba`; model loading isolated to local trusted artifact |
| `clinical_ml/predict.py` | Original backend prediction service | New local-only CLI adapter, **not** a verbatim original module |
| `tests/test_inference_contract.py` | Public inference component | Interface tests use an explicit stub estimator, **not** clinical data or the original trained model |
| `preprocessing/`, `feature_selection/`, `models/`, `evaluation/` | Private research work | Generic experimental utilities, **not established as a line-by-line reproduction of training** |

## What is not validated here

- The original research dataset, preprocessing and feature encoding, exact train/test split, hyperparameter search and exported model artifact have not been reproduced or independently audited in this public repository.
- A passing CI run validates the public code's interface and error handling, not discrimination, calibration, performance on real patients, transportability or safety.
- The public repository does not include real patient records, real predictions, or the trained model. Do not add those assets without proper institutional approval and privacy/provenance review.
- The model uses 12 input columns; matching names and order alone does **not** establish the expected type, unit, or category encodings.
- The original code returns the second `predict_proba` column as the positive-class probability. This assumes a binary model whose positive class is the second class. Confirm `classes_` for the actual model before interpreting it.
- The `predicted_label` field in the CLI is a **research model label**. It is not a clinical treatment recommendation.

## Reproducing a legitimate validation

In an authorized private environment: identify the exact model version and training script, inspect target definition and label encoding, perform privacy checks, verify patient-level split integrity, and evaluate on the untouched real test cohort with appropriate confidence intervals and calibration. Record dataset version, n, class prevalence, preprocessing fit scope, random seeds, thresholds, ROC AUC, recall/sensitivity, specificity, precision, F1 and failure analysis.

**Do not publish scores from synthetic demonstrations as clinical results.** No measured clinical metrics are claimed by this repository.

## Publication checklist

- [ ] Institutional permission to publish the extracted source
- [ ] Repository history examined for data, credentials and model artifacts
- [ ] Complete source and historical training protocol independently reviewed
- [ ] No patient-level examples or notebook outputs committed
- [ ] Trained artifact and raw data remain private
- [ ] CI green for code-level contract tests
- [ ] Separate authorized clinical evaluation before any clinical claims

This is a research engineering portfolio project, **not a medical device**.
