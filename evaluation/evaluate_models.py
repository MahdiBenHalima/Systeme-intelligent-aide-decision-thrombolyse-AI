import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)
from sklearn.model_selection import cross_validate


def evaluate_on_test(model, X_test, y_test):
    """
    Evaluate a trained model on a test set.
    """

    y_pred = model.predict(X_test)

    results = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(
            y_test,
            y_pred,
            zero_division=0
        ),
        "recall": recall_score(
            y_test,
            y_pred,
            zero_division=0
        ),
        "f1": f1_score(
            y_test,
            y_pred,
            zero_division=0
        )
    }

    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
        results["auc"] = roc_auc_score(
            y_test,
            y_proba
        )

    results["confusion_matrix"] = confusion_matrix(
        y_test,
        y_pred
    )

    return results


def crossval_scores(
    model,
    X,
    y,
    cv
):
    """
    Evaluate a model using stratified cross-validation.
    """

    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "auc": "roc_auc"
    }

    scores = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        return_train_score=False,
        error_score="raise"
    )

    results = {}

    for metric in scoring:
        results[metric] = scores[
            f"test_{metric}"
        ].mean()

    return results


def results_to_dataframe(results):
    """
    Convert model evaluation results into a DataFrame.
    """

    return pd.DataFrame(results).T