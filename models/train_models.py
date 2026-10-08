from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.model_selection import StratifiedKFold


RANDOM_STATE = 42
N_SPLITS = 5


def get_models():
    """
    Return the Machine Learning models used for comparison.
    """

    models = {
        "LogisticRegression": LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_STATE
        ),

        "RandomForest": RandomForestClassifier(
            n_estimators=300,
            random_state=RANDOM_STATE,
            class_weight="balanced"
        ),

        "XGBoost": XGBClassifier(
            random_state=RANDOM_STATE,
            eval_metric="logloss"
        )
    }

    return models


def build_pipeline(preprocessor, model):
    """
    Combine preprocessing and Machine Learning model
    into a single pipeline.
    """

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )


def create_cv():
    """
    Create stratified cross-validation strategy.
    """

    return StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE
    )