import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean dataframe column names.
    """
    df = df.copy()
    df.columns = df.columns.str.strip()
    return df


def remove_identifier_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove potential patient/dossier identifiers.
    """
    id_cols = [
        "ndossier",
        "n_dossier",
        "numero_dossier",
        "id",
        "patient_id"
    ]

    return df.drop(
        columns=[c for c in id_cols if c in df.columns],
        errors="ignore"
    )


def prepare_features(
    df: pd.DataFrame,
    target: str
):
    """
    Separate features and target.
    """
    df = clean_columns(df)
    df = remove_identifier_columns(df)

    X = df.drop(columns=[target])
    y = df[target]

    return X, y


def build_preprocessor(X: pd.DataFrame):
    """
    Build the preprocessing pipeline used before model training.
    """

    numeric_cols = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_cols = [
        c for c in X.columns
        if c not in numeric_cols
    ]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent")
            )
        ]
    )

    return numeric_cols, categorical_cols, numeric_transformer, categorical_transformer