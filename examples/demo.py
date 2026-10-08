import sys
from pathlib import Path

import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Permet d'importer les modules du projet
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from preprocessing.preprocessing import build_preprocessor
from models.train_models import (
    get_models,
    build_pipeline,
    create_cv
)
from evaluation.evaluate_models import (
    evaluate_on_test,
    crossval_scores
)


def main():

    # --------------------------------------------------
    # 1. Génération d'un dataset de démonstration
    # --------------------------------------------------

    X, y = make_classification(
        n_samples=500,
        n_features=15,
        n_informative=8,
        n_redundant=2,
        n_classes=2,
        random_state=42
    )

    X = pd.DataFrame(
        X,
        columns=[
            f"feature_{i}"
            for i in range(X.shape[1])
        ]
    )

    y = pd.Series(
        y,
        name="target"
    )

    print("Dataset shape:", X.shape)

    # --------------------------------------------------
    # 2. Séparation train / test
    # --------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    # --------------------------------------------------
    # 3. Préprocessing
    # --------------------------------------------------

    (
        numeric_cols,
        categorical_cols,
        numeric_transformer,
        categorical_transformer
    ) = build_preprocessor(X_train)

    from sklearn.compose import ColumnTransformer

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                numeric_transformer,
                numeric_cols
            ),
            (
                "cat",
                categorical_transformer,
                categorical_cols
            )
        ]
    )

    # --------------------------------------------------
    # 4. Modèles
    # --------------------------------------------------

    models = get_models()
    cv = create_cv()

    all_results = {}

    # --------------------------------------------------
    # 5. Entraînement et évaluation
    # --------------------------------------------------

    for name, model in models.items():

        print(f"\n===== {name} =====")

        pipeline = build_pipeline(
            preprocessor,
            model
        )

        # Cross-validation
        cv_results = crossval_scores(
            pipeline,
            X_train,
            y_train,
            cv
        )

        print("Cross-validation:")
        for metric, value in cv_results.items():
            print(f"{metric}: {value:.4f}")

        # Entraînement final
        pipeline.fit(
            X_train,
            y_train
        )

        # Test
        test_results = evaluate_on_test(
            pipeline,
            X_test,
            y_test
        )

        print("\nTest:")
        for metric, value in test_results.items():

            if metric != "confusion_matrix":
                print(
                    f"{metric}: {value:.4f}"
                )

        print(
            "Confusion matrix:"
        )
        print(
            test_results["confusion_matrix"]
        )

        all_results[name] = {
            "cv": cv_results,
            "test": test_results
        }


if __name__ == "__main__":
    main()