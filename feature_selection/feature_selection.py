import pandas as pd

from sklearn.feature_selection import SelectKBest, mutual_info_classif


def select_features(
    X: pd.DataFrame,
    y: pd.Series,
    k: int = 12
):
    """
    Select the k most informative features using
    mutual information.
    """

    selector = SelectKBest(
        score_func=mutual_info_classif,
        k=min(k, X.shape[1])
    )

    X_selected = selector.fit_transform(X, y)

    selected_features = X.columns[
        selector.get_support()
    ].tolist()

    return X_selected, selected_features, selector