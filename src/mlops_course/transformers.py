from typing import Self

import pandas as pd
from sklearn.base import BaseEstimator, OneToOneFeatureMixin, TransformerMixin


class CategoryTransformer(OneToOneFeatureMixin, TransformerMixin, BaseEstimator):
    """Transformer for treating categorical variables.

    This is a simplified version with restricted error handling
    """

    def __init__(self):  # noqa
        self._fitted = False

    def fit(self, X: pd.Series, y: pd.Series | None = None, **fit_params) -> Self:  # noqa: ANN003
        """Persist observed categories during fitting."""
        if isinstance(X, pd.DataFrame):
            raise ValueError("CategoryTransformer only accepts single column series")
        X = X.astype("category")
        self._categories = X.cat.categories
        self._fitted = True
        return self

    def transform(self, X: pd.Series, y: pd.DataFrame | pd.Series | None = None) -> pd.DataFrame:
        """Transform columns to categorical type using the categories observed during fitting."""
        # Seems that column transformer passes series but expected dataframes
        return X.astype("category").cat.set_categories(self._categories).to_frame()
