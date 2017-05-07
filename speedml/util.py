"""
Speedml utility methods. Contact author https://twitter.com/manavsehgal. Code, docs and demos https://speedml.com.
"""

import pandas as pd
import numpy as np

from sklearn.base import TransformerMixin

# We'll impute missing values using the median for numeric columns and the most
# common value for string columns.
# Code by 'sveitser' at http://stackoverflow.com/a/25562948
class DataFrameImputer(TransformerMixin):
    def fit(self, X, y=None):
        """
        Uses ``X`` dataset to fill empty values for numeric features with the median value, otherwise fills most common value for text features.
        """
        self.fill = pd.Series([X[c].value_counts().index[0]
            if X[c].dtype == np.dtype('O') else X[c].median() for c in X],
            index=X.columns)
        return self
    def transform(self, X, y=None):
        """
        Calls the self.fill rule defined in fit method.
        """
        return X.fillna(self.fill)
