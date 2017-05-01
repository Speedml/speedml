import pandas as pd
import numpy as np

from sklearn.base import TransformerMixin

# We'll impute missing values using the median for numeric columns and the most
# common value for string columns.
# Code by 'sveitser' at http://stackoverflow.com/a/25562948
class DataFrameImputer(TransformerMixin):
    def fit(self, X, y=None):
        self.fill = pd.Series([X[c].value_counts().index[0]
            if X[c].dtype == np.dtype('O') else X[c].median() for c in X],
            index=X.columns)
        return self
    def transform(self, X, y=None):
        return X.fillna(self.fill)
