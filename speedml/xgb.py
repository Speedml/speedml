"""
Speedml Xgb component with methods that work on XGBoost model workflow. Contact author https://twitter.com/manavsehgal. Code and demos https://github.com/Speedml.
"""

from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
from builtins import *

from speedml.base import Base

import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GridSearchCV

class Xgb(Base):
    def hyper(self, select_params, fixed_params):
        """
        Tune XGBoost hyper-parameters by selecting from permutations of values from the ``select_params`` dictionary. Remaining parameters with single values are specified by the ``fixed_params`` dictionary. Returns a dataframe with ranking of ``select_params`` items.
        """
        optimized_GBM = GridSearchCV(xgb.XGBClassifier(**fixed_params), select_params, scoring = 'accuracy', cv = 5, n_jobs = -1)
        optimized_GBM.fit(Base.train_X, Base.train_y)
        df = pd.DataFrame(optimized_GBM.cv_results_)[['rank_test_score', 'params']].sort_values(by='rank_test_score')
        df.rename(columns = {'rank_test_score': 'rank'}, inplace = True)
        return df

    def cv(self, grid_params):
        """
        Calculate the Cross-Validation (CV) score for XGBoost model based on ``grid_params`` parameters. Sets xgb.cv_results variable to the resulting dataframe.
        """
        xgdmat = xgb.DMatrix(Base.train_X, Base.train_y)
        self.cv_results = xgb.cv(
            params = grid_params, dtrain = xgdmat,
            num_boost_round = 1000, nfold = 5,
            metrics = ['error'], early_stopping_rounds = 20)

    def params(self, params):
        """
        Sets Base.xgb_params to ``params`` dictionary.
        """
        Base.xgb_params = params

    def classifier(self):
        """
        Creates the XGBoost Classifier with Base.xgb_params dictionary of model hyper-parameters.
        """
        self.clf = xgb.XGBClassifier(**Base.xgb_params)

    def fit(self):
        """
        Sets Base.xgb_model with trained XGBoost model.
        """
        Base.xgb_model = self.clf.fit(Base.train_X, Base.train_y)

    def predict(self):
        """
        Sets xgb.predictions with predictions from the XGBoost model.
        """
        self.predictions = Base.xgb_model.predict(Base.test_X)
