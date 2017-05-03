"""
Speedml Xgb component with methods that work on XGBoost model workflow. Author @manavsehgal. Docs https://speedml.com.
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
        optimized_GBM = GridSearchCV(xgb.XGBClassifier(**fixed_params), select_params, scoring = 'accuracy', cv = 5, n_jobs = -1)
        optimized_GBM.fit(Base.train_X, Base.train_y)
        df = pd.DataFrame(optimized_GBM.cv_results_)[['rank_test_score', 'params']].sort_values(by='rank_test_score')
        df.rename(columns = {'rank_test_score': 'rank'}, inplace = True)
        return df

    def cv(self, grid_params):
        xgdmat = xgb.DMatrix(Base.train_X, Base.train_y)
        self.cv_results = xgb.cv(
            params = grid_params, dtrain = xgdmat,
            num_boost_round = 1000, nfold = 5,
            metrics = ['error'], early_stopping_rounds = 20)

    def params(self, params):
        Base.xgb_params = params

    def classifier(self):
        self.clf = xgb.XGBClassifier(**Base.xgb_params)

    def fit(self):
        Base.xgb_model = self.clf.fit(Base.train_X, Base.train_y)

    def predict(self):
        self.predictions = Base.xgb_model.predict(Base.test_X)
