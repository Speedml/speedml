"""
Speedml Xgb component with methods that work on XGBoost model workflow. Contact author https://twitter.com/manavsehgal. Code, docs and demos https://speedml.com.
"""

from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
from builtins import *

from .base import Base

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel

class Xgb(Base):
    def sample_accuracy(self):
        """
        Calculate the accuracy of an XGBoost model based on number of correct labels in prediction.
        """
        train_preds = Base.xgb_model.predict(Base.train_X)
        rounded_preds = np.round(train_preds).astype(int).flatten()
        correct = np.where(rounded_preds == Base.train_y)[0]
        correct_labels = len(correct)
        total_labels = Base.train_y.shape[0]
        self.sample_accuracy = round(correct_labels / total_labels * 100, 2)
        message = 'Accuracy = {}%. Found {} correct of {} total labels'
        return message.format(self.sample_accuracy,
                              correct_labels,
                              total_labels)

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
        self.error = self.cv_results.get_value(len(self.cv_results) - 1, 'test-error-mean')

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

    def feature_selection(self):
        """
        Returns threshold and accuracy for ``n`` number of features.
        """
        Base.data_n()
        X = Base.train_n.drop([Base.target], axis=1)
        Y = Base.train[Base.target]

        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=7)

        # Fit model on all training data
        model = xgb.XGBClassifier()
        model.fit(X_train, y_train)

        # Make predictions for test data and evaluate
        y_pred = model.predict(X_test)
        predictions = [round(value) for value in y_pred]
        accuracy = accuracy_score(y_test, predictions)
        self.feature_accuracy = round(accuracy * 100.0, 2)
        print("Accuracy: %f%%" % (self.feature_accuracy))

        # Fit model using each importance as a threshold
        thresholds = np.sort(model.feature_importances_)
        for thresh in thresholds:
        	# Select features using threshold
        	selection = SelectFromModel(model, threshold=thresh, prefit=True)
        	select_X_train = selection.transform(X_train)

        	# Train model
        	selection_model = xgb.XGBClassifier()
        	selection_model.fit(select_X_train, y_train)

        	# Evalation model
        	select_X_test = selection.transform(X_test)
        	y_pred = selection_model.predict(select_X_test)
        	predictions = [round(value) for value in y_pred]
        	accuracy = accuracy_score(y_test, predictions)
        	print ("Thresh=%.3f, n=%d, Accuracy: %.2f%%" % (thresh, select_X_train.shape[1], accuracy*100.0))
