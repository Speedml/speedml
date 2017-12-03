"""
Speedml Model component with methods that work on sklearn models workflow. Contact author https://twitter.com/manavsehgal. Code, docs and demos https://speedml.com.
"""

from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
from builtins import *

from .base import Base

import pandas as pd
import numpy as np

import xgboost as xgb

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, log_loss

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression

class Model(Base):
    def data(self):
        """
        Prepare model input data ``Base.train_y`` as Series, ``Base.train_X``, and ``Base.test_X`` datasets as Matrix.
        """
        Base.train_y = Base.train[Base.target]
        Base.train_X = Base.train.drop([Base.target], axis=1).as_matrix()
        Base.test_X = Base.test.as_matrix()
        message = 'train_X: {} train_y: {} test_X: {}'
        return message.format(Base.train_X.shape,
                              Base.train_y.shape,
                              Base.test_X.shape)

    def evaluate(self):
        """
        Model evaluation across multiple classifiers based on accuracy of predictions.
        """
        classifiers = [
            xgb.XGBClassifier(**Base.xgb_params),
            KNeighborsClassifier(3),
            SVC(probability=True),
            DecisionTreeClassifier(),
            RandomForestClassifier(),
            AdaBoostClassifier(),
            GradientBoostingClassifier(),
            GaussianNB(),
            LogisticRegression()]

        log_cols = ["Classifier", "Accuracy"]
        Base.model_ranking = pd.DataFrame(columns=log_cols)

        sss = StratifiedShuffleSplit(n_splits=10, test_size=0.1, random_state=0)

        X = Base.train_X
        y = Base.train_y

        acc_dict = {}

        for train_index, test_index in sss.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            for clf in classifiers:
                name = clf.__class__.__name__
                clf.fit(X_train, y_train)
                train_predictions = clf.predict(X_test)
                acc = accuracy_score(y_test, train_predictions)
                if name in acc_dict:
                    acc_dict[name] += acc
                else:
                    acc_dict[name] = acc

        for clf in acc_dict:
            acc_dict[clf] = acc_dict[clf] / 10.0
            log_entry = pd.DataFrame([[clf, acc_dict[clf]]], columns=log_cols)
            Base.model_ranking = Base.model_ranking.append(log_entry)
            Base.model_ranking = Base.model_ranking.sort_values(by='Accuracy', ascending=False)

    def ranks(self):
        """
        Returns DataFrame of model ranking sorted by Accuracy.
        """
        self.xgb_accuracy = Base.model_ranking[Base.model_ranking['Classifier'] == 'XGBClassifier']['Accuracy'][0]
        return Base.model_ranking.sort_values(by='Accuracy', ascending=False)
