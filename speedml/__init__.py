"""
Speedml is a Python package to speed start machine learning projects. Author @manavsehgal. Docs https://speedml.com.
"""

from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
from builtins import *

from speedml.base import Base
from speedml.plot import Plot
from speedml.feature import Feature
from speedml.xgb import Xgb

import numpy as np
import pandas as pd

import xgboost as xgb

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression

class Speedml(Base):
    """
    Speedml is a simple and powerful API wrapping best machine learning Python libraries and popular strategies used by top data scientists. Author @manavsehgal.
    """
    def __init__(self, train, test, target, uid=None):
        Base.target = target

        if train.endswith('.csv'):
            Base.train = pd.read_csv(train)
            Base.test = pd.read_csv(test)

        if train.endswith('.json'):
            Base.train = pd.read_json(train)
            Base.test = pd.read_json(test)

        if not Base.train.empty and not Base.test.empty:
            if uid:
                Base.uid = Base.test.pop(uid)
                Base.train = Base.train.drop([uid], axis=1)

            self.plot = Plot()
            self.feature = Feature()
            self.xgb = Xgb()

            Base.data_n()
        else:
            print('ERROR: SpeedML can only process .csv and .json files.')

    def shape(self):
        """
        Print shape (samples, features) of train, test datasets and number of numerical features in each dataset.
        """
        message = 'Shape: train {} test {}\n'
        message += 'Numerical: train_n ({}) test_n ({})'
        print(message.format(Base.train.shape,
                             Base.test.shape,
                             Base.train_n.shape[1],
                             Base.test_n.shape[1]))

    def crosstab(self, a, b):
        return pd.crosstab(Base.train[a], Base.train[b])

    def model_data(self):
        Base.train_y = Base.train[Base.target]
        Base.train_X = Base.train.drop([Base.target], axis=1).as_matrix()
        Base.test_X = Base.test.as_matrix()
        print('train_X: {} train_y: {} test_X: {}'.format(Base.train_X.shape, Base.train_y.shape, Base.test_X.shape))

    def sample_accuracy(self):
        train_preds = Base.xgb_model.predict(Base.train_X)
        rounded_preds = np.round(train_preds).astype(int).flatten()
        correct = np.where(rounded_preds == Base.train_y)[0]
        correct_labels = len(correct)
        total_labels = Base.train_y.shape[0]
        self.accuracy = round(correct_labels / total_labels * 100, 2)
        print("Accuracy = {}%. Found {} correct of {} total labels".format(self.accuracy, correct_labels, total_labels))

    def save_results(self, columns, file_path):
        submission = pd.DataFrame(columns)
        submission.to_csv(file_path, index=False)
        print('Results saved.')

    def evaluate_models(self):
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

        self.plot.model_ranking = Base.model_ranking

    def model_ranks(self):
        return Base.model_ranking.sort_values(by='Accuracy', ascending=False)

    def feature_selection(self):
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
        print("Accuracy: %.2f%%" % (accuracy * 100.0))

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

# END: class SpeedML -----------------------------------------
