from speedml.plot import Plot
from speedml.feature import Feature

import numpy as np
import pandas as pd

import xgboost as xgb
import graphviz

from sklearn.model_selection import GridSearchCV
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

class Speedml(object):
    '''
    Speedml is a simple and powerful API wrapping best machine learning Python libraries and popular strategies used by top data scientists. Author @manavsehgal.
    '''
    def __init__(self, train, test, target, uid=None):
        self.target = target

        if train.endswith('.csv'):
            self.train = pd.read_csv(train)
            self.test = pd.read_csv(test)

        if train.endswith('.json'):
            self.train = pd.read_json(train)
            self.test = pd.read_json(test)

        if not self.train.empty and not self.test.empty:
            if uid:
                self.uid = self.test.pop(uid)
                self.train = self.train.drop([uid], axis=1)

            self.plot = Plot(self.train, self.test,
                             self.target,
                             self.uid if uid else None)
            self.feature = Feature(self.train, self.test,
                                   self.target,
                                   self.uid if uid else None)
            self.data_n()
        else:
            print('ERROR: SpeedML can only process .csv and .json files.')

    def data_n(self):
        self.train_n = self.train.select_dtypes(include=[np.number])
        self.test_n = self.test.select_dtypes(include=[np.number])
        self.plot.data_n()
        self.feature.data_n()

    def shape(self):
        '''
        Print shape (samples, features) of train, test datasets and number of numerical features in each dataset.
        '''
        message = 'Shape: train {} test {}\n'
        message += 'Numerical: train_n ({}) test_n ({})'
        print(message.format(self.train.shape,
                             self.test.shape,
                             self.train_n.shape[1],
                             self.test_n.shape[1]))

    def crosstab(self, a, b):
        return pd.crosstab(self.train[a], self.train[b])

    def model_data(self):
        self.train_y = self.train[self.target]
        self.train_X = self.train.drop([self.target], axis=1).as_matrix()
        self.test_X = self.test.as_matrix()
        print('train_X: {} train_y: {} test_X: {}'.format(self.train_X.shape, self.train_y.shape, self.test_X.shape))

    def xgb_hyper(self, select_params, fixed_params):
        optimized_GBM = GridSearchCV(xgb.XGBClassifier(**fixed_params), select_params, scoring = 'accuracy', cv = 5, n_jobs = -1)
        optimized_GBM.fit(self.train_X, self.train_y)
        df = pd.DataFrame(optimized_GBM.cv_results_)[['rank_test_score', 'params']].sort_values(by='rank_test_score')
        df.rename(columns = {'rank_test_score': 'rank'}, inplace = True)
        return df

    def xgb_cv(self, grid_params):
        xgdmat = xgb.DMatrix(self.train_X, self.train_y)

        self.xgb_cv_results = xgb.cv(params = grid_params, dtrain = xgdmat, num_boost_round = 1000, nfold = 5, metrics = ['error'], early_stopping_rounds = 20)
        # Look for early stopping that minimizes error

    def xgb_params(self, params):
        self.xgb_params = params

    def xgb_classifier(self):
        self.xgb_clf = xgb.XGBClassifier(**self.xgb_params)

    def xgb_fit(self):
        self.xgb_model = self.xgb_clf.fit(self.train_X, self.train_y)
        self.plot.xgb_model = self.xgb_model

    def xgb_predict(self):
        self.xgb_predictions = self.xgb_model.predict(self.test_X)

    def xgb_tree(self):
        xgb.plot_tree(self.xgb_model)

    def xgb_graph(self):
        xgb.to_graphviz(self.xgb_model)

    def sample_accuracy(self):
        train_preds = self.xgb_model.predict(self.train_X)
        rounded_preds = np.round(train_preds).astype(int).flatten()
        correct = np.where(rounded_preds == self.train_y)[0]
        correct_labels = len(correct)
        total_labels = self.train_y.shape[0]
        self.accuracy = round(correct_labels / total_labels * 100, 2)
        print("Accuracy = {}%. Found {} correct of {} total labels".format(self.accuracy, correct_labels, total_labels))

    def save_results(self, columns, file_path):
        submission = pd.DataFrame(columns)
        submission.to_csv(file_path, index=False)
        print('Results saved.')

    def evaluate_models(self):
        classifiers = [
            xgb.XGBClassifier(**self.xgb_params),
            KNeighborsClassifier(3),
            SVC(probability=True),
            DecisionTreeClassifier(),
            RandomForestClassifier(),
            AdaBoostClassifier(),
            GradientBoostingClassifier(),
            GaussianNB(),
            LogisticRegression()]

        log_cols = ["Classifier", "Accuracy"]
        self.model_ranking = pd.DataFrame(columns=log_cols)

        sss = StratifiedShuffleSplit(n_splits=10, test_size=0.1, random_state=0)

        X = self.train_X
        y = self.train_y

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
            self.model_ranking = self.model_ranking.append(log_entry)
            self.model_ranking = self.model_ranking.sort_values(by='Accuracy', ascending=False)

        self.plot.model_ranking = self.model_ranking

    def model_ranks(self):
        return self.model_ranking.sort_values(by='Accuracy', ascending=False)

    def feature_selection(self):
        X = self.train_n.drop([self.target], axis=1)
        Y = self.train[self.target]

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
