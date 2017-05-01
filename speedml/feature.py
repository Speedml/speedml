from speedml.util import DataFrameImputer

import numpy as np
from sklearn.preprocessing import LabelEncoder
import re

class Feature(object):
    def data_n(self):
        self.train_n = self.train.select_dtypes(include=[np.number])
        self.test_n = self.test.select_dtypes(include=[np.number])

    def drop(self, features):
        '''
        Drop list of features from train and test datasets.

        Params
        ------
        features: List of features to drop from train and test datasets.

        Side Effects
        ------------
        Updated numeric datasets with self.data_n().
        '''
        self.train = self.train.drop(features, axis=1)
        self.test = self.test.drop(features, axis=1)
        self.data_n()

    def impute(self):
        self.test[self.target] = -1
        combine = self.train.append(self.test)
        combine = DataFrameImputer().fit_transform(combine)
        self.train = combine[0:self.train.shape[0]]
        self.test = combine[self.train.shape[0]::]
        self.test = self.test.drop([self.target], axis=1)
        self.data_n()

    def ordinal_to_numeric(self, a, map_to_numbers):
        self.train[a] = self.train[a].apply(lambda x: map_to_numbers[x])
        self.test[a] = self.test[a].apply(lambda x: map_to_numbers[x])
        self.data_n()

    def fillna(self, a, new):
        """
        a: feature where to check for NaN values.
        new: new value to replace the matched NaN values.
        """
        self.train[a] = self.train[a].fillna(new)
        self.test[a] = self.test[a].fillna(new)

    def replace(self, a, match, new):
        """
        a: feature where to replace text.
        existing: string or list of strings to match.
        new: new text to replace the matched string or list of strings.
        """
        self.train[a] = self.train[a].replace(match, new)
        self.test[a] = self.test[a].replace(match, new)

    def outliers_fix(self, a, lower = None, upper = None):
        """
        a = feature name
        lower = percentile >0 but <100
        upper = percentile <100 but >0
        """
        if upper:
            upper_value = np.percentile(self.train[a].values, upper)
            change = self.train.loc[self.train[a] > upper_value, a].shape[0]
            self.train.loc[self.train[a] > upper_value, a] = upper_value
            self.data_n()
            print('{} or {:.2f}% outliers fixed.'.format(change, change/self.train.shape[0]*100))

        if lower:
            lower_value = np.percentile(self.train[a].values, lower)
            change = self.train.loc[self.train[a] < lower_value, a].shape[0]
            self.train.loc[self.train[a] < lower_value, a] = lower_value
            self.data_n()
            print('{} or {:.2f}% outliers fixed.'.format(change, change/self.train.shape[0]*100))

    def density(self, a):
        vals = self.train[a].value_counts()
        dvals = vals.to_dict()
        self.train[a + '_density'] = self.train[a].apply(lambda x: dvals.get(x, vals.min()))
        self.test[a + '_density'] = self.test[a].apply(lambda x: dvals.get(x, vals.min()))
        self.data_n()

    def add(self, a, num):
        self.train[a] = self.train[a] + num
        self.test[a] = self.test[a] + num
        self.data_n()

    def sum(self, new, a, b):
        self.train[new] = self.train[a] + self.train[b]
        self.test[new] = self.test[a] + self.test[b]
        self.data_n()

    def diff(self, new, a, b):
        self.train[new] = self.train[a] - self.train[b]
        self.test[new] = self.test[a] - self.test[b]
        self.data_n()

    def product(self, new, a, b):
        self.train[new] = self.train[a] * self.train[b]
        self.test[new] = self.test[a] * self.test[b]
        self.data_n()

    def divide(self, new, a, b):
        self.train[new] = self.train[a] / self.train[b]
        self.test[new] = self.test[a] / self.test[b]
        # Histograms require finite values
        self.train[new] = self.train[new].replace([np.inf, -np.inf], 0)
        self.test[new] = self.test[new].replace([np.inf, -np.inf], 0)
        self.data_n()

    def round(self, new, a, precision):
        self.train[new] = round(self.train[a], precision)
        self.test[new] = round(self.test[a], precision)
        self.data_n()

    def concat(self, new, a, sep, b):
        self.train[new] = self.train[a].astype(str) + sep + self.train[b].astype(str)
        self.test[new] = self.test[a].astype(str) + sep + self.test[b].astype(str)

    def list_len(self, new, a):
        self.train[new] = self.train[a].apply(len)
        self.test[new] = self.test[a].apply(len)
        self.data_n()

    def word_count(self, new, a):
        self.train[new] = self.train[a].apply(lambda x: len(x.split(" ")))
        self.test[new] = self.test[a].apply(lambda x: len(x.split(" ")))
        self.data_n()

    def _regex_text(self, regex, text):
        regex_search = re.search(regex, text)
        # If the word exists, extract and return it.
        if regex_search:
            return regex_search.group(1)
        return ""

    def regex_extract(self, a, regex, new=None):
        '''
        new: New feature to extract regex matched text. If new=None then replace existing feature specified by a.
        a: Existing feature to match regex.
        regex: Regular expression to use for matching and text extraction.
        '''
        self.train[new if new else a] = self.train[a].apply(lambda x: self._regex_text(regex=regex, text=x))
        self.test[new if new else a] = self.test[a].apply(lambda x: self._regex_text(regex=regex, text=x))

    def labels(self, features):
        self.test[self.target] = -1
        combine = self.train.append(self.test)

        le = LabelEncoder()
        for feature in features:
            combine[feature] = le.fit_transform(combine[feature])

        self.train = combine[0:self.train.shape[0]]
        self.test = combine[self.train.shape[0]::]
        self.test = self.test.drop([self.target], axis=1)
        self.data_n()
