"""
Speedml Feature component with methods that work on features or the feature engineering workflow. Author @manavsehgal. Docs https://speedml.com.
"""

from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
from builtins import *

from speedml.base import Base
from speedml.util import DataFrameImputer

import numpy as np
from sklearn.preprocessing import LabelEncoder
import re

class Feature(Base):
    def drop(self, features):
        """
        Drop one or more list of strings naming ``features`` from train and test datasets.
        """
        start = Base.train.shape[1]

        Base.train = Base.train.drop(features, axis=1)
        Base.test = Base.test.drop(features, axis=1)
        Base.data_n()

        end = Base.train.shape[1]
        message = 'Dropped {} features.'
        return message.format(start - end)

    def impute(self):
        start = Base.train.isnull().sum().sum()

        Base.test[Base.target] = -1
        combine = Base.train.append(Base.test)
        combine = DataFrameImputer().fit_transform(combine)
        Base.train = combine[0:Base.train.shape[0]]
        Base.test = combine[Base.train.shape[0]::]
        Base.test = Base.test.drop([Base.target], axis=1)
        Base.data_n()

        end = Base.train.isnull().sum().sum()
        message = 'Imputed {} empty values to {}.'
        return message.format(start, end)

    def ordinal_to_numeric(self, a, map_to_numbers):
        Base.train[a] = Base.train[a].apply(lambda x: map_to_numbers[x])
        Base.test[a] = Base.test[a].apply(lambda x: map_to_numbers[x])
        Base.data_n()

    def fillna(self, a, new):
        """
        Fills empty or null values in ``a`` feature name with ``new`` string value.
        """
        Base.train[a] = Base.train[a].fillna(new)
        Base.test[a] = Base.test[a].fillna(new)

    def replace(self, a, match, new):
        """
        In feature ``a`` values ``match`` string or list of strings and replace with a ``new`` string.
        """
        Base.train[a] = Base.train[a].replace(match, new)
        Base.test[a] = Base.test[a].replace(match, new)

    def outliers_fix(self, a, lower = None, upper = None):
        """
        Fix outliers for ``lower`` or ``upper`` or both percentile of values within ``a`` feature.
        """
        if upper:
            upper_value = np.percentile(Base.train[a].values, upper)
            change = Base.train.loc[Base.train[a] > upper_value, a].shape[0]
            Base.train.loc[Base.train[a] > upper_value, a] = upper_value
            Base.data_n()
            print('{} or {:.2f}% outliers fixed.'.format(change, change/Base.train.shape[0]*100))

        if lower:
            lower_value = np.percentile(Base.train[a].values, lower)
            change = Base.train.loc[Base.train[a] < lower_value, a].shape[0]
            Base.train.loc[Base.train[a] < lower_value, a] = lower_value
            Base.data_n()
            print('{} or {:.2f}% outliers fixed.'.format(change, change/Base.train.shape[0]*100))

    def density(self, a):
        vals = Base.train[a].value_counts()
        dvals = vals.to_dict()
        Base.train[a + '_density'] = Base.train[a].apply(lambda x: dvals.get(x, vals.min()))
        Base.test[a + '_density'] = Base.test[a].apply(lambda x: dvals.get(x, vals.min()))
        Base.data_n()

    def add(self, a, num):
        Base.train[a] = Base.train[a] + num
        Base.test[a] = Base.test[a] + num
        Base.data_n()

    def sum(self, new, a, b):
        Base.train[new] = Base.train[a] + Base.train[b]
        Base.test[new] = Base.test[a] + Base.test[b]
        Base.data_n()

    def diff(self, new, a, b):
        Base.train[new] = Base.train[a] - Base.train[b]
        Base.test[new] = Base.test[a] - Base.test[b]
        Base.data_n()

    def product(self, new, a, b):
        Base.train[new] = Base.train[a] * Base.train[b]
        Base.test[new] = Base.test[a] * Base.test[b]
        Base.data_n()

    def divide(self, new, a, b):
        Base.train[new] = Base.train[a] / Base.train[b]
        Base.test[new] = Base.test[a] / Base.test[b]
        # Histograms require finite values
        Base.train[new] = Base.train[new].replace([np.inf, -np.inf], 0)
        Base.test[new] = Base.test[new].replace([np.inf, -np.inf], 0)
        Base.data_n()

    def round(self, new, a, precision):
        Base.train[new] = round(Base.train[a], precision)
        Base.test[new] = round(Base.test[a], precision)
        Base.data_n()

    def concat(self, new, a, sep, b):
        Base.train[new] = Base.train[a].astype(str) + sep + Base.train[b].astype(str)
        Base.test[new] = Base.test[a].astype(str) + sep + Base.test[b].astype(str)

    def list_len(self, new, a):
        Base.train[new] = Base.train[a].apply(len)
        Base.test[new] = Base.test[a].apply(len)
        Base.data_n()

    def word_count(self, new, a):
        Base.train[new] = Base.train[a].apply(lambda x: len(x.split(" ")))
        Base.test[new] = Base.test[a].apply(lambda x: len(x.split(" ")))
        Base.data_n()

    def _regex_text(self, regex, text):
        regex_search = re.search(regex, text)
        # If the word exists, extract and return it.
        if regex_search:
            return regex_search.group(1)
        return ""

    def regex_extract(self, a, regex, new=None):
        """
        Args:
            a (str): Existing feature to match regex.
            regex (str): Regular expression to use for matching and text extraction.
            new (str): New feature to extract regex matched text. If new=None then replace existing feature specified by ``a``.
        """
        Base.train[new if new else a] = Base.train[a].apply(lambda x: self._regex_text(regex=regex, text=x))
        Base.test[new if new else a] = Base.test[a].apply(lambda x: self._regex_text(regex=regex, text=x))

    def labels(self, features):
        Base.test[Base.target] = -1
        combine = Base.train.append(Base.test)

        le = LabelEncoder()
        for feature in features:
            combine[feature] = le.fit_transform(combine[feature])

        Base.train = combine[0:Base.train.shape[0]]
        Base.test = combine[Base.train.shape[0]::]
        Base.test = Base.test.drop([Base.target], axis=1)
        Base.data_n()
