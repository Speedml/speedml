"""
Speedml is a Python package to speed start machine learning projects. Contact author https://twitter.com/manavsehgal. Code, docs and demos https://speedml.com.
"""

from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
from builtins import *

from speedml.base import Base
from speedml.plot import Plot
from speedml.feature import Feature
from speedml.xgb import Xgb
from speedml.model import Model

import pandas as pd

class Speedml(Base):
    def __init__(self, train, test, target, uid=None):
        """
        Open datasets ``train`` and ``test`` as CSV or JSON files and store in pandas DataFrames ``Base.train`` and ``Base.test``. Set ``Base.target`` and ``Base.uid`` values based on parameters. Initialize ``Plot``, ``Feature``, and ``Xgb`` components.
        """
        Base.version = 'v0.9.1'
        Base.outpath = 'output/'
        Base.inpath = '../input/'

        Base.target = target

        if train.endswith('.csv'):
            Base.train = pd.read_csv(Base.inpath + train)
            Base.test = pd.read_csv(Base.inpath + test)

        if train.endswith('.json'):
            Base.train = pd.read_json(Base.inpath + train)
            Base.test = pd.read_json(Base.inpath + test)

        if not Base.train.empty and not Base.test.empty:
            if uid:
                Base.uid = Base.test.pop(uid)
                Base.train = Base.train.drop([uid], axis=1)

            self.plot = Plot()
            self.feature = Feature()
            self.xgb = Xgb()
            self.model = Model()

            Base.data_n()
        else:
            print('ERROR: SpeedML can only process .csv and .json file extensions.')

    def info(self):
        """
        Runs DataFrame.info() on both Train and Test datasets.
        """
        self.train.info()
        print('-'*40)
        self.test.info()

    def about(self):
        """
        Describes Speedml package and current status of machine learning workflow.
        """
        message  = 'You are running Speedml {}\n'.format(Base.version)
        message += 'Train dataset is {:.2f}% model ready.\n'.format(Base.train_n.shape[1] / Base.train.shape[1] * 100)
        nulls_by_features = Base.train.isnull().sum() + Base.test.isnull().sum()
        nulls = nulls_by_features[1].sum()
        message += 'Datasets contain {:d} null values.'.format(int(nulls))
        return message

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

    def save_results(self, columns, file_name):
        """
        Saves the ``columns`` dictionary input to a DataFrame as ``file_name`` CSV file.
        """
        submission = pd.DataFrame(columns)
        submission.to_csv(Base.outpath + file_name,
                          index=False)
        return 'Results saved.'
