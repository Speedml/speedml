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
            self.model = Model()

            Base.data_n()
        else:
            print('ERROR: SpeedML can only process .csv and .json file extensions.')

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

    def save_results(self, columns, file_path):
        """
        Saves the ``columns`` dictionary input to a DataFrame as ``file_path`` CSV file.
        """
        submission = pd.DataFrame(columns)
        submission.to_csv(file_path, index=False)
        return 'Results saved.'
