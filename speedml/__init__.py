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
import os

from IPython.core.interactiveshell import InteractiveShell

# Used by Speedml.about
_RELEASE = 'v0.9.2'

class Speedml(Base):
    def __init__(self, train, test, target, uid=None):
        """
        Open datasets ``train`` and ``test`` as CSV or JSON files and store in pandas DataFrames ``Base.train`` and ``Base.test``. Set ``Base.target`` and ``Base.uid`` values based on parameters. Initialize ``Plot``, ``Feature``, and ``Xgb`` components.
        """
        self._setup_environment()

        Base.target = target

        # TODO: Add more file formats supported by pandas.read_
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

    def _setup_environment(self):
        # Used by data out path 'internally' within Speedml methods
        Base.outpath = 'output/'

        # Setup for Notebook environment
        try:
            __IPYTHON__
        except NameError:
            Base.notebook = False
        else:
            Base.notebook = True
            # Multiple outputs from single input cell
            InteractiveShell.ast_node_interactivity = "all"
            # Plots inline within Notebook output
            ipython = get_ipython()
            ipython.magic('matplotlib inline')

    def info(self):
        """
        Runs DataFrame.info() on both Train and Test datasets.
        """
        self.train.info()
        print('-'*40)
        self.test.info()

    def eda(self):
        """
        Performs speed exploratory data analysis (EDA) on the current state of datasets. Returns metrics and recommendations as a dataframe.
        """
        about_index = ['Speedml Release', 'Null Values',
                       '#Samples Train', '#Samples Test', '#Features', 'Numerical over Text']

        about_samples = []

        about_samples.append([_RELEASE, 'Visit https://speedml.com for release notes.'])

        nulls_by_features = Base.train.isnull().sum() + Base.test.isnull().sum()
        nulls = nulls_by_features[1].sum()
        about_samples.append([nulls, 'Aim for zero nulls.'])

        about_samples.append([
            Base.train.shape[0],
            'Much larger than #Features to avoid over-fitting.'])

        about_samples.append([
            Base.test.shape[0],
            'Cannot drop Test samples.'])

        about_samples.append([
            Base.train.shape[1],
            'Compare with n=count during feature selection.'])

        numerical_ratio = int(Base.train_n.shape[1] / Base.train.shape[1] * 100)
        about_samples.append(['{}%'.format(numerical_ratio),
                              'Aim for 100% numerical.'])

        numerical_features = Base.train_n.columns.values

        if numerical_features != []:
            high_cardinality_num = []
            categorical_num = []
            continuous = []
            for feature in numerical_features:
                repeating = Base.train[feature].value_counts()
                if repeating.count() > 90/100 * Base.train.shape[0]:
                    continuous.append((feature, repeating.count()))
                    if feature == Base.target:
                        target_analysis = ['Model ready.', 'Use regression models.']
                    continue
                if repeating.count() > 10:
                    high_cardinality_num.append((feature, repeating.count()))
                    if feature == Base.target:
                        target_analysis = ['Pre-process.', 'Dimensionality reduction?']
                    continue
                if repeating.count() > 1:
                    categorical_num.append((feature, repeating.count()))
                    if feature == Base.target:
                        target_analysis = ['Model ready.', 'Use classification models.']
                    continue

            about_index += ['Numerical High-cardinality',
                            'Numerical Categorical',
                            'Numerical Continuous']

            about_samples.append([
                high_cardinality_num,
                '(>10) categories. Engineer with density method.'])

            about_samples.append([
                categorical_num,
                'Violin plots for outliers.'])

            about_samples.append([
                continuous,
                '~90% unique. Scatter plots for outliers.'])


        if Base.train_n.shape[1] != Base.train.shape[1]:
            text_features = []
            text_features = list(set(Base.train.columns.values) - set(numerical_features))

            if text_features != []:
                about_index += [
                    'Text High-cardinality',
                    'Text Categorical', 'Text Unique']
                high_cardinality_text = []
                categorical_text = []
                text = []
                for feature in text_features:
                    repeating = Base.train[feature].value_counts()
                    if repeating.count() > 90/100 * Base.train.shape[0]:
                        text.append((feature, repeating.count()))
                        if feature == Base.target:
                            target_analysis = ['ERROR.', 'Unique text cannot be processed as a target variable.']
                        continue
                    if repeating.count() > 10:
                        high_cardinality_text.append((feature, repeating.count()))
                        if feature == Base.target:
                            target_analysis = ['Pre-process.', 'Label to numerical. Pre-process high-cardinality.']
                        continue
                    if repeating.count() > 1:
                        categorical_text.append((feature, repeating.count()))
                        if feature == Base.target:
                            target_analysis = ['Model ready.', 'Labels or mapping to numerical. Use classification models.']
                        continue

                about_samples.append([
                    high_cardinality_text,
                    '(>10) categories. Labels to numeric.'])

                about_samples.append([
                    categorical_text,
                    'Mapping or Labels to numeric.'])

                about_samples.append([
                    text,
                    '~90% unique. Extract or drop.'])

        about_index += ['Target Analysis ({})'.format(Base.target)]
        about_samples.append(target_analysis)

        about_df = pd.DataFrame(about_samples,
                                index=about_index,
                                columns=['Results', 'Observations'])

        return about_df

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
        submission.to_csv(file_path,
                          index=False)
        return 'Results saved.'

    def slug(self):
        performance_slug = 'e{:.2f}-m{:.2f}-s{:.2f}-f{:.2f}'.format(
            self.xgb.error * 100,
            self.model.xgb_accuracy * 100,
            self.xgb.sample_accuracy,
            self.xgb.feature_accuracy)
        return performance_slug
