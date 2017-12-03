"""
Speedml is a Python package to speed start machine learning projects. Contact author https://twitter.com/manavsehgal. Code, docs and demos https://speedml.com.
"""

from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
from builtins import *

from .base import Base
from .plot import Plot
from .feature import Feature
from .xgb import Xgb
from .model import Model

import numpy as np
import pandas as pd
import os

from IPython.core.interactiveshell import InteractiveShell

# Used by Speedml.about
_RELEASE = 'v0.9.3'

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

            self.np = np
            self.pd = pd
        else:
            print('ERROR: SpeedML can only process .csv and .json file extensions.')

    def configure(self, option=None, value=None):
        """
        Configure Speedml defaults with ``option`` configuration parameter, ``value`` setting. When method is called without parameters it simply returns the current config dictionary, otherwise returns the updated configuration.
        """
        if option and value:
            Base._config[option] = value
        return Base._config

    def _setup_environment(self):
        Base._config = {}
        # Used by data out path 'internally' within Speedml methods
        Base._config['outpath'] = 'output/'
        # Positive and negative skew within +- this value
        Base._config['outlier_threshold'] = 3
        # #Features/#Samples Train < this value
        Base._config['overfit_threshold'] = 0.01
        # Feature is high-cardinality if categories > this value
        Base._config['high_cardinality'] = 10
        # Unique (continuous) if Base._config['unique_ratio']% non-repeat values
        Base._config['unique_ratio'] = 80

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
        Performs speed exploratory data analysis (EDA) on the current state of datasets. Returns metrics and recommendations as a dataframe. Progressively hides metrics as they achieve workflow completion goals or meet the configured defaults and thresholds.
        """
        Base.data_n()

        eda_metrics = []

        eda_index = ['Speedml Release']
        eda_metrics.append([_RELEASE, 'Visit https://speedml.com for release notes.'])

        nulls_by_features = Base.train.isnull().sum() + Base.test.isnull().sum()
        nulls = nulls_by_features[1].sum()
        if nulls:
            eda_index.append('Nulls')
            eda_metrics.append([nulls, 'Use feature.impute.'])

        skew = Base.train_n.skew()
        skew_upper = skew[skew > Base._config['outlier_threshold']]
        skew_lower = skew[skew < -Base._config['outlier_threshold']]
        if not skew_upper.empty:
            eda_index.append('Outliers Upper')
            eda_metrics.append(
                [skew_upper.axes[0].tolist(),
                 'Positive skew (> {}). Use feature.outliers(upper).'.format(
                     Base._config['outlier_threshold'])])
        if not skew_lower.empty:
            eda_index.append('Outliers Lower')
            eda_metrics.append(
                [skew_lower.axes[0].tolist(),
                 'Negative skew (< -{}). Use feature.outliers(lower).'.format(
                     Base._config['outlier_threshold'])])

        eda_index.append('Shape')
        feature_by_sample = Base.train.shape[1] / Base.train.shape[1]
        message = '#Features / #Samples > {}. Over-fitting.'.format(Base._config['overfit_threshold'])
        message = message if feature_by_sample < Base._config['overfit_threshold'] else ''
        eda_metrics.append([self.shape(), message])

        numerical_ratio = int(Base.train_n.shape[1] / Base.train.shape[1] * 100)
        if numerical_ratio < 100:
            eda_index.append('Numerical Ratio')
            eda_metrics.append(['{}%'.format(numerical_ratio),
                                  'Aim for 100% numerical.'])

        numerical_features = Base.train_n.columns.values

        if numerical_features != []:
            high_cardinality_num = []
            categorical_num = []
            continuous = []
            for feature in numerical_features:
                repeating = Base.train[feature].value_counts()
                if repeating.count() > (Base._config['unique_ratio'])/100*Base.train.shape[0]:
                    continuous.append(feature)
                    if feature == Base.target:
                        target_analysis = ['Model ready.',
                                           'Use regression models.']
                    continue
                if repeating.count() > Base._config['high_cardinality']:
                    high_cardinality_num.append(feature)
                    if feature == Base.target:
                        target_analysis = ['Pre-process.',
                                           'Dimensionality reduction?']
                    continue
                if repeating.count() > 1:
                    categorical_num.append(feature)
                    if feature == Base.target:
                        target_analysis = ['Model ready.',
                                           'Use classification models.']
                    continue

            if high_cardinality_num:
                eda_index.append('Numerical High-cardinality')
                eda_metrics.append([
                    high_cardinality_num,
                    '(>{}) categories. Use feature.density'.format(
                        Base._config['high_cardinality'])])

            if categorical_num:
                eda_index.append('Numerical Categorical')
                eda_metrics.append([
                    categorical_num,
                    ' Use plot.ordinal.'])

            if continuous:
                eda_index.append('Numerical Continuous')
                eda_metrics.append([
                    continuous,
                    '~{}% unique. Use plot.continuous.'.format(Base._config['unique_ratio'])])

        if Base.train_n.shape[1] != Base.train.shape[1]:
            text_features = []
            text_features = list(set(Base.train.columns.values) - set(numerical_features))

            if text_features != []:
                high_cardinality_text = []
                categorical_text = []
                text = []
                for feature in text_features:
                    repeating = Base.train[feature].value_counts()
                    if repeating.count() > (Base._config['unique_ratio'])/100*Base.train.shape[0]:
                        text.append(feature)
                        if feature == Base.target:
                            target_analysis = [
                                'ERROR.',
                                'Unique text cannot be a target variable.']
                        continue
                    if repeating.count() > Base._config['high_cardinality']:
                        high_cardinality_text.append(feature)
                        if feature == Base.target:
                            target_analysis = [
                                'Pre-process.',
                                'Use feature.labels.']
                        continue
                    if repeating.count() > 1:
                        categorical_text.append(feature)
                        if feature == Base.target:
                            target_analysis = [
                                'Pre-process.',
                                'Use feature.labels or feature.mapping.']
                        continue

                if high_cardinality_text:
                    eda_index.append('Text High-cardinality')
                    eda_metrics.append([
                        high_cardinality_text,
                        '(>{}) categories. Use feature.labels.'.format(Base._config['high_cardinality'])])

                if categorical_text:
                    eda_index.append('Text Categorical')
                    eda_metrics.append([
                        categorical_text,
                        'Use feature.labels or feature.mapping.'])

                if text:
                    eda_index.append('Text Unique')
                    eda_metrics.append([
                        text,
                        '~{}% unique. Use feature.extract or feature.drop.'.format(Base._config['unique_ratio'])])

        eda_index += ['Target Analysis ({})'.format(Base.target)]
        eda_metrics.append(target_analysis)

        eda_df = pd.DataFrame(eda_metrics,
                                index=eda_index,
                                columns=['Results', 'Observations'])

        return eda_df

    def shape(self):
        """
        Print shape (samples, features) of train, test datasets and number of numerical features in each dataset.
        """
        Base.data_n()
        message = 'train {} | test {}'
        return message.format(Base.train.shape, Base.test.shape)

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
