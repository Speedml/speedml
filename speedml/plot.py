"""
Speedml Plot component with methods that work on plots or the Exploratory Data Analysis (EDA) workflow. Contact author https://twitter.com/manavsehgal. Code, docs and demos https://speedml.com.
"""

from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
from builtins import *

from .base import Base

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns

import xgboost as xgb

from sklearn.ensemble import ExtraTreesClassifier

class Plot(Base):
    def crosstab(self, x, y):
        """
        Return a dataframe cross-tabulating values from feature ``x`` and ``y``.
        """
        return pd.crosstab(Base.train[x], Base.train[y])

    def bar(self, x, y):
        """
        Bar plot ``x`` across ``y`` feature values.
        """
        plt.figure(figsize=(8,4))
        sns.barplot(x, y, data=Base.train)
        plt.xlabel(x, fontsize=12)
        plt.ylabel(y, fontsize=12)
        plt.show();

    def strip(self, x, y):
        """
        Stripplot plot ``x`` across ``y`` feature values.
        """
        plt.figure(figsize=(8,4))
        sns.stripplot(x, y, hue=Base.target, data=Base.train, jitter=True)
        plt.xlabel(x, fontsize=12)
        plt.ylabel(y, fontsize=12)
        plt.show();

    def distribute(self):
        """
        Plot multiple feature distribution histogram plots for all numeric features. This helps understand skew of distribution from normal to quickly and relatively identify outliers in the dataset.
        """
        Base.data_n()
        features = len(Base.train_n.columns)
        plt.figure()
        Base.train_n.hist(figsize=(features * 1.1, features * 1.1));

    def correlate(self):
        """
        Plot correlation matrix heatmap for numerical features of the training dataset. Use this plot to understand if certain features are duplicate, are of low importance, or possibly high importance for our model.
        """
        Base.data_n()
        corr = Base.train_n.corr()
        features = Base.train_n.shape[1]
        cell_size = features * 1.2 if features < 9 else features * 0.5
        plt.figure(figsize=(cell_size, cell_size))
        sns.heatmap(corr, vmax=1, linewidths=.5, square=True,
                    annot=True if features < 9 else False)
        plt.title('feature correlations in train_n dataset');

    def ordinal(self, y):
        """
        Plot ordinal features (categorical numeric) using Violin plot against target feature. Use this to determine outliers within ordinal features spread across associated target feature values.
        """
        Base.data_n()
        plt.figure(figsize=(8,4))
        sns.violinplot(x=Base.target, y=y, data=Base.train_n)
        plt.xlabel(Base.target, fontsize=12)
        plt.ylabel(y, fontsize=12)
        plt.show();

    def continuous(self, y):
        """
        Plot continuous features (numeric) using scatter plot. Use this to determine outliers within continuous features.
        """
        Base.data_n()
        plt.figure(figsize=(8,6))
        plt.scatter(range(Base.train_n.shape[0]), np.sort(Base.train_n[y].values))
        plt.xlabel('Samples', fontsize=12)
        plt.ylabel(y, fontsize=12)
        plt.show();

    def model_ranks(self):
        """
        Plot ranking among accuracy offered by various models based on our datasets.
        """
        plt.xlabel('Accuracy')
        plt.title('Classifier Accuracy')

        sns.set_color_codes("muted")
        sns.barplot(x='Accuracy', y='Classifier', data=Base.model_ranking, color="b");

    def _create_feature_map(self, features):
        outfile = open(Base._config['outpath'] + 'xgb.fmap', 'w')
        i = 0
        for feat in features:
            outfile.write('{0}\t{1}\tq\n'.format(i, feat))
            i = i + 1

        outfile.close()

    def _plot_importance(self, feature, importance):
        ranking = pd.DataFrame({'Feature': feature,
                               'Importance': importance})
        ranking = ranking.sort_values(by='Importance', ascending=False)
        fig, ax = plt.subplots(figsize=(9, ranking.shape[0]/2.5))
        y_pos = np.arange(ranking.shape[0])
        importance = ranking['Importance']
        ax.barh(y_pos, importance, align='center')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(ranking['Feature'])
        ax.invert_yaxis()
        ax.set_xlabel('Importance')
        ax.set_title('Feature Importance')
        plt.show()

    def importance(self):
        """
        Plot importance of features based on ExtraTreesClassifier.
        """
        Base.data_n()
        X = Base.train_n
        y = X[Base.target].copy()
        X = X.drop([Base.target], axis=1)
        model = ExtraTreesClassifier()
        model.fit(X, y)
        self._plot_importance(X.columns, model.feature_importances_)

    def xgb_importance(self):
        """
        Plot importance of features based on XGBoost.
        """
        Base.data_n()
        X = Base.train_n
        X = X.drop([Base.target], axis=1)
        self._create_feature_map(X.columns)
        fscore = Base.xgb_model.get_booster().get_fscore(fmap=Base._config['outpath'] + 'xgb.fmap')
        self._plot_importance(list(fscore.keys()), list(fscore.values()))
