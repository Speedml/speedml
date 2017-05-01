import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns

import xgboost as xgb
from sklearn.ensemble import ExtraTreesClassifier

class Plot(object):
    def __init__(self, train, test, target, uid):
        '''
        Only init the class variables which do not change during their lifetime.
        Rest of the variables are set in the parent Speedml class when these are updated.
        '''
        self.train = train
        self.test = test
        self.target = target
        self.uid = uid

    def data_n(self):
        self.train_n = self.train.select_dtypes(include=[np.number])
        self.test_n = self.test.select_dtypes(include=[np.number])

    def distribute(self):
        features = len(self.train_n.columns)
        plt.figure()
        self.train_n.hist(figsize=(features * 1.1, features * 1.1));

    def correlate(self):
        corr = self.train_n.corr()
        features = self.train_n.shape[1]
        cell_size = features * 1.2 if features < 12 else features * 0.5
        plt.figure(figsize=(cell_size, cell_size))
        sns.heatmap(corr, vmax=1, annot=True if features < 12 else False, square=True)
        plt.title('feature correlations in train_n dataset');

    def ordinal(self, a):
        plt.figure(figsize=(8,4))
        sns.violinplot(x=self.target, y=a, data=self.train_n)
        plt.xlabel(self.target, fontsize=12)
        plt.ylabel(a, fontsize=12)
        plt.show();

    def continuous(self, a):
        plt.figure(figsize=(8,6))
        plt.scatter(range(self.train_n.shape[0]), np.sort(self.train_n[a].values))
        plt.xlabel('Samples', fontsize=12)
        plt.ylabel(a, fontsize=12)
        plt.show();

    def model_ranks(self):
        plt.xlabel('Accuracy')
        plt.title('Classifier Accuracy')

        sns.set_color_codes("muted")
        sns.barplot(x='Accuracy', y='Classifier', data=self.model_ranking, color="b");

    def _create_feature_map(self, features):
        outfile = open('data/xgb.fmap', 'w')
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
        self.data_n()
        X = self.train_n
        y = X[self.target].copy()
        X = X.drop([self.target], axis=1)
        model = ExtraTreesClassifier()
        model.fit(X, y)
        self._plot_importance(X.columns, model.feature_importances_)

    def xgb_importance(self):
        self.data_n()
        X = self.train_n
        X = X.drop([self.target], axis=1)
        self._create_feature_map(X.columns)
        fscore = self.xgb_model.booster().get_fscore(fmap='data/xgb.fmap')
        self._plot_importance(list(fscore.keys()), list(fscore.values()))
