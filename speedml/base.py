import numpy as np

class Base(object):
    @staticmethod
    def data_n():
        """
        Updates train_n and test_n numeric datasets (used for model data creation) based on numeric datatypes from train and test datasets.
        """
        Base.train_n = Base.train.select_dtypes(include=[np.number])
        Base.test_n = Base.test.select_dtypes(include=[np.number])
