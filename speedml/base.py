import numpy as np

class Base(object):
    def data_n():
        Base.train_n = Base.train.select_dtypes(include=[np.number])
        Base.test_n = Base.test.select_dtypes(include=[np.number])
