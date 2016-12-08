#!user/bin/env python
# _*_ coding: utf-8 _*_
# Author: Peng Hao
# Email: haopengbuaa@gmail.com
# Created: 25/11/16


import pandas as pd
from math import sqrt
from sklearn.metrics import mean_squared_error


class Recommender(object):
    def __init__(self, tol=1e-5, maxIter=100):
        self.tol = tol
        self.maxIter = maxIter

    def fit(self):
        self._fit()

    def _fit(self):
        raise NotImplementedError()

    def predict(self):
        return self._predict()

    def _predict(self):
        raise NotImplementedError()

    def evaluate(self, prediction, tar_te_mat):
        rmse = sqrt(mean_squared_error(tar_te_mat.data, prediction[tar_te_mat.nonzero()]))
        return rmse
