#!user/bin/env python
# _*_ coding: utf-8 _*_
# Author: Peng Hao
# Email: haopengbuaa@gmail.com
# Created: 25/11/16


import scipy.sparse as ssp
import numpy as np
from math import sqrt
from sklearn.metrics import mean_squared_error


class SRecommender(object):
    """Base class,
    which every single domain recommendation model has to inherit.
    """
    def __init__(self, tol=1e-5, maxIter=100):
        self.tol = tol
        self.maxIter = maxIter

    def _setup_input(self, train_mat):
        try:
            if isinstance(train_mat, ssp.csr_matrix):
                self.train_mat = train_mat
        except:
            self.train_mat = ssp.csr_matrix(train_mat)

        if self.train_mat.ndim < 2:
            raise ValueError('Training rating matrix should have two dimensions:'
                             + 'users*items')

        self.rmax = np.max(self.train_mat.data)
        self.rmin = np.min(self.train_mat.data)

    def fit(self, train_mat):
        """Train the model on training matrix.
        """
        # check data input
        self._setup_input(train_mat)
        # train the model
        self._fit()

    def _fit(self):
        raise NotImplementedError()

    def test(self, test_mat):
        """Test the model on given test matrix.
        """
        if not isinstance(test_mat, ssp.csr_matrix):
            test_mat = ssp.csr_matrix(test_mat)

        predictions = [self.predict(uidx, iidx)
                       for uidx, iidx in zip(*test_mat.nonzero())]
        return predictions


    def predict(self, u, i):
        est = self._predict(u, i)
        # clip estimation into range [self.rmin, self.rmax]
        est = min(self.rmin, est)
        est = max(self.rmax, est)
        return est

    def _predict(self, u, i):
        raise NotImplementedError()

    def evaluate(self, predictions, tar_te_mat):
        rmse = sqrt(mean_squared_error(tar_te_mat.data, predictions))
        return rmse


class CRecommender(object):
    """Base class,
    which every cross-domain recommendation model has to inherit.
    """
    def __init__(self, tol=1e-5, maxIter=100):
        self.tol = tol
        self.maxIter = maxIter

    def _setup_input(self, src_tr_rate_mat, tar_tr_rate_mat):
        try:
            if isinstance(src_tr_rate_mat, ssp.csr_matrix):
                self.src_tr_rate_mat = src_tr_rate_mat
        except:
            self.src_tr_rate_mat = ssp.csr_matrix(src_tr_rate_mat)

        try:
            if isinstance(tar_tr_rate_mat, ssp.csr_matrix):
                self.tar_tr_rate_mat = tar_tr_rate_mat
        except:
            self.tar_tr_rate_mat = ssp.csr_matrix(tar_tr_rate_mat)

        if self.src_tr_rate_mat.ndim < 2:
            raise ValueError('Source train rating matrix should have two dimensions:'
                             + 'users*items')
        if self.tar_tr_rate_mat.ndim < 2:
            raise ValueError('Target train rating matrix should have two dimensions:'
                             + 'users*items')

        self.rmax = np.max(self.tar_tr_rate_mat.data)
        self.rmin = np.min(self.tar_tr_rate_mat.data)

    def fit(self, src_tr_rate_mat, tar_tr_rate_mat, *args):
        """Train the model on training matrix.
        """
        # check data input
        self._setup_input(src_tr_rate_mat, tar_tr_rate_mat)
        # train the model
        self._fit(args)

    def _fit(self, args):
        raise NotImplementedError()

    def test(self, test_mat):
        """Test the model on given test matrix.
        """
        if not isinstance(test_mat, ssp.csr_matrix):
            test_mat = ssp.csr_matrix(test_mat)

        predictions = [self.predict(uidx, iidx)
                       for uidx, iidx in zip(*test_mat.nonzero())]
        return predictions

    def predict(self, u, i):
        est = self._predict(u, i)
        # clip estimation into range [self.rmin, self.rmax]
        if est < self.rmin:
            return self.rmin
        if est > self.rmax:
            return self.rmax
        return est

    def _predict(self, u, i):
        raise NotImplementedError()

    def evaluate(self, predictions, tar_te_mat):
        rmse = sqrt(mean_squared_error(tar_te_mat.data, predictions))
        return rmse