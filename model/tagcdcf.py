#!user/bin/env python
# Author: Peng Hao
# Email: haopengbuaa@gmail.com
# Created: 23/11/16
# _*_ coding: utf-8 _*_
#

import numpy as np
import scipy.sparse as ssp
from base import CRecommender
from util.util import single_sigmoid, sigmoid, sigmoid_grad, normalize
from math import sqrt
from joblib import Parallel, delayed
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity
np.random.seed(999)


class TagCDCF(CRecommender):
    """
    This class is used to implement the algorithm proposed in the
    following paper:
    'Yue Shi, Martha Larson, Alan Hanjalic, 2011,
    Tags as Bridges between Domains:
    Improving Recommendation with Tag-Induced Cross-Domain
    Collaborative Filtering.'
    """

    def __init__(self,
                 reg_cross_u=0.001,
                 reg_cross_i=0.001,
                 reg_lambda=0.01,
                 num_factor=10,
                 verbose=True):
        super(TagCDCF, self).__init__()
        # regularization parameter to control cross domain user similarity
        self.reg_cross_u = reg_cross_u
        # regularization parameter to control cross domain item similarity
        self.reg_cross_i = reg_cross_i
        # regularization parameter for latent feature matrices
        self.reg_lambda = reg_lambda
        # the number of maximum iteration times
        self.num_factor = num_factor
        # display training process
        self.verbose = verbose


    def _fit(self, args):
        src_user_tag_mat, tar_user_tag_mat, \
        src_item_tag_mat, tar_item_tag_mat = args
        # compute cross-domain similarities
        self.cross_user_sim = cosine_similarity(src_user_tag_mat,
                                                tar_user_tag_mat,
                                                dense_output=False)
        self.cross_item_sim = cosine_similarity(src_item_tag_mat,
                                                tar_item_tag_mat,
                                                dense_output=False)
        # set indicator matrix for cross domain similarity matrices
        self.user_sim_indicator = self.cross_user_sim.sign()
        self.item_sim_indicator = self.cross_item_sim.sign()

        # initialize latent user and item feature matrices
        self.src_user = np.random.normal(scale=0.5, size=(self.src_tr_rate_mat.shape[0],
                                                          self.num_factor))
        self.src_item = np.random.normal(scale=0.5, size=(self.src_tr_rate_mat.shape[1],
                                                          self.num_factor))
        self.tar_user = np.random.normal(scale=0.5, size=(self.tar_tr_rate_mat.shape[0],
                                                          self.num_factor))
        self.tar_item = np.random.normal(scale=0.5, size=(self.tar_tr_rate_mat.shape[1],
                                                          self.num_factor))
        # set indicator matrix
        self.src_rate_indicator = self.src_tr_rate_mat.sign()
        self.tar_rate_indicator = self.tar_tr_rate_mat.sign()

        # normalize rating matrix values to [0,1]
        normalize(self.tar_tr_rate_mat)

        # train the model
        self._train()

    def _train(self):
        err_last = float('Nan')
        step = 0.0001
        t = 0
        old_src_u = np.copy(self.src_user)
        old_src_i = np.copy(self.src_item)
        old_tar_u = np.copy(self.tar_user)
        old_tar_i = np.copy(self.tar_item)
        while t < self.maxIter:
            step *= 2.0
            temp = self._gradient_src_user()
            self.src_user -= step * (temp + self.reg_lambda * self.src_user)
            temp = self._gradient_src_item()
            self.src_item -= step * (temp + self.reg_lambda * self.src_item)
            temp = self._gradient_tar_user()
            self.tar_user -= step * (temp + self.reg_lambda * self.tar_user)
            temp = self._gradient_tar_item()
            self.tar_item -= step * (temp + self.reg_lambda * self.tar_item)
            err = self._loss()
            print err
            while err > err_last:
                step *= 0.5
                #copy
                self.src_user = np.copy(old_src_u)
                self.src_item = np.copy(old_src_i)
                self.tar_user = np.copy(old_tar_u)
                self.tar_item = np.copy(old_tar_i)
                temp = self._gradient_src_user()
                self.src_user -= step * (temp + self.reg_lambda * self.src_user)
                temp = self._gradient_src_item()
                self.src_item -= step * (temp + self.reg_lambda * self.src_item)
                temp = self._gradient_tar_user()
                self.tar_user -= step * (temp + self.reg_lambda * self.tar_user)
                temp = self._gradient_tar_item()
                self.tar_item -= step * (temp + self.reg_lambda * self.tar_item)
                err = self._loss()
                print err
            # keep temp results
            old_src_u = np.copy(self.src_user)
            old_src_i = np.copy(self.src_item)
            old_tar_u = np.copy(self.tar_user)
            old_tar_i = np.copy(self.tar_item)
            del_err = (err_last - err) / err_last
            if del_err < self.tol:
                break
            else:
                t += 1
                err_last = err
                # Make predictions with updated parameters
                predictions = [self.predict(uidx, iidx) for uidx, iidx in
                               zip(*self.tar_tr_rate_mat.nonzero())]
                train_rmse = sqrt(mean_squared_error(
                        self.tar_tr_rate_mat.data, predictions))
                if self.verbose:
                    print("[Epoch {0}/{1}] | Train rmse={2:.4f} ".format(t,
                                                                         self.maxIter,
                                                                         train_rmse))

    def _predict(self, u, i):
        """
        Make predictions.

        Returns
        -----------------
        predictions : numpy.ndarray
            The prediction on
        """
        est = single_sigmoid(self.tar_user[u].dot(self.tar_item[i])) * self.rmax
        return est

    def _loss(self):
        """
        compute objective function.

        Returns
        ----------
        err: float
        Computed loss function value.
        """
        UsVs = sigmoid(np.dot(self.src_user, self.src_item.T))
        UtVt = sigmoid(np.dot(self.tar_user, self.tar_item.T))
        src_acc_sum = UsVs[self.src_tr_rate_mat.nonzero()] - self.src_tr_rate_mat.data
        tar_acc_sum = UtVt[self.tar_tr_rate_mat.nonzero()] - self.tar_tr_rate_mat.data
        e1 = np.sum((src_acc_sum)**2) + np.sum((tar_acc_sum)**2)

        UsUt = sigmoid(np.dot(self.src_user, self.tar_user.T))
        e2 = np.sum((UsUt[self.cross_user_sim.nonzero()] - self.cross_user_sim.data) ** 2)

        VsVt = sigmoid(np.dot(self.src_item, self.tar_item.T))
        e3 = np.sum((VsVt[self.cross_item_sim.nonzero()] - self.cross_item_sim.data) ** 2)

        e4 = (np.sum(self.src_user ** 2) +
              np.sum(self.tar_user ** 2) +
              np.sum(self.src_item ** 2) +
              np.sum(self.tar_item ** 2)
              )

        err = (0.5 * e1 +
               0.5 * self.reg_cross_u * e2 +
               0.5 * self.reg_cross_i * e3 +
               0.5 * self.reg_lambda * e4)
        return err

    def _gradient_src_user(self, n_jobs=8, batch_size=500):
        """
        Update latent user feature factors of source domain.

        Returns
        --------------------
        tmp: numpy.ndarray
            A updated source user latent feature matrix.
        """
        m, f = self.src_user.shape  # m: n_users, f: n_factors

        start_idx = range(0, m, batch_size)
        end_idx = start_idx[1:] + [m]
        res = Parallel(n_jobs=n_jobs)(
                delayed(_batch_grad_src_user)(lo, hi, f, self.src_user, self.src_item,
                                              self.src_rate_indicator,
                                              self.src_tr_rate_mat, self.tar_user,
                                              self.user_sim_indicator,
                                              self.cross_user_sim,
                                              self.reg_cross_u
                                              )
                for lo, hi in zip(start_idx, end_idx))
        grad_src_user = np.vstack(res)
        return grad_src_user

    def _gradient_src_item(self, n_jobs=8, batch_size=500):
        """
        Update latent item feature factors of source domain.

        Returns
        --------------------
        tmp: numpy.ndarray
            A updated source item latent feature matrix.
        """
        m, f = self.src_item.shape  # m: n_items, f: n_factors

        start_idx = range(0, m, batch_size)
        end_idx = start_idx[1:] + [m]
        res = Parallel(n_jobs=n_jobs)(
                delayed(_batch_grad_src_item)(lo, hi, f, self.src_user, self.src_item,
                                              self.src_rate_indicator,
                                              self.src_tr_rate_mat, self.tar_item,
                                              self.item_sim_indicator,
                                              self.cross_item_sim,
                                              self.reg_cross_i
                                              )
                for lo, hi in zip(start_idx, end_idx))
        grad_src_item = np.vstack(res)
        return grad_src_item

    def _gradient_tar_user(self, n_jobs=8, batch_size=500):
        """
        Update latent user feature factors of target domain.

        Returns
        --------------------
        tmp: numpy.ndarray
            A updated target user latent feature matrix.
        """
        m, f = self.tar_user.shape  # m: n_items, f: n_factors

        start_idx = range(0, m, batch_size)
        end_idx = start_idx[1:] + [m]
        res = Parallel(n_jobs=n_jobs)(
                delayed(_batch_grad_tar_user)(lo, hi, f, self.tar_user, self.tar_item,
                                              self.tar_rate_indicator,
                                              self.tar_tr_rate_mat, self.src_user,
                                              self.user_sim_indicator,
                                              self.cross_user_sim,
                                              self.reg_cross_u
                                              )
                for lo, hi in zip(start_idx, end_idx))
        grad_tar_user = np.vstack(res)
        return grad_tar_user

    def _gradient_tar_item(self, n_jobs=8, batch_size=500):
        """
        Update latent item feature factors of target domain.

        Returns
        --------------------
        tmp: numpy.ndarray
            A updated target item latent feature matrix.
        """
        m, f = self.tar_item.shape  # m: n_items, f: n_factors

        start_idx = range(0, m, batch_size)
        end_idx = start_idx[1:] + [m]
        res = Parallel(n_jobs=n_jobs)(
                delayed(_batch_grad_tar_item)(lo, hi, f, self.tar_user, self.tar_item,
                                              self.tar_rate_indicator,
                                              self.tar_tr_rate_mat, self.src_item,
                                              self.item_sim_indicator,
                                              self.cross_item_sim,
                                              self.reg_cross_i
                                              )
                for lo, hi in zip(start_idx, end_idx))
        grad_tar_item = np.vstack(res)
        return grad_tar_item



def _batch_grad_src_user(lo, hi, f, src_user, src_item, src_rate_indicator,
                         src_tr_rate_mat, tar_user, user_sim_indicator, cross_user_sim,
                         reg_cross_u):
    tmp = np.empty((hi - lo, f), dtype=src_user.dtype)
    UVT = np.dot(src_user[lo:hi], src_item.T)
    part1 = (src_rate_indicator[lo:hi].multiply(
            sigmoid(UVT)) - src_tr_rate_mat[lo:hi]).multiply(
            sigmoid_grad(UVT)).dot(src_item)

    UUT = np.dot(src_user[lo:hi], tar_user.T)
    part2 = (user_sim_indicator[lo:hi].multiply(
            sigmoid(UUT)) - cross_user_sim[lo:hi]).multiply(
            sigmoid_grad(UUT)).dot(tar_user)
    tmp[0:hi-lo] = part1 + reg_cross_u * part2
    return tmp


def _batch_grad_src_item(lo, hi, f, src_user, src_item, src_rate_indicator,
                         src_tr_rate_mat, tar_item, item_sim_indicator, cross_item_sim,
                         reg_cross_i):
    tmp = np.empty((hi - lo, f), dtype=src_item.dtype)
    UVT = np.dot(src_user, src_item[lo:hi].T)
    part1 = ((src_rate_indicator.tocsc()[:,lo:hi].multiply(
            sigmoid(UVT)) - src_tr_rate_mat.tocsc()[:, lo:hi]).multiply(
            sigmoid_grad(UVT))).T.dot(src_user)

    VVT = np.dot(src_item[lo:hi], tar_item.T)
    part2 = (item_sim_indicator[lo:hi].multiply(
            sigmoid(VVT)) - cross_item_sim[lo:hi]).multiply(sigmoid_grad(
            VVT)).dot(tar_item)
    tmp[0:hi-lo] = part1 + reg_cross_i * part2
    return tmp


def _batch_grad_tar_user(lo, hi, f, tar_user, tar_item, tar_rate_indicator,
                         tar_tr_rate_mat, src_user, user_sim_indicator, cross_user_sim,
                         reg_cross_u):
    tmp = np.empty((hi - lo, f), dtype=tar_user.dtype)
    UVT = np.dot(tar_user[lo:hi], tar_item.T)
    part1 = (tar_rate_indicator[lo:hi].multiply(sigmoid(UVT)) -
             tar_tr_rate_mat[lo:hi]).multiply(
            sigmoid_grad(UVT)).dot(tar_item)

    UUT = np.dot(src_user, tar_user[lo:hi].T)
    part2 = ((user_sim_indicator.tocsc()[:, lo:hi].multiply(
            sigmoid(UUT)) - cross_user_sim.tocsc()[:, lo:hi]).multiply(
            sigmoid_grad(UUT))).T.dot(src_user)

    tmp[0:hi-lo] = part1 + reg_cross_u * part2
    return tmp


def _batch_grad_tar_item(lo, hi, f, tar_user, tar_item, tar_rate_indicator,
                         tar_tr_rate_mat, src_item, item_sim_indicator, cross_item_sim,
                         reg_cross_i):
    tmp = np.empty((hi - lo, f), dtype=tar_item.dtype)
    UVT = np.dot(tar_user, tar_item[lo:hi].T)
    part1 = ((tar_rate_indicator.tocsc()[:, lo:hi].multiply(
            sigmoid(UVT)) - tar_tr_rate_mat.tocsc()[:, lo:hi]).multiply(
            sigmoid_grad(UVT))).T.dot(tar_user)

    VVT = np.dot(src_item, tar_item[lo:hi].T)
    part2 = ((item_sim_indicator.tocsc()[:, lo:hi].multiply(
            sigmoid(VVT)) - cross_item_sim.tocsc()[:, lo:hi]).multiply(
            sigmoid_grad(VVT))).T.dot(src_item)

    tmp[0:hi-lo] = part1 + reg_cross_i * part2
    return tmp
