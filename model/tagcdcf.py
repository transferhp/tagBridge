#!user/bin/env python
# Author: Peng Hao
# Email: haopengbuaa@gmail.com
# Created: 23/11/16
# _*_ coding: utf-8 _*_
#

import numpy as np
from base import Recommender
from util.util import sigmoid, sigmoid_grad, normalize, bound_prediction
from math import sqrt
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity
np.random.seed(999)


class TagCDCF(Recommender):
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

    def _find_col_id(self, mdict, mkeys):
        """Find column index by column name."""
        return [mdict[key] for key in mkeys]

    def _fit(self):
        # collect common tags
        com_tags = set(self.src_tagid.keys()) & set(self.tar_tagid.keys())

        # slicing tagging matrix by common tags
        tar_cols = self._find_col_id(self.tar_tagid, com_tags)
        src_cols = self._find_col_id(self.src_tagid, com_tags)
        src_user_comTag_mat = self.src_user_tag_mat.tocsc()[:, src_cols]
        tar_user_comTag_mat = self.tar_user_tag_mat.tocsc()[:, tar_cols]
        src_item_comTag_mat = self.src_item_tag_mat.tocsc()[:, src_cols]
        tar_item_comTag_mat = self.tar_item_tag_mat.tocsc()[:, tar_cols]

        # compute cross-domain similarities
        self.cross_user_sim = cosine_similarity(src_user_comTag_mat.sign(),
                                                tar_user_comTag_mat.sign(),
                                                dense_output=False)
        self.cross_item_sim = cosine_similarity(src_item_comTag_mat.sign(),
                                                tar_item_comTag_mat.sign(),
                                                dense_output=False)
        # set indicator matrix for cross domain similarity matrices
        self.user_sim_indicator = self.cross_user_sim.sign()
        self.item_sim_indicator = self.cross_item_sim.sign()

        # initialize latent user and item feature matrices
        self.src_user = np.random.rand(self.src_tr_rate_mat.shape[0], self.num_factor)
        self.src_item = np.random.rand(self.src_tr_rate_mat.shape[1], self.num_factor)
        self.tar_user = np.random.rand(self.tar_tr_rate_mat.shape[0], self.num_factor)
        self.tar_item = np.random.rand(self.tar_tr_rate_mat.shape[1], self.num_factor)
        # set indicator matrix
        self.src_rate_indicator = self.src_tr_rate_mat.sign()
        self.tar_rate_indicator = self.tar_tr_rate_mat.sign()

        # normalize rating matrix values to [0,1]
        self.max_val, self.min_val, self.tar_tr_rate_mat = normalize(self.tar_tr_rate_mat)

        # train the model
        self._train()

    def _train(self):
        err_last = float('Nan')
        step = 0.01
        t = 0
        while t < self.maxIter:
            step *= 1.05
            temp = self._gradient_src_user()
            nextU1 = self.src_user - step * (
                temp + self.reg_lambda * self.src_user)
            temp = self._gradient_src_item()
            nextV1 = self.src_item - step * (
                temp + self.reg_lambda * self.src_item)
            temp = self._gradient_tar_user()
            nextU2 = self.tar_user - step * (
                temp + self.reg_lambda * self.tar_user)
            temp = self._gradient_tar_item()
            nextV2 = self.tar_item - step * (
                temp + self.reg_lambda * self.tar_item)
            err = self._loss(nextU1, nextV1, nextU2, nextV2)
            print err
            while err > err_last:
                step *= 0.95
                temp = self._gradient_src_user()
                nextU1 = self.src_user - step * (
                    temp + self.reg_lambda * self.src_user)
                temp = self._gradient_src_item()
                nextV1 = self.src_item - step * (
                    temp + self.reg_lambda * self.src_item)
                temp = self._gradient_tar_user()
                nextU2 = self.tar_user - step * (
                    temp + self.reg_lambda * self.tar_user)
                temp = self._gradient_tar_item()
                nextV2 = self.tar_item - step * (
                    temp + self.reg_lambda * self.tar_item)
                err = self._loss(nextU1, nextV1, nextU2, nextV2)
                print err
            self.src_user = nextU1
            self.src_item = nextV1
            self.tar_user = nextU2
            self.tar_item = nextV2
            del_err = (err_last - err) / err_last
            if del_err < self.tol:
                break
            else:
                t += 1
                err_last = err
                # Make predictions with updated parameters
                predictions = self.predict()
                train_rmse = sqrt(mean_squared_error(
                        self.tar_tr_rate_mat.data,
                        predictions[self.tar_tr_rate_mat.nonzero()]))
                if self.verbose:
                    print("[Epoch {0}/{1}] | Train rmse={2:.4f} ".format(t,
                                                                         self.maxIter,
                                                                         train_rmse))

    def _predict(self):
        """
        Make predictions.

        Returns
        -----------------
        predictions : numpy.ndarray
            The prediction on
        """
        predictions = sigmoid(self.tar_user.dot(self.tar_item.T)) * self.max_val
        predictions = bound_prediction(predictions, self.max_val, self.min_val)
        return predictions

    def _loss(self, u1, v1, u2, v2):
        """
        compute objective function.

        Parameters
        ----------
        u1 : numpy.ndarray
        User latent feature matrix for source domain.
        v1 : numpy.ndarray
        Item latent feature matrix for source domain.
        u2 : numpy.ndarray
        User latent feature matrix for target domain.
        v2 : numpy.ndarray
        Item latent feature matrix for target domain.

        Returns
        ----------
        err: float
        Computed loss function value.
        """
        e1 = (
            (self.src_rate_indicator.multiply(sigmoid(np.dot(u1, v1.T))) -
             self.src_tr_rate_mat).power(2).sum() +
            (self.tar_rate_indicator.multiply(sigmoid(np.dot(u2, v2.T))) -
             self.tar_tr_rate_mat).power(2).sum()
        )

        e2 = (self.user_sim_indicator.multiply(
                sigmoid(np.dot(u1, u2.T))) -
              self.cross_user_sim).power(2).sum()

        e3 = (self.item_sim_indicator.multiply(
                sigmoid(np.dot(v1, v2.T))) -
              self.cross_item_sim).power(2).sum()

        e4 = (np.sum(u1 ** 2) +
              np.sum(u2 ** 2) +
              np.sum(v1 ** 2) +
              np.sum(v2 ** 2)
              )

        err = (0.5 * e1 +
               0.5 * self.reg_cross_u * e2 +
               0.5 * self.reg_cross_i * e3 +
               0.5 * self.reg_lambda * e4)
        return err

    def _gradient_src_user(self):
        """
        Update latent user feature factors of source domain.

        Returns
        --------------------
        tmp: numpy.ndarray
            A updated source user latent feature matrix.
        """
        tmp = (
            (self.src_rate_indicator.multiply(sigmoid(np.dot(self.src_user, self.src_item.T))) -
             self.src_tr_rate_mat).multiply(sigmoid_grad(
                    np.dot(self.src_user, self.src_item.T))).dot(self.src_item) +
            self.reg_cross_u * (self.user_sim_indicator.multiply(sigmoid(
                    np.dot(self.src_user, self.tar_user.T))) - self.cross_user_sim).multiply(
                    sigmoid_grad(np.dot(self.src_user, self.tar_user.T))).dot(
                    self.tar_user)
        )

        return tmp

    def _gradient_src_item(self):
        """
        Update latent item feature factors of source domain.

        Returns
        --------------------
        tmp: numpy.ndarray
            A updated source item latent feature matrix.
        """
        tmp = (
            (self.src_rate_indicator.multiply(sigmoid(np.dot(self.src_user, self.src_item.T))) -
             self.src_tr_rate_mat).multiply(sigmoid_grad(np.dot(self.src_user,
                                                     self.src_item.T))).T.dot(
                    self.src_user) +
            self.reg_cross_i * (self.item_sim_indicator.multiply(sigmoid(np.dot(self.src_item, self.tar_item.T))) -
                                self.cross_item_sim).multiply(sigmoid_grad(
                    np.dot(self.src_item, self.tar_item.T))).dot(self.tar_item)
        )

        return tmp

    def _gradient_tar_user(self):
        """
        Update latent user feature factors of target domain.

        Returns
        --------------------
        tmp: numpy.ndarray
            A updated target user latent feature matrix.
        """
        tmp = (
            (self.tar_rate_indicator.multiply(sigmoid(np.dot(self.tar_user, self.tar_item.T))) -
             self.tar_tr_rate_mat).multiply(sigmoid_grad(
                    np.dot(self.tar_user, self.tar_item.T))).dot(self.tar_item) +
            self.reg_cross_u * (self.user_sim_indicator.multiply(sigmoid(np.dot(self.src_user, self.tar_user.T))) -
                                self.cross_user_sim).multiply(sigmoid_grad(
                    np.dot(self.src_user, self.tar_user.T))).T.dot(self.src_user)
        )

        return tmp

    def _gradient_tar_item(self):
        """
        Update latent item feature factors of target domain.

        Returns
        --------------------
        tmp: numpy.ndarray
            A updated target item latent feature matrix.
        """
        tmp = (
            (self.tar_rate_indicator.multiply(sigmoid(np.dot(self.tar_user, self.tar_item.T))) -
             self.tar_tr_rate_mat).multiply(sigmoid_grad(np.dot(self.tar_user,
                                                     self.tar_item.T))).T.dot(
                    self.tar_user) +
            self.reg_cross_i * (self.item_sim_indicator.multiply(sigmoid(np.dot(self.src_item, self.tar_item.T))) -
                                self.cross_item_sim).multiply(sigmoid_grad(
                    np.dot(self.src_item, self.tar_item.T))).T.dot(self.src_item)
        )
        return tmp
