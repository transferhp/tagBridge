from base import SRecommender
import scipy.sparse as ssp
cimport numpy as np
import numpy as np


class SVD(SRecommender):
    def __init__(self,n_factors=100, n_epochs=20, biased=True, lr_all=.005,
                 reg_all=.02, lr_bu=None, lr_bi=None, lr_pu=None, lr_qi=None,
                 reg_bu=None, reg_bi=None, reg_pu=None, reg_qi=None,
                 verbose=False):
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.biased = biased
        self.lr_bu = lr_bu if lr_bu is not None else lr_all
        self.lr_bi = lr_bi if lr_bi is not None else lr_all
        self.lr_pu = lr_pu if lr_pu is not None else lr_all
        self.lr_qi = lr_qi if lr_qi is not None else lr_all
        self.reg_bu = reg_bu if reg_bu is not None else reg_all
        self.reg_bi = reg_bi if reg_bi is not None else reg_all
        self.reg_pu = reg_pu if reg_pu is not None else reg_all
        self.reg_qi = reg_qi if reg_qi is not None else reg_all
        self.verbose = verbose

    def _fit(self):
        self.sgd()

    def _predict(self, u, i):
        pred = np.mean(self.train_mat) if self.biased else 0
        pred += self.bu[u] + self.bi[i] + self.pu[u].dot(self.qi[i])
        return pred

    def sgd(self):
        # user biases
        cdef np.ndarray[np.double_t] bu = np.zeros(self.train_mat.shape[0], np.double)
        # item biases
        cdef np.ndarray[np.double_t] bi = np.zeros(self.train_mat.shape[1], np.double)
        # user factors
        cdef np.ndarray[np.double_t, ndim=2] pu = (
                        np.zeros((self.train_mat.shape[0], self.n_factors), np.double) + .1
            )
        # item factors
        cdef np.ndarray[np.double_t, ndim=2] qi = (
                        np.zeros((self.train_mat.shape[1], self.n_factors), np.double)
                        + .1
            )


        cdef int u = 0
        cdef int i = 0
        cdef double r = 0
        cdef double global_mean = np.mean(self.train_mat)
        cdef double err = 0

        cdef double lr_bu = self.lr_bu
        cdef double lr_bi = self.lr_bi
        cdef double lr_pu = self.lr_pu
        cdef double lr_qi = self.lr_qi

        cdef double reg_bu = self.reg_bu
        cdef double reg_bi = self.reg_bi
        cdef double reg_pu = self.reg_pu
        cdef double reg_qi = self.reg_qi

        cdef int f = 0
        cdef double dot = 0
        cdef double puf = 0
        cdef double qif = 0

        if not self.biased:
            global_mean = 0

            for current_epoch in range(self.n_epochs):
                if self.verbose:
                    print(" Processing epoch {}".format(current_epoch))
                for u, i, r in zip(*ssp.find(self.train_mat)):
                    # compute current error
                    dot = 0  # <q_i, p_u>
                    for f in range(self.n_factors):
                        dot += qi[i, f] * pu[u, f]
                        err = r - (global_mean + bu[u] + bi[i] + dot)

                        # update biases
                        if self.biased:
                            bu[u] += lr_bu * (err - reg_bu * bu[u])
                            bi[i] += lr_bi * (err - reg_bi * bi[i])

                        # update factors
                        for f in range(self.n_factors):
                            puf = pu[u, f]
                            qif = qi[i, f]
                            pu[u, f] += lr_pu * (err * qif - reg_pu * puf)
                            qi[i, f] += lr_qi * (err * puf - reg_qi * qif)

        self.bu = bu
        self.bi = bi
        self.pu = pu
        self.qi = qi
