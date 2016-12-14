#!user/bin/env python
# _*_ coding: utf-8 _*_
# Author: Peng Hao
# Email: haopengbuaa@gmail.com
# Created: 1/12/16

from scipy.special import expit
import os
import math
import numpy as np
import pandas as pd
import scipy.sparse as ssp


def single_sigmoid(x):
    return 1 / (1 + math.exp(-x))

def sigmoid(x):
    return expit(x)


def sigmoid_grad(x):
    ex = sigmoid(x)
    y = ex / (1 + ex) ** 2
    return y


def read_dict(file_name):
    with open(file_name, 'rb') as f:
        mydict = dict()
        for line in f:
            line = line.strip()  # delete '\n'
            k, v = line.rsplit(':', 1)
            mydict[k] = int(v)
    return mydict


def split_by_user(df, ratio=0.75):
    """Train/test splitting base on users"""
    train = []
    test = []
    for u, g in df.groupby('user_id'):
        size = len(g)
        if size <= 3:  # have less than 3 ratings
            continue
        else:
            # Randomly select 20 percent of his/her rating scores
            # and corresponding tag assignments as test set.
            mask = np.random.rand(size) < ratio
            train.append(g[mask])
            test.append(g[~mask])

    # Concatenate sub data frames
    tr_df = pd.concat(train)
    te_df = pd.concat(test)
    return tr_df, te_df


def mapping2id(samples):
    """Mapping a list of samples to dictionary in form of {sample:id}"""
    return dict((sample, i) for (i, sample) in enumerate(samples))


def normalize(csr_mat):
    """Normalize sparse csr_matrix to make sure
    each value in the matrix is within the range of [0, 1].

    Parameters
    ---------------
    csr_mat: (n_users * n_items) sparse rating matrix
              ssp.csr_matrix
    """
    if not isinstance(csr_mat, ssp.csr_matrix):
        raise ValueError('Input matrix should be in sparse csr format.')

    nnz_data = csr_mat.data
    # max_val = np.max(nnz_data)
    # csr_mat.data /= max_val
    # minMax normalization
    csr_mat.data = (nnz_data - np.mean(nnz_data)) / np.std(nnz_data)


def bound_prediction(predictions, max_val, min_val):
    predictions[predictions > max_val] = max_val
    predictions[predictions < min_val] = min_val
    return predictions


def rating_mat_batch(df, uid, iid):
    row = []
    col = []
    value = []
    for _, line in df.iterrows():
        u = int(line['user_id'])
        i = int(line['item_id'])
        rid = uid[u]
        cid = iid[i]
        row.append(rid)
        col.append(cid)
        value.append(float(line['rating']))
    rate_mat = ssp.coo_matrix((value, (row, col)),
                              shape=(len(uid), len(iid)), dtype='float32')
    return rate_mat


def tagging_mat_batch(df, sample2id, tid, task=None):
    row = []
    col = []
    for _, line in df.iterrows():
        if task == 'user':
            sample = int(line['user_id'])
        else:
            sample = int(line['item_id'])
        tag = str(line['tag'])
        rid = sample2id[sample]
        cid = tid[tag]
        row.append(rid)
        col.append(cid)
    rate_mat = ssp.coo_matrix((np.ones_like(row), (row, col)),
                              shape=(len(sample2id), len(tid)), dtype='float32')
    return rate_mat


def load_rating_mat(data_dir, n_users, n_items, flag=None):
    if flag == 'train':
        indices = np.load(os.path.join(data_dir, 'train_rating_mat_indices.npy'))
        indptr = np.load(os.path.join(data_dir, 'train_rating_mat_indptr.npy'))
        data = np.load(os.path.join(data_dir, 'train_rating_mat_data.npy'))
        rating_mat = ssp.csr_matrix((data, indices, indptr), shape=(n_users, n_items))
        return rating_mat

    if flag == 'test':
        indices = np.load(os.path.join(data_dir, 'test_rating_mat_indices.npy'))
        indptr = np.load(os.path.join(data_dir, 'test_rating_mat_indptr.npy'))
        data = np.load(os.path.join(data_dir, 'test_rating_mat_data.npy'))
        rating_mat = ssp.csr_matrix((data, indices, indptr), shape=(n_users, n_items))
        return rating_mat
    raise ValueError('"train" or "test" should be provided to decide which rating '
                     'matrix to be loaded!')


def load_tagging_mat(data_dir, n_rows, n_cols, flag=None):
    if flag == 'user':
        indices = np.load(os.path.join(data_dir, 'user_tagging_mat_indices.npy'))
        indptr = np.load(os.path.join(data_dir, 'user_tagging_mat_indptr.npy'))
        data = np.load(os.path.join(data_dir, 'user_tagging_mat_data.npy'))
        return ssp.csr_matrix((data, indices, indptr), shape=(n_rows, n_cols))

    if flag == 'item':
        indices = np.load(os.path.join(data_dir, 'item_tagging_mat_indices.npy'))
        indptr = np.load(os.path.join(data_dir, 'item_tagging_mat_indptr.npy'))
        data = np.load(os.path.join(data_dir, 'item_tagging_mat_data.npy'))
        return ssp.csr_matrix((data, indices, indptr), shape=(n_rows, n_cols))

    raise ValueError('"user" or "item" should be provided to decide which tagging '
                    'matrix to be loaded!')


def find_col_idx(col_id, sample_list):
    return [col_id[sample] for sample in sample_list]