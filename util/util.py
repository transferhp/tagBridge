#!user/bin/env python
# _*_ coding: utf-8 _*_
# Author: Peng Hao
# Email: haopengbuaa@gmail.com
# Created: 1/12/16

from scipy.special import expit
import csv
import numpy as np
import pandas as pd
import scipy.sparse as ssp


def sigmoid(x):
    return expit(x)


def sigmoid_grad(x):
    ex = sigmoid(x)
    y = ex / (1 + ex) ** 2
    return y


def write_dict(file_name, myDict):
    with open(file_name, 'wb') as f:
        writer = csv.writer(f)
        for row in myDict.iteritems():
            writer.writerow(row)


def read_dict(file_name):
    with open(file_name, 'rb') as csv_file:
        reader = csv.reader(csv_file)
        mydict = dict(reader)
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

    Returns
    ---------------
    Maximum and minimum values in the matrix and normalized rating matrix.
    """
    if not isinstance(csr_mat, ssp.csr_matrix):
        raise ValueError('Input matrix should be in sparse csr format.')

    nnz_data = csr_mat.data
    max_val = np.max(nnz_data)
    min_val = np.min(nnz_data)
    # minMax normalization
    csr_mat.data = (nnz_data - np.mean(nnz_data)) / np.std(nnz_data)
    return max_val, min_val, csr_mat


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


def tagging_mat_batch(df, sample2id, tid, task='user'):
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
