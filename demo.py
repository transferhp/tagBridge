"""
!user/bin/env python
_*_ coding: utf-8 _*_

Created on: Dec 02, 2016

Author: Peng Hao
Email: haopengbuaa@gmail.com

"""

import time
import os
import pandas as pd
from collections import OrderedDict
from util.util import read_dict, load_rating_mat, load_tagging_mat, find_col_idx
from model import TagCDCF
from model import SVD


def run():
    # set model candidates
    algorithms = OrderedDict()
    algorithms['svd'] = run_svd
    algorithms['tagcdcf'] = run_tagcdcf
    # algorithms['tagicofi'] = run_tagicofi

    # Call recommendation models to run
    results = pd.DataFrame()
    for name, fit_predict in algorithms.items():
        start = time.time()
        rmse = fit_predict()
        spent_time = time.time() - start
        results.ix[name, 'time'] = spent_time
        results.ix[name, 'rmse'] = rmse

    print results


def run_tagcdcf():
    tar_dir = './data/movielens/ml-10m/pro/'
    src_dir = './data/librarything/pro/'

    # load source rating matrix
    src_userid = read_dict(os.path.join(src_dir, 'unique_user_id.txt'))
    src_itemid = read_dict(os.path.join(src_dir, 'unique_item_id.txt'))
    src_rating_mat = load_rating_mat(src_dir, len(src_userid), len(src_itemid),
                                     flag='train')

    # load target rating matrix
    tar_userid = read_dict(os.path.join(tar_dir, 'unique_user_id.txt'))
    tar_itemid = read_dict(os.path.join(tar_dir, 'unique_item_id.txt'))
    tar_rating_mat = load_rating_mat(tar_dir, len(tar_userid), len(tar_itemid),
                                     flag='train')

    # load source train tags
    src_tagid = read_dict(os.path.join(src_dir, 'unique_tag_id.txt'))

    # load target train tags
    tar_tagid = read_dict(os.path.join(tar_dir, 'unique_tag_id.txt'))

    # compute common tags
    com_tags = set(src_tagid.keys()) & set(tar_tagid.keys())
    # find column index for common tags in both source and target domains
    src_col_idx = find_col_idx(src_tagid, com_tags)
    tar_col_idx = find_col_idx(tar_tagid, com_tags)

    # load source tagging matrix
    src_user_tagging_mat = load_tagging_mat(src_dir, len(src_userid), len(src_tagid),
                                            flag='user')
    src_item_tagging_mat = load_tagging_mat(src_dir, len(src_itemid), len(src_tagid),
                                            flag='item')
    # slicing by common tags
    src_user_tag_binary = src_user_tagging_mat.tocsc()[:, src_col_idx].sign()
    src_item_tag_binary = src_item_tagging_mat.tocsc()[:, src_col_idx].sign()

    # load target tagging matrix
    tar_user_tagging_mat = load_tagging_mat(tar_dir, len(tar_userid), len(tar_tagid),
                                            flag='user')
    tar_item_tagging_mat = load_tagging_mat(tar_dir, len(tar_itemid), len(tar_tagid),
                                            flag='item')
    # slicing by common tags
    tar_user_tag_binary = tar_user_tagging_mat.tocsc()[:, tar_col_idx].sign()
    tar_item_tag_binary = tar_item_tagging_mat.tocsc()[:, tar_col_idx].sign()

    inputs = [src_rating_mat, tar_rating_mat,
            src_user_tag_binary, tar_user_tag_binary,
            src_item_tag_binary, tar_item_tag_binary]

    # train the model
    model = TagCDCF(reg_cross_u=0.001, reg_cross_i=0.001, reg_lambda=10, num_factor=10)
    model.fit(*inputs)

    # load target test rating matrix
    test_rating_mat = load_rating_mat(tar_dir, len(tar_userid), len(tar_itemid),
                                      flag='test')
    # test algorithm on specific test set
    predictions = model.test(test_rating_mat)

    rmse = model.evaluate(predictions, test_rating_mat)
    tmp = pd.DataFrame({'pred': predictions, 'truth': test_rating_mat.data})
    tmp.to_csv("prediction_truth_comparison.csv")
    return rmse


def run_tagicofi():
    pass

def run_svd():
    data_dir = './data/movielens/ml-10m/pro/'
    # load training data
    uid = read_dict(os.path.join(data_dir, 'unique_user_id.txt'))
    iid = read_dict(os.path.join(data_dir, 'unique_item_id.txt'))
    train_rating_mat = load_rating_mat(data_dir, len(uid), len(iid), flag='train')

    # train the model
    model = SVD()
    model.fit(train_rating_mat)

    # load target test rating matrix
    test_rating_mat = load_rating_mat(data_dir, len(uid), len(iid), flag='test')
    # test algorithm on specific test set
    predictions = model.test(test_rating_mat)

    rmse = model.evaluate(predictions, test_rating_mat)
    tmp = pd.DataFrame({'pred': predictions, 'truth': test_rating_mat.data})
    tmp.to_csv("prediction_truth_comparison.csv")
    return rmse


if __name__ == '__main__':
    run()