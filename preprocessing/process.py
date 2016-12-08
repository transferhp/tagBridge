#!user/bin/env python
# _*_ coding: utf-8 _*_
# Author: Peng Hao
# Email: haopengbuaa@gmail.com
# Created: 23/11/16
#
# Loading raw data.


import os
import numpy as np
import scipy.sparse as ssp
import pandas as pd
from joblib import Parallel, delayed
from util.util import mapping2id, split_by_user, rating_mat_batch, tagging_mat_batch


class Preprocess():

    @staticmethod
    def save_id(sampleid, file_path):
        # write to file
        with open (file_path, 'w') as f:
            for s, id in sampleid.items():
                f.write('{0} : {1}\n'.format(s, id))

    def process_ml10m(self):
        data_dir = '../data/movielens/ml-10m/'
        print('Started to process ml-10m dataset....')

        # Load raw rating data
        names = ['user_id', 'item_id', 'rating', 'ts']
        rate_df = pd.read_csv(os.path.join(data_dir, 'ratings.dat'),
                              names=names,
                              engine='python',
                              sep='::')
        del rate_df['ts']

        # Load raw tagging data, note 'NA' in original tagging data
        names = ['user_id', 'item_id', 'tag', 'ts']
        tag_df = pd.read_csv(os.path.join(data_dir, 'tags.dat'),
                           names=names,
                           engine='python',
                           keep_default_na=False,
                           sep='::')
        del tag_df['ts']

        # filter out users and items that have less than 3 unique tags
        tag_df = self.filter_user_item(tag_df, 3)

        # TODO: save filtered tag_df?
        # tag_df.to_csv(os.path.join(data_dir, 'pro/', 'filtered_tag_df.csv'),index=False)

        # mapping left users, items and tags to id
        users = tag_df['user_id'].unique().tolist()
        userid = mapping2id(users)
        items = tag_df['item_id'].unique().tolist()
        itemid = mapping2id(items)
        tags = np.unique(tag_df['tag']).tolist()
        tagid = mapping2id(tags)

        self.save_id(userid, os.path.join(data_dir, 'pro/', 'unique_user_id.txt'))
        self.save_id(itemid, os.path.join(data_dir, 'pro/', 'unique_item_id.txt'))
        self.save_id(tagid, os.path.join(data_dir, 'pro/', 'unique_tag_id.txt'))

        # filter rate_df by keeping only qualified users and items
        rate_df = rate_df[rate_df['user_id'].isin(users)]
        rate_df = rate_df[rate_df['item_id'].isin(items)]
        # TODO: save filtered rate_df?

        # train/test splitting
        train_df, test_df = split_by_user(rate_df, ratio=0.8)

        print "There are total of {} unique users in the training set " \
              "and {} unique users in the entire dataset".format(
              (len(train_df['user_id'].unique())), len(userid))

        print "There are total of {} unique items in the training set " \
              "and {} unique items in the entire dataset".format(
                (len(train_df['item_id'].unique())), len(itemid))

        # For test data,
        # only keep the users and items that appear in the training sets
        test_df = test_df[test_df['user_id'].isin(train_df['user_id'].unique())]
        test_df = test_df[test_df['item_id'].isin(train_df['item_id'].unique())]

        # make rating matrix
        print('Making training rating matrix...')
        batch_size = 2000
        start_idx = range(0, len(train_df), batch_size)
        end_idx = start_idx[1:] + [len(train_df)]
        train_rate_mat = ssp.csr_matrix((len(userid), len(itemid)), dtype='float32')
        tmp = Parallel(n_jobs=-1)(delayed(rating_mat_batch)(
                train_df[lo:hi], userid, itemid) for lo, hi in zip(start_idx,end_idx))
        for X in tmp:
            train_rate_mat += X
        print('....done!')
        print('Making test rating matrix....')
        batch_size = 2000
        start_idx = range(0, len(test_df), batch_size)
        end_idx = start_idx[1:] + [len(test_df)]
        test_rate_mat = ssp.csr_matrix((len(userid), len(itemid)), dtype='float32')
        tmp = Parallel(n_jobs=-1)(delayed(rating_mat_batch)(
                test_df[lo:hi], userid, itemid) for lo, hi in zip(start_idx, end_idx))
        for X in tmp:
            test_rate_mat += X
        print('....done!')

        # save both train and test rating matrices
        np.save(os.path.join(data_dir, 'pro/', 'train_rating_mat_data.npy'),
                train_rate_mat.data)
        np.save(os.path.join(data_dir, 'pro/', 'train_rating_mat_indices.npy'),
                train_rate_mat.indices)
        np.save(os.path.join(data_dir, 'pro/', 'train_rating_mat_indptr.npy'),
                train_rate_mat.indptr)
        np.save(os.path.join(data_dir, 'pro/', 'test_rating_mat_data.npy'),
                test_rate_mat.data)
        np.save(os.path.join(data_dir, 'pro/', 'test_rating_mat_indices.npy'),
                test_rate_mat.indices)
        np.save(os.path.join(data_dir, 'pro/', 'test_rating_mat_indptr.npy'),
                test_rate_mat.indptr)

        # generate tagging data for training set
        tag_df = tag_df[tag_df['user_id'].isin(train_df['user_id'].unique())]
        tag_df = tag_df[tag_df['item_id'].isin(train_df['item_id'].unique())]
        train_tags = tag_df['tag'].unique().tolist()
        with open(os.path.join(data_dir, 'pro/', 'tags_in_train.txt'), 'w') as f:
            for tag in train_tags:
                f.write('{}\n'.format(tag))

        batch_size = 2000
        start_idx = range(0, len(tag_df), batch_size)
        end_idx = start_idx[1:] + [len(tag_df)]

        # make user tagging matrix
        print('Making user tagging matrix....')
        user_tag_mat = ssp.csr_matrix((len(users), len(tags)), dtype='float32')
        u_tmp = Parallel(n_jobs=-1)(delayed(tagging_mat_batch)(
                tag_df[lo:hi], userid, tagid, task='user') for lo, hi in zip(start_idx,
                                                                             end_idx))
        for tmp in u_tmp:
            user_tag_mat += tmp
        print('....done!')

        # make item tagging matrix
        print('Making item tagging matrix....')
        item_tag_mat = ssp.csr_matrix((len(items), len(tags)), dtype='float32')
        i_tmp = Parallel(n_jobs=-1)(delayed(tagging_mat_batch)(
                tag_df[lo:hi], itemid, tagid, task='item') for lo, hi in zip(start_idx,
                                                                             end_idx))
        for tmp in i_tmp:
            item_tag_mat += tmp
        print('....done!')

        # save user tagging matrix
        np.save(os.path.join(data_dir, 'pro/', 'user_tagging_mat_data.npy'),
                user_tag_mat.data)
        np.save(os.path.join(data_dir, 'pro/', 'user_tagging_mat_indices.npy'),
                user_tag_mat.indices)
        np.save(os.path.join(data_dir, 'pro/', 'user_tagging_mat_indptr.npy'),
                user_tag_mat.indptr)

        # save item tagging matrix
        np.save(os.path.join(data_dir, 'pro/', 'item_tagging_mat_data.npy'),
                item_tag_mat.data)
        np.save(os.path.join(data_dir, 'pro/', 'item_tagging_mat_indices.npy'),
                item_tag_mat.indices)
        np.save(os.path.join(data_dir, 'pro/', 'item_tagging_mat_indptr.npy'),
                item_tag_mat.indptr)
        print('Processing ml-10m dataset finished!')


    def process_ml1m(self):
        data_dir = '../data/movielens/ml-1m/'

        # Load only rating data
        names = ['user_id', 'item_id', 'rating', 'ts']
        actions = pd.read_csv(os.path.join(data_dir, 'ratings.dat'),
                              names=names,
                              engine='python',
                              sep='::')
        del actions['ts']

        # save cleaned data
        if not os.path.isdir(os.path.join(data_dir, 'pro')):
            os.mkdir(os.path.join(data_dir, 'pro'))
        actions.to_csv(os.path.join(data_dir, 'pro/', 'ml_1m.csv'), index=False)


    @staticmethod
    def _read_raw_lt_data(data_path):
        """
        Read raw LibraryThing dataset line by line,
        and save loaded data in the form of
        [user_id, item_id, rating_score, lower_case_tag].

        Return
        -----------
        data: list
        The data structure that keeps the information of LibraryThing dataset.

        """
        data = []

        with open(os.path.join(data_path, 'UI2.txt'), 'r') as myfile:
            for line in myfile:
                line = line.strip('\n')
                user_id = line.split()[0]
                item_id = line.split()[1]
                double_rating = line.split()[2]
                # convert each tag into lower case
                tag = " ".join([x.lower() for x in line.split()[3:]])
                rating = float(double_rating) / 2.0
                data.append([int(user_id), int(item_id), rating, tag])
        return data


    def process_lt(self):
        data_dir = '../data/librarything/'
        print('Started to process lt dataset....')

        # load raw data
        actions = self._read_raw_lt_data(data_dir)
        # change data format
        actions = pd.DataFrame(data=actions,
                               columns=['user_id', 'item_id', 'rating', 'tag'])

        # Drop inconsistent rating scores on same user-item-tag triplet,
        # while keep the first appeared one in the original dataset.
        actions = actions.groupby(['user_id', 'item_id']).apply(
                lambda x: x[x['rating'] == x['rating'].unique()[0]])

        # filter out users and items that have less than 5 unique tags
        actions = self.filter_user_item(actions, 5)

        # mapping left users, items and tags to id
        users = actions['user_id'].unique().tolist()
        userid = mapping2id(users)
        items = actions['item_id'].unique().tolist()
        itemid = mapping2id(items)
        tags = actions['tag'].unique().tolist()
        tagid = mapping2id(tags)

        self.save_id(userid, os.path.join(data_dir, 'pro/', 'unique_user_id.txt'))
        self.save_id(itemid, os.path.join(data_dir, 'pro/', 'unique_item_id.txt'))
        self.save_id(tagid, os.path.join(data_dir, 'pro/', 'unique_tag_id.txt'))

        # train/test splitting
        train_df, test_df = split_by_user(actions[['user_id', 'item_id',
                                                   'rating']].drop_duplicates(),
                                          ratio=0.8)

        print "There are total of {} unique users in the training set " \
              "and {} unique users in the entire dataset".format(
                (len(train_df['user_id'].unique())), len(userid))

        print "There are total of {} unique items in the training set " \
              "and {} unique items in the entire dataset".format(
                (len(train_df['item_id'].unique())), len(itemid))

        # For test data,
        # only keep the users and items that appear in the training sets
        test_df = test_df[test_df['user_id'].isin(train_df['user_id'].unique())]
        test_df = test_df[test_df['item_id'].isin(train_df['item_id'].unique())]

        # make rating matrix
        print('Making training rating matrix...')
        batch_size = 2000
        start_idx = range(0, len(train_df), batch_size)
        end_idx = start_idx[1:] + [len(train_df)]
        train_rate_mat = ssp.csr_matrix((len(userid), len(itemid)), dtype='float32')
        tmp = Parallel(n_jobs=-1)(delayed(rating_mat_batch)(
                train_df[lo:hi], userid, itemid) for lo, hi in zip(start_idx, end_idx))
        for X in tmp:
            train_rate_mat += X
        print('....done!')
        print('Making test rating matrix....')
        batch_size = 2000
        start_idx = range(0, len(test_df), batch_size)
        end_idx = start_idx[1:] + [len(test_df)]
        test_rate_mat = ssp.csr_matrix((len(userid), len(itemid)), dtype='float32')
        tmp = Parallel(n_jobs=-1)(delayed(rating_mat_batch)(
                test_df[lo:hi], userid, itemid) for lo, hi in zip(start_idx, end_idx))
        for X in tmp:
            test_rate_mat += X
        print('....done!')

        # save both train and test rating matrices
        np.save(os.path.join(data_dir, 'pro/', 'train_rating_mat_data.npy'),
                train_rate_mat.data)
        np.save(os.path.join(data_dir, 'pro/', 'train_rating_mat_indices.npy'),
                train_rate_mat.indices)
        np.save(os.path.join(data_dir, 'pro/', 'train_rating_mat_indptr.npy'),
                train_rate_mat.indptr)
        np.save(os.path.join(data_dir, 'pro/', 'test_rating_mat_data.npy'),
                test_rate_mat.data)
        np.save(os.path.join(data_dir, 'pro/', 'test_rating_mat_indices.npy'),
                test_rate_mat.indices)
        np.save(os.path.join(data_dir, 'pro/', 'test_rating_mat_indptr.npy'),
                test_rate_mat.indptr)

        # generate tagging data for training set
        tag_df = actions[actions['user_id'].isin(train_df['user_id'].unique())]
        tag_df = tag_df[tag_df['item_id'].isin(train_df['item_id'].unique())]
        train_tags = tag_df['tag'].unique().tolist()
        with open(os.path.join(data_dir, 'pro/', 'tags_in_train.txt'), 'w') as f:
            for tag in train_tags:
                f.write('{}\n'.format(tag))

        batch_size = 2000
        start_idx = range(0, len(tag_df), batch_size)
        end_idx = start_idx[1:] + [len(tag_df)]

        # make user tagging matrix
        print('Making user tagging matrix....')
        user_tag_mat = ssp.csr_matrix((len(users), len(tags)), dtype='float32')
        u_tmp = Parallel(n_jobs=-1)(delayed(tagging_mat_batch)(
                tag_df[lo:hi], userid, tagid, task='user') for lo, hi in zip(start_idx,
                                                                             end_idx))
        for tmp in u_tmp:
            user_tag_mat += tmp
        print('....done!')

        # make item tagging matrix
        print('Making item tagging matrix....')
        item_tag_mat = ssp.csr_matrix((len(items), len(tags)), dtype='float32')
        i_tmp = Parallel(n_jobs=-1)(delayed(tagging_mat_batch)(
                tag_df[lo:hi], itemid, tagid, task='item') for lo, hi in zip(start_idx,
                                                                             end_idx))
        for tmp in i_tmp:
            item_tag_mat += tmp
        print('....done!')

        # save user tagging matrix
        np.save(os.path.join(data_dir, 'pro/', 'user_tagging_mat_data.npy'),
                user_tag_mat.data)
        np.save(os.path.join(data_dir, 'pro/', 'user_tagging_mat_indices.npy'),
                user_tag_mat.indices)
        np.save(os.path.join(data_dir, 'pro/', 'user_tagging_mat_indptr.npy'),
                user_tag_mat.indptr)

        # save item tagging matrix
        np.save(os.path.join(data_dir, 'pro/', 'item_tagging_mat_data.npy'),
                item_tag_mat.data)
        np.save(os.path.join(data_dir, 'pro/', 'item_tagging_mat_indices.npy'),
                item_tag_mat.indices)
        np.save(os.path.join(data_dir, 'pro/', 'item_tagging_mat_indptr.npy'),
                item_tag_mat.indptr)
        print('Processing lt dataset finished!')


    @staticmethod
    def filter_user_item(df, num):
        old = len(df)
        # Filter out users who distributed less than n unique tags
        df = df.groupby('user_id').filter(lambda x: len(x['tag'].unique()) >= num)
        # Filter out items which are attached less than n unique tags
        df = df.groupby('item_id').filter(lambda x: len(x['tag'].unique()) >= num)
        new = len(df)
        if new == old: # data frame length not change
            return df
        else:
            return Preprocess.filter_user_item(df, num)


if __name__ == '__main__':
    processor = Preprocess()
    # pre-processing raw ml-10m dataset
    processor.process_ml10m()
    # pre-processing raw lt dataset
    processor.process_lt()

