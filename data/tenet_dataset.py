from collections import defaultdict

import numpy as np
import scipy.sparse as sp
from data.dataset import Dataset


class TenetDataset(Dataset):
    def __init__(self, args):
        Dataset.__init__(self, args)

        self.user_items_dct = self.get_user_items_dict(self.user_lists_dct, self.list_items_dct)
        self.user_item_matrix_sp = self.get_sparse_matrix_from_dict(self.user_items_dct, self.num_user, self.num_item)
        self.list_item_matrix_sp = self.get_sparse_matrix_from_dict(self.list_items_dct, self.num_list, self.num_item)
        self.item_list_matrix_sp = self.get_sparse_matrix_from_dict(self.list_items_dct, self.num_item, self.num_list,
                                                                    reverse=True)

        self.user_user_comm_mat_sp = self.mat_mult_sp(self.user_item_matrix_sp,
                                                      self.user_item_matrix_sp.T)
        self.item_item_comm_mat_sp = self.mat_mult_sp(self.item_list_matrix_sp,
                                                      self.item_list_matrix_sp.T)
        self.list_list_comm_mat_sp = self.mat_mult_sp(self.list_item_matrix_sp,
                                                      self.list_item_matrix_sp.T)

        self.user_adj_mat = self.user_user_comm_mat_sp
        self.list_adj_mat = self.list_list_comm_mat_sp
        self.item_adj_mat = self.item_item_comm_mat_sp
        print("hello")

    def mat_mult_sp(self, mat1, mat2):
        return mat1 * mat2

    def get_sparse_matrix_from_dict(self, dct, num_row, num_col, reverse=False):
        sp_mat = sp.lil_matrix((num_row, num_col))

        for key in dct:
            values = dct[key]
            for value in values:
                if not reverse:
                    sp_mat[key, value] = 1
                else:
                    sp_mat[value, key] = 1
        return sp_mat

    def get_user_items_dict(self, user_lists_dct, list_items_dct):
        user_items_dct = defaultdict(set)
        for user in user_lists_dct:
            for lst in user_lists_dct[user]:
                user_items_dct[user] = user_items_dct[user].union(set(list_items_dct[lst]))

        return user_items_dct


class Batch(object):
    def __init__(self, num_instances, batch_size, shuffle=True):
        self.num_instances = num_instances
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.start = 0
        self.epoch_completed = False
        self.indices = np.arange(0, num_instances)
        self.initialize_epoch()

    def initialize_epoch(self):
        self.initialize_next_epoch()

    def initialize_next_epoch(self):
        self.epoch_completed = False
        self.start = 0
        if self.shuffle:
            np.random.shuffle(self.indices)

    def get_next_batch_indices(self):
        start = self.start
        batch_size = self.batch_size
        if start + batch_size < (self.num_instances - 1):
            end = start + batch_size
            self.start = end
        else:
            end = self.num_instances
            self.epoch_completed = True
        return self.indices[start:end]

    def has_next_batch(self):
        return not self.epoch_completed
