import numpy as np


class NegativeSamples(object):
    def __init__(self, sp_matrix, num_negatives, params):
        self.sp_matrix = sp_matrix
        self.num_negatives = num_negatives
        self.num_rating = sp_matrix.nnz
        self.num_item = sp_matrix.shape[-1]  ##

        self.num_user = params.num_user
        self.list_user_vec = params.list_user_vec

        self.list_pos_arr, self.item_pos_arr, self.rating_pos_arr = self.get_positive_instances(sp_matrix)
        self.user_pos_arr = self.list_user_vec[self.list_pos_arr]

        self.list_neg_arr = np.repeat(self.list_pos_arr, self.num_negatives)
        self.rating_neg_arr = np.repeat([0], len(self.rating_pos_arr) * self.num_negatives)

        self.list_arr = np.concatenate([self.list_pos_arr, self.list_neg_arr])
        self.rating_arr = np.concatenate([self.rating_pos_arr, self.rating_neg_arr])
        self.rating_arr = self.rating_arr.astype(np.float16)

        self.item_arr, self.item_neg_arr, self.user_arr, self.user_neg_arr = None, None, None, None

    def get_positive_instances(self, mat):
        pos_mat = mat.tocsc().tocoo()
        list_pos_arr, item_pos_arr = pos_mat.row, pos_mat.col
        rating_pos_arr = np.repeat([1], len(list_pos_arr))
        return list_pos_arr, item_pos_arr, rating_pos_arr

    def generate_negative_item_samples(self):
        neg_item_arr = np.random.choice(self.num_item - 1, self.num_negatives * self.num_rating) + 1
        return neg_item_arr

    def generate_negative_user_samples(self):
        neg_user_arr = np.random.choice(self.num_user - 1, self.num_negatives * self.num_rating) + 1
        return neg_user_arr

    def generate_instances(self):
        self.item_neg_arr = self.generate_negative_item_samples()
        self.item_arr = np.concatenate([self.item_pos_arr, self.item_neg_arr])
        self.user_neg_arr = self.generate_negative_user_samples()
        self.user_arr = np.concatenate([self.user_pos_arr, self.user_neg_arr])
        return self.user_arr, self.list_arr, self.item_arr, self.rating_arr


class ListNegativeSamples(object):
    def __init__(self, train_matrix_item_seq, num_negatives, params):
        self.train_matrix_item_seq = train_matrix_item_seq
        self.num_negatives = num_negatives

        self.list_input = np.arange(1, params.num_list)
        self.seq, self.seq_pos = self.get_seq_and_seq_pos(train_matrix_item_seq)
        self.num_item = params.num_item
        self.params = params
        self.seq_neg = None

    def get_seq_and_seq_pos(self, seq_mat):
        seq_in_mat = np.roll(seq_mat, 1, axis=1)
        seq_in_mat[:, 0] = 0
        seq_out_mat = seq_mat.copy()
        row = np.arange(0, len(seq_mat))
        seq_out_mat[row, (seq_mat != 0).argmax(axis=1)] = 0
        return seq_in_mat[1:len(seq_mat)], seq_out_mat[1:len(seq_mat)]

    def generate_negative_seq_mat(self, seq_pos):
        num_row, num_col = seq_pos.shape
        seq_neg = np.random.choice(self.num_item - 1, num_row * num_col).reshape(num_row, num_col) + 1
        seq_neg = seq_neg * (seq_pos != 0)
        return seq_neg

    def generate_instances(self, ):
        self.seq_neg = self.generate_negative_seq_mat(self.seq_pos)
        return self.list_input, self.seq, self.seq_pos, self.seq_neg
