import heapq
import math
import multiprocessing

import numpy as np
import torch

from data.tenet_dataset import Batch

_posItemlst = None
_itemMatrix = None
_predMatrix = None
_k = None
_matShape = None


def evaluate_model(posItemlst, itemMatrix, predMatrix, k, num_thread):
    global _posItemlst
    global _itemMatrix
    global _predMatrix
    global _k
    global _matShape

    _posItemlst = posItemlst
    _itemMatrix = itemMatrix
    _predMatrix = predMatrix
    _k = k
    _matShape = itemMatrix.shape
    num_inst = _matShape[0]

    hits, ndcgs, maps = [], [], []
    if num_thread > 1:
        pool = multiprocessing.Pool(processes=num_thread)
        res = pool.map(eval_one_rating, range(num_inst))

        pool.close()
        pool.join()
        hits = [r[0] for r in res]
        ndcgs = [r[1] for r in res]
        maps = [r[2] for r in res]
        return hits, ndcgs, maps

    # Single thread        
    for ind in range(num_inst):
        (hr, ndcg, mapval) = eval_one_rating(ind)
        hits.append(hr)
        ndcgs.append(ndcg)
        maps.append(mapval)
    return hits, ndcgs, maps


def eval_one_rating(ind):
    map_item_score = {}
    predictions = _predMatrix[ind]
    items = _itemMatrix[ind]
    gtItem = _posItemlst[ind]

    for i in range(_matShape[1]):
        item = items[i]
        map_item_score[item] = predictions[i]
    ranklist = heapq.nlargest(_k, map_item_score, key=map_item_score.get)
    hr = getHitRatio(ranklist, gtItem)
    ndcg = getNDCG(ranklist, gtItem)
    mapval = getMAP(ranklist, gtItem)
    return hr, ndcg, mapval


def getHitRatio(ranklist, gtItem):
    for item in ranklist:
        if item == gtItem:
            return 1
    return 0


def getNDCG(ranklist, gtItem):
    for i in range(len(ranklist)):
        item = ranklist[i]
        if item == gtItem:
            return math.log(2) / math.log(i + 2)
    return 0


def getMAP(ranklist, gtItem):
    for i in range(len(ranklist)):
        item = ranklist[i]
        if item == gtItem:
            return 1.0 / (i + 1)
    return 0


class ValidTestError(object):
    def __init__(self, params):
        self.params = params
        self.validNegativesDict = params.validNegativesDict
        self.testNegativesDict = params.testNegativesDict

        self.num_valid_instances = params.num_valid_instances
        self.num_test_instances = params.num_test_instances
        self.num_thread = params.num_thread
        self.num_valid_negatives = self.get_num_valid_negative_samples(self.validNegativesDict)
        self.valid_dim = self.num_valid_negatives + 1

        self.epoch_mod = params.epoch_mod
        self.valid_batch_siz = params.valid_batch_siz
        self.at_k = params.at_k

        self.validArrDubles, self.valid_pos_items = self.get_dict_to_dubles(self.validNegativesDict)
        self.testArrDubles, self.test_pos_items = self.get_dict_to_dubles(self.testNegativesDict)
        self.list_user_vec = params.list_user_vec

    def get_num_valid_negative_samples(self, validDict):
        for key in validDict:
            return len(self.validNegativesDict[key])
        return None

    def get_dict_to_dubles(self, dct):
        list_lst, item_lst = [], []
        pos_item_lst = []
        for key, value in dct.items():
            lst_id, itm_id = key
            lists = list(np.full(self.valid_dim, lst_id, dtype='int32'))  # +1 to add pos item
            items = [itm_id]
            pos_item_lst.append(itm_id)
            items += list(value)  # first is positive item

            list_lst += lists
            item_lst += items

        return (np.array(list_lst), np.array(item_lst)), np.array(pos_item_lst)

    def get_update(self, model, device, valid_flag=True):
        model.eval()
        if valid_flag:
            (list_input, item_input) = self.validArrDubles
            num_inst = self.num_valid_instances * self.valid_dim
            posItemlst = self.valid_pos_items
            matShape = (self.num_valid_instances, self.valid_dim)
        else:
            (list_input, item_input) = self.testArrDubles
            num_inst = self.num_test_instances * self.valid_dim
            posItemlst = self.test_pos_items
            matShape = (self.num_test_instances, self.valid_dim)

        batch_siz = self.valid_batch_siz * self.valid_dim

        full_pred_torch_lst = []
        list_input_ten = torch.from_numpy(list_input.astype(np.long)).to(device)
        item_input_ten = torch.from_numpy(item_input.astype(np.long)).to(device)
        user_input = self.list_user_vec[list_input]
        user_input_ten = torch.from_numpy(user_input.astype(np.long)).to(device)
        batch = Batch(num_inst, batch_siz, shuffle=False)
        while batch.has_next_batch():
            batch_indices = batch.get_next_batch_indices()

            if valid_flag:
                item_seq = torch.from_numpy(
                    self.params.train_matrix_item_seq[list_input[batch_indices]].astype(np.long)).to(
                    device)
            else:
                item_seq = torch.from_numpy(
                    self.params.train_matrix_item_seq_for_test[list_input[batch_indices]].astype(np.long)).to(
                    device)
            y_pred = model(user_indices=user_input_ten[batch_indices], list_indices=list_input_ten[batch_indices],
                           item_seq=item_seq,
                           test_item_indices=item_input_ten[batch_indices], train=False, network='seq')
            full_pred_torch_lst.append(y_pred.detach().cpu().numpy())

        full_pred_np = np.concatenate(full_pred_torch_lst)

        predMatrix = np.array(full_pred_np).reshape(matShape)
        itemMatrix = np.array(item_input).reshape(matShape)

        (hits, ndcgs, maps) = evaluate_model(posItemlst=posItemlst, itemMatrix=itemMatrix, predMatrix=predMatrix,
                                             k=self.at_k, num_thread=self.num_thread)
        return hits, ndcgs, maps
