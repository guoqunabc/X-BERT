#!/usr/bin/env python
# encoding: utf-8

import os
import math
import numpy as np
import random
import torch
import shutil
import json
from sklearn.datasets import load_svmlight_file
import scipy.sparse as sp



class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text, label):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            label: list of integers. The labels of the example.
        """
        self.guid = guid
        self.text = text
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, output_ids, output_mask):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.output_ids = output_ids
        self.output_mask = output_mask


def repack_output(output_ids, output_mask, num_labels, device):
    batch_size = output_ids.size(0)
    idx_arr = torch.nonzero(output_mask)
    rows = idx_arr[:,0]
    cols = output_ids[idx_arr[:,0], idx_arr[:,1]]
    c_true = torch.zeros((batch_size,num_labels), dtype=torch.float, device=device)
    c_true[rows, cols] = 1.0
    return c_true


# convert array of indices into one-hot vector
# Y_pad_seq: batch_size x max_len
# Y_one_hot: batch_size x NUM_LABEL
def indice_to_onehot(Y_pad_seq, TRG_vocab, NUM_LABEL=None, device=None):
    batch_size, max_len = Y_pad_seq.size()
    tmp = torch.unsqueeze(Y_pad_seq, 2)
    Y_one_hot = torch.zeros((batch_size, max_len, NUM_LABEL),
                            dtype=torch.float32, device=device)
    Y_one_hot.scatter_(2, tmp, 1)
    Y_one_hot = Y_one_hot.sum(dim=1)

    # masking out <pad>
    Y_one_hot[:,TRG_vocab.stoi['<pad>']] = 0.0
    return Y_one_hot


# Y_list: list(list(y))
# label start from 0 indexing
def Ylist_to_Ysparse(Y_list, L=None):
    rows, cols, vals = [], [], []
    for row_idx, y_list in enumerate(Y_list):
        n_label_per_row = len(y_list)
        rows += [row_idx] * n_label_per_row
        cols += list(map(int, y_list))
        vals += [1] * n_label_per_row

    NUM_DATA = max(rows) + 1
    NUM_LABEL = max(cols) + 1 if L is None else L
    Y_csr_mat = sp.csr_matrix( (vals, (rows, cols)), shape=(NUM_DATA, NUM_LABEL) )
    return Y_csr_mat


# load MLC svm file into sparse matrix
# L: number of labels
# D: number of data dimension
def load_mlc_svmfile(file_path, L=None, D=None):
    assert(os.path.exists(file_path))

    if D is None:
        X, Y_list = load_svmlight_file(file_path, multilabel=True, zero_based=False)
    else:
        X, Y_list = load_svmlight_file(file_path, n_features=D, multilabel=True, zero_based=False)

    # create label sparse matrix
    Y = Ylist_to_Ysparse(Y_list, L=L)
    return X, Y, Y_list


def create_exp_dir(path, scripts_to_save=None, options=None):
    if not os.path.exists(path):
        os.mkdir(path)
    #else:
    #    shutil.rmtree(path)
    #    os.mkdir(path)

    print('Experiment dir : {}'.format(path))
    if scripts_to_save is not None:
        if not os.path.exists(os.path.join(path, 'scripts')):
            os.mkdir(os.path.join(path, 'scripts'))
        for script in scripts_to_save:
            dst_file = os.path.join(path, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file)

    # dump the args
    if options is not None:
        with open(os.path.join(path, 'options.json'), 'w') as f:
            json.dump(vars(options), f)
