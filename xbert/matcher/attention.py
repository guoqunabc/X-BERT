#!/usr/bin/env python
# encoding: utf-8

# modefied from prakashpandey9 github (Dec-15th, 2018)
# Github: https://github.com/prakashpandey9/Text-Classification-Pytorch/blob/master/main.py

from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import copy
import csv
import glob
import json
import logging
import math
import numpy as np
import os
import random
import re
import pickle
import shutil
import tarfile
import tempfile
import scipy as sp
import scipy.sparse as smat
import sys

import time
from os import path
from io import open

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
from torch.optim import lr_scheduler
torch.backends.cudnn.enabled=False    # torch bug when loading rnn models...

import xbert.Constants as Constants
import xbert.data_utils as data_utils
import xbert.rf_linear as rf_linear
import xbert.rf_util as rf_util
from xbert.matcher.bert import HingeLoss, transform_prediction


# global variable within the module

device = None
n_gpu = None
logger = None



def calc_grad_norm(net):
    total_grad_norm = 0.0
    for p in list(filter(lambda p: p.grad is not None, net.parameters())):
        total_grad_norm += p.grad.data.norm(2).item()
    return total_grad_norm

class DataLoader(object):
    ''' Data iterator for X-Attention '''
    def __init__(self, data, set_option='trn', batch_size=64, device=torch.device("cuda")):
        self.device = device
        self.set_option = set_option
        self.prepare_loader(data, set_option=set_option)
        self.prepare_data()
        self.prepare_batch(batch_size=batch_size)

    def prepare_loader(self, data, set_option='trn'):
        self.xseq_list = data[set_option]['xseq']
        self.cseq_list = data[set_option]['cseq']
        self.yseq_list = data[set_option]['yseq']
        self.C = data['C']
        self.stoi = data['stoi']
        self.itos = data['itos']

    # each data point: (xseq, cseq, ypseq, ynseq) where
    # where x_i is a list of wid
    # and y_i is a list of yid
    def prepare_data(self):
        self._dset = {'x': self.xseq_list, 'c': self.cseq_list, 'y': self.yseq_list}
        assert(len(self._dset['x']) == len(self._dset['c']))

    def prepare_batch(self, batch_size=64):
        self._n_sample = len(self._dset['x'])
        self._n_batch = int(math.ceil(self._n_sample / float(batch_size)))  # for the last batch
        self._batch_size = batch_size
        self._n_cluster = self.C.shape[1]
        self._n_label = self.C.shape[0]

    def shuffle(self):
        tmp = list(zip(self._dset['x'], self._dset['c'], self._dset['y']))
        random.shuffle(tmp)
        self._dset['x'], self._dset['c'], self._dset['y'] = zip(*tmp)

    def sort_by_len(self, key_idx=0):
        tmp = list(zip(self._dset['x'], self._dset['c'], self._dset['y']))
        sorted_tmp = sorted(tmp, key=lambda x: len(x[key_idx]))
        self._dset['x'], self._dset['c'], self._dset['y'] = zip(*sorted_tmp)

    def _batchify(self, xseq_list, cseq_list):
        # doing padding and masking
        def pad_to_longest(insts):
            max_len = max(len(inst) for inst in insts)
            inst_data = np.array([
                inst + [Constants.PAD] * (max_len - len(inst))
                for inst in insts])
            return inst_data

        # batch of x_seq with padding into torch tensor
        # x_pt: batch_size x max_len
        x_data = pad_to_longest(xseq_list)
        x_pt = torch.tensor(x_data, dtype=torch.int64, device=self.device)

        # batch of c_seq into torch tensor
        # c_pt: batch_size x _n_cluster
        c_data = np.zeros((len(cseq_list), self._n_cluster))
        for i, cseq in enumerate(cseq_list):
            for j, cid in enumerate(cseq):
                c_data[i, cid] = 1
        c_pt = torch.tensor(c_data, dtype=torch.float32, device=self.device)
        return x_pt, c_pt


    def __getitem__(self, index):
        if index == self._n_batch:
            raise StopIteration()

        s = index * self._batch_size
        e = min((index + 1) * self._batch_size, self._n_sample)
        x_pt, c_pt = self._batchify(self._dset['x'][s:e], self._dset['c'][s:e])
        ret_obj = {'x': x_pt, 'c': c_pt}
        return ret_obj

    def __len__(self):
        return self._n_batch


class CNN(nn.Module):
    def __init__(self, batch_size, hidden_size, output_size, in_channels, out_channels,
                 kernel_heights, stride, padding, pooling_size, keep_probab, vocab_size,
                 embedding_length, weights):
        super(CNN, self).__init__()
        self.batch_size = batch_size
        self.output_size = output_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_heights = kernel_heights
        self.stride = stride
        self.padding = padding
        self.pooling_size = pooling_size
        self.vocab_size = vocab_size
        self.embedding_length = embedding_length
        self.l_embed_dropout = nn.Dropout(p=0.25)
        self.word_embeddings = nn.Embedding(vocab_size, embedding_length)
        if weights is not None:
            # Assigning the look-up table to the pre-trained GloVe word embedding.
            self.word_embeddings.weight = nn.Parameter(weights, requires_grad=False)

        # convolution layer
        self.conv1 = nn.Conv2d(in_channels, out_channels, (kernel_heights[0], embedding_length), stride, padding)
        self.conv2 = nn.Conv2d(in_channels, out_channels, (kernel_heights[1], embedding_length), stride, padding)
        self.conv3 = nn.Conv2d(in_channels, out_channels, (kernel_heights[2], embedding_length), stride, padding)
        #self.conv4 = nn.Conv2d(in_channels, out_channels, (kernel_heights[3], embedding_length), stride, padding)

        self.mlp = nn.Sequential(
            nn.Linear(len(kernel_heights)*out_channels*pooling_size, hidden_size),
            nn.ReLU(),
            #nn.Linear(2*hidden_size, hidden_size),
            #nn.ReLU(),
        )
        self.l_hidden_dropout = nn.Dropout(p=keep_probab)
        self.label = nn.Linear(hidden_size, output_size)

    def conv_block(self, input, conv_layer):
        conv_out = conv_layer(input)
        # conv_out.size() = (batch_size, out_channels, dim, 1)
        activation = F.relu(conv_out.squeeze(3))
        # activation.size() = (batch_size, out_channels, dim1)
        max_out = F.adaptive_max_pool1d(activation, output_size=self.pooling_size)
        #max_out = F.max_pool1d(activation, activation.size()[2]).squeeze(2)# maxpool_out.size() = (batch_size, out_channels)
        # max_out.size() = (batch_size, out_channels, pooling_size)
        return max_out

    def forward(self, input_sentences, batch_size=None):
        input = self.word_embeddings(input_sentences)
        input = self.l_embed_dropout(input)
        batch_size, num_seq, embed_dim = input.size()
        input = input.unsqueeze(1)
        # input.size() = (batch_size, 1, num_seq, embedding_length)

        max_out1 = self.conv_block(input, self.conv1)
        max_out2 = self.conv_block(input, self.conv2)
        max_out3 = self.conv_block(input, self.conv3)
        #max_out4 = self.conv_block(input, self.conv4)
        all_out = torch.cat((max_out1, max_out2, max_out3), dim=1).contiguous()
        #all_out = torch.cat((max_out1, max_out2, max_out3, max_out4), dim=1).contiguous()
        all_out = all_out.view(batch_size, -1)

        # all_out.size() = (batch_size, num_kernels*out_channels*pool_size)
        fc_in = self.l_hidden_dropout(all_out)
        # fc_in.size() = (batch_size, num_kernels*out_channels)
        fc_out = self.mlp(fc_in)
        # fc_out.size() = (batch_size, hidden_size)
        logits = self.label(fc_out)

        return logits


class SelfAttention(nn.Module):
    def __init__(self, batch_size, output_size, hidden_size, vocab_size, embedding_length, weights):
        super(SelfAttention, self).__init__()

        """
        Arguments
        ---------
        batch_size : Size of the batch which is same as the batch_size of the data returned by the TorchText BucketIterator
        output_size : 2 = (pos, neg)
        hidden_sie : Size of the hidden_state of the LSTM
        vocab_size : Size of the vocabulary containing unique words
        embedding_length : Embeddding dimension of GloVe word embeddings
        weights : Pre-trained GloVe word_embeddings which we will use to create our word_embedding look-up table

        --------

        """

        self.batch_size = batch_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.embedding_length = embedding_length
        self.weights = weights

        self.word_embeddings = nn.Embedding(vocab_size, embedding_length)
        self.word_embeddings.weights = nn.Parameter(weights, requires_grad=False)
        self.bilstm = nn.LSTM(embedding_length, hidden_size, bidirectional=True)
        # We will use da = 350, r = 30 & penalization_coeff = 1 as per given in the self-attention original ICLR paper
        self.W_s1 = nn.Linear(2*hidden_size, 350)
        self.W_s2 = nn.Linear(350, 30)
        self.fc_layer = nn.Sequential(
            nn.Linear(30*2*hidden_size, 2000),
            nn.ReLU(),
        )
        self.label = nn.Linear(2000, output_size)

    def attention_net(self, lstm_output):

        """
        Now we will use self attention mechanism to produce a matrix embedding of the input sentence in which every row represents an
        encoding of the inout sentence but giving an attention to a specific part of the sentence. We will use 30 such embedding of
        the input sentence and then finally we will concatenate all the 30 sentence embedding vectors and connect it to a fully
        connected layer of size 2000 which will be connected to the output layer of size 2 returning logits for our two classes i.e.,
        pos & neg.

        Arguments
        ---------

        lstm_output = A tensor containing hidden states corresponding to each time step of the LSTM network.
        ---------

        Returns : Final Attention weight matrix for all the 30 different sentence embedding in which each of 30 embeddings give
        attention to different parts of the input sentence.

        Tensor size : lstm_output.size() = (batch_size, num_seq, 2*hidden_size)
        attn_weight_matrix.size() = (batch_size, 30, num_seq)

        """
        attn_weight_matrix = self.W_s2(F.tanh(self.W_s1(lstm_output)))
        attn_weight_matrix = attn_weight_matrix.permute(0, 2, 1)
        attn_weight_matrix = F.softmax(attn_weight_matrix, dim=2)

        return attn_weight_matrix

    def forward(self, input_sentences, batch_size=None):

        """
        Parameters
        ----------
        input_sentence: input_sentence of shape = (batch_size, num_sequences)
        batch_size : default = None. Used only for prediction on a single sentence after training (batch_size = 1)

        Returns
        -------
        Output of the linear layer containing logits for pos & neg class.

        """

        input = self.word_embeddings(input_sentences)
        #input = self.l_embed_dropout(input)
        input = input.permute(1, 0, 2)
        if batch_size is None:
            h_0 = Variable(torch.zeros(2, self.batch_size, self.hidden_size).cuda())
            c_0 = Variable(torch.zeros(2, self.batch_size, self.hidden_size).cuda())
        else:
            h_0 = Variable(torch.zeros(2, batch_size, self.hidden_size).cuda())
            c_0 = Variable(torch.zeros(2, batch_size, self.hidden_size).cuda())

        output, (h_n, c_n) = self.bilstm(input, (h_0, c_0))
        output = output.permute(1, 0, 2)
        # output.size() = (batch_size, num_seq, 2*hidden_size)
        # h_n.size() = (1, batch_size, hidden_size)
        # c_n.size() = (1, batch_size, hidden_size)
        attn_weight_matrix = self.attention_net(output)
        # attn_weight_matrix.size() = (batch_size, r, num_seq)
        # output.size() = (batch_size, num_seq, 2*hidden_size)
        hidden_matrix = torch.bmm(attn_weight_matrix, output)
        # hidden_matrix.size() = (batch_size, r, 2*hidden_size)
        # Let's now concatenate the hidden_matrix and connect it to the fully connected layer.
        fc_out = self.fc_layer(hidden_matrix.view(-1, hidden_matrix.size()[1]*hidden_matrix.size()[2]))
        logits = self.label(fc_out)
        # logits.size() = (batch_size, output_size)

        return logits


class AttentionMatcher(object):
    def __init__(self, model=None, criterion=None, num_clusters=None, vocab_size=None):
        self.model = model
        self.criterion = criterion
        self.num_clusters = num_clusters
        self.vocab_size = vocab_size

    @staticmethod
    def get_args_and_set_logger():
        global logger
        logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                            datefmt = '%m/%d/%Y %H:%M:%S',
                            level = logging.INFO)
        logger = logging.getLogger(__name__)
        parser = argparse.ArgumentParser(description='')

        ## Required parameters
        parser.add_argument('-i', '--data_bin_path', default=None, type=str, required=True,
                            help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
        parser.add_argument("-o", "--output_dir", default=None, type=str, required=True,
                            help="The output directory where the model predictions and checkpoints will be written.")
        parser.add_argument('--arch-name', default='self-attend', type=str, help='self-attend|conv1d')
        parser.add_argument("--init_checkpoint_dir", default=None, type=str, required=False,
                            help="The directory where the model checkpoints will be intialized from.")

        ## Other parameters
        parser.add_argument("--do_train",
            action='store_true',
            help="Whether to run training.")
        parser.add_argument("--do_eval",
            action='store_true',
            help="Whether to run eval on the dev set.")
        parser.add_argument("--stop_by_dev",
            action='store_true',
            help="Whether to run eval on the dev set.")

        # model parameters
        parser.add_argument('--embed-size', default=512, type=int, help='#dims of word embedding')
        parser.add_argument('--hidden-size', default=512, type=int, help='last hidden layer dimension of netH')
        parser.add_argument('--filter-size', default=128, type=int, help='#filters in conv1d of netH')
        parser.add_argument('--pooling-size', default=4, type=int, help='#pooling units after conv1d of netH')
        parser.add_argument('--pretrained-path', default=None, type=str, help='pretrained matcher path')

        # optimizer
        parser.add_argument('--optim', default='adam', help='adam | rmsprop | sgd', type=str)
        parser.add_argument('--lr', default=1e-4, help='learning rate', type=float)
        parser.add_argument('--momentum', default=0.9, help='sgd-based momentum', type=float)
        parser.add_argument('--factor', default=0.1, help='new_lr = lr * factor', type=float)
        parser.add_argument('--patience', default=10, help='Number of epochs with no improvement after which learning rate will be reduced', type=int)
        parser.add_argument('--min-lr', default=1e-7, help='A lower bound on the learning rate of all param groups or each group respectively', type=float)
        parser.add_argument('--random-seed', default=1126, type=int, help='random seed')

        # loss func
        parser.add_argument('--loss-func', default='l2-hinge', help='bce | l1-hinge | l2-hinge', type=str)
        parser.add_argument('--margin', default=1.0, help='margin in hinge loss', type=float)

        # training
        parser.add_argument('--cuda', action='store_true', help='use CUDA')
        parser.add_argument('--train_batch_size', default=128, type=int, help='batch size for training')
        parser.add_argument('--eval_batch_size', default=512, type=int, help='batch size for evaluation')
        parser.add_argument('--num_train_epochs', default=200, type=int, help='max epoch for training')
        parser.add_argument('--log_interval', default=10, type=int, help='log interval')
        parser.add_argument('--eval_interval', default=50, type=int, help='eval interval')
        # prediction
        parser.add_argument('--only_topk', default=10, type=int, help='store topk prediction for matching stage')
        args = parser.parse_args()
        print(args)

        return {'parser': parser, 'logger': logger, 'args': args}

    @staticmethod
    def bootstrap_for_training(args):
        ''' set device for training, and fix random seed'''
        global n_gpu, device
        np.random.seed(args.random_seed)
        random.seed(args.random_seed)
        torch.manual_seed(args.random_seed)
        if torch.cuda.is_available() and not args.cuda:
            logging("WARNING: You have a CUDA device, so you should probably run with --cuda")
        device = torch.device("cuda" if args.cuda else "cpu")

    @staticmethod
    def load_data(args):
        global device
        with open(args.data_bin_path, 'rb') as fin:
            data_dict = pickle.load(fin)
        trn_features = data_dict['trn']
        val_features = data_dict['val']
        tst_features = data_dict['tst']
        NUM_TOKEN = len(data_dict['stoi'])
        NUM_LABEL = data_dict['C'].shape[0]
        NUM_CLUSTER = data_dict['C'].shape[1]
        logger.info('TRN {} VAL {} TST {}'.format(len(trn_features['xseq']), len(val_features['xseq']), len(tst_features['xseq'])))
        logger.info('NUM_LABEL {}'.format(NUM_LABEL))
        logger.info('NUM_CLUSTER {}'.format(NUM_CLUSTER))

        # load Y csr matrix
        C_val = data_utils.Ylist_to_Ysparse(data_dict['val']['cseq'], L=NUM_CLUSTER)
        C_tst = data_utils.Ylist_to_Ysparse(data_dict['tst']['cseq'], L=NUM_CLUSTER)

        # data iterator
        trn_iter = DataLoader(data_dict, set_option='trn',
                              batch_size=args.train_batch_size, device=device)
        val_iter = DataLoader(data_dict, set_option='val',
                              batch_size=args.eval_batch_size, device=device)
        tst_iter = DataLoader(data_dict, set_option='tst',
                              batch_size=args.eval_batch_size, device=device)

        return {'trn_features': trn_features, 'val_features': val_features, 'tst_features': tst_features,
                'trn_iter': trn_iter, 'val_iter': val_iter, 'tst_iter': tst_iter,
                'num_clusters': NUM_CLUSTER, 'vocab_size': NUM_TOKEN,
                'NUM_LABEL': NUM_LABEL, 'NUM_CLUSTER': NUM_CLUSTER, 'C_val': C_val, 'C_tst': C_tst}

    def prepare_attention_model(self, args, num_clusters, vocab_size, arch_name='self-attend', loss_func='l2-hinge'):
        ''' Initialize/Load an X-attention model for sequeence classification'''
        global device
        if arch_name == 'conv1d':
            model = CNN(batch_size=args.train_batch_size,
                        hidden_size=args.hidden_size, output_size=num_clusters,
                        in_channels=1, out_channels=args.filter_size,
                        kernel_heights=[1,3,5], stride=2, padding=0,
                        pooling_size=args.pooling_size,
                        keep_probab=0.5, vocab_size=vocab_size,
                        embedding_length=args.embed_size, weights=None).to(device)
        elif arch_name == 'self-attend':
            model = SelfAttention(batch_size=args.train_batch_size,
                                  output_size=num_clusters, hidden_size=args.hidden_size,
                                  vocab_size=vocab_size,
                                  embedding_length=args.embed_size, weights=None).to(device)
        else:
            raise NotImplementedError('unknown arch {}'.format(arch))
        print(model)

        if loss_func == 'bce':
            criterion = nn.BCEWithLogitsLoss()
        elif loss_func == 'l1-hinge':
            criterion = HingeLoss(margin=args.margin, squared=False).to(device)
        elif loss_func == 'l2-hinge':
            criterion = HingeLoss(margin=args.margin, squared=True).to(device)
        else:
            raise NotImplementedError('unknown loss function {}'.format(loss_func))

        pretrained_path = path.join(args.init_checkpoint_dir, 'pytorch_model.bin')
        if path.exists(pretrained_path):
            model.load_state_dict(torch.load(pretrained_path))

        # overwrite
        self.model = model
        self.criterion = criterion
        self.num_clusters = num_clusters
        self.vocab_size = vocab_size
        self.loss_func = loss_func
        self.arch_name = arch_name

    def load(self, args):
        matcher_json_file = os.path.join(args.init_checkpoint_dir, 'xttention.json')
        with open(matcher_json_file, 'r') as fin:
            matcher_json_dict = json.load(fin)
        num_clusters = matcher_json_dict['num_clusters']
        vocab_size = matcher_json_dict['vocab_size']
        loss_func = matcher_json_dict['loss_func']
        arch_name = matcher_json_dict['arch_name']
        self.prepare_attention_model(args, num_clusters, vocab_size,
                                     arch_name=arch_name,
                                     loss_func=loss_func)

    def save(self, args):
        model_to_save = self.model
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

        output_model_file = os.path.join(args.output_dir, 'pytorch_model.bin')
        torch.save(model_to_save.state_dict(), output_model_file)

        output_xbert_file = os.path.join(args.output_dir, 'xttention.json')
        with open(output_xbert_file, 'w') as fout:
            xbert_dict = {'num_clusters': self.num_clusters,
                          'vocab_size': self.vocab_size,
                          'arch_name': self.arch_name,
                          'loss_func': self.loss_func}
            json.dump(xbert_dict, fout)

    def predict(self, args, eval_iter, C_eval_true, topk=10, verbose=True):
        '''Prediction interface'''
        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d", eval_iter._n_sample)
        logger.info("  Batch size = %d", args.eval_batch_size)

        self.model.eval()
        total_loss = 0.
        total_example = 0.
        with torch.no_grad():
            rows, cols, vals = [], [], []
            for bidx, batch in enumerate(eval_iter):
                x, c_true = batch['x'], batch['c']
                cur_batch_size, cur_src_len = x.size()
                c_pred = self.model(x, batch_size=cur_batch_size)
                loss = self.criterion(c_pred, c_true).item()
                total_loss += cur_batch_size * loss

                # get topk prediction rows,cols,vals
                cpred_topk_vals, cpred_topk_cols = c_pred.topk(topk, dim=1)
                cpred_topk_rows = (total_example + torch.arange(cur_batch_size))
                cpred_topk_rows = cpred_topk_rows.view(cur_batch_size, 1).expand_as(cpred_topk_cols)
                total_example += cur_batch_size

                # append
                rows += cpred_topk_rows.numpy().flatten().tolist()
                cols += cpred_topk_cols.cpu().numpy().flatten().tolist()
                vals += cpred_topk_vals.cpu().numpy().flatten().tolist()

        eval_loss = total_loss / total_example
        m = int(total_example)
        n = self.num_clusters
        pred_csr_codes = smat.csr_matrix( (vals, (rows,cols)), shape=(m,n) )
        pred_csr_codes = rf_util.smat_util.sorted_csr(pred_csr_codes, only_topk=None)
        C_eval_pred = pred_csr_codes

        # evaluation
        eval_metrics = rf_linear.Metrics.generate(C_eval_true, C_eval_pred, topk=args.only_topk)
        if verbose:
            logger.info('| matcher_eval_prec {}'.format(' '.join("{:4.2f}".format(100*v) for v in eval_metrics.prec)))
            logger.info('| matcher_eval_recl {}'.format(' '.join("{:4.2f}".format(100*v) for v in eval_metrics.recall)))
            logger.info('-' * 89)
        return eval_loss, eval_metrics, C_eval_pred


    def train(self, args, trn_iter, eval_iter=None, C_eval=None):

        # Prepare optimizer
        if args.optim == 'adam':
            optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr)
        elif args.optim == 'rmsprop':
            optimizer = torch.optim.RMSprop(self.model.parameters(), lr=args.lr, momentum=args.momentum)
        elif args.optim == 'sgd':
            optimizer = torch.optim.SGD(self.model.parameters(), lr=args.lr, momentum=args.momentum)
        else:
            raise ValueError('unknown optim %s [adam | rmsprop | sgd]' % (args.optim))
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'max',
                                                   factor=args.factor,
                                                   patience=args.patience,
                                                   min_lr=1e-6, verbose=True)

        # Start Batch Training
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", trn_iter._n_sample)
        logger.info("  Batch size = %d", args.train_batch_size)
        global_step = 0
        total_loss = 0.

        self.model.train()
        total_run_time = 0.0
        best_matcher_prec = -1
        for epoch in range(1, args.num_train_epochs + 1):
            trn_iter.shuffle()
            for step, batch in enumerate(trn_iter):
                start_time = time.time()
                x, c_true = batch['x'], batch['c']
                cur_batch_size, cur_src_len = x.size()
                self.model.zero_grad()
                c_pred = self.model(x, batch_size=cur_batch_size)
                loss = self.criterion(c_pred, c_true)

                # backward
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                total_run_time += time.time() - start_time

                # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
                #torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                #for p in self.model.parameters():
                #    p.data.add_(-lr, p.grad.data)

                global_step += 1
                # print training log
                if step % args.log_interval == 0 and step > 0:
                    elapsed = time.time() - start_time
                    cur_loss = total_loss / args.log_interval

                    #calc_grad_norm = lambda net: sum(p.grad.data.norm(2).item() for p in net.parameters if p.grad is not None)
                    cur_gnorm = calc_grad_norm(self.model)
                    logger.info('| epoch {:3d} | {:4d}/{:4d} batches | ms/batch {:5.4f} | train_loss {:e} | gnorm {:e}'.format(
                        epoch, step, len(trn_iter), elapsed * 1000 / args.log_interval, cur_loss, cur_gnorm))
                    total_loss = 0.0

                # eval on dev set and save best model
                if step % args.eval_interval == 0 and step > 0 and args.stop_by_dev:
                    eval_loss, eval_metrics, C_eval_pred = self.predict(args, eval_iter, C_eval, topk=args.only_topk, verbose=False)
                    logger.info('-' * 89)
                    logger.info('| epoch {:3d} evaluation | time: {:5.4f}s | eval_loss {:e}'.format(
                        epoch, total_run_time, eval_loss))
                    logger.info('| matcher_eval_prec {}'.format(' '.join("{:4.2f}".format(100*v) for v in eval_metrics.prec)))
                    logger.info('| matcher_eval_recl {}'.format(' '.join("{:4.2f}".format(100*v) for v in eval_metrics.recall)))

                    avg_matcher_prec = np.mean(eval_metrics.prec)
                    if avg_matcher_prec > best_matcher_prec and epoch > 0:
                        logger.info('| **** saving model at global_step {} ****'.format(global_step))
                        best_matcher_prec = avg_matcher_prec
                        self.save(args)
                    logger.info('-' * 89)
                    self.model.train()
                    scheduler.step(avg_matcher_prec)


def main():
    # get args
    args = AttentionMatcher.get_args_and_set_logger()['args']

    # load data
    AttentionMatcher.bootstrap_for_training(args)     # need to first initialize device variable
    data = AttentionMatcher.load_data(args)           # for load_data function
    trn_iter = data['trn_iter']
    val_iter = data['val_iter']
    tst_iter = data['tst_iter']
    num_clusters = data['num_clusters']
    vocab_size = data['vocab_size']
    C_val = data['C_val']
    C_tst = data['C_tst']

    # if no init_checkpoint_dir,
    # start with random intialization and train
    model = AttentionMatcher()
    if args.init_checkpoint_dir is None:
        logger.info("start with random initialization")
        model.prepare_attention_model(args, num_clusters, vocab_size, arch_name=args.arch_name, loss_func=args.loss_func)
    else:
        logger.info("start with init_checkpoint_dir")
        model.load(args)

    # do_train and save model
    if args.do_train:
        model.train(args, trn_iter, eval_iter=val_iter, C_eval=C_val)

    # do_eval on test set and save prediction output
    if args.do_eval:
        eval_loss, eval_metrics, C_tst_pred = model.predict(args, tst_iter, C_tst, topk=args.only_topk)
        pred_csr_codes = C_tst_pred
        pred_csr_codes = rf_util.smat_util.sorted_csr(pred_csr_codes, only_topk=args.only_topk)
        pred_csr_codes = transform_prediction(pred_csr_codes, transform='lpsvm-l2')
        prediction_path = os.path.join(args.output_dir, 'C_eval_pred.npz')
        smat.save_npz(prediction_path, pred_csr_codes)


if __name__ == '__main__':
    main()
