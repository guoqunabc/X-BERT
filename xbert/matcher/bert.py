# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch BERT model."""

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
import torch.nn.functional as F

from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from pytorch_pretrained_bert.modeling import BertModel, BertPreTrainedModel, BertConfig, WEIGHTS_NAME, CONFIG_NAME
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam, WarmupLinearSchedule

import xbert.data_utils as data_utils
import xbert.rf_linear as rf_linear
import xbert.rf_util as rf_util

# global variable within the module

device = None
n_gpu = None
logger = None

class BertForSequenceClassification(BertPreTrainedModel):
    """BERT model for classification.
    This module is composed of the BERT model with a linear layer on top of
    the pooled output.

    Params:
        `config`: a BertConfig class instance with the configuration to build a new model.
        `num_labels`: the number of classes for the classifier. Default = 2.

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `labels`: labels for the classification output: torch.LongTensor of shape [batch_size]
            with indices selected in [0, ..., num_labels].

    Outputs:
        if `labels` is not `None`:
            Outputs the CrossEntropy classification loss of the output with the labels.
        if `labels` is `None`:
            Outputs the classification logits of shape [batch_size, num_labels].

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    num_labels = 2

    model = BertForSequenceClassification(config, num_labels)
    logits = model(input_ids, token_type_ids, input_mask)
    ```
    """
    def __init__(self, config, num_labels):
        super(BertForSequenceClassification, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits

class HingeLoss(nn.Module):
    """criterion for loss function

       y: 0/1 ground truth matrix of size: batch_size x output_size
       f: real number pred matrix of size: batch_size x output_size
    """
    def __init__(self, margin=1.0, squared=True):
        super(HingeLoss, self).__init__()
        self.margin = margin
        self.squared = squared

    def forward(self, f, y):
        # convert y into {-1,1}
        y_new = 2.*y - 1.0
        loss = F.relu(self.margin - y_new*f)
        if self.squared:
            loss = loss**2
        return loss.mean()

class BertMatcher(object):

    def __init__(self, model=None, criterion=None, num_clusters=None):
        self.model = model
        self.criterion = criterion
        self.num_clusters = num_clusters

    @staticmethod
    def get_args_and_set_logger():
        global logger
        logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                            datefmt = '%m/%d/%Y %H:%M:%S',
                            level = logging.INFO)
        logger = logging.getLogger(__name__)
        parser = argparse.ArgumentParser(description='')

        ## Required parameters
        parser.add_argument("-i", "--data_bin_path", default=None, type=str, required=True,
                            help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
        parser.add_argument("-o", "--output_dir", default=None, type=str, required=True,
                            help="The output directory where the model predictions and checkpoints will be written.")
        parser.add_argument("--bert_model", default=None, type=str, required=True,
                            help="Bert pre-trained model selected in the list: bert-base-uncased, "
                            "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                            "bert-base-multilingual-cased, bert-base-chinese.")
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
        parser.add_argument("--cache_dir",
                            default="",
                            type=str,
                            help="Where do you want to store the pre-trained models downloaded from s3")
        parser.add_argument("--train_batch_size",
                            default=32,
                            type=int,
                            help="Total batch size for training.")
        parser.add_argument("--eval_batch_size",
                            default=64,
                            type=int,
                            help="Total batch size for eval.")
        parser.add_argument("--loss_func",
                            default='l2-hinge',
                            type=str,
                            help="loss function: bce | l1-hinge | l2-hinge")
        parser.add_argument('--margin', default=1.0, help='margin in hinge loss', type=float)
        parser.add_argument("--learning_rate",
                            default=5e-5,
                            type=float,
                            help="The initial learning rate for Adam.")
        parser.add_argument("--num_train_epochs",
                            default=10,
                            type=int,
                            help="Total number of training epochs to perform.")
        parser.add_argument("--warmup_proportion",
                            default=0.1,
                            type=float,
                            help="Proportion of training to perform linear learning rate warmup for. "
                                 "E.g., 0.1 = 10%% of training.")
        parser.add_argument("--no_cuda",
                            action='store_true',
                            help="Whether not to use CUDA when available")
        parser.add_argument("--local_rank",
                            type=int,
                            default=-1,
                            help="local_rank for distributed training on gpus")
        parser.add_argument('--seed',
                            type=int,
                            default=42,
                            help="random seed for initialization")
        parser.add_argument('--gradient_accumulation_steps',
                            type=int,
                            default=1,
                            help="Number of updates steps to accumulate before performing a backward/update pass.")
        parser.add_argument('--fp16',
                            action='store_true',
                            help="Whether to use 16-bit float precision instead of 32-bit")
        parser.add_argument('--loss_scale',
                            type=float, default=0,
                            help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                                 "0 (default value): dynamic loss scaling.\n"
                                 "Positive power of 2: static loss scaling value.\n")
        parser.add_argument('--server_ip', type=str, default='', help="Can be used for distant debugging.")
        parser.add_argument('--server_port', type=str, default='', help="Can be used for distant debugging.")
        parser.add_argument('--log_interval', default=50, type=int, help='log interval')
        parser.add_argument('--eval_interval', default=100, type=int, help='eval interval')
        parser.add_argument('--only_topk', default=10, type=int, help='store topk prediction for matching stage')

        args = parser.parse_args()
        print(args)

        return {'parser': parser, 'logger': logger, 'args': args}

    @staticmethod
    def bootstrap_for_training(args):
        ''' set device for multi-gpu training, and fix random seed, and exp logging '''
        global n_gpu, device
        if args.server_ip and args.server_port:
            # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
            import ptvsd
            print("Waiting for debugger attach")
            ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
            ptvsd.wait_for_attach()

        if args.local_rank == -1 or args.no_cuda:
            device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
            n_gpu = torch.cuda.device_count()
        else:
            torch.cuda.set_device(args.local_rank)
            device = torch.device("cuda", args.local_rank)
            n_gpu = 1
            # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
            torch.distributed.init_process_group(backend='nccl')
        logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
                        device, n_gpu, bool(args.local_rank != -1), args.fp16))

        if args.gradient_accumulation_steps < 1:
            raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                                args.gradient_accumulation_steps))

        args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if n_gpu > 0:
            torch.cuda.manual_seed_all(args.seed)

        #if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        #    raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

    @staticmethod
    def load_data(args):
        with open(args.data_bin_path, 'rb') as fin:
            data_dict = pickle.load(fin)
        trn_features = data_dict['trn_features']
        val_features = data_dict['val_features']
        tst_features = data_dict['tst_features']
        NUM_LABEL = data_dict['C'].shape[0]
        NUM_CLUSTER = data_dict['C'].shape[1]
        logger.info('TRN {} VAL {} TST {}'.format(len(trn_features), len(val_features), len(tst_features)))
        logger.info('NUM_LABEL {}'.format(NUM_LABEL))
        logger.info('NUM_CLUSTER {}'.format(NUM_CLUSTER))

        # load Y csr matrix
        C_val = data_utils.Ylist_to_Ysparse(data_dict['val']['cseq'], L=NUM_CLUSTER)
        C_tst = data_utils.Ylist_to_Ysparse(data_dict['tst']['cseq'], L=NUM_CLUSTER)
        return {'trn_features': trn_features, 'val_features': val_features, 'tst_features': tst_features,
                'num_clusters': NUM_CLUSTER, 'vocab_size': 30522,
                'NUM_LABEL': NUM_LABEL, 'NUM_CLUSTER': NUM_CLUSTER, 'C_val': C_val, 'C_tst': C_tst}

    def prepare_bert_model(self, args, num_clusters, loss_func='l2-hinge', init_checkpoint_dir=None):
        ''' Load a pretrained BertModel for sequence classification'''
        cache_dir = args.cache_dir if args.cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed_{}'.format(args.local_rank))
        if init_checkpoint_dir is None:
            model = BertForSequenceClassification.from_pretrained(args.bert_model, cache_dir=cache_dir, num_labels=num_clusters)
        else:
            model = BertForSequenceClassification.from_pretrained(init_checkpoint_dir, cache_dir=cache_dir, num_labels=num_clusters)

        if args.fp16:
            model.half()
        model.to(device)
        if args.local_rank != -1:
            try:
                from apex.parallel import DistributedDataParallel as DDP
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

            model = DDP(model)
        elif n_gpu > 1:
            model = torch.nn.DataParallel(model)

        # Prepare Loss Criterion
        if loss_func == 'bce':
            criterion = nn.BCEWithLogitsLoss()
        elif loss_func == 'l1-hinge':
            criterion = HingeLoss(margin=args.margin, squared=False).to(device)
        elif loss_func == 'l2-hinge':
            criterion = HingeLoss(margin=args.margin, squared=True).to(device)
        else:
            raise NotImplementedError('unknown loss function {}'.format(loss_func))

        # overwrite
        self.model = model
        self.criterion = criterion
        self.num_clusters = num_clusters
        self.loss_func = loss_func

    def load(self, args):
        matcher_json_file = os.path.join(args.init_checkpoint_dir, 'xbert.json')
        with open(matcher_json_file, 'r') as fin:
            matcher_json_dict = json.load(fin)
        num_clusters = matcher_json_dict['num_clusters']
        loss_func = matcher_json_dict['loss_func']
        self.prepare_bert_model(args, num_clusters, loss_func=loss_func, init_checkpoint_dir=args.init_checkpoint_dir)

    def save(self, args):
        # Save a trained model, configuration and tokenizer
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model  # Only save the model it-self

        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        # If we save using the predefined names, we can load using `from_pretrained`
        output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)
        output_config_file = os.path.join(args.output_dir, CONFIG_NAME)

        torch.save(model_to_save.state_dict(), output_model_file)
        model_to_save.config.to_json_file(output_config_file)
        #tokenizer.save_vocabulary(args.output_dir)

        output_xbert_file = os.path.join(args.output_dir, 'xbert.json')
        with open(output_xbert_file, 'w') as fout:
            xbert_dict = {'num_clusters': self.num_clusters,
                          'loss_func': self.loss_func}
            json.dump(xbert_dict, fout)

    def predict(self, args, eval_features, C_eval_true, topk=10, verbose=True):
        '''Prediction interface'''
        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d", len(eval_features))
        logger.info("  Batch size = %d", args.eval_batch_size)
        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        all_output_ids = torch.tensor([f.output_ids for f in eval_features], dtype=torch.long)
        all_output_mask = torch.tensor([f.output_mask for f in eval_features], dtype=torch.long)
        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_output_ids, all_output_mask)

        # Run prediction for full data
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

        self.model.eval()
        total_loss = 0.
        total_example = 0.
        rows, cols, vals = [], [], []
        for batch in eval_dataloader:
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, output_ids, output_mask = batch
            cur_batch_size = input_ids.size(0)

            with torch.no_grad():
                c_pred = self.model(input_ids, segment_ids, input_mask)
                c_true = data_utils.repack_output(output_ids, output_mask, self.num_clusters, device)
                loss = self.criterion(c_pred, c_true)
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


    def train(self, args, trn_features, eval_features=None, C_eval=None):
        # Prepare optimizer
        num_train_optimization_steps = int(len(trn_features) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
        if args.local_rank != -1:
            num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()

        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
        if args.fp16:
            try:
                from apex.optimizers import FP16_Optimizer
                from apex.optimizers import FusedAdam
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

            optimizer = FusedAdam(optimizer_grouped_parameters,
                                  lr=args.learning_rate,
                                  bias_correction=False,
                                  max_grad_norm=1.0)
            if args.loss_scale == 0:
                optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
            else:
                optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)
            warmup_linear = WarmupLinearSchedule(warmup=args.warmup_proportion,t_total=num_train_optimization_steps)
        else:
            optimizer = BertAdam(optimizer_grouped_parameters,
                                 lr=args.learning_rate,
                                 warmup=args.warmup_proportion,
                                 t_total=num_train_optimization_steps)

        # Start Batch Training
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(trn_features))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)
        global_step = 0
        nb_tr_steps = 0
        tr_loss = 0
        all_input_ids = torch.tensor([f.input_ids for f in trn_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in trn_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in trn_features], dtype=torch.long)
        all_output_ids = torch.tensor([f.output_ids for f in trn_features], dtype=torch.long)
        all_output_mask = torch.tensor([f.output_mask for f in trn_features], dtype=torch.long)
        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_output_ids, all_output_mask)
        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

        self.model.train()
        total_run_time = 0.0
        best_matcher_prec = -1
        for epoch in range(1, args.num_train_epochs):
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch in enumerate(train_dataloader):
                start_time = time.time()
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, output_ids, output_mask = batch
                c_pred = self.model(input_ids, segment_ids, input_mask)
                c_true = data_utils.repack_output(output_ids, output_mask, self.num_clusters, device)
                loss = self.criterion(c_pred, c_true)

                if n_gpu > 1:
                    loss = loss.mean() # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                if args.fp16:
                    optimizer.backward(loss)
                else:
                    loss.backward()

                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                total_run_time += time.time() - start_time
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    if args.fp16:
                        # modify learning rate with special warm up BERT uses
                        # if args.fp16 is False, BertAdam is used that handles this automatically
                        lr_this_step = args.learning_rate * warmup_linear.get_lr(global_step, args.warmup_proportion)
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr_this_step
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1

                # print training log
                if step % args.log_interval == 0 and step > 0:
                    elapsed = time.time() - start_time
                    cur_loss = tr_loss / nb_tr_steps
                    logger.info("| epoch {:3d} | {:4d}/{:4d} batches | ms/batch {:5.4f} | train_loss {:e}".format(
                        epoch, step, len(train_dataloader), elapsed * 1000 / args.log_interval, cur_loss))

                # eval on dev set and save best model
                if step % args.eval_interval == 0 and step > 0 and args.stop_by_dev:
                    eval_loss, eval_metrics, C_eval_pred = self.predict(args, eval_features, C_eval, topk=args.only_topk, verbose=False)
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
                    self.model.train()    # after model.eval(), reset model.train()

        return self

# transform model prediction optimized under margin-loss
# into smoother curve for ranker
def transform_prediction(csr_codes, transform='lpsvm-l2'):
    if transform == 'sigmoid':
        csr_codes.data[:] = rf_linear.Transform.sigmoid(csr_codes.data[:])
    elif transform == 'lpsvm-l2':
        csr_codes.data[:] = rf_linear.Transform.lpsvm(2, csr_codes.data[:])
    elif transform == 'lpsvm-l3':
        csr_codes.data[:] = rf_linear.Transform.lpsvm(3, csr_codes.data[:])
    else:
        raise NotImplementedError('unknown transform {}'.format(transform))
    return csr_codes



def main():
    # get args
    args = BertMatcher.get_args_and_set_logger()['args']

    # load data
    data = BertMatcher.load_data(args)
    trn_features = data['trn_features']
    val_features = data['val_features']
    tst_features = data['tst_features']
    num_clusters = data['num_clusters']
    C_val = data['C_val']
    C_tst = data['C_tst']

    # if no init_checkpoint_dir,
    # start with random intialization and train
    BertMatcher.bootstrap_for_training(args)
    model = BertMatcher()
    if args.init_checkpoint_dir is None:
        logger.info("start with random initialization")
        model.prepare_bert_model(args, num_clusters, loss_func=args.loss_func)
    else:
        logger.info("start with init_checkpoint_dir")
        model.load(args)

    # do_train and save model
    if args.do_train:
        model.train(args, trn_features, eval_features=val_features, C_eval=C_val)

    # do_eval on test set and save prediction output
    if args.do_eval:
        eval_loss, eval_metrics, C_tst_pred = model.predict(args, tst_features, C_tst, topk=args.only_topk)
        pred_csr_codes = C_tst_pred
        pred_csr_codes = rf_util.smat_util.sorted_csr(pred_csr_codes, only_topk=args.only_topk)
        pred_csr_codes = transform_prediction(pred_csr_codes, transform='lpsvm-l2')
        prediction_path = os.path.join(args.output_dir, 'C_eval_pred.npz')
        smat.save_npz(prediction_path, pred_csr_codes)


if __name__ == '__main__':
    main()
