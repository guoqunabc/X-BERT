# coding=utf-8

import argparse
import sys
import os
from os import path, system
import time
import collections
import itertools
import pickle
import json
import ctypes
from ctypes import *

import scipy as sp
import scipy.sparse as smat
from sklearn.preprocessing import normalize as sk_normalize

import xbert.indexer as indexer
import xbert.rf_util as rf_util
import xbert.rf_linear as rf_linear
from xbert.rf_util import PyMatrix
from xbert.rf_linear import Data, MLProblem, Metrics, HierarchicalMLModel, ml_train

# solver_type
solver_dict = {
    #'L2R_LR':0,
    'L2R_L2LOSS_SVC_DUAL':1,
    #'L2R_L2LOSS_SVC':2,
    'L2R_L1LOSS_SVC_DUAL':3,
    #'MCSVM_CS':4,
    'L1R_L2LOSS_SVC':5,
    #'L1R_LR':6,
    'L2R_LR_DUAL':7
    }


class LinearModel(object):

    def __init__(self, model=None):
        self.model = model

    def __getitem__(self, key):
        return LinearModel(self.model[key])

    def __add__(self, other):
        return LinearModel(self.model + other.model)

    def save(self, model_folder):
        self.model.save(model_folder)

    @classmethod
    def load(cls, model_folder):
        return cls(HierarchicalMLModel.load(model_folder))

    @classmethod
    def train(cls, X, Y, C,
            mode='full-model',
            hierarchical=True,
            solver_type=solver_dict['L2R_L2LOSS_SVC_DUAL'],
            Cp=1.0, Cn=1.0, threshold=0.1, max_iter=100, threads=-1):
        if mode == 'full-model':
            prob = MLProblem(X, Y, C)
        elif mode == 'matcher':
            assert C is not None
            Y = Y.dot(C)
            prob = MLProblem(X, Y)
        elif mode == 'ranker':
            assert C is not None
            prob = MLProblem(X, Y, C)
            hierarchical=False

        model = ml_train(prob,
                         hierarchical=hierarchical,
                         min_labels=2,
                         solver_type=solver_type,
                         Cp=Cp, Cn=Cn,
                         threshold=threshold,
                         threads=threads,
                         max_iter=max_iter)
        return cls(model)

    def predict(self, X, csr_codes=None, beam_size=10, only_topk=10):
        cond_prob = True
        pred_csr = self.model.predict(X,
                                      only_topk=only_topk,
                                      csr_codes=csr_codes,
                                      beam_size=beam_size,
                                      cond_prob=cond_prob)
        return pred_csr

class SubCommand(object):
    def __init__(self):
        pass

    @classmethod
    def add_parser(cls, super_parser):
        pass

    @staticmethod
    def add_arguments(parser):
        pass

class LinearTrainCommand(SubCommand):

    @staticmethod
    def run(args):
        C = smat.load_npz(args.input_code_path)
        X = smat.load_npz(args.input_inst_feat)
        Y = smat.load_npz(args.input_inst_label)

        model = LinearModel.train(X, Y, C,
                    solver_type=solver_dict[args.solver_type],
                    Cp=args.Cp,
                    Cn=args.Cn,
                    threshold=args.threshold,
                    threads=args.threads,
                    )
        model.save(args.output_ranker_folder)


    @classmethod
    def add_parser(cls, super_parser):
        parser = super_parser.add_parser('train',
                aliases=[], help="Train a linear ranker with codes")
        cls.add_arguments(parser)
        parser.set_defaults(run=cls.run)

    @staticmethod
    def add_arguments(parser):
        parser.add_argument("-x", "--input-inst-feat", type=str, required=True, metavar="PATH",
                help="path to the npz file of the feature matrix (CSR, nr_insts * nr_feats)")

        parser.add_argument("-y", "--input-inst-label", type=str, required=True, metavar="PATH",
                help="path to the npz file of the label matrix (CSR, nr_insts * nr_labels)")

        parser.add_argument("-c", "--input-code-path", type=str, required=True, metavar="PATH",
                help="path to the npz file of the indexing codes (CSR, nr_labels * nr_codes)")

        parser.add_argument("-o", "--output-ranker-folder", type=str, required=True, metavar="DIR",
                help="directory for storing linear ranker")

        parser.add_argument("-s", "--solver-type", type=str, default='L2R_L2LOSS_SVC_DUAL', metavar="SOLVER_STR",
                help="{} (default L2R_L2LOSS_SVC_DUAL)".format(' | '.join(solver_dict.keys())))

        parser.add_argument("--Cp", type=float, default=1.0, metavar="VAL",
                help="coefficient for positive class in the loss function (default 1.0)")

        parser.add_argument("--Cn", type=float, default=1.0, metavar="VAL",
                help="coefficient for negative class in the loss function (default 1.0)")

        parser.add_argument("-t", "--threshold", type=float, default=0.1, metavar="VAL",
                help="threshold to sparsify the model weights (default 0.1)")

        parser.add_argument("-n", "--threads", type=int, default=-1, metavar="INT",
                help="number of threads to use (defautl -1 to denote all the CPUs)")



class LinearPredictCommand(SubCommand):
    @staticmethod
    def run(args):
        Xt = smat.load_npz(args.input_inst_feat)
        model = LinearModel.load(args.input_ranker_folder)
        # input_csr_code is
        # get only ranker part if csr_codes from a matcher is provided
        if args.input_csr_code is not None and os.path.exists(args.input_csr_code):
            csr_codes = smat.load_npz(args.input_csr_code)
            model = model[-1]
        else:
            csr_codes = None
        Yt_pred = model.predict(Xt, csr_codes=csr_codes, beam_size=args.beam_size, only_topk=args.only_topk)
        if args.input_inst_label is not None and path.exists(args.input_inst_label):
            Yt = smat.load_npz(args.input_inst_label) if args.input_inst_label else None
            metric = Metrics.generate(Yt, Yt_pred, topk=10)
            print('==== tst_set evaluation ====')
            print(metric)

        smat.save_npz(args.output_path, Yt_pred)

    @classmethod
    def add_parser(cls, super_parser):
        parser = super_parser.add_parser('predict',
                aliases=[], help="Generate predictions based on the given ranker")
        cls.add_arguments(parser)
        parser.set_defaults(run=cls.run)

    @staticmethod
    def add_arguments(parser):
        parser.add_argument("-m", "--input-ranker-folder", type=str, required=True,
                help="path to the ranker folder")

        parser.add_argument("-x", "--input-inst-feat", type=str, required=True,
                help="path to the npz file of the feature matrix (CSR)")

        parser.add_argument("-y", "--input-inst-label", type=str, required=False,
                help="path to the npz file of the label matrix (CSR) for computing metrics")

        parser.add_argument("-o", "--output-path", type=str, required=True,
                help="path to the npz file of output prediction (CSR)")

        parser.add_argument("-c", "--input-csr-code", type=str, required=False,
                help="path to the npz file of the csr codes generated by the matcher")

        parser.add_argument("-t", "--csr-code-transform", type=str, default='lpsvm-l2',
                help="transform of csr codes generated by the matcher sigmoid | lpsvm-l2 | l2svm-l3 (default lpsvm-l2)")

        parser.add_argument("-k", "--only-topk", type=int, default=20,
                help="number of top labels in the prediction")

        parser.add_argument("-b", "--beam-size", type=int, default=10,
                help="size of beam search in the prediction")


def get_parser():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(
                    help="subcommands",
                    metavar="SUBCOMMAND")
    subparsers.required = True
    LinearTrainCommand.add_parser(subparsers)
    LinearPredictCommand.add_parser(subparsers)
    return parser


if __name__ == '__main__':

    parser = get_parser()
    args = parser.parse_args()
    args.run(args)
