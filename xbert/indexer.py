#!/usr/bin/env python
# encoding: utf-8

import argparse
import json
import os
from os import path
import scipy as sp
import scipy.sparse as smat
import ctypes
from ctypes import *

from xbert.rf_util import PyMatrix, fillprototype, load_dynamic_library

class RandomProject(object):

    """Encode and decode a label into a K-way D-dimensional code.

    feat_mat: L by P matrix
    L: number of label
    P: label feature dimension
    K: range of the code = {0, 1, 2, ..., K-1}
    D: length of the codes for each label (number of hashing functions)
    """

    def __init__(self, feat_mat, kdim, depth, algo, seed):
        self.feat_mat = feat_mat
        self.code_delim = ' | '
        self.K, self.D = kdim, depth
        self.L, self.P = feat_mat.shape
        self.algo = algo
        self.random_matrix = np.random.randn(self.P, self.D)

    def get_codes(self):
        if self.algo == 2:      # ordinal
            Z_quant = self.ordinal_quantization(self.feat_mat)
        elif self.algo == 3:    # uniform
            Z_quant = self.uniform_quantization(self.feat_mat)
        else:
            raise NotImplementedError('unknown algo {}'.format(self.algo))
        self.hash_func = np.array([self.K**d for d in range(0, self.D)])
        return Z_quant.dot(self.hash_func)

    # Z: L by D projected label embedding
    # Z_quant: L by D quantized projected label embeddings
    # quantize it by ordinal ranking
    def ordinal_quantization(self, label_embedding):
        Z = label_embedding.dot(self.random_matrix)
        Z_argsort = np.argsort(Z, axis=0)
        bin_size = math.ceil(self.L *1. / self.K)
        Z_quant = []
        for d in range(self.D):
            rank = np.zeros(self.L, dtype=np.int64)
            rank[Z_argsort[:,d]] = np.arange(self.L)
            quantized_rows = (rank // bin_size).tolist()
            Z_quant.append(quantized_rows)
        Z_quant = np.array(Z_quant).T
        return Z_quant

    # quantize it by min/max of each cols into K bins
    def uniform_quantization(self, label_embedding):
        Z = label_embedding.dot(self.random_matrix)
        Z_quant = []
        for d in range(self.D):
            bins = np.linspace( min(Z[:,d]), max(Z[:,d]), self.K )
            quantized_rows = np.digitize(Z[:,d], bins) - 1 # bits = {0, 1, ..., K-1}
            Z_quant.append(quantized_rows)
        Z_quant = np.array(Z_quant).T
        return Z_quant


    def prepare_coding(self, Z_quant):
        # L dimensional array,
        # each entry is the hash code (row idx)
        self.hash_code_arr = Z_quant.dot(self.hash_func)
        rows, cols, vals = [], [], []
        for l in range(self.L):
            rows.append(self.hash_code_arr[l])
            cols.append(l)
            vals.append(1)

        m = self.K**self.D
        n = self.L
        M = sp.csr_matrix( (vals, (rows, cols)), shape=(m,n) )
        self.code2label_mat = M

        code2label_set = {}
        for code in np.nonzero(M.indptr[1:] - M.indptr[:-1])[0]:
            code2label_set[code] = set(M.indices[M.indptr[code]:M.indptr[code+1]])
        self.code2label_set = code2label_set


class corelib(object):
    def __init__(self, dirname, soname, forced_rebuild=False):
        self.clib_float32 = load_dynamic_library(dirname, soname + '_float32', forced_rebuild=forced_rebuild)
        self.clib_float64 = load_dynamic_library(dirname, soname + '_float64', forced_rebuild=forced_rebuild)
        arg_list = [
                    POINTER(PyMatrix),
                    c_uint32,
                    c_uint32,
                    c_int32,
                    c_uint32,
                    c_int32,
                    POINTER(c_uint32)
                    ]
        fillprototype(self.clib_float32.get_codes, None, arg_list)
        fillprototype(self.clib_float64.get_codes, None, arg_list)

    def get_codes(self, py_feat_mat, depth, algo, seed, codes, verbose=0, max_iter=10, threads=-1):
        clib = self.clib_float32
        if py_feat_mat.dtype == sp.float64:
            clib = self.clib_float64
            if verbose != 0:
                print('perform float64 computation')
        else:
            clib = self.clib_float32
            if verbose != 0:
                print('perform float32 computation')
        clib.get_codes(byref(py_feat_mat), depth, algo, seed, max_iter, threads, codes.ctypes.data_as(POINTER(c_uint32)))

forced_rebuild = False
corelib_path = path.join(path.dirname(path.abspath(__file__)), 'corelib/')
soname = 'rf_linear'
clib = corelib(corelib_path, soname, forced_rebuild)

# SEmatic-aware Code
class SeC(object):
    def __init__(self, kdim, depth, algo, seed, codes):
        assert(kdim == 2)
        self.kdim = kdim
        self.depth = depth
        self.algo = algo
        self.seed = seed
        self.codes = codes
        self.indptr = sp.cumsum(sp.bincount(codes + 1, minlength=(self.nr_codes + 1)), dtype=sp.uint64)
        self.indices = sp.argsort(codes * sp.float64(self.nr_elements) + sp.arange(self.nr_elements))

    @property
    def nr_elements(self):
        return len(self.codes)

    @property
    def nr_codes(self):
        return 1 << self.depth

    def __len__(self):
        return len(self.codes)

    def get_code_for_element(self, eid):
        assert 0 <= eid and eid < self.nr_elements
        return self.codes[eid]

    def get_elements_with_code(self, code):
        assert 0 <= code and code < self.nr_codes
        begin, end = self.indptr[code], self.indptr[code + 1]
        return self.indices[begin:end]

    def get_csc_matrix(self):
        return smat.csc_matrix((sp.ones_like(self.indices, dtype=sp.float64), self.indices, self.indptr), shape=(self.nr_elements, self.nr_codes))

    def print(self):
        print('nr_codes: {}'.format(self.nr_codes))
        print('nr_elements: {}'.format(self.nr_elements))
        print('algo: {}'.format(Indexer.algos[self.algo]))
        for nid in range(self.nr_codes):
            labels = ' '.join(map(str, self.get_elements_with_code(nid)))
            print('code({nid}): {labels}'.format(nid=nid, labels=labels))


class Indexer(object):
    KMEANS=0
    KDTREE=1    # KDTREE with Roound-Robin feature splits
    ORDINAL=2   # random projection with ordinal quantization
    UNIFORM=3   # random projection with uniform quantization
    BALANCED_ORDINAL=4  # random projection with balaced ordinal quantization
    SKMEANS=5   # Spherical KMEANS
    KDTREE_CYCLIC=11  # KDTREE with cyclic feature splits( 0,...,0, 1,...,1, 2,...,2)

    algos = {v:k for k, v in vars().items() if isinstance(v, int)}

    def __init__(self, feat_mat):
        self.py_feat_mat = PyMatrix.init_from(feat_mat)

    @property
    def feat_mat(self):
        return self.py_feat_mat.buf

    def estimate_depth_with_nr_clusters(self, nr_clusters):
        nr_labels = self.feat_mat.rows
        depth = int(sp.log2(nr_clusters))
        return depth

    def estimate_depth_with_cluster_size(self, cluster_size):
        return self.estimate_depth_with_nr_clusters(nr_labels // cluster_size + 1)

    def ordinal_gen(self, kdim, depth, seed):
        sp.random.seed(seed)
        random_matrix = sp.randn(self.feat_mat.shape[1], depth)
        X = self.feat_mat.dot(random_matrix)
        m = self.feat_mat.shape[0] // kdim + [1, 0][self.feat_mat.shape[0] % kdim != 0]
        X = sp.argsort(sp.argsort(X, axis=0), axis=0) // m
        print(X)
        codes = sp.array((X * (kdim ** sp.arange(depth)).reshape(1, -1)).sum(axis=1), dtype=sp.uint32)
        return codes

    def balaced_ordinal_gen(self, kdim, depth, seed, threads=-1):
        assert int(2**sp.log2(kdim)) == kdim
        sp.random.seed(seed)
        random_matrix = sp.randn(self.feat_mat.shape[1], depth)
        X = PyMatrix(self.feat_mat.dot(random_matrix))
        codes = sp.zeros(X.rows, dtype=sp.uint32)
        new_depth = depth * int(sp.log2(kdim))
        clib.get_codes(X, new_depth, Indexer.KDTREE_CYCLIC, seed, codes, threads=threads)
        return codes

    def gen(self, kdim, depth, algo, seed, max_iter=10, threads=-1):
        assert algo in [Indexer.KMEANS, Indexer.KDTREE, Indexer.ORDINAL, Indexer.UNIFORM, Indexer.BALANCED_ORDINAL, Indexer.KDTREE_CYCLIC, Indexer.SKMEANS]
        if algo in [Indexer.KMEANS, Indexer.KDTREE, Indexer.KDTREE_CYCLIC, Indexer.SKMEANS]:
            feat_mat = self.py_feat_mat
            codes = sp.zeros(feat_mat.rows, dtype=sp.uint32)
            clib.get_codes(feat_mat, depth, algo, seed, codes, max_iter=max_iter, threads=threads)
        elif algo in [Indexer.ORDINAL, Indexer.UNIFORM]:
            rp_clf = RandomProject(self.feat_mat, kdim, depth, algo, seed)
            codes = rp_clf.get_codes()
        elif algo in [Indexer.BALANCED_ORDINAL]:
            assert int(2**sp.log2(kdim)) == kdim
            codes = self.balaced_ordinal_gen(kdim, depth, seed, threads=threads)
        else:
            raise NotImplementedError('unknown algo {}'.format(algo))
        return SeC(kdim, depth, algo, seed, codes)


def run_test(data_folder='./datasets/Eurlex-4K'):
    import xbert.rf_linear as rf_linear
    data = rf_linear.Data.load(data_folder, label_emb=None)
    L = smat.load_npz(data_folder + '/L.pifa.npz')
    code = Indexer(L).gen(kdim=2, depth=6, algo=0, seed=5, max_iter=20, threads=-1)
    code.print()
    code = Indexer(L).gen(kdim=2, depth=6, algo=5, seed=5, max_iter=20, threads=-1)
    code.print()

def main(args):
    # set hyper-parameters
    input_feat_path = args.input_feat_path
    depth = args.depth
    kdim = args.kdim
    algo = args.algo
    seed = args.seed
    verbose = args.verbose
    max_iter = args.max_iter
    threads = args.threads
    output_code_dir = args.output_code_dir
    if verbose:
        print('depth {} kdim {} algo {}'.format(depth, kdim, algo))

    # load label feature matrix (nr_labels * nr_features)
    if path.exists(input_feat_path):
        feat_mat = smat.load_npz(input_feat_path)
    else:
        raise ValueError('label embedding path does not exist {}'.format(input_feat_path))

    # Indexing algorithm
    # C: nr_labels x nr_codes, stored in csr sparse matrix
    code = Indexer(feat_mat).gen(kdim=kdim, depth=depth, algo=algo, seed=seed, max_iter=max_iter, threads=threads)
    if verbose:
        code.print()
    C = code.get_csc_matrix()
    if verbose:
        print('C', C.shape)

    # save code and args
    output_code_path = path.join(output_code_dir, 'code.npz')
    smat.save_npz('{}'.format(output_code_path), C, compressed=False)
    output_config_path = path.join(output_code_dir, 'config.json')
    with open(output_config_path, 'w') as fout:
        json.dump(vars(args), fout)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("-i", "--input-feat-path", type=str, required=True,
            default='./datasets/Eurlex-4K/L.pifa.npz',
            help="path to the npz file of input label feature matrix (nr_labels * nr_features, CSR)")
    parser.add_argument("-o", "--output-code-dir", type=str, required=True,
            default='./save_models/Eurlex-4K/indexer/code.npz',
            help="path to the output npz file of indexing codes (nr_labels * nr_codes, CSR)")
    parser.add_argument("-d", "--depth", type=int, required=True,
            default=6,
            help="The depth of hierarchical 2-means")
    # optional
    parser.add_argument("--algo", type=int, default=5, help="0 for KMEANS 5 for SKMEANS (default 5)")
    parser.add_argument("--seed", type=int, default=0, help="random seed (default 0)")
    parser.add_argument("--kdim", type=int, default=2)
    parser.add_argument("--threads", type=int, default=-1)
    parser.add_argument("--max-iter", type=int, default=20)
    parser.add_argument("-v", "--verbose", type=bool, default=False)

    args = parser.parse_args()
    print(args)
    main(args)
