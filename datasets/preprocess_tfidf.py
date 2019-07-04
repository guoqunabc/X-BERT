#!/usr/bin/env python3 -u

import argparse
import os, sys
from sklearn.datasets import load_svmlight_file
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import normalize
import pickle
import scipy.sparse as sp


# mlc2seq format (each line):
# l_1,..,l_k \t w_1 w_2 ... w_t
def parse_mlc2seq_format(data_path):
    assert(os.path.isfile(data_path))
    with open(data_path) as fin:
        labels, corpus = [], []
        for line in fin:
            tmp = line.strip().split('\t', 1)
            labels.append(tmp[0])
            corpus.append(tmp[1])
    return labels, corpus


def main(args):
    # load data
    print('loading corpus and labels...')
    trn_labels, trn_corpus = parse_mlc2seq_format('{}/mlc2seq/train.txt'.format(args.input_data_dir))
    val_labels, val_corpus = parse_mlc2seq_format('{}/mlc2seq/valid.txt'.format(args.input_data_dir))
    tst_labels, tst_corpus = parse_mlc2seq_format('{}/mlc2seq/test.txt'.format(args.input_data_dir))

    # create features
    print('creating features...')
    if args.feat_type == 0 or args.feat_type == 1:
        vectorizer = CountVectorizer(
                ngram_range=(1, args.ngram),
                min_df=args.min_df,)

        X_wordcount_trn = vectorizer.fit_transform(trn_corpus)
        X_wordcount_val = vectorizer.transform(val_corpus)
        X_wordcount_tst = vectorizer.transform(tst_corpus)

        def convert_feat(X, feat_type=None):
            if feat_type == 0:
                X_ret = X.copy()
                X_ret[X_ret>0] = 1
            elif feat_type == 1:
                X_ret = X.copy()
            else:
                raise NotImplementedError('unknown feature_type!')
        X_ret_trn = convert_feat(X_wordcount_trn, feat_type=args.feat_type)
        X_ret_val = convert_feat(X_wordcount_val, feat_type=args.feat_type)
        X_ret_tst = convert_feat(X_wordcount_tst, feat_type=args.feat_type)

    elif args.feat_type == 2:
        vectorizer = TfidfVectorizer(
                ngram_range=(1, args.ngram),
                min_df=args.min_df,)
        X_ret_trn = vectorizer.fit_transform(trn_corpus)
        X_ret_val = vectorizer.transform(val_corpus)
        X_ret_tst = vectorizer.transform(tst_corpus)

    else:
        raise NotImplementedError('unknown feature_type!')


    # convert list of label strings into list of list of label
    # label index start from 0
    def convert_label_to_Y(labels, K_in=None):
        rows, cols ,vals = [], [], []
        for i, label in enumerate(labels):
            label_list = list(map(int, label.split(',')))
            rows += [i] * len(label_list)
            cols += label_list
            vals += [1] * len(label_list)

        K_out = max(cols) + 1 if K_in is None else K_in
        Y = sp.csr_matrix( (vals, (rows,cols)), shape=(len(labels),K_out) )
        return Y

    # save vectorizer as pickle
    print('saving vectorized features into libsvm format and dictionary into pickle...')
    out_file_name = '{}/vectorizer.pkl'.format(args.input_data_dir)
    with open(out_file_name, 'wb') as fout:
        pickle.dump(vectorizer, fout, protocol=pickle.HIGHEST_PROTOCOL)
	
	# save X and Y
	def save_npz(X, Y, set_flag='trn'):
		sp.save_npz('{}/X.{}.npz'.format(args.input_data_dir, set_flag), X)
		sp.save_npz('{}/Y.{}.npz'.format(args.input_data_dir, set_flag), Y)
    
	# write file back into libsvm format
    Y_ret_trn = convert_label_to_Y(trn_labels, K_in=None)
    save_npz(X_ret_trn, Y_ret_trn, , set_flag='trn')

    Y_ret_val = convert_label_to_Y(val_labels, K_in=Y_ret_trn.shape[1])
    save_npz(X_ret_val, Y_ret_val, , set_flag='val')

    Y_ret_tst = convert_label_to_Y(tst_labels, K_in=Y_ret_trn.shape[1])
    save_npz(X_ret_tst, Y_ret_tst, , set_flag='tst')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='read mlc2seq/[train|valid|test].txt and produce n-gram tfidf features')
    parser.add_argument('-i', '--input-data-dir', type=str, required=True, help='dataset directory: data_dir/mlcseq/*.txt')
    parser.add_argument('--feat-type', type=int, default=2, help='0: binary feature, 1: word count, 2: TF-IDF')
    # parser.add_argument('--normalize', type=int, required=True, help='0: no normalization, 1: L1 normalization, 2: L2 normalization')
    parser.add_argument('--ngram', type=int, default=1, help='ngram features')
    parser.add_argument('--min-df', type=int, default=1, help='keep words whose df >= min_df (absolute count)')
    args = parser.parse_args()
    print(args)
    main(args)
