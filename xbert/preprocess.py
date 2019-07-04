#!/usr/bin/env python
# encoding: utf-8

import argparse
from collections import Counter
import itertools
import json
import os
from os import path
import logging
import numpy as np
import pickle
from tqdm import tqdm
import scipy as sp
import scipy.sparse as smat
from pytorch_pretrained_bert.tokenization import BertTokenizer

import xbert.Constants as Constants
from xbert.data_utils import InputExample, InputFeatures


logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
        datefmt = '%m/%d/%Y %H:%M:%S',
        level = logging.INFO)
logger = logging.getLogger(__name__)


# Dataset is a set of quadruple:
# D = {(x, c, yp, yn)_i}, i=1,...,N
# xseq  is a list of string
# cseq  is a list of cluster_id
# yseq is a list of positive labels
def load_mlc2seq_data(data_path, csr_codes, matcher='xbert'):
    xseq_list, cseq_list, yseq_list = [], [], []
    with open(data_path, 'r') as fin:
        for idx, line in enumerate(tqdm(fin)):
            label_str, text_str = line.strip().split('\t')
            assert(len(text_str) > 0)
            assert(len(label_str) > 0)

            # xseq
            if matcher == 'xbert':
                xseq = text_str
            elif matcher == 'xttention':
                xseq = text_str.split()
            else:
                raise ValueError('unknown matcher!')
            xseq_list.append(xseq)
            # yseq
            yseq = list(map(int, label_str.split(',')))
            yseq_list.append(yseq)
            # cseq
            cseq = [csr_codes[y] for y in yseq]
            cseq_list.append(cseq)

    return xseq_list, cseq_list, yseq_list


# self-defined helper functions for preprocessing
# matcher=xttention
def build_vocab_map(sentences, max_vocab_size=50000):
    ''' Trim vocab by number of occurence '''
    word_counts = Counter(itertools.chain(*sentences))

    #wid2word = [Constants.PAD_WORD, Constants.UNK_WORD]
    wid2word = [Constants.UNK_WORD, Constants.PAD_WORD]
    wid2word = wid2word + [x[0] for x in word_counts.most_common(max_vocab_size+1) if x[0] != Constants.PAD_WORD]
    word2wid = {x: i for i, x in enumerate(wid2word)}
    print('[Info] Original x_vocabulary size = {},'.format(len(word_counts)))
    print('[Info] Trimmed x_vocabulary size = {},'.format(len(word2wid)))

    wid2word = {v: k for k, v in word2wid.items()}
    return word2wid, wid2word

def convert_raw_to_id(raw_x_list, word2wid):
    ignored_doc_count = 0
    ret_x_list = []
    for i, x_word_list in enumerate(raw_x_list):
        # filter out document whose wid are all <UNK>
        x_wid_list = [word2wid[w] if w in word2wid else Constants.UNK for w in x_word_list]
        #if all([x_wid == Constants.UNK for x_wid in x_wid_list]):
        #    ignored_doc_count += 1
        #    continue
        ret_x_list.append(x_wid_list)
    print('[Info] Trimmed documents = {},'.format(len(ret_x_list)),
          'Ignored documents = {}'.format(ignored_doc_count))
    return ret_x_list

def write_dict_map(key2val, save_file_name):
    with open(save_file_name, 'w') as fout:
        for key in sorted(key2val):
            fout.write('{} ||| {}\n'.format(key, key2val[key]))


# self-defined helper functions for preprocessing
# matcher=xbert
def create_examples(xseq_list, cseq_list, set_type):
    """Creates examples for the training and dev sets."""

    examples = []
    for i, (xseq, cseq) in enumerate(zip(xseq_list, cseq_list)):
        guid = "%s-%s" % (set_type, i)
        examples.append(InputExample(guid=guid, text=xseq, label=cseq))
    return examples

def convert_examples_to_features(examples, tokenizer, max_xseq_len=512, max_cseq_len=128):
    """Loads a data file into a list of `InputBatch`s."""
    features = []
    xseq_lens, cseq_lens = [], []
    for (ex_index, example) in enumerate(examples):
        tokens = tokenizer.tokenize(example.text)
        xseq_lens.append(len(tokens))

        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens) > max_xseq_len - 2:
            tokens = tokens[:(max_xseq_len - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = ["[CLS]"] + tokens + ["[SEP]"]
        segment_ids = [0] * len(tokens)
        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_xseq_len - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_xseq_len
        assert len(input_mask) == max_xseq_len
        assert len(segment_ids) == max_xseq_len

        # labels
        labels = example.label
        cseq_lens.append(len(labels))
        if len(labels) > max_cseq_len:
            labels = labels[:max_cseq_len]
        output_ids = labels
        output_mask = [1] * len(output_ids)
        padding = [0] * (max_cseq_len - len(output_ids))
        output_ids += padding
        output_mask += padding

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("output_ids: %s" % " ".join([str(x) for x in output_ids]))
            logger.info("output_mask: %s" % " ".join([str(x) for x in output_mask]))

        tmp = InputFeatures(input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
                output_ids=output_ids,
                output_mask=output_mask)
        features.append(tmp)
    return features, xseq_lens, cseq_lens

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def main(args):
    # set hyper-parameters
    input_data_dir = args.input_data_dir
    input_code_path = args.input_code_path
    output_data_dir = args.output_data_dir

    # load existing code
    C = smat.load_npz(input_code_path)
    csr_codes = C.nonzero()[1]

    # a short glimpse at clustering result for fast debugging
    logger.info('NUM_LABELS: {}'.format(C.shape[0]))
    logger.info('NUM_CLUSTERS: {}'.format(C.shape[1]))

    # load data in mlc2seq format
    logger.info('loading mlc2seq data into quadruple set')
    dir_path = '{}/mlc2seq'.format(args.input_data_dir)
    trn_path = os.path.join(dir_path, 'train.txt')
    val_path = os.path.join(dir_path, 'valid.txt')
    tst_path = os.path.join(dir_path, 'test.txt')
    trn_xseq_list, trn_cseq_list, trn_yseq_list = load_mlc2seq_data(trn_path, csr_codes, matcher=args.matcher)
    val_xseq_list, val_cseq_list, val_yseq_list = load_mlc2seq_data(val_path, csr_codes, matcher=args.matcher)
    tst_xseq_list, tst_cseq_list, tst_yseq_list = load_mlc2seq_data(tst_path, csr_codes, matcher=args.matcher)

    if args.matcher == 'xbert':
        # BERT preprocess and tokenizer pipeline
        tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)
        trn_examples = create_examples(trn_xseq_list, trn_cseq_list, 'trn')
        val_examples = create_examples(val_xseq_list, val_cseq_list, 'val')
        tst_examples = create_examples(tst_xseq_list, tst_cseq_list, 'tst')

        trn_features, xseq_lens, cseq_lens = convert_examples_to_features(trn_examples, tokenizer, args.max_xseq_len, args.max_cseq_len)
        logger.info('trn_xseq: min={} max={} mean={} median={}'.format(np.min(xseq_lens), np.max(xseq_lens), np.mean(xseq_lens), np.median(xseq_lens)))
        logger.info('trn_cseq: min={} max={} mean={} median={}'.format(np.min(cseq_lens), np.max(cseq_lens), np.mean(cseq_lens), np.median(cseq_lens)))
        val_features, xseq_lens, cseq_lens = convert_examples_to_features(val_examples, tokenizer, args.max_xseq_len, args.max_cseq_len)
        logger.info('val_xseq: min={} max={} mean={} median={}'.format(np.min(xseq_lens), np.max(xseq_lens), np.mean(xseq_lens), np.median(xseq_lens)))
        logger.info('val_cseq: min={} max={} mean={} median={}'.format(np.min(cseq_lens), np.max(cseq_lens), np.mean(cseq_lens), np.median(cseq_lens)))
        tst_features, xseq_lens, cseq_lens = convert_examples_to_features(tst_examples, tokenizer, args.max_xseq_len, args.max_cseq_len)
        logger.info('tst_xseq: min={} max={} mean={} median={}'.format(np.min(xseq_lens), np.max(xseq_lens), np.mean(xseq_lens), np.median(xseq_lens)))
        logger.info('tst_cseq: min={} max={} mean={} median={}'.format(np.min(cseq_lens), np.max(cseq_lens), np.mean(cseq_lens), np.median(cseq_lens)))

        # save data dict
        data = {
                'args': args,
                'C': C,
                'trn': {
                    'cseq': trn_cseq_list,
                    'yseq': trn_yseq_list},
                'val': {
                    'cseq': val_cseq_list,
                    'yseq': val_yseq_list},
                'tst': {
                    'cseq': tst_cseq_list,
                    'yseq': tst_yseq_list},
                'tokenizer': tokenizer,
                'trn_features': trn_features,
                'val_features': val_features,
                'tst_features': tst_features,
                }

    elif args.matcher == 'xttention':
        #  Build vocabulary and category map
        print('| building word vocabulary from document texts')
        all_xseq_list = trn_xseq_list + val_xseq_list + tst_xseq_list
        stoi, itos = build_vocab_map(all_xseq_list, max_vocab_size=args.vocab_size)

        # convert str to int
        trn_xseq_list = convert_raw_to_id(trn_xseq_list, stoi)
        val_xseq_list = convert_raw_to_id(val_xseq_list, stoi)
        tst_xseq_list = convert_raw_to_id(tst_xseq_list, stoi)

        # save data dict
        data = {
            'args': args,
            'C': C,
            'stoi': stoi,
            'itos': itos,
            'trn': {
                'xseq': trn_xseq_list,
                'cseq': trn_cseq_list,
                'yseq': trn_yseq_list},
            'val': {
                'xseq': val_xseq_list,
                'cseq': val_cseq_list,
                'yseq': val_yseq_list},
            'tst': {
                'xseq': tst_xseq_list,
                'cseq': tst_cseq_list,
                'yseq': tst_yseq_list}
        }

    else:
        raise NotImplementedError('unknown matcher {}: currently support [xbert|xttention]'.format(args.matcher))

    logger.info('Dumping the processed data to pickle file')
    output_data_path = path.join(output_data_dir, 'data_dict.pt')
    with open(output_data_path, 'wb') as fout:
        pickle.dump(data, fout, protocol=pickle.HIGHEST_PROTOCOL)
    output_config_path = path.join(output_data_dir, 'config.json')
    with open(output_config_path, 'w') as fout:
        json.dump(vars(args), fout)
    logger.info('Finish.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("-m", '--matcher', type=str, required=True,
            default='xbert',
            help='preprocess for matcher [xbert|xttention]')
    parser.add_argument("-i", "--input-data-dir", type=str, required=True, metavar="DIR",
            default='./datasets/Eurlex-4K',
            help="path to the dataset directory containing mls2seq/")
    parser.add_argument("-c", "--input-code-path", type=str, required=True, metavar="PATH",
            default='./save_models/Eurlex-4K/indexer/code.npz',
            help="path to the npz file of the indexing codes (CSR, nr_labels * nr_codes)")
    parser.add_argument("-o", "--output-data-dir", type=str, required=True, metavar="DIR",
            default='./save_models/Eurlex-4K/elmo-a0-s0/data-data-xbert',
            help="directory for storing data_dict.pkl")

    # for xttention data pre-processing
    parser.add_argument('--vocab-size', type=int, default=80000)

    # preprocessing document text
    parser.add_argument("--bert_model", default='bert-base-uncased', type=str,
            help="Bert pre-trained model selected in the list: bert-base-uncased, "
            "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
            "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--max_xseq_len",
            default=512,
            type=int,
            help="The maximum total input sequence length after WordPiece tokenization. \n"
            "Sequences longer than this will be truncated, and sequences shorter \n"
            "than this will be padded.")
    parser.add_argument("--max_cseq_len",
            default=32,
            type=int,
            help="The maximum total output sequence length. \n"
            "Sequences longer than this will be truncated, and sequences shorter \n"
            "than this will be padded.")
    parser.add_argument("--do_lower_case",
            action='store_true',
            help="Set this flag if you are using an uncased model.")

    args = parser.parse_args()
    print(args)
    main(args)
