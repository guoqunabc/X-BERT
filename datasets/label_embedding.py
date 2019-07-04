#!/usr/bin/env python
# encoding: utf-8

import argparse
import h5py
import os
import numpy as np
from sklearn.datasets import load_svmlight_file
from sklearn.preprocessing import MultiLabelBinarizer
import scipy.sparse as sp
import pickle
from sklearn.preprocessing import normalize
from tqdm import tqdm


def main(args):
	if args.embed_type == 'elmo':

		# clean label text, which is dataset dependent!
		# save emlo embedding to disk
		elmo_layer_path = "{}.elmo_layers.hdf5".format(args.dataset)
		if not os.path.exists(elmo_layer_path):
			cmd="python clean_label_text.py --dataset {} > ./label_text.txt".format(args.dataset)
			print(cmd)
			os.system(cmd)
			
			cmd="allennlp elmo ./label_text.txt {} --all --cuda-device 0".format(elmo_layer_path)
			print(cmd)
			os.system(cmd)
		 
		# load emlo embedding and generate sentence embedding
		# by mean pooling over time and concate all three layers embedding
		h5py_file = h5py.File(elmo_layer_path, 'r')
		num_label = len(h5py_file) - 1
		label_embedding = np.zeros((num_label, 1024 * 3))
		for idx in tqdm(range(num_label)):
			embedding = h5py_file.get(str(idx))
			embedding = np.mean(embedding, axis=1).reshape(-1)
			label_embedding[idx] = embedding
		label_embedding = sp.csr_matrix(label_embedding)
	
	elif args.embed_type == 'pifa':
		# load TF-IDF and label matrix
		X = sp.load_npz("./{}/X.trn.npz".format(args.dataset))
		Y = sp.load_npz("./{}/Y.trn.npz".format(args.dataset))
		assert(Y.getformat() == 'csr')
		print('X', type(X), X.shape)
		print('Y', type(Y), Y.shape)
		# create label embedding
		Y_avg = normalize(Y, axis=1, norm='l2')
		label_embedding = sp.csr_matrix(Y_avg.T.dot(X))
		label_embedding = normalize(label_embedding, axis=1, norm='l2')

	else:
		raise NotImplementedError('unknown embed_type {}'.format(args.embed_type))

	# save label embedding
	print('label_embedding', type(label_embedding), label_embedding.shape)
	label_embedding_path = "{}/L.{}.npz".format(args.dataset, args.embed_type)
	sp.save_npz(label_embedding_path, label_embedding)	



if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--dataset', default='Eurlex-4K', type=str, help='dataset')
	parser.add_argument('--embed-type', default='pifa', type=str, help='elmo|pifa')
	args = parser.parse_args()
	print(args)
	main(args)
