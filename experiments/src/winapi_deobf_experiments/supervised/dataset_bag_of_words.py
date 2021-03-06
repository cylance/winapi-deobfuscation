
import numpy as np 

from winapi_deobf_experiments.util.io import read_filepath_of_delimited_ints_as_vector
from winapi_deobf_experiments.supervised.dataset import Dataset

def get_bag_of_words_dataset_for_classification(corpus, K, label_tracker, seed=1234):

	np.random.seed(seed=seed)
	train_pct=_get_train_pct(label_tracker)
	X = _get_bag_of_words_matrix(corpus)
	Xproj = _rotate_bag_of_words_onto_dominant_right_singular_vectors(X, K)

	N=len(corpus.numeric_corpus)
	train_idxs=np.random.choice(N, int(N*train_pct))
	test_idxs=np.setdiff1d(range(N),train_idxs)
	features_train=Xproj[train_idxs,:]
	features_test=Xproj[test_idxs,:]
	labels_train=[corpus.get_numeric_label(x) for x in train_idxs]
	labels_test=[corpus.get_numeric_label(x) for x in test_idxs]

	#TD: add assertion that # train idxs + # test idxs has right size. 
	dataset=Dataset(features_train=features_train, features_test=features_test, 
		labels_train=labels_train, labels_test=labels_test, train_idxs=train_idxs, 
		test_idxs=test_idxs)
	return dataset 

def _get_bag_of_words_matrix(corpus):
	W=len(corpus.vocab_dict)
	N=len(corpus.numeric_corpus)
	X=np.zeros((N,W))
	for (i,numeric_words) in enumerate(corpus.numeric_corpus):
		for numeric_word in numeric_words:
			X[i,numeric_word]=1.0
	return X 

def _rotate_bag_of_words_onto_dominant_right_singular_vectors(X, K):
	U, d, V = np.linalg.svd(X, full_matrices=True)
	Xproj=np.matmul(X,np.transpose(V[1:(K+3),:]))
	return Xproj 

def _get_train_pct(label_tracker):
	train_pcts=[]
	for label, label_info in label_tracker.iteritems():
		train_pcts.append(label_info["train_pct"])
	if np.sum(np.diff(train_pcts))!=0.0:
		raise ValueError("Experiment 1 assumes a uniform training pct across labels")
	return train_pcts[0]