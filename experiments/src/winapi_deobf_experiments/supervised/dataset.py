
# Can look to see how well the API call's last argument's latent state distributions work as predictive features 
# for predicting the API call itself. 
# This was used on Vadim's API call data. 

import numpy as np 
import csv
from sklearn.linear_model import LogisticRegression
np.set_printoptions(suppress=True)

from winapi_deobf_experiments.util.io import read_filepath_of_delimited_ints_as_vector


class Dataset(object):
	__acceptable_keys_list_shortform = ["features", "labels", "train_idxs", "test_idxs"]
	__acceptable_keys_list_longform = ["features_train", "features_test", "labels_train", "labels_test", "train_idxs", "test_idxs"]
	
	def __init__(self, **kwargs):
		#My attempt at mimicking an overloaded function in Julia. 
		if set(kwargs.keys())==set(self.__acceptable_keys_list_shortform):
			self.features_train = np.take(kwargs["features"], kwargs["train_idxs"], axis=0)
			self.features_test = np.take(kwargs["features"], kwargs["test_idxs"], axis=0)
			self.labels_train = np.take(kwargs["labels"], kwargs["train_idxs"], axis=0)
			self.labels_test = np.take(kwargs["labels"], kwargs["test_idxs"], axis=0)
			self.train_idxs = kwargs["train_idxs"]
			self.test_idxs = kwargs["test_idxs"]
		elif  set(kwargs.keys())==set(self.__acceptable_keys_list_longform):
 			[self.__setattr__(key, kwargs.get(key)) for key in self.__acceptable_keys_list_longform]
		else:
			raise NotImplementedError("You did not provide an understandable set of arguments to class Dataset")

def get_dataset_for_classification(unsupervised_results_filepaths, K, dict_api_calls_to_n_args=None, \
	use_separate_hmms_for_each_api_function=True):

	M = _get_number_of_hmm_models(dict_api_calls_to_n_args, use_separate_hmms_for_each_api_function)
	labels, features= _get_labels_and_features_from_filepath(unsupervised_results_filepaths.path_to_dumped_data, \
		label_idx=-1, M=M, feature_end_idx=-1, K=K) 

	train_idxs=read_filepath_of_delimited_ints_as_vector(unsupervised_results_filepaths.train_idxs_filepath)
	test_idxs=read_filepath_of_delimited_ints_as_vector(unsupervised_results_filepaths.test_idxs_filepath)
	
	#TD: add assertion that # train idxs + # test idxs has right size. 
	dataset=Dataset(features=features, labels=labels, train_idxs=train_idxs, test_idxs=test_idxs)
	return dataset 

def _get_number_of_hmm_models(dict_api_calls_to_n_args, use_separate_hmms_for_each_api_function):
	if use_separate_hmms_for_each_api_function:
		M=len(dict_api_calls_to_n_args.keys()) #num_models! 
	else: 
		M=1	
	return M 

def _get_labels_and_features_from_filepath(filepath, label_idx, M, feature_end_idx, K):
	"""
			Arguments: 
			M: Int. Number of models 
	"""
	feature_start_idx=-M*(K+3)-1
	labels, features=[],[]
	with open(filepath,'rb') as csvfile:
		csv_reader = csv.reader(csvfile, delimiter=',')
		for (idx,row) in enumerate(csv_reader):
			label=float(row[label_idx])
			if not np.isnan(label):#	Rows that have nan's are ignored (this should be vestigal...)
				feature_vec=np.array(row[feature_start_idx:feature_end_idx], dtype=np.float)
				labels.append(label)
				features.append(feature_vec)
	features=np.concatenate(features).reshape(np.shape(features)) #convert list of arrays to 2d array 	
	_check_that_hidden_state_marginals_sum_to_one_for_each_model(features, K, M)
	return np.array(labels, dtype="int"), features 

def _check_that_hidden_state_marginals_sum_to_one_for_each_model(features, K, M):
	n_samples=len(features)
	for m in range(M):
		for i in range(n_samples):
			sum_of_hidden_state_marginals= np.sum(np.exp(features[i,m*(K+3):m*(K+3)+K]))
			#this handles default unlikely value where everything is unlikely. 
			if not np.isclose(sum_of_hidden_state_marginals, 1.0) and not np.isclose(sum_of_hidden_state_marginals, 0.0):  
				raise ValueError("The latent state marginals in the feature matrix don't sum to 1 after exponentiation" \
					"for model %d and sample %d, the sum was %.04f" %(m, i, sum_of_hidden_state_marginals))

def ensure_indexing_alignment_of_dataset_and_corpus_for_expt_2(dataset, corpus):
	"""
	Arguments:
		dataset: Instance of class DatasetForOneK
		corpus: Instance of class Corpus. 
	"""
	for i in range(len(dataset.labels_train)):
		assert(corpus.numeric_labels[dataset.train_idxs[i]]==dataset.labels_train[i])
