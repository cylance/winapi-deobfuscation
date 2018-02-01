
from __future__ import division
#from builtins import range #pyhsmm package i think 
import os, sys
import numpy as np
np.seterr(divide='ignore') # these warnings are usually harmless for this code

from winapi_deobf_experiments.preprocess.corpus import *
from winapi_deobf_experiments.unsupervised.results import UnsupervisedResults
from winapi_deobf_experiments.unsupervised.train import fit_single_hmm_to_all_api_functions, fit_separate_hmms_for_each_api_function

def create_unsupervised_data(K, corpus, configs, use_separate_hmms_for_each_api_function=True,
		dict_api_calls_to_n_args=None):
	"""
		Returns:
			results_paths: A dictionary mapping hmm latent state parameter k to an inner dictionary, where the 
				inner dictionary provides the filepaths for a few things: path_to_dumped_data, path_to_fvm,
				train_idxs_filepath, and test_idxs_filepath 
	"""
	print("\n Now dumping features with K=%d \n" %(K))
	if use_separate_hmms_for_each_api_function:
		hmms=fit_separate_hmms_for_each_api_function(corpus, K, configs, dict_api_calls_to_n_args)
	else:
		hmms=fit_single_hmm_to_all_api_functions(corpus, K, configs)

	### RESULTS
	results = UnsupervisedResults(configs, hmms, corpus, K)  
	unsupervised_results_paths  = results.redump_data(streaming=True, store_empirical_transition_matrix=False, \
		store_latent_state_series=False, one_time_series_per_row=True, use_likelihood_feats=True, \
		use_latent_state_marginals=True, add_numeric_labels=True, dict_model_names_to_n_args=dict_api_calls_to_n_args,
		train_test_or_all="all", default_unlikely_value=configs.default_unlikely_value) 
	print("For K=%d, now redumping data to %s, train_idxs at %s, fvm at %s" \
		%(K, unsupervised_results_paths.path_to_dumped_data, unsupervised_results_paths.train_idxs_filepath, \
		unsupervised_results_paths.path_to_fvm))
	return unsupervised_results_paths
