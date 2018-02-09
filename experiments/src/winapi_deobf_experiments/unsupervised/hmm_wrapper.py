"""
Goal: Generate new features with HMM features attached 

"""


from __future__ import division
from builtins import range #pyhsmm package i think 
import os, sys
import numpy as np
np.seterr(divide='ignore') # these warnings are usually harmless for this code

import pyhsmm
from pyhsmm.util.text import progprint_xrange

from winapi_deobf_experiments.unsupervised.hmm_trainer import fit_separate_hmms_for_each_label, fit_single_hmm_to_all_labels
from winapi_deobf_experiments.unsupervised.hmm_redumper import HMM_Redumper

def create_unsupervised_data(corpus, configs, dict_labels_to_fixed_ts_length=None):
	"""
	Returns:
		results_paths: A dictionary mapping hmm latent state parameter k to an inner dictionary, where the 
			inner dictionary provides the filepaths for a few things: path_to_dumped_data, path_to_fvm,
			train_idxs_filepath, and test_idxs_filepath 
	"""
	print("\n Now dumping features with K=%d \n" %(configs.K))
	if configs.use_separate_hmms_for_each_label:
		hmms=fit_separate_hmms_for_each_label(corpus, configs, configs.K, dict_labels_to_fixed_ts_length)
	else:
		hmms=fit_single_hmm_to_all_labels(corpus, configs, configs.K)

	### RESULTS
	results = HMM_Redumper(configs.results_super_dir, configs.run_summary, hmms, corpus, configs.K)  
	unsupervised_results_paths  = results.redump_data(streaming=True, feature_types=set(configs.feature_types), \
			add_numeric_labels=True, dict_labels_to_fixed_ts_length=dict_labels_to_fixed_ts_length, 
			train_test_or_all="all", default_unlikely_value=configs.default_unlikely_value) 
	print("For K=%d, now redumping data to %s, train_idxs at %s, fvm at %s" \
		%(configs.K, unsupervised_results_paths.paths_to_redumped_data, unsupervised_results_paths.path_to_train_idxs, \
		unsupervised_results_paths.path_to_fvm))
	return unsupervised_results_paths
