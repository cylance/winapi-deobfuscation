"""
Goal: Generate new features with HMM features attached 

"""


from __future__ import division
from builtins import range #pyhsmm package i think 
import os, sys
import numpy as np
np.seterr(divide='ignore') # these warnings are usually harmless for this code
import multiprocessing as mp
import functools 
import inspect 

import pyhsmm
from pyhsmm.util.text import progprint_xrange



def fit_single_hmm_to_all_labels(corpus, configs, K):
	"""
		Used when we're fitting a single hmm to the entire API call dataset. 
	"""

	obs_type=_get_obs_type_from_corpus(corpus)
	
	np.random.seed(seed=configs.seed)
	hmms={}

	print("Fitting hmm model 1/1.")

	#initialize model 
	hmm = _initialize_hmm(corpus, K, obs_type)
	#grab relevant data 
	train_cut=[corpus.numeric_corpus[idx] for idx in corpus.cut.use_idxs]
	# transform corpus to useable Data (i.e. integer time-series) and add to hmm class 
	add_corpus_to_HMM(hmm, train_cut) 	

	#fit hmm 
	hmm=_fit_hmm(hmm, algo_name=configs.model_type)

	# store in raw dictionary form 
	hmms["single_model"]=hmm

	return hmms 

def fit_separate_hmms_for_each_label(corpus, configs, K, dict_labels_to_fixed_ts_length=None):
	"""
	Used when we're fitting a separate hmm to each api function. 
	"""

	obs_type=_get_obs_type_from_corpus(corpus)

	np.random.seed(seed=configs.seed)
	hmms={}

	for (idx, label) in enumerate(set(corpus.labels)):
		print("Constructing hmm model %d of %d" %(idx+1, len(set(corpus.labels))))
		last_obs_idx=get_max_num_obs_for_this_label(dict_labels_to_fixed_ts_length, label)
		hmm = _initialize_hmm(corpus, K, obs_type)
		train_cut=[corpus.numeric_corpus[idx][:last_obs_idx] for idx in corpus.cut.use_idxs \
			if corpus.labels[idx]==label] #retain idx if (a) in 80% train set and (b) has matching api call
		add_corpus_to_HMM(hmm,train_cut) 

		### ### Fit Model
		hmm=_fit_hmm(hmm, algo_name=configs.model_type)

		### ### Store in dictionary form 
		hmms[label]=hmm
	return hmms 

def _get_obs_type_from_corpus(corpus):
	parent_types_as_strings=str(inspect.getmro(type(corpus)))
	is_categorical = "CategoricalCorpus" in parent_types_as_strings 
	is_numeric =  "NumericCorpus" in parent_types_as_strings 
	if is_categorical == is_numeric:
		raise ValueError("Corpus type should be exclusively either categorical or numeric")
	if is_categorical:
		return "categorical"
	else:
		return "numeric"

def get_max_num_obs_for_this_label(dict_labels_to_fixed_ts_length, label):
	if not dict_labels_to_fixed_ts_length:
		last_obs_idx=None 
	else:
		last_obs_idx=dict_labels_to_fixed_ts_length[label]
	return last_obs_idx 

def add_corpus_to_HMM(hmm, numeric_corpus):
	for time_series in numeric_corpus:
		hmm.add_data(time_series)

def _initialize_hmm(corpus, K, obs_type):
	if obs_type is not "numeric" and not "categorical":
		raise ValueError("data_type must be numeric or categorical, but it's listed as %s"\
			%(corpus.obs_type))
	elif obs_type is "categorical":
		hmm = _initialize_hmm_categorical(corpus, K)
	elif obs_type is "numeric":
		hmm = _initialize_hmm_numeric(corpus, K)
	return hmm

def _initialize_hmm_categorical(corpus, K, alpha_0=1.0, alpha=3., init_state_concentration=1.):
	obs_hypparams=dict(alpha_0=alpha_0, K=corpus.W) 	#get hyperparameters 
	hmm = pyhsmm.models.HMM(
		alpha=alpha,init_state_concentration= init_state_concentration, # these are only used for initialization
		obs_distns=[pyhsmm.distributions.Categorical(**obs_hypparams) for i in range(K)])
	return hmm

def _initialize_hmm_numeric(corpus, K, mu_0_for_all_dims=4.0, sigma_0_for_all_dims=4.0, 
	kappa_0=0.3, nu_0_increment=5.0, alpha=3., init_state_concentration=1.0):
	obs_dim=np.shape(corpus.numeric_corpus[0])[1] #dimension of observations 
	obs_hypparams = {'mu_0':np.eye(obs_dim)*mu_0_for_all_dims,
	                'sigma_0':np.eye(obs_dim)*sigma_0_for_all_dims,
	                'kappa_0':kappa_0,
	                'nu_0':obs_dim+nu_0_increment}
	#TD: **** revisit these interps
	obs_distns = [pyhsmm.distributions.Gaussian(**obs_hypparams) for state in range(K)]
	hmm = pyhsmm.models.HMM(
		alpha=alpha, init_state_concentration= init_state_concentration, # these are only used for initialization
		obs_distns=obs_distns)
	return hmm 


def _fit_hmm(hmm, algo_name):
	if algo_name=="Baum-Welch":
		print('Gibbs sampling for initialization')
		for idx in progprint_xrange(25):
			hmm.resample_model()
		## in models line 440, we se that this is resampling the parameters and resampling the states 
		print('fit EM model')
		likes = hmm.EM_fit()
	else:
		raise NotImplementedError
	return hmm 




