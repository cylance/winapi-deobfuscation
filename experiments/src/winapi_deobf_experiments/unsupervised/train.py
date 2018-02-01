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

from winapi_deobf_experiments.preprocess.corpus import make_corpus

### ### TRAINING ROUTINE 
# (Didn't use API for now to get more fine grained control. 

def fit_single_hmm_to_all_api_functions(corpus, K, configs, single_cut=True):
	"""
		Used when we're fitting a single hmm to the entire API call dataset. 
	"""
	if not single_cut:
		raise NotImplementedError
	else:
		assert len(corpus.cuts.cuts.keys())==1 #should only be one cut for api call expts 
		only_cut_name = corpus.cuts.cuts.keys()[0]
	
	hmms={}

	print("Fitting hmm model 1/1.")

	#initialize model 
	hmm = _initialize_hmm(corpus.W, K)
	#grab relevant data 
	train_cut=[corpus.numeric_corpus[idx] for idx in corpus.cuts.cuts[only_cut_name].use_idxs]
	# transform corpus to useable Data (i.e. integer time-series) and add to hmm class 
	add_corpus_to_HMM(hmm,train_cut) 	

	#fit hmm 
	hmm=_fit_hmm(hmm, algo_name=configs.model_type)

	# store in raw dictionary form 
	hmms[only_cut_name]=hmm

	return hmms 

def fit_separate_hmms_for_each_api_function(corpus, K, configs, dict_api_calls_to_n_args):
	"""
		Used when we're fitting a separate hmm to each api function. 
	"""
	hmms={}
	assert len(corpus.cuts.cuts.keys())==1 #should only be one cut for api call expts 
	only_cut_name=corpus.cuts.cuts.keys()[0]

	for (idx, api_call) in enumerate(set(corpus.labels)):
		print("Constructing hmm model %d of %d" %(idx, len(set(corpus.labels))))
		n_args_this_api_call=dict_api_calls_to_n_args[api_call]
		hmm = _initialize_hmm(corpus.W, K)
		train_cut=[corpus.numeric_corpus[idx][:n_args_this_api_call] for idx in corpus.cuts.cuts[only_cut_name].use_idxs \
			if corpus.labels[idx]==api_call] #retain idx if (a) in 80% train set and (b) has matching api call
		add_corpus_to_HMM(hmm,train_cut) 	### ### transform corpus to useable Data (i.e. integer time-series)

		### ### Fit Model
		hmm=_fit_hmm(hmm, algo_name=configs.model_type)

		### ### Store in dictionary form 
		hmms[api_call]=hmm
	return hmms 


def add_corpus_to_HMM(hmm, numeric_corpus):
	for process_stream in numeric_corpus:
		hmm.add_data(process_stream)

def _initialize_hmm(W, K, alpha_0=1.0, alpha=3., init_state_concentration=1.):
	obs_hypparams=dict(alpha_0=alpha_0, K=W) 	#get hyperparameters 
	hmm = pyhsmm.models.HMM(
		alpha=alpha,init_state_concentration= init_state_concentration, # these are only used for initialization
		obs_distns=[pyhsmm.distributions.Categorical(**obs_hypparams) for i in range(K)])
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




