
import numpy as np 
import copy #to copy arrays 
import os 
import pickle 
import time, datetime #to get date, also to time things with @timeit decorator
import csv #for saving modelexploration results 

from winapi_deobf_experiments.unsupervised.hmm_feature_maker import HMMFeatureMaker 
from winapi_deobf_experiments.util.io import write_vector, read_text_file_as_list_of_strings, \
	ensure_dirs, read_first_line, write_rows_to_text, file_len, save_obj
from winapi_deobf_experiments.util.representation import Bunch, interleave_lists 
from winapi_deobf_experiments.util.math_funs import generate_feature_idx_bound_tuples_per_group, \
	convert_np_array_into_comma_separated_string, nudge_prob_matrix_away_from_zero

class UnsupervisedResults:
	"""
		To generate and save results.  Originally set up as a separate class after corpus and hmm 
		because its attributes (e.g. description) could use information from both
		the corpus and the hmm; thus we don't "pollute" the corpus class with info not available
		before running the hmm. 
	"""			


	def __init__(self, configs, hmms, corpus, K, model_names=None,
		make_model_subdirs=False):
		# Having another "corpus" in here would carry high storage costs. 
		# But luckily, Python seems to be making pointers, not copies.  
		# (Changing unsupervised_results.corpus.blah also changes corpus.blah)
		self.hmms = hmms
		self.corpus = corpus
		self.configs=configs
		self.K=K 
		self.model_names = None 
		self.description=None
		self.local_results_dir = None 
		self._initialize(model_names, corpus, make_model_subdirs)
		#workaround, because Python doesn't support default args which depend on vals of other args. 


	### ### Initialization Methods 
	#TD: generalize this for single time series case 
	#TD: make configs a class for better code isolation if I change the config entries later
	def _initialize(self, model_names, corpus, make_model_subdirs=False):
		#TD:  Initialization currently identical for Results and EDA classes; smooth this up. 
		self._set_model_names(model_names)
		self._set_description()
		self.local_results_dir=os.path.join(self.configs.results_super_dir,self.description)
		self._ensure_directories(make_model_subdirs)


	def _set_description(self):
		""" 
			Note this function must be run (and so its class must be initalized) --at least-- after constructing the corpus and vocab, b/c
			it uses that information.    It could potentially also use information from the hmm.
		"""
		self.description=""
		self.description=self.description+self.configs.run_summary+"_"
		self.description=self.description+"K"+"="+str(self.K)+"_"
		self.description=self.description+"T"+"="+str(len(self.corpus.corpus))+"_" #num documents in corpus
		self.description=self.description+"W"+"="+str(self.corpus.W)+"_"
		self.description=self.description+"Date"+"="+time.strftime("%m-%d-%Y")

	def _set_model_names(self, model_names):
		if model_names is None:
			self.model_names = self.hmms.keys()
		else:
			self.model_names = model_names 	

	def _ensure_directories(self, make_model_subdirs=False):
		if not os.path.isdir(self.configs.results_super_dir):
			os.mkdir(self.configs.results_super_dir) 
		if not os.path.isdir(self.local_results_dir):
			os.mkdir(self.local_results_dir) 
		if make_model_subdirs:
			for model_name in self.model_names:
				#TD: make array or dictionary storing these separate paths direclty in the class. 
				model_subdir=os.path.join(self.local_results_dir, model_name)
				if not os.path.isdir(model_subdir):
					os.mkdir(model_subdir) 

	def redump_data(self, streaming=True, store_empirical_transition_matrix=False, \
			store_latent_state_series=False, one_time_series_per_row=True, use_likelihood_feats=True, \
			use_latent_state_marginals=True, add_numeric_labels=True, dict_model_names_to_n_args=None,
			train_test_or_all="all", default_unlikely_value=-200.0, single_cut=True):
		"""
			We append new features to the original data and dump the transformed data elsewhere. 
			The dumps are organized in the same way as the original raw data. 

			Parameters:
				streaming:  If True, the features (latent state marginals and event/sequence likelihoods)
					are obtained in a streaming way, i.e. based on partial sequences (up to time t) rather
					than the full sequences. 

		"""
		if not single_cut:
			raise NotImplementedError
		else:
			assert len(self.corpus.cuts.cuts.keys())==1 #should only be one cut for api call expts 
			only_cut_name = self.corpus.cuts.cuts.keys()[0]
		self._get_model_parameters(store_empirical_transition_matrix, store_latent_state_series)
		#one HMMFeatureMaker instance for each cut. 
		HMMFeatureMakerArray=[HMMFeatureMaker(self.hmms[model_name], model_name, streaming, use_latent_state_marginals, use_likelihood_feats) \
			for model_name in self.model_names]
		assert len(self.corpus.filepath_trackers)==1
		filepath, filepath_tracker = self.corpus.filepath_trackers.keys()[0], self.corpus.filepath_trackers.values()[0]
		rl = read_text_file_as_list_of_strings(filepath)
		#TD: save new_filepaths somewhere 
		main_results_dir=self._get_and_ensure_data_redump_dir(filepath)
		path_to_fvm=os.path.join(os.path.dirname(main_results_dir),"fvm.txt")
		self._make_fvm(path_to_fvm, HMMFeatureMakerArray, add_numeric_labels)
		train_idxs_filepath, test_idxs_filepath = self._write_train_and_test_idxs_for_filepath(main_results_dir, only_cut_name)
		path_to_dumped_data= os.path.join(main_results_dir, os.path.basename(filepath))
		print("Now obtaining feature for file %s" % (filepath))
		feature_matrix=self.get_prediction_features_for_this_filepath(filepath, only_cut_name, HMMFeatureMakerArray, \
			streaming, one_time_series_per_row, use_likelihood_feats, use_latent_state_marginals, \
			add_numeric_labels, dict_model_names_to_n_args, train_test_or_all, default_unlikely_value)
		print("Now dumping dataset with new features appended to %s" % (path_to_dumped_data))
		with open(path_to_dumped_data,'w') as f:
			for (row_idx,row) in enumerate(feature_matrix):
				old_line=rl[row_idx]
				new_line=old_line.rstrip()+","+convert_np_array_into_comma_separated_string(feature_matrix[row_idx,:])+"\n" #removes new line adds in data "2"
				f.write(new_line)
		return UnsupervisedResultsPaths(path_to_dumped_data, path_to_fvm, train_idxs_filepath,test_idxs_filepath)
			
	def _make_fvm(self, path_to_fvm, HMMFeatureMakerArray, add_numeric_labels):
		"""
			FVM for redumped dataset that describes ONLY the new features (appended at end)
			Assumes that the HMMFeatureMakerArray class generates features in order [latent state marginals, likelihood features]
		"""
		#reads in first line of original dataset 
		if os.path.exists(path_to_fvm):
			os.remove(path_to_fvm)
		for idx, fma in enumerate(HMMFeatureMakerArray):
			 write_rows_to_text(path_to_fvm, fma.feature_names, mode="a+", return_delimited=True)
		if add_numeric_labels:
			write_rows_to_text(path_to_fvm, ["numeric_label"], mode="a+", return_delimited=True)

	def _write_train_and_test_idxs_for_filepath(self, new_dir, only_cut_name):
		train_idxs=self.corpus.cuts.cuts[only_cut_name].use_idxs 
		test_idxs=self.corpus.cuts.cuts[only_cut_name].withheld_idxs
		train_idxs_filepath=os.path.join(new_dir,"train_idxs")
		test_idxs_filepath=os.path.join(new_dir,"test_idxs")
		write_vector(train_idxs_filepath, train_idxs)
		write_vector(test_idxs_filepath, test_idxs)
		return train_idxs_filepath, test_idxs_filepath 

	def _get_and_ensure_data_redump_dir(self, filepath):
		new_dir=os.path.join(self.local_results_dir, os.path.basename(os.path.dirname(filepath))+"_new_dump")	
		ensure_dirs(new_dir) #recursive mkdir 
		return new_dir 

	def get_prediction_features_for_this_filepath(self, filepath, only_cut_name, HMMFeatureMakerArray, streaming=False,
			one_time_series_per_row=True, use_likelihood_feats=True, use_latent_state_marginals=True, add_numeric_labels=True,
			dict_model_names_to_n_args=None, train_test_or_all="all", default_unlikely_value=-200.0):
		"""
			API call dataset has one time series per line (and T.S. have diff't lengths across lines)
			We'll just return the features from the last observation then.   We can use in particular
			the marginal probs across latent states (and maybe the streaming_mean_log_likelihood_of_events_in_partial_seq)
			See the FVM for the feature names. 

			Attributes:
				streaming:  Bool.  If true, the HMM is fit in a streaming way (we only use forward algorithm, 
					not forward-backwards; i.e. we don't use any sequence tokens which are past the current value 
					in determining latent state distributions, obs likelihoods, etc.). 
				one_time_series_per_row: Bool.  If true, the original and dumped data have one sequence 
					per row (a la API call data) rather than one event/observation/token per row.
				add_numeric_labels: Bool. 
				dict_model_names_to_n_args: Dict mapping strings to ints.  If none, will use whole sample.  If provided, will truncate
					samples during test phase to the appropriate number of args for that model. 
				train_test_or_all: String.  If "train" we generate features for the train indices for the cut, if "test" we generate for the 
					test indices, if "all" we use all. 
		"""

		if not use_latent_state_marginals and not use_likelihood_feats:
			raise ValueError("You have requested forming features without using latent state marginals OR likelihood features."\
				"Nonsense!")
		if not streaming:
			raise NotImplementedError("Need to implement the Non-streaming version of this,"\
					 	"i.e fit using both forward AND backward algorithm")
		nFeatsPerModel=self._get_n_features_per_model(use_likelihood_feats, use_latent_state_marginals)
		nFeatsForNumericLabel=1 if add_numeric_labels else 0 
		first_time_series_idx_for_filepath=self.corpus.filepath_trackers[filepath].start_idx_of_time_series_for_filepath
		last_time_series_idx_for_filepath=self.corpus.filepath_trackers[filepath].stop_idx_of_time_series_for_filepath
		F,L=first_time_series_idx_for_filepath,last_time_series_idx_for_filepath
		num_events_in_filepath=sum(map(lambda x: len(x), self.corpus.numeric_corpus[F:(L+1)]))
		nModels=len(HMMFeatureMakerArray)
		# We're adding features in a hcatted way, one set of features (with nFeats) for each Model, so 
		# this gives us the 1st and last index of features for each of the nModels models 
		feature_idx_bounds_array=self._get_feature_idx_bounds_array(nFeatsPerModel, nModels, add_numeric_labels)
		if one_time_series_per_row:
			feature_matrix=np.zeros((L-F+1,nFeatsPerModel*nModels+nFeatsForNumericLabel))
		else:
			feature_matrix=np.zeros((num_events_in_filepath,nFeatsPerModel*nModels+nFeatsForNumericLabel))
		n_events_so_far=0
		for ts_idx in range(first_time_series_idx_for_filepath, last_time_series_idx_for_filepath+1):
			print("Now generating new features for time series %d of %d" %(ts_idx, L))
			sample=self.corpus.numeric_corpus[ts_idx]
			T=len(sample)
			for (model_idx, model_fm) in enumerate(HMMFeatureMakerArray):
				log_state_marginals, log_likelihoods=self.get_log_state_marginals_and_log_likelihoods_for_sample(sample, model_fm, \
					dict_model_names_to_n_args, default_unlikely_value=default_unlikely_value)
				if use_latent_state_marginals:
					if use_likelihood_feats:
						new_features=np.hstack((log_state_marginals, log_likelihoods))
					else:
						new_features=log_state_marginals
				else:
					new_features=log_likelihoods
				if train_test_or_all=="train":
					prohibited_indices=self.corpus.cuts.cuts[only_cut_name].withheld_idxs
				elif train_test_or_all=="test":
					prohibited_indices=self.corpus.cuts.cuts[only_cut_name].use_idxs
				elif train_test_or_all=="all":
					prohibited_indices=[]
				if ts_idx in prohibited_indices:
					new_features*=np.nan 
				if one_time_series_per_row:
					feature_matrix[ts_idx,feature_idx_bounds_array[model_idx][0]:feature_idx_bounds_array[model_idx][1]]=new_features[-1,:] #TD add in the right feature indices.  
				else:
					feature_matrix[n_events_so_far:(n_events_so_far+T),feature_idx_bounds_array[model_idx][0]:\
						feature_idx_bounds_array[model_idx][1]]=new_features #TD add in the right feature indices.
			n_events_so_far+=T
			#add in the extra numeric label:
			if add_numeric_labels:
				numeric_label= self.corpus.numeric_labels[ts_idx]
				numeric_label_feature_idx=feature_idx_bounds_array[-1][0]
				if one_time_series_per_row:
					feature_matrix[ts_idx,numeric_label_feature_idx]=numeric_label  
				else:
					numeric_labels=np.ones_like(log_likelihoods[:,0])*numeric_label
					feature_matrix[n_events_so_far:(n_events_so_far+T),numeric_label_feature_idx:\
						numeric_label_feature_idx]=nnumeric_labels#TD add in the right feature indices.
		return feature_matrix 

	def get_log_state_marginals_and_log_likelihoods_for_sample(self, sample, model_fm, dict_model_names_to_n_args,
			default_unlikely_value=-200.0):
		T=len(sample)
		if dict_model_names_to_n_args:
			#extract, from each sample, only the relevant n_args for that model 
			n_args_for_model=dict_model_names_to_n_args[model_fm.model_name]
			sample=sample[:n_args_for_model]
		log_alpha=model_fm.forward_algorithm_for_one_sample(sample)
		log_state_marginals=model_fm.get_streaming_log_state_marginals_for_one_sample(log_alpha)
		self._check_that_latent_state_marginals_sum_to_one(log_state_marginals)
		a,b,c=model_fm.derive_likelihood_features_for_one_sample(log_alpha)
		log_likelihoods=np.hstack((a,b,c))
		if dict_model_names_to_n_args:
			if n_args_for_model>T: #model needs more args than are avialable; automatic exclusion
				log_state_marginals=np.ones_like(log_state_marginals)*default_unlikely_value
				log_likelihoods=np.ones_like(log_likelihoods)*default_unlikely_value
		return log_state_marginals, log_likelihoods 

 	def _get_feature_idx_bounds_array(self, nFeats,nModels, add_numeric_labels):
 		"""
 			Returns:	
 				List of tuples of ints.  Length of list is equal to the number of models, plus 1 if 
 					we're adding numeric labels.  The tuples have form (a,b) where a is the 1st index of features 
 					relevant to that model, and b is the 2nd. 
 		"""
 		feature_idx_bounds_array=generate_feature_idx_bound_tuples_per_group(nFeats,nModels)
 		if add_numeric_labels:
 			numeric_label_feature_idx=feature_idx_bounds_array[-1][1]
 			feature_idx_bounds_array.append((numeric_label_feature_idx,numeric_label_feature_idx))
 		return feature_idx_bounds_array 

	def _get_n_features_per_model(self, use_likelihood_feats, use_latent_state_marginals):
		nFeats=0
		if use_likelihood_feats: 
			nFeats+=3
		if use_latent_state_marginals:
			nFeats+=self.K
		return nFeats 

	def _check_that_latent_state_marginals_sum_to_one(self, log_state_marginals):
		assert np.isclose(np.sum(np.exp(log_state_marginals)), len(log_state_marginals)) 

	def _get_number_of_events_for_a_filepath(self, filepath):
		first_time_series_idx_for_filepath=self.corpus.filepath_trackers[filepath].start_idx_of_time_series_for_filepaths
		last_time_series_idx_for_filepath=self.corpus.filepath_trackers[filepath].stop_idx_of_time_series_for_filepaths
		num_events_from_filepath=sum([len(x) for x in self.corpus.corpus[first_time_series_idx_for_filepath:(last_time_series_idx_for_filepath+1)]])
		return num_events_from_filepath 

	def _get_new_filepath(self, new_data_dir, filepath):
		new_filepath=new_data_dir+os.path.split(os.path.split(filepath)[0])[1]+"_"+os.path.split(filepath)[1]
		return new_filepath 

	### ### Methods to Construct Results (Model Parameters)
	# TD: currently over whole corpus, refine this so we get it for each cut. 
	def _get_empirical_transition_matrix(self):
		W = self.corpus.W
		event_names = [get_key_given_value(self.corpus.vocab_dict,i) for i in range(W)]
		bigram_counts = _get_bigram_frequencies(self.corpus.corpus)
		self.corpus.empirical_transition_matrix = _row_normalize_if_not_zero(_get_empirical_transition_count_matrix(bigram_counts, event_names, W))

	def _get_emission_prob_matrix(self, cut_name):
		hmm=self.hmms[cut_name]
		(W,K)= self.corpus.W, self.K
		mat=np.zeros((W,K))
		for k in range(K):
			mat[:,k]=hmm.obs_distns[k].params['weights']
		return mat 

	def _get_initial_state_distribution(self, cut_name):
		hmm=self.hmms[cut_name]
		return hmm.init_state_distn.pi_0

	def _get_transition_matrix(self, cut_name):
		hmm=self.hmms[cut_name]
		return hmm.trans_distn.trans_matrix

	def _get_latent_state_series(self, cut_name):
		hmm=self.hmms[cut_name]
		return hmm.stateseqs

	def _get_model_parameters(self, store_empirical_transition_matrix=True, store_latent_state_series=True):
		### General Stuff
		if store_empirical_transition_matrix: #Note: it can take a while to construct 
			self._get_empirical_transition_matrix()
		### Cut specific Stuff
		for model_name in self.model_names:
			self.hmms[model_name].transition_matrix=self._get_transition_matrix(model_name)
			self.hmms[model_name].initial_state_distribution=self._get_initial_state_distribution(model_name)
			self.hmms[model_name].emission_prob_matrix=self._get_emission_prob_matrix(model_name)	
			if store_latent_state_series: 	#Note: it can be huge 
				self.hmms[model_name].latent_state_series=self._get_latent_state_series(model_name)

class UnsupervisedResultsPaths:
	def __init__(self, path_to_dumped_data, path_to_fvm, train_idxs_filepath,test_idxs_filepath):
		self.path_to_dumped_data=path_to_dumped_data
		self.path_to_fvm=path_to_fvm 
		self.train_idxs_filepath=train_idxs_filepath
		self.test_idxs_filepath=test_idxs_filepath 	
		self.main_results_dir=os.path.dirname(path_to_dumped_data)
