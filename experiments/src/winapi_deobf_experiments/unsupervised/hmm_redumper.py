
import numpy as np 
import copy #to copy arrays 
import os 
import pickle 
import time, datetime #to get date, also to time things with @timeit decorator
import csv #for saving modelexploration results 

from winapi_deobf_experiments.unsupervised.hmm_feature_maker import HMMFeatureMaker 
from winapi_deobf_experiments.util.io import write_vector, read_text_file_as_list_of_strings, \
	write_rows_to_text, ensure_dirs 
from winapi_deobf_experiments.util.math_funs import generate_feature_idx_bound_tuples_per_group, \
	convert_np_array_into_comma_separated_string

class HMM_Redumper:
	"""
	Used to re-dump data with HMM features 

	Attributes:
		results_super_dir: String 
			File directory where results are dumped. 
		run_summary: String 
			Descriptive name used to create subfolders where results are dumped.
		corpus: Corpus 
			Contains all the time series vectors and associated info. 
		hmms: Dict<string, hmm>
			Maps a model descriptor to an instance of pyhsmm's hmm class.
		K: Int
			Number of latent states.
		model_names: List of strings 
			Gives the names for the different models trained.  
		results_dir: String 
			Directory where results will be saved.  Combines the "super_dir" from the configs 
			with a descriptive subdirectory that depends upon properties of the corpus 
			(e.g. number of time series) and training session (e.g. date performed)
	"""			


	def __init__(self, results_super_dir, run_summary, hmms, corpus, K, model_names=None,
		make_model_subdirs=False):
		# Having another "corpus" in here would carry high storage costs. 
		# But luckily, Python seems to be making pointers, not copies.  
		# (Changing unsupervised_results.corpus.blah also changes corpus.blah)
		#TD: generalize this for single time series case 
		#TD:  Initialization currently identical for Results and EDA classes; smooth this up. 
		self.hmms = hmms
		self.corpus = corpus
		self.K=K 
		self.model_names = self._get_model_names(model_names)
		self.results_dir=self._get_and_ensure_results_dir(results_super_dir, run_summary)

	def _get_and_ensure_results_dir(self, results_super_dir, run_summary):
		description = self._get_description(run_summary)
		results_dir=os.path.join(results_super_dir, description)
		ensure_dirs(results_dir)
		return results_dir 

	def _get_description(self, run_summary):
		""" 
			Note this function must be run (and so its class must be initalized) --at least-- after constructing the corpus and vocab, b/c
			it uses that information.    It could potentially also use information from the hmm.
		"""
		description=""
		description=description+run_summary+"_"
		description=description+"K"+"="+str(self.K)+"_"
		description=description+"T"+"="+str(len(self.corpus.corpus))+"_" #num documents in corpus
		description=description+"Date"+"="+time.strftime("%m-%d-%Y")
		return description

	def _get_model_names(self, model_names):
		if model_names is None:
			model_names = self.hmms.keys()
		else:
			model_names = model_names 
		return model_names 	

	def redump_data(self, streaming=True, store_latent_state_series=False, one_time_series_per_row=True, \
			feature_types={"log_likelihoods", "log_state_marginals"}, add_numeric_labels=True, 
			dict_labels_to_fixed_ts_length=None, train_test_or_all="all", default_unlikely_value=-200.0):
		"""
		Dump to results folder at self.results_dir (which has descriptive folder name with number of 
			time series analyzed, date of model training, etc.)

		Within that folder is:
			(a) FVM 
			(b) Set of folders, one for each of the original data folders, which contains:
				(b1) train and test idxs 
				(b2) redumped data with predictive features appended. 

		We append new features to the original data and dump the transformed data elsewhere. 
		The dumps are organized in the same way as the original raw data. 

		Parameters:
			streaming:  If True, the features (latent state marginals and event/sequence likelihoods)
				are obtained in a streaming way, i.e. based on partial sequences (up to time t) rather
				than the full sequences. 

		"""
		if type(feature_types) is not set:
			raise ValueError("Attribute feature_types must be a set, but it has type %s" %(type(feature_types)))

		#one HMMFeatureMaker instance for each cut. 
		HMMFeatureMakerArray=[HMMFeatureMaker(self.hmms[model_name], model_name, streaming, feature_types=feature_types) \
			for model_name in self.model_names]

		#write_corpus level stuff 
		path_to_fvm = self._make_fvm(HMMFeatureMakerArray, add_numeric_labels)
		path_to_train_idxs, path_to_test_idxs = self._write_train_and_test_idxs()
		paths_to_redumped_data=[]

		#loop over files starts here  
		for (path_to_orig_data, filepath_tracker) in self.corpus.filepath_trackers.iteritems():
			#path_to_orig_data, filepath_tracker = self.corpus.filepath_trackers.keys()[0], self.corpus.filepath_trackers.values()[0]
			print("Now obtaining feature for file %s" % (path_to_orig_data))
			feature_matrix=self._get_prediction_features_for_this_filepath(path_to_orig_data, HMMFeatureMakerArray, \
				streaming, one_time_series_per_row, feature_types,
				add_numeric_labels, dict_labels_to_fixed_ts_length, train_test_or_all, default_unlikely_value)
			path_to_redumped_data = self._redump_data_for_one_filepath(path_to_orig_data, feature_matrix)
			paths_to_redumped_data.append(path_to_redumped_data)

		return HMM_Redumper_Paths(paths_to_redumped_data, path_to_fvm, path_to_train_idxs,  path_to_test_idxs)

	def _make_fvm(self, HMMFeatureMakerArray, add_numeric_labels):
		"""
		FVM for redumped dataset that describes ONLY the new features (appended at end)
		Assumes that the HMMFeatureMakerArray class generates features in order [latent state marginals, 
		likelihood features]
		"""
		path_to_fvm=os.path.join(self.results_dir, "fvm.txt")
		#reads in first line of original dataset 
		if os.path.exists(path_to_fvm):
			os.remove(path_to_fvm)
		for idx, fma in enumerate(HMMFeatureMakerArray):
			 write_rows_to_text(path_to_fvm, fma.feature_names, mode="a+", return_delimited=True)
		if add_numeric_labels:
			write_rows_to_text(path_to_fvm, ["numeric_label"], mode="a+", return_delimited=True)
		return path_to_fvm 	

	def _write_train_and_test_idxs(self):
		train_idxs=self.corpus.cut.use_idxs 
		test_idxs=self.corpus.cut.withheld_idxs
		path_to_train_idxs=os.path.join(self.results_dir,"train_idxs")
		path_to_test_idxs=os.path.join(self.results_dir,"test_idxs")
		write_vector(path_to_train_idxs, train_idxs)
		write_vector(path_to_test_idxs, test_idxs)
		return path_to_train_idxs, path_to_test_idxs 

	def _redump_data_for_one_filepath(self, path_to_orig_data, feature_matrix):
		#TD: a bit convoluted; can i just state the desired path and then ensure the dir exists. 
		data_dir_for_redump = self._state_and_ensure_data_redump_dir(path_to_orig_data)
		path_to_redumped_data= os.path.join(data_dir_for_redump, os.path.basename(path_to_orig_data))
		print("Now dumping dataset with new features appended to %s" % (path_to_redumped_data))
		rl = read_text_file_as_list_of_strings(path_to_orig_data)
		with open(path_to_redumped_data,'w') as f:
			for (row_idx,row) in enumerate(feature_matrix):
				old_line=rl[row_idx]
				new_line=old_line.rstrip()+","+convert_np_array_into_comma_separated_string(feature_matrix[row_idx,:])+"\n" #removes new line adds in data "2"
				f.write(new_line)
		return path_to_redumped_data

	def _state_and_ensure_data_redump_dir(self, filepath):
		new_dir=os.path.join(self.results_dir, os.path.basename(os.path.dirname(filepath))+"_new_dump")	
		ensure_dirs(new_dir) #recursive mkdir 
		return new_dir 

	def _get_prediction_features_for_this_filepath(self, filepath, HMMFeatureMakerArray, streaming=False,
			one_time_series_per_row=True, feature_types={"log_likelihoods", "log_state_marginals"}, add_numeric_labels=True,
			dict_labels_to_fixed_ts_length=None, train_test_or_all="all", default_unlikely_value=-200.0):
		"""
		API call dataset has one time series per line (and T.S. have diff't lengths across lines)
		We'll just return the features from the last observation then.   We can use in particular
		the marginal probs across latent states (and maybe the streaming_mean_log_likelihood_of_events_in_partial_seq)
		See the FVM for the feature names. 

		Attributes:
			streaming:  Bool.  
				If true, the HMM is fit in a streaming way (we only use forward algorithm, 
				not forward-backwards; i.e. we don't use any sequence tokens which are past the current value 
				in determining latent state distributions, obs likelihoods, etc.). 
			one_time_series_per_row: Bool.  
				If true, the original and dumped data have one sequence 
				per row (a la API call data) rather than one event/observation/token per row.
			feature_types: Set of strings.
				Should be "log_likelihoods", "log_state_marginals" or Both. 
			add_numeric_labels: Bool. 
			dict_labels_to_fixed_ts_length: Dict<string,int> or None. 
			 	If none, will use whole sample.  If provided, will truncate
				samples during test phase to the appropriate number of args for that model. 
			train_test_or_all: String. 
			 	If "train" we generate features for the train indices for the cut, if "test" we generate for the 
				test indices, if "all" we use all. 
		"""

		if not streaming or not one_time_series_per_row:
			raise NotImplementedError("Need to implement the Non-streaming version of this,"\
					 	"i.e fit using both forward AND backward algorithm")
		F=self.corpus.filepath_trackers[filepath].start_idx_of_time_series_for_filepath
		L=self.corpus.filepath_trackers[filepath].stop_idx_of_time_series_for_filepath
		feature_matrix, feature_idx_bounds_array= self._get_empty_feature_matrix_and_idx_ranges_for_models(\
		 	filepath, feature_types, F, L, add_numeric_labels, one_time_series_per_row, n_models=len(HMMFeatureMakerArray))
		n_events_so_far=0
		prohibited_idxs=self._get_prohibited_indices(train_test_or_all)
		#for each time series in filepath, scroll through all the hmm models and get predictions. 
		#add in the numeric label if necesaary. 
		for ts_idx in range(F,L+1):
			print("Now generating new features for time series %d of %d" %(ts_idx, L))
			sample=self.corpus.numeric_corpus[ts_idx]
			T=len(sample)
			for (model_idx, hmm_feat_maker) in enumerate(HMMFeatureMakerArray):
				log_state_marginals, log_likelihoods=hmm_feat_maker.get_log_state_marginals_and_log_likelihoods_for_sample(sample, \
					dict_labels_to_fixed_ts_length, default_unlikely_value)
				new_features_for_ts=self._get_new_features_based_on_feature_type(log_state_marginals, log_likelihoods, feature_types)
				new_features_for_ts_reduced=self._only_keep_features_relevant_to_last_observation(new_features_for_ts)
				new_features_for_ts_reduced_redacted=self._blank_out_features_if_the_time_series_has_a_prohibited_idx(\
					new_features_for_ts_reduced, ts_idx, prohibited_idxs)
				feature_matrix[ts_idx,feature_idx_bounds_array[model_idx][0]:feature_idx_bounds_array[model_idx][1]]=new_features_for_ts_reduced_redacted #TD add in the right feature indices.  
				n_events_so_far+=T
			if add_numeric_labels:
				feature_matrix=self._update_row_of_feature_matrix_to_have_numeric_label_appended(feature_matrix, \
					ts_idx, feature_idx_bounds_array)
		return feature_matrix 

	def _get_empty_feature_matrix_and_idx_ranges_for_models(self, filepath, feature_types, F, L, add_numeric_labels, \
 		one_time_series_per_row, n_models):
 		"""
 		Arguments:
 			filepath: String 
 			feature_types: Set of strings.   
 				Should contain either "log_likelihoods", "log_state_marginals", or both.
 				Otherwise, an error will be raised. 
 			F: Int 
 				First time series index for this filepath; i.e. match up the filepath's information with the indexing 
 				strategy for the overall corpus where we store time series e.g. in corpus.numeric_corpus. 
 			L: Int
 				Last time series index for this filepath; i.e. match up the filepath's information with the indexing 
 				strategy for the overall corpus where we store time series e.g. in corpus.numeric_corpus. 
 			add_numeric_labels: Bool 
 			one_time_series_per_row: Bool 
 			n_models: Int  
 				Number of models being used overall in this training phase. 
 		"""
 		#TD: make the below not CamelCase
 		nFeatsPerModel=self._get_n_features_per_model(feature_types)
		nFeatsForNumericLabel=1 if add_numeric_labels else 0 
		#TD: doesn't seem to be used anywhere 
		#num_events_in_filepath=sum(map(lambda x: len(x), self.corpus.numeric_corpus[F:(L+1)]))
		# We're adding features in a hcatted way, one set of features (with nFeats) for each Model, so 
		# this gives us the 1st and last index of features for each of the nModels models 
		feature_idx_bounds_array=self._get_feature_idx_bounds_array(nFeatsPerModel, n_models, add_numeric_labels)
		if one_time_series_per_row:
			feature_matrix=np.zeros((L-F+1,nFeatsPerModel*n_models+nFeatsForNumericLabel))
		else:
			raise NotImplementedError 
		return feature_matrix, feature_idx_bounds_array

	def _update_row_of_feature_matrix_to_have_numeric_label_appended(self, feature_matrix, ts_idx, feature_idx_bounds_array):
		numeric_label= self.corpus.get_numeric_label(ts_idx)
		#TD: don't really need to recompute the below every time
		numeric_label_feature_idx=feature_idx_bounds_array[-1][0]
		feature_matrix[ts_idx,numeric_label_feature_idx]=numeric_label 
		return feature_matrix  

	def _blank_out_features_if_the_time_series_has_a_prohibited_idx(self, features, ts_idx, prohibited_idxs):
		if ts_idx in prohibited_idxs:
			features*=np.nan 
		return features 

	def _only_keep_features_relevant_to_last_observation(self, new_features):
		"""
		New features have T rows, where T is the # of obs in a t.s.  We just extract the last row; this way we can have the same number 
		of features across time series, even if the time series have different lenghts 
		"""
		return new_features[-1,:]

	def _get_prohibited_indices(self, train_test_or_all):
		if train_test_or_all=="train":
			prohibited_indices=self.corpus.cut.withheld_idxs
		elif train_test_or_all=="test":
			prohibited_indices=self.corpus.cut.use_idxs
		elif train_test_or_all=="all":
			prohibited_indices=[]
		return prohibited_indices 



	def _get_new_features_based_on_feature_type(self, log_state_marginals, log_likelihoods, feature_types):
		if feature_types=={"log_likelihoods"}:
			new_features=log_likelihoods
		elif feature_types=={"log_likelihoods", "log_state_marginals"}:
			new_features=np.hstack((log_state_marginals, log_likelihoods))
		elif feature_types=={"log_state_marginals"}:
			new_features=log_state_marginals
		else:
			raise ValueError("You provided feature_types %s which I do not understand" %(feature_types))
		return new_features 

 	def _get_feature_idx_bounds_array(self, nFeats, nModels, add_numeric_labels):
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

	def _get_n_features_per_model(self, feature_types):
		nFeats=0
		if "log_likelihoods" in feature_types: 
			nFeats+=3
		if "log_state_marginals" in feature_types:
			nFeats+=self.K 
		if nFeats==0:
			raise ValueError("Can't find any features with feature_types set to %s" %(feature_types))
		return nFeats 

	# def _get_empirical_transition_matrix(self):
	# 	W = self.corpus.W
	# 	event_names = [get_key_given_value(self.corpus.vocab_dict,i) for i in range(W)]
	# 	bigram_counts = _get_bigram_frequencies(self.corpus.corpus)
	# 	empirical_transition_matrix = _row_normalize_if_not_zero(_get_empirical_transition_count_matrix(bigram_counts, event_names, W))
	# 	return empirical_transition_matrix

class HMM_Redumper_Paths:
	"""
	Just a nice struct for storing paths related to a data redump. 
	"""
	def __init__(self, paths_to_redumped_data, path_to_fvm, path_to_train_idxs,  path_to_test_idxs):
		self.paths_to_redumped_data=paths_to_redumped_data
		self.path_to_fvm=path_to_fvm 
		self.path_to_train_idxs=path_to_train_idxs
		self.path_to_test_idxs=path_to_test_idxs	
		self.main_results_dir=os.path.dirname(path_to_train_idxs)
