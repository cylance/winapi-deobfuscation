
from abc import ABCMeta, abstractmethod
import numpy as np 
import glob, os #for bulding the vocabulary 
import copy #to copy arrays 
import math 
import numpy as np 

from winapi_deobf_experiments.util.io import *

class Cut:
	"""
	A cut is a (corpus-level) list of indices to use and to withhold during training, based on the 
	(label-based) training percentages provided in the cut_definition. 

	Attributes:
		use_idxs:  List of ints.   
			Maintains a single set of indices across all filepaths. 
		withheld_idxs: List of ints.   
			Maintains a single set of indices across all filepaths. 
	""" 
	def __init__(self, label_tracker, filepath_trackers, seed):
		"""
		Arguments:
			label_tracker: Dict<string, Dict>>
				Dictionary mapping labels to inner dicts, which maps "data" to a list of filepaths 
				and "train_pct" to a float. 
			filepath_trackers: Dict<string, FilepathTracker>
				Dictionary mapping filepaths to filepath tracker instances. 

		"""
		use_idxs=[]
		withheld_idxs=[]
		for filepath, ft in filepath_trackers.iteritems():
			train_indices_filepath, test_indices_filepath=self._get_use_and_withhold_idxs_for_filepath( \
				ft.N_samples_in_filepath, ft.start_idx_of_time_series_for_filepath,\
				train_pct_for_filepath=label_tracker[ft.filepath_label]["train_pct"], seed=seed) 
				#train_pct_for_filepath entry assumes there will be a single unique match 
			use_idxs.extend(train_indices_filepath.tolist())
			withheld_idxs.extend(test_indices_filepath.tolist())
		self.use_idxs=use_idxs
		self.withheld_idxs=withheld_idxs

	def _get_use_and_withhold_idxs_for_filepath(self, N_samples_in_filepath, 
			start_idx_for_filepath,  train_pct_for_filepath, seed):
		"""
		
		Given characteristics for a filepath (# samples, training pct, and starting idx relative to the corpus),
		uses random selection to return train and test (corpus-based) indices.

		Arguments:
			N_samples_in_filepath: Int 
			start_idx_for_filepath: Int 
			train_pct_for_filepath: Float. 
			seed: Int
				Sets random seed. 
		"""
		np.random.seed(seed=seed)
		permuted_row_idxs=np.random.permutation(N_samples_in_filepath)
		samples_to_use_in_filepath_as_float=train_pct_for_filepath*N_samples_in_filepath
		decimal_part, int_part = math.modf(samples_to_use_in_filepath_as_float)
		last_train_idx_in_this_file=int(int_part)+np.random.binomial(n=1, p=decimal_part) #important if only one sample per file 
		train_indices = permuted_row_idxs[:last_train_idx_in_this_file]+start_idx_for_filepath		
		test_indices = permuted_row_idxs[last_train_idx_in_this_file:]+start_idx_for_filepath		
		return train_indices, test_indices
