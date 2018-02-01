
from abc import ABCMeta, abstractmethod
import numpy as np 
import glob, os #for bulding the vocabulary 
import copy #to copy arrays 

from winapi_deobf_experiments.util.io import *

class Cut:
	"""
	The Cut class is a particular "point of view" on the dataset, and there can be many
	such cuts applied to a dataset.   For instance, we could look just at the good samples, or just the bad, or both.  For each
	cut instance, we will learn our own model parameters; and this will be based on a certain 
	list of train and test indices derived from the configurations which set a desired pct of good  
	and bad data in that "Cut"

	Attributes:
		use_idxs:  List of ints.   Maintains a single set of indices across all filepaths. 
		withheld_idxs: List of ints.   Maintains a single set of indices across all filepaths. 
	""" 
	def __init__(self, cut_name, cut_definition):
		self.cut_name = cut_name 
		self.cut_definition = cut_definition 
		self.use_idxs=[]
		self.withheld_idxs=[]
		self.emission_prob_matrix = None
		self.initial_state_distribution = None 
		self.transition_matrix = None
		self.latent_state_series = None 

	"""
	This is done by desired pct of train and test conditioned on filepath_label (good vs. bad).
	Note that truncated_sample_size_per_filepath, when set, overrides the requested train and test pcts for good/bad. 
	"""
	def _get_use_and_withhold_idxs_for_filepath(self, filepath, filepath_label, N_samples_to_use_in_filepath, seed):
		permuted_row_idxs=np.random.permutation(N_samples_to_use_in_filepath)
		train_pct_for_filepath=[val for (key,val) in self.cut_definition.iteritems() if key==filepath_label][0] #assume there will be a single unique match 
		last_train_idx_in_this_file=int(np.floor(train_pct_for_filepath*N_samples_to_use_in_filepath))
		train_indices = permuted_row_idxs[:last_train_idx_in_this_file]			
		test_indices = permuted_row_idxs[last_train_idx_in_this_file:]		
		return train_indices, test_indices

	def get_cut_idxs(self, filepath_trackers, seed):
		ft=filepath_trackers
		counter = 0		
		for filepath, ft in filepath_trackers.iteritems():
			train_indices_filepath, test_indices_filepath=self._get_use_and_withhold_idxs_for_filepath(filepath, ft.filepath_label, \
				ft.N_samples_in_filepath, seed) 
			self.use_idxs.extend((train_indices_filepath+counter).tolist())
			self.withheld_idxs.extend((test_indices_filepath+counter).tolist())
			counter += (len(train_indices_filepath)+len(test_indices_filepath))

class Cuts:
	def __init__(self):
		self.cuts={}

	def make_cuts(self, configs, filepath_trackers, seed=1234):
		for cut_name, cut_definition in configs.cuts_dictionary.iteritems():
			self.cuts[cut_name] = Cut(cut_name, cut_definition) #initiate the CutClass
			self.cuts[cut_name].get_cut_idxs(filepath_trackers, seed)
