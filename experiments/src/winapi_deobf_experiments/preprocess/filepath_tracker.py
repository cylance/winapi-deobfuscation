
from abc import ABCMeta, abstractmethod
import numpy as np 
import glob, os 
import copy

from winapi_deobf_experiments.util.io import *

class FilepathTracker:
	"""
	Provides information about a filepath.  Specifically what it contributes to the corpus as a whole. 

	Attributes:
		filepath_label: String 
		start_idx_of_time_series_for_filepath: Int  
		stop_idx_of_time_series_for_filepath: Int 
		N_samples_per_filepath: Int 
	"""

	def __init__(self, N_samples_in_filepath, filepath_label, start_idx_of_time_series_for_filepath, \
			stop_idx_of_time_series_for_filepath):
		self.filepath_label = filepath_label 
		self.start_idx_of_time_series_for_filepath = start_idx_of_time_series_for_filepath
		self.stop_idx_of_time_series_for_filepath = stop_idx_of_time_series_for_filepath
		self.N_samples_in_filepath = N_samples_in_filepath
