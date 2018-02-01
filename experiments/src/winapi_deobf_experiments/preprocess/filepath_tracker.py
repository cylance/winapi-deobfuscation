
from abc import ABCMeta, abstractmethod
import numpy as np 
import glob, os 
import copy

from winapi_deobf_experiments.util.io import *

class FilepathTracker:
	"""
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

# class APIFilepathTrackers(FilepathTrackers):


# 	def __init__(self, dir_filter, local_data_labels_and_dirs):
# 		FilepathTrackers.__init__(self, dir_filter, local_data_labels_and_dirs)
# 		self.dicts_api_calls_to_n_args=

# 	def form_dict_api_calls_to_n_args(self, data_dir, dir_filter):
# 		d={}
# 		for data_filepath in glob.glob(os.path.join(data_dir, dir_filter)):
# 			with open(data_filepath, "r") as f:
# 				rl = f.readlines()
# 				for (line_idx, line) in enumerate(rl):
# 					pieces = line.split(",")
# 					n_args=int(pieces[-1].strip("\n"))
# 					api_call=pieces[-2]
# 					if api_call not in d.keys():
# 						d[api_call]=n_args
# 					else:
# 						assert d[api_call]==n_args
# 		self.dict_api_calls_to_n_args=d

