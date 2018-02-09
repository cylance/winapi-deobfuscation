from __future__ import division
import os, sys, glob 

def form_dict_api_calls_to_n_args(label_tracker, dir_filter):
	"""
	Arguments:
		label_tracker: Dict<string, Dict>>
			Dictionary mapping labels to inner dicts, which maps "data_dirs" to a list of strings, 
			interpreted as filepaths, and "train_pct" to a float. 
		dir_filter: String 
			Input data is filtered based on this string. 
	Returns: Dict<string, int>
		Dictionary maping the name of an api function to the integer 
		number of arguments it takes. 
	"""
	d={}
	for label, label_info in label_tracker.iteritems():
		for data_dir in label_info["data_dirs"]:
			for data_filepath in glob.glob(os.path.join(data_dir, dir_filter)):
				with open(data_filepath, "r") as f:
					rl = f.readlines()
					for (line_idx, line) in enumerate(rl):
						pieces = line.split(",")
						n_args=int(pieces[-1].strip("\n"))
						api_call=pieces[-2]
						if api_call not in d.keys():
							d[api_call]=n_args
						else:
							assert d[api_call]==n_args
					return d