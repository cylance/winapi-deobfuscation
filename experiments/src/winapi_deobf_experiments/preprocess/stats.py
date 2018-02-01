from __future__ import division
import os, sys, glob 

def form_dict_api_calls_to_n_args(data_dir, dir_filter):
	d={}
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