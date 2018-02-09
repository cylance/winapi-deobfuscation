import yaml #for configs 
import os
import warnings 
import pickle 

def write_vector(filepath, vec, delimiter="\n", do_not_overwrite=False):
	"""
	Converts vector elements to strings, and then writes to a text file.
	"""
	if os.path.exists(filepath):
		if do_not_overwrite:
			raise ValueError("Trying to write to filepath %s which already exists" %(filepath))
		else: 
			warnings.warn("Overwriting filepath %s which already exists" %(filepath))
	with open(filepath,'w') as f:
		for (idx, val) in enumerate(vec):
			f.write(str(val)+delimiter)

def read_filepath_of_delimited_ints_as_vector(filepath, delimiter="\n"):
	"""
	Takes filepath containing delimited integers and returns 
	it as a vector 
	"""
	ints_as_strings=read_text_file_as_list_of_strings(filepath)
	int_vector=[int(x.strip(delimiter)) for x in ints_as_strings]
	return int_vector

def read_text_file_as_list_of_strings(filepath):
	with open(filepath, "r") as f: 
		rl=f.readlines()
	return rl 

def load_configs(local_configs_path):
	with open(local_configs_path, 'r') as f:
		configs = yaml.load(f)	
	return configs 
	
def write_rows_to_text(filepath, rows, mode="w",return_delimited=False):
	"""
	rows: a list of strings 
	"""
	if return_delimited:
		with open(filepath, mode) as f:
			f.writelines("%s\n" % r for r in rows)
	else:
		with open(filepath, mode) as f:
			f.writelines("%s" % r for r in rows)

def ensure_dirs(directory):
	"""
	Recursive version of ensure_dir
	"""
	if not os.path.isdir(directory):
		os.makedirs(directory) 

def read_first_line(filepath):
	with open(filepath, 'r') as f:
		first_line = f.readline()
	return first_line 

def file_len(fname):
	with open(fname) as f:
		for i, l in enumerate(f):
			pass
	return i + 1

def save_obj(obj, filepath): #note: save as .pkl 
	with open(filepath, 'wb') as f:
		pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

