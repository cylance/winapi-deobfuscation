
# the Corpus class (and FileCorpus class) are abstract base classes
from abc import ABCMeta, abstractmethod
import numpy as np 
import glob, os #for bulding the vocabulary 
import copy #to copy arrays 

from winapi_deobf_experiments.util.io import *
from winapi_deobf_experiments.util.representation import Bunch
from winapi_deobf_experiments.preprocess.filepath_tracker import FilepathTracker
from winapi_deobf_experiments.preprocess.cut import Cut

##########################################################################################################################
###########################################################################################################################
###########################################################################################################################

def make_corpus(configs):
	corpus=	CorpusFactory(configs.corpus_type).corpus_class(configs, configs.seed)
	return corpus 


class CorpusFactory:
	"""
	Determines which derived class (of the Corpus base class)
	to use based on a string. 

	Attributes:
		corpus_class: A non-ABC subtype of the Corpus class below. 

	"""
	def __init__(self, corpus_type):
		self.corpus_class={
			"APICorpus": APICorpus, 
			"APIFlexiCorpus": APIFlexiCorpus,
			"ETSCorpus": ETSCorpus
			}.get(corpus_type)
		if not self.corpus_class:
			raise ValueError('corpus_type for the Corpus Class is unexpected value: %s' % corpus_type)

###########################################################################################################################
###########################################################################################################################
###########################################################################################################################



class Corpus:
	"""

	Takes a set of directories (each with potentially many filepaths, than in turn have potentially
	many time series), with provided string labels, and creates a "corpus", which is a flattened 
	representation of the time series extracted fromd disk. 

	To read in your own distinct datatype for your own distinct problem, you just need to define 
	your own corpus subclass that determines how the data is read in.  In particular, just need to 
	define a custom _read_filepath_corpus_and_get_labels(filepath, label) method. 

	Attributes:
		corpus:  A list of lists of type obs_type.  
			Each corpus[i] represents a single sequence, and corpus[i][j] is an event observed from sequence.
		numeric_corpus: list of np arrays of numbers.  
			Same structure as above, but now the rep is a float for a NumericCorpus and an int for a CategoricalCorpus.  
		labels: List of strings.  
			Labels[i] is the label for the ith sequence. 
		labels_to_numeric_dict:  Dict <string, int>
			Maps a label to an integer identifier. 
		filepath_trackers: Dict <string,filepath_tracker)
			Maps filepaths to information (number of t.s. samples in the filepath, first and last index, filepath label)
		cuts: Cut 
			The cut is a list of  indices of the corpus/numeric_corpus to use or withhold during training. 
			determined via random selection based on labels
		obs_type: String 
			"numeric" for NumericCorpus, "categorical" for CategoricalCorpus 

	Current hierarchy (* = abstract base class):
		Corpus* -> DiscreteCorpus* -> APICorpus -> APIFlexiCorpus
				-> NumericCorpus* -> ETSCorpus
	"""
	
	__metaclass__ = ABCMeta 	 

	def __init__(self, configs, seed=1234):
		"""
		Arguments:
			configs: A Bunch instance.  
				The yaml configs file as a (key, value) dictionary is represented instead as a 
				class instance (so we can call e.g. configs.instance_type instead of configs["instance_type"])
		"""
		self._check_well_formedness_of_configs(configs)
		self.corpus = []
		self.numeric_corpus = None #TD: make type explicit
		self.labels = []
		self.labels_to_numeric_dict = dict()
		self.filepath_trackers= {}
		self._set_corpus_and_labels_and_filepath_trackers(configs.label_tracker, configs.dir_filter) #TD: parallelize this
		self._set_numeric_repr_of_labels() #TD: parallelize this
		self.cut=Cut(configs.label_tracker, self.filepath_trackers, seed)

	def _check_well_formedness_of_configs(self, configs):
		for label, label_info in configs.label_tracker.iteritems():
			if set(label_info.keys())!={"data_dirs", "train_pct"}:
				raise ValueError("The label_tracker in the configs is not well-formed; label %s"\
					"needs to have keys for data and train_pct" %(label))

	def _set_corpus_and_labels_and_filepath_trackers(self, label_tracker, dir_filter):
		"""
		Arguments:
			label_tracker: Dict<string, Dict>>
				Dictionary mapping labels to inner dicts, which maps "data_dir" to a list of strings, 
				interpreted as filepaths, and "train_pct" to a float. 
			dir_filter: String 
				Tells which files to include from the label_tracker's data_dirs. 

		"""
		samples_used_so_far = 0 
		for label, label_info in label_tracker.iteritems():
			dirs_with_label=label_info["data_dirs"]
			for (dir_idx,directory) in enumerate(dirs_with_label):
					for filepath in glob.glob(os.path.join(directory, dir_filter)):
						fp_corpus, fp_labels = self._read_filepath_corpus_and_get_labels(filepath, label)
						self.corpus += fp_corpus
						self.labels += fp_labels
						N_samples_in_filepath = len(fp_labels)
						fp_tracker = FilepathTracker(N_samples_in_filepath=len(fp_labels), 
							filepath_label=label, 
							start_idx_of_time_series_for_filepath=samples_used_so_far,
							stop_idx_of_time_series_for_filepath=samples_used_so_far+N_samples_in_filepath-1)
						self.filepath_trackers[filepath]=fp_tracker
						samples_used_so_far+=N_samples_in_filepath

	@abstractmethod
	def _read_filepath_corpus_and_get_labels(self, filepath, label):
		"""
		Returns the corpus and labels (see class def. of these attributes) just from a particular filepath. 
		"""
		pass


	def _set_numeric_repr_of_labels(self):
		"""
		We want to represent labels for time series using integers rather than just strings. 
		"""
		labels_to_numeric_dict={}
		numeric_labels=[0]*len(self.labels)
		label_set=set(self.labels)
		for word in label_set:
			labels_to_numeric_dict[word] = len(labels_to_numeric_dict)
		self.labels_to_numeric_dict=labels_to_numeric_dict

	@abstractmethod
	def _construct_numeric_corpus(self):
		pass 

	def get_numeric_label(self, idx):
		return self.labels_to_numeric_dict[self.labels[idx]]

class NumericCorpus(Corpus):
	"""
	Additional attributes:
		NONE SO FAR. 

	"""
	__metaclass__ = ABCMeta 

	def __init__(self, configs, seed=1234):
		Corpus.__init__(self, configs, seed)
		self._construct_numeric_corpus()
		self.obs_type="numeric"

	def _construct_numeric_corpus(self):
		self.numeric_corpus=self.corpus 
		#TD: check further that this creates a pointer and not a copy of the data.
		#so far it seems to be creating a pointer as adding additional fields with different names
		#does not increase the size of the pickled file. 


class CategoricalCorpus(Corpus):
	"""
	Additional attributes:
		vocab_dict: Dictionary <obs_type, int>
			Keys are all the tokens in the training set and values are their number of occurences. 
		W: Int. 
			The number of items in vocab_dict (i.e. the size of the "alphabet" = space of possible tokens.)

	"""
	__metaclass__ = ABCMeta 

	def __init__(self, configs, seed=1234):
		Corpus.__init__(self, configs, seed)
		self._construct_vocabulary()
		self._construct_numeric_corpus()
		self.obs_type="categorical"

	def _construct_numeric_corpus(self):
		self.numeric_corpus = map(lambda process: np.array([self.vocab_dict[x] for x in process], dtype=np.int32), self.corpus)

	def _construct_vocabulary(self):
		self.vocab_dict=dict()
		vocab_set=self._construct_vocabulary_as_set()
		for word in vocab_set:
			self.vocab_dict[word] = len(self.vocab_dict)
		self.W=len(self.vocab_dict)

	def _construct_vocabulary_as_set(self):
		vocab_set = set()
		for idx, time_series in enumerate(self.corpus):
			vocab_set=vocab_set.union(set(time_series)) 
		return vocab_set



class APICorpus(CategoricalCorpus):
	"""
	Handles files such as Vadim's API Call Data in the non-flexi case.

	That is, this handles a file whose rows have the following format:
		arg_1, arg_2, ..., arg_n, ApiName,
	where the comma-delimited "items" have the following interpetations:
		the first n args of a row are the correct number of args for that api
		the ApiName is the name of the api function, 
	For example, one row might be:
		0x2400,var_414h,1,0x0,var_20Ch,3,var_418h,FormatMessage

	Create a new time series sample each time the row changes.   
	Last value in the row is the API call (label); omit from time series and replace with "period."
	"""

	def _read_filepath_corpus_and_get_labels(self,filepath, label):
		"""
		Gets the corpus (i.e. collection of time series) and labels for a given filepath in the data directory. 
		"""

		with open(filepath, "r") as f:
			rl = f.readlines()
			filepath_corpus, filepath_labels = [], [];
			for (line_idx, line) in enumerate(rl):
				pieces = line.split(",")
				api_call, api_call_idx=self._find_the_api_call(pieces)
				api_args = pieces[:api_call_idx]
				filepath_corpus.append(api_args)
				filepath_labels.append(api_call)
		return filepath_corpus, filepath_labels 


	def _find_the_api_call(self, pieces):
		for (idx, piece) in enumerate(pieces):
			if piece is '':
				api_call=pieces[idx-1]
				api_call_idx=idx-1
				break
			elif "\r\n" in piece:
				api_call=pieces[idx].strip("\r\n")
				api_call_idx=idx
				break 
		return api_call, api_call_idx 

class APIFlexiCorpus(APICorpus):
	"""
	Handles files such as Vadim's API Call Data in the flexi corpus case:

	That is, it handles a file whose rows have the following format:
		arg1, arg2, ..., argN, ApiName, n_args 
	where the comma-delimited "items" have the following interpetations:
		ApiName is the name of the api function, 
		n_args is the actual number of arguments the known api call takes 
		the first n args of a row are the correct args for that api
		the (n+1)st to (N)th args of the row are other things found on the stack
	For example, one row might be:
		0x2400,var_414h,1,0x0,var_20Ch,3,var_418h,var_20Ah,1,3,arg_Ch,arg_Ch,FormatMessage,7

	Create a new time series sample each time the row changes.   
	Last two values in the row ar are the ApiName and n_args; omit from time series and replace with "period."
	"""

	def _find_the_api_call(self, pieces):
		### TD: This overrides the APICorpus method; would be clearer to have APICorpus be a base class
		# where we pass on _find_the_api_call until reaching one of 2 derived classes, APIFlexiCorpus
		# and APIStaticCorpus
		for (idx, piece) in enumerate(pieces):
			if piece is '':
				api_call=pieces[idx-2]
				api_call_idx=idx-2
				break
			elif "\n" in piece:
				api_call=pieces[idx-1]
				api_call_idx=idx-1
				break 
		return api_call, api_call_idx 

class ETSCorpus(NumericCorpus):
	"""
	Handles files where each row is a time series of floats. 

	ETSCorpus stands for "Entropy Time Series" corpus, which is where
	I used this strategy. 
	"""

	def _read_filepath_corpus_and_get_labels(self, filepath, label):
		"""
		Gets the corpus (i.e. collection of time series) and labels for a given filepath in the data directory. 

		Note: for numeric time series, pyhsmm requires each time series to be an 2-dim np.array 
		with dimension (#obs, 1)

		"""

		with open(filepath, "r") as f:
			rl = f.readlines()
			filepath_corpus, filepath_labels = [], [];
			for (line_idx, line) in enumerate(rl):
				ts_list=[float(x) for x in line.split(",")]
				ts=np.array(ts_list)[:,np.newaxis]
				#label=self._label_maker(filepath)
				filepath_corpus.append(ts)
				filepath_labels.append(label)
		return filepath_corpus, filepath_labels 


