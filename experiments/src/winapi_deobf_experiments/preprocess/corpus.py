
# the Corpus class (and FileCorpus class) are abstract base classes
from abc import ABCMeta, abstractmethod
import numpy as np 
import glob, os #for bulding the vocabulary 
import copy #to copy arrays 

from winapi_deobf_experiments.util.io import *
from winapi_deobf_experiments.util.representation import Bunch
from winapi_deobf_experiments.preprocess.filepath_tracker import FilepathTracker
from winapi_deobf_experiments.preprocess.cuts import Cuts

##########################################################################################################################
###########################################################################################################################
###########################################################################################################################

def make_corpus(configs, local_data_dir, corpus_type):

	### ### Load Time Series  
	local_data_labels_and_dirs={"no_valence": [local_data_dir]}
	corpus=	CorpusFactory(corpus_type).corpus_class(configs, local_data_labels_and_dirs)
	corpus.make_corpus_and_labels_and_filepath_trackers() #TD: parallelize this
	corpus.make_cuts()

	### ### Load Vocabulary 
	corpus.construct_vocabulary()
	corpus.construct_numeric_corpus()
	return corpus 


class CorpusFactory:
	"""
	Automatically determines which derived class (e.g ProcessCorpus, FileByPIDsDeInterleaved, etc.)
	to use based on a field in the configurations yaml file.  
	"""
	def __init__(self, corpus_type):
		self.corpus_class=None
		self.get_corpus_class(corpus_type)

	def get_corpus_class(self, corpus_type):
		self.corpus_class={
			"APICorpus": APICorpus, 
			"APIFlexiCorpus": APIFlexiCorpus
			}.get(corpus_type)
		if not self.corpus_class:
			raise ValueError('corpus_type for the Corpus Class is unexpected value: %s' % corpus_type)
		return self.corpus_class


###########################################################################################################################
###########################################################################################################################
###########################################################################################################################



class Corpus:
	"""
		An instance of this class contains original corpus and post-model summary 
		information to be exported.

		TD: fill this in

		Attributes:
			configs: A Bunch instance.  The yaml configs file as a (key, value) dictionary is represented instead as a 
				class instance (so we can call e.g. configs.instance_type instead of configs["instance_type"])
			corpus:  A list of lists of strings.  Each corpus[i] represents a single sequence, and corpus[i][j] is an event observed from sequence. 
			numeric_corpus: list of np arrays of integers.  Each numeric_corpus[i] represents a single sequence, and numeric_corpus[i][j] is the 
					numeric indicator (e.g. 24) of the label for that sequence. 
			labels: List of strings.  labels[i] is the label for the ith sequence. 
			vocab_dict: Dictionary. Keys are all the tokens in the training set and values are their number of occurences. 
			W: Int. The number of items in vocab_dict (i.e. the size of the "alphabet" = space of possible tokens.)
			filepaths: List of strings.  Gives filepaths to data. 
	"""
	
	__metaclass__ = ABCMeta 	 

	def __init__(self, configs, local_data_labels_and_dirs):
		self.configs=configs
		self.local_data_labels_and_dirs=local_data_labels_and_dirs
		self.filepath_trackers= {}
		self.local_results_dir = None
		self.cuts = Cuts();
		self.corpus = []
		self.labels = []
		self.numeric_corpus = None #TD: make type explicit
		self.numeric_labels = None 
		self.non_detonation_idxs =[];
		self.W = None 
		#self.vocab_set = set()
		self.vocab_dict = dict()
		self.label_dict = dict()
		self.empirical_transition_matrix = None

	def make_corpus_and_labels_and_filepath_trackers(self):
		self._make_corpus_and_labels_and_filepath_trackers()
		self._vectorize_labels()

	def _make_corpus_and_labels_and_filepath_trackers(self):
		samples_used_so_far = 0 
		for local_data_dir_label, local_data_dirs in self.local_data_labels_and_dirs.iteritems():
			for (dir_idx,directory) in enumerate(local_data_dirs):
					for filepath in glob.glob(os.path.join(directory, self.configs.dir_filter)):
						this_corpus, these_labels = self._read_filepath_corpus_and_get_labels(filepath, \
							local_data_dir_label)
						self.corpus += this_corpus
						self.labels += these_labels
						N_samples_in_filepath = len(these_labels)
						filepath_tracker = FilepathTracker(N_samples_in_filepath=len(these_labels), 
							filepath_label=local_data_dir_label, 
							start_idx_of_time_series_for_filepath=samples_used_so_far,
							stop_idx_of_time_series_for_filepath=samples_used_so_far+N_samples_in_filepath-1)
						self.filepath_trackers[filepath]=filepath_tracker
						samples_used_so_far+=N_samples_in_filepath

	def _vectorize_labels(self):
		label_dict={}
		numeric_labels=[0]*len(self.labels)
		label_set=set(self.labels)
		for word in label_set:
			label_dict[word] = len(label_dict)
		for (idx,label) in enumerate(self.labels):
			numeric_labels[idx]=label_dict[label]
		self.numeric_labels=numeric_labels
		self.label_dict=label_dict

	@abstractmethod
	def _read_filepath_corpus_and_get_labels(self, filepath, label):
		pass
	
	def make_cuts(self, seed=1234):
		self.cuts.make_cuts(self.configs, self.filepath_trackers, seed)

	#### get vocabulary
	def construct_vocabulary(self):
		vocab_set=self._construct_vocabulary_as_set()
		for word in vocab_set:
			self.vocab_dict[word] = len(self.vocab_dict)
		self.W=len(self.vocab_dict)
		return self.vocab_dict

	def _construct_vocabulary_as_set(self):
		vocab_set = set()
		for idx, single_process_stream in enumerate(self.corpus):
			vocab_set=vocab_set.union(set(single_process_stream)) 
		return vocab_set

	def construct_numeric_corpus(self):
		self.numeric_corpus = map(lambda process: np.array([self.vocab_dict[x] for x in process], dtype=np.int32), self.corpus)


class APICorpus(Corpus):
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

		#TD: fix warning below
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


