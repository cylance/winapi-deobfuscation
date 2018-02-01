"""
	'Flexi' means now we don't know how many arguments should go with each API call. 

	Notes: 
		1) Code currently works assuming there is only one "file" cut (i.e. set of training/test idxs across
		multiple datasets), and that the models are constructed elsewhere. 
		2) This code operates on directories, not files. 

"""

import os
import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib
matplotlib.use('TkAgg')
#two lines above to avoid error "**RuntimeError**: Python is not installed as a framework."
import matplotlib.pyplot as plt  
matplotlib.rcParams['font.size'] = 8
import pprint 
pp = pprint.PrettyPrinter(indent=3)

from winapi_deobf_experiments.preprocess.stats import form_dict_api_calls_to_n_args
from winapi_deobf_experiments.preprocess.corpus import  make_corpus
from winapi_deobf_experiments.supervised.dataset import get_dataset_for_classification, \
	ensure_indexing_alignment_of_dataset_and_corpus_for_expt_2
from winapi_deobf_experiments.supervised.dataset_bag_of_words import get_bag_of_words_dataset_for_classification
from winapi_deobf_experiments.supervised.evaluate import get_top_n_model_accuracies, get_top_n_baserate_accuracies, \
	get_lr_predicted_labels 
from winapi_deobf_experiments.unsupervised.train import fit_single_hmm_to_all_api_functions
from winapi_deobf_experiments.unsupervised.results import UnsupervisedResults
from winapi_deobf_experiments.unsupervised.wrapper import create_unsupervised_data
from winapi_deobf_experiments.util.draw import plot_latent_trajectory, plot_accuracies, create_confusion_matrix_plot
from winapi_deobf_experiments.util.representation import construct_configs

def run_experiment_1(parsed_api_data_dir="data/expt1/paper/", train_pct=.8, K=10, \
	path_to_configs="src/winapi_deobf_experiments/configs.yaml"):
	"""
	Arguments:
		parsed_api_data_dir: String 
			The local directory that contains the parsed api function calls to be processed,
			where the data format is that produced by the data_collection folder of this repo.
			
			This experiment takes, as input, a directory of files whose rows have the following format:
				arg_1, arg_2, ..., arg_n, ApiName,
			where the comma-delimited "items" have the following interpetations:
				the first n args of a row are the correct number of args for that api
				the ApiName is the name of the api function, 
			For example, one row might be:
				0x2400,var_414h,1,0x0,var_20Ch,3,var_418h,FormatMessage
		train_pct: Float. 
			Percentage of the data to use for training the model (both the hmm and the classifier model).
		K: Int. 	
			Number of latent states for the HMM model. 
		path_to_configs: String
			Path the the .yaml formatted configurations file.
	"""
	print("\n\n .... Now running code for experiment 1  ...\n\n")
	
	### ### SETUP
	configs = construct_configs(path_to_configs, train_pct=train_pct, run_summary="Expt_1", dir_filter="*csv")

	### ### GENERATE CORPUS FROM FILEPATHS
	corpus = make_corpus(configs, parsed_api_data_dir, corpus_type="APICorpus")

	### ### GET UNSUPERVISED FEATURES
	hmms=fit_single_hmm_to_all_api_functions(corpus, K, configs)
	results = UnsupervisedResults(configs, hmms, corpus, K)  
	unsupervised_results_paths  = results.redump_data(streaming=True, store_empirical_transition_matrix=False, \
		store_latent_state_series=False, one_time_series_per_row=True, use_likelihood_feats=True, \
		use_latent_state_marginals=True, add_numeric_labels=True, dict_model_names_to_n_args=None,
		train_test_or_all="all", default_unlikely_value=configs.default_unlikely_value) 
	results_dir=unsupervised_results_paths.main_results_dir 

	### ### SUPERVISED CLASSIFICATION (Train multi-class logistic regression classifier and get predictions)
	### ### 1) USING SEQUENTIAL (HMM) VECTORIZATION
	dataset = get_dataset_for_classification(unsupervised_results_paths, K, dict_api_calls_to_n_args=None, \
		use_separate_hmms_for_each_api_function=False)
	yhat_test = get_lr_predicted_labels(dataset.features_train, dataset.features_test, dataset.labels_train)
	acc_sequential=np.mean(yhat_test==dataset.labels_test) #73.1% in paper
	print("\n\n Expt1: Predictive Accuracy using Sequential Vectorization is %.04f" %(acc_sequential))
	create_confusion_matrix_plot(ys=dataset.labels_test, ys_predicted=yhat_test, 
		path_to_save_fig=os.path.join(results_dir,"confusion_with_hmm_vectorization.pdf"))
	
	### ### 2) USING BAG OF WORDS VECTORIZATION 
	dataset = get_bag_of_words_dataset_for_classification(corpus, K, train_pct=train_pct)
	yhat_test = get_lr_predicted_labels(dataset.features_train, dataset.features_test, dataset.labels_train)
	acc_bag_of_words=np.mean(yhat_test==dataset.labels_test) # 51.56% in paper.
	print("\n\n Expt1: Predictive Accuracy using Bag of Words Vectorization is %.04f" %(acc_bag_of_words))
	create_confusion_matrix_plot(ys=dataset.labels_test, ys_predicted=yhat_test, 
		path_to_save_fig=os.path.join(results_dir,"confusion_with_bag_of_words_vectorization.pdf"))


def run_experiment_2(parsed_api_data_dir="data/expt2/paper/", train_pct=.8, K=10, 
	path_to_configs="src/winapi_deobf_experiments/configs.yaml"):
	"""
	Arguments:
		parsed_api_data_dir: String 
			The local directory that contains the parsed api function calls to be processed,
			where the data format is that produced by the data_collection folder of this repo.

			This experiment takes, as input, a directory of files whose rows have the following format:
				arg1, arg2, ..., argN, ApiName, n_args 
			where the comma-delimited "items" have the following interpetations:
				ApiName is the name of the api function, 
				n_args is the actual number of arguments the known api call takes 
				the first n args of a row are the correct args for that api
				the (n+1)st to (N)th args of the row are other things found on the stack
			For example, one row might be:
				0x2400,var_414h,1,0x0,var_20Ch,3,var_418h,var_20Ah,1,3,arg_Ch,arg_Ch,FormatMessage,7
		train_pct: Float. 
			Percentage of the data to use for training the model (both the hmm and the classifier model).
		K: Int. 	
			Number of latent states for the HMM model. 
		path_to_configs: String
			Path the the .yaml formatted configurations file.
	"""


	print("\n\n .... Now running code for experiment 2  ...\n\n")

	### ### SETUP 
	configs = construct_configs(path_to_configs, train_pct=train_pct, run_summary="Expt_2", dir_filter="*txt")

	### ### GENERATE CORPUS FROM FILEPATHS
	corpus= make_corpus(configs, parsed_api_data_dir, corpus_type="APIFlexiCorpus")
	dict_api_calls_to_n_args=form_dict_api_calls_to_n_args(parsed_api_data_dir, dir_filter=configs.dir_filter)

	### ### GET UNSUPERVISED FEATURES (Create it, or can reconstruct if already done.)
	unsupervised_results_filepaths=create_unsupervised_data(K, corpus, configs, \
		use_separate_hmms_for_each_api_function=True, dict_api_calls_to_n_args=dict_api_calls_to_n_args)
	results_dir=unsupervised_results_filepaths.main_results_dir 

	### ### SUPERVISED CLASSIFICATION (Train multi-class logistic regression classifier and get predictions)
	dataset = get_dataset_for_classification(unsupervised_results_filepaths, K, \
		 dict_api_calls_to_n_args, use_separate_hmms_for_each_api_function=True)
	ensure_indexing_alignment_of_dataset_and_corpus_for_expt_2(dataset, corpus)
	logistic=LogisticRegression(multi_class="multinomial", solver="lbfgs")
	logistic.fit(dataset.features_train, dataset.labels_train)
	top_n_model_accs=get_top_n_model_accuracies(logistic, dataset.features_test, dataset.labels_test, n_max=5)
	top_n_baserate_accs=get_top_n_baserate_accuracies(dataset.labels_train, dataset.labels_test, n_max=5)

	### ### PLOTS
	plot_accuracies(top_n_model_accs, top_n_baserate_accs, 
			method1_name="Generic deobfuscator", method2_name="Baserate",
			title="Predictive performance of generic API deobfuscator", ylim=[0.0,1.0], 
			path_to_save_fig=os.path.join(results_dir,"expt_2_accuracy.pdf"))
	plot_latent_trajectory(path_to_save_fig=os.path.join(results_dir,"latent_trajectory.pdf"))


if __name__ == "__main__":
	run_experiment_1()
	run_experiment_2()
