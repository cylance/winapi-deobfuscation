
import os
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
#two lines above to avoid error "**RuntimeError**: Python is not installed as a framework."
import matplotlib.pyplot as plt  
matplotlib.rcParams['font.size'] = 8
import pprint 
pp = pprint.PrettyPrinter(indent=3)

from sklearn.linear_model import LogisticRegression

from winapi_deobf_experiments.preprocess.corpus import make_corpus
from winapi_deobf_experiments.preprocess.stats import form_dict_api_calls_to_n_args
from winapi_deobf_experiments.supervised.dataset import get_dataset_for_classification, \
	ensure_indexing_alignment_of_dataset_and_corpus_for_expt_2
from winapi_deobf_experiments.supervised.dataset_bag_of_words import get_bag_of_words_dataset_for_classification
from winapi_deobf_experiments.supervised.evaluate import get_top_n_model_accuracies, get_top_n_baserate_accuracies, \
	get_lr_predicted_labels 
from winapi_deobf_experiments.util.draw import plot_latent_trajectory, plot_accuracies, create_confusion_matrix_plot
from winapi_deobf_experiments.util.representation import construct_configs
from winapi_deobf_experiments.unsupervised.hmm_wrapper import create_unsupervised_data


def api_expt_1(path_to_configs="src/winapi_deobf_experiments/configs_expt_1.yaml"):
	"""

	Runs Experiment 1 of the paper "Towards Generic Deobfuscation of Windows API Calls"
	by Vadim Kotov and Mike Wojnowicz. 

	Args:
		path_to_configs: String 
			Filepath of yaml configurations file for the experiment. 

	"""

	### GET CONFIGS (as a Bunch object)
	configs=construct_configs(path_to_configs)
	print("\n Configurations file:")
	print vars(configs)
	_run_expt_1(configs)


def api_expt_2(path_to_configs="src/winapi_deobf_experiments/configs_expt_2.yaml"):
	"""

	Runs Experiment 2 of the paper "Towards Generic Deobfuscation of Windows API Calls"
	by Vadim Kotov and Mike Wojnowicz. 

	Args:
		path_to_configs: String 
			Filepath of yaml configurations file for the experiment. 

	"""

	### GET CONFIGS (as a Bunch object)
	configs=construct_configs(path_to_configs)
	print("\n Configurations file:")
	print vars(configs)
	_run_expt_2(configs)


def _run_expt_1(configs):
	"""

	The main engine of api_expt_1() is extracted here to allow test_api.py in unit tests 
	to tweak the configs towards a small run.

	Args:
		configs:  A Bunch instance. 
	"""

	print("\n\n ... Now running Experiment 1... ")

	### ### GENERATE CORPUS AND DATASET 
	corpus, results_dir, dataset = _get_corpus_and_dataset(configs, \
		dict_api_calls_to_n_args=None)

	### ### SUPERVISED CLASSIFICATION (Train multi-class logistic regression classifier and get predictions)
	### ### 1) USING SEQUENTIAL (HMM) VECTORIZATION
	yhat_test = get_lr_predicted_labels(dataset.features_train, dataset.features_test, dataset.labels_train)
	acc_sequential=np.mean(yhat_test==dataset.labels_test) 
	print("\n\n Expt1: Predictive Accuracy using Sequential Vectorization is %.04f" %(acc_sequential))
	create_confusion_matrix_plot(ys=dataset.labels_test, ys_predicted=yhat_test, 
		path_to_save_fig=os.path.join(results_dir,"confusion_with_hmm_vectorization.pdf"))

	### ### 2) USING BAG OF WORDS (BOW) VECTORIZATION 
	dataset_bow = get_bag_of_words_dataset_for_classification(corpus, configs.K, configs.label_tracker, seed=configs.seed)
	yhat_test_bow = get_lr_predicted_labels(dataset_bow.features_train, dataset_bow.features_test, dataset_bow.labels_train)
	acc_bow=np.mean(yhat_test_bow==dataset_bow.labels_test) 
	print("\n\n Expt1: Predictive Accuracy using Bag of Words Vectorization is %.04f" %(acc_bow))
	create_confusion_matrix_plot(ys=dataset_bow.labels_test, ys_predicted=yhat_test_bow, 
		path_to_save_fig=os.path.join(results_dir,"confusion_with_bag_of_words_vectorization.pdf"))


def _run_expt_2(configs):
	"""
	The main engine of api_expt_2() is extracted here to allow test_api.py in unit tests 
	to tweak the configs towards a small run.

	Args:
		configs:  A Bunch instance. 
	"""

	print("\n\n ... Now running Experiment 2... ")
	
	### ### GENERATE CORPUS AND DATASET 
	dict_api_calls_to_n_args=form_dict_api_calls_to_n_args(configs.label_tracker, \
		dir_filter=configs.dir_filter)
	corpus, results_dir, dataset = _get_corpus_and_dataset(configs, \
		dict_api_calls_to_n_args = dict_api_calls_to_n_args)

	### ### SUPERVISED CLASSIFICATION (Train multi-class logistic regression classifier and get predictions)
	ensure_indexing_alignment_of_dataset_and_corpus_for_expt_2(dataset, corpus)
	logistic=LogisticRegression(multi_class="multinomial", solver="lbfgs")
	logistic.fit(dataset.features_train, dataset.labels_train)
	top_n_model_accs=get_top_n_model_accuracies(logistic, dataset.features_test, dataset.labels_test, n_max=5)
	top_n_baserate_accs=get_top_n_baserate_accuracies(dataset.labels_train, dataset.labels_test, n_max=5)

	### PLOTS
	plot_accuracies(top_n_model_accs, top_n_baserate_accs, 
			method1_name="Generic deobfuscator", method2_name="Baserate",
			title="Predictive performance of generic API deobfuscator", ylim=[0.0,1.0], 
			path_to_save_fig=os.path.join(results_dir,"expt_2_accuracy.pdf"))
	plot_latent_trajectory(path_to_save_fig=os.path.join(results_dir,"latent_trajectory.pdf"))

def _get_corpus_and_dataset(configs, dict_api_calls_to_n_args):

	### ### GENERATE CORPUS FROM FILEPATHS
	corpus=	make_corpus(configs)

	### ### GET UNSUPERVISED FEATURES
	unsupervised_results_filepaths=create_unsupervised_data(corpus, configs, dict_api_calls_to_n_args)
	results_dir=unsupervised_results_filepaths.main_results_dir

	### ### GET DATASET FOR SUPERVISED CLASSIFICATION 
	dataset = get_dataset_for_classification(unsupervised_results_filepaths, configs.K, \
	 dict_api_calls_to_n_args=dict_api_calls_to_n_args, use_separate_hmms_for_each_api_function=False, \
	 feature_types=set(configs.feature_types))
	return corpus, results_dir, dataset 

if __name__=="__main__":
	api_expt_1()
	api_expt_2()


