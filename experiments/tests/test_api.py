from nose import tools as nt
import matplotlib
matplotlib.use('TkAgg')

import winapi_deobf_experiments.api as api


def test_experiment_1():
	print("\n\n .... Now testing code for experiment 1 with sample data ...\n\n")
	api.run_experiment_1(parsed_api_data_dir="data/expt1/sample/", train_pct=.99, K=2)

def test_experiment_2():
	print("\n\n .... Now testing code for experiment 2 with sample data ...\n\n")
	api.run_experiment_2(parsed_api_data_dir="data/expt2/sample/", train_pct=.99, K=2)
