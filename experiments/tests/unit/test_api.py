
from winapi_deobf_experiments.api import _run_expt_1, _run_expt_2
from winapi_deobf_experiments.util.representation import construct_configs 

def test_expt_1(path_to_configs="src/winapi_deobf_experiments/configs_expt_1.yaml"):

	print("\n\n ... Now directing Experiment 1 to a a compact model and a small snippet of sample data ... ")
	configs=construct_configs(path_to_configs)

	#reset some params for a quick sample run. 
	configs.K=2
	configs.label_tracker["all_labels"]["data_dirs"]=["data/expt1/sample/"]
	print("\n Configurations file:")
	print vars(configs)

	_run_expt_1(configs)

def test_expt_2(path_to_configs="src/winapi_deobf_experiments/configs_expt_2.yaml"):

	print("\n\n ... Now directing Experiment 2 to a a compact model and a small snippet of sample data ... ")
	configs=construct_configs(path_to_configs)

	#reset some params for a quick sample run. 
	configs.K=2
	configs.label_tracker["all_labels"]["data_dirs"]=["data/expt2/sample2/"]
	print("\n Configurations file:")
	print vars(configs)

	_run_expt_2(configs)