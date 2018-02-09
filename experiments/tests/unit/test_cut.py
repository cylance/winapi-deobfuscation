
import nose.tools as nt 

from winapi_deobf_experiments.preprocess.cut import Cut
from winapi_deobf_experiments.preprocess.filepath_tracker import FilepathTracker 

def test_cut():

	#generate fake filepath characteristics. 
	filepath="fake_filepath"
	filepath_label="good"
	desired_train_pct=.80
	N_samples_in_filepath=1000
	start_idx=0
	stop_idx=999
	seeds=[1234, 1235]
	filepath_tracker=FilepathTracker(N_samples_in_filepath=N_samples_in_filepath, 
		filepath_label=filepath_label, start_idx_of_time_series_for_filepath=start_idx, \
			stop_idx_of_time_series_for_filepath=stop_idx)

	#generate 
	label_tracker={filepath_label: {"train_pct": desired_train_pct, "data_dirs": [filepath]}}
	filepath_trackers={filepath: filepath_tracker} #dict mapping filepaths to filepath_tracker instances. 
	print("\n \n ... Now testing that we get the same random selection of samples when we set the seed the same ... ")
	cut=Cut(label_tracker, filepath_trackers, seed=seeds[0])
	use_idxs_seed_1_run_1=cut.use_idxs
	withheld_idxs_seed_1_run_1=cut.withheld_idxs
	cut=Cut(label_tracker, filepath_trackers, seed=seeds[0])
	use_idxs_seed_1_run_2=cut.use_idxs
	withheld_idxs_seed_1_run_2=cut.withheld_idxs
	nt.ok_(use_idxs_seed_1_run_1==use_idxs_seed_1_run_2)
	nt.ok_(withheld_idxs_seed_1_run_1==withheld_idxs_seed_1_run_2)
	print("\n \n ... Now testing that we get different random selection of samples when we set the seed differently... ")
	cut=Cut(label_tracker, filepath_trackers, seed=seeds[1])
	use_idxs_seed_2=cut.use_idxs
	withheld_idxs_seed_2=cut.withheld_idxs
	nt.ok_(use_idxs_seed_1_run_1!=use_idxs_seed_2)
	nt.ok_(withheld_idxs_seed_1_run_1!=withheld_idxs_seed_2)

