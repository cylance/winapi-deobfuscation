"""
Can't do a block comment in a .yaml file, so we define the configs here

	Attributes:
		results_super_dir: String.
			Local directory to where results will be sent. 
		model_type: String 
			Type of HMM model fit.  Currently only 'Baum-Welch' is supported.  
			Baum-Welch does a maximum likelihood fit to the dataset.  Later extensions will more completely incorporate VB 
			and other variants. 
		default_unlikely_value: Float. 
			Missing data in the log likelihoods get this number assigned to it, which should be very small. 
		label_tracker: Dict<string, Dict>>
			Dictionary mapping labels to inner dicts, which maps "data_dirs" to a list of strings, 
			interpreted as filepaths, and "train_pct" to a float. 
		run_summary: String 
			Descriptive phrase used in forming results subdirectories describing the run. 
		dir_filter: String:
			Input data is filtered based on this string. 
		corpus_type: String 
			A string corresponding to a Corpus derived class that is Not an abstract base class. 
		use_separate_hmms_for_each_label: Bool
			If True, fits a separate HMM model for every label provided in the label_tracker.
		K: Int
			Number of latent states for HMM 
		feature_types: List of strings. 
			Current support "log_likelihoods", "log_state_marginals", or both. 
			Would like this to be a set, but it's not supported by yaml. 
		seed: Int 
			Random seed to initialize train/test splits and HMM initalization. 

The construct_configs() function has the following property:
	
	Returns: 	 
		The yaml configs file as a (key, value) dictionary is represented instead as a 
		class instance (so we can call e.g. configs.instance_type instead of configs["instance_type"])

"""