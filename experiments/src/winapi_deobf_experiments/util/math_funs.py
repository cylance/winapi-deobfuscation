
import numpy as np 

def convert_np_array_into_comma_separated_string(array):
	my_string=""
	for k in range(len(array)):
		my_string+="%s," % str(array[k])
	return my_string.rstrip(",")

def nudge_prob_matrix_away_from_zero(matrix_whose_columns_are_prob_vecs, eps_upper_bound=1e-4,
									 eps_lower_bound=1e-20):
	"""

	This function will find "m", the smallest value in a probability matrix, add m to all values
	in each probability vector, and renormalize.  This will  correct for exact 0 probabilities 
	(as might be obtained in ML type algorithms).   [Why? Exact 0's throw off likelihood computations;
	these are determined iteratively via the forward and backwards algorithms and are done on the log scale;
	but probabilities equal to 0 at time-step t become -inf on the log scale, which throw off likelihood computations
	for all time steps t+1, t+2, ... . ]

	Arguments:
		matrix_whose_columns_are_prob_vecs: A ndarray whose columns sum to 1. 
		eps_upper_bound:  "m" might be fairly large; this sets the minimum possible value we'll add in
						 so as to prevent large unintended perturbations to the original dataset.
		eps_lower_bound:  Conversely, "m" might be extremely small, such that it's essentially already
						0.  This sets the maximum possible value we'll add in.  

	"""
	X=matrix_whose_columns_are_prob_vecs
	orig_array_is_1d=np.ndim(X)==1 
	X=X[:,np.newaxis] if orig_array_is_1d else X 
	eps_X=max(min(np.ndarray.min(X[X>0]),eps_upper_bound),eps_lower_bound)	
	Nrows,Ncols=np.shape(X) #or  (num els in prob vec, num prob vecs)
	for i in range(Ncols):		
		X[:,i]=(X[:,i]+eps_X)/(1+eps_X*Nrows)
	X=X[:,0] if orig_array_is_1d else X
	return X

def generate_feature_idx_bound_tuples_per_group(nFeats,nGroups):
	feature_idx_bounds=[((nFeats*i),nFeats*(i+1)) for i in range(nGroups)]
	return feature_idx_bounds
