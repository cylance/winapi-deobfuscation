
import numpy as np 
from sklearn.linear_model import LogisticRegression
np.set_printoptions(suppress=True)

def get_lr_predicted_labels(features_train, features_test, labels_train):
	logistic=LogisticRegression(multi_class="multinomial", solver="lbfgs")
	logistic.fit(features_train, labels_train)
	yhat_test = logistic.predict(features_test)
	return yhat_test 
	
def get_top_n_model_accuracies(trained_model, features, labels, n_max):
	"""
	Gets the (standard) accuracy and the pct of time that the correct predictions were in the top n predictions
	"""
	yhat=trained_model.predict(features)
	#standard_acc=np.mean(yhat==labels)
	top_n_accs=np.zeros(n_max)
	for n in range(n_max):
		top_n_acc=_compute_prob_true_label_is_in_top_n_predictions(trained_model, features, labels, n=n+1)
		print("Top n=%d acc: %.04f" %(n+1, top_n_acc))
		top_n_accs[n]=top_n_acc
	return top_n_accs

def _compute_prob_true_label_is_in_top_n_predictions(model, features_test, labels_test, n=3):
	"""
	Arguments:
		model: A model instance from scikitlearn (e.g. logistic regression, fvm)
		features_test: numpy.ndarray.  has dimensionality n_test_samples x n_features 
		labels_test: numpy.ndarray. has dimensionality n_test_samples x 1 
		n: Int. Tells how many predictions to consider. 

	"""
	log_p_hats=model.predict_log_proba(features_test)
	n_successes=0
	for (i,api_scores) in enumerate(log_p_hats):
		top_results=api_scores.argsort()[-n:][::-1] 
		if labels_test[i] in top_results:
			n_successes+=1
	return float(n_successes)/len(labels_test)

def get_top_n_baserate_accuracies(labels_train, labels_test, n_max):
	label_counts_train=np.bincount([int(x) for x in labels_train])
	label_freqs_train=label_counts_train/float(sum(label_counts_train))
	idxs_by_sorted_frequencies=np.argsort(label_freqs_train)[::-1] #make it most to least frequent 
	top_n_accs=np.zeros(n_max)
 	for n in range(n_max):
		top_n_acc=np.mean([label in idxs_by_sorted_frequencies[:n+1] for label in labels_test])
		print("Baserate top n=%d acc: %.04f" %(n+1, top_n_acc))
		top_n_accs[n]=top_n_acc
	return top_n_accs







