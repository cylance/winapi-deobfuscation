
import numpy as np 
from numpy import transpose as trans #for prediction dumps 
from scipy.misc import logsumexp as logsumexp #for prediction dumps 

from winapi_deobf_experiments.util.representation import get_key_given_value 
from winapi_deobf_experiments.util.math_funs import nudge_prob_matrix_away_from_zero


class CategoricalHMMParameters:
	"""
	(1) Extracts useful HMM parameters from an instance of pyhsmm.models.HMM
	(2) Nudges away from 0 
	(3) Additionally epresents as logs. 

	Attributes:
		initial_state_distribution: np.ndarray 
			Initial state distribution.  Transferred over from pyhsmm.models.HMM
		transition_matrix: np.ndarray 
			Transferred over from pyhsmm.models.HMM
		emission_prob_matrix: np.ndarray 
			Constructed via pyhsmm.models.HMM
		K: Int 
			Number of hidden states. 
		log_A:
			Transition matrix, cleaned and logged. 
		log_B:
			Emission prob matrix, cleaned and logged. 
		log_pi:
			Init state distn, cleaned and logged. 

	"""

	def __init__(self, hmm):
		"""
		Arguments:
			hmm: instance of pyhsmm.models.HMM
		"""
		self.initial_state_distribution=hmm.init_state_distn.pi_0
		self.transition_matrix=hmm.trans_distn.trans_matrix
		self.emission_prob_matrix, self.K=self._get_K_and_emission_prob_matrix(hmm)
		self._set_cleaned_and_logged_parameters()
		#if store_latent_state_series:
		#	self.latent_state_series=hmm.stateseqs 

	def _get_K_and_emission_prob_matrix(self, hmm):
		K=len(hmm.obs_distns)
		W=len(hmm.obs_distns[0].params["weights"])
		mat=np.zeros((W,K))
		for k in range(K):
			mat[:,k]=hmm.obs_distns[k].params['weights']
		return mat,K 

	def _set_cleaned_and_logged_parameters(self):
		"""
		Nudge parameters away from zero and take logs. 
		"""
		A,B,pi=self._nudge_parameters_away_from_zero()
		self.log_A,self.log_B,self.log_pi=np.log(A),np.log(B),np.log(pi)

	def _nudge_parameters_away_from_zero(self):
		A=trans(nudge_prob_matrix_away_from_zero(trans(self.transition_matrix)))
		B=nudge_prob_matrix_away_from_zero(self.emission_prob_matrix)
		pi=nudge_prob_matrix_away_from_zero(self.initial_state_distribution)	
		return A,B,pi 


class HMMFeatureMaker:
	""" 
	1) Extracts interpretable features (e.g. transiton matrix) from trained hmm model.
	2) Has methods to spit out the estimated latent state probabilities 
	and log likelihood information in a streaming/forward sense; i.e. all features at time t are based only on 
	observations O_1,...O_t and not O_{t+1},...,O_T.

	Attributes:
		hmm_params: Instance of CategoricalHMMParameters.
		model_name: String
		feature_names: List of strings 
			Describes all the features 


	"""
	def __init__(self, hmm, model_name, streaming=True, feature_types={"log_likelihoods", "log_state_marginals"}):
		"""
		Arguments:
			feature_type: String 
				Either "likelihood_feats" (currently supported) or "latent_state_marginals" (not supported)
		"""	
		if not streaming:
			raise NotImplementedError("Right now can only do streaming feautres")
		if ("log_likelihoods" not in feature_types) and ("log_state_marginals" not in feature_types):
			raise ValueError("Feature types must include either log_likelihoods or log_state_marginals; "\
				"you provided %s" %(feature_types))
		self.hmm_params=CategoricalHMMParameters(hmm)
		self.model_name=model_name 
		self.feature_names=self._get_feature_names(streaming, feature_types)

	def _get_feature_names(self, streaming, feature_types):
		#TD: use some nicer automatic way to fill in "_" between words. 
		mode="streaming" if streaming else "complete_sequence"
		feature_names=[]
		if "log_state_marginals" in feature_types:
			latent_state_features=[mode+"_latent_state_marginal"+str(k)+"_prob_from_"+self.model_name+"_model" for k in range(self.hmm_params.K)]
			feature_names.extend(latent_state_features)
		if "log_likelihoods" in feature_types:
			ll_features_pre=[mode+"_log_likelihood_of_event", mode+"_log_likelihood_of_partial_seq", mode+"_mean_log_likelihood_of_events_in_partial_seq"]
			ll_features=[feat+"_from_"+self.model_name+"_model" for feat in ll_features_pre]
			feature_names.extend(ll_features)
		return feature_names

	def forward_algorithm_for_one_sample(self, sample):
		"""
		Parameters:
			Sample: An array of integers representing a time series of events which has 
			been vectorized and preprocessed in the same way as the data which yielded
			the model parameters. 

		Returns:		
			Log alpha: A TxK matrix whose entries are the logs of the forward variable,
			so the (t,k)th entry is the log of P(O_1,...O_t, Q_t=k | lambda).  
		"""

		T=len(sample)
		log_alpha=np.zeros((T,self.hmm_params.K)) #recall; all negatives, more negative means more unlikely. 

		#initialize the forward algorithm 
		curr_obs=sample[0]
		log_b=self.hmm_params.log_B[curr_obs,:]
		#b=B[obs,:]
		#[x*y for x,y in zip(pi,b)]
		log_alpha[0,:]=self.hmm_params.log_pi+log_b 

		#do thre rest of the forward algorthm. 
		for t in range(1,T):
			curr_obs=sample[t]
			log_b=self.hmm_params.log_B[curr_obs,:]
			for k in range(self.hmm_params.K):
				log_alpha[t,k]=logsumexp(log_alpha[(t-1),:]+self.hmm_params.log_A[:,k])+log_b[k]			
		#summative contribution of O_t to log [P(O_1:t | alpha)] 
		return log_alpha 

	def _row_normalize_the_log_forward_variable(self, log_alpha):
		"""
		This does normalization on the forward (alpha) variable; we want to convert the alpha variable P(O_{1:t}, Q_t | lambda)
		to the latent state marginals P(Q_t =k | O_{1:t}, lambda) to get a probability 
		distribution over latent states. 
		"""
		logS=logsumexp(log_alpha,1)[:,np.newaxis]
		streaming_log_state_marginals=log_alpha-logS
		return streaming_log_state_marginals

	def get_streaming_log_state_marginals_for_one_sample(self, log_alpha):
		"""

		Returns:
			log_state_marginals: A TxK matrix whose entries are the logs of P(Q_t =k | O_{1:t}, lambda).
			i.e. the state marginals over the sequence observed so far.  
			Note that t<T, the final observation for the time series. 
		"""
		streaming_log_state_marginals=self._row_normalize_the_log_forward_variable(log_alpha)
		return streaming_log_state_marginals

	def _row_normalize_the_log_forward_variable(self, log_alpha):
		"""
		This does normalization on the forward (alpha) variable; we want to convert the alpha variable P(O_{1:t}, Q_t | lambda)
		to the latent state marginals P(Q_t =k | O_{1:t}, lambda) to get a probability 
		distribution over latent states. 
		"""
		logS=logsumexp(log_alpha,1)[:,np.newaxis]
		streaming_log_state_marginals=log_alpha-logS
		return streaming_log_state_marginals
		
	def get_complete_log_state_marginals_for_one_sample(self, hmm, sample):
		"""
		Returns:
			state_marginals: A TxK matrix whose entries are the logs of P(Q_t =k | O_T, lambda),
			i.e. the state marginals over the entire time series. 
			Note that T is the final observation in the time series.  
		"""
		raise NotImplementedError("Currently can only support streaming log state marginals")
		# Code below has problems; seems to return nan's
		# log_state_marginals=np.log(hmm.heldout_state_marginals(sample))
		# return log_state_marginals

	def derive_likelihood_features_for_one_sample(self, log_prob_observation_sequence_and_latent_states):
		"""
		Returns:
			ll_individual_observations: Contribution to streaming HMM log likelihood of sequence so far, for one event, i.e. log P(O_1:t | alpha) - log P(O_1:(t-1) | alpha) ; note we marginalize over latent state dist'n 
			ll_partial_sequences:  Streaming HMM log likelihood of sequence so far (i.e.: log P(O_1:t | alpha), so we marginalize over latent state dist'n) 
			ll_individual_observations_running_mean: The quantity "ll_partial_sequences" is guaranteed to monotonically decrease
									in the number of elements in the time series; this normalizes by the number of event so far 
									so that we can more adequately do things like guage whether/when the sample has started to "get weird"
		"""
		T=np.shape(log_prob_observation_sequence_and_latent_states)[0]
		ll_partial_sequences=logsumexp(log_prob_observation_sequence_and_latent_states,1) #log of P(O_1:t | alpha)
		ll_individual_observations= np.insert(np.diff(ll_partial_sequences),0,ll_partial_sequences[0])
		ll_individual_observations_running_mean=np.array([ll_partial_sequences[i]/(i+1) for i in range(T)]) #"log likelihood counter"
		return ll_individual_observations[:,np.newaxis], ll_partial_sequences[:,np.newaxis], ll_individual_observations_running_mean[:,np.newaxis]

	def get_log_state_marginals_and_log_likelihoods_for_sample(self, sample, dict_labels_to_fixed_ts_length=None,
			default_unlikely_value=-200.0):
		"""
		Arguments: 
			sample: np.ndarray 
				A time series of length T 
			dict_labels_to_fixed_ts_length: dict<string,int> or None. 
				Dictionary mapping model names to a fixed time series length.
				Optional. 
			default_unlikely_value: float. 
				If dict_labels_to_fixed_ts_length is providd, and the sample's time series length is short 
				short of the stipulated fixed length for self.model_name in dict_labels_to_fixed_ts_length, we
				fill in the "missing observations" with the default_unlikely_value.

		Returns:
			log_state_marginals: np.ndarray 
				Matrix of shape (T,K) whose entry (t,k) provides the log marginal prob 
				of being in latent state k at time t 
			log_likelihoods: np.ndarray
				Matrix of shape (T,3) whose entry (t,:) provides the log likelihood of the individ obs, 
				log like of partial sequence, and running mean log likelihood of individ obs. 

		"""
		T=len(sample)
		if dict_labels_to_fixed_ts_length:
			if self.model_name not in dict_labels_to_fixed_ts_length.keys():
				return ValueError("If you provide a non-None value for dict_labels_to_fixed_ts_length " \
					"then its keys must match the model_names for the HMMFeatureMaker.  Yet now we're trying to" \
					"generate features using label %s which is not in the dict" %(self.model_name))
			#extract, from each sample, only the relevant n_args for that model 
			n_obs_for_model=dict_labels_to_fixed_ts_length[self.model_name]
			sample=sample[:n_obs_for_model]
		log_alpha=self.forward_algorithm_for_one_sample(sample)
		log_state_marginals=self.get_streaming_log_state_marginals_for_one_sample(log_alpha)
		self._check_that_latent_state_marginals_sum_to_one(log_state_marginals)
		a,b,c=self.derive_likelihood_features_for_one_sample(log_alpha)
		log_likelihoods=np.hstack((a,b,c))
		#if the given label's time series is short of the stipulated fixed length,
		#we replace it with the default_unlikely_value
		if dict_labels_to_fixed_ts_length:
			if n_obs_for_model>T: #model needs more args than are avialable; automatic exclusion
				log_state_marginals=np.ones_like(log_state_marginals)*default_unlikely_value
				log_likelihoods=np.ones_like(log_likelihoods)*default_unlikely_value
		return log_state_marginals, log_likelihoods 

	def _check_that_latent_state_marginals_sum_to_one(self, log_state_marginals):
		assert np.isclose(np.sum(np.exp(log_state_marginals)), len(log_state_marginals)) 
