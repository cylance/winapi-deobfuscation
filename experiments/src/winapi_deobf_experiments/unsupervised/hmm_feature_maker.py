
import numpy as np 
from numpy import transpose as trans #for prediction dumps 
from scipy.misc import logsumexp as logsumexp #for prediction dumps 

from winapi_deobf_experiments.util.representation import get_key_given_value 
from winapi_deobf_experiments.util.math_funs import nudge_prob_matrix_away_from_zero


class HMMFeatureMaker:
	""" 
	Stores transformed params (for one cut only).  Has methods to spit out the estimated latent state probabilities 
	and log likelihood information in a streaming/forward sense; i.e. all features at time t are based only on 
	observations O_1,...O_t and not O_{t+1},...,O_T.
	"""
	def __init__(self, model, model_name, streaming, use_latent_state_marginals, use_likelihood_feats):	
		self.log_A=None
		self.log_B=None 
		self.log_pi=None
		self.model=model
		self.model_name=model_name 
		self.K=self.model.trans_distn.trans_matrix.shape[0]
		self.feature_names=[]
		self._prep_forward_algorithm_parameters_for_a_cut()
		self._make_feature_names(streaming, use_latent_state_marginals, use_likelihood_feats)

	def _prep_forward_algorithm_parameters_for_a_cut(self):
		A_cut=self.model.transition_matrix
		B_cut=self.model.emission_prob_matrix
		pi_cut=self.model.initial_state_distribution
		A_cut,B_cut,pi_cut=self._nudge_parameters_away_from_zero(A_cut,B_cut,pi_cut)
		self.log_A,self.log_B,self.log_pi=np.log(A_cut),np.log(B_cut),np.log(pi_cut)

	def _make_feature_names(self, streaming, use_latent_state_marginals, use_likelihood_feats):
		if not use_likelihood_feats and not use_latent_state_marginals:
			raise ValueError("To make the new fvm we need to either be adding likelihood features and/or latent state marginals" \
		 		"but you have specified both to be false")
		mode="streaming" if streaming else "complete_sequence"
		ll_features_pre=[mode+"_log_likelihood_of_event", mode+"_log_likelihood_of_partial_seq", mode+"_mean_log_likelihood_of_events_in_partial_seq"]
		ll_features=[feat+"_from_"+self.model_name+"_model" for feat in ll_features_pre]
		latent_state_features=[mode+"_latent_state_marginal"+str(k)+"_prob_from_"+self.model_name+"_model" for k in range(self.K)]
		self.feature_names=[]
		if use_latent_state_marginals:
			self.feature_names.extend(latent_state_features)
		if use_likelihood_feats:
			self.feature_names.extend(ll_features)

	def _nudge_parameters_away_from_zero(self,A,B,pi):
		B=nudge_prob_matrix_away_from_zero(B)
		A=trans(nudge_prob_matrix_away_from_zero(trans(A)))
		pi=nudge_prob_matrix_away_from_zero(pi)	
		return A,B,pi 

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
		log_alpha=np.zeros((T,self.K)) #recall; all negatives, more negative means more unlikely. 

		#initialize the forward algorithm 
		curr_obs=sample[0]
		log_b=self.log_B[curr_obs,:]
		#b=B[obs,:]
		#[x*y for x,y in zip(pi,b)]
		log_alpha[0,:]=self.log_pi+log_b 

		#do thre rest of the forward algorthm. 
		for t in range(1,T):
			curr_obs=sample[t]
			log_b=self.log_B[curr_obs,:]
			for k in range(self.K):
				log_alpha[t,k]=logsumexp(log_alpha[(t-1),:]+self.log_A[:,k])+log_b[k]			
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

	def get_complete_log_state_marginals_for_one_sample(self, sample):
		"""
		Returns:
			state_marginals: A TxK matrix whose entries are the logs of P(Q_t =k | O_T, lambda),
			i.e. the state marginals over the entire time series. 
			Note that T is the final observation in the time series.  
		"""
		log_state_marginals=np.log(self.model.heldout_state_marginals(sample))
		return log_state_marginals

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
