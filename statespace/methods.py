import numpy as np
from primitive.linalg import invert_covariance
from primitive.methods import InferenceModule

# Base Kalman filtering class:

class KalmanFilter(InferenceModule):

    def set_state_dims(self, D):
        """This method creates an instance attribute D that represents the number of dimensions of the state. 

        This is required in general for initialisation of variable arrays containing mean vectors and covariance matrices at times t.
        """
        self.D = D

    def initialise(self):
        """This method initialises the instance history attribute history which contains the results of the estimation method.

        The history attribute is a list of iteration history variables. An iteration may refer to any computation that involves a change
        in time or a change in model parameters for a fixed time interval.

        Additionally, the instance attributes that represent the mean x_est, covariance P_est and log_evidence are created for ease of operation.
        """
        # Initialise sampling history:
        self.history = []

        # Initialise estimate attributes
        self.x_est = np.zeros(shape=(self.y.shape[0]+1, self.D, 1))
        self.P_est = np.zeros(shape=(self.y.shape[0]+1, self.D, self.D))

        self.log_evidence = []

    def initialise_iteration(self, t, x_init, P_init):
        """This method initialises an iteration history attribute and sets the initial point given by a time t, mean vector x_init and covariance matrix P_init. 

        An iteration may refer to any computation that involves a change in time for fixed model parameters or a change in model parameters for a fixed time interval.

        The iteration history is a list of dictionaries of model parameters, time, mean vector and covariance matrix associated with separate time points t and
        particles, i.e. an iteration may be index by time and particle id. 
        
        At the start of each iteration the iteration memory attribute is reset. Thus, it should be appended to the history attribute before this line.

        When inheriting from this class, the child class can extend the dictionary key words to include other variables. These should also be made arguments for
        the method. In the future, we can automate this copy/paste operation by implementing the method using kwargs.
        """
        # Initialise iteration history:
        self.iteration_history = []
        self.iteration_history.append(self.model.get_parameter_values() | {"time":t, "mean":x_init, "cov":P_init})

    #### DEVNOTE #### t_series, x_series in Levy models are considered parameters of our model.
    def kalman_iteration(self, y, x_init, P_init, s, t):
        """This method implements the prediction and correction steps of a Kalman filter iteration and returns the resulting mean vector x_est, covariance matrix P_est
        and log_likelihood.

        The associated matrices for the state transition A, state noise covariance Q, observation matrix H and observation noise covariance are computed using the
        model object. This implicitly assumes that no change in model parameters is made during this iteration.

        Additional changes in model parameters may be used in the Kalman iteration by extending this function.
        """
        A = self.model.expA(t-s)
        noise_mean, Q = self.model.I.conditional_moments(s=s, t=t)

        # Predict:
        x_pred = A @ x_init + noise_mean
        P_pred = A @ P_init @ A.T + Q

        # Update:
        residual_pred = y - self.model.H @ x_pred
        residual_pred_cov = self.model.H @ P_pred @ self.model.H.T + self.model.eps.covariance()
        kalman_gain = P_pred @ self.model.H.T @ invert_covariance(residual_pred_cov)

        x_est = x_pred + kalman_gain @ residual_pred
        P_est = (np.eye(x_init.shape[0]) - kalman_gain @ self.model.H) @ P_pred

        # Calculate log marginal likelihood:
        log_det = np.linalg.slogdet(residual_pred_cov)[1]
        log_marginal_likelihood = -0.5 * (log_det + residual_pred @ invert_covariance(residual_pred_cov) @ residual_pred + y.shape[0] * np.log(2 * np.pi))
        
        return x_est, P_est, log_marginal_likelihood
    
    def kalman_sweep(self):
        """This method implements a full sweep of the given input-output data set for fixed model parameters. This may be required to compute the marginal
        likelihood of the data set. The results will be appended to the iteration_history attribute.
         
        The iteration in this context represents a change in time for fixed model parameters.
        """

        extended_times = np.vstack((self.X[0], self.X)) # This is required to include the prior on x_est(times=0)
        for i, t in enumerate(self.X, 0):
            # Previous time step is s.
            s = extended_times[i]

            # Filtering:
            self.x_est[i+1], self.P_est[i+1], log_likelihood = self.kalman_iteration(y=self.y[i], x_init=self.x_est[i], P_init=self.P_est[i], s=s, t=self.X[i])

            # Save results:
            self.iteration_history.append(self.model.get_parameter_values() | {"time":self.X[i], "mean":self.x_est[i+1], "cov":self.P_est[i+1]})

    def filter(self, times, y, x_init, P_init):
        """This method is the default filtering functionality for changes in time with fixed model parameters.
        """

        # Set training variables:
        self.set_training_variables(y=y, X=times, Xeval=None)
        self.set_state_dims(D=x_init.shape[0])
        
        # Initialise history:
        self.initialise()

        # Start iterations:
        self.x_est[0] = x_init
        self.P_est[0] = P_init

        self.initialise_iteration(self.X[0], self.x_est[0], self.P_est[0])
        self.kalman_sweep()

        self.history.append(self.iteration_history)

        return self.history


class MetropolisHastingsKalmanFilter(KalmanFilter):

    def initialise(self, x_init, P_init):

        # Initialise subordinator:
        low = self.X.min()
        high = self.X.max()
        self.model.I.subordinator.initialise_proposal_samples(low, high)

        # Initialise sampling history:
        self.history = []

        # Initialise estimate attributes
        self.x_est = np.zeros(shape=(self.y.shape[0]+1, self.D, 1))
        self.P_est = np.zeros(shape=(self.y.shape[0]+1, self.D, self.D))
        self.log_evidence = []

        # Start iterations:
        self.x_est[0] = x_init
        self.P_est[0] = P_init

    def initialise_iteration(self, t, x_init, P_init, log_likelihood):

        # Initialise sampling history:
        self.iteration_history = [self.model.get_parameter_values() | {"time":t, 
                                                                        "mean":x_init, 
                                                                        "cov":P_init, 
                                                                        "log_likelihood":log_likelihood,
                                                                        "t_series":self.model.I.subordinator.t_series, 
                                                                         "x_series":self.model.I.subordinator.x_series
                                                                     }
                                ]
        
        self.iteration_log_evidence = [log_likelihood]


    def kalman_iteration(self, y, x_init, P_init, s, t, t_series, x_series):

        A = self.model.expA(t - s)
        noise_mean, Q = self.model.I.proposed_conditional_moments(s, t, t_series, x_series)

        # Predict:
        x_pred = A @ x_init + noise_mean
        P_pred = A @ P_init @ A.T + Q

        # Update:
        residual_pred = y - self.model.H @ x_pred
        residual_pred_cov = self.model.H @ P_pred @ self.model.H.T + self.model.eps.covariance()
        kalman_gain = P_pred @ self.model.H.T @ invert_covariance(residual_pred_cov)

        x_est = x_pred + kalman_gain @ residual_pred
        P_est = (np.eye(x_init.shape[0]) - kalman_gain @ self.model.H) @ P_pred

        # Calculate log marginal likelihood:
        log_det = np.linalg.slogdet(residual_pred_cov)[1]
        log_marginal_likelihood = -0.5 * (log_det + residual_pred @ invert_covariance(residual_pred_cov) @ residual_pred + y.shape[0] * np.log(2 * np.pi))

        return x_est, P_est, log_marginal_likelihood
    
    def get_mixture_moments(self):
        means = np.array([sample['mean'] for sample in self.iteration_history])
        post_mix_mean = means.mean(axis=0)

        residual_mean = (means - post_mix_mean)
        mixture_adjustment = residual_mean @ np.transpose(residual_mean, axes=(0,2,1))

        post_mix_cov = (np.array([sample['cov'] for sample in self.iteration_history]) + mixture_adjustment).mean(axis=0)

        return post_mix_mean, post_mix_cov

    def filter(self, times, y, x_init, P_init, n_samples=10):

        # Set training variables:
        self.set_training_variables(y=y, X=times, Xeval=None)
        self.set_state_dims(D=x_init.shape[0])
        
        # Initialise history:
        self.initialise(x_init, P_init)

        extended_times = np.vstack((self.X[0]-1, self.X)) # This is required to include the prior on x_est(times=0-1). The choice is arbitrary
        for i, t in enumerate(times, 0):
            # Previous time step is s.
            s = extended_times[i]

            self.initialise_iteration(t=t, x_init=self.x_est[i], P_init=self.P_est[i], log_likelihood=-10*np.ones((1, 1)))

            # Start sampling:
            for _ in range(n_samples):
                
                # Propose subordinator jumps:
                proposal_t_series, proposal_x_series = self.model.I.subordinator.propose_subordinator((s,t))

                # Filtering:
                proposed_x_est, proposed_P_est, proposed_log_likelihood = self.kalman_iteration(y=self.y[i],
                                                                                x_init=self.x_est[i],
                                                                                P_init=self.P_est[i],
                                                                                s=s, 
                                                                                t=self.X[i],
                                                                                t_series=proposal_t_series,
                                                                                x_series=proposal_x_series
                                                                            )
            

                acceptance_prob = np.min([1, np.exp(proposed_log_likelihood[0][0] - self.iteration_log_evidence[-1][0][0])])
                u = np.random.uniform(low=0.0, high=1.0)

                if u < acceptance_prob:
                    self.model.I.subordinator.set_proposal_samples(proposal_t_series, proposal_x_series)

                    # Instead of setting the value of self.x_est and self.P_est here, set them after the inner iterations are complete and use the mixture-of-Gaussians.
                    #self.x_est[i+1] = proposed_x_est
                    #self.P_est[i+1] = proposed_P_est
                    self.iteration_log_evidence.append(proposed_log_likelihood)
                    self.iteration_history.append(self.model.get_parameter_values() | {"time":t, 
                                                                                       "mean":proposed_x_est, 
                                                                                       "cov":proposed_P_est, 
                                                                                       "t_series":proposal_t_series, 
                                                                                       "x_series":proposal_x_series
                                                                                      }
                                                 )
                else:
                    self.iteration_log_evidence.append(self.iteration_log_evidence[-1])
                    self.iteration_history.append(self.iteration_history[-1])

            self.x_est[i+1], self.P_est[i+1] = self.get_mixture_moments()

            self.log_evidence.append(self.iteration_log_evidence)
            self.history.append(self.iteration_history)

        return self.history