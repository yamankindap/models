import numpy as np
from primitives.linalg import invert_covariance
from primitives.methods import InferenceModule

class KalmanFilter(InferenceModule):

    def initialise(self):

        # Initialise sampling history:
        self.history = []

    def kalman_iteration(self, y, x_init, P_init, s, t):

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
        
        return x_est, P_est

    def filtering(self, times, y, x_init, P_init):
        
        # Initialise history:
        self.initialise()
        self.history.append(self.parameters | {"time":np.array([0.]), "mean":x_init, "cov":P_init})
        
        # Initialise estimate
        x_est = np.zeros(shape=(y.shape[0]+1, x_init.shape[0], 1))
        P_est = np.zeros(shape=(y.shape[0]+1, P_init.shape[0], P_init.shape[1]))

        extended_times = np.vstack((0, times))

        # Start iterations:
        x_est[0] = x_init
        P_est[0] = P_init

        for i, t in enumerate(times, 1):
            s = extended_times[i-1]

            x_est[i], P_est[i] = self.kalman_iteration(y=y[i-1], x_init=x_est[i-1], P_init=P_est[i-1], s=s, t=t)

            self.history.append(self.parameters | {"time":t, "mean":x_est[i], "cov":P_est[i]})

        return self.history





    def initialise_iteration(self, i):

        # Initialise iteration history:
        self.iteration_history = []

        # Initialise 

        # GP specific step:
        mean, cov, log_marginal_likelihood = self.model.posterior_density(self.y[0:1+i], self.X[0:1+i], self.X[i:1+i])

        # Initialise sampling history:
        self.iteration_history.append(self.parameters | {"mean":mean, "cov":cov, "log_marginal_likelihood":log_marginal_likelihood})
        
    def sample(self, i):

        for key in self.parameters.keys():
            # Propose new values:
            parameter_proposal = self.parameters.copy()
            parameter_proposal[key] = self.proposal[key].propose(x=self.parameters[key], shape=None)
            self.model.set_parameter_values(**parameter_proposal)

            # GP specific step:
            mean, cov, log_marginal_likelihood = self.model.posterior_density(self.y[0:1+i], self.X[0:1+i], self.X[i:1+i])

            # Metropolis-Hastings step:
            proposal_joint_log_likelihood = log_marginal_likelihood.item() + self.prior[key].log_likelihood(parameter_proposal[key]) + self.proposal[key].log_likelihood(self.parameters[key], parameter_proposal[key])
            previous_joint_log_likelihood = self.iteration_history[-1]["log_marginal_likelihood"].item() + self.prior[key].log_likelihood(self.iteration_history[-1][key]) + self.proposal[key].log_likelihood(parameter_proposal[key], self.parameters[key])

            acceptance_prob = np.min([1, np.exp(proposal_joint_log_likelihood - previous_joint_log_likelihood)])
            u = np.random.uniform(low=0.0, high=1.0)

            if u < acceptance_prob:
                self.parameters = parameter_proposal
                self.iteration_history.append(self.model.get_parameter_values() | {"mean":mean, "cov":cov, "log_marginal_likelihood":log_marginal_likelihood})
            else:
                self.model.set_parameter_values(**self.parameters)
                self.iteration_history.append(self.iteration_history[-1])


    def filter(self, y, X, Xeval=None, n_samples=10):

        # Set training variables:
        self.set_training_variables(y, X, Xeval)

        # Initialise sampling history:
        self.initialise()

        # Start filtering:
        for i in range(X.shape[0]):

            self.initialise_iteration(i)

            # Start sampling:
            for _ in range(n_samples):
                self.sample(i)

            self.history.append(self.iteration_history)

        return self.history