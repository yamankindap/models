import numpy as np
from primitives.priors import PriorModule
from primitives.proposals import ProposalModule

from primitives.methods import InferenceModule


# Gaussian process Metropolis-Within-Gibbs sampler for kernel parameters.

class MH_Within_Gibbs_sampler_for_GP_parameters(InferenceModule):

    def initialise(self):
        # GP specific step:
        mean, cov, log_marginal_likelihood = self.model.posterior_density(self.y, self.X, self.Xeval)

        # Initialise sampling history:
        self.history = [self.parameters | {"mean":mean, "cov":cov, "log_marginal_likelihood":log_marginal_likelihood}]

    def sample(self):

        for key in self.parameters.keys():
            # Propose new values:
            parameter_proposal = self.parameters.copy()
            parameter_proposal[key] = self.proposal[key].propose(x=self.parameters[key], shape=None)
            self.model.set_parameter_values(**parameter_proposal)

            # GP specific step:
            mean, cov, log_marginal_likelihood = self.model.posterior_density(self.y, self.X, self.Xeval)

            # Metropolis-Hastings step:
            proposal_joint_log_likelihood = log_marginal_likelihood.item() + self.prior[key].log_likelihood(parameter_proposal[key]) + self.proposal[key].log_likelihood(self.parameters[key], parameter_proposal[key])
            previous_joint_log_likelihood = self.history[-1]["log_marginal_likelihood"].item() + self.prior[key].log_likelihood(self.history[-1][key]) + self.proposal[key].log_likelihood(parameter_proposal[key], self.parameters[key])

            acceptance_prob = np.min([1, np.exp(proposal_joint_log_likelihood - previous_joint_log_likelihood)])
            u = np.random.uniform(low=0.0, high=1.0)

            if u < acceptance_prob:
                self.parameters = parameter_proposal
                self.history.append(self.model.get_parameter_values() | {"mean":mean, "cov":cov, "log_marginal_likelihood":log_marginal_likelihood})
            else:
                self.model.set_parameter_values(**self.parameters)
                self.history.append(self.history[-1])

    def fit(self, y, X, Xeval=None, n_samples=10):

        # Set training variables:
        self.set_training_variables(y, X, Xeval)

        # Initialise sampling history:
        self.initialise()

        # Start sampling:
        for _ in range(n_samples):
            self.sample()

        return self.history

# Gaussian process Sequential Metropolis-Within-Gibbs sampler for kernel parameters.

class Sequential_MH_Within_Gibbs_sampler_for_GP_parameters(InferenceModule):

    def initialise(self):

        # Initialise sampling history:
        self.history = []

    def initialise_iteration(self, i):

        # Initialise iteration history:
        self.iteration_history = []

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


# Non Gaussian process Metropolis-Within-Gibbs sampler for kernel parameters and subordinator function.

class MH_Within_Gibbs_sampler_for_NGP(InferenceModule):

    def initialise(self):

        # Initialise subordinator:
        low = self.Xeval.min()
        high = self.Xeval.max()
        self.model.subordinator.initialise_proposal_samples(low, high)
        
        # Conditional GP inference step:
        mean, cov, log_marginal_likelihood = self.model.posterior_density(self.y, self.X, self.Xeval)

        # Initialise sampling history:
        self.history = [self.parameters | {"mean":mean, 
                                           "cov":cov, 
                                           "log_marginal_likelihood":log_marginal_likelihood, 
                                           "t_series":self.model.subordinator.t_series, 
                                           "x_series":self.model.subordinator.x_series
                                          }
                       ]

    # Construct a grid of intervals of size n.
    def proposal_grid(self, low, high, n):
        step_size = (high-low)/n
        try:
            grid = np.sort(np.array([np.arange(start=low, stop=high+step_size, step=step_size), np.arange(start=low, stop=high+step_size, step=step_size)]).flatten())[1:-1].reshape(n,2)
        except:
            grid = np.sort(np.array([np.arange(start=low, stop=high+step_size, step=step_size), np.arange(start=low, stop=high+step_size, step=step_size)]).flatten())[1:-1].reshape(n+1,2)
        return grid

    def sample_subordinator(self, proposal_intervals):
        
        for proposal_interval in proposal_intervals:
            
            proposal_t_series, proposal_x_series = self.model.subordinator.propose_subordinator(proposal_interval)
            proposal_mean, proposal_cov, proposal_log_marginal_likelihood = self.model.proposal_posterior_density(self.y, self.X, self.Xeval, proposal_t_series, proposal_x_series)

            acceptance_prob = np.min([1, np.exp(proposal_log_marginal_likelihood.item() - self.history[-1]["log_marginal_likelihood"].item())])
            u = np.random.uniform(low=0.0, high=1.0)

            if u < acceptance_prob:
                self.model.subordinator.set_proposal_samples(proposal_t_series, proposal_x_series)
                self.history.append(self.parameters | {"mean":proposal_mean, "cov":proposal_cov, "log_marginal_likelihood":proposal_log_marginal_likelihood, "t_series":proposal_t_series, "x_series":proposal_x_series})
            else:
                self.history.append(self.history[-1])

        
    def sample(self):

        for key in self.parameters.keys():
            # Propose new values:
            parameter_proposal = self.parameters.copy()
            parameter_proposal[key] = self.proposal[key].propose(x=self.parameters[key], shape=None)
            self.model.set_parameter_values(**parameter_proposal)

            # GP specific step:
            mean, cov, log_marginal_likelihood = self.model.posterior_density(self.y, self.X, self.Xeval)

            # Metropolis-Hastings step:
            proposal_joint_log_likelihood = log_marginal_likelihood.item() + self.prior[key].log_likelihood(parameter_proposal[key]) + self.proposal[key].log_likelihood(self.parameters[key], parameter_proposal[key])
            previous_joint_log_likelihood = self.history[-1]["log_marginal_likelihood"].item() + self.prior[key].log_likelihood(self.history[-1][key]) + self.proposal[key].log_likelihood(parameter_proposal[key], self.parameters[key])

            acceptance_prob = np.min([1, np.exp(proposal_joint_log_likelihood - previous_joint_log_likelihood)])
            u = np.random.uniform(low=0.0, high=1.0)

            if u < acceptance_prob:
                self.parameters = parameter_proposal
                self.history.append(self.model.get_parameter_values() | {"mean":mean, "cov":cov, "log_marginal_likelihood":log_marginal_likelihood})
            else:
                self.model.set_parameter_values(**self.parameters)
                self.history.append(self.history[-1])

    def fit(self, y, X, Xeval, n_samples=10, n_intervals=20):

        # Set training variables:
        self.set_training_variables(y, X, Xeval)

        # Initialise sampling history:
        self.initialise()

        proposal_intervals = self.proposal_grid(low=Xeval.min(), high=Xeval.max(), n=n_intervals)

        # Start sampling:
        for _ in range(n_samples):

            self.sample_subordinator(proposal_intervals)

            self.sample()

        return self.history


