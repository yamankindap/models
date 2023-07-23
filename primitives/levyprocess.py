import numpy as np
from primitives.utils import incgammal

from primitives.parameters import ParameterInterface

# A Levy process is specifically defined as a 1 dimensional stochastic process. 
# The multidimensional case will be handled under a separate LevyField object that take LevyProcess objects as inputs.
class LevyProcess(ParameterInterface):
    parameter_keys = None

    def __len__(self):
        return 1

    def set_name(self, name):
        if not hasattr(self, 'name'):
            self.name = name
        else:
            print('The process is already named as {}.'.format(self.name))

    def stochastic_integral(self, evaluation_points, t_series, x_series):
        """This method assumes that the evaluation_points is 1 dimensional with shape (N, 1).
        Returns an array of same shape.

        This needs to be modified for multidimensional X matrices.
        """
        W = [x_series[t_series<point].sum() for point in evaluation_points]
        return np.array(W).reshape(evaluation_points.shape[0], -1)
    
    # The following methods are required to treat a LevyProcess as a proposal object.

    def likelihood(self, x_prime, x):
        """ The density g( \dot | \theta ) evaluated at \theta^\prime.
        """
        pass

    def log_likelihood(self, x_prime, x):
        """ The log density g( \dot | \theta ) evaluated at \theta^\prime.
        """
        pass

    def propose(self, x, shape):
        pass


class GammaProcess(LevyProcess):
    parameter_keys = ["beta", "C"]

    def __init__(self, **kwargs):
        # Set parameters using the ParameterInterface parent.
        super().__init__(**kwargs)

        # Set sampling hyperparameters.
        self.set_hyperparameters(M=10, tolerance=0.01, pt=0.05)

    def set_hyperparameters(self, M, tolerance, pt):
        """The simulation parameters are treated as hyperparameters for subordinator simulation.
        """
        self.M = M
        self.tolerance = tolerance
        self.pt = pt
    
    def h_gamma(self, gamma):
        return 1/(self.beta*(np.exp(gamma/self.C)-1))

    def unit_expected_residual(self, c):
        return (self.C/self.beta)*incgammal(1, self.beta*c)

    def unit_variance_residual(self, c):
        return (self.C/self.beta**2)*incgammal(2, self.beta*c)
    
    def simulate_from_series_representation(self, rate=1.0, M=100, gamma_0=0.0):
        gamma_sequence = np.random.exponential(scale=1/rate, size=M)
        gamma_sequence[0] += gamma_0 
        gamma_sequence = gamma_sequence.cumsum()
        x_series = self.h_gamma(gamma_sequence)
        thinning_function = (1+self.beta*x_series)*np.exp(-self.beta*x_series)
        u = np.random.uniform(low=0.0, high=1.0, size=x_series.size)
        x_series = x_series[u < thinning_function]
        return gamma_sequence, x_series
    
    def simulate_adaptively_truncated_jump_series(self, rate=1.0):
        # Adaptive truncation is based on Theorem 3 of Kindap, Godsill 2023 (the theorem number may change in publication). 
        # Gaussian approximation of the residual is not valid for the Gamma process.
        gamma_sequence, x_series = self.simulate_from_series_representation(rate=rate, M=self.M, gamma_0=0.0)
        truncation_level = self.h_gamma(gamma_sequence[-1])
        residual_expected_value = rate*self.unit_expected_residual(truncation_level)
        residual_variance = rate*self.unit_variance_residual(truncation_level)
        E_c = self.tolerance*x_series.sum()

        while (residual_variance/((E_c - residual_expected_value)**2) > self.pt) or (E_c < residual_expected_value):
            gamma_sequence_extension, x_series_extension = self.simulate_from_series_representation(rate=rate, M=self.M, gamma_0=gamma_sequence[-1])
            gamma_sequence = np.concatenate((gamma_sequence, gamma_sequence_extension))
            x_series = np.concatenate((x_series, x_series_extension))
            truncation_level = self.h_gamma(gamma_sequence[-1])
            residual_expected_value = rate*self.unit_expected_residual(truncation_level)
            residual_variance = rate*self.unit_variance_residual(truncation_level)
            E_c = self.tolerance*x_series.sum()
        return x_series, truncation_level
    
    def simulate_points(self, rate, low, high):
        """Returns the times and jump sizes associated with the point process representation of a Levy process. Returns the points.
        """
        x_series, truncation_level = self.simulate_adaptively_truncated_jump_series(rate=rate)
        t_series = np.random.uniform(low=low, high=high, size=x_series.shape)
        return t_series, x_series  
    
    def initialise_proposal_samples(self, low, high):
        """Simulates random times and jumps sizes from the prior over the whole evaluation input space.
        """
        self.t_series, self.x_series = self.simulate_points(rate=(high-low), low=low, high=high) 

    def set_proposal_samples(self, proposal_t_series, proposal_x_series):
        self.t_series = proposal_t_series
        self.x_series = proposal_x_series

    def propose_subordinator(self, proposal_interval):
        """On the given interval, removes the previous points and simulates new random points. Returns the proposal points.
        """
        conditioning_x_series = self.x_series[(proposal_interval[1] < self.t_series)]
        conditioning_x_series = np.concatenate((conditioning_x_series, self.x_series[(self.t_series < proposal_interval[0])]))
        conditioning_t_series = self.t_series[(proposal_interval[1] < self.t_series)]
        conditioning_t_series = np.concatenate((conditioning_t_series, self.t_series[(self.t_series < proposal_interval[0])]))
        proposed_t_series, proposed_x_series = self.simulate_points(rate=(proposal_interval[1]-proposal_interval[0]), low=proposal_interval[0], high=proposal_interval[1])
        proposal_x_series = np.concatenate((conditioning_x_series, proposed_x_series))
        proposal_t_series = np.concatenate((conditioning_t_series, proposed_t_series))
        return proposal_t_series, proposal_x_series
    
    def likelihood(self, x_prime, x):
        """ The likelihood should consider both the poisson process likelihood of the times and the joint jump density. 
        """
        pass

    def log_likelihood(self, x_prime, x):
        """ The log density g( \dot | \theta ) evaluated at \theta^\prime.
        """
        pass

    def propose(self, x, shape):
        pass


# The Stable process class is only available as an edge parameter setting of the Tempered Stable class.
# Future versions will consider the full implementation of a stable process as a subordinator process.
class StableProcess(LevyProcess):
    parameter_keys = ["alpha", "C"]

    def __init__(self, **kwargs):
        # Set parameters using the ParameterInterface parent.
        super().__init__(**kwargs)

        # Set sampling hyperparameters.
        self.check_parameter_constraints()

    def check_parameter_constraints(self):
        if (self.alpha >= 1):
            raise ValueError('The alpha parameter is set to greater than or equal to 1.')

    def h_stable(self, gamma):
        return np.power((self.alpha/self.C)*gamma, np.divide(-1,self.alpha))
    
    def unit_expected_residual(self, c):
        return (self.C/(1-self.alpha))*(c**(1-self.alpha))

    def unit_variance_residual(self, c):
        return (self.C/(2-self.alpha))*(c**(2-self.alpha))

    def simulate_from_series_representation(self, rate=1.0, M=1000, gamma_0=0.0):
        gamma_sequence = np.random.exponential(scale=1/rate, size=M)
        gamma_sequence[0] += gamma_0 
        gamma_sequence = gamma_sequence.cumsum()
        x_series = self.h_stable(gamma_sequence)
        return gamma_sequence, x_series


class TemperedStableProcess(LevyProcess):
    parameter_keys = ["alpha", "beta", "C"]

    def __init__(self, **kwargs):
        # Set parameters using the ParameterInterface parent.
        super().__init__(**kwargs)

        # Set sampling hyperparameters.
        self.set_hyperparameters(M=100, tolerance=0.01, pt=0.05)

    def set_hyperparameters(self, M, tolerance, pt):
        """The simulation parameters are treated as hyperparameters for subordinator simulation.
        """
        self.M = M
        self.tolerance = tolerance
        self.pt = pt
        self.set_residual_approximation_method(mode="Gaussian")

    def h_stable(self, gamma):
        return np.power((self.alpha/self.C)*gamma, np.divide(-1,self.alpha))

    def unit_expected_residual(self, c):
        return (self.C*self.beta**(self.alpha-1))*incgammal(1-self.alpha, self.beta*c)

    def unit_variance_residual(self, c):
        return (self.C*self.beta**(self.alpha-2))*incgammal(2-self.alpha, self.beta*c)

    # Residual approximation:
    def set_residual_approximation_method(self, mode):
        if mode is None:
            print('Residual approximation mode is set to add the expected residual value.')
            self.simulate_residual = self.simulate_residual_drift
        elif mode == 'mean-only':
            print('Residual approximation mode is set to add the expected residual value.')
            self.simulate_residual = self.simulate_residual_drift
        elif mode == 'Gaussian':
            print('Residual approximation mode is set to Gaussian approximation.')
            self.simulate_residual = self.simulate_residual_gaussians
        else:
            raise ValueError('The mode can only be set to `mean-only` or `Gaussian`.')
        
    def residual_stats(self, rate, truncation_level):
        R_mu = rate*self.unit_expected_residual(truncation_level)
        R_var = rate*self.unit_variance_residual(truncation_level)
        return R_mu, R_var

    def simulate_residual_gaussians(self, low, high, truncation_level, size):
        R_mu = (high-low)*self.unit_expected_residual(truncation_level)
        R_var = (high-low)*self.unit_variance_residual(truncation_level)

        t_series = np.linspace(low, high, num=size)
        delta = t_series[1] - t_series[0]

        residual_jumps = np.random.normal(loc=delta * R_mu, scale=np.sqrt(delta * R_var), size=size-1)
        return t_series[1:].flatten(), residual_jumps

    def simulate_residual_drift(self, low, high, truncation_level, size):
        R_mu = (high-low)*self.unit_expected_residual(truncation_level)

        t_series = np.linspace(low, high, num=size+1) # This series includes 0, which is later removed.
        delta = t_series[1] - t_series[0]

        residual_jumps = np.random.normal(loc=delta * R_mu, scale=0, size=size)
        return t_series[1:].flatten(), residual_jumps

    # Simulation functions:
    def simulate_from_series_representation(self, rate=1.0, M=100, gamma_0=0.0):
        gamma_sequence = np.random.exponential(scale=1/rate, size=M)
        gamma_sequence[0] += gamma_0 
        gamma_sequence = gamma_sequence.cumsum()
        x_series = self.h_stable(gamma_sequence)
        thinning_function = np.exp(-self.beta*x_series)
        u = np.random.uniform(low=0.0, high=1.0, size=x_series.size)
        x_series = x_series[u < thinning_function]
        return gamma_sequence, x_series
    
    # This function doesn't work properly thus there is a max_iter variable. 
    # The probabilistic bound may be readjusted to include the fact that the residual mean will be approximated.
    def simulate_adaptively_truncated_jump_series(self, rate=1.0):
        gamma_sequence, x_series = self.simulate_from_series_representation(rate=rate, M=self.M, gamma_0=0.0)
        truncation_level = self.h_stable(gamma_sequence[-1])
        residual_expected_value = rate*self.unit_expected_residual(truncation_level)
        residual_variance = rate*self.unit_variance_residual(truncation_level)
        E_c = self.tolerance*x_series.sum() + residual_expected_value
        max_iter = 50
        idx = 0
        while (residual_variance/((E_c - residual_expected_value)**2) > self.pt) or (E_c < residual_expected_value):
            gamma_sequence_extension, x_series_extension = self.simulate_from_series_representation(rate=rate, M=self.M, gamma_0=gamma_sequence[-1])
            gamma_sequence = np.concatenate((gamma_sequence, gamma_sequence_extension))
            x_series = np.concatenate((x_series, x_series_extension))
            truncation_level = self.h_stable(gamma_sequence[-1])
            residual_expected_value = rate*self.unit_expected_residual(truncation_level)
            residual_variance = rate*self.unit_variance_residual(truncation_level)
            E_c = self.tolerance*x_series.sum() + residual_expected_value

            idx += 1
            if idx > max_iter:
                print('Max iter reached.')
                break
        return x_series, truncation_level
    
    def simulate_points(self, rate, low, high):
        """Returns the times and jump sizes associated with the point process representation of a Levy process. Returns the points.
        """
        x_series, truncation_level = self.simulate_adaptively_truncated_jump_series(rate=rate)
        t_series = np.random.uniform(low=low, high=high, size=x_series.shape)
        residual_t_series, residual_jumps = self.simulate_residual(low=low, high=high, truncation_level=truncation_level, size=x_series.size)
        return np.concatenate((t_series, residual_t_series)), np.concatenate((x_series, residual_jumps))    

    def initialise_proposal_samples(self, low, high):
        """Simulates random times and jumps sizes from the prior over the whole evaluation input space.
        """
        self.t_series, self.x_series = self.simulate_points(rate=(high-low), low=low, high=high) 

    def set_proposal_samples(self, proposal_t_series, proposal_x_series):
        self.t_series = proposal_t_series
        self.x_series = proposal_x_series

    def propose_subordinator(self, proposal_interval):
        """On the given interval, removes the previous points and simulates new random points. Returns the proposal points.
        """
        conditioning_x_series = self.x_series[(proposal_interval[1] < self.t_series)]
        conditioning_x_series = np.concatenate((conditioning_x_series, self.x_series[(self.t_series < proposal_interval[0])]))
        conditioning_t_series = self.t_series[(proposal_interval[1] < self.t_series)]
        conditioning_t_series = np.concatenate((conditioning_t_series, self.t_series[(self.t_series < proposal_interval[0])]))
        proposed_t_series, proposed_x_series = self.simulate_points(rate=(proposal_interval[1]-proposal_interval[0]), low=proposal_interval[0], high=proposal_interval[1])
        proposal_x_series = np.concatenate((conditioning_x_series, proposed_x_series))
        proposal_t_series = np.concatenate((conditioning_t_series, proposed_t_series))
        return proposal_t_series, proposal_x_series
    
    def likelihood(self, x_prime, x):
        """ The likelihood should consider both the poisson process likelihood of the times and the joint jump density. 
        """
        pass

    def log_likelihood(self, x_prime, x):
        """ The log density g( \dot | \theta ) evaluated at \theta^\prime.
        """
        pass

    def propose(self, x, shape):
        pass