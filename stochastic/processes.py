import numpy as np

from scipy.special import kv
from scipy.special import gamma as gammafnc
from scipy.special import gammainc, gammaincc, gammaincinv
from scipy.special import hankel1, hankel2

from primitive.utils import incgammal
from primitive.utils import incgammau

from primitive.stochastics import LevyProcess

class GammaProcess(LevyProcess):
    """The gamma process class produces random jump times and sizes of a gamma Levy process.
    It is designed to produce multiple realisations with a single call. 
    The main method is .simulate_points() which returns random jump times and sizes for a given number of realisations.
    The number of realisations are assumed to be equal to the number of particles (as in particle filtering).
    The default single realisation shape is (1, M) where M is the number of jump times and sizes.
    This shape is generalised to (n_particles, M) for multiple realisation versions.
    """
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
    
    def simulate_from_series_representation(self, rate=1.0, M=100, gamma_0=0.0, size=1):
        gamma_sequence = np.random.exponential(scale=1/rate, size=(size,M))
        gamma_sequence[:,0] += gamma_0
        gamma_sequence = gamma_sequence.cumsum(axis=1)

        x_series = self.h_gamma(gamma_sequence)
        thinning_function = (1+self.beta*x_series)*np.exp(-self.beta*x_series)
        u = np.random.uniform(low=0.0, high=1.0, size=x_series.shape)
        x_series[u > thinning_function] = 0.
        return gamma_sequence, x_series
    
    def simulate_adaptively_truncated_jump_series(self, rate=1.0, size=1):
        # Adaptive truncation is based on Theorem 3 of Kindap, Godsill 2023 (the theorem number may change in publication). 
        # Gaussian approximation of the residual is not valid for the Gamma process.
        gamma_sequence, x_series = self.simulate_from_series_representation(rate, M=self.M, gamma_0=0., size=size)

        truncation_level = self.h_gamma(gamma_sequence[:,-1])

        residual_expected_value = rate*self.unit_expected_residual(truncation_level)
        residual_variance = rate*self.unit_variance_residual(truncation_level)
        E_c = self.tolerance*x_series.sum(axis=1)

        condition1 = (residual_variance/((E_c - residual_expected_value)**2) > self.pt)
        condition2 = (E_c < residual_expected_value)

        while condition1.any() or condition2.any():
            gamma_sequence_extension, x_series_extension = self.simulate_from_series_representation(rate=rate, M=self.M, gamma_0=gamma_sequence[:,-1], size=size)
            gamma_sequence = np.concatenate((gamma_sequence, gamma_sequence_extension), axis=1)
            x_series = np.concatenate((x_series, x_series_extension), axis=1)
            truncation_level = self.h_gamma(gamma_sequence[:,-1])
            residual_expected_value = rate*self.unit_expected_residual(truncation_level)
            residual_variance = rate*self.unit_variance_residual(truncation_level)
            E_c = self.tolerance*x_series.sum(axis=1)

            condition1 = (residual_variance/((E_c - residual_expected_value)**2) > self.pt)
            condition2 = (E_c < residual_expected_value)
            
        return x_series, truncation_level
    
    def simulate_points(self, rate, low, high, n_particles=1):
        """Returns the times and jump sizes associated with the point process representation of a Levy process. Returns the points.
        """
        x_series, truncation_level = self.simulate_adaptively_truncated_jump_series(rate=rate, size=n_particles)
        t_series = np.random.uniform(low=low, high=high, size=x_series.shape)
        return t_series, x_series  
    
    def initialise_proposal_samples(self, low, high, n_particles=1):
        """Simulates random times and jumps sizes from the prior over the whole evaluation input space.
        """
        self.t_series, self.x_series = self.simulate_points(rate=(high-low), low=low, high=high, n_particles=n_particles) 

    def set_proposal_samples(self, proposal_t_series, proposal_x_series):
        self.t_series = proposal_t_series
        self.x_series = proposal_x_series

    def propose_subordinator(self, proposal_interval):
        """On the given interval, removes the previous points and simulates new random points. Returns the proposal points.
        NOTE that this function only works when there is a single particle under consideration. This may be fixed by
        finding a method to keep the shape of x_series constant while applying conditional selection.
        """
        conditioning_x_series = self.x_series[(proposal_interval[1] < self.t_series)]
        conditioning_x_series = np.concatenate((conditioning_x_series, self.x_series[(self.t_series < proposal_interval[0])])).reshape(1, -1)
        conditioning_t_series = self.t_series[(proposal_interval[1] < self.t_series)]
        conditioning_t_series = np.concatenate((conditioning_t_series, self.t_series[(self.t_series < proposal_interval[0])])).reshape(1, -1)
        proposed_t_series, proposed_x_series = self.simulate_points(rate=(proposal_interval[1]-proposal_interval[0]), low=proposal_interval[0], high=proposal_interval[1])
        proposal_x_series = np.concatenate((conditioning_x_series, proposed_x_series), axis=1)
        proposal_t_series = np.concatenate((conditioning_t_series, proposed_t_series), axis=1)  
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
    
    def simulate_from_series_representation(self, rate=1.0, M=1000, gamma_0=0.0, size=1):
        gamma_sequence = np.random.exponential(scale=1/rate, size=(size,M))
        gamma_sequence[:,0] += gamma_0
        gamma_sequence = gamma_sequence.cumsum(axis=1)
        x_series = self.h_stable(gamma_sequence)
        return gamma_sequence, x_series
    
    def simulate_points(self, rate, low, high, n_particles=1):
        """Returns the times and jump sizes associated with the point process representation of a Levy process. Returns the points.
        """
        gamma_sequence, x_series = self.simulate_from_series_representation(rate=rate, size=n_particles)
        t_series = np.random.uniform(low=low, high=high, size=x_series.shape)
        return t_series, x_series  


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

    def simulate_residual_gaussians(self, low, high, truncation_level, shape):
        n_samples = 50
        # n_samples = shape[1]

        R_mu = (high-low)*self.unit_expected_residual(truncation_level)
        R_var = (high-low)*self.unit_variance_residual(truncation_level)

        t_series = np.linspace(low, high, num=n_samples+1) # This series includes 0, which is later removed.
        delta = t_series[1] - t_series[0]

        residual_jumps = np.random.normal(loc=delta * R_mu, scale=np.sqrt(delta * R_var), size=(n_samples, R_mu.shape[0])).T

        # Broadcast linspaced times to number of particles.
        t_series = np.broadcast_to(t_series[1:][np.newaxis], shape=(shape[0], n_samples))

        return t_series, residual_jumps

    def simulate_residual_drift(self, low, high, truncation_level, shape):
        R_mu = (high-low)*self.unit_expected_residual(truncation_level)

        t_series = np.linspace(low, high, num=shape[1]+1) # This series includes 0, which is later removed.
        delta = t_series[1] - t_series[0]

        residual_jumps = np.random.normal(loc=delta * R_mu, scale=0, size=(shape[1], R_mu.shape[0])).T

        # Broadcast linspaced times to number of particles.
        t_series = np.broadcast_to(t_series[1:][np.newaxis], shape=(shape[0],shape[1]))

        return t_series, residual_jumps

    # Simulation functions:
    def simulate_from_series_representation(self, rate=1.0, M=100, gamma_0=0.0, size=1):
        gamma_sequence = np.random.exponential(scale=1/rate, size=(size,M))
        gamma_sequence[:,0] += gamma_0
        gamma_sequence = gamma_sequence.cumsum(axis=1)

        x_series = self.h_stable(gamma_sequence)
        thinning_function = np.exp(-self.beta*x_series)
        u = np.random.uniform(low=0.0, high=1.0, size=x_series.shape)
        x_series[u > thinning_function] = 0.
        return gamma_sequence, x_series

    def simulate_adaptively_truncated_jump_series(self, rate=1.0, size=1):
        # Adaptive truncation is based on Theorem 3 of Kindap, Godsill 2023 (the theorem number may change in publication). 
        # Gaussian approximation of the residual is not valid for the Gamma process.
        gamma_sequence, x_series = self.simulate_from_series_representation(rate, M=self.M, gamma_0=0., size=size)
        truncation_level = self.h_stable(gamma_sequence[:,-1])

        residual_expected_value = rate*self.unit_expected_residual(truncation_level)
        residual_variance = rate*self.unit_variance_residual(truncation_level)

        E_c = self.tolerance*x_series.sum(axis=1) + residual_expected_value

        max_iter = 50
        idx = 0
        condition1 = (residual_variance/((E_c - residual_expected_value)**2) > self.pt) 
        condition2 = (E_c < residual_expected_value)

        while condition1.any() or condition2.any():
            gamma_sequence_extension, x_series_extension = self.simulate_from_series_representation(rate=rate, M=self.M, gamma_0=gamma_sequence[:,-1], size=size)
            gamma_sequence = np.concatenate((gamma_sequence, gamma_sequence_extension), axis=1)
            x_series = np.concatenate((x_series, x_series_extension), axis=1)
            truncation_level = self.h_stable(gamma_sequence[:,-1])
            residual_expected_value = rate*self.unit_expected_residual(truncation_level)
            residual_variance = rate*self.unit_variance_residual(truncation_level)
            E_c = self.tolerance*x_series.sum(axis=1) + residual_expected_value

            condition1 = (residual_variance/((E_c - residual_expected_value)**2) > self.pt)
            condition2 = (E_c < residual_expected_value)

            idx += 1
            if idx > max_iter:
                print('Max iter reached.')
                break
            
        return x_series, truncation_level
    
    def simulate_points(self, rate, low, high, n_particles=1):
        """Returns the times and jump sizes associated with the point process representation of a Levy process. Returns the points.
        """
        x_series, truncation_level = self.simulate_adaptively_truncated_jump_series(rate=rate, size=n_particles)
        t_series = np.random.uniform(low=low, high=high, size=x_series.shape)
        residual_t_series, residual_jumps = self.simulate_residual(low=low, high=high, truncation_level=truncation_level, shape=x_series.shape)

        return np.concatenate((t_series, residual_t_series), axis=1), np.concatenate((x_series, residual_jumps),  axis=1)


    def initialise_proposal_samples(self, low, high, n_particles=1):
        """Simulates random times and jumps sizes from the prior over the whole evaluation input space.
        """
        self.t_series, self.x_series = self.simulate_points(rate=(high-low), low=low, high=high, n_particles=n_particles)

    def set_proposal_samples(self, proposal_t_series, proposal_x_series):
        self.t_series = proposal_t_series
        self.x_series = proposal_x_series


    def propose_subordinator(self, proposal_interval):
        """On the given interval, removes the previous points and simulates new random points. Returns the proposal points.
        NOTE that this function only works when there is a single particle under consideration. This may be fixed by
        finding a method to keep the shape of x_series constant while applying conditional selection.
        """

        conditioning_x_series = self.x_series[(proposal_interval[1] < self.t_series)]
        conditioning_x_series = np.concatenate((conditioning_x_series, self.x_series[(self.t_series < proposal_interval[0])])).reshape(1, -1)
        conditioning_t_series = self.t_series[(proposal_interval[1] < self.t_series)]
        conditioning_t_series = np.concatenate((conditioning_t_series, self.t_series[(self.t_series < proposal_interval[0])])).reshape(1, -1)
        proposed_t_series, proposed_x_series = self.simulate_points(rate=(proposal_interval[1]-proposal_interval[0]), low=proposal_interval[0], high=proposal_interval[1])
        proposal_x_series = np.concatenate((conditioning_x_series, proposed_x_series), axis=1)
        proposal_t_series = np.concatenate((conditioning_t_series, proposed_t_series), axis=1)

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


class GeneralisedInverseGaussianProcess(LevyProcess):
    parameter_keys = ["lam", "gamma", "delta"]

    def __init__(self, **kwargs):
        # Set parameters using the ParameterInterface parent.
        super().__init__(**kwargs)

        self.abs_lam = np.abs(kwargs["lam"])

        # Set sampling hyperparameters.
        self.set_hyperparameters(M_gamma=10, M_stable=100, tolerance=0.01, pt=0.05)
        self.max_iter = 50 # This can also be given as an argument.

        # The parameter values assigned here are temporary.
        self.gamma_process = GammaProcess(**{"beta":np.float64(1.), "C":np.float64(1.)})
        self.gamma_process2 = GammaProcess(**{"beta":np.float64(1.), "C":np.float64(1.)})
        self.tempered_stable_process = TemperedStableProcess(**{"alpha":np.float64(0.6), "beta":np.float64(0.01), "C":np.float64(1.)})

        # Define a third gamma process for the positive lam extension
        if (self.lam > 0):
            C = np.float64(self.lam)
            beta = np.float64(0.5*self.gamma**2)
            self.pos_ext_gamma_process = GammaProcess(**{"beta":beta, "C":C})

        self.set_simulation_method()
        self.set_residual_approximation_method(mode="Gaussian")

    def set_hyperparameters(self, M_gamma, M_stable, tolerance, pt):
        """The simulation parameters are treated as hyperparameters for subordinator simulation.
        """
        self.M_gamma = M_gamma
        self.M_stable = M_stable
        self.tolerance = tolerance
        self.pt = pt

    # Residual approximation module:
    def _exact_residual_stats(self, rate, truncation_level_gamma, truncation_level_TS):
        residual_expected_value_GIG = rate*self.tempered_stable_process.unit_expected_residual(truncation_level_TS)
        residual_variance_GIG = rate*self.tempered_stable_process.unit_variance_residual(truncation_level_TS)
        return residual_expected_value_GIG, residual_variance_GIG
    
    def _lower_bound_residual_stats(self, rate, truncation_level_gamma, truncation_level_TS):
        residual_expected_value_GIG = rate*self.lb_gamma_process.unit_expected_residual(truncation_level_gamma) + rate*self.lb_tempered_stable_process.unit_expected_residual(truncation_level_TS)
        residual_variance_GIG = rate*self.lb_gamma_process.unit_variance_residual(truncation_level_gamma) + rate*self.lb_tempered_stable_process.unit_variance_residual(truncation_level_TS)
        return residual_expected_value_GIG, residual_variance_GIG

    # Define the related function using mean or gaussian here....
    def set_residual_approximation_method(self, mode):
        if (self.abs_lam >= 0.5):
            if (self.abs_lam == 0.5):
                print('Residual approximation method is set to exact method.')
                self.residual_stats = self._exact_residual_stats
            else:
                # Initialise the lower bounding point processes for residual approximation
                print('Residual approximation method is set to lower bounding method.')
                z0 = self.cornerpoint()
                H0 = z0*self.H_squared(z0)
                C_gamma_B = np.float64(z0/((np.pi**2)*H0*self.abs_lam))
                beta_gamma_B = np.float64(0.5*self.gamma**2 + (self.abs_lam/(1+self.abs_lam))*(z0**2)/(2*self.delta**2))
                self.lb_gamma_process = GammaProcess(**{"beta":beta_gamma_B, "C":C_gamma_B})
                beta_0 = np.float64(1.95) # This parameter value can be optimised further in the future...
                C_TS_B = np.float64((2*self.delta*np.sqrt(np.e)*np.sqrt(beta_0-1))/((np.pi**2)*H0*beta_0))
                beta_TS_B = np.float64(0.5*self.gamma**2 + (beta_0*z0**2)/(2*self.delta**2))
                self.lb_tempered_stable_process = TemperedStableProcess(**{"alpha":0.5, "beta":beta_TS_B, "C":C_TS_B})
                # Select the appropriate residual_gaussian_sequence() function
                self.residual_stats = self._lower_bound_residual_stats
        else:
            print('Residual approximation method is set to lower bounding method.')
            z1 = self.cornerpoint()
            C_gamma_A = np.float64(z1/(2*np.pi*self.abs_lam))
            beta_gamma_A = np.float64(0.5*self.gamma**2 + (self.abs_lam/(1+self.abs_lam))*(z1**2)/(2*self.delta**2))
            self.lb_gamma_process = GammaProcess(**{"beta":beta_gamma_A, "C":C_gamma_A})
            beta_0 = np.float64(1.95) # This parameter value can be optimised further in the future...
            C_TS_A = np.float64((self.delta*np.sqrt(np.e)*np.sqrt(beta_0-1))/(np.pi*beta_0))
            beta_TS_A = np.float64(0.5*self.gamma**2 + (beta_0*z1**2)/(2*self.delta**2))
            self.lb_tempered_stable_process = TemperedStableProcess(**{"alpha":0.5, "beta":beta_TS_A, "C":C_TS_A})
            self.residual_stats = self._lower_bound_residual_stats

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

    def simulate_residual_gaussians(self, low, high, truncation_level_gamma, truncation_level_TS, shape):
        n_samples = 50
        # n_samples = shape[1]

        R_mu, R_var = self.residual_stats((high-low), truncation_level_gamma, truncation_level_TS)

        t_series = np.linspace(low, high, num=n_samples+1) # This series includes 0, which is later removed.
        delta = t_series[1] - t_series[0]

        residual_jumps = np.random.normal(loc=delta * R_mu, scale=np.sqrt(delta * R_var), size=(n_samples, R_mu.shape[0])).T

        # Broadcast linspaced times to number of particles.
        t_series = np.broadcast_to(t_series[1:][np.newaxis], shape=(shape[0], n_samples))
        
        return t_series, residual_jumps

    def simulate_residual_drift(self, low, high, truncation_level_gamma, truncation_level_TS, shape):
        R_mu, R_var = self.residual_stats((high-low), truncation_level_gamma, truncation_level_TS)

        t_series = np.linspace(low, high, num=shape[1]+1)
        delta = t_series[1] - t_series[0]

        residual_jumps = np.random.normal(loc=delta * R_mu, scale=0, size=(shape[1], R_mu.shape[0])).T

        # Broadcast linspaced times to number of particles.
        t_series = np.broadcast_to(t_series[1:][np.newaxis], shape=(shape[0],shape[1]))

        return t_series, residual_jumps

    # Auxiliary functionality
    def cornerpoint(self):
        return np.power(np.power(float(2), 1-2*self.abs_lam)*np.pi/np.power(gammafnc(self.abs_lam), 2), 1/(1-2*self.abs_lam))

    def H_squared(self, z):
        return np.real(hankel1(self.abs_lam, z)*hankel2(self.abs_lam, z))

    def probability_density(self, x):
        return np.power(self.gamma/self.delta, self.lam)*(1/(2*kv(self.lam, self.delta*self.gamma))*np.power(x, self.lam-1)*np.exp(-(self.gamma**2*x+self.delta**2/x)/2))

    def random_sample(self, size):
        def thinning_function(delta, x):
            return np.exp(-(1/2)*(np.power(delta, 2)*(1/x)))
        def reciprocal_sample(x, i):
            return x**(i)
        def random_GIG(lam, gamma, delta, size=1):
            i = 1
            if lam < 0:
                tmp = gamma
                gamma = delta
                delta = tmp
                lam = -lam
                i = -1
            shape = lam
            scale = 2/np.power(gamma, 2)
            gamma_rv = np.random.gamma(shape=shape, scale=scale, size=size)
            u = np.random.uniform(low=0.0, high=1.0, size=size)
            sample = gamma_rv[u < thinning_function(delta, gamma_rv)]
            return reciprocal_sample(sample, i)
        sample = np.array([])
        while sample.size < size:
            sample = np.concatenate((sample, random_GIG(self.lam, self.gamma, self.delta, size=size)))
        return sample[np.random.randint(low=0, high=sample.size, size=size)]

    # Select simulation method and set corresponding parameters
    def _simulate_with_positive_extension(self, rate, size=1):
        x_series, truncation_level = self.simulate_Q_GIG(rate=rate, size=size)
        x_P_series = self.simulate_adaptive_positive_extension_series(rate=rate, size=size)
        return np.concatenate((x_series, x_P_series), axis=1), truncation_level

    def set_simulation_method(self, method=None):
        # Automatically select a method for simulation
        if method is None:
            if (self.abs_lam >= 0.5):
                if (self.gamma == 0) or (self.abs_lam == 0.5):
                    print('Simulation method is set to GIG paper version.')
                    # Set parameters of the tempered stable process...
                    alpha = np.float64(0.5)
                    C = np.float64(self.delta*gammafnc(0.5)/(np.sqrt(2)*np.pi))
                    beta = np.float64(0.5*self.gamma**2)

                    if (self.gamma == 0):
                        print('The dominating point process is set as a stable process.')
                        self.tempered_stable_process = StableProcess(**{"alpha":alpha, "C":C})
                    else:
                        self.tempered_stable_process.set_parameter_values(**{"alpha":alpha, "beta":beta, "C":C})

                    self.simulate_Q_GIG = self.simulate_adaptive_series_setting_1
                    if (self.lam > 0):
                        print('An independent gamma process extension will be made.')
                        self.simulate_jumps = self._simulate_with_positive_extension
                    else:
                        self.simulate_jumps = self.simulate_Q_GIG
                else:
                    print('Simulation method is set to improved version.')
                    # Set parameters of the two gamma and one TS processes...
                    z1 = self.cornerpoint()
                    C1 = np.float64(z1/(np.pi*self.abs_lam*2*(1+self.abs_lam)))
                    beta1 = np.float64(0.5*self.gamma**2)
                    self.gamma_process.set_parameter_values(**{"beta":beta1, "C":C1})

                    C2 = np.float64(z1/(np.pi*2*(1+self.abs_lam)))
                    beta2 = np.float64(0.5*self.gamma**2 + (z1**2)/(2*self.delta**2))
                    self.gamma_process2.set_parameter_values(**{"beta":beta2, "C":C2})

                    C = np.float64(self.delta/(np.sqrt(2*np.pi)))
                    alpha = np.float64(0.5)
                    beta = np.float64(0.5*self.gamma**2 + (z1**2)/(2*self.delta**2))
                    self.tempered_stable_process.set_parameter_values(**{"alpha":alpha, "beta":beta, "C":C})

                    self.simulate_Q_GIG = self.simulate_adaptive_combined_series_setting_1
                    if (self.lam > 0):
                        print('An independent gamma process extension will be made.')
                        self.simulate_jumps = self._simulate_with_positive_extension
                    else:
                        self.simulate_jumps = self.simulate_Q_GIG
            else:
                print('Simulation method is set to improved version for 0 < |lam| < 0.5.')
                # Set parameters of the two gamma and one TS processes...
                z0 = self.cornerpoint()
                H0 = z0*self.H_squared(z0)
                C1 = np.float64(z0/((np.pi**2)*H0*self.abs_lam*(1+self.abs_lam)))
                beta1 = np.float64(0.5*self.gamma**2)
                self.gamma_process.set_parameter_values(**{"beta":beta1, "C":C1})

                C2 = np.float64(z0/((np.pi**2)*(1+self.abs_lam)*H0))
                beta2 = np.float64(0.5*self.gamma**2 + (z0**2)/(2*self.delta**2))
                self.gamma_process2.set_parameter_values(**{"beta":beta2, "C":C2})

                C = np.float64(np.sqrt(2*self.delta**2)*gammafnc(0.5)/(H0*np.pi**2))
                alpha = np.float64(0.5)
                beta = np.float64(0.5*self.gamma**2)
                self.tempered_stable_process.set_parameter_values(**{"alpha":alpha, "beta":beta, "C":C})

                self.simulate_Q_GIG = self.simulate_adaptive_combined_series_setting_2
                if (self.lam > 0):
                    print('An independent gamma process extension will be made.')
                    self.simulate_jumps = self._simulate_with_positive_extension
                else:
                    self.simulate_jumps = self.simulate_Q_GIG
        else:
            raise ValueError('The manual selection functionality for simulation method is NOT implemented.')
        
    # Positive lam extension module
    def simulate_adaptive_positive_extension_series(self, rate=1.0, size=1):
        gamma_sequence, x_series = self.pos_ext_gamma_process.simulate_from_series_representation(rate=rate, M=self.M_gamma, gamma_0=0., size=size)

        truncation_level = self.pos_ext_gamma_process.h_gamma(gamma_sequence[:,-1])
        residual_expected_value = rate*self.pos_ext_gamma_process.unit_expected_residual(truncation_level)
        residual_variance = rate*self.pos_ext_gamma_process.unit_variance_residual(truncation_level)
        E_c = self.tolerance*x_series.sum(axis=1)
        itr = 1

        condition1 = (residual_variance/((E_c - residual_expected_value)**2) > self.pt)
        condition2 = (E_c < residual_expected_value)

        while condition1.any() or condition2.any():
            
            # Debug code:
            if (itr > self.max_iter):
                #print('Max iteration reached.')
                break

            gamma_sequence_extension, x_series_extension = self.pos_ext_gamma_process.simulate_from_series_representation(rate=rate, M=self.M_gamma, gamma_0=gamma_sequence[:,-1], size=size)
            gamma_sequence = np.concatenate((gamma_sequence, gamma_sequence_extension), axis=1)
            x_series = np.concatenate((x_series, x_series_extension), axis=1)

            truncation_level = self.pos_ext_gamma_process.h_gamma(gamma_sequence[:,-1])
            residual_expected_value = rate*self.pos_ext_gamma_process.unit_expected_residual(truncation_level)
            residual_variance = rate*self.pos_ext_gamma_process.unit_variance_residual(truncation_level)
            E_c = self.tolerance*x_series.sum(axis=1)
            itr += 1

            condition1 = (residual_variance/((E_c - residual_expected_value)**2) > self.pt)
            condition2 = (E_c < residual_expected_value)
        # We do not use residual approximation in this setting since Asmussen and Rosinski 2001 shows it is not valid for the gamma process.
        return x_series
    
    # Jump magnitude simulation:
    ## GIG-paper:
    def simulate_adaptive_series_setting_1(self, rate=1.0, size=1):
        gamma_sequence, x_series = self.simulate_series_setting_1(rate=rate, M=self.M_stable, gamma_0=0., size=size)
        truncation_level = self.tempered_stable_process.h_stable(gamma_sequence[:,-1])
        residual_expected_value = rate*self.tempered_stable_process.unit_expected_residual(truncation_level)
        residual_variance = rate*self.tempered_stable_process.unit_variance_residual(truncation_level)
        E_c = self.tolerance*x_series.sum(axis=1)
        itr = 1

        condition1 = (residual_variance/((E_c - residual_expected_value)**2) > self.pt)
        condition2 = (E_c < residual_expected_value)

        # Adaptive truncation is based on Theorem 3 of Kindap, Godsill 2023. 
        while condition1.any() or condition2.any():

            # Debug code:
            if (itr > self.max_iter):
                #print('Max iteration reached.')
                break

            gamma_sequence_extension, x_series_extension = self.simulate_series_setting_1(rate=rate, M=self.M_stable, gamma_0=gamma_sequence[:,-1], size=size)
            gamma_sequence = np.concatenate((gamma_sequence, gamma_sequence_extension), axis=1)
            x_series = np.concatenate((x_series, x_series_extension), axis=1)
            truncation_level = self.tempered_stable_process.h_stable(gamma_sequence[:,-1])
            residual_expected_value = rate*self.tempered_stable_process.unit_expected_residual(truncation_level)
            residual_variance = rate*self.tempered_stable_process.unit_variance_residual(truncation_level)
            E_c = self.tolerance*x_series.sum(axis=1)
            itr += 1

        return x_series, truncation_level

    def simulate_series_setting_1(self, rate, M, gamma_0=0., size=1):
        gamma_sequence, x_series = self.tempered_stable_process.simulate_from_series_representation(rate=rate, M=M, gamma_0=gamma_0, size=size)
        z_series = np.sqrt(np.random.gamma(shape=0.5, scale=np.power(x_series/(2*self.delta**2), -1.0)))
        hankel_squared = self.H_squared(z_series)
        acceptance_prob = 2/(hankel_squared*z_series*np.pi)
        u = np.random.uniform(low=0.0, high=1.0, size=x_series.shape)
        x_series[u > acceptance_prob] = 0.
        return gamma_sequence, x_series

    ## GH-paper:
    def simulate_adaptive_combined_series_setting_1(self, rate=1.0, size=1):

        # Simulate jump magnitudes:
        gamma_sequence_N_Ga_1, x_series_N_Ga_1 = self.simulate_left_bounding_series_setting_1(rate=rate, M=self.M_gamma, size=size)
        gamma_sequence_N_Ga_2, x_series_N_Ga_2 = self.simulate_left_bounding_series_setting_1_alternative(rate=rate, M=self.M_gamma, size=size)
        gamma_sequence_N2, x_series_N2 = self.simulate_right_bounding_series_setting_1(rate=rate, M=self.M_stable, size=size)

        x_series = np.concatenate((x_series_N_Ga_1, x_series_N_Ga_2), axis=1)
        x_series = np.concatenate((x_series, x_series_N2), axis=1)

        truncation_level_N_Ga_1 = self.gamma_process.h_gamma(gamma_sequence_N_Ga_1[:,-1])
        truncation_level_N_Ga_2 = self.gamma_process2.h_gamma(gamma_sequence_N_Ga_2[:,-1])
        truncation_level_N2 = self.tempered_stable_process.h_stable(gamma_sequence_N2[:,-1])

        # Residual statistics:
        residual_expected_value_N_Ga_1 = rate*self.gamma_process.unit_expected_residual(truncation_level_N_Ga_1)
        residual_variance_N_Ga_1 = rate*self.gamma_process.unit_variance_residual(truncation_level_N_Ga_1)
        residual_expected_value_N_Ga_2 = rate*self.gamma_process2.unit_expected_residual(truncation_level_N_Ga_2)
        residual_variance_N_Ga_2 = rate*self.gamma_process2.unit_variance_residual(truncation_level_N_Ga_2)
        residual_expected_value_N2 = rate*self.tempered_stable_process.unit_expected_residual(truncation_level_N2)
        residual_variance_N2 = rate*self.tempered_stable_process.unit_variance_residual(truncation_level_N2)
        residual_expected_value = residual_expected_value_N_Ga_1 + residual_expected_value_N_Ga_2 + residual_expected_value_N2
        residual_variance = residual_variance_N_Ga_1 + residual_variance_N_Ga_2 + residual_variance_N2

        ## Select process for simulation:
        ### First select process for individual dimensions
        selection = np.argmax(np.array([truncation_level_N_Ga_1, truncation_level_N_Ga_2, truncation_level_N2]), axis=0)
        ### Then randomly select a dimension based on empirical counts
        selection = np.random.choice(selection)

        # Adaptive simulation:
        _mean_lower_bound, _var_lower_bound = self.residual_stats(rate, truncation_level_gamma=truncation_level_N2, truncation_level_TS=truncation_level_N2)
        E_c = self.tolerance*x_series.sum(axis=1) + _mean_lower_bound
        itr = 1

        condition1 = (residual_variance/((E_c - residual_expected_value)**2) > self.pt)
        condition2 = (E_c < residual_expected_value)

        # Adaptive truncation is based on Corollary 5 of Kindap, Godsill 2023. 
        while condition1.any() or condition2.any():
            
            # Debug code:
            if (itr > self.max_iter):
                #print('Max iteration reached.')
                break

            if (selection == 2):
                gamma_sequence_extension, x_series_extension = self.simulate_right_bounding_series_setting_1(rate=rate, M=self.M_stable, gamma_0=gamma_sequence_N2[:,-1], size=size)
                gamma_sequence_N2 = np.concatenate((gamma_sequence_N2, gamma_sequence_extension), axis=1)
                x_series_N2 = np.concatenate((x_series_N2, x_series_extension), axis=1)
                truncation_level_N2 = self.tempered_stable_process.h_stable(gamma_sequence_N2[:,-1])
                residual_expected_value_N2 = rate*self.tempered_stable_process.unit_expected_residual(truncation_level_N2)
                residual_variance_N2 = rate*self.tempered_stable_process.unit_variance_residual(truncation_level_N2)
            elif (selection == 0):
                gamma_sequence_extension, x_series_extension = self.simulate_left_bounding_series_setting_1(rate=rate, M=self.M_gamma, gamma_0=gamma_sequence_N_Ga_1[:,-1], size=size)
                gamma_sequence_N_Ga_1 = np.concatenate((gamma_sequence_N_Ga_1, gamma_sequence_extension), axis=1)
                x_series_N_Ga_1 = np.concatenate((x_series_N_Ga_1, x_series_extension), axis=1)
                truncation_level_N_Ga_1 = self.gamma_process.h_gamma(gamma_sequence_N_Ga_1[:,-1])
                residual_expected_value_N_Ga_1 = rate*self.gamma_process.unit_expected_residual(truncation_level_N_Ga_1)
                residual_variance_N_Ga_1 = rate*self.gamma_process.unit_variance_residual(truncation_level_N_Ga_1)
            else:
                gamma_sequence_extension, x_series_extension = self.simulate_left_bounding_series_setting_1_alternative(rate=rate, M=self.M_gamma, gamma_0=gamma_sequence_N_Ga_2[:,-1], size=size)
                gamma_sequence_N_Ga_2 = np.concatenate((gamma_sequence_N_Ga_2, gamma_sequence_extension), axis=1)
                x_series_N_Ga_2 = np.concatenate((x_series_N_Ga_2, x_series_extension), axis=1)
                truncation_level_N_Ga_2 = self.gamma_process2.h_gamma(gamma_sequence_N_Ga_2[:,-1])
                residual_expected_value_N_Ga_2 = rate*self.gamma_process2.unit_expected_residual(truncation_level_N_Ga_2)
                residual_variance_N_Ga_2 = rate*self.gamma_process2.unit_variance_residual(truncation_level_N_Ga_2)
            
            x_series = np.concatenate((x_series, x_series_extension), axis=1)
            
            residual_expected_value = residual_expected_value_N_Ga_1 + residual_expected_value_N_Ga_2 + residual_expected_value_N2
            residual_variance = residual_variance_N_Ga_1 + residual_variance_N_Ga_2 + residual_variance_N2

            ### First select process for individual dimensions
            selection = np.argmax(np.array([truncation_level_N_Ga_1, truncation_level_N_Ga_2, truncation_level_N2]), axis=0)
            ### Then randomly select a dimension based on empirical counts
            selection = np.random.choice(selection)

            _mean_lower_bound, _var_lower_bound = self.residual_stats(rate, truncation_level_gamma=truncation_level_N2, truncation_level_TS=truncation_level_N2)
            E_c = self.tolerance*x_series.sum(axis=1) + _mean_lower_bound
            itr += 1

        truncation_level = truncation_level_N2
        return x_series, truncation_level

    def simulate_left_bounding_series_setting_1(self, rate, M, gamma_0=0.0, size=1):
        z1 = self.cornerpoint()
        gamma_sequence, x_series = self.gamma_process.simulate_from_series_representation(rate=rate, M=M, gamma_0=gamma_0, size=size)
        envelope_fnc = (((2*self.delta**2)**self.abs_lam)*incgammal(self.abs_lam, (z1**2)*x_series/(2*self.delta**2))*self.abs_lam*(1+self.abs_lam)
                        /((x_series**self.abs_lam)*(z1**(2*self.abs_lam))*(1+self.abs_lam*np.exp(-(z1**2)*x_series/(2*self.delta**2)))))
        u = np.random.uniform(low=0.0, high=1.0, size=x_series.shape)
        x_series[u > envelope_fnc] = 0.
        u_z = np.random.uniform(low=0.0, high=1.0, size=x_series.shape)    
        z_series = np.sqrt(((2*self.delta**2)/x_series)*gammaincinv(self.abs_lam, u_z*gammainc(self.abs_lam, (z1**2)*x_series/(2*self.delta**2))))
        hankel_squared = self.H_squared(z_series)
        acceptance_prob = 2/(hankel_squared*np.pi*((z_series**(2*self.abs_lam))/(z1**(2*self.abs_lam-1))))
        u = np.random.uniform(low=0.0, high=1.0, size=x_series.shape)
        x_series[u > acceptance_prob] = 0.
        return gamma_sequence, x_series

    def simulate_left_bounding_series_setting_1_alternative(self, rate, M, gamma_0=0.0, size=1):
        z1 = self.cornerpoint()
        gamma_sequence, x_series = self.gamma_process2.simulate_from_series_representation(rate=rate, M=M, gamma_0=gamma_0, size=size)
        envelope_fnc = (((2*self.delta**2)**self.abs_lam)*incgammal(self.abs_lam, (z1**2)*x_series/(2*self.delta**2))*self.abs_lam*(1+self.abs_lam)
                        /((x_series**self.abs_lam)*(z1**(2*self.abs_lam))*(1+self.abs_lam*np.exp(-(z1**2)*x_series/(2*self.delta**2)))))
        u = np.random.uniform(low=0.0, high=1.0, size=x_series.shape)
        x_series[u > envelope_fnc] = 0.
        u_z = np.random.uniform(low=0.0, high=1.0, size=x_series.shape)    
        z_series = np.sqrt(((2*self.delta**2)/x_series)*gammaincinv(self.abs_lam, u_z*gammainc(self.abs_lam, (z1**2)*x_series/(2*self.delta**2))))
        hankel_squared = self.H_squared(z_series)
        acceptance_prob = 2/(hankel_squared*np.pi*((z_series**(2*self.abs_lam))/(z1**(2*self.abs_lam-1))))
        u = np.random.uniform(low=0.0, high=1.0, size=x_series.shape)
        x_series[u > acceptance_prob] = 0.
        return gamma_sequence, x_series

    def simulate_right_bounding_series_setting_1(self, rate, M, gamma_0=0.0, size=1):
        z1 = self.cornerpoint()
        gamma_sequence, x_series = self.tempered_stable_process.simulate_from_series_representation(rate=rate, M=M, gamma_0=gamma_0, size=size)
        envelope_fnc = incgammau(0.5, (z1**2)*x_series/(2*self.delta**2))/(np.sqrt(np.pi)*np.exp(-(z1**2)*x_series/(2*self.delta**2)))
        u = np.random.uniform(low=0.0, high=1.0, size=x_series.shape)
        x_series[u > envelope_fnc] = 0.
        u_z = np.random.uniform(low=0.0, high=1.0, size=x_series.shape)
        z_series = np.sqrt(((2*self.delta**2)/x_series)*gammaincinv(0.5, u_z*(gammaincc(0.5, (z1**2)*x_series/(2*self.delta**2)))
                                                            + gammainc(0.5, (z1**2)*x_series/(2*self.delta**2))))
        hankel_squared = self.H_squared(z_series)
        acceptance_prob = 2/(hankel_squared*z_series*np.pi)
        u = np.random.uniform(low=0.0, high=1.0, size=x_series.shape)
        x_series[u > acceptance_prob] = 0.
        return gamma_sequence, x_series

    def simulate_adaptive_combined_series_setting_2(self, rate=1.0, size=1):
        # Simulate jump magnitudes:
        gamma_sequence_N_Ga_1, x_series_N_Ga_1 = self.simulate_left_bounding_series_setting_2(rate=rate, M=self.M_gamma, size=size)
        gamma_sequence_N_Ga_2, x_series_N_Ga_2 = self.simulate_left_bounding_series_setting_2_alternative(rate=rate, M=self.M_gamma, size=size)
        gamma_sequence_N2, x_series_N2 = self.simulate_right_bounding_series_setting_2(rate=rate, M=self.M_stable, size=size)
        x_series = np.concatenate((x_series_N_Ga_1, x_series_N_Ga_2), axis=1)
        x_series = np.concatenate((x_series, x_series_N2), axis=1)

        truncation_level_N_Ga_1 = self.gamma_process.h_gamma(gamma_sequence_N_Ga_1[:,-1])
        truncation_level_N_Ga_2 = self.gamma_process2.h_gamma(gamma_sequence_N_Ga_2[:,-1])
        truncation_level_N2 = self.tempered_stable_process.h_stable(gamma_sequence_N2[:,-1])

        # Residual statistics:
        residual_expected_value_N_Ga_1 = rate*self.gamma_process.unit_expected_residual(truncation_level_N_Ga_1)
        residual_variance_N_Ga_1 = rate*self.gamma_process.unit_variance_residual(truncation_level_N_Ga_1)
        residual_expected_value_N_Ga_2 = rate*self.gamma_process2.unit_expected_residual(truncation_level_N_Ga_2)
        residual_variance_N_Ga_2 = rate*self.gamma_process2.unit_variance_residual(truncation_level_N_Ga_2)
        residual_expected_value_N2 = rate*self.tempered_stable_process.unit_expected_residual(truncation_level_N2)
        residual_variance_N2 = rate*self.tempered_stable_process.unit_variance_residual(truncation_level_N2)
        residual_expected_value = residual_expected_value_N_Ga_1 + residual_expected_value_N_Ga_2 + residual_expected_value_N2
        residual_variance = residual_variance_N_Ga_1 + residual_variance_N_Ga_2 + residual_variance_N2

        ## Select process for simulation:
        ### First select process for individual dimensions
        selection = np.argmax(np.array([truncation_level_N_Ga_1, truncation_level_N_Ga_2, truncation_level_N2]), axis=0)
        ### Then randomly select a dimension based on empirical counts
        selection = np.random.choice(selection)

        # Adaptive simulation:
        _mean_lower_bound, _var_lower_bound = self.residual_stats(rate, truncation_level_gamma=truncation_level_N2, truncation_level_TS=truncation_level_N2)
        E_c = self.tolerance*x_series.sum(axis=1) + _mean_lower_bound
        itr = 1

        condition1 = (residual_variance/((E_c - residual_expected_value)**2) > self.pt)
        condition2 = (E_c < residual_expected_value)

        # Adaptive truncation is based on Corollary 5 of Kindap, Godsill 2023. 
        while condition1.any() or condition2.any():
            
            # Debug code:
            if (itr > self.max_iter):
                #print('Max iteration reached.')
                break

            if (selection == 2):
                gamma_sequence_extension, x_series_extension = self.simulate_right_bounding_series_setting_2(rate=rate, M=self.M_stable, gamma_0=gamma_sequence_N2[:,-1], size=size)
                gamma_sequence_N2 = np.concatenate((gamma_sequence_N2, gamma_sequence_extension), axis=1)
                x_series_N2 = np.concatenate((x_series_N2, x_series_extension), axis=1)
                truncation_level_N2 = self.tempered_stable_process.h_stable(gamma_sequence_N2[:,-1])
                residual_expected_value_N2 = rate*self.tempered_stable_process.unit_expected_residual(truncation_level_N2)
                residual_variance_N2 = rate*self.tempered_stable_process.unit_variance_residual(truncation_level_N2)
            elif (selection == 0):
                gamma_sequence_extension, x_series_extension = self.simulate_left_bounding_series_setting_2(rate=rate, M=self.M_gamma, gamma_0=gamma_sequence_N_Ga_1[:,-1], size=size)
                gamma_sequence_N_Ga_1 = np.concatenate((gamma_sequence_N_Ga_1, gamma_sequence_extension), axis=1)
                x_series_N_Ga_1 = np.concatenate((x_series_N_Ga_1, x_series_extension), axis=1)
                truncation_level_N_Ga_1 = self.gamma_process.h_gamma(gamma_sequence_N_Ga_1[:,-1])
                residual_expected_value_N_Ga_1 = rate*self.gamma_process.unit_expected_residual(truncation_level_N_Ga_1)
                residual_variance_N_Ga_1 = rate*self.gamma_process.unit_variance_residual(truncation_level_N_Ga_1)
            else:
                gamma_sequence_extension, x_series_extension = self.simulate_left_bounding_series_setting_2_alternative(rate=rate, M=self.M_gamma, gamma_0=gamma_sequence_N_Ga_2[:,-1], size=size)
                gamma_sequence_N_Ga_2 = np.concatenate((gamma_sequence_N_Ga_2, gamma_sequence_extension), axis=1)
                x_series_N_Ga_2 = np.concatenate((x_series_N_Ga_2, x_series_extension), axis=1)
                truncation_level_N_Ga_2 = self.gamma_process2.h_gamma(gamma_sequence_N_Ga_2[:,-1])
                residual_expected_value_N_Ga_2 = rate*self.gamma_process2.unit_expected_residual(truncation_level_N_Ga_2)
                residual_variance_N_Ga_2 = rate*self.gamma_process2.unit_variance_residual(truncation_level_N_Ga_2)
            
            x_series = np.concatenate((x_series, x_series_extension), axis=1)
            
            residual_expected_value = residual_expected_value_N_Ga_1 + residual_expected_value_N_Ga_2 + residual_expected_value_N2
            residual_variance = residual_variance_N_Ga_1 + residual_variance_N_Ga_2 + residual_variance_N2

            ### First select process for individual dimensions
            selection = np.argmax(np.array([truncation_level_N_Ga_1, truncation_level_N_Ga_2, truncation_level_N2]), axis=0)
            ### Then randomly select a dimension based on empirical counts
            selection = np.random.choice(selection)

            _mean_lower_bound, _var_lower_bound = self.residual_stats(rate, truncation_level_gamma=truncation_level_N2, truncation_level_TS=truncation_level_N2)
            E_c = self.tolerance*x_series.sum(axis=1) + _mean_lower_bound
            itr += 1

        truncation_level = truncation_level_N2
        return x_series, truncation_level

    def simulate_left_bounding_series_setting_2(self, rate, M, gamma_0=0.0, size=1):
        z0 = self.cornerpoint()
        H0 = z0*self.H_squared(z0)
        gamma_sequence, x_series = self.gamma_process.simulate_from_series_representation(rate=rate, M=M, gamma_0=gamma_0, size=size)
        envelope_fnc = (((2*self.delta**2)**self.abs_lam)* incgammal(self.abs_lam, (z0**2)*x_series/(2*self.delta**2))*self.abs_lam*(1+self.abs_lam)/
            ((x_series**self.abs_lam)*(z0**(2*self.abs_lam))*(1+self.abs_lam*np.exp(-(z0**2)*x_series/(2*self.delta**2)))))
        u = np.random.uniform(low=0.0, high=1.0, size=x_series.shape)
        x_series[u > envelope_fnc] = 0.

        u_z = np.random.uniform(low=0.0, high=1.0, size=x_series.shape)
        z_series = np.sqrt(((2*self.delta**2)/x_series)*gammaincinv(self.abs_lam, u_z*(gammainc(self.abs_lam, (z0**2)*x_series/(2*self.delta**2)))))
        hankel_squared = self.H_squared(z_series)
        acceptance_prob = H0/(hankel_squared*((z_series**(2*self.abs_lam))/(z0**(2*self.abs_lam-1))))
        u = np.random.uniform(low=0.0, high=1.0, size=x_series.shape)
        x_series[u > acceptance_prob] = 0.
        return gamma_sequence, x_series
    
    def simulate_left_bounding_series_setting_2_alternative(self, rate, M, gamma_0=0.0, size=1):
        z0 = self.cornerpoint()
        H0 = z0*self.H_squared(z0)
        gamma_sequence, x_series = self.gamma_process2.simulate_from_series_representation(rate=rate, M=M, gamma_0=gamma_0, size=size)
        envelope_fnc = (((2*self.delta**2)**self.abs_lam)* incgammal(self.abs_lam, (z0**2)*x_series/(2*self.delta**2))*self.abs_lam*(1+self.abs_lam)/
            ((x_series**self.abs_lam)*(z0**(2*self.abs_lam))*(1+self.abs_lam*np.exp(-(z0**2)*x_series/(2*self.delta**2)))))
        u = np.random.uniform(low=0.0, high=1.0, size=x_series.shape)
        x_series[u > envelope_fnc] = 0.

        u_z = np.random.uniform(low=0.0, high=1.0, size=x_series.shape)
        z_series = np.sqrt(((2*self.delta**2)/x_series)*gammaincinv(self.abs_lam, u_z*(gammainc(self.abs_lam, (z0**2)*x_series/(2*self.delta**2)))))
        hankel_squared = self.H_squared(z_series)
        acceptance_prob = H0/(hankel_squared*((z_series**(2*self.abs_lam))/(z0**(2*self.abs_lam-1))))
        u = np.random.uniform(low=0.0, high=1.0, size=x_series.shape)
        x_series[u > acceptance_prob] = 0.
        return gamma_sequence, x_series

    def simulate_right_bounding_series_setting_2(self, rate, M, gamma_0=0.0, size=1):
        z0 = self.cornerpoint()
        H0 = z0*self.H_squared(z0)
        gamma_sequence, x_series = self.tempered_stable_process.simulate_from_series_representation(rate=rate, M=M, gamma_0=gamma_0, size=size)
        envelope_fnc = gammaincc(0.5, (z0**2)*x_series/(2*self.delta**2))
        u = np.random.uniform(low=0.0, high=1.0, size=x_series.shape)
        x_series[u > envelope_fnc] = 0.
        u_z = np.random.uniform(low=0.0, high=1.0, size=x_series.shape)
        z_series = np.sqrt(((2*self.delta**2)/x_series)*gammaincinv(0.5, u_z*(gammaincc(0.5, (z0**2)*x_series/(2*self.delta**2)))
                                                            +gammainc(0.5, (z0**2)*x_series/(2*self.delta**2))))
        hankel_squared = self.H_squared(z_series)
        acceptance_prob = H0/(hankel_squared*z_series)
        u = np.random.uniform(low=0.0, high=1.0, size=x_series.shape)
        x_series[u > acceptance_prob] = 0.
        return gamma_sequence, x_series

    def simulate_points(self, rate, low, high, n_particles=1):
        """Returns the times and jump sizes associated with the point process representation of a Levy process. Returns the points.
        """
        x_series, truncation_level = self.simulate_jumps(rate=rate, size=n_particles)
        t_series = np.random.uniform(low=low, high=high, size=x_series.shape)
        # Simulate residual must depend on t_series as well.
        residual_t_series, residual_jumps = self.simulate_residual(low=low, high=high, truncation_level_gamma=truncation_level, truncation_level_TS=truncation_level, shape=x_series.shape)
        return np.concatenate((t_series, residual_t_series), axis=1), np.concatenate((x_series, residual_jumps), axis=1)

    def initialise_proposal_samples(self, low, high, n_particles=1):
        """Simulates random times and jumps sizes from the prior over the whole evaluation input space.
        """
        self.t_series, self.x_series = self.simulate_points(rate=(high-low), low=low, high=high, n_particles=n_particles) 

    def set_proposal_samples(self, proposal_t_series, proposal_x_series):
        self.t_series = proposal_t_series
        self.x_series = proposal_x_series

    def propose_subordinator(self, proposal_interval):
        """On the given interval, removes the previous points and simulates new random points. Returns the proposal points.
        NOTE that this function only works when there is a single particle under consideration. This may be fixed by
        finding a method to keep the shape of x_series constant while applying conditional selection.
        """
        conditioning_x_series = self.x_series[(proposal_interval[1] < self.t_series)]
        conditioning_x_series = np.concatenate((conditioning_x_series, self.x_series[(self.t_series < proposal_interval[0])])).reshape(1, -1)
        conditioning_t_series = self.t_series[(proposal_interval[1] < self.t_series)]
        conditioning_t_series = np.concatenate((conditioning_t_series, self.t_series[(self.t_series < proposal_interval[0])])).reshape(1, -1)
        proposed_t_series, proposed_x_series = self.simulate_points(rate=(proposal_interval[1]-proposal_interval[0]), low=proposal_interval[0], high=proposal_interval[1])
        proposal_x_series = np.concatenate((conditioning_x_series, proposed_x_series), axis=1)
        proposal_t_series = np.concatenate((conditioning_t_series, proposed_t_series), axis=1)  
        return proposal_t_series, proposal_x_series
