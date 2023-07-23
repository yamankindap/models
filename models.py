import numpy as np
from primitives.linalg import invert_covariance

# Specific probabilistic model class:

class GaussianProcess:

    def __init__(self, mean_function, covariance_function, noise_covariance_function):

        # Set system and measurement models:
        ## System model:
        self.mean_function = mean_function
        self.covariance_function = covariance_function
        ## Measurement model:
        self.noise_covariance_function = noise_covariance_function

    def get_parameter_values(self):
        # We might need to exclude some parameters from learning and add a key removal stage to below:
        return self.covariance_function.get_parameter_values() | self.mean_function.get_parameter_values()
    
    def set_parameter_values(self, **kwargs):
        self.covariance_function.set_parameter_values(**kwargs)
        self.mean_function.set_parameter_values(**kwargs)

    def sample(self, X, size=1):
        """Sample independent realisations with the given size.
        """
        system_cov = self.covariance_function(X)
        fX = np.random.multivariate_normal(mean=self.mean_function(X, ndims=1).flatten(), cov=system_cov, size=size).T

        measurement_cov = self.noise_covariance_function(X)
        noise = np.random.multivariate_normal(mean=self.mean_function(X, ndims=1).flatten(), cov=measurement_cov, size=size).T

        y = fX + noise
        return fX, y

    def block_Gaussian_conditioning(self, y, a, b, A, B, C):
        """Conditioning operation for a joint Gaussian.
        The function assumes the following joint Gaussian distribution P(x, y):
        mean = | a | 
               | b |
        covariance = | A   B |
                     | B.T C |
        Returns the mean vector and covariance matrix for P(x | y).
        """
        C_inv = invert_covariance(C)

        mean = a + B @ C_inv @ (y - b)
        cov = A - B @ C_inv @ B.T
        return mean, cov, C_inv

    def posterior_density(self, y, X, Xeval):
        """Given training inputs X and outputs y, return the posterior mean vector and covariance matrix evaluated on Xeval.
        """

        # Compute prior mean vectors:
        mu_eval = self.mean_function(Xeval, ndims=y.shape[1])
        mu = self.mean_function(X, ndims=y.shape[1])
        
        # Compute prior covariance matrices:
        SigmaXX = self.covariance_function(X, X)
        SigmaXevalX = self.covariance_function(Xeval, X)
        SigmaXevalXeval = self.covariance_function(Xeval, Xeval)
        Omega = self.noise_covariance_function(X)

        # Posterior inference for f(Xeval):
        mean, cov, C_inv = self.block_Gaussian_conditioning(y, mu_eval, mu, SigmaXevalXeval, SigmaXevalX, (SigmaXX + Omega))

        # Marginal likelihood computation:
        log_marginal_likelihood = 0.5 * (- y.T @ C_inv @ y - np.linalg.slogdet(SigmaXX + Omega)[1] - y.shape[0] * np.log(2*np.pi))

        return mean, cov, log_marginal_likelihood


class NonGaussianProcess:

    def __init__(self, mean_function, covariance_function, noise_covariance_function, subordinator):

        # Set system and measurement models:
        ## System model:
        self.mean_function = mean_function
        self.covariance_function = covariance_function

        ## There may be multiple subordinators associated with different dimensions. These may be represented with a dictionary or a LevyField object.
        self.subordinator = subordinator

        ## Measurement model:
        self.noise_covariance_function = noise_covariance_function

    def get_parameter_values(self):
        # We might need to exclude some parameters from learning and add a key removal stage to below:
        # We might include subordinator parameters here as well.
        return self.covariance_function.get_parameter_values() | self.mean_function.get_parameter_values()
    
    def set_parameter_values(self, **kwargs):
        self.covariance_function.set_parameter_values(**kwargs)
        self.mean_function.set_parameter_values(**kwargs)

    def sample(self, X, size=1):
        """Sample independent realisations with the given size.
        """
        low = X.min()
        high = X.max()
        self.subordinator.initialise_proposal_samples(low, high)
        Wx = self.subordinator.stochastic_integral(X, self.subordinator.t_series, self.subordinator.x_series)

        system_cov = self.covariance_function(Wx)
        fX = np.random.multivariate_normal(mean=self.mean_function(X, ndims=1).flatten(), cov=system_cov, size=size).T

        measurement_cov = self.noise_covariance_function(X)
        noise = np.random.multivariate_normal(mean=self.mean_function(X, ndims=1).flatten(), cov=measurement_cov, size=size).T

        y = fX + noise
        return fX, y
    
    def block_Gaussian_conditioning(self, y, a, b, A, B, C):
        """Conditioning operation for a joint Gaussian.
        The function assumes the following joint Gaussian distribution P(x, y):
        mean = | a | 
               | b |
        covariance = | A   B |
                     | B.T C |
        Returns the mean vector and covariance matrix for P(x | y).
        """
        C_inv = invert_covariance(C)

        mean = a + B @ C_inv @ (y - b)
        cov = A - B @ C_inv @ B.T
        return mean, cov, C_inv

    def posterior_density(self, y, X, Xeval):
        """Given training inputs X and outputs y, return the posterior mean vector and covariance matrix evaluated on Xeval.
        """

        # Compute prior mean vectors:
        mu_eval = self.mean_function(Xeval, ndims=y.shape[1])
        mu = self.mean_function(X, ndims=y.shape[1])
        
        # Compute prior covariance matrices:
        WX = self.subordinator.stochastic_integral(X, self.subordinator.t_series, self.subordinator.x_series)
        WXeval = self.subordinator.stochastic_integral(Xeval, self.subordinator.t_series, self.subordinator.x_series)

        SigmaXX = self.covariance_function(WX, WX)
        SigmaXevalX = self.covariance_function(WXeval, WX)
        SigmaXevalXeval = self.covariance_function(WXeval, WXeval)
        Omega = self.noise_covariance_function(WX)

        # Posterior inference for f(Xeval):
        mean, cov, C_inv = self.block_Gaussian_conditioning(y, mu_eval, mu, SigmaXevalXeval, SigmaXevalX, (SigmaXX + Omega))

        # Marginal likelihood computation:
        log_marginal_likelihood = 0.5 * (- y.T @ C_inv @ y - np.linalg.slogdet(SigmaXX + Omega)[1] - y.shape[0] * np.log(2*np.pi))

        return mean, cov, log_marginal_likelihood
    
    def proposal_posterior_density(self, y, X, Xeval, t_series, x_series):

        # Compute prior mean vectors:
        mu_eval = self.mean_function(Xeval, ndims=y.shape[1])
        mu = self.mean_function(X, ndims=y.shape[1])
        
        # Compute prior covariance matrices:
        WX = self.subordinator.stochastic_integral(X, t_series, x_series)
        WXeval = self.subordinator.stochastic_integral(Xeval, t_series, x_series)

        SigmaXX = self.covariance_function(WX, WX)
        SigmaXevalX = self.covariance_function(WXeval, WX)
        SigmaXevalXeval = self.covariance_function(WXeval, WXeval)
        Omega = self.noise_covariance_function(WX)

        # Posterior inference for f(Xeval):
        mean, cov, C_inv = self.block_Gaussian_conditioning(y, mu_eval, mu, SigmaXevalXeval, SigmaXevalX, (SigmaXX + Omega))

        # Marginal likelihood computation:
        log_marginal_likelihood = 0.5 * (- y.T @ C_inv @ y - np.linalg.slogdet(SigmaXX + Omega)[1] - y.shape[0] * np.log(2*np.pi))

        return mean, cov, log_marginal_likelihood
    
