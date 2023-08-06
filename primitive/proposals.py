import numpy as np
from scipy.special import gamma as gammafnc

from primitive.parameters import ParameterInterface

# Base proposal function class:

class ProposalModule(ParameterInterface):
    parameter_keys = None

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

# Specific proposal function classes:

class GaussianProposalGenerator(ProposalModule):
    parameter_keys = ["sigma"]

    def likelihood(self, x_prime, x):
        return 1/(np.sqrt(2*np.pi)*self.sigma) * np.exp( -0.5 * ((x_prime - x)/self.sigma)**2 )
    
    def log_likelihood(self, x_prime, x):
        return -0.5 * ((x_prime - x)/self.sigma)**2 - np.log( np.sqrt(2*np.pi)*self.sigma )

    def propose(self, x, shape):
        return x + np.random.normal(loc=0., scale=self.sigma, size=shape)
    
class LaplaceProposalGenerator(ProposalModule):
    parameter_keys = ["sigma"]

    def likelihood(self, x_prime, x):
        return 1/(2*self.sigma) * np.exp( -np.abs(x_prime - x) / self.sigma )
    
    def log_likelihood(self, x_prime, x):
        return - (np.abs(x_prime - x)/self.sigma) - np.log(2*self.sigma)

    def propose(self, x, shape):
        return x + np.random.laplace(loc=0., scale=self.sigma, size=shape)

class FoldedLaplaceProposalGenerator(ProposalModule):
    parameter_keys = ["sigma"]

    def likelihood(self, x_prime, x):
        scale = 1 / self.sigma
        if (x_prime < x):
            return scale * np.exp(-x/self.sigma) * np.cosh(x_prime/self.sigma)
        else:
            return scale * np.exp(-x_prime/self.sigma) * np.cosh(x/self.sigma)
        
    def log_likelihood(self, x_prime, x):
        scale = -1 * np.log(self.sigma)
        if (x_prime < x):
            return scale - x/self.sigma + np.log( np.cosh(x_prime/self.sigma) )
        else:
            return scale - x_prime/self.sigma + np.log( np.cosh(x/self.sigma) )

    def propose(self, x, shape):
        return np.abs(x + np.random.laplace(loc=0., scale=self.sigma, size=shape))

class FoldedGaussianProposalGenerator(ProposalModule):
    parameter_keys = ["sigma"]

    def likelihood(self, x_prime, x):
        return 1/(np.sqrt(2*np.pi)*self.sigma) * np.exp( -0.5 * ((x_prime - x)/self.sigma)**2 ) + 1/(np.sqrt(2*np.pi)*self.sigma) * np.exp( -0.5 * ((x_prime + x)/self.sigma)**2 )
    
    def log_likelihood(self, x_prime, x):
        return -0.5 * ((x_prime - x)/self.sigma)**2 - np.log( np.sqrt(2*np.pi)*self.sigma ) - 0.5 * ((x_prime + x)/self.sigma)**2 - np.log( np.sqrt(2*np.pi)*self.sigma ) 

    def propose(self, x, shape):
        return np.abs(x + np.random.normal(loc=0., scale=self.sigma, size=shape))
    
class InverseGammaProposalGenerator(ProposalModule):
    parameter_keys = ["alpha", "beta"]

    def likelihood(self, x_prime, x):
        return ( self.beta**self.alpha / gammafnc(self.alpha) ) * x_prime**(-self.alpha-1) * np.exp(-self.beta/x_prime)
    
    def log_likelihood(self, x_prime, x):
        return self.alpha * np.log(self.beta) - np.log(gammafnc(self.alpha)) - (self.alpha + 1) * np.log(x_prime) - self.beta / x_prime

    def propose(self, x, shape):
        return 1 / np.random.gamma(shape=self.alpha, scale=1/self.beta, size=shape)
    