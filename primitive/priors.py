import numpy as np
from scipy.special import gamma as gammafnc

from primitive.parameters import ParameterInterface

# Base prior distribution class:

class PriorModule(ParameterInterface):
    parameter_keys = None
        
    def likelihood(self, x):
        """ The density g( \dot | \theta ) evaluated at \theta^\prime.
        """
        pass

    def log_likelihood(self, x):
        """ The log density g( \dot | \theta ) evaluated at \theta^\prime.
        """
        pass

class InverseGammaPrior(PriorModule):
    parameter_keys = ["alpha", "beta"]

    def likelihood(self, x):
        """ The density g( \dot | \theta ) evaluated at \theta^\prime.
        """
        return ( self.beta**self.alpha / gammafnc(self.alpha) ) * x**(-self.alpha-1) * np.exp(-self.beta/x)
    
    def log_likelihood(self, x):
        return self.alpha * np.log(self.beta) - np.log(gammafnc(self.alpha)) - (self.alpha + 1) * np.log(x) - self.beta / x
    
    def mean(self):
        if (self.alpha > 1):
            return self.beta / (self.alpha - 1)
        return None
    
    def variance(self):
        if (self.alpha > 2):
            return self.beta**2 / ((self.alpha - 1)**2 * (self.alpha - 2))
        return None

