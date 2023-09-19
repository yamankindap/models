import numpy as np

from primitive.parameters import ParameterInterface

# Base stochastic integral object:

class ForcingFunction(ParameterInterface):
    """The ForcingFunction class represents the forcing (driving) function in a stochastic differential equation (SDE). 

    It is primarily intended as a template for building stochastic forcing functions for SDEs driven by various different processes.

    A ForcingFunction can be interpreted as a random variable whose scale is defined through a time interval (s, t).

    The ForcingFunction class and all its child classes are not designed to be instantiated as individual objects. They are sub-modules used by a Model object.
    """
    parameter_keys = None

    def get_parameter_values(self):
        """A ForcingFunction is a parameterised class but the shape parameter is excluded from the resulting dictionary since it is assumed to not change
        during sampling procedures.
        """
        parameters = super().get_parameter_values()
        # Remove the shape key as it will not be changed after initialisation.
        return {key: value for key, value in parameters.items() if key not in ["shape"]}

    def sample(self, s=None, t=None):
        pass

    def __call__(self, s=None, t=None):
        return self.sample(s=s, t=t)
    
# Independent Brownian motion
    
class BrownianMotion(ForcingFunction):
    parameter_keys = ["shape", "sigma"]

    def sample(self, s=None, t=None):
        """Generates samples from a multivariate Gaussian with independent dimensions with a scale proportional to the length of an interval (s, t).

        This function does not implement any further discretisation of the interval (s, t) and simply generates a single independent homoscedastic Gaussian vector as the driving noise term.
        """
        dW = np.sqrt(t - s) * self.sigma * np.random.randn(self.shape[0], self.shape[1])
        return dW

# INCOMPLETE
# Stochastic integral driven by a Brownian motion
# There are two types of BrownianMotionDrivenIntegral objects. 
# The first uses the analytical solution directly to provide the noise covariance. This requires the attribute Q to be set.
# The second type uses an Euler-type discretisation of the solution to approximate the path of a particle. This procedure is similar to the generalised shot-noise method
# solution for Levy processes.

class BrownianMotionDrivenIntegral(BrownianMotion):
    parameter_keys = ["shape", "sigma"]

    def __init__(self, **kwargs):
        # Set variable parameters using the ParameterInterface class.
        super().__init__(**kwargs)

        # Set simulation type:
        self.type = kwargs.get("type", None)
        if self.type is None:
            raise ValueError("The keyword argument 'type' is required. The two choices are: 'analytical' and 'euler'.")
        elif (self.type == "analytical"):
            self.sample = self.sample_analytical
        elif (self.type == "euler"):
            self.sample = self.sample_euler
        else:
            raise ValueError("The keyword argument 'type' is set to an unknown type.")
        
    def set_unit_noise_covariance(self, Q):
        """The attribute Q set by this method is assigned by a separate object as required. 

        Specifically, Q represents the noise covariance matrix for a given model. 

        Q may be a fixed function with argument dt that represents the length of a time interval or it may be a LinearOperator object with parameters that can be included in sampling.
        """
        setattr(self, "Q", Q)

    def sample_analytical(self, s=None, t=None):
        dW = np.random.multivariate_normal(mean=np.zeros((self.Q.shape[0])), cov=self.sigma*self.Q(t-s), size=1).T
        return dW
    
    def sample_euler(self, s=None, t=None):
        pass

    def conditional_moments(self, s=None, t=None):
        return 0, self.sigma*self.Q(t-s)

class NormalVarianceMeanProcessDrivenIntegral(ForcingFunction):
    parameter_keys = ["shape", "mu", "sigma"]

    def __init__(self, **kwargs):
        # Set parameters using the ParameterInterface class.
        super().__init__(**kwargs)

        self.subordinator = kwargs.get("subordinator", None)
        if self.subordinator is None:
            self.process = kwargs.get("process", None)

            if self.process is None:
                raise ValueError("The forcing function is not initialised. Arguments must contain a subordinator or a process.")
    
    def sample(self, s=None, t=None):
        mean, cov = self.conditional_moments(s, t)
        dW = np.random.multivariate_normal(mean=mean.flatten(), cov=cov, size=1).T
        return dW
    
    def set_ssm_attributes(self, h, ft, expA):
        """The attributes set by this method are assigned by a separate object as required. 

        Specifically, the attributes h, ft, and expA are terms that explicitly show up in the solution of an SDE.
        """
        setattr(self, "h", h)
        setattr(self, "ft", ft)
        setattr(self, "expA", expA)

    def conditional_moments(self, s, t):
        mask = (s < self.subordinator.t_series) & (self.subordinator.t_series <= t)
        x_series = self.subordinator.x_series[mask]
        t_series = self.subordinator.t_series[mask]

        mean = np.zeros(self.h.shape)
        for i in range(x_series.size):
            mean += self.ft(t-t_series[i]) @ np.array([[self.mu]]) @ np.array([[x_series[i]]])

        cov = np.zeros(self.expA.shape)
        for i in range(x_series.size):
            mat = self.ft(t-t_series[i])
            cov += mat @ mat.T * np.array([[self.sigma**2]]) * np.array([[x_series[i]]])

        return mean, cov
    
    def proposed_conditional_moments(self, s, t, t_series, x_series):
        mask = (s < t_series) & (t_series <= t)
        x_series = x_series[mask]
        t_series = t_series[mask]

        mean = np.zeros(self.h.shape)
        for i in range(x_series.size):
            mean += self.ft(t-t_series[i]) @ np.array([[self.mu]]) @ np.array([[x_series[i]]])

        cov = np.zeros(self.expA.shape)
        for i in range(x_series.size):
            mat = self.ft(t-t_series[i])
            cov += mat @ mat.T * np.array([[self.sigma**2]]) * np.array([[x_series[i]]])

        return mean, cov