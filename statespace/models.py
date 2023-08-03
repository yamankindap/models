import numpy as np

from primitives.parameters import ParameterInterface
from primitives.linalg import LinearOperator



# Forcing function objects:

class ForcingFunction(ParameterInterface):
    parameter_keys = None

    def get_parameter_values(self):
        parameters = super().get_parameter_values()
        # Remove the shape key as it will not be changed after initialisation.
        del parameters["shape"]
        return parameters

    def sample(self, s=None, t=None):
        pass

    def __call__(self, s=None, t=None):
        return self.sample(s=s, t=t)
    
class BrownianMotion(ForcingFunction):
    parameter_keys = ["shape", "sigma"]

    def sample(self, s=None, t=None):
        dW = np.sqrt(t - s) * self.sigma * np.random.randn(self.shape[0], self.shape[1])
        return dW
    

class NormalVarianceMeanProcess(ForcingFunction):
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
        # Create/modify instance parameters named after the key and stores value.
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
    

# Measurement noise objects:

class Noise(ParameterInterface):
    parameter_keys = None

    def get_parameter_values(self):
        parameters = super().get_parameter_values()
        # Remove the shape key as it will not be changed after initialisation.
        del parameters["shape"]
        return parameters

    def sample(self, t=None):
        pass

    def __call__(self, t=None):
        return self.sample(t=t)
    
class GaussianNoise(Noise):
    parameter_keys = ["shape", "sigma_eps"]

    def sample(self, t=None):
        return self.sigma_eps * np.random.randn(self.shape[0], self.shape[1])



# Base State space model object:

class BaseStateSpaceModel:

    def __init__(self, **kwargs):

        # System transition matrix
        self.A = kwargs.get("A", None)

        # System forcing function
        self.I = kwargs.get("I", None)

        # Measurement matrix
        self.H = kwargs.get("H", None)

        # Measurement noise:
        self.eps = kwargs.get("eps", None)

    def set_configuration(self, **kwargs):
        # System transition matrix
        self.A = kwargs.get("A", None)

        # System forcing function
        self.I = kwargs.get("I", None)

        # Measurement matrix
        self.H = kwargs.get("H", None)

        # Measurement noise:
        self.eps = kwargs.get("eps", None)

    def get_parameter_values(self):
        pass
    
    def set_parameter_values(self, **kwargs):
        pass

    def sample(self, times, size=1):
        # Initialise state and measurement arrays:
        ## The number of columns in x_init should be configurable.
        ## We assume that X starts with [0 ... 0]^T
        x_init = np.zeros((self.A.shape[1], 1))

        ## The third dimension of this array has to be same as the number of columns in x_init.
        x = np.zeros(shape=(times.shape[0], x_init.shape[0], 1))

        ## The number of columns in y should be configurable.
        y = np.zeros(shape=(times.shape[0], self.H.shape[0], 1))

        x[0] = x_init
        y[0] = self.H @ x[0] + self.eps(t=times[0])

        for i in range(1, times.shape[0]):
            dt = times[i] - times[i-1]

            x[i] = self.A(dt) @ x[i-1] + self.I(s=times[i-1], t=times[i])
            y[i] = self.H @ x[i] + self.eps(t=times[i])

        return x, y


# Vector SDE model definitions:

class h_vector(LinearOperator):
    parameter_keys = ["shape", "indicator_dim", "value"]

    def __init__(self, shape=(2,1)):
        super().__init__(**{"shape":shape, "indicator_dim":(-1,0), "value":1.})

    def get_parameter_values(self):
        # There are no parameters in the h vector for Langevin dynamics.
        return {}

    def compute_matrix(self, dt=None):
        h = np.zeros(self.shape)
        h[self.indicator_dim[0]][self.indicator_dim[1]] = self.value
        return h


# Langevin model definitions:

## Linear operators for the Langevin model

class expA_Langevin(LinearOperator):
    parameter_keys = ["shape", "theta"]

    def compute_matrix(self, dt):
        expA = np.zeros(self.shape)
        expA[0][0] = 1.
        expA[0][1] = (np.exp(self.theta*dt) - 1) / self.theta
        expA[1][1] = np.exp(self.theta*dt)
        return expA

## Langevin model object:

class NVMLangevinModel(BaseStateSpaceModel):

    def __init__(self, subordinator, theta, mu=0., sigma=1., sigma_eps=0.1, shape=(2,1)):

        # State-space model attributes:
        self.expA = expA_Langevin(**{"shape":(shape[0], shape[0]), "theta":theta})
        self.h = h_vector(shape=(shape[0], 1))
        self.ft = lambda dt: self.expA(dt) @ self.h()

        # System noise
        # This will be changed for Levy processes:
        # system_noise = BrownianMotion(**{"shape":(1,1), "sigma":1.})

        # def tmp(s, t):
        #     return self.ft(t-s) @ system_noise(s=s, t=t)

        # System noise
        system_noise = NormalVarianceMeanProcess(**{"shape":(1,1), "mu":mu, "sigma":sigma, "subordinator":subordinator})
        system_noise.set_ssm_attributes(h=self.h, ft=self.ft, expA=self.expA)

        # Observation model
        H = np.zeros((1,2))
        H[0][0] = 1

        config = {"A":self.expA, "I":system_noise, "H":H, "eps":GaussianNoise(**{"shape":(1,1), "sigma_eps":sigma_eps})}
        super().__init__(**config)

    def get_parameter_values(self):
        return self.expA.get_parameter_values() | self.I.get_parameter_values() | self.eps.get_parameter_values()
    
    def set_parameter_values(self, **kwargs):
        self.expA.set_parameter_values(**kwargs)

    def sample(self, times, size=1):
        # Initialise the subordinator jumps
        low = np.min(times)
        high = np.max(times)
        self.I.subordinator.initialise_proposal_samples(low=low, high=high)

        x, y = super().sample(times=times, size=size)

        return x, y
