import numpy as np

from primitive.linalg import LinearOperator

from stochastic.integrals import NormalVarianceMeanProcessDrivenIntegral
from stochastic.variables import GaussianNoise

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
    

# Constant velocity model definitions:

## Linear operators for the Constant velocity model

class expA_ConstantVelocity(LinearOperator):
    parameter_keys = ["shape"]

    def compute_matrix(self, dt):
        expA = np.zeros(self.shape)
        expA[0][0] = 1.
        expA[0][1] = dt
        expA[1][1] = 1.
        return expA

## Constant velocity model object:

## The only significant difference of this object from NVMLangevinModel is self.expA. There may be a better way to implement this.

class NVMConstantVelocityModel(BaseStateSpaceModel):

    def __init__(self, subordinator, mu=0., sigma=1., sigma_eps=0.1, shape=(2,1)):

        # State-space model attributes:
        self.expA = expA_ConstantVelocity(**{"shape":(shape[0], shape[0])})
        self.h = h_vector(shape=(shape[0], 1))
        self.ft = lambda dt: self.expA(dt) @ self.h()

        # System noise
        system_noise = NormalVarianceMeanProcessDrivenIntegral(**{"shape":(1,1), "mu":mu, "sigma":sigma, "subordinator":subordinator})
        system_noise.set_ssm_attributes(h=self.h, ft=self.ft, expA=self.expA)

        # Observation model
        H = np.zeros((1,2))
        H[0][0] = 1

        config = {"A":self.expA, "I":system_noise, "H":H, "eps":GaussianNoise(**{"shape":(1,1), "sigma_eps":sigma_eps})}
        super().__init__(**config)

    def get_parameter_values(self):
        return self.I.get_parameter_values() | self.eps.get_parameter_values()
    
    def set_parameter_values(self, **kwargs):
        self.expA.set_parameter_values(**kwargs)

    def sample(self, times, size=1):
        # Initialise the subordinator jumps
        low = np.min(times)
        high = np.max(times)
        self.I.subordinator.initialise_proposal_samples(low=low, high=high)

        x, y = super().sample(times=times, size=size)

        return x, y


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

        # System noise
        system_noise = NormalVarianceMeanProcessDrivenIntegral(**{"shape":(1,1), "mu":mu, "sigma":sigma, "subordinator":subordinator})
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
