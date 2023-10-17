import numpy as np

from primitive.linalg import LinearOperator

from stochastic.integrals import NormalVarianceMeanProcessDrivenIntegral, BrownianMotionDrivenIntegral
from stochastic.variables import GaussianNoise

# Base State space model object:

class BaseStateSpaceModel:

    def __init__(self, **kwargs):
        """A state-space model (SSM) is defined as

            X(t) = A X(s) + I(s, t)
            Y(t) = H X(t) + eps(t)

        To initialise an SSM, the required keyword arguments are A, I, H, and eps. 

        The argument A must be a LinearOperator object that has a valid compute_matrix method.

        The argument I is a ForcingFunction object defined in stochastic.integrals.py. It represents a stochastic integral that can be sampled given a time interval (s, t).

        The argument H is a fixed numpy array with the correct dimensions. It can be generalised to a LinearOperator if a parameterised matrix is required.

        The argument eps is a random variable that is implemented as a Noise object defined in stochastic.variables.py. It may in general be time dependent.
        """
        # System transition matrix
        self.A = kwargs.get("A", None)

        # System forcing function (stochastic integral)
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
        """The assumption that x_init is always zeros is not fully general. The transition function is dependent on the actual value of x[0]. Hence the
        function assumes that times[0] = 0. and times has at least two values.
        """
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

# The h_vector selects the state dimension that a stochastic process directly forces. 
# The default h_vector defined here only selects the last dimension of the state.

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
    

# INCOMPLETE 
# The compute_matrix methods are manually defined for 2 dimensional cases.

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
    
class Q_ConstantVelocity(LinearOperator):
    parameter_keys = ["shape"]

    def compute_matrix(self, dt):
        Q = np.zeros(self.shape)
        Q[0][0] = dt**3 / 3
        Q[0][1] = dt**2 / 2
        Q[1][0] = dt**2 / 2
        Q[1][1] = dt
        return Q

## Constant velocity model object:

class BrownianConstantVelocityModel(BaseStateSpaceModel):

    def __init__(self, sigma=1., sigma_eps=0.1, shape=(2,1)):

        # State-space model attributes:
        self.expA = expA_ConstantVelocity(**{"shape":(shape[0], shape[0])})
        self.h = h_vector(shape=(shape[0], 1))
        self.ft = lambda dt: self.expA(dt) @ self.h()
        self.unit_Q = Q_ConstantVelocity(**{"shape":(shape[0], shape[0])})

        # System noise
        system_noise = BrownianMotionDrivenIntegral(**{"shape":shape, "sigma":sigma, "type":"analytical"})
        system_noise.set_unit_noise_covariance(Q=self.unit_Q)
        
        # Observation model
        H = np.zeros((1,2))
        H[0][0] = 1

        config = {"A":self.expA, "I":system_noise, "H":H, "eps":GaussianNoise(**{"shape":(1,1), "sigma_eps":sigma_eps})}
        super().__init__(**config)

    def get_parameter_values(self):
        return self.I.get_parameter_values() | self.eps.get_parameter_values()
    
    def set_parameter_values(self, **kwargs):
        self.I.set_parameter_values(**kwargs)
        self.eps.set_parameter_values(**kwargs)

## The only significant difference of this object from NVMLangevinModel is self.expA. There may be a better way to implement this.

class NVMConstantVelocityModel(BaseStateSpaceModel):

    def __init__(self, subordinator, mu=0., sigma=1., sigma_eps=0.1, shape=(2,1)):

        # State-space model attributes:
        self.expA = expA_ConstantVelocity(**{"shape":(shape[0], shape[0])})
        self.h = h_vector(shape=(shape[0], 1))
        self.ft = lambda dt: self.expA(dt) @ self.h()

        # System noise
        system_noise = NormalVarianceMeanProcessDrivenIntegral(**{"shape":shape, "mu":mu, "sigma":sigma, "subordinator":subordinator})
        system_noise.set_ssm_attributes(h=self.h, ft=self.ft, expA=self.expA)

        # Observation model
        H = np.zeros((1,2))
        H[0][0] = 1

        config = {"A":self.expA, "I":system_noise, "H":H, "eps":GaussianNoise(**{"shape":(1,1), "sigma_eps":sigma_eps})}
        super().__init__(**config)

    def get_parameter_values(self):
        return self.I.get_parameter_values() | self.eps.get_parameter_values()
    
    def set_parameter_values(self, **kwargs):
        self.I.set_parameter_values(**kwargs)
        self.eps.set_parameter_values(**kwargs)

    def sample(self, times, size=1):
        # Initialise the subordinator jumps
        ## The function assumes that when size > 1, all samples are conditional on a single subordinator realisation.
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
    
class Q_Langevin(LinearOperator):
    parameter_keys = ["shape", "theta"]

    def compute_matrix(self, dt):
        Q = np.zeros(self.shape)
        K = -1. * self.theta

        Q[0][0] = ( dt - (2/K) * (1 - np.exp(self.theta*dt)) + (1/(2*K)) * (1 - np.exp(2*self.theta*dt)) ) / K**2
        Q[0][1] = ( (1/K) * (1 - np.exp(self.theta*dt)) - (1/(2*K)) * (1 - np.exp(2*self.theta*dt)) ) / K
        Q[1][0] = Q[0][1]
        Q[1][1] = (1 - np.exp(2*self.theta*dt)) / (2*K)
        return Q

## Langevin model object:

class BrownianLangevinModel(BaseStateSpaceModel):

    def __init__(self, theta, sigma=1., sigma_eps=0.1, shape=(2,1)):

        # State-space model attributes:
        self.expA = expA_Langevin(**{"shape":(shape[0], shape[0]), "theta":theta})
        self.h = h_vector(shape=(shape[0], 1))
        self.ft = lambda dt: self.expA(dt) @ self.h()
        self.unit_Q = Q_Langevin(**{"shape":(shape[0], shape[0]), "theta":theta})

        # System noise
        system_noise = BrownianMotionDrivenIntegral(**{"shape":shape, "sigma":sigma, "type":"analytical"})
        system_noise.set_unit_noise_covariance(Q=self.unit_Q)
        
        # Observation model
        H = np.zeros((1,2))
        H[0][0] = 1

        config = {"A":self.expA, "I":system_noise, "H":H, "eps":GaussianNoise(**{"shape":(1,1), "sigma_eps":sigma_eps})}
        super().__init__(**config)

    def get_parameter_values(self):
        return self.expA.get_parameter_values() | self.I.get_parameter_values() | self.eps.get_parameter_values()
    
    def set_parameter_values(self, **kwargs):
        self.expA.set_parameter_values(**kwargs)
        self.I.set_parameter_values(**kwargs)
        self.eps.set_parameter_values(**kwargs)


class NVMLangevinModel(BaseStateSpaceModel):

    def __init__(self, subordinator, theta, mu=0., sigma=1., sigma_eps=0.1, shape=(2,1)):

        # State-space model attributes:
        self.expA = expA_Langevin(**{"shape":(shape[0], shape[0]), "theta":theta})
        self.h = h_vector(shape=(shape[0], 1))
        self.ft = lambda dt: self.expA(dt) @ self.h()

        # System noise
        system_noise = NormalVarianceMeanProcessDrivenIntegral(**{"shape":shape, "mu":mu, "sigma":sigma, "subordinator":subordinator})
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
        ## The function assumes that when size > 1, all samples are conditional on a single subordinator realisation.
        low = np.min(times)
        high = np.max(times)
        self.I.subordinator.initialise_proposal_samples(low=low, high=high)

        x, y = super().sample(times=times, size=size)

        return x, y
