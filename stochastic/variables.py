import numpy as np
from primitive.parameters import ParameterInterface


# Measurement noise objects:

class Noise(ParameterInterface):
    parameter_keys = None

    def get_parameter_values(self):
        parameters = super().get_parameter_values()
        # Remove the shape key as it will not be changed after initialisation.
        return {key: value for key, value in parameters.items() if key not in ["shape"]}

    def sample(self, t=None):
        pass

    def __call__(self, t=None):
        return self.sample(t=t)
    
class GaussianNoise(Noise):
    parameter_keys = ["shape", "sigma_eps"]

    def covariance(self):
        return self.sigma_eps**2 * np.ones((self.shape[0], self.shape[1]))

    def sample(self, t=None):
        return self.sigma_eps * np.random.randn(self.shape[0], self.shape[1])