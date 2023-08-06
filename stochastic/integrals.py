import numpy as np
from primitive.parameters import ParameterInterface


class ForcingFunction(ParameterInterface):
    parameter_keys = None

    def get_parameter_values(self):
        parameters = super().get_parameter_values()
        # Remove the shape key as it will not be changed after initialisation.
        return {key: value for key, value in parameters.items() if key not in ["shape"]}

    def sample(self, s=None, t=None):
        pass

    def __call__(self, s=None, t=None):
        return self.sample(s=s, t=t)
    
class BrownianMotion(ForcingFunction):
    parameter_keys = ["shape", "sigma"]

    def sample(self, s=None, t=None):
        dW = np.sqrt(t - s) * self.sigma * np.random.randn(self.shape[0], self.shape[1])
        return dW
    

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