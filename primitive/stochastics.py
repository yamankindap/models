import numpy as np
from primitive.parameters import ParameterInterface


# A Levy process is specifically defined as a 1 dimensional stochastic process. 
# The multidimensional case will be handled under a separate LevyField object that take LevyProcess objects as inputs.

class LevyProcess(ParameterInterface):
    parameter_keys = None

    def __len__(self):
        return 1

    def set_name(self, name):
        if not hasattr(self, 'name'):
            self.name = name
        else:
            print('The process is already named as {}.'.format(self.name))

    def stochastic_integral(self, evaluation_points, t_series, x_series):
        """This method assumes that the evaluation_points is 1 dimensional with shape (N, 1).
        Returns an array of same shape.

        This needs to be modified for multidimensional X matrices.
        """
        W = [x_series[t_series<point].sum() for point in evaluation_points]
        return np.array(W).reshape(evaluation_points.shape[0], -1)
    
    # The following methods are required to treat a LevyProcess as a proposal object.

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

