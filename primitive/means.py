import numpy as np

from primitive.parameters import ParameterInterface

# Base mean function class:

class MeanFunction(ParameterInterface):
    parameter_keys = None
    
    def __call__(self, X, ndims):
        return self.evaluate(X=X, ndims=ndims)
    
    def evaluate(self, X, ndims):
        pass

# Specific mean function classes:

class ZeroMeanFunction(MeanFunction):
    parameter_keys = []

    def evaluate(self, X, ndims):
        return np.zeros((X.shape[0], ndims))