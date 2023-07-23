import numpy as np
from scipy.special import gamma as gammafnc
from scipy.special import kv

from primitives.parameters import ParameterInterface

# Base kernel function class:

class StationaryKernel(ParameterInterface):
    parameter_keys = None

    def compute_distance_matrix(self, X1, X2=None):
        if X2 is None:
            X2 = X1
        
        X_norms_squared = np.sum(X1**2, axis=1, keepdims=True)
        if X1 is X2:
            Xprime_norms_squared = X_norms_squared.T
        else:
            Xprime_norms_squared = np.sum(X2**2, axis=1, keepdims=True).T

        dist_squared = np.abs(X_norms_squared + Xprime_norms_squared - 2 * np.einsum('ij,kj->ik', X1, X2))
        return np.sqrt(dist_squared)
    
    def compute_kernel_matrix(self, X1, X2=None):
        pass

    def __call__(self, X1, X2=None):
        return self.compute_kernel_matrix(X1=X1, X2=X2)


# Specific kernel function classes:

class WhiteKernel(StationaryKernel):
    parameter_keys = ["sigma"]

    def compute_kernel_matrix(self, X1, X2=None):
        return self.sigma**2 * np.eye(X1.shape[0])

class SquaredExponentialKernel(StationaryKernel):
    parameter_keys = ["length_scale", "sigma"]
        
    def compute_kernel_matrix(self, X1, X2=None):
        dist_matrix = self.compute_distance_matrix(X1, X2)
        kernel_matrix = self.sigma**2 * np.exp(-0.5 * dist_matrix**2 / self.length_scale**2)
        return kernel_matrix

class MaternKernel(StationaryKernel):
    parameter_keys = ["length_scale", "sigma", "nu"]

    def compute_kernel_matrix(self, X1, X2=None):
        distance_matrix = self.compute_distance_matrix(X1, X2)

        r = np.sqrt(2*self.nu) * distance_matrix / self.length_scale
        kernel_matrix = np.zeros_like(distance_matrix)

        # Set diagonal entries
        np.fill_diagonal(kernel_matrix, self.sigma**2)

        # Compute off-diagonal entries
        mask = (r > 0)
        kernel_matrix[mask] = self.sigma**2 * (2**(1-self.nu) / gammafnc(self.nu)) * (r[mask]**self.nu) * kv(self.nu, r[mask])

        # Set invalid entries to zero
        kernel_matrix[np.isnan(kernel_matrix) | np.isinf(kernel_matrix)] = 0

        return kernel_matrix
    
class PeriodicKernel(StationaryKernel):
    parameter_keys = ["length_scale", "sigma", "periodicity"]

    def compute_kernel_matrix(self, X1, X2=None):
        distance_matrix = self.compute_distance_matrix(X1, X2)

        r = np.sqrt(2) * np.sin(np.pi * distance_matrix / self.periodicity) / self.length_scale
        kernel_matrix = self.sigma**2 * np.exp(-1 * r**2)

        return kernel_matrix
    
class OverlapKernel(StationaryKernel):

    def compute_kernel_matrix(self, X1, X2=None):
        matrix = np.zeros((X1.shape[0],X2.shape[0]))
        n_categories = X1.shape[1]

        for row in range(matrix.shape[0]):
            for col in range(matrix.shape[1]):
                if (row <= col):
                    matrix[row][col] = np.count_nonzero(X1[row] == X2[col])/n_categories

        matrix = matrix + matrix.T

        for i in range(matrix.shape[0]):
            matrix[i][i] = matrix[i][i]/2

        return matrix