import numpy as np

# General linear algebra operation implementations:

def invert_covariance(cov, alpha=1e-9):
    """Inverts a covariance matrix that is not necessarily positive-definite. The alpha parameter is added to the diagonal before invertion.
    """
    cov = cov + alpha*np.eye(cov.shape[0])
    # Compute the Cholesky decomposition of the covariance matrix
    L = np.linalg.cholesky(cov)
    # Invert the lower triangular matrix using forward substitutions
    Linv = np.linalg.solve(L, np.eye(len(cov)))
    # Invert the upper triangular matrix using backward substitutions
    cov_inv = np.dot(Linv.T, Linv)
    return cov_inv