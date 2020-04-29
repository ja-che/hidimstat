import numpy as np


def gaonkar(X, y, rcond=1e-3):
    """Gaonkar 2013 procedure

    Parameters
    -----------
        X : ndarray or scipy.sparse matrix, (n_samples, n_features)
            Data
        y : ndarray, shape (n_samples,) or (n_samples, n_targets)
            Target. Will be cast to X's dtype if necessary
        rcond : float, optional
            Cutoff for small singular values.
            Singular values smaller (in modulus) than
            `rcond` * largest_singular_value (again, in modulus)
            are set to zero. Broadcasts against the stack of matrices
        """

    X = np.asarray(X)
    n_samples, n_features = X.shape

    K = _manual_inverting(np.dot(X, X.T), rcond=rcond)
    sum_K = np.sum(K)

    L = - np.outer(np.sum(K, axis=0), np.sum(K, axis=1)) / sum_K
    C = np.dot(X.T, K + L)

    beta_hat = np.dot(C, y)

    scale_beta = np.sqrt(np.sum(C ** 2, axis=1))

    return beta_hat, scale_beta


def _manual_inverting(X, rcond=1e-3, full_rank=False):
    'Inverting taking care of low eigenvalues to increase numerical stability'

    X = np.asarray(X)
    n_samples, n_features = X.shape

    if n_samples != n_features:
        raise ValueError('The matrix is not a square matrix')

    U, s, V = np.linalg.svd(X, full_matrices=False)
    rank = np.sum(s > rcond * s.max())
    s_inv = np.zeros(np.size(s))
    s_inv[:rank] = 1 / s[:rank]

    if full_rank:
        s_inv[rank:] = 1 / (rcond * s.max())

    X_inv = np.linalg.multi_dot([U, np.diag(s_inv), V])

    return X_inv
