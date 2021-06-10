import numpy as np
from numpy.linalg import norm
from sklearn.linear_model import LassoCV
from sklearn.model_selection import KFold


def reid(X, y, eps=1e-2, tol=1e-4, max_iter=1e+4, n_jobs=1, seed=0):
    """Estimation of noise standard deviation using Reid procedure

    Parameters
    -----------
    X : ndarray or scipy.sparse matrix, shape (n_samples, n_features)
        Data.

    y : ndarray, shape (n_samples,) or (n_samples, n_targets)
        Target. Will be cast to X's dtype if necessary.

    tol : float, optional
        The tolerance for the optimization: if the updates are
        smaller than ``tol``, the optimization code checks the
        dual gap for optimality and continues until it is smaller
        than ``tol``.

    max_iter : int, optional
        The maximum number of iterations.

    n_jobs : int or None, optional (default=1)
        Number of CPUs to use during the cross validation.

    seed: int, optional (default=0)
        Seed passed in the KFold object which is used to cross-validate
        LassoCV. This seed controls the partitioning randomness.

    Returns
    -------
    sigma_hat : float
        Estimated noise standard deviation.

    beta_hat : array, shape (n_features,)
        Estimated parameter vector.
    """

    X = np.asarray(X)
    n_samples, n_features = X.shape

    if max_iter // 5 <= n_features:
        max_iter = n_features * 5
        print("'max_iter' has been increased to {}".format(max_iter))

    cv = KFold(n_splits=5, shuffle=True, random_state=seed)

    clf_lasso_cv = \
        LassoCV(eps=eps, normalize=False, fit_intercept=False,
                cv=cv, tol=tol, max_iter=max_iter, n_jobs=n_jobs)

    clf_lasso_cv.fit(X, y)
    beta_hat = clf_lasso_cv.coef_
    error = clf_lasso_cv.predict(X) - y
    coef_max = np.max(np.abs(beta_hat))

    if coef_max == 0:
        support = 0
    else:
        support = np.sum(np.abs(beta_hat) > tol * coef_max)

    # avoid dividing by 0
    if support >= n_samples:
        support = n_samples - 1

    sigma_hat = np.sqrt((1. / (n_samples - support)) * norm(error) ** 2)

    return sigma_hat, beta_hat


def empirical_snr(X, y, beta, epsilon=None):
    """Compute the SNR for the linear model: y = X beta + epsilon

    Parameters
    -----------
    X : ndarray or scipy.sparse matrix, (n_samples, n_features)
        Data
    y : ndarray, shape (n_samples,)
        Target. Will be cast to X's dtype if necessary
    beta : ndarray, shape (n_features,)
        True parameter vector.
    epsilon : ndarray, shape (n_samples,), opitonal (default=None)
        True error vector.
    """
    X = np.asarray(X)

    signal = np.dot(X, beta)

    if epsilon is None:
        epsilon = y - signal

    sig_signal = np.linalg.norm(signal - np.mean(signal))
    sig_noise = np.linalg.norm(epsilon - np.mean(epsilon))
    snr_hat = (sig_signal / sig_noise) ** 2

    return snr_hat
