import numpy as np
from numpy.linalg import norm
from scipy.linalg import toeplitz, solve
from sklearn.linear_model import LassoCV, MultiTaskLassoCV
from sklearn.model_selection import KFold


def reid(X, y, eps=1e-2, tol=1e-4, max_iter=1e+4, n_jobs=1, seed=0):
    """Estimation of noise standard deviation using Reid procedure

    Parameters
    -----------
    X : ndarray or scipy.sparse matrix, shape (n_samples, n_features)
        Data.

    y : ndarray, shape (n_samples,)
        Target.

    eps: float, optional (default=1e-2)
        Length of the cross-validation path.
        eps=1e-2 means that alpha_min / alpha_max = 1e-2.

    tol : float, optional (default=1e-4)
        The tolerance for the optimization: if the updates are
        smaller than ``tol``, the optimization code checks the
        dual gap for optimality and continues until it is smaller
        than ``tol``.

    max_iter : int, optional (default=1e+4)
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

    sigma_hat = norm(error) / np.sqrt(n_samples - support)

    return sigma_hat, beta_hat


def group_reid(X, Y, fit_Y=True, stationary=True, method='simple', order=1,
               eps=1e-2, tol=1e-4, max_iter=1e+4, n_jobs=1, seed=0):

    """Estimation of the covariance matrix using group Reid procedure

    Parameters
    -----------
    X : ndarray or scipy.sparse matrix, shape (n_samples, n_features)
        Data.

    Y : ndarray, shape (n_samples, n_targets)
        Target.

    fit_Y : bool, optional (default=True)
        If True, Y will be regressed against X by MultiTaskLassoCV
        and the covariance matrix is estimated on the residuals.
        Otherwise, covariance matrix is estimated directly on Y.

    stationary : bool, optional (default=True)
        If True, noise is considered to have the same magnitude at each
        time point. Otherwise, magnitude of the noise is not constant.

    method : bool, optional (default='simple')
        If 'simple', the correlation matrix is estimated by taking the
        median of the correlation between two consecutive time points
        and the noise standard deviation at each time point is estimated
        by taking the median of the standard deviations for every time steps.
        If 'AR', the order of the AR model is given by `order` and the
        Yule-Walker method is used to estimate the covariance matrix.

    eps : float, optional (default=1e-2)
        Length of the cross-validation path.
        eps=1e-2 means that alpha_min / alpha_max = 1e-2.

    tol : float, optional (default=1e-4)
        The tolerance for the optimization: if the updates are
        smaller than ``tol``, the optimization code checks the
        dual gap for optimality and continues until it is smaller
        than ``tol``.

    max_iter : int, optional (default=1e+4)
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
    n_targets = Y.shape[1]

    if method == 'simple':
        print('Group reid: simple cov estimation')
    else:
        print('Group reid: ' + method + str(order) + ' cov estimation')

    if (max_iter // 5) <= n_features:
        max_iter = n_features * 5
        print("'max_iter' has been increased to {}".format(max_iter))

    cv = KFold(n_splits=5, shuffle=True, random_state=seed)

    if fit_Y:

        clf_mtlcv = \
            MultiTaskLassoCV(eps=eps, normalize=False, fit_intercept=False,
                             cv=cv, tol=tol, max_iter=max_iter,
                             n_jobs=n_jobs)

        clf_mtlcv.fit(X, Y)
        Beta_hat = clf_mtlcv.coef_
        Error = clf_mtlcv.predict(X) - Y
        coef_max = np.max(np.abs(Beta_hat))
        row_max = np.max(np.sum(np.abs(Beta_hat), axis=0))

        if coef_max == 0:
            support = 0
        else:
            support = np.sum(np.sum(np.abs(Beta_hat), axis=0) > tol * row_max)

    else:

        Beta_hat = np.zeros((n_features, n_targets))
        Error = np.copy(Y)
        support = 0

    # avoid dividing by 0
    if support >= n_samples:
        support = n_samples - 1

    sigma_hat_raw = norm(Error, axis=0) / np.sqrt(n_samples - support)

    if stationary:
        sigma_hat = np.median(sigma_hat_raw) * np.ones(n_targets)
        Corr_emp = np.corrcoef(Error.T)
    else:
        sigma_hat = sigma_hat_raw
        Error_resc = Error / sigma_hat
        Corr_emp = np.corrcoef(Error_resc.T)

    # Median method
    if not stationary or method == 'simple':

        rho_hat = np.median(np.diag(Corr_emp, 1))
        Corr_hat = \
            toeplitz(np.geomspace(1, rho_hat ** (n_targets - 1), n_targets))
        Cov_hat = np.outer(sigma_hat, sigma_hat) * Corr_hat

    # Yule-Walker method
    elif stationary and method == 'AR':

        if order > n_targets - 1:
            raise ValueError('The requested AR order is to high with ' +
                             'respect to the number of time points.')

        rho_ar = np.zeros(order + 1)
        rho_ar[0] = 1

        for i in range(1, order + 1):
            rho_ar[i] = np.median(np.diag(Corr_emp, i))

        A = toeplitz(rho_ar[:-1])
        coef_ar = solve(A, rho_ar[1:])

        Error_estimate = np.zeros((n_samples, n_targets - order))

        for i in range(order):
            # time window used to estimate the error from AR model
            start = order - i - 1
            end = - i - 1
            Error_estimate += coef_ar[i] * Error[:, start:end]

        epsilon = Error[:, order:] - Error_estimate
        sigma_eps = np.median(norm(epsilon, axis=0) / np.sqrt(n_samples))

        rho_ar_full = np.zeros(n_targets)
        rho_ar_full[:order+1] = rho_ar

        for i in range(order + 1, n_targets):
            start = i - order
            end = i
            rho_ar_full[i] = np.dot(coef_ar[::-1], rho_ar_full[start:end])

        Corr_hat = toeplitz(rho_ar_full)
        sigma_hat[:] = sigma_eps / np.sqrt((1 - np.dot(coef_ar, rho_ar[1:])))
        Cov_hat = np.outer(sigma_hat, sigma_hat) * Corr_hat

    else:
        raise ValueError('Unknown method for estimating the covariance matrix')

    return Cov_hat, Beta_hat


def empirical_snr(X, y, beta, epsilon=None):
    """Compute the SNR for the linear model: y = X beta + epsilon

    Parameters
    -----------
    X : ndarray or scipy.sparse matrix, shape (n_samples, n_features)
        Data.
    y : ndarray, shape (n_samples,)
        Target.
    beta : ndarray, shape (n_features,)
        True parameter vector.
    epsilon : ndarray, shape (n_samples,), optional (default=None)
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
