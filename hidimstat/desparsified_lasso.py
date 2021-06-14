import numpy as np
import scipy.stats as st
from joblib import Parallel, delayed
from sklearn.utils.validation import check_memory
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso

from .noise_std import reid


def _compute_all_residuals(X, alphas, Gram=None, max_iter=5000, tol=1e-3,
                           method='lasso', c=0.01, n_jobs=1, verbose=0):
    """Nodewise Lasso. Compute all the residuals: regressing each column of the
    design matrix against the other columns"""

    n_samples, n_features = X.shape

    results = \
        Parallel(n_jobs=n_jobs, verbose=verbose)(
            delayed(_compute_residuals)
                (X=X,
                 column_index=i,
                 alpha=alphas[i],
                 Gram=Gram,
                 max_iter=max_iter,
                 tol=tol,
                 method=method,
                 c=c)
            for i in range(n_features))

    results = np.asarray(results)
    Z = np.stack(results[:, 0], axis=1)
    omega_diag = np.stack(results[:, 1])

    return Z, omega_diag


def _compute_residuals(X, column_index, alpha=None, Gram=None, max_iter=5000,
                       tol=1e-3, method='lasso', c=0.01):
    """Compute the residuals of the regression of a given column of the
    design matrix against the other columns"""

    n_samples, n_features = X.shape
    i = column_index

    X_new = np.delete(X, i, axis=1)
    y = np.copy(X[:, i])

    if method != 'lasso':
        ValueError("The only regression method available is 'lasso'")

    if Gram is None:
        Gram_loc = np.dot(X_new.T, X_new)
    else:
        Gram_loc = np.delete(np.delete(Gram, i, axis=0), i, axis=1)

    if alpha is None:
        k = c * (1. / n_samples)
        alpha = k * np.max(np.abs(np.dot(X_new, y)))

    clf_lasso_loc = Lasso(alpha=alpha, precompute=Gram_loc, max_iter=max_iter,
                          tol=tol)

    clf_lasso_loc.fit(X_new, y)
    z = y - clf_lasso_loc.predict(X_new)

    omega_diag_i = n_samples * np.sum(z ** 2) / np.dot(y, z) ** 2

    return z, omega_diag_i


def desparsified_lasso_confint(X, y, normalize=True, dof_ajdustement=False,
                               confidence=0.95, max_iter=5000, tol=1e-3,
                               residual_method='lasso', c=0.01, n_jobs=1,
                               memory=None, verbose=0):

    """Desparsified Lasso with confidence intervals

    Parameters
    -----------
    X : ndarray, shape (n_samples, n_features)
        Data.

    y : ndarray, shape (n_samples,)
        Target.

    normalize : bool, optional (default=True)
        If True, the regressors X will be normalized before regression and
        the target y will be centered.

    dof_ajdustement : bool, optional (default=False)
        If True, makes the degrees of freedom adjustement (cf. [4]_ and [5]_).
        Otherwise, the original Desparsified Lasso estimator is computed
        (cf. [1]_ and [2]_ and [3]_).

    confidence : float, optional (default=0.95)
        Confidence level used to compute the confidence intervals.
        Each value should be in the range [0, 1].

    max_iter : int, optional (default=5000)
        The maximum number of iterations when regressing, by Lasso,
        each column of the design matrix against the others.

    tol : float, optional (default=1e-3)
        The tolerance for the optimization of the Lasso problems: if the
        updates are smaller than `tol`, the optimization code checks the
        dual gap for optimality and continues until it is smaller than `tol`.

    residual_method : string, optional (default='lasso')
        The method for the computind the residuals of the Nodewise Lasso.
        Currently the only method available is 'lasso'.

    c : float, optional (default=0.01)
        Only used if method='lasso'. Then alpha = c * alpha_max.

    n_jobs : int or None, optional (default=1)
        Number of CPUs to use during the Nodewise Lasso.

    memory : str or joblib.Memory object, optional (default=None)
        Used to cache the output of the computation of the Nodewise Lasso.
        By default, no caching is done. If a string is given, it is the path
        to the caching directory.

    verbose: int, optional (default=1)
        The verbosity level: if non zero, progress messages are printed
        when computing the Nodewise Lasso in parralel.
        The frequency of the messages increases with the verbosity level.

    Returns
    -------
    beta_hat : array, shape (n_features,)
        Estimated parameter vector.

    cb_min : array, shape (n_features)
        Lower bound of the confidence intervals on the parameter vector.

    cb_max : array, shape (n_features)
        Upper bound of the confidence intervals on the parameter vector.
    """

    X = np.asarray(X)

    n_samples, n_features = X.shape

    Z = np.zeros((n_samples, n_features))
    omega_diag = np.zeros(n_features)
    omega_invsqrt_diag = np.zeros(n_features)

    quantile = st.norm.ppf(1 - (1 - confidence) / 2)

    memory = check_memory(memory)

    if normalize:

        y = y - np.mean(y)
        X = StandardScaler().fit_transform(X)
        Gram = np.dot(X.T, X)

        k = c * (1. / n_samples)
        alphas = k * np.max(np.abs(Gram - np.diag(np.diag(Gram))), axis=0)

    else:

        Gram = None
        alphas = None

    # Calculating precision matrix (Nodewise Lasso)
    Z, omega_diag = memory.cache(_compute_all_residuals, ignore=['n_jobs'])(
        X, alphas, Gram=Gram, max_iter=max_iter, tol=tol,
        method=residual_method, c=c, n_jobs=n_jobs, verbose=verbose)

    # Lasso regression
    sigma_hat, beta_lasso = reid(X, y, n_jobs=n_jobs)

    # Computing the degrees of freedom adjustement
    if dof_ajdustement:
        coef_max = np.max(np.abs(beta_lasso))
        support = np.sum(np.abs(beta_lasso) > 0.01 * coef_max)
        support = min(support, n_samples - 1)
        dof_factor = n_samples / (n_samples - support)
    else:
        dof_factor = 1

    # Computing Desparsified Lasso estimator and confidence intervals
    beta_bias = dof_factor * np.dot(y.T, Z) / np.sum(X * Z, axis=0)

    P = ((Z.T.dot(X)).T / np.sum(X * Z, axis=0)).T
    P_nodiag = P - np.diag(np.diag(P))
    Id = np.identity(n_features)
    P_nodiag = dof_factor * P_nodiag + (dof_factor - 1) * Id

    beta_hat = beta_bias - P_nodiag.dot(beta_lasso)

    omega_diag = omega_diag * dof_factor ** 2
    omega_invsqrt_diag = omega_diag ** (-0.5)

    confint_radius = np.abs(quantile * sigma_hat /
                            (np.sqrt(n_samples) * omega_invsqrt_diag))
    cb_max = beta_hat + confint_radius
    cb_min = beta_hat - confint_radius

    return beta_hat, cb_min, cb_max
