import numpy as np
import scipy.stats as st
from joblib import Parallel, delayed
from sklearn.linear_model import Lasso, LassoCV

from .noise_std import reid


def _compute_residuals(X, column_index, alpha=None, Gram=None, max_iter=5000,
                       tol=1e-3, method='lasso', c=0.01):
    """Nodewise Lasso"""

    n_samples, n_features = X.shape
    i = column_index

    X_new = np.delete(X, i, axis=1)
    y = np.copy(X[:, i])

    if method == 'lasso' and Gram is None:
        Gram_loc = np.dot(X_new.T, X_new)
    elif method == 'lasso' and Gram is not None:
        Gram_loc = np.delete(np.delete(Gram, i, axis=0), i, axis=1)

    if method == 'lasso' and alpha is None:
        k = c * (1. / n_samples)
        alpha = k * np.max(np.abs(np.dot(X_new, y)))

    if method == 'lasso':
        clf_lasso_loc = Lasso(alpha=alpha, precompute=Gram_loc,
                              max_iter=max_iter, tol=tol)

    elif method == 'lasso_cv':
        clf_lasso_loc = LassoCV(max_iter=max_iter, tol=tol, cv=3)

    if method in ['lasso', 'lasso_cv']:
        clf_lasso_loc.fit(X_new, y)
        z = y - clf_lasso_loc.predict(X_new)

    omega_diag = n_samples * np.sum(z ** 2) / np.dot(y, z) ** 2

    return z, omega_diag


def desparsified_lasso_confint(X, y, confidence=0.95, max_iter=5000,
                               tol=1e-3, method="lasso", c=0.01, n_jobs=1):
    """Desparsified Lasso with confidence intervals

    Parameters
    -----------
        X : ndarray or scipy.sparse matrix, (n_samples, n_features)
            Data
        y : ndarray, shape (n_samples,) or (n_samples, n_targets)
            Target. Will be cast to X's dtype if necessary
        confidence : float, optional
            Confidence level used to compute the confidence intervals.
            Each value should be in the range [0, 1].
        tol : float, optional
            The tolerance for the optimization: if the updates are
            smaller than ``tol``, the optimization code checks the
            dual gap for optimality and continues until it is smaller
            than ``tol``.
        method : string, optional
            The method for the nodewise lasso: "lasso" or "lasso_cv"
        c : float, optional
            Only used if method="lasso". Then alpha = c * alpha_max.
        n_jobs : int or None, optional (default=1)
            Number of CPUs to use during the cross validation.
        """

    X = np.asarray(X)

    n_samples, n_features = X.shape

    Z = np.zeros((n_samples, n_features))
    omega_diag = np.zeros(n_features)
    omega_invsqrt_diag = np.zeros(n_features)

    quantile = st.norm.ppf(1 - (1 - confidence) / 2)

    if method == "lasso":

        Gram = np.dot(X.T, X)

        k = c * (1. / n_samples)
        alphas = k * np.max(np.abs(Gram - np.diag(np.diag(Gram))), axis=0)

    else:

        Gram = None
        alphas = None

    # Calculating Omega Matrix
    results = \
        Parallel(n_jobs=n_jobs)(
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

    omega_invsqrt_diag = omega_diag ** (-0.5)

    # Lasso regression
    clf_lasso_cv = LassoCV(cv=5, max_iter=max_iter, n_jobs=n_jobs)
    clf_lasso_cv.fit(X, y)
    beta_lasso = clf_lasso_cv.coef_

    # Estimating the coefficient vector
    beta_bias = y.T.dot(Z) / np.sum(X * Z, axis=0)

    P = ((Z.T.dot(X)).T / np.sum(X * Z, axis=0)).T
    P_nodiag = P - np.diag(np.diag(P))

    beta_hat = beta_bias - P_nodiag.dot(beta_lasso)

    sigma_hat = reid(X, y)

    confint_radius = np.abs(quantile * sigma_hat /
                            (np.sqrt(n_samples) * omega_invsqrt_diag))
    cb_max = beta_hat + confint_radius
    cb_min = beta_hat - confint_radius

    return beta_hat, cb_min, cb_max
