import numpy as np
from numpy.linalg import norm
from sklearn.linear_model import LassoCV, LassoLarsCV


def reid(X, y, method="lars", tol=1e-4, max_iter=1e+3, n_jobs=1):
    """Estimation of noise standard deviation using Reid procedure

    Parameters
    -----------
    X : ndarray or scipy.sparse matrix, (n_samples, n_features)
        Data
    y : ndarray, shape (n_samples,) or (n_samples, n_targets)
        Target. Will be cast to X's dtype if necessary
    method : string, optional
        The method for the CV-lasso: "lars" or "lasso"
    tol : float, optional
        The tolerance for the optimization: if the updates are
        smaller than ``tol``, the optimization code checks the
        dual gap for optimality and continues until it is smaller
        than ``tol``.
    max_iter : int, optional
        The maximum number of iterations
    n_jobs : int or None, optional (default=1)
        Number of CPUs to use during the cross validation.
    """

    X = np.asarray(X)
    n_samples, n_features = X.shape

    if int(max_iter / 5) <= n_features:
        max_iter = n_features * 5

    if method == "lars":
        clf_lars_cv = LassoLarsCV(max_iter=max_iter, normalize=False, cv=3,
                                  n_jobs=n_jobs)
        clf_lars_cv.fit(X, y)
        error = clf_lars_cv.predict(X) - y
        support = sum(clf_lars_cv.coef_ != 0)

    elif method == "lasso":
        clf_lasso_cv = LassoCV(tol=tol, max_iter=max_iter, cv=3, n_jobs=n_jobs)
        clf_lasso_cv.fit(X, y)
        error = clf_lasso_cv.predict(X) - y
        support = sum(clf_lasso_cv.coef_ != 0)

    sigma_hat = np.sqrt((1. / (n_samples - support)) * norm(error) ** 2)

    return sigma_hat
