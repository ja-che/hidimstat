import numpy as np
from numpy.linalg import norm
from sklearn.svm import LinearSVR
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline


def standardized_svr(X, y, Cs=None, return_C=False, n_jobs=1):
    """Cross-validated SVR

    Parameters
    -----------
    X : ndarray or scipy.sparse matrix, (n_samples, n_features)
        Data.

    y : ndarray, shape (n_samples,) or (n_samples, n_targets)
        Target. Will be cast to X's dtype if necessary.

    Cs : ndarray, default=None
        List of Cs where to compute the models.
        If None, Cs are set automatically.

    return_C : bool, default=False
        If True, we return the hyper-parameter selected by cross-validation.

    n_jobs : int or None, optional (default=1)
        Number of CPUs to use during the cross validation.

    Returns
    -------
    beta_hat : array, shape (n_features,)
        Estimated parameter vector.

    scale : ndarray, shape (n_features,)
        Value of the standard deviation of the parameters.

    C : float
        If return_C is True, the hyper-parameter selected
        by cross-validation 'C' is returned.
    """

    n_samples, n_features = X.shape

    if Cs is None:
        Cs = np.logspace(-7, 1, 9)

    steps = [('SVR', LinearSVR())]
    pipeline = Pipeline(steps)
    parameters = {'SVR__C': Cs}

    grid = GridSearchCV(pipeline, param_grid=parameters, n_jobs=n_jobs)
    grid.fit(X, y)

    C = grid.best_params_['SVR__C']
    beta_hat = grid.best_estimator_.named_steps['SVR'].coef_

    std = norm(beta_hat) / np.sqrt(n_features)
    scale = std * np.ones(beta_hat.size)

    if return_C:
        return beta_hat, scale, C

    return beta_hat, scale
