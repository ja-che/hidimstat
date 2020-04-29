import numpy as np
from numpy.linalg import norm
from sklearn.svm import LinearSVR
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline


def standardized_svr(X, y, Cs=np.logspace(-7, 1, 9), return_C=False, n_jobs=1):

    n_samples, n_features = X.shape

    steps = [('SVR', LinearSVR())]

    pipeline = Pipeline(steps)

    parameters = {'SVR__C': Cs}

    grid = GridSearchCV(pipeline, param_grid=parameters, n_jobs=n_jobs)
    grid.fit(X, y)

    C = grid.best_params_['SVR__C']

    clf_svr = LinearSVR(C=C)
    clf_svr.fit(X, y)

    beta_hat = clf_svr.coef_
    std = norm(beta_hat) / np.sqrt(n_features)

    scale = std * np.ones(beta_hat.size)

    if return_C:
        return beta_hat, scale, C

    return beta_hat, scale
