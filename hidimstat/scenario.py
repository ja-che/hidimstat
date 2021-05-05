import numpy as np


def design_matrix_toeplitz_cov(seed=0, n=100, p=500, rho=0.0):
    """Generate a design matrix with a 1D structure and Toeplitz covariance
    matrix with a more efficient code

    Parameters
    -----------
    n : int
        Number of samples
    p : int
        Number of features
    rho: float
        Level of correlation between neighboring features
    """

    np.random.seed(seed)

    X = np.zeros((n, p))
    X[:, 0] = np.random.randn(n)

    for i in np.arange(1, p):
        epsilon = ((1 - rho ** 2) ** 0.5) * np.random.randn(n)
        X[:, i] = rho * X[:, i - 1] + epsilon

    return X


def scenario(scenario='Toeplitz', seed=0, n_samples=100, n_features=500,
             effect_small=0.25, effect_medium=0.5, effect_large=1,
             effect_s_nb=4, effect_m_nb=4, effect_l_nb=4, sigma=1,
             rho=0.0, shuffle=True):
    """Generate the 1D data with 3 levels for the weight vector

    Parameters
    -----------
    scenario : string
        Type of scenario generated
    n_samples : int
        Number of samples
    n_features : int
        Number of features
    effect_small : float
        Value of the low non zero coefficients of the weight vector
    effect_medium : float
        Value of the medium non zero coefficients of the weight vector
    effect_large : float
        Value of the high non zero coefficients of the weight vector
    effect_s_nb : int
        Number of coefficients with a low non zero value
    effect_m_nb : int
        Number of coefficients with a medium non zero value
    effect_l_nb : int
        Number of coefficients with a high non zero value
    sigma: float
        Standard deviation of the additive White Gaussian noise
    rho: float
        Level of correlation between neighboring features
    shuffle : boolean
        Shuffle the features (changing data structure) if True.
    """

    if scenario == 'Toeplitz':
        X = design_matrix_toeplitz_cov(seed, n_samples, n_features, rho)

    if shuffle:
        np.random.shuffle(X.T)

    nb_effects = effect_s_nb + effect_m_nb + effect_l_nb

    beta = np.zeros(n_features)
    beta[0:effect_s_nb] = effect_small
    beta[effect_s_nb:effect_s_nb + effect_m_nb] = effect_medium
    beta[effect_s_nb + effect_m_nb:nb_effects] = effect_large

    epsilon = sigma * np.random.randn(n_samples)

    y = np.dot(X, beta) + epsilon

    return y, beta, X, epsilon
