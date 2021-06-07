import numpy as np
from scipy import ndimage

ROI_SIZE_2D = 2
SHAPE_2D = (12, 12)

ROI_SIZE_3D = 2
SHAPE_3D = (12, 12, 12)


def multivariate_1D_simulation(n_samples=100, n_features=500,
                               support_size=10, sigma=1.0,
                               rho=0.0, shuffle=True, seed=0):
    """Generate 1D data with Toeplitz design matrix

    Parameters
    -----------
    n_samples : int
        Number of samples
    n_features : int
        Number of features
    support_size: int
        Size of the support
    sigma: float
        Standard deviation of the additive White Gaussian noise
    rho: float
        Level of correlation between neighboring features (if shuffle is False)
    shuffle : bool
        Shuffle the features (changing data structure) if True.
    seed: int
        Seed used for generating design matrix and noise

    Returns
    -------
    X: ndarray, shape (n_samples, n_features)
        Design matrix
    y: ndarray, shape (n_samples,)
        Target
    beta: ndarray, shape (n_features,)
        Parameter vector
    epsilon: ndarray, shape (n_samples,)
        Additive white Gaussian noise
    """

    rng = np.random.default_rng(seed)

    X = np.zeros((n_samples, n_features))
    X[:, 0] = rng.standard_normal(n_samples)

    for i in np.arange(1, n_features):
        epsilon = ((1 - rho ** 2) ** 0.5) * rng.standard_normal(n_samples)
        X[:, i] = rho * X[:, i - 1] + epsilon

    if shuffle:
        rng.shuffle(X.T)

    beta = np.zeros(n_features)
    beta[0:support_size] = 1.0

    epsilon = sigma * rng.standard_normal(n_samples)
    y = np.dot(X, beta) + epsilon

    return X, y, beta, epsilon


def generate_2D_weight(shape, roi_size):
    """Create a 2D weight map with four ROIs

    Parameters
    -----------
    shape : tuple (n_x, n_z)
        Shape of the data in the simulation
    roi_size : int
        Size of the edge of the ROIs

    Returns
    -------
    w : ndarray, shape (n_x, n_z)
        2D weight map
    """

    w = np.zeros(shape + (5,))
    w[0:roi_size, 0:roi_size, 0] = 1.0
    w[-roi_size:, -roi_size:, 1] = 1.0
    w[0:roi_size, -roi_size:, 2] = 1.0
    w[-roi_size:, 0:roi_size, 3] = 1.0

    return w


def generate_3D_weight(shape, roi_size):
    """Create a 3D weight map with five ROIs

    Parameters
    -----------
    shape : tuple (n_x, n_y, n_z)
        Shape of the data in the simulation
    roi_size : int
        Size of the edge of the ROIs

    Returns
    -------
    w : ndarray, shape (n_x, n_y, n_z)
        3D weight map
    """

    w = np.zeros(shape + (5,))
    w[0:roi_size, 0:roi_size, 0:roi_size, 0] = -1.0
    w[-roi_size:, -roi_size:, 0:roi_size, 1] = 1.0
    w[0:roi_size, -roi_size:, -roi_size:, 2] = -1.0
    w[-roi_size:, 0:roi_size, -roi_size:, 3] = 1.0
    w[(shape[0] - roi_size) // 2:(shape[0] + roi_size) // 2,
      (shape[1] - roi_size) // 2:(shape[1] + roi_size) // 2,
      (shape[2] - roi_size) // 2:(shape[2] + roi_size) // 2, 4] = 1.0
    return w


def multivariate_simulation(n_samples=100,
                            shape=SHAPE_2D,
                            roi_size=ROI_SIZE_2D,
                            sigma=1.0,
                            smooth_X=1.0,
                            return_shaped_data=True,
                            seed=0):
    """Generate a multivariate simulation with 2D or 3D data

    Parameters
    -----------
    n_samples : int
        Number of samples
    shape : tuple (n_x, n_y) or (n_x, n_y, n_z)
        Shape of the data in the simulation
    roi_size : int
        Size of the edge of the ROIs
    sigma: float
        Standard deviation of the additive white Gaussian noise
    smooth_X : float
        Level of smoothing using a Gaussian filter
    return_shaped_data : bool
        If true, the function returns shaped data and weight map
    seed: int
        Seed used for generating design matrix and noise

    Returns
    -------
    X: ndarray, shape (n_samples, n_features)
        Design matrix
    y: ndarray, shape (n_samples,)
        Target
    beta: ndarray, shape (n_features,)
        Parameter vector (flattened weight map)
    epsilon: ndarray, shape (n_samples,)
        Additive white Gaussian noise
    X_: ndarray, shape (n_samples, n_x, n_y) or (n_samples, n_x, n_y, n_z)
        Reshaped design matrix
    w : ndarray, shape (n_x, n_y) or (n_x, n_y, n_z)
        2D or 3D weight map
    """

    rng = np.random.default_rng(seed)

    if len(shape) == 2:
        w = generate_2D_weight(shape, roi_size)
    elif len(shape) == 3:
        w = generate_3D_weight(shape, roi_size)

    beta = w.sum(-1).ravel()
    X_ = rng.standard_normal((n_samples,) + shape)
    X = []

    for i in np.arange(n_samples):
        Xi = ndimage.filters.gaussian_filter(X_[i], smooth_X)
        X.append(Xi.ravel())

    X = np.asarray(X)
    X_ = X.reshape((n_samples,) + shape)

    epsilon = sigma * rng.standard_normal(n_samples)
    y = np.dot(X, beta) + epsilon

    if return_shaped_data:
        return X, y, beta, epsilon, X_, w

    return X, y, beta, epsilon
