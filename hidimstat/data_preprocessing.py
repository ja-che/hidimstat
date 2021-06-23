import numpy as np
from mne.inverse_sparse.mxne_inverse import _prepare_gain


def preprocess_meg_eeg_data(evoked, forward, noise_cov, loose=0., depth=0.,
                            pca=False, rank=None):
    """Preprocess MEG or EEG data to produce the whitened MEG/EEG measurements
    (target) and the preprocessed gain matrix (design matrix).

    Parameters
    ----------
    evoked : instance of mne.Evoked
        The evoked data.

    forward : instance of Forward
        The forward solution.

    noise_cov : instance of Covariance
        The noise covariance.

    loose : float in [0, 1] or 'auto'
        Value that weights the source variances of the dipole components
        that are parallel (tangential) to the cortical surface. If loose
        is 0 then the solution is computed with fixed orientation.
        If loose is 1, it corresponds to free orientations.
        The default value ('auto') is set to 0.2 for surface-oriented source
        space and set to 1.0 for volumic or discrete source space.

    depth : None or float in [0, 1]
        Depth weighting coefficients. If None, no depth weighting is performed.

    pca : bool, optional (default=False)
        If True, Whitener is reduced.
        If False, Whitener is not reduced.

    rank : None or int
        Rank reduction of the whitener. If None rank is estimated from data.



    Returns
    -------

    """

    all_ch_names = evoked.ch_names

    # Handle depth weighting and whitening (here is no weights)
    forward, gain, gain_info, whitener, source_weighting, mask = \
        _prepare_gain(forward, evoked.info, noise_cov, pca=pca, depth=depth,
                      loose=loose, weights=None, weights_min=None, rank=rank)

    # Select channels of interest
    sel = [all_ch_names.index(name) for name in gain_info['ch_names']]

    M = evoked.data[sel]
    M = np.dot(whitener, M)

    return gain, M
