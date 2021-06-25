"""
Support recovery on MEG data (2D)
========================================

"""

import numpy as np
import mne
from mne.datasets import sample
from mne.inverse_sparse.mxne_inverse import _prepare_gain, _make_sparse_stc
from sklearn.cluster import FeatureAgglomeration
from sklearn.cluster._agglomerative import _fix_connectivity

from hidimstat.clustered_inference import clustered_inference
from hidimstat.ensemble_clustered_inference import \
    ensemble_clustered_inference
from hidimstat.stat_tools import zscore_from_pval


def preprocess_meg_eeg_data(evoked, forward, noise_cov, loose=0., depth=0.,
                            pca=False, rank=None):
    """Preprocess MEG or EEG data to produce the whitened MEG/EEG measurements
    (target) and the preprocessed gain matrix (design matrix). This function
    is mainly wrapping the `_prepare_gain` MNE function.

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
    G : array, shape (n_channels, n_dipoles)
        The preprocessed gain matrix.

    M : array, shape (n_channels, n_times)
        The whitened MEG/EEG measurements.

    forward : instance of Forward
        The preprocessed forward solution.
    """

    all_ch_names = evoked.ch_names

    # Handle depth weighting and whitening (here is no weights)
    forward, G, gain_info, whitener, source_weighting, mask = \
        _prepare_gain(forward, evoked.info, noise_cov, pca=pca, depth=depth,
                      loose=loose, weights=None, weights_min=None, rank=rank)

    # Select channels of interest
    sel = [all_ch_names.index(name) for name in gain_info['ch_names']]

    M = evoked.data[sel]
    M = np.dot(whitener, M)

    return G, M, forward


def _compute_stc(zscore_active_set, active_set, evoked, forward):
    """Wrapper of `_make_sparse_stc`"""

    X = np.atleast_2d(zscore_active_set)

    if X.shape[1] > 1 and X.shape[0] == 1:
        X = X.T

    stc = _make_sparse_stc(X, active_set, forward, tmin=evoked.times[0],
                           tstep=1. / evoked.info['sfreq'])
    return stc


# Downloading data
subject = 'sample'
data_path = sample.data_path()
fwd_fname = data_path + '/MEG/sample/sample_audvis-meg-eeg-oct-6-fwd.fif'
ave_fname = data_path + '/MEG/sample/sample_audvis-ave.fif'
raw_fname = data_path + '/MEG/sample/sample_audvis_raw.fif'
cov_fname = data_path + '/MEG/sample/sample_audvis-shrunk-cov.fif'
subjects_dir = data_path + '/subjects'
condition = 'Left Auditory'

# Read noise covariance matrix
noise_cov = mne.read_cov(cov_fname)

# Handling average file
evoked = mne.read_evokeds(ave_fname, condition=condition, baseline=(None, 0))
evoked = evoked.pick_types('grad')
# Selecting relevant time window
evoked.plot()
evoked.crop(tmin=0.05, tmax=0.1)

# Handling forward solution
forward = mne.read_forward_solution(fwd_fname)
# Collecting features' connectivity
connectivity = mne.source_estimate.spatial_src_adjacency(forward['src'])

# Choosing frequency and number of clusters used for compression.
# Reducing the frequency to 100Hz to make inference faster
t_step = 0.01
step = int(t_step * evoked.info['sfreq'])
evoked.decimate(step)
t_min = evoked.times[0]
t_step = 1. / evoked.info['sfreq']
# Taking n_clusters > 2000 might lead to an unpowerfull inference.
# Taking n_clusters < 500 might compress too much the data leading
# to a compress problem not close enough to the original problem.
# For MEG data n_clusters = 1000 is generally a good default choice.
n_clusters = 1000
# Setting theoretical FWER target
fwer_target = 0.1
correction_clust_inf = 1. / n_clusters
zscore_target = zscore_from_pval((fwer_target / 2) * correction_clust_inf)


# Preprocessing MEG data
X, Y, forward = preprocess_meg_eeg_data(evoked, forward, noise_cov)

# Initializing FeatureAgglomeration object used for the clustering step
connectivity_fixed, _ = \
    _fix_connectivity(X.T, connectivity, affinity="euclidean")
ward = FeatureAgglomeration(n_clusters=n_clusters, connectivity=connectivity)

# Making the inference with the clustered inference algorithm
inference_method = 'desparsified-group-lasso'
beta_hat, pval, pval_corr, one_minus_pval, one_minus_pval_corr = \
    clustered_inference(X, Y, ward, n_clusters, method=inference_method)

# Extracting active set (support)
active_set = np.logical_or(pval_corr < fwer_target / 2,
                           one_minus_pval_corr < fwer_target / 2)
active_set_full = np.copy(active_set)
active_set_full[:] = True

# Computing z-scores
zscore = zscore_from_pval(pval, one_minus_pval)
zscore_active_set = zscore[active_set]

# Building mne.SourceEstimate object
stc = _compute_stc(zscore_active_set, active_set, evoked, forward)

# Plotting
mne.viz.set_3d_backend("pyvista")
max_stc = np.max(np.abs(stc._data))
clim = dict(pos_lims=(3, zscore_target, max_stc), kind='value')
brain = stc.plot(subject=subject, hemi='lh', clim=clim,
                 subjects_dir=subjects_dir)
brain.show_view('lat')
brain.add_text(0.05, 0.9, 'audio - cd-MTLasso (AR1)', 'title', font_size=30)
brain.save_image('figures/meg_audio_cd-MTLasso.png')

interactive_plot = False
if interactive_plot:
    brain = stc.plot(subject=subject, hemi='both',
                     subjects_dir=subjects_dir, clim=clim)

# Runing the ensmeble clustered inference algorithm on temporal data
# might take several minutes on standard device with `n_jobs=1` (around 10 mn)
run_ensemble_clustered_inference = False
if run_ensemble_clustered_inference:

    # Making the inference with the ensembled clustered inference algorithm
    beta_hat, pval, pval_corr, one_minus_pval, one_minus_pval_corr = \
        ensemble_clustered_inference(X, Y, ward, n_clusters,
                                     inference_method=inference_method)

    # Extracting active set (support)
    active_set = np.logical_or(pval_corr < fwer_target / 2,
                               one_minus_pval_corr < fwer_target / 2)
    active_set_full = np.copy(active_set)
    active_set_full[:] = True

    # Computing z-scores
    zscore = zscore_from_pval(pval, one_minus_pval)
    zscore_active_set = zscore[active_set]

    # Building mne.SourceEstimate object
    stc = _compute_stc(zscore_active_set, active_set, evoked, forward)

    # Plotting
    mne.viz.set_3d_backend("pyvista")
    max_stc = np.max(np.abs(stc._data))
    clim = dict(pos_lims=(3, zscore_target, max_stc), kind='value')
    brain = stc.plot(subject=subject, hemi='lh', clim=clim,
                     subjects_dir=subjects_dir)
    brain.show_view('lat')
    brain.add_text(0.05, 0.9, 'audio - ecd-MTLasso (AR1)',
                   'title', font_size=30)

    interactive_plot = False
    if interactive_plot:
        brain = stc.plot(subject=subject, hemi='both',
                         subjects_dir=subjects_dir, clim=clim)
