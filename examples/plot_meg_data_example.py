"""
Support recovery on MEG data (2D)
========================================

"""

import os
import numpy as np
import mne
from mne.datasets import sample, somato
from mne.inverse_sparse.mxne_inverse import _prepare_gain, _make_sparse_stc
from mne.minimum_norm import make_inverse_operator, apply_inverse
from sklearn.cluster import FeatureAgglomeration
from sklearn.cluster._agglomerative import _fix_connectivity

from hidimstat.clustered_inference import clustered_inference
from hidimstat.ensemble_clustered_inference import \
    ensemble_clustered_inference
from hidimstat.stat_tools import zscore_from_pval


def preprocess_meg_eeg_data(evoked, forward, noise_cov, loose=0., depth=0.,
                            pca=False):
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
                      loose=loose, weights=None, weights_min=None, rank=None)

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


# Choose the experiment (task)
list_cond = ['audio', 'visual', 'somato']
cond = list_cond[0]

# Downloading data
if cond in ['audio', 'visual']:

    subject = 'sample'
    data_path = sample.data_path()
    fwd_fname_suffix = 'MEG/sample/sample_audvis-meg-eeg-oct-6-fwd.fif'
    fwd_fname = os.path.join(data_path, fwd_fname_suffix)
    ave_fname = os.path.join(data_path, 'MEG/sample/sample_audvis-ave.fif')
    raw_fname = os.path.join(data_path, 'MEG/sample/sample_audvis_raw.fif')
    cov_fname_suffix = 'MEG/sample/sample_audvis-shrunk-cov.fif'
    cov_fname = os.path.join(data_path, cov_fname_suffix)
    cov_fname = data_path + '/' + cov_fname_suffix
    subjects_dir = os.path.join(data_path, 'subjects')

    if cond == 'audio':
        condition = 'Left Auditory'
    elif cond == 'visual':
        condition = 'Left visual'

    # Read noise covariance matrix
    noise_cov = mne.read_cov(cov_fname)

    # Handling average file
    evoked = mne.read_evokeds(ave_fname, condition=condition,
                              baseline=(None, 0))
    evoked = evoked.pick_types('grad')

    # Selecting relevant time window
    evoked.plot()
    t_min, t_max = 0.05, 0.1
    t_step = 0.01

    pca = False

elif cond == 'somato':

    data_path = somato.data_path()
    subject = '01'
    subjects_dir = data_path + '/derivatives/freesurfer/subjects'
    task = 'somato'
    raw_fname = os.path.join(data_path, f'sub-{subject}', 'meg',
                             f'sub-{subject}_task-{task}_meg.fif')
    fwd_fname = os.path.join(data_path, 'derivatives', f'sub-{subject}',
                             f'sub-{subject}_task-{task}-fwd.fif')
    condition = 'Unknown'

    # Read evoked
    raw = mne.io.read_raw_fif(raw_fname)
    events = mne.find_events(raw, stim_channel='STI 014')
    reject = dict(grad=4000e-13, eog=350e-6)
    picks = mne.pick_types(raw.info, meg=True, eeg=True, eog=True)

    event_id, tmin, tmax = 1, -.2, .25
    epochs = mne.Epochs(raw, events, event_id, tmin, tmax, picks=picks,
                        reject=reject, preload=True)
    evoked = epochs.average()
    evoked = evoked.pick_types('grad')

    # Compute noise covariance matrix
    noise_cov = mne.compute_covariance(epochs, rank='info', tmax=0.)

    # Selecting relevant time window: focusing on early signal
    evoked.plot()
    t_min, t_max = 0.03, 0.05
    t_step = 1.0 / 300

    # We must reduce the whitener since data were preprocessed for removal
    # of environmental noise leading to an effective number of 64 samples.
    pca = True

# Handling forward solution
forward = mne.read_forward_solution(fwd_fname)
# Collecting features' connectivity
connectivity = mne.source_estimate.spatial_src_adjacency(forward['src'])

# Croping evoked according to relevant time window
evoked.crop(tmin=t_min, tmax=t_max)
# Choosing frequency and number of clusters used for compression.
# Reducing the frequency to 100Hz to make inference faster
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
zscore_threshold = zscore_from_pval((fwer_target / 2) * correction_clust_inf)

# Preprocessing MEG data
X, Y, forward = preprocess_meg_eeg_data(evoked, forward, noise_cov, pca=pca)

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

# Plotting parameters
if cond == 'audio':
    hemi = 'lh'
    view = 'lateral'
elif cond == 'visual':
    hemi = 'rh'
    view = 'medial'
elif cond == 'somato':
    hemi = 'rh'
    view = 'lateral'

# Plotting
mne.viz.set_3d_backend("pyvista")

if active_set.sum() != 0:
    max_stc = np.max(np.abs(stc.data))
else:
    max_stc = 6

clim = dict(pos_lims=(3, zscore_threshold, max_stc), kind='value')
brain = stc.plot(subject=subject, hemi=hemi, clim=clim,
                 subjects_dir=subjects_dir)
brain.show_view(view)
brain.add_text(0.05, 0.9, f'{cond} - cd-MTLasso (AR1)', 'title', font_size=30)

save_fig = False
if save_fig:
    brain.save_image('figures/meg_audio_cd-MTLasso.png')

interactive_plot = False
if interactive_plot:
    brain = stc.plot(subject=subject, hemi='both',
                     subjects_dir=subjects_dir, clim=clim)

#  Compare with sLORETA
lambda2 = 1. / 9
inv = make_inverse_operator(evoked.info, forward, noise_cov, loose=0.,
                            depth=0., fixed=True)
stc_full = apply_inverse(evoked, inv, lambda2=lambda2, method='sLORETA')
stc_full = stc_full.mean()

# Computing estimated support by sLORETA
n_features = stc_full.data.size
correction = 1. / n_features
zscore_threshold_no_clust = zscore_from_pval((fwer_target / 2) * correction)
active_set = np.abs(stc_full.data) > zscore_threshold_no_clust
active_set = active_set.flatten()

sLORETA_solution = np.atleast_2d(stc_full.data[active_set]).flatten()

stc = _make_sparse_stc(sLORETA_solution, active_set, forward, stc_full.tmin,
                       tstep=stc_full.tstep)

# Plotting sLORETA solution
mne.viz.set_3d_backend("pyvista")

if active_set.sum() != 0:
    max_stc = np.max(np.abs(stc.data))
else:
    max_stc = 6

clim = dict(pos_lims=(3, zscore_threshold_no_clust, max_stc), kind='value')
brain = stc.plot(subject=subject, hemi=hemi, clim=clim,
                 subjects_dir=subjects_dir)
brain.show_view(view)
brain.add_text(0.05, 0.9, f'{cond} - sLORETA', 'title', font_size=30)

# Runing the ensemble clustered inference algorithm on temporal data
# might take several minutes on standard device with `n_jobs=1` (around 10 min)
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
    clim = dict(pos_lims=(3, zscore_threshold, max_stc), kind='value')
    brain = stc.plot(subject=subject, hemi=hemi, clim=clim,
                     subjects_dir=subjects_dir)
    brain.show_view(view)
    brain.add_text(0.05, 0.9, f'{cond} - ecd-MTLasso (AR1)',
                   'title', font_size=30)

    interactive_plot = False
    if interactive_plot:
        brain = stc.plot(subject=subject, hemi='both',
                         subjects_dir=subjects_dir, clim=clim)
