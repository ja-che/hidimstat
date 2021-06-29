"""
Support recovery on fMRI data
=============================

This example show how to recover the support of a decoder map with
statistical guarantees working with the Haxby dataset.

"""

#############################################################################
# Imports needed for this script
# ------------------------------
import numpy as np
import pandas as pd
from sklearn.utils import Bunch
from sklearn.cluster import FeatureAgglomeration
from sklearn.feature_extraction import image
from nilearn import datasets
from nilearn.input_data import NiftiMasker
from nilearn.image import mean_img
from nilearn.plotting import plot_stat_map, show

from hidimstat.stat_tools import zscore_from_pval, pval_from_scale
from hidimstat.standardized_svr import standardized_svr
from hidimstat.permutation_test import permutation_test_cv
from hidimstat.gaonkar import gaonkar
from hidimstat.clustered_inference import clustered_inference
from hidimstat.ensemble_clustered_inference import ensemble_clustered_inference


#############################################################################
# Function to fetch and preprocess Haxby dataset
# ----------------------------------------------
def preprocess_haxby(subject=2, memory=None, comparison='face-house'):
    '''Gathering and preprocessing Haxby dataset for a given subject'''

    # Gathering Data
    haxby_dataset = datasets.fetch_haxby(subjects=[subject])
    fmri_filename = haxby_dataset.func[0]

    behavioral = pd.read_csv(haxby_dataset.session_target[0], sep=" ")

    conditions = pd.DataFrame.to_numpy(behavioral['labels'])
    session_label = pd.DataFrame.to_numpy(behavioral['chunks'])

    if comparison == 'face-house':
        condition_mask = np.logical_or(conditions == 'face',
                                       conditions == 'house')
        y = np.asarray((conditions[condition_mask] == 'face') * 2 - 1)
    elif comparison == 'cat-chair':
        condition_mask = np.logical_or(conditions == 'cat',
                                       conditions == 'chair')
        y = np.asarray((conditions[condition_mask] == 'cat') * 2 - 1)
    else:
        raise ValueError(f'Not a valid comparison: {comparison}')

    groups = session_label[condition_mask]

    # Loading anatomical image (back-ground image)
    if haxby_dataset.anat[0] is None:
        bg_img = None
    else:
        bg_img = mean_img(haxby_dataset.anat)

    # Loading mask
    mask_img = haxby_dataset.mask
    masker = NiftiMasker(mask_img=mask_img, standardize=True,
                         smoothing_fwhm=None, memory=memory)

    # Computing masked data
    fmri_masked = masker.fit_transform(fmri_filename)

    X = np.asarray(fmri_masked)[condition_mask, :]

    return Bunch(X=X, y=y, groups=groups, bg_img=bg_img, masker=masker)


#############################################################################
# Gathering and preprocessing Haxby dataset for a given subject
# -------------------------------------------------------------
list_subject = [1, 2, 3, 4, 5, 6]
list_comparison = ['face-house', 'cat-chair']
# Choose subject and contrast. By default subject=2 and comparison='face-house'
data = preprocess_haxby(subject=list_subject[1], comparison=list_comparison[0])
X, y, groups, masker = data.X, data.y, data.groups, data.masker
mask = masker.mask_img_.get_fdata().astype(bool)

#############################################################################
# Initializing FeatureAgglomeration object needed to perform the clustering
# -------------------------------------------------------------------------
# For fMRI data n_clusters = 500 is generally a good default choice.
n_clusters = 500
# Deriving connectivity.
shape = mask.shape
n_x, n_y, n_z = shape[0], shape[1], shape[2]
connectivity = image.grid_to_graph(n_x=n_x, n_y=n_y, n_z=n_z, mask=mask)
# Initializing FeatureAgglomeration object
ward = FeatureAgglomeration(n_clusters=n_clusters, connectivity=connectivity)

#############################################################################
# Making the inference with several algorithms
# --------------------------------------------
# Setting theoretical FWER target
n_samples, n_features = X.shape
fwer_target = 0.1
# No need of correction for permutation tests
zscore_threshold_corr = zscore_from_pval((fwer_target / 2))
# Other methods need to be corrected
correction = 1. / n_features
zscore_threshold_no_clust = zscore_from_pval((fwer_target / 2) * correction)
correction_clust = 1. / n_clusters
zscore_threshold_clust = zscore_from_pval((fwer_target / 2) * correction_clust)

# Recovering the support with SVR decoder thresholded parametrically
# We computed the regularization parameter by CV (C = 0.1)
beta_hat, scale = standardized_svr(X, y, Cs=[0.1])
pval, _, one_minus_pval, _ = pval_from_scale(beta_hat, scale)
zscore_std_svr = zscore_from_pval(pval, one_minus_pval)

# Recovering the support with SVR decoder thresholded by permutation test
# This inference takes around 15 minutes.
permutation_test_inference = False
if permutation_test_inference:
    # We computed the regularization parameter by CV (C = 0.1)
    pval_corr, one_minus_pval_corr = \
        permutation_test_cv(X, y, n_permutations=50, C=0.1)
    zscore_permutation_test = zscore_from_pval(pval_corr, one_minus_pval_corr)

# Recovering the support with Gaonkar algorithm
beta_hat, scale = gaonkar(X, y)
pval, _, one_minus_pval, _ = pval_from_scale(beta_hat, scale)
zscore_gaonkar = zscore_from_pval(pval, one_minus_pval)

# Recovering the support with clustered inference
beta_hat, pval, pval_corr, one_minus_pval, one_minus_pval_corr = \
    clustered_inference(X, y, ward, n_clusters)
zscore_cdl = zscore_from_pval(pval, one_minus_pval)

# Recovering the support with ensemble clustered inference
beta_hat, pval, pval_corr, one_minus_pval, one_minus_pval_corr = \
    ensemble_clustered_inference(X, y, ward, n_clusters, groups=groups)
zscore_ecdl = zscore_from_pval(pval, one_minus_pval)

#############################################################################
# Plotting the results
# --------------------
bg_img = data.bg_img
cut_coords = [-25, -40, -5]
# cut_coords = None

zscore_img = masker.inverse_transform(zscore_std_svr)
plot_stat_map(zscore_img, threshold=zscore_threshold_no_clust, bg_img=bg_img,
              cut_coords=cut_coords, title='SVR parametric threshold')

if permutation_test_inference:
    zscore_img = masker.inverse_transform(zscore_permutation_test)
    plot_stat_map(zscore_img, threshold=zscore_threshold_corr, bg_img=bg_img,
                  cut_coords=cut_coords, title='SVR permutation-test thresh.')

zscore_img = masker.inverse_transform(zscore_gaonkar)
plot_stat_map(zscore_img, threshold=zscore_threshold_no_clust, bg_img=bg_img,
              cut_coords=cut_coords, title='Gaonkar algorithm')

zscore_img = masker.inverse_transform(zscore_cdl)
plot_stat_map(zscore_img, threshold=zscore_threshold_clust, bg_img=bg_img,
              cut_coords=cut_coords, title='CluDL')

zscore_img = masker.inverse_transform(zscore_ecdl)
plot_stat_map(zscore_img, threshold=zscore_threshold_clust, bg_img=bg_img,
              cut_coords=cut_coords, title='EnCluDL')

show()
