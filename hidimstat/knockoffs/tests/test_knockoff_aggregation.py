import numpy as np
from hidimstat.knockoffs import knockoff_aggregation, model_x_knockoff
from hidimstat.knockoffs.data_simulation import simu_data
from hidimstat.knockoffs.utils import cal_fdp_power

n = 300
p = 100
n_bootstraps = 25
fdr = 0.5
SEED = 0

X, y, _, non_zero_index = simu_data(n, p, seed=SEED)


def test_knockoff_aggregation():
    selected, aggregated_pval, pvals = knockoff_aggregation(
        X, y, fdr=fdr, n_bootstraps=n_bootstraps, verbose=True, random_state=0)

    fdp, power = cal_fdp_power(selected, non_zero_index)
    assert pvals.shape == (n_bootstraps, p)
    assert fdp < 0.2
    assert power > 0.1

    # Single AKO (or vanilla KO)
    selected = knockoff_aggregation(
        X, y, fdr=fdr, verbose=False, n_bootstraps=1, random_state=1)

    selected_ko = model_x_knockoff(X, y, fdr=fdr, seed=SEED+1)

    np.testing.assert_array_equal(selected, selected_ko)

    fdp, power = cal_fdp_power(selected, non_zero_index)
    assert fdp < 0.2
    assert power > 0.1
