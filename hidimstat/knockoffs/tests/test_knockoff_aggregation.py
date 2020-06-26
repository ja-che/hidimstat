import numpy as np
from hidimstat.knockoffs import knockoff_aggregation, model_x_knockoff
from hidimstat.knockoffs.data_simulation import simu_data
from hidimstat.knockoffs.utils import cal_fdp_power

n = 600
p = 1000
n_bootstraps = 25
fdr = 0.1
SEED = 0

X, y, _, non_zero_index = simu_data(n, p, seed=SEED)


def test_knockoff_aggregation():
    selected, aggregated_pval, pvals = knockoff_aggregation(
        X, y, verbose=True, n_bootstraps=n_bootstraps, random_state=None)

    fdp, power = cal_fdp_power(selected, non_zero_index)
    assert pvals.shape == (n_bootstraps, p)
    assert fdp < fdr * 2
    assert power > 0.1

    # Single AKO (or vanilla KO)
    selected = knockoff_aggregation(
        X, y, verbose=False, n_bootstraps=1, random_state=SEED)

    selected_ko = model_x_knockoff(X, y, seed=SEED)

    np.testing.assert_array_equal(selected, selected_ko)

    fdp, power = cal_fdp_power(selected, non_zero_index)
    assert fdp < fdr * 2
    assert power > 0.1
