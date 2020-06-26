# HiDimStat: High-dimensional statistical inference tool for Python
[![build][TravisCI]][travis]  [![coverage][CodeCov]][cov]

## Installation 

HiDimStat working only with Python 3, ideally Python 3.6+. For installation,
run the following from terminal

```
git clone https://github.com/ja-che/hidimstat.git
pip install -e ./
# or python setup.py install
```

## Dependencies

```
joblib
numpy
scipy
scikit-learn
```

For running tests it is also needed to installed `pytest`.


## Documentation & Examples [Work In Progress]

As of now in the `examples` folder there is a Python script to reproduce Figure
1 in Nguyen et al. (2020) (see References below). __Warning__: this script
should take quite a long time to run.

`python plot_fig_1_nguyen_et_al.py`

<p align="center">
  <img src="./examples/figures/fig1_nguyen_et_al.png"  alt="Histogram of FDP & Power for KO vs. AKO" width="500">
</p>


## References

* Chevalier, J. A., Salmon, J., & Thirion, B. (2018). __Statistical inference
  with ensemble of clustered desparsified lasso__. In _International Conference
  on Medical Image Computing and Computer-Assisted Intervention_
  (pp. 638-646). Springer, Cham.

* T.-B. Nguyen, J.-A. Chevalier, B.Thirion, & S. Arlot. (2020). __Aggregation
  of Multiple Knockoffs__. In Proceedings of the 37th International Conference on
  Machine Learning, Vienna, Austria, PMLR 119.

* Candes, E., Fan, Y., Janson, L., & Lv, J. (2018). __Panning for gold: Model-X
  knockoffs for high dimensional controlled variable selection__. _Journal of the
  Royal Statistical Society Series B_, 80(3), 551-577.

* Zhang, C. H., & Zhang, S. S. (2014). __Confidence intervals for low dimensional
  parameters in high dimensional linear models__. _Journal of the Royal
  Statistical Society: Series B: Statistical Methodology_, 217-242.


[TravisCI]: https://travis-ci.com/ja-che/hidimstat.svg?branch=master "travisCI status"
[travis]: https://travis-ci.com/ja-che/hidimstat

[CodeCov]: https://codecov.io/gh/ja-che/hidimstat/branch/master/graph/badge.svg "CodeCov status"
[cov]: https://codecov.io/gh/ja-che/hidimstat
