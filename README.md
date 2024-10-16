# HiDimStat: High-dimensional statistical inference tool for Python
[![build][TravisCI]][travis]  [![coverage][CodeCov]][cov]

The HiDimStat package provides statistical inference methods to solve the
problem of support recovery in the context of high-dimensional and
spatially structured data.

**Update on Oct 2024: this repository is no longer maintained.** Please refer to the new up-to-date github repo at https://github.com/mind-INRIA/hidimstat.


## Installation

HiDimStat working only with Python 3, ideally Python 3.6+. For installation,
run the following from terminal

```bash
pip install hidimstat
```

Or if you want the latest version available (for example to contribute to
the development of this project:

```
pip install -U git+https://github.com/ja-che/hidimstat.git
```

or

```bash
git clone https://github.com/ja-che/hidimstat.git
cd hidimstat
pip install -e .
```

## Dependencies

```
joblib
numpy
scipy
scikit-learn
```

To run examples it is neccessary to install `matplotlib`, and to run tests it
is also needed to install `pytest`.

## Documentation & Examples

All the documentation of HiDimStat is available at https://ja-che.github.io/hidimstat/.

As of now in the `examples` folder there are three Python scripts that
illustrate how to use the main HiDimStat functions.
In each script we handle a different kind of dataset:
``plot_2D_simulation_example.py`` handles a simulated dataset with a 2D
spatial structure,
``plot_fmri_data_example.py`` solves the decoding problem on Haxby fMRI dataset,
``plot_meg_data_example.py`` tackles the source localization problem on several
MEG/EEG datasets.


```bash
# For example run the following command in terminal
python plot_2D_simulation_example.py
```

## References

The algorithms developed in this package have been detailed in several
conference/journal articles that can be downloaded at
https://ja-che.github.io/research.html.

#### Main references:

Ensemble of Clustered desparsified Lasso (ECDL):

* Chevalier, J. A., Salmon, J., & Thirion, B. (2018). __Statistical inference
  with ensemble of clustered desparsified lasso__. In _International Conference
  on Medical Image Computing and Computer-Assisted Intervention_
  (pp. 638-646). Springer, Cham.

* Chevalier, J. A., Nguyen, T. B., Thirion, B., & Salmon, J. (2021). __Spatially relaxed inference on high-dimensional linear models__. arXiv preprint arXiv:2106.02590.

Aggregation of multiple Knockoffs (AKO):

* Nguyen T.-B., Chevalier J.-A., Thirion B., & Arlot S. (2020). __Aggregation
  of Multiple Knockoffs__. In _Proceedings of the 37th International Conference on
  Machine Learning_, Vienna, Austria, PMLR 119.

Application to decoding (fMRI data):

* Chevalier, J. A., Nguyen T.-B., Salmon, J., Varoquaux, G. & Thirion, B. (2021). __Decoding with confidence: Statistical control on decoder maps__. In _NeuroImage_, 234, 117921.

Application to source localization (MEG/EEG data):

* Chevalier, J. A., Gramfort, A., Salmon, J., & Thirion, B. (2020). __Statistical control for spatio-temporal MEG/EEG source imaging with desparsified multi-task Lasso__. In _Proceedings of the 34th Conference on Neural Information Processing Systems (NeurIPS 2020)_, Vancouver, Canada.

If you use our packages, we would appreciate citations to the relevant aforementioned papers.

#### Other useful references:

For de-sparsified(or de-biased) Lasso:

* Javanmard, A., & Montanari, A. (2014). __Confidence intervals and hypothesis
  testing for high-dimensional regression__. _The Journal of Machine Learning
  Research_, 15(1), 2869-2909.

* Zhang, C. H., & Zhang, S. S. (2014). __Confidence intervals for low dimensional
  parameters in high dimensional linear models__. _Journal of the Royal
  Statistical Society: Series B: Statistical Methodology_, 217-242.

* Van de Geer, S., Bühlmann, P., Ritov, Y. A., & Dezeure, R. (2014). __On
  asymptotically optimal confidence regions and tests for high-dimensional
  models__. _The Annals of Statistics_, 42(3), 1166-1202.

For Knockoffs Inference:

* Barber, R. F; Candès, E. J. (2015). __Controlling the false discovery rate
  via knockoffs__. _Annals of Statistics_. 43 , no. 5,
  2055--2085. doi:10.1214/15-AOS1337. https://projecteuclid.org/euclid.aos/1438606853

* Candès, E., Fan, Y., Janson, L., & Lv, J. (2018). __Panning for gold: Model-X
  knockoffs for high dimensional controlled variable selection__. _Journal of the
  Royal Statistical Society Series B_, 80(3), 551-577.

## License

This project is licensed under the BSD 2-Clause License.

## Acknowledgments

This project has been funded by Labex DigiCosme (ANR-11-LABEX-0045-DIGICOSME)
as part of the program "Investissement d’Avenir" (ANR-11-IDEX-0003-02), by the
Fast Big project (ANR-17-CE23-0011) and the KARAIB AI Chair
(ANR-20-CHIA-0025-01). This study has also been supported by the European
Union’s Horizon 2020 research and innovation program
(Grant Agreement No. 945539, Human Brain Project SGA3).


[TravisCI]: https://travis-ci.com/ja-che/hidimstat.svg?branch=main "travisCI status"
[travis]: https://travis-ci.com/ja-che/hidimstat

[CodeCov]: https://codecov.io/gh/ja-che/hidimstat/branch/main/graph/badge.svg "CodeCov status"
[cov]: https://codecov.io/gh/ja-che/hidimstat
