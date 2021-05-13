.. hidimstat documentation master file, created by
   sphinx-quickstart on Fri April 23 12:22:52 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

HiDimStat: High-dimensional statistical inference tool for Python
=================================================================
|Build Status| |codecov|

Installation
------------

HiDimStat working only with Python 3, ideally Python 3.6+. For installation,
run the following from terminal::

  git clone https://github.com/ja-che/hidimstat.git
  cd hidimstat
  pip install -e .
  # or python setup.py install


Dependencies
------------

hidimstat depends on the following packages::

  joblib
  numpy
  scipy
  scikit-learn


To run examples it is neccessary to install ```matplotlib``, and to run tests it
is also needed to install ``pytest``.


Documentation & Examples
------------------------

As of now in the `examples` folder there is a Python script to reproduce Figure
1 in Nguyen et al. 2020 (see References below).

.. warning::

  this scrip should take quite a long time to run.

.. code-block::

  # Run this command in terminal
  python plot_fig_1_nguyen_et_al.py


References
----------

Main references
~~~~~~~~~~~~~~~

Ensemble of Clustered desparsified Lasso (ECDL):

* Chevalier, J. A., Salmon, J., & Thirion, B. (2018). Statistical inference
  with ensemble of clustered desparsified lasso. In International Conference
  on Medical Image Computing and Computer-Assisted Intervention
  (pp. 638-646). Springer, Cham.

Aggregation of multiple Knockoffs (AKO):

* Nguyen T.-B., Chevalier J.-A., Thirion B., & Arlot S. (2020). Aggregation
  of Multiple Knockoffs. In Proceedings of the 37th International Conference on
  Machine Learning, Vienna, Austria, PMLR 119.

If you use our packages, we would appreciate citations to the aforementioned papers.

Other useful references
~~~~~~~~~~~~~~~~~~~~~~~

For de-sparsified(or de-biased) Lasso:

* Javanmard, A., & Montanari, A. (2014). Confidence intervals and hypothesis
  testing for high-dimensional regression. The Journal of Machine Learning
  Research, 15(1), 2869-2909.

* Zhang, C. H., & Zhang, S. S. (2014). Confidence intervals for low dimensional
  parameters in high dimensional linear models. Journal of the Royal
  Statistical Society: Series B: Statistical Methodology, 217-242.

For Knockoffs Inference:

* Barber, R. F; Candès, E. J. (2015). Controlling the false discovery rate
  via knockoffs. Annals of Statistics. 43 , no. 5,
  2055--2085. doi:10.1214/15-AOS1337. https://projecteuclid.org/euclid.aos/1438606853

* Candès, E., Fan, Y., Janson, L., & Lv, J. (2018). Panning for gold: Model-X
  knockoffs for high dimensional controlled variable selection. Journal of the
  Royal Statistical Society Series B, 80(3), 551-577.

.. |Build Status| image:: https://travis-ci.com/ja-che/hidimstat.svg?branch=main
   :target: https://codecov.io/gh/ja-che/hidimstat

.. |codecov| image:: https://codecov.io/gh/ja-che/hidimstat/branch/main/graph/badge.svg
   :target: https://codecov.io/gh/ja-che/hidimstat


Build the documentation
-----------------------

To build the documentation you will need to run:

.. code-block::

    pip install -U sphinx_gallery sphinx_bootstrap_theme
    cd doc
    make html

API
---

.. toctree::
    :maxdepth: 1

    api.rst
