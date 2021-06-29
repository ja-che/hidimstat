"""Parallel util function."""

# Author: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#
# License: Simplified BSD

import os
from warnings import warn


if 'MNE_FORCE_SERIAL' in os.environ:
    _force_serial = True
else:
    _force_serial = None


def parallel_func(func, n_jobs, **kwargs):
    """Return parallel instance with delayed function.

    Util function to use joblib only if available

    Parameters
    ----------
    func : callable
        A function.
    n_jobs : int
        Number of jobs to run in parallel.
    **kwargs
        To pass to joblib.Parallel instance.

    Returns
    -------
    parallel: instance of joblib.Parallel or list
        The parallel object.
    my_func: callable
        ``func`` if not parallel or delayed(func).
    n_jobs: int
        Number of jobs >= 0.
    """
    # for a single job, we don't need joblib
    if n_jobs != 1:
        try:
            from joblib import Parallel, delayed
        except ImportError:
            warn('joblib not installed. Cannot run in parallel.')
            n_jobs = 1

    if n_jobs == 1:
        n_jobs = 1
        my_func = func
        parallel = list
    else:
        n_jobs = check_n_jobs(n_jobs)
        parallel = Parallel(n_jobs, **kwargs)
        my_func = delayed(func)

    return parallel, my_func, n_jobs


def _check_wrapper(fun):
    def run(*args, **kwargs):
        try:
            return fun(*args, **kwargs)
        except RuntimeError as err:
            msg = str(err.args[0]) if err.args else ''
            if msg.startswith('The task could not be sent to the workers'):
                raise RuntimeError(
                    msg + ' Consider using joblib memmap caching to get '
                    'around this problem. See mne.set_mmap_min_size, '
                    'mne.set_cache_dir, and buffer_size parallel function '
                    'arguments (if applicable).')
            raise
    return run


def check_n_jobs(n_jobs):
    """Check n_jobs in particular for negative values.

    Parameters
    ----------
    n_jobs : int
        The number of jobs.

    Returns
    -------
    n_jobs : int
        The checked number of jobs. Always positive (or 'cuda' if
        applicable).
    """
    if _force_serial:
        n_jobs = 1
        print('... MNE_FORCE_SERIAL set. Processing in forced serial mode.')
    elif n_jobs <= 0:
        try:
            import multiprocessing
            n_cores = multiprocessing.cpu_count()
            n_jobs = min(n_cores + n_jobs + 1, n_cores)
            if n_jobs <= 0:
                raise ValueError('If n_jobs has a negative value it must not '
                                 'be less than the number of CPUs present. '
                                 'You\'ve got %s CPUs' % n_cores)
        except ImportError:
            # only warn if they tried to use something other than 1 job
            if n_jobs != 1:
                warn('multiprocessing not installed. Cannot run in parallel.')
                n_jobs = 1

    return n_jobs
