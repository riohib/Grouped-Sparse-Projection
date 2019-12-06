from __future__ import division
import numpy as np
import scipy.io
import pylab as plt

def sparse_opt(b, k):
    """ Project a vector onto a sparsity constraint
    Solves the projection problem by taking into account the
    symmetry of l1 and l2 constraints.
    Parameters
    ----------
    b : sorted vector in decreasing value
    k : Ratio of l1/l2 norms of a vector
    Returns
    -------
    z : closest vector satisfying the required sparsity constraint.
    """
    n = len(b)
    sumb = np.cumsum(b)
    normb = np.cumsum(b * b)
    pnormb = np.arange(1, n + 1) * normb
    y = (pnormb - sumb * sumb) / (np.arange(1, n + 1) - k * k)
    bot = np.int(np.ceil(k * k))
    z = np.zeros(n)
    if bot > n:
        print('Looks like the sparsity measure is not between 0 and 1\n')
        return
    obj = (-np.sqrt(y) * (np.arange(1, n + 1) + k) + sumb) / np.arange(1, n + 1)
    indx = np.argmax(obj[bot:n])
    p = indx + bot - 1
    p = min(p, n - 1)
    p = max(p, bot)
    lam = np.sqrt(y[p])
    mue = -sumb[p] / (p + 1) + k / (p + 1) * (lam)
    # z[:p + 1] = (b[:p + 1] + mue) / (lam)
    # # mue = -sumb[p] / (p + 1) + k / (p + 1) * (lam)
    z[:p + 1] = (b[:p + 1] + mue) / (lam + 1e-15)
    return z
