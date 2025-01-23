#!/usr/bin/env python3
"""Here's some documentation"""

import numpy as np

expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """Documentation"""

    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None, None

    n, d = X.shape

    if not isinstance(kmin, int) or kmin <= 0 or kmin > n:
        return None, None, None, None

    if kmax is None:
        kmax = n
    elif not isinstance(kmax, int) or kmax < kmin or kmax > n:
        return None, None, None, None

    if isinstance(iterations, int) or iterations <= 0:
        return None, None, None, None

    if not isinstance(tol, (float, int)) or tol < 0:
        return None, None, None, None

    if not isinstance(verbose, bool):
        return None, None, None, None

    return None
