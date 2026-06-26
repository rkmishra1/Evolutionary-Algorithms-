"""
Uniform weight vector generation on the unit hyperplane.

References:
    [1] I. Das and J. E. Dennis, Normal-boundary intersection: A new method
        for generating the Pareto surface in nonlinear multicriteria optimization
        problems, SIAM Journal on Optimization, 1998, 8(3): 631-657.
    [2] K. Deb and H. Jain, An evolutionary many-objective optimization
        algorithm using reference-point based non-dominated sorting approach,
        part I: Solving problems with box constraints, IEEE Transactions on
        Evolutionary Computation, 2014, 18(4): 577-601.
"""

import numpy as np
from itertools import combinations
from math import comb


def uniform_point(n: int, m: int) -> tuple[np.ndarray, int]:
    """Generate approximately *n* uniformly distributed points with *m*
    objectives on the unit hyperplane.

    Due to the requirement of uniform distribution, the actual number of
    points may be slightly smaller than *n*.

    Parameters
    ----------
    n : int
        Desired (approximate) number of weight vectors.
    m : int
        Number of objectives.

    Returns
    -------
    W : np.ndarray, shape (N, m)
        Weight vectors.
    N : int
        Actual number of weight vectors generated.
    """
    # Determine H1
    h1 = 1
    while comb(h1 + m, m - 1) <= n:
        h1 += 1

    # Generate the primary layer of weight vectors
    # Equivalent to: nchoosek(1:H1+M-1, M-1)
    indices = np.array(list(combinations(range(1, h1 + m), m - 1)))
    w = indices - np.arange(m - 1)  # subtract 0:M-2
    w = w - 1
    # W = ([W, H1] - [0, W]) / H1
    w = np.column_stack([w, np.full(len(w), h1)]) - \
        np.column_stack([np.zeros(len(w)), w])
    w = w / h1

    # Two-layer approach for many-objective problems
    if h1 < m:
        h2 = 0
        while comb(h1 + m - 1, m - 1) + comb(h2 + m, m - 1) <= n:
            h2 += 1
        if h2 > 0:
            indices2 = np.array(list(combinations(range(1, h2 + m), m - 1)))
            w2 = indices2 - np.arange(m - 1) - 1
            w2 = np.column_stack([w2, np.full(len(w2), h2)]) - \
                 np.column_stack([np.zeros(len(w2)), w2])
            w2 = w2 / h2
            w = np.vstack([w, w2 / 2 + 1 / (2 * m)])

    w = np.maximum(w, 1e-6)
    return w, w.shape[0]
