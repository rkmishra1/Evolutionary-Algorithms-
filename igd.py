"""Inverted Generational Distance (IGD) performance metric."""

import numpy as np
from scipy.spatial.distance import cdist


def igd(pop_obj: np.ndarray, pf: np.ndarray) -> float:
    """Compute the Inverted Generational Distance between a population's
    objective values and the true Pareto front.

    Parameters
    ----------
    pop_obj : np.ndarray, shape (n, m)
        Objective values of the obtained population.
    pf : np.ndarray, shape (p, m)
        Reference points on the true Pareto front.

    Returns
    -------
    float
        IGD value (lower is better).
    """
    distances = cdist(pf, pop_obj)
    min_distances = np.min(distances, axis=1)
    return float(np.mean(min_distances))
