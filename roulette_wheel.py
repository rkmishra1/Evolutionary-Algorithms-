"""Roulette wheel selection for the MOEAD-ABC algorithm."""

import numpy as np


def roulette_wheel_selection(p: np.ndarray) -> int:
    """Select an index using roulette wheel (fitness-proportionate) selection.

    Parameters
    ----------
    p : np.ndarray
        Probability vector (must sum to 1).

    Returns
    -------
    int
        Selected index (0-based).
    """
    r = np.random.rand()
    c = np.cumsum(p)
    return int(np.searchsorted(c, r))
