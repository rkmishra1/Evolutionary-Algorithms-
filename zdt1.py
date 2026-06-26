"""ZDT1 benchmark multi-objective problem.

Reference:
    E. Zitzler, K. Deb, and L. Thiele, Comparison of multiobjective
    evolutionary algorithms: Empirical results, Evolutionary Computation,
    2000, 8(2): 173-195.
"""

from __future__ import annotations

import numpy as np
from problem import Problem


class ZDT1(Problem):
    """ZDT1 — a bi-objective benchmark with a convex Pareto front."""

    def __init__(self, global_config) -> None:
        super().__init__(global_config)
        self.global_config.M = 2
        if self.global_config.D is None or self.global_config.D == 0:
            self.global_config.D = 30
        d = self.global_config.D
        self.global_config.lower = np.zeros(d)
        self.global_config.upper = np.ones(d)
        self.global_config.encoding = "real"

    def cal_obj(self, pop_dec: np.ndarray) -> np.ndarray:
        """Evaluate ZDT1 objectives.

        f1 = x1
        g  = 1 + 9 * mean(x2..xD)
        h  = 1 - sqrt(f1 / g)
        f2 = g * h
        """
        f1 = pop_dec[:, 0]
        g = 1.0 + 9.0 * np.sum(pop_dec[:, 1:], axis=1) / (pop_dec.shape[1] - 1)
        # Note: original MATLAB uses sum without dividing by (D-1) for g,
        # but re-checking the MATLAB code: g = 1 + 9*sum(x2..xD).
        # Let's match exactly.
        g = 1.0 + 9.0 * np.sum(pop_dec[:, 1:], axis=1)
        h = 1.0 - np.sqrt(f1 / g)
        f2 = g * h
        return np.column_stack([f1, f2])

    def pf(self, n: int = 10000) -> np.ndarray:
        """Sample *n* reference points on the true Pareto front."""
        f1 = np.linspace(0, 1, n)
        f2 = 1.0 - np.sqrt(f1)
        return np.column_stack([f1, f2])
