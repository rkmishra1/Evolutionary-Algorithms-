"""ZDT4 benchmark multi-objective problem.

Reference:
    E. Zitzler, K. Deb, and L. Thiele, Comparison of multiobjective
    evolutionary algorithms: Empirical results, Evolutionary Computation,
    2000, 8(2): 173-195.
"""

from __future__ import annotations

import numpy as np
from problem import Problem

class ZDT4(Problem):
    """ZDT4 — a bi-objective benchmark with 21^9 local Pareto fronts, testing global convergence."""

    def __init__(self, global_config) -> None:
        super().__init__(global_config)
        self.global_config.M = 2
        if self.global_config.D is None or self.global_config.D == 0:
            self.global_config.D = 10
        d = self.global_config.D
        self.global_config.lower = np.zeros(d)
        self.global_config.lower[1:] = -5.0
        self.global_config.upper = np.ones(d)
        self.global_config.upper[1:] = 5.0
        self.global_config.encoding = "real"

    def cal_obj(self, pop_dec: np.ndarray) -> np.ndarray:
        """Evaluate ZDT4 objectives.

        f1 = x1
        g  = 1 + 10*(D-1) + sum(x_i^2 - 10*cos(4*pi*x_i))
        h  = 1 - sqrt(f1 / g)
        f2 = g * h
        """
        f1 = pop_dec[:, 0]
        x2_end = pop_dec[:, 1:]
        D_minus_1 = pop_dec.shape[1] - 1
        g = 1.0 + 10.0 * D_minus_1 + np.sum(x2_end ** 2 - 10.0 * np.cos(4.0 * np.pi * x2_end), axis=1)
        h = 1.0 - np.sqrt(f1 / g)
        f2 = g * h
        return np.column_stack([f1, f2])

    def pf(self, n: int = 1000) -> np.ndarray:
        """Sample *n* reference points on the true Pareto front."""
        f1 = np.linspace(0, 1, n)
        f2 = 1.0 - np.sqrt(f1)
        return np.column_stack([f1, f2])
