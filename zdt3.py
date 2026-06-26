"""ZDT3 benchmark multi-objective problem.

Reference:
    E. Zitzler, K. Deb, and L. Thiele, Comparison of multiobjective
    evolutionary algorithms: Empirical results, Evolutionary Computation,
    2000, 8(2): 173-195.
"""

from __future__ import annotations

import numpy as np
from problem import Problem

class ZDT3(Problem):
    """ZDT3 — a bi-objective benchmark with a discontinuous Pareto front."""

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
        """Evaluate ZDT3 objectives.

        f1 = x1
        g  = 1 + 9 * mean(x2..xD)
        h  = 1 - sqrt(f1 / g) - (f1 / g) * sin(10 * pi * f1)
        f2 = g * h
        """
        f1 = pop_dec[:, 0]
        g = 1.0 + 9.0 * np.sum(pop_dec[:, 1:], axis=1)
        h = 1.0 - np.sqrt(f1 / g) - (f1 / g) * np.sin(10.0 * np.pi * f1)
        f2 = g * h
        return np.column_stack([f1, f2])

    def pf(self, n: int = 1000) -> np.ndarray:
        """Sample *n* reference points on the true Pareto front."""
        f1 = np.linspace(0, 1, n * 5)  # denser sample for filtering
        f2 = 1.0 - np.sqrt(f1) - f1 * np.sin(10.0 * np.pi * f1)
        
        points = np.column_stack([f1, f2])
        # Simple non-dominated filtering to extract only the true Pareto optimal segments
        is_efficient = np.ones(points.shape[0], dtype=bool)
        for i, c in enumerate(points):
            if is_efficient[i]:
                # Keep points that are not dominated by any other point
                is_efficient[is_efficient] = ~((np.all(points[is_efficient] >= c, axis=1)) & (np.any(points[is_efficient] > c, axis=1)))
                is_efficient[i] = True
                
        # Downsample to approximately n points
        filtered_points = points[is_efficient]
        indices = np.linspace(0, len(filtered_points) - 1, min(n, len(filtered_points)), dtype=int)
        return filtered_points[indices]
