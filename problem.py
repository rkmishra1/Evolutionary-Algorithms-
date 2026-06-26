"""Base class for multi-objective test problems."""

from __future__ import annotations

import numpy as np


class Problem:
    """Base class for multi-objective optimisation problems.

    Sub-classes must override :meth:`__init__` (to set problem parameters on
    the :class:`GlobalConfig`) and typically :meth:`cal_obj` and :meth:`pf`.
    """

    def __init__(self, global_config) -> None:
        self.global_config = global_config

    # ------------------------------------------------------------------
    # Population initialisation
    # ------------------------------------------------------------------
    def init(self, n: int) -> np.ndarray:
        """Generate *n* random decision vectors within bounds.

        Parameters
        ----------
        n : int
            Number of individuals.

        Returns
        -------
        np.ndarray, shape (n, D)
        """
        encoding = self.global_config.encoding
        d = self.global_config.D

        if encoding == "binary":
            return np.random.randint(0, 2, size=(n, d))
        elif encoding == "permutation":
            return np.argsort(np.random.rand(n, d), axis=1)
        else:  # 'real'
            lower = self.global_config.lower
            upper = self.global_config.upper
            return np.random.uniform(lower, upper, size=(n, d))

    # ------------------------------------------------------------------
    # Evaluation helpers
    # ------------------------------------------------------------------
    def cal_dec(self, pop_dec: np.ndarray) -> np.ndarray:
        """Repair infeasible decision variables (identity by default)."""
        return pop_dec

    def cal_obj(self, pop_dec: np.ndarray) -> np.ndarray:
        """Calculate objective values.  Override in sub-classes."""
        obj1 = pop_dec[:, 0] + np.sum(pop_dec[:, 1:], axis=1)
        obj2 = 1 - pop_dec[:, 0] + np.sum(pop_dec[:, 1:], axis=1)
        return np.column_stack([obj1, obj2])

    def cal_con(self, pop_dec: np.ndarray) -> np.ndarray:
        """Calculate constraint violations (none by default)."""
        return np.zeros((pop_dec.shape[0], 1))

    def pf(self, n: int = 10000) -> np.ndarray:
        """Sample reference points on the true Pareto front."""
        return np.ones((1, self.global_config.M))
