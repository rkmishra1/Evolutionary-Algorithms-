"""Individual (solution) representation for multi-objective optimisation."""

from __future__ import annotations

import numpy as np


class Individual:
    """A single solution in the population.

    Attributes
    ----------
    dec : np.ndarray, shape (D,)
        Decision variables.
    obj : np.ndarray, shape (M,)
        Objective values.
    con : np.ndarray, shape (C,)
        Constraint violations (≤0 means feasible).
    """

    def __init__(
        self,
        dec: np.ndarray | None = None,
        obj: np.ndarray | None = None,
        con: np.ndarray | None = None,
    ) -> None:
        self.dec = dec
        self.obj = obj
        self.con = con

    def copy(self) -> Individual:
        """Return a shallow copy of this individual, preserving dynamic attributes."""
        new_ind = Individual(
            dec=self.dec.copy() if self.dec is not None else None,
            obj=self.obj.copy() if self.obj is not None else None,
            con=self.con.copy() if self.con is not None else None,
        )
        for k, v in self.__dict__.items():
            if k not in ("dec", "obj", "con"):
                if isinstance(v, np.ndarray):
                    setattr(new_ind, k, v.copy())
                elif isinstance(v, list):
                    setattr(new_ind, k, list(v))
                else:
                    setattr(new_ind, k, v)
        return new_ind


# ------------------------------------------------------------------
# Population-level helpers (operate on list[Individual])
# ------------------------------------------------------------------

def pop_decs(population: list[Individual]) -> np.ndarray:
    """Stack all decision vectors into a matrix (n × D)."""
    return np.vstack([ind.dec for ind in population])


def pop_objs(population: list[Individual]) -> np.ndarray:
    """Stack all objective vectors into a matrix (n × M)."""
    return np.vstack([ind.obj for ind in population])


def pop_cons(population: list[Individual]) -> np.ndarray:
    """Stack all constraint vectors into a matrix (n × C)."""
    return np.vstack([ind.con for ind in population])


def create_individuals(
    decs: np.ndarray,
    problem,
    global_config,
) -> list[Individual]:
    """Evaluate a batch of decision vectors and return :class:`Individual`
    objects.

    Parameters
    ----------
    decs : np.ndarray, shape (n, D)
        Decision variable matrix.
    problem : Problem
        The problem instance (provides ``cal_dec``, ``cal_obj``, ``cal_con``).
    global_config : GlobalConfig
        Global settings (provides bounds clamping and evaluation counter).

    Returns
    -------
    list[Individual]
        Evaluated individuals.
    """
    n = decs.shape[0]

    # Clamp to bounds
    if global_config.lower is not None and global_config.upper is not None:
        decs = np.clip(decs, global_config.lower, global_config.upper)

    # Evaluate
    decs = problem.cal_dec(decs)
    objs = problem.cal_obj(decs)
    cons = problem.cal_con(decs)

    individuals = []
    for i in range(n):
        individuals.append(Individual(
            dec=decs[i].copy(),
            obj=objs[i].copy(),
            con=cons[i].copy(),
        ))

    # Update evaluation counter
    global_config.evaluated += n
    return individuals
