"""Global experiment configuration for the MOEAD-ABC framework.

Equivalent to GLOBAL.m in the MATLAB implementation.
"""

from __future__ import annotations

import time
import numpy as np
from individual import Individual, create_individuals, pop_objs, pop_cons


class GlobalConfig:
    """Stores all experiment-level parameters and controls the run loop.

    Parameters
    ----------
    N : int
        Population size (default 100).
    M : int or None
        Number of objectives (set by the problem).
    D : int or None
        Number of decision variables (set by the problem).
    algorithm : callable or None
        Algorithm function ``algorithm(global_config)``.
    problem_class : type or None
        Problem class to instantiate.
    evaluation : int
        Maximum number of objective evaluations (default 10 000).
    run : int
        Run number (default 1).
    save : int
        Number of intermediate populations to save (default 0 → display only).
    agg_type : int
        Aggregation function type for MOEAD-ABC (default 1).
    """

    def __init__(
        self,
        N: int = 100,
        M: int | None = None,
        D: int | None = None,
        algorithm=None,
        problem_class=None,
        evaluation: int = 10000,
        run: int = 1,
        save: int = 0,
        agg_type: int = 1,
    ) -> None:
        # Public parameters
        self.N = N
        self.M = M
        self.D = D
        self.evaluation = evaluation
        self.run = run
        self.save = save
        self.agg_type = agg_type

        # Problem & algorithm
        self.algorithm = algorithm
        self.problem_class = problem_class
        self.problem = None  # Instantiated in start()

        # Bounds (set by problem.__init__)
        self.lower: np.ndarray | None = None
        self.upper: np.ndarray | None = None
        self.encoding: str = "real"

        # Runtime bookkeeping
        self.evaluated: int = 0
        self.runtime: float = 0.0
        self.result: list = []
        self.PF: np.ndarray | None = None
        self._start_time: float = 0.0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def gen(self) -> int:
        """Current generation number."""
        return int(np.ceil(self.evaluated / self.N)) if self.N > 0 else 0

    @property
    def maxgen(self) -> int:
        """Maximum generation number."""
        return int(np.ceil(self.evaluation / self.N)) if self.N > 0 else 0

    # ------------------------------------------------------------------
    # Core workflow
    # ------------------------------------------------------------------
    def start(self) -> None:
        """Instantiate the problem, compute the true PF, and run the
        algorithm."""
        # Instantiate the problem
        self.problem = self.problem_class(self)

        # Get the true Pareto front
        self.PF = self.problem.pf(10000)

        # Run the algorithm
        self._start_time = time.perf_counter()
        try:
            self.algorithm(self)
        except _Termination:
            pass
        self.evaluated = max(self.evaluated, self.evaluation)
        self.runtime = time.perf_counter() - self._start_time

        # Final output
        self._output()

    def initialization(self, n: int | None = None) -> list[Individual]:
        """Randomly generate an initial population of *n* individuals."""
        if n is None:
            n = self.N
        decs = self.problem.init(n)
        return create_individuals(decs, self.problem, self)

    def not_termination(self, population: list[Individual]) -> bool:
        """Check whether the algorithm should continue.

        Stores the current population, prints progress, and raises
        :class:`_Termination` when the evaluation budget is exhausted.

        Parameters
        ----------
        population : list[Individual]
            Current population.

        Returns
        -------
        bool
            ``True`` if the evaluation budget has **not** been exhausted.
        """
        # Accumulate runtime
        self.runtime = time.perf_counter() - self._start_time

        # Save the population snapshot
        num = 10 if self.save <= 0 else self.save
        index = max(
            0,
            min(
                num - 1,
                int(np.ceil(num * self.evaluated / self.evaluation)) - 1,
            ),
        )
        # Extend result list if necessary
        while len(self.result) <= index:
            self.result.append(None)
        self.result[index] = (self.evaluated, list(population))

        # Progress display
        alg_name = getattr(self.algorithm, "__name__", str(self.algorithm))
        prob_name = type(self.problem).__name__
        pct = self.evaluated / self.evaluation * 100
        print(
            f"\r{alg_name} on {prob_name}, {self.M} objectives "
            f"{self.D} variables, run {self.run} "
            f"({pct:6.2f}%), {self.runtime:.2f}s passed...",
            end="",
            flush=True,
        )

        # Termination check
        if self.evaluated >= self.evaluation:
            raise _Termination
        return True

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------
    def _output(self) -> None:
        """Display or save results after algorithm termination."""
        print()  # newline after progress
        if self.save == 0 and self.result:
            # Get last population
            _, pop = self.result[-1]
            if pop:
                from igd import igd
                from draw import draw

                objs = pop_objs(pop)
                score = igd(objs, self.PF)
                print(f"IGD : {score:.4e}  Runtime : {self.runtime:.2f}s")
                prob_name = type(self.problem).__name__
                alg_name = getattr(self.algorithm, "__name__", str(self.algorithm))
                draw(objs, title=f"{alg_name} on {prob_name}")
        elif self.save > 0:
            import os
            alg_name = getattr(self.algorithm, "__name__", str(self.algorithm))
            prob_name = type(self.problem).__name__
            folder = os.path.join("Data", alg_name)
            os.makedirs(folder, exist_ok=True)
            fname = f"{alg_name}_{prob_name}_M{self.M}_D{self.D}_{self.run}.npz"
            # Save all snapshots
            np.savez(
                os.path.join(folder, fname),
                **{
                    f"eval_{i}": pop_objs(pop)
                    for i, (_, pop) in enumerate(self.result)
                    if pop is not None
                },
                runtime=self.runtime,
            )
            print(f"Results saved to {os.path.join(folder, fname)}")


class _Termination(Exception):
    """Internal exception to halt the algorithm when evaluations are
    exhausted — mirrors MATLAB's ``error('GLOBAL:Termination', ...)``.
    """
