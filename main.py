#!/usr/bin/env python3
"""Command-line entry point for the MOEAD-ABC framework.

Usage examples:

    # Run MOEAD-ABC on ZDT1 with default settings
    python main.py -algorithm moeadabc -problem zdt1 -N 100 -M 2

    # With custom evaluation budget and aggregation type
    python main.py -algorithm moeadabc -problem zdt1 -N 200 -M 2 -evaluation 20000 -agg_type 3

    # Save 5 intermediate population snapshots
    python main.py -algorithm moeadabc -problem zdt1 -N 100 -M 2 -save 5 -run 1
"""

from __future__ import annotations

import argparse
import sys

from global_config import GlobalConfig


# ------------------------------------------------------------------
# Registry of available algorithms and problems
# ------------------------------------------------------------------
ALGORITHMS = {}
PROBLEMS = {}


def _register_defaults() -> None:
    """Lazy-import and register the built-in algorithms and problems."""
    from moeadabc import moeadabc
    from nsga2 import nsga2
    from mopso import mopso
    from spea2 import spea2
    from pesa2 import pesa2
    from moead import moead
    from nsga3 import nsga3
    from zdt1 import ZDT1
    from zdt2 import ZDT2
    from zdt3 import ZDT3
    from zdt4 import ZDT4

    ALGORITHMS["moeadabc"] = moeadabc
    ALGORITHMS["nsga2"] = nsga2
    ALGORITHMS["mopso"] = mopso
    ALGORITHMS["spea2"] = spea2
    ALGORITHMS["pesa2"] = pesa2
    ALGORITHMS["moead"] = moead
    ALGORITHMS["nsga3"] = nsga3
    PROBLEMS["zdt1"] = ZDT1
    PROBLEMS["zdt2"] = ZDT2
    PROBLEMS["zdt3"] = ZDT3
    PROBLEMS["zdt4"] = ZDT4


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------
def main() -> None:
    _register_defaults()

    parser = argparse.ArgumentParser(
        description="MOEAD-ABC: Multi-Objective Artificial Bee Colony "
                    "Algorithm based on Decomposition",
    )
    parser.add_argument(
        "-algorithm", type=str, default="moeadabc",
        choices=list(ALGORITHMS.keys()),
        help="Algorithm to run (default: moeadabc)",
    )
    parser.add_argument(
        "-problem", type=str, default="zdt1",
        choices=list(PROBLEMS.keys()),
        help="Test problem (default: zdt1)",
    )
    parser.add_argument("-N", type=int, default=100, help="Population size")
    parser.add_argument("-M", type=int, default=2, help="Number of objectives")
    parser.add_argument(
        "-D", type=int, default=0,
        help="Number of decision variables (0 → use problem default)",
    )
    parser.add_argument(
        "-evaluation", type=int, default=10000,
        help="Maximum number of objective evaluations",
    )
    parser.add_argument("-run", type=int, default=1, help="Run number")
    parser.add_argument(
        "-save", type=int, default=0,
        help="Number of intermediate populations to save (0 → display only)",
    )
    parser.add_argument(
        "-agg_type", type=int, default=1, choices=[1, 2, 3, 4],
        help="Aggregation function type: 1=PBI, 2=Tchebycheff, "
             "3=Normalised Tchebycheff, 4=Modified Tchebycheff",
    )

    args = parser.parse_args()

    config = GlobalConfig(
        N=args.N,
        M=args.M,
        D=args.D if args.D > 0 else None,
        algorithm=ALGORITHMS[args.algorithm],
        problem_class=PROBLEMS[args.problem],
        evaluation=args.evaluation,
        run=args.run,
        save=args.save,
        agg_type=args.agg_type,
    )
    config.start()


if __name__ == "__main__":
    main()
