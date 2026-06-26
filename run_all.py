#!/usr/bin/env python3
"""Run experiments on ZDT1, ZDT2, ZDT3, and ZDT4 for all 7 multiobjective algorithms,
calculate IGD metrics, generate plots, and save results.
"""

from __future__ import annotations

import os
import sys
import numpy as np

# Use Agg backend for non-interactive plotting
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Add current path to import registered modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from global_config import GlobalConfig
from individual import pop_objs
from igd import igd

# Import problems
from zdt1 import ZDT1
from zdt2 import ZDT2
from zdt3 import ZDT3
from zdt4 import ZDT4

# Import algorithms
from moeadabc import moeadabc
from nsga2 import nsga2
from mopso import mopso
from spea2 import spea2
from pesa2 import pesa2
from moead import moead
from nsga3 import nsga3

def main():
    # Setup directories
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # We save results and figures in folders inside implementations_and_results directory
    # so we can easily commit them to the new repository root
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    figures_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
    
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    algorithms = {
        "nsga2": nsga2,
        "mopso": mopso,
        "spea2": spea2,
        "pesa2": pesa2,
        "moead": moead,
        "nsga3": nsga3,
        "moeadabc": moeadabc,
    }

    problems = {
        "ZDT1": ZDT1,
        "ZDT2": ZDT2,
        "ZDT3": ZDT3,
        "ZDT4": ZDT4,
    }

    colors = {
        "nsga2": "#1f77b4",
        "mopso": "#ff7f0e",
        "spea2": "#2ca02c",
        "pesa2": "#d62728",
        "moead": "#9467bd",
        "nsga3": "#8c564b",
        "moeadabc": "#e377c2"
    }

    all_summaries = {}

    for prob_name, prob_class in problems.items():
        print(f"\n##################################################")
        print(f" BENCHMARKING PROBLEM: {prob_name}")
        print(f"##################################################")

        # Reference true Pareto front
        temp_config = GlobalConfig(N=100, M=2, D=None, problem_class=prob_class)
        temp_problem = prob_class(temp_config)
        true_pf = temp_problem.pf(1000)

        # Save true Pareto front to CSV for plotting if needed
        true_pf_path = os.path.join(results_dir, f"{prob_name}_true_pf.csv")
        np.savetxt(true_pf_path, true_pf, delimiter=",", header="f1,f2", comments="")

        summary_data = []

        # Combined comparison figure setup
        plt.figure(figsize=(10, 8))
        # Plot true Pareto front (as discontinuous scatter or continuous line)
        if prob_name == "ZDT3":
            # For discontinuous front, scatter is cleaner
            plt.scatter(true_pf[:, 0], true_pf[:, 1], c="black", s=8, label="True Pareto Front", alpha=0.8)
        else:
            plt.plot(true_pf[:, 0], true_pf[:, 1], "k-", label="True Pareto Front", linewidth=2.5, alpha=0.8)

        for name, alg_func in algorithms.items():
            print(f"\n--- Running {name.upper()} on {prob_name} ---")
            
            # Instantiate global config (D=None lets the problem choose default dimension)
            config = GlobalConfig(
                N=100,
                M=2,
                D=None,
                algorithm=alg_func,
                problem_class=prob_class,
                evaluation=10000,
                save=1,
            )
            
            config.start()
            
            if not config.result:
                print(f"Error: No results found for {name} on {prob_name}")
                continue
                
            _, final_pop = config.result[-1]
            final_objs = pop_objs(final_pop)
            
            # Calculate IGD
            score = igd(final_objs, true_pf)
            print(f"Finished: IGD = {score:.4e}, Runtime = {config.runtime:.2f}s")
            
            # Save CSV results
            csv_path = os.path.join(results_dir, f"{name}_{prob_name}.csv")
            np.savetxt(csv_path, final_objs, delimiter=",", header="f1,f2", comments="")
            
            # Save individual plot in Yarpiz style
            plt.figure(figsize=(8, 6))
            if prob_name == "ZDT3":
                plt.scatter(true_pf[:, 0], true_pf[:, 1], c="gray", s=4, label="True Pareto Front", alpha=0.5)
            else:
                plt.plot(true_pf[:, 0], true_pf[:, 1], "k-", label="True Pareto Front", linewidth=1.5, alpha=0.5)
            
            # Obtained front in Yarpiz style: red stars
            plt.plot(final_objs[:, 0], final_objs[:, 1], "r*", markersize=8, label="Non-dominated Solutions")
            
            # Title based on algorithm type
            if name in ("mopso", "pesa2", "moead"):
                plt.title(f"Repository ({name.upper()} on {prob_name})\nIGD: {score:.4e}, Runtime: {config.runtime:.2f}s")
            else:
                plt.title(f"Non-dominated Solutions F_1 ({name.upper()} on {prob_name})\nIGD: {score:.4e}, Runtime: {config.runtime:.2f}s")
                
            plt.xlabel("1st Objective")
            plt.ylabel("2nd Objective")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            
            fig_path = os.path.join(figures_dir, f"{name}_{prob_name}.png")
            plt.savefig(fig_path, dpi=150)
            plt.close()
            
            # Add to combined comparison plot
            plt.figure(1)
            plt.scatter(final_objs[:, 0], final_objs[:, 1], color=colors[name], s=15, alpha=0.7, label=name.upper())

            summary_data.append({
                "Algorithm": name.upper(),
                "IGD": score,
                "Runtime (s)": config.runtime,
                "Evaluations": config.evaluated
            })

        # Save combined comparison figure
        plt.figure(1)
        plt.title(f"Pareto Fronts Comparison on {prob_name}")
        plt.xlabel("Objective 1 (f1)")
        plt.ylabel("Objective 2 (f2)")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend()
        plt.tight_layout()
        combined_fig_path = os.path.join(figures_dir, f"combined_comparison_{prob_name}.png")
        plt.savefig(combined_fig_path, dpi=200)
        plt.close()
        print(f"\nSaved combined comparison figure to {combined_fig_path}")

        all_summaries[prob_name] = summary_data

    # Print overall summary table to stdout
    for prob_name, summary_data in all_summaries.items():
        print("\n" + "="*70)
        print(f" SUMMARY FOR {prob_name}")
        print("="*70)
        print(f"{'Algorithm':<15} | {'IGD Score':<12} | {'Runtime (s)':<12} | {'Evaluations':<12}")
        print("="*70)
        for row in summary_data:
            print(f"{row['Algorithm']:<15} | {row['IGD']:<12.4e} | {row['Runtime (s)']:<12.2f} | {row['Evaluations']:<12}")
        print("="*70)

if __name__ == "__main__":
    main()
