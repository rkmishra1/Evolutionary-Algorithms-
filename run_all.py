#!/usr/bin/env python3
"""Run experiments on ZDT1 for all 7 multiobjective algorithms,
calculate IGD metrics, generate plots, and save results.
"""

from __future__ import annotations

import os
import sys
import time
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
from zdt1 import ZDT1

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
    results_dir = os.path.join(repo_root, "results")
    figures_dir = os.path.join(repo_root, "figures")
    
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

    # Reference true Pareto front
    temp_config = GlobalConfig(N=100, M=2, D=30, problem_class=ZDT1)
    temp_problem = ZDT1(temp_config)
    true_pf = temp_problem.pf(1000)

    summary_data = []

    # Combined comparison figure setup
    plt.figure(figsize=(10, 8))
    plt.plot(true_pf[:, 0], true_pf[:, 1], "k-", label="True Pareto Front", linewidth=2.5, alpha=0.8)
    
    colors = {
        "nsga2": "#1f77b4",
        "mopso": "#ff7f0e",
        "spea2": "#2ca02c",
        "pesa2": "#d62728",
        "moead": "#9467bd",
        "nsga3": "#8c564b",
        "moeadabc": "#e377c2"
    }

    for name, alg_func in algorithms.items():
        print(f"\n==================================================")
        print(f"Running {name.upper()} on ZDT1...")
        print(f"==================================================")
        
        # We set save = 1 to prevent interactive plotting from drawing
        config = GlobalConfig(
            N=100,
            M=2,
            D=30,
            algorithm=alg_func,
            problem_class=ZDT1,
            evaluation=10000,
            save=1,  # Saves to file and avoids blocking draw() calls
        )
        
        config.start()
        
        # Retrieve final population from results
        if not config.result:
            print(f"Error: No results found for {name}")
            continue
            
        _, final_pop = config.result[-1]
        final_objs = pop_objs(final_pop)
        
        # Calculate IGD
        score = igd(final_objs, true_pf)
        print(f"\nFinished {name}: IGD = {score:.4e}, Runtime = {config.runtime:.2f}s")
        
        # Save CSV results
        csv_path = os.path.join(results_dir, f"{name}_ZDT1.csv")
        np.savetxt(csv_path, final_objs, delimiter=",", header="f1,f2", comments="")
        print(f"Saved CSV to {csv_path}")
        
        # Save individual plot
        plt.figure(figsize=(8, 6))
        plt.plot(true_pf[:, 0], true_pf[:, 1], "k-", label="True Pareto Front", linewidth=2)
        plt.scatter(final_objs[:, 0], final_objs[:, 1], c="red", s=25, edgecolors="k", label=f"Obtained PF ({name.upper()})")
        plt.title(f"Pareto Front for {name.upper()} on ZDT1\n(IGD: {score:.4e}, Runtime: {config.runtime:.2f}s)")
        plt.xlabel("Objective 1 (f1)")
        plt.ylabel("Objective 2 (f2)")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend()
        plt.tight_layout()
        
        fig_path = os.path.join(figures_dir, f"{name}_ZDT1.png")
        plt.savefig(fig_path, dpi=150)
        plt.close()
        print(f"Saved figure to {fig_path}")
        
        # Add to combined comparison plot
        # Need to re-activate the first figure
        plt.figure(1)
        plt.scatter(final_objs[:, 0], final_objs[:, 1], color=colors[name], s=15, alpha=0.7, label=name.upper())

        summary_data.append({
            "Algorithm": name.upper(),
            "IGD": score,
            "Runtime (s)": config.runtime,
            "Population Size": config.N,
            "Evaluations": config.evaluated
        })

    # Save combined comparison figure
    plt.figure(1)
    plt.title("Pareto Fronts Comparison on ZDT1")
    plt.xlabel("Objective 1 (f1)")
    plt.ylabel("Objective 2 (f2)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    combined_fig_path = os.path.join(figures_dir, "combined_comparison.png")
    plt.savefig(combined_fig_path, dpi=200)
    plt.close()
    print(f"\nSaved combined comparison figure to {combined_fig_path}")

    # Output Summary Table
    print("\n" + "="*70)
    print(f"{'Algorithm':<15} | {'IGD Score':<12} | {'Runtime (s)':<12} | {'Evaluations':<12}")
    print("="*70)
    for row in summary_data:
        print(f"{row['Algorithm']:<15} | {row['IGD']:<12.4e} | {row['Runtime (s)']:<12.2f} | {row['Evaluations']:<12}")
    print("="*70)

if __name__ == "__main__":
    main()
