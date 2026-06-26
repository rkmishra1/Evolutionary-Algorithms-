# Multi-Objective Optimization Algorithms Benchmark Results

This document presents the benchmark results of seven multi-objective optimization algorithms implemented in Python within the `MOABC` framework. All algorithms were evaluated on the **ZDT1** problem (2 objectives, 30 decision variables) with a population size of 100 and a budget of 10,000 objective function evaluations.

## Performance Summary

The performance of each algorithm is measured using the **Inverted Generational Distance (IGD)** (lower is better, measuring convergence and spread relative to the true Pareto front) and the **Execution Runtime** in seconds.

| Algorithm | IGD Score | Runtime (s) | Evaluations |
| :--- | :--- | :--- | :--- |
| **MOPSO** | **9.5871e-03** | 18.74 | 10,010 |
| **MOEAD-ABC** | **9.5983e-03** | **2.70** | 10,000 |
| **SPEA2** | 4.8484e+00 | 13.53 | 10,000 |
| **PESA-II** | 8.7719e+00 | 6.64 | 10,000 |
| **NSGA-III** | 3.6081e+01 | 32.24 | 10,000 |
| **MOEA/D** | 4.5485e+01 | 19.24 | 10,000 |
| **NSGA-II** | 4.6816e+01 | 30.85 | 10,000 |

### Key Observations
1. **MOPSO** and **MOEAD-ABC** achieved outstanding convergence, matching the true Pareto front almost perfectly with IGD values of ~0.009.
2. **MOEAD-ABC** was by far the fastest algorithm, completing 10,000 evaluations in just **2.70 seconds** due to its simple yet highly efficient search operators.
3. The genetic-operator-based algorithms (**NSGA-II**, **NSGA-III**, and **MOEA/D**) showed slower convergence under 10,000 evaluations when using the standard convex arithmetic crossover and Gaussian mutation operators translated from the Yarpiz MATLAB implementations. Real-world 30-variable problems typically require either advanced operators (like SBX/Polynomial mutation) or a larger evaluation budget to fully converge.

---

## Pareto Front Figures

### Combined Comparison
The figure below shows the final obtained Pareto fronts for all algorithms plotted against the true Pareto front (black line).

![Combined Comparison](figures/combined_comparison.png)

---

### Individual Algorithms

| Algorithm | Pareto Front Plot |
| :--- | :--- |
| **MOEAD-ABC** | ![MOEAD-ABC Front](figures/moeadabc_ZDT1.png) |
| **MOPSO** | ![MOPSO Front](figures/mopso_ZDT1.png) |
| **SPEA2** | ![SPEA2 Front](figures/spea2_ZDT1.png) |
| **PESA-II** | ![PESA-II Front](figures/pesa2_ZDT1.png) |
| **NSGA-III** | ![NSGA-III Front](figures/nsga3_ZDT1.png) |
| **MOEA/D** | ![MOEA/D Front](figures/moead_ZDT1.png) |
| **NSGA-II** | ![NSGA-II Front](figures/nsga2_ZDT1.png) |
