# Multi-Objective Optimization Algorithms Benchmark Results

This document presents the benchmark results of seven multi-objective optimization algorithms evaluated on four standard test problems: **ZDT1, ZDT2, ZDT3, and ZDT4**.

All problems are bi-objective (\(M=2\)). 
- **ZDT1**, **ZDT2**, and **ZDT3** are configured with 30 decision variables (\(D=30\)).
- **ZDT4** is configured with 10 decision variables (\(D=10\)).

All algorithms ran with a population size of 100 and a budget of 10,000 objective function evaluations. Runtimes and Inverted Generational Distance (IGD) scores are measured relative to the true Pareto front.

---

## 📊 Performance Summary

The performance of each algorithm is measured using the **Inverted Generational Distance (IGD)** (lower is better, measuring convergence and spread) and the **Execution Runtime** in seconds.

### ZDT1 Benchmark Summary
*30 variables, convex Pareto front: \(f_2 = 1 - \sqrt{f_1}\)*

| Algorithm | IGD Score | Runtime (s) | Evaluations |
| :--- | :---: | :---: | :---: |
| **MOPSO** | **7.8354e-03** | 13.01 | 10,035 |
| **MOEAD-ABC** | **8.3809e-03** | **2.46** | 10,000 |
| **PESA-II** | 3.4757e+00 | 5.86 | 10,000 |
| **SPEA2** | 5.6250e+00 | 13.19 | 10,000 |
| **NSGA-III** | 3.5385e+01 | 30.15 | 10,000 |
| **MOEA/D** | 3.7034e+01 | 12.07 | 10,000 |
| **NSGA-II** | 4.5366e+01 | 27.37 | 10,000 |

### ZDT2 Benchmark Summary
*30 variables, non-convex Pareto front: \(f_2 = 1 - f_1^2\)*

| Algorithm | IGD Score | Runtime (s) | Evaluations |
| :--- | :---: | :---: | :---: |
| **MOPSO** | **9.0240e-03** | 16.28 | 10,009 |
| **MOEAD-ABC** | **1.7681e-02** | **2.51** | 10,000 |
| **SPEA2** | 1.8645e+00 | 29.13 | 10,000 |
| **PESA-II** | 1.0445e+01 | 12.03 | 10,000 |
| **NSGA-III** | 3.6981e+01 | 31.23 | 10,000 |
| **NSGA-II** | 4.0734e+01 | 29.28 | 10,000 |
| **MOEA/D** | 6.0194e+01 | 15.79 | 10,000 |

### ZDT3 Benchmark Summary
*30 variables, discontinuous Pareto front with 5 segments*

| Algorithm | IGD Score | Runtime (s) | Evaluations |
| :--- | :---: | :---: | :---: |
| **MOPSO** | **8.4670e-03** | 7.85 | 10,053 |
| **MOEAD-ABC** | **1.5876e-02** | **2.45** | 10,000 |
| **SPEA2** | 2.7380e+00 | 12.91 | 10,000 |
| **PESA-II** | 5.8431e+00 | 5.80 | 10,000 |
| **NSGA-III** | 3.8027e+01 | 30.26 | 10,000 |
| **MOEA/D** | 4.2150e+01 | 10.87 | 10,000 |
| **NSGA-II** | 4.3182e+01 | 29.79 | 10,000 |

### ZDT4 Benchmark Summary
*10 variables, multi-modal search space with \(21^9\) local Pareto-optimal fronts*

| Algorithm | IGD Score | Runtime (s) | Evaluations |
| :--- | :---: | :---: | :---: |
| **MOEAD-ABC** | **4.8329e-01** | **2.34** | 10,000 |
| **MOPSO** | 2.4772e+00 | 4.00 | 10,046 |
| **NSGA-III** | 3.5079e+00 | 31.77 | 10,000 |
| **NSGA-II** | 4.1737e+00 | 29.98 | 10,000 |
| **SPEA2** | 4.9914e+00 | 13.55 | 10,000 |
| **MOEA/D** | 6.1168e+00 | 9.02 | 10,000 |
| **PESA-II** | 2.3022e+01 | 7.06 | 10,000 |

---

## 📈 Pareto Front Plots

### ZDT1 Results
![Combined ZDT1](figures/combined_comparison_ZDT1.png)

| **MOEAD-ABC** | **MOPSO** | **SPEA2** | **PESA-II** |
| :---: | :---: | :---: | :---: |
| ![moeadabc_ZDT1](figures/moeadabc_ZDT1.png) | ![mopso_ZDT1](figures/mopso_ZDT1.png) | ![spea2_ZDT1](figures/spea2_ZDT1.png) | ![pesa2_ZDT1](figures/pesa2_ZDT1.png) |
| **NSGA-III** | **MOEA/D** | **NSGA-II** | |
| ![nsga3_ZDT1](figures/nsga3_ZDT1.png) | ![moead_ZDT1](figures/moead_ZDT1.png) | ![nsga2_ZDT1](figures/nsga2_ZDT1.png) | |

### ZDT2 Results
![Combined ZDT2](figures/combined_comparison_ZDT2.png)

| **MOEAD-ABC** | **MOPSO** | **SPEA2** | **PESA-II** |
| :---: | :---: | :---: | :---: |
| ![moeadabc_ZDT2](figures/moeadabc_ZDT2.png) | ![mopso_ZDT2](figures/mopso_ZDT2.png) | ![spea2_ZDT2](figures/spea2_ZDT2.png) | ![pesa2_ZDT2](figures/pesa2_ZDT2.png) |
| **NSGA-III** | **MOEA/D** | **NSGA-II** | |
| ![nsga3_ZDT2](figures/nsga3_ZDT2.png) | ![moead_ZDT2](figures/moead_ZDT2.png) | ![nsga2_ZDT2](figures/nsga2_ZDT2.png) | |

### ZDT3 Results
![Combined ZDT3](figures/combined_comparison_ZDT3.png)

| **MOEAD-ABC** | **MOPSO** | **SPEA2** | **PESA-II** |
| :---: | :---: | :---: | :---: |
| ![moeadabc_ZDT3](figures/moeadabc_ZDT3.png) | ![mopso_ZDT3](figures/mopso_ZDT3.png) | ![spea2_ZDT3](figures/spea2_ZDT3.png) | ![pesa2_ZDT3](figures/pesa2_ZDT3.png) |
| **NSGA-III** | **MOEA/D** | **NSGA-II** | |
| ![nsga3_ZDT3](figures/nsga3_ZDT3.png) | ![moead_ZDT3](figures/moead_ZDT3.png) | ![nsga2_ZDT3](figures/nsga2_ZDT3.png) | |

### ZDT4 Results
![Combined ZDT4](figures/combined_comparison_ZDT4.png)

| **MOEAD-ABC** | **MOPSO** | **SPEA2** | **PESA-II** |
| :---: | :---: | :---: | :---: |
| ![moeadabc_ZDT4](figures/moeadabc_ZDT4.png) | ![mopso_ZDT4](figures/mopso_ZDT4.png) | ![spea2_ZDT4](figures/spea2_ZDT4.png) | ![pesa2_ZDT4](figures/pesa2_ZDT4.png) |
| **NSGA-III** | **MOEA/D** | **NSGA-II** | |
| ![nsga3_ZDT4](figures/nsga3_ZDT4.png) | ![moead_ZDT4](figures/moead_ZDT4.png) | ![nsga2_ZDT4](figures/nsga2_ZDT4.png) | |
