<div align="center">

# 🧬 Evolutionary Algorithms Benchmark

### Seven multi-objective optimisation algorithms · Python · ZDT1 - ZDT4

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](#quick-start)
[![Algorithms](https://img.shields.io/badge/Algorithms-7-brightgreen)](#algorithms)
[![Benchmark](https://img.shields.io/badge/Benchmark-ZDT1--ZDT4-orange)](#benchmark-setup)
[![Results](https://img.shields.io/badge/Results-RESULTS.md-blue)](RESULTS.md)

</div>

---

## Overview

This repository benchmarks **seven classic multi-objective evolutionary algorithms**, each implemented from scratch in pure Python. All algorithms are evaluated across four classic test problems: **ZDT1, ZDT2, ZDT3, and ZDT4** under identical conditions, then compared by convergence quality (IGD) and wall-clock runtime.

> **Multi-objective optimisation** seeks a set of trade-off solutions (the *Pareto front*) rather than a single optimum — e.g. minimising cost *and* maximising performance simultaneously. IGD measures how closely an algorithm's output approximates the true Pareto front (lower = better).

---

## Algorithms

Grouped by paradigm:

### 🐝 Swarm-based
| Algorithm | Description | File |
|---|---|---|
| **MOEAD-ABC** | ABC search within a decomposition framework | [`moeadabc.py`](moeadabc.py) |
| **MOPSO** | Particle swarm with Pareto archiving | [`mopso.py`](mopso.py) |

### 🔀 Decomposition-based
| Algorithm | Description | File |
|---|---|---|
| **MOEA/D** | Scalarisation via weight vectors | [`moead.py`](moead.py) |

### 🏆 Pareto-dominance-based
| Algorithm | Description | File |
|---|---|---|
| **NSGA-II** | Fast non-dominated sort + crowding distance | [`nsga2.py`](nsga2.py) |
| **NSGA-III** | Reference-point guided selection | [`nsga3.py`](nsga3.py) |
| **SPEA2** | Strength-based fitness + archive | [`spea2.py`](spea2.py) |
| **PESA-II** | Grid-based density + Pareto archive | [`pesa2.py`](pesa2.py) |

---

## Quick Start

```bash
# Run all 7 algorithms on ZDT1, ZDT2, ZDT3, ZDT4 and save results + figures
python run_all.py

# Run a single algorithm on a specific problem
python main.py -algorithm moeadabc -problem zdt2 -N 100
```

---

## Benchmark Setup

- **Population Size (\(N\))**: 100
- **Evaluation Budget**: 10,000 objective function evaluations
- **Problems**:
  - **ZDT1**: 30 variables, convex front (\(f_2 = 1 - \sqrt{f_1}\))
  - **ZDT2**: 30 variables, non-convex front (\(f_2 = 1 - f_1^2\))
  - **ZDT3**: 30 variables, discontinuous front (\(f_2 = 1 - \sqrt{f_1} - f_1 \sin(10 \pi f_1)\))
  - **ZDT4**: 10 variables, multi-modal search space with \(21^9\) local fronts
- **Metric**: IGD ↓ (lower is better, measuring convergence and diversity)

---

## Results Summary

A brief summary of IGD scores and Runtimes (s) for each problem:

### ZDT1 (Convex)
- 🥇 **MOPSO** (IGD: **7.84e-03**, Runtime: 13.01s)
- 🥈 **MOEAD-ABC** (IGD: **8.38e-03**, Runtime: **2.46s** ⚡)
- PESA-II: 3.48e+00 | SPEA2: 5.63e+00 | NSGA-III: 3.54e+01 | MOEA/D: 3.70e+01 | NSGA-II: 4.54e+01

### ZDT2 (Non-Convex)
- 🥇 **MOPSO** (IGD: **9.02e-03**, Runtime: 16.28s)
- 🥈 **MOEAD-ABC** (IGD: **1.77e-02**, Runtime: **2.51s** ⚡)
- SPEA2: 1.86e+00 | PESA-II: 1.04e+01 | NSGA-III: 3.70e+01 | NSGA-II: 4.07e+01 | MOEA/D: 6.02e+01

### ZDT3 (Discontinuous)
- 🥇 **MOPSO** (IGD: **8.47e-03**, Runtime: 7.85s)
- 🥈 **MOEAD-ABC** (IGD: **1.59e-02**, Runtime: **2.45s** ⚡)
- SPEA2: 2.74e+00 | PESA-II: 5.84e+00 | NSGA-III: 3.80e+01 | MOEA/D: 4.22e+01 | NSGA-II: 4.32e+01

### ZDT4 (Multi-Modal)
- 🥇 **MOEAD-ABC** (IGD: **4.83e-01**, Runtime: **2.34s** ⚡)
- 🥈 **MOPSO** (IGD: **2.48e+00**, Runtime: 4.00s)
- NSGA-III: 3.51e+00 | NSGA-II: 4.17e+00 | SPEA2: 4.99e+00 | MOEA/D: 6.12e+00 | PESA-II: 2.30e+01

---

## Pareto Front Comparison Plots

### ZDT1
![All algorithms vs. true ZDT1 Pareto front](figures/combined_comparison_ZDT1.png)

### ZDT2
![All algorithms vs. true ZDT2 Pareto front](figures/combined_comparison_ZDT2.png)

### ZDT3
![All algorithms vs. true ZDT3 Pareto front](figures/combined_comparison_ZDT3.png)

### ZDT4
![All algorithms vs. true ZDT4 Pareto front](figures/combined_comparison_ZDT4.png)

> Refer to [RESULTS.md](RESULTS.md) for full metrics tables, individual algorithm plots, and an in-depth performance analysis.

---

## Repository Structure

```
Evolutionary-Algorithms/
├── main.py           # Single-algorithm entry point (CLI)
├── run_all.py        # Run all algorithms → results/ + figures/
├── individual.py     # Shared solution representation
│
├── moeadabc.py       # MOEAD-ABC
├── moead.py          # MOEA/D
├── mopso.py          # MOPSO
├── nsga2.py          # NSGA-II
├── nsga3.py          # NSGA-III
├── spea2.py          # SPEA2
├── pesa2.py          # PESA-II
│
├── zdt1.py, zdt2.py, zdt3.py, zdt4.py # Problem definitions
├── figures/          # Pareto front plots (PNG)
├── results/          # Raw numerical outputs
├── matlab/           # Original MATLAB reference code
└── RESULTS.md        # Full benchmark report with analysis
```

---

## License

Released under the [MIT License](LICENSE).
