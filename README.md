<div align="center">

# 🧬 Evolutionary Algorithms Benchmark

### Seven multi-objective optimisation algorithms · Python · ZDT1 – ZDT4

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](#quick-start)
[![Algorithms](https://img.shields.io/badge/Algorithms-7-brightgreen)](#algorithms)
[![Problems](https://img.shields.io/badge/Problems-ZDT1–ZDT4-orange)](#benchmark-setup)
[![Results](https://img.shields.io/badge/Full%20Report-RESULTS.md-blue)](RESULTS.md)

</div>

---

## Overview

This repository benchmarks **seven classic multi-objective evolutionary algorithms**, each implemented from scratch in pure Python, across four standard ZDT test problems under identical conditions.

> **Multi-objective optimisation** seeks a set of trade-off solutions — the *Pareto front* — rather than a single optimum (e.g. minimise cost *and* maximise performance simultaneously). **IGD** (Inverted Generational Distance) measures how closely an algorithm approximates the true Pareto front. Lower is better.

---

## Algorithms

### 🐝 Swarm-based
| Algorithm | Description | File |
|---|---|---|
| **MOEAD-ABC** | Artificial Bee Colony search within a decomposition framework | [`moeadabc.py`](moeadabc.py) |
| **MOPSO** | Particle Swarm Optimisation with Pareto archiving | [`mopso.py`](mopso.py) |

### 🔀 Decomposition-based
| Algorithm | Description | File |
|---|---|---|
| **MOEA/D** | Scalarisation into scalar sub-problems via weight vectors | [`moead.py`](moead.py) |

### 🏆 Pareto-dominance-based
| Algorithm | Description | File |
|---|---|---|
| **NSGA-II** | Fast non-dominated sort with crowding distance | [`nsga2.py`](nsga2.py) |
| **NSGA-III** | Reference-point guided selection for many objectives | [`nsga3.py`](nsga3.py) |
| **SPEA2** | Strength-based fitness assignment with external archive | [`spea2.py`](spea2.py) |
| **PESA-II** | Grid-based density estimation with Pareto archive | [`pesa2.py`](pesa2.py) |

---

## Quick Start

```bash
# Run all 7 algorithms on ZDT1–ZDT4 and save results + figures
python run_all.py

# Run a single algorithm on a specific problem
python main.py -algorithm moeadabc -problem zdt2 -N 100
```

---

## Benchmark Setup

| Parameter | Value |
|---|---|
| **Population size** | 100 |
| **Evaluation budget** | 10,000 objective function evaluations |
| **Metric** | IGD ↓ (lower = better convergence & diversity) |

### Test Problems

| Problem | Variables | Front shape |
|---|:---:|---|
| **ZDT1** | 30 | Convex |
| **ZDT2** | 30 | Non-convex |
| **ZDT3** | 30 | Discontinuous (5 segments) |
| **ZDT4** | 10 | Multi-modal — 21⁹ local fronts |

---

## Results

### ZDT1 — Convex front

| Rank | Algorithm | IGD ↓ | Runtime (s) |
|:---:|---|:---:|:---:|
| 🥇 | **MOPSO** | **7.84e-03** | 13.01 |
| 🥈 | **MOEAD-ABC** | **8.38e-03** | **2.46** ⚡ |
| 3 | PESA-II | 3.48e+00 | — |
| 4 | SPEA2 | 5.63e+00 | — |
| 5 | NSGA-III | 3.54e+01 | — |
| 6 | MOEA/D | 3.70e+01 | — |
| 7 | NSGA-II | 4.54e+01 | — |

### ZDT2 — Non-convex front

| Rank | Algorithm | IGD ↓ | Runtime (s) |
|:---:|---|:---:|:---:|
| 🥇 | **MOPSO** | **9.02e-03** | 16.28 |
| 🥈 | **MOEAD-ABC** | **1.77e-02** | **2.51** ⚡ |
| 3 | SPEA2 | 1.86e+00 | — |
| 4 | PESA-II | 1.04e+01 | — |
| 5 | NSGA-III | 3.70e+01 | — |
| 6 | NSGA-II | 4.07e+01 | — |
| 7 | MOEA/D | 6.02e+01 | — |

### ZDT3 — Discontinuous front

| Rank | Algorithm | IGD ↓ | Runtime (s) |
|:---:|---|:---:|:---:|
| 🥇 | **MOPSO** | **8.47e-03** | 7.85 |
| 🥈 | **MOEAD-ABC** | **1.59e-02** | **2.45** ⚡ |
| 3 | SPEA2 | 2.74e+00 | — |
| 4 | PESA-II | 5.84e+00 | — |
| 5 | NSGA-III | 3.80e+01 | — |
| 6 | MOEA/D | 4.22e+01 | — |
| 7 | NSGA-II | 4.32e+01 | — |

### ZDT4 — Multi-modal front

| Rank | Algorithm | IGD ↓ | Runtime (s) |
|:---:|---|:---:|:---:|
| 🥇 | **MOEAD-ABC** | **4.83e-01** | **2.34** ⚡ |
| 🥈 | **MOPSO** | **2.48e+00** | 4.00 |
| 3 | NSGA-III | 3.51e+00 | — |
| 4 | NSGA-II | 4.17e+00 | — |
| 5 | SPEA2 | 4.99e+00 | — |
| 6 | MOEA/D | 6.12e+00 | — |
| 7 | PESA-II | 2.30e+01 | — |

### Key Takeaways

- 🐝 **MOEAD-ABC** is the most efficient algorithm — **3–7× faster** than MOPSO with near-identical quality on ZDT1–3, and **outright winner** on the harder ZDT4 multi-modal problem
- 🌊 **MOPSO** is the top performer on convex/non-convex/discontinuous fronts when runtime is not a constraint
- 📉 GA-based algorithms (NSGA-II/III, MOEA/D) need larger evaluation budgets or advanced operators (SBX / polynomial mutation) to close the gap on 30-variable problems

---

## Pareto Front Plots

<table>
  <tr>
    <td align="center"><b>ZDT1 — Convex</b><br><img src="figures/combined_comparison_ZDT1.png"/></td>
    <td align="center"><b>ZDT2 — Non-convex</b><br><img src="figures/combined_comparison_ZDT2.png"/></td>
  </tr>
  <tr>
    <td align="center"><b>ZDT3 — Discontinuous</b><br><img src="figures/combined_comparison_ZDT3.png"/></td>
    <td align="center"><b>ZDT4 — Multi-modal</b><br><img src="figures/combined_comparison_ZDT4.png"/></td>
  </tr>
</table>

> Individual per-algorithm plots and full metric tables are in [RESULTS.md](RESULTS.md).

---

## Repository Structure

```
Evolutionary-Algorithms/
├── main.py              # Single-algorithm CLI entry point
├── run_all.py           # Run all 7 algorithms on all 4 problems
├── individual.py        # Shared solution representation
│
├── moeadabc.py          # MOEAD-ABC
├── moead.py             # MOEA/D
├── mopso.py             # MOPSO
├── nsga2.py             # NSGA-II
├── nsga3.py             # NSGA-III
├── spea2.py             # SPEA2
├── pesa2.py             # PESA-II
│
├── zdt1.py – zdt4.py   # Problem definitions
├── figures/             # Pareto front plots (PNG)
├── results/             # Raw numerical outputs
└── RESULTS.md           # Full benchmark report
```

---

## License

Released under the [MIT License](LICENSE).
