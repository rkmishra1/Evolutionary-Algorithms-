<div align="center">

# 🧬 Evolutionary Algorithms Benchmark

### Seven multi-objective optimisation algorithms · Python · ZDT1

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](#quick-start)
[![Algorithms](https://img.shields.io/badge/Algorithms-7-brightgreen)](#algorithms)
[![Benchmark](https://img.shields.io/badge/Benchmark-ZDT1-orange)](#benchmark-setup)
[![Results](https://img.shields.io/badge/Results-RESULTS.md-blue)](RESULTS.md)

</div>

---

## Overview

This repository benchmarks **seven classic multi-objective evolutionary algorithms**, each implemented from scratch in pure Python. Every algorithm is evaluated on the **ZDT1** test problem under identical conditions, then ranked by convergence quality (IGD) and wall-clock runtime.

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
# Run all 7 algorithms and save results + figures
python run_all.py

# Run a single algorithm
python main.py -algorithm moeadabc -problem zdt1 -N 100
```

---

## Benchmark Setup

| Parameter | Value |
|---|---|
| **Problem** | ZDT1 — 2 objectives, 30 decision variables |
| **Population** | 100 |
| **Evaluations** | 10,000 |
| **Metric** | IGD ↓ (lower is better) |

---

## Results

### 📊 Performance Summary

| Rank | Algorithm | Family | IGD ↓ | Runtime (s) |
|:---:|---|---|:---:|:---:|
| 🥇 | **MOPSO** | Swarm | **9.59e-03** | 18.74 |
| 🥈 | **MOEAD-ABC** | Swarm | **9.60e-03** | **2.70** ⚡ |
| 3 | SPEA2 | Pareto | 4.85e+00 | 13.53 |
| 4 | PESA-II | Pareto | 8.77e+00 | 6.64 |
| 5 | NSGA-III | Pareto | 3.61e+01 | 32.24 |
| 6 | MOEA/D | Decomposition | 4.55e+01 | 19.24 |
| 7 | NSGA-II | Pareto | 4.68e+01 | 30.85 |

**Takeaways**

- 🎯 **MOPSO** and **MOEAD-ABC** both reach near-perfect Pareto approximation (IGD ≈ 0.009)
- ⚡ **MOEAD-ABC** is **7× faster** than MOPSO at the same quality — the most efficient algorithm overall
- 📉 GA-based algorithms (NSGA-II/III, MOEA/D) require larger evaluation budgets or advanced operators (SBX / polynomial mutation) to fully converge on 30-variable problems

---

### 📈 Pareto Front Comparison

![All algorithms vs. true ZDT1 Pareto front](figures/combined_comparison.png)

<details>
<summary>▶ Individual algorithm plots</summary>

<br>

| MOEAD-ABC | MOPSO |
|:---:|:---:|
| ![MOEAD-ABC](figures/moeadabc_ZDT1.png) | ![MOPSO](figures/mopso_ZDT1.png) |

| SPEA2 | PESA-II |
|:---:|:---:|
| ![SPEA2](figures/spea2_ZDT1.png) | ![PESA-II](figures/pesa2_ZDT1.png) |

| NSGA-III | MOEA/D | NSGA-II |
|:---:|:---:|:---:|
| ![NSGA-III](figures/nsga3_ZDT1.png) | ![MOEA/D](figures/moead_ZDT1.png) | ![NSGA-II](figures/nsga2_ZDT1.png) |

</details>

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
├── figures/          # Pareto front plots (PNG)
├── results/          # Raw numerical outputs
└── RESULTS.md        # Full benchmark report with analysis
```

---

## Citation

If you use this code in your research, please cite the relevant algorithm papers. See [RESULTS.md](RESULTS.md) for a full discussion of results.

---

## License

Released under the [MIT License](LICENSE).
