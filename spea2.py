"""SPEA2: Strength Pareto Evolutionary Algorithm 2.

Reference:
    E. Zitzler, M. Laumanns, and L. Thiele, SPEA2: Improving the Strength Pareto
    Evolutionary Algorithm, Technical Report 103, Computer Engineering and
    Networks Laboratory (TIK), ETH Zurich, 2001.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist
from individual import Individual, create_individuals

def dominates(p: Individual, q: Individual) -> bool:
    """Return True if p dominates q."""
    return np.all(p.obj <= q.obj) and np.any(p.obj < q.obj)

def binary_tournament_selection(pop: list[Individual]) -> Individual:
    """Select one individual from population using binary tournament based on fitness F."""
    i1, i2 = np.random.randint(0, len(pop), size=2)
    p1, p2 = pop[i1], pop[i2]
    return p1 if p1.F < p2.F else p2

def spea2(global_config) -> None:
    """Run the SPEA2 algorithm."""
    nPop = global_config.N
    nArchive = 100  # Default archive size
    D = global_config.D
    VarMin = global_config.lower
    VarMax = global_config.upper

    K = int(np.round(np.sqrt(nPop + nArchive)))

    pCrossover = 0.7
    nCrossover = 2 * int(np.round(pCrossover * nPop / 2))
    pMutation = 1.0 - pCrossover
    nMutation = nPop - nCrossover

    gamma = 0.1
    h = 0.2
    sigma_mut = h * (VarMax - VarMin)

    # Initialize population and archive
    pop = global_config.initialization()
    archive: list[Individual] = []

    # Main Loop
    while global_config.not_termination(archive if archive else pop):
        Q = pop + archive
        nQ = len(Q)

        # Domination matrices and Strength S calculation
        dom = np.zeros((nQ, nQ), dtype=bool)
        for i in range(nQ):
            for j in range(nQ):
                if i != j and dominates(Q[i], Q[j]):
                    dom[i, j] = True

        S = np.sum(dom, axis=1)
        for i in range(nQ):
            Q[i].S = S[i]

        # Raw fitness R calculation
        for i in range(nQ):
            Q[i].R = np.sum(S[dom[:, i]])

        # Density D and Fitness F calculation using standardized Euclidean distance
        Z = np.vstack([ind.obj for ind in Q])
        std_Z = np.std(Z, axis=0, ddof=1)
        std_Z[std_Z == 0] = 1e-10
        Z_norm = Z / std_Z
        
        SIGMA = cdist(Z_norm, Z_norm, metric="euclidean")
        SIGMA_sorted = np.sort(SIGMA, axis=0)

        for i in range(nQ):
            Q[i].sigma = SIGMA_sorted[:, i]
            # K-th nearest neighbor (0-indexed equivalent of 1-indexed K is K-1)
            Q[i].sigmaK = Q[i].sigma[K - 1]
            Q[i].D = 1.0 / (Q[i].sigmaK + 2)
            Q[i].F = Q[i].R + Q[i].D

        # Non-dominated count
        nd_indices = [i for i in range(nQ) if Q[i].R == 0]
        nND = len(nd_indices)

        if nND <= nArchive:
            # Sort Q based on fitness F
            Q_sorted = sorted(Q, key=lambda ind: ind.F)
            archive = [ind.copy() for ind in Q_sorted[:min(nArchive, nQ)]]
        else:
            # Pruning non-dominated individuals to nArchive size
            archive = [Q[idx].copy() for idx in nd_indices]
            
            # Sub-distance matrix for non-dominated individuals
            SIGMA_arc = SIGMA[nd_indices, :][:, nd_indices]
            SIGMA_arc_sorted = np.sort(SIGMA_arc, axis=0)

            k = 1  # 0-indexed equivalent of 2nd neighbor (1st is self at distance 0)
            while len(archive) > nArchive:
                row_vals = SIGMA_arc_sorted[k, :]
                while np.min(row_vals) == np.max(row_vals) and k < SIGMA_arc_sorted.shape[0] - 1:
                    k += 1
                    row_vals = SIGMA_arc_sorted[k, :]
                
                j = np.argmin(row_vals)
                archive.pop(j)
                SIGMA_arc_sorted = np.delete(SIGMA_arc_sorted, j, axis=1)

        # Crossover & Mutation for next generation
        offspring_decs = []

        # Crossover
        for _ in range(nCrossover // 2):
            p1 = binary_tournament_selection(archive)
            p2 = binary_tournament_selection(archive)
            alpha = np.random.uniform(-gamma, 1.0 + gamma, size=D)
            c1_dec = alpha * p1.dec + (1.0 - alpha) * p2.dec
            c2_dec = alpha * p2.dec + (1.0 - alpha) * p1.dec
            offspring_decs.append(c1_dec)
            offspring_decs.append(c2_dec)

        # Mutation
        for _ in range(nMutation):
            p = binary_tournament_selection(archive)
            y_dec = p.dec + sigma_mut * np.random.randn(D)
            offspring_decs.append(y_dec)

        # Evaluate and update pop
        offspring_decs = np.vstack(offspring_decs)
        pop = create_individuals(offspring_decs, global_config.problem, global_config)
