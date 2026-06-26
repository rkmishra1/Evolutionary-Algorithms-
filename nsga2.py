"""NSGA-II: Non-dominated Sorting Genetic Algorithm II.

Reference:
    K. Deb, A. Pratap, S. Agarwal, and T. Meyarivan, A fast and elitist
    multiobjective genetic algorithm: NSGA-II, IEEE Transactions on
    Evolutionary Computation, 2002, 6(2): 182-197.
"""

from __future__ import annotations

import numpy as np
from individual import Individual, create_individuals

def dominates(p: Individual, q: Individual) -> bool:
    """Return True if p dominates q."""
    return np.all(p.obj <= q.obj) and np.any(p.obj < q.obj)

def non_dominated_sorting(pop: list[Individual]) -> tuple[list[Individual], list[list[int]]]:
    """Perform non-dominated sorting on the population."""
    n = len(pop)
    for ind in pop:
        ind.domination_set = []
        ind.dominated_count = 0
        ind.rank = 0

    F = [[]]
    for i in range(n):
        p = pop[i]
        for j in range(n):
            if i == j:
                continue
            q = pop[j]
            if dominates(p, q):
                p.domination_set.append(j)
            elif dominates(q, p):
                p.dominated_count += 1
        
        if p.dominated_count == 0:
            p.rank = 1
            F[0].append(i)

    k = 0
    while True:
        Q = []
        for i in F[k]:
            p = pop[i]
            for j in p.domination_set:
                q = pop[j]
                q.dominated_count -= 1
                if q.dominated_count == 0:
                    q.rank = k + 2
                    Q.append(j)
        if not Q:
            break
        F.append(Q)
        k += 1

    return pop, F

def calc_crowding_distance(pop: list[Individual], F: list[list[int]]) -> list[Individual]:
    """Calculate crowding distance for each individual in the fronts."""
    for front in F:
        if not front:
            continue
        n = len(front)
        for idx in front:
            pop[idx].crowding_distance = 0.0

        m = len(pop[front[0]].obj)
        for j in range(m):
            sorted_indices = sorted(front, key=lambda idx: pop[idx].obj[j])
            
            pop[sorted_indices[0]].crowding_distance = float("inf")
            pop[sorted_indices[-1]].crowding_distance = float("inf")

            min_val = pop[sorted_indices[0]].obj[j]
            max_val = pop[sorted_indices[-1]].obj[j]
            diff = max_val - min_val
            if diff == 0:
                diff = 1e-10

            for i in range(1, n - 1):
                prev_val = pop[sorted_indices[i - 1]].obj[j]
                next_val = pop[sorted_indices[i + 1]].obj[j]
                pop[sorted_indices[i]].crowding_distance += (next_val - prev_val) / diff
    return pop

def sort_population(pop: list[Individual]) -> list[Individual]:
    """Sort the population based on rank and crowding distance."""
    sorted_pop = list(pop)
    sorted_pop.sort(key=lambda ind: (ind.rank, -ind.crowding_distance))
    
    # Re-assign ranks to front groupings
    ranks = [ind.rank for ind in sorted_pop]
    max_rank = max(ranks) if ranks else 0
    F = [[] for _ in range(max_rank)]
    for idx, ind in enumerate(sorted_pop):
        F[ind.rank - 1].append(idx)
        
    return sorted_pop

def nsga2(global_config) -> None:
    """Run the NSGA-II algorithm."""
    nPop = global_config.N
    D = global_config.D

    pCrossover = 0.7
    nCrossover = 2 * int(np.round(pCrossover * nPop / 2))
    pMutation = 0.4
    nMutation = int(np.round(pMutation * nPop))
    mu = 0.02
    sigma = 0.1 * (global_config.upper - global_config.lower)

    # Initialization
    pop = global_config.initialization()
    pop, F = non_dominated_sorting(pop)
    pop = calc_crowding_distance(pop, F)
    pop = sort_population(pop)

    while global_config.not_termination(pop):
        offspring_decs = []
        
        # Crossover
        for _ in range(nCrossover // 2):
            i1, i2 = np.random.randint(0, nPop, size=2)
            p1, p2 = pop[i1], pop[i2]
            alpha = np.random.uniform(0, 1, size=D)
            c1_dec = alpha * p1.dec + (1.0 - alpha) * p2.dec
            c2_dec = alpha * p2.dec + (1.0 - alpha) * p1.dec
            offspring_decs.append(c1_dec)
            offspring_decs.append(c2_dec)
            
        # Mutation
        for _ in range(nMutation):
            i = np.random.randint(0, nPop)
            p = pop[i]
            y_dec = p.dec.copy()
            nMu = int(np.ceil(mu * D))
            j = np.random.choice(D, size=nMu, replace=False)
            noise = sigma[j] * np.random.randn(nMu)
            y_dec[j] = y_dec[j] + noise
            offspring_decs.append(y_dec)

        # Evaluate offspring
        offspring_decs = np.vstack(offspring_decs)
        offspring_pop = create_individuals(offspring_decs, global_config.problem, global_config)
        
        # Merge
        pop = pop + offspring_pop
        
        # Sort and truncate
        pop, F = non_dominated_sorting(pop)
        pop = calc_crowding_distance(pop, F)
        pop = sort_population(pop)
        pop = pop[:nPop]
        
        # Re-sort to update ranks and distances of truncated population
        pop, F = non_dominated_sorting(pop)
        pop = calc_crowding_distance(pop, F)
        pop = sort_population(pop)
