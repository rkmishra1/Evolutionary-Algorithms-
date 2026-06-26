"""MOEA/D: Multi-Objective Evolutionary Algorithm based on Decomposition.

Reference:
    Q. Zhang and H. Li, MOEA/D: A multiobjective evolutionary algorithm based
    on decomposition, IEEE Transactions on Evolutionary Computation, 2007,
    11(6): 712-731.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist
from individual import Individual, create_individuals

def dominates(p: Individual, q: Individual) -> bool:
    """Return True if p dominates q."""
    return np.all(p.obj <= q.obj) and np.any(p.obj < q.obj)

def determine_domination(pop: list[Individual]) -> list[Individual]:
    """Determine domination status of all individuals."""
    n = len(pop)
    for ind in pop:
        ind.is_dominated = False

    for i in range(n - 1):
        for j in range(i + 1, n):
            if dominates(pop[i], pop[j]):
                pop[j].is_dominated = True
            elif dominates(pop[j], pop[i]):
                pop[i].is_dominated = True
    return pop

def decomposed_cost(ind: Individual, z: np.ndarray, lambd: np.ndarray) -> float:
    """Calculate Decomposed Tchebycheff Cost."""
    return float(np.max(lambd * np.abs(ind.obj - z)))

def moead(global_config) -> None:
    """Run the MOEA/D algorithm."""
    nPop = global_config.N
    m = global_config.M
    D = global_config.D
    nArchive = 100  # Default EP size
    VarMin = global_config.lower
    VarMax = global_config.upper

    # Neighborhood size
    T = int(np.minimum(np.maximum(np.ceil(0.15 * nPop), 2), 15))

    # Create sub-problems (weights and neighborhood)
    lambdas = np.random.rand(nPop, m)
    lambdas = lambdas / np.linalg.norm(lambdas, axis=1, keepdims=True)
    
    dist_matrix = cdist(lambdas, lambdas)
    neighbors = np.argsort(dist_matrix, axis=1)[:, :T]

    # Initialize population
    pop = global_config.initialization()
    
    # Initialize ideal point
    z = np.min(np.vstack([ind.obj for ind in pop]), axis=0)

    # Compute initial decomposed costs
    for i in range(nPop):
        pop[i].g = decomposed_cost(pop[i], z, lambdas[i])

    # Determine Population Domination Status
    pop = determine_domination(pop)
    
    # Initialize External Population (EP)
    EP = [ind.copy() for ind in pop if not ind.is_dominated]

    gamma = 0.5

    # Main Loop
    while global_config.not_termination(EP if EP else pop):
        for i in range(nPop):
            # Select 2 random neighbors
            k = np.random.choice(T, size=2, replace=False)
            j1 = neighbors[i, k[0]]
            j2 = neighbors[i, k[1]]
            p1, p2 = pop[j1], pop[j2]

            # Reproduction (BLX-alpha crossover)
            alpha = np.random.uniform(-gamma, 1.0 + gamma, size=D)
            y_dec = alpha * p1.dec + (1.0 - alpha) * p2.dec
            y_dec = np.clip(y_dec, VarMin, VarMax)

            # Evaluate child
            y_list = create_individuals(y_dec.reshape(1, -1), global_config.problem, global_config)
            y = y_list[0]

            # Update ideal point
            z = np.minimum(z, y.obj)

            # Update neighbors
            for j in neighbors[i]:
                y_g = decomposed_cost(y, z, lambdas[j])
                # If child's decomposed cost is better, replace the neighbor
                if y_g <= pop[j].g:
                    pop[j] = y.copy()
                    pop[j].g = y_g

        # Update External Population (EP)
        pop = determine_domination(pop)
        ndpop = [ind.copy() for ind in pop if not ind.is_dominated]
        
        EP = EP + ndpop
        EP = determine_domination(EP)
        EP = [ind for ind in EP if not ind.is_dominated]

        # Truncate EP to nArchive size
        if len(EP) > nArchive:
            extra = len(EP) - nArchive
            indices_to_delete = np.random.choice(len(EP), size=extra, replace=False)
            keep_indices = set(range(len(EP))) - set(indices_to_delete)
            EP = [EP[idx] for idx in sorted(keep_indices)]
