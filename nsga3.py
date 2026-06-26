"""NSGA-III: Non-dominated Sorting Genetic Algorithm III.

Reference:
    K. Deb and H. Jain, An evolutionary many-objective optimization algorithm
    using reference-point based non-dominated sorting approach, part I: Solving
    problems with box constraints, IEEE Transactions on Evolutionary
    Computation, 2014, 18(4): 577-601.
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

def get_fixed_row_sum_integer_matrix(M: int, row_sum: int) -> np.ndarray:
    """Helper to generate combinations for reference points."""
    if M == 1:
        return np.array([[row_sum]])
    A = []
    for i in range(row_sum + 1):
        B = get_fixed_row_sum_integer_matrix(M - 1, row_sum - i)
        prefix = np.full((B.shape[0], 1), i)
        A.append(np.column_stack([prefix, B]))
    return np.vstack(A)

def generate_reference_points(M: int, p: int) -> np.ndarray:
    """Generate reference points on a hyperplane."""
    return get_fixed_row_sum_integer_matrix(M, p) / p

def update_ideal_point(pop: list[Individual], prev_zmin: np.ndarray | None) -> np.ndarray:
    """Track and update the ideal point."""
    costs = np.vstack([ind.obj for ind in pop])
    current_min = np.min(costs, axis=0)
    if prev_zmin is None:
        return current_min
    return np.minimum(prev_zmin, current_min)

def perform_scalarizing(z: np.ndarray, params: dict) -> dict:
    """Find the extreme points in translated cost space."""
    n_obj = z.shape[0]
    n_pop = z.shape[1]
    
    if params.get("zmax") is not None:
        zmax = params["zmax"]
        smin = params["smin"]
    else:
        zmax = np.zeros((n_obj, n_obj))
        smin = np.full(n_obj, np.inf)
        
    for j in range(n_obj):
        w = np.full(n_obj, 1e-10)
        w[j] = 1.0
        
        s = np.max(z / w.reshape(-1, 1), axis=0)
        smin_j = np.min(s)
        ind = np.argmin(s)
        
        if smin_j < smin[j]:
            zmax[:, j] = z[:, ind]
            smin[j] = smin_j
            
    params["zmax"] = zmax
    params["smin"] = smin
    return params

def find_hyperplane_intercepts(zmax: np.ndarray, fp: np.ndarray) -> np.ndarray:
    """Find intercepts of the hyperplane constructed by extreme points."""
    try:
        w = np.linalg.solve(zmax.T, np.ones(zmax.shape[0]))
        a = 1.0 / w
        if np.any(a <= 1e-4) or np.any(np.isnan(a)):
            raise ValueError
    except (np.linalg.LinAlgError, ValueError):
        a = np.max(fp, axis=1)
        a[a <= 1e-4] = 1e-4
    return a

def normalize_population(pop: list[Individual], params: dict) -> tuple[list[Individual], dict]:
    """Perform normalization on the population."""
    params["zmin"] = update_ideal_point(pop, params.get("zmin"))
    
    # Translate costs
    fp = np.vstack([ind.obj for ind in pop]).T - params["zmin"].reshape(-1, 1)
    
    params = perform_scalarizing(fp, params)
    
    a = find_hyperplane_intercepts(params["zmax"], fp)
    
    for i, ind in enumerate(pop):
        ind.normalized_cost = fp[:, i] / a
        
    return pop, params

def associate_to_reference_point(pop: list[Individual], params: dict) -> tuple[list[Individual], np.ndarray, np.ndarray]:
    """Associate each individual with a reference point."""
    Zr = params["Zr"]
    nZr = params["nZr"]
    n = len(pop)
    
    rho = np.zeros(nZr)
    d = np.zeros((n, nZr))
    
    for i in range(n):
        z = pop[i].normalized_cost
        for j in range(nZr):
            w = Zr[j] / np.linalg.norm(Zr[j])
            d[i, j] = np.linalg.norm(z - np.dot(w, z) * w)
            
        jmin = np.argmin(d[i, :])
        pop[i].associated_ref = jmin
        pop[i].distance_to_associated_ref = d[i, jmin]
        rho[jmin] += 1
        
    return pop, d, rho

def sort_and_select_population(pop: list[Individual], params: dict) -> tuple[list[Individual], list[list[int]], dict]:
    """Perform NSGA-III reference point-based selection."""
    pop, params = normalize_population(pop, params)
    pop, F = non_dominated_sorting(pop)
    
    nPop = params["nPop"]
    if len(pop) == nPop:
        return pop, F, params
        
    pop, d, rho = associate_to_reference_point(pop, params)
    
    newpop = []
    last_front_idx = -1
    for l in range(len(F)):
        if len(newpop) + len(F[l]) > nPop:
            last_front_idx = l
            break
        for idx in F[l]:
            newpop.append(pop[idx])
            
    last_front = list(F[last_front_idx])
    
    # Niching loop
    while True:
        j = np.argmin(rho)
        
        associated_from_last_front = []
        for idx in last_front:
            if pop[idx].associated_ref == j:
                associated_from_last_front.append(idx)
                
        if not associated_from_last_front:
            rho[j] = float("inf")
            continue
            
        if rho[j] == 0:
            ddj = d[associated_from_last_front, j]
            new_member_ind = np.argmin(ddj)
        else:
            new_member_ind = np.random.randint(len(associated_from_last_front))
            
        member_to_add = associated_from_last_front[new_member_ind]
        last_front.remove(member_to_add)
        newpop.append(pop[member_to_add])
        
        rho[j] += 1
        
        if len(newpop) >= nPop:
            break
            
    newpop, F = non_dominated_sorting(newpop)
    return newpop, F, params

def nsga3(global_config) -> None:
    """Run the NSGA-III algorithm."""
    nPop = global_config.N
    D = global_config.D
    M = global_config.M

    nDivision = 10
    Zr = generate_reference_points(M, nDivision)
    nZr = Zr.shape[0]

    pCrossover = 0.5
    nCrossover = 2 * int(np.round(pCrossover * nPop / 2))
    pMutation = 0.5
    nMutation = int(np.round(pMutation * nPop))
    mu = 0.02
    sigma = 0.1 * (global_config.upper - global_config.lower)

    params = {
        "nPop": nPop,
        "Zr": Zr,
        "nZr": nZr,
        "zmin": None,
        "zmax": None,
        "smin": None,
    }

    # Initialize population
    pop = global_config.initialization()
    pop, F, params = sort_and_select_population(pop, params)

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
        
        # Sort and select
        pop, F, params = sort_and_select_population(pop, params)
