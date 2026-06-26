"""MOPSO: Multi-Objective Particle Swarm Optimization.

Reference:
    C. A. Coello Coello, G. T. Pulido, and M. S. Lechuga, Handling multiple
    objectives with particle swarm optimization, IEEE Transactions on
    Evolutionary Computation, 2004, 8(3): 256-273.
"""

from __future__ import annotations

import numpy as np
from individual import Individual, create_individuals
from roulette_wheel import roulette_wheel_selection

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

def create_grid(pop: list[Individual], n_grid: int, alpha: float) -> list[dict]:
    """Create grid divisions for leader/deletion selection."""
    costs = np.vstack([ind.obj for ind in pop])
    cmin = np.min(costs, axis=0)
    cmax = np.max(costs, axis=0)
    
    dc = cmax - cmin
    cmin = cmin - alpha * dc
    cmax = cmax + alpha * dc
    
    n_obj = costs.shape[1]
    grid = []
    
    for j in range(n_obj):
        cj = np.linspace(cmin[j], cmax[j], n_grid + 1)
        grid.append({
            "LB": np.concatenate([[-np.inf], cj]),
            "UB": np.concatenate([cj, [np.inf]])
        })
    return grid

def find_grid_index(ind: Individual, grid: list[dict]) -> Individual:
    """Find and assign 1-D grid cell index for an individual."""
    n_obj = len(ind.obj)
    n_grid_plus_2 = len(grid[0]["LB"])
    
    sub_indices = []
    for j in range(n_obj):
        # Find first UB that cost is strictly smaller than
        idx = np.where(ind.obj[j] < grid[j]["UB"])[0][0]
        sub_indices.append(idx)
        
    ind.grid_subindex = sub_indices
    
    grid_index = sub_indices[0]
    for j in range(1, n_obj):
        grid_index = grid_index * n_grid_plus_2 + sub_indices[j]
        
    ind.grid_index = grid_index
    return ind

def select_leader(rep: list[Individual], beta: float) -> Individual:
    """Select a leader particle from the repository based on cell density."""
    gi = np.array([ind.grid_index for ind in rep])
    oc = np.unique(gi)
    
    n_particles = np.array([np.sum(gi == cell) for cell in oc])
    
    p = np.exp(-beta * n_particles)
    p = p / np.sum(p)
    
    sci = roulette_wheel_selection(p)
    sc = oc[sci]
    
    scm = np.where(gi == sc)[0]
    smi = np.random.choice(scm)
    
    return rep[smi]

def delete_one_rep_member(rep: list[Individual], gamma: float) -> list[Individual]:
    """Remove one member from a dense grid cell in the repository."""
    gi = np.array([ind.grid_index for ind in rep])
    oc = np.unique(gi)
    
    n_particles = np.array([np.sum(gi == cell) for cell in oc])
    
    p = np.exp(gamma * n_particles)
    p = p / np.sum(p)
    
    sci = roulette_wheel_selection(p)
    sc = oc[sci]
    
    scm = np.where(gi == sc)[0]
    smi = np.random.choice(scm)
    
    # Remove selected member
    rep.pop(smi)
    return rep

def mopso_mutate(x_dec: np.ndarray, pm: float, VarMin: np.ndarray, VarMax: np.ndarray) -> np.ndarray:
    """Apply MOPSO-specific mutation to a decision vector."""
    D = len(x_dec)
    j = np.random.randint(0, D)
    dx = pm * (VarMax[j] - VarMin[j])
    
    lb = max(x_dec[j] - dx, VarMin[j])
    ub = min(x_dec[j] + dx, VarMax[j])
    
    xnew = x_dec.copy()
    xnew[j] = np.random.uniform(lb, ub)
    return xnew

def mopso(global_config) -> None:
    """Run the MOPSO algorithm."""
    nPop = global_config.N
    nRep = 100  # Default repository size
    w = 0.5
    wdamp = 0.99
    c1 = 1.0
    c2 = 2.0
    nGrid = 7
    alpha = 0.1
    beta = 2.0
    gamma = 2.0
    mu = 0.1

    VarMin = global_config.lower
    VarMax = global_config.upper
    D = global_config.D

    # Initialize particles
    pop = global_config.initialization()
    
    # Initialize velocities and personal bests
    for ind in pop:
        ind.velocity = np.zeros(D)
        ind.best_dec = ind.dec.copy()
        ind.best_obj = ind.obj.copy()

    # Determine Domination and populate repository
    pop = determine_domination(pop)
    rep = [ind.copy() for ind in pop if not ind.is_dominated]

    grid = create_grid(rep, nGrid, alpha)
    for i in range(len(rep)):
        rep[i] = find_grid_index(rep[i], grid)

    # Main Loop
    while global_config.not_termination(rep):
        gen = global_config.gen
        maxgen = global_config.maxgen
        
        # Iteration-based mutation probability
        if maxgen <= 1:
            pm = 0.0
        else:
            pm = (1.0 - (gen - 1) / (maxgen - 1)) ** (1.0 / mu)
            pm = np.clip(pm, 0.0, 1.0)

        for i in range(nPop):
            leader = select_leader(rep, beta)
            
            # Particle velocity update
            r1 = np.random.rand(D)
            r2 = np.random.rand(D)
            pop[i].velocity = (
                w * pop[i].velocity
                + c1 * r1 * (pop[i].best_dec - pop[i].dec)
                + c2 * r2 * (leader.dec - pop[i].dec)
            )
            
            # Position update and clamp
            new_dec = pop[i].dec + pop[i].velocity
            new_dec = np.clip(new_dec, VarMin, VarMax)
            
            # Evaluate updated position
            updated_list = create_individuals(new_dec.reshape(1, -1), global_config.problem, global_config)
            updated = updated_list[0]
            
            # Apply mutation
            if np.random.rand() < pm:
                mutated_dec = mopso_mutate(updated.dec, pm, VarMin, VarMax)
                mutated_list = create_individuals(mutated_dec.reshape(1, -1), global_config.problem, global_config)
                mutated = mutated_list[0]
                
                # Check dominance between mutated position and current updated position
                if dominates(mutated, updated):
                    updated = mutated
                elif dominates(updated, mutated):
                    pass
                else:
                    if np.random.rand() < 0.5:
                        updated = mutated
            
            # Update particle position and velocity
            pop[i].dec = updated.dec
            pop[i].obj = updated.obj
            pop[i].con = updated.con
            
            # Update personal best
            if dominates(pop[i], Individual(dec=pop[i].best_dec, obj=pop[i].best_obj)):
                pop[i].best_dec = pop[i].dec.copy()
                pop[i].best_obj = pop[i].obj.copy()
            elif dominates(Individual(dec=pop[i].best_dec, obj=pop[i].best_obj), pop[i]):
                pass
            else:
                if np.random.rand() < 0.5:
                    pop[i].best_dec = pop[i].dec.copy()
                    pop[i].best_obj = pop[i].obj.copy()

        # Add new non-dominated particles to repository
        pop = determine_domination(pop)
        rep = rep + [ind.copy() for ind in pop if not ind.is_dominated]
        
        # Clean up repository
        rep = determine_domination(rep)
        rep = [ind for ind in rep if not ind.is_dominated]
        
        # Grid update
        grid = create_grid(rep, nGrid, alpha)
        for i in range(len(rep)):
            rep[i] = find_grid_index(rep[i], grid)
            
        # Prune repository if full
        if len(rep) > nRep:
            extra = len(rep) - nRep
            for _ in range(extra):
                rep = delete_one_rep_member(rep, gamma)
                
            # Re-update grid indices after deletion
            grid = create_grid(rep, nGrid, alpha)
            for i in range(len(rep)):
                rep[i] = find_grid_index(rep[i], grid)

        # Damping inertia weight
        w = w * wdamp
