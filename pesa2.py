"""PESA-II: Pareto Envelope-based Selection Algorithm II.

Reference:
    D. W. Corne, N. R. Jerram, J. D. Knowles, and M. J. Oates, PESA-II: Reducing
    the selective pressure in evolutionary algorithms, Proceedings of the
    Genetic and Evolutionary Computation Conference (GECCO'2001), 2001, 828-835.
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

def get_grid_boundaries(archive: list[Individual], n_grid: int, inflation_factor: float) -> list[np.ndarray]:
    """Calculate the grid cell boundaries based on archive costs."""
    costs = np.vstack([ind.obj for ind in archive])
    zmin = np.min(costs, axis=0)
    zmax = np.max(costs, axis=0)
    
    dz = zmax - zmin
    alpha = inflation_factor / 2.0
    zmin = zmin - alpha * dz
    zmax = zmax + alpha * dz
    
    n_obj = costs.shape[1]
    boundaries = []
    
    for j in range(n_obj):
        cj = np.linspace(zmin[j], zmax[j], n_grid + 1)
        boundaries.append(np.concatenate([[-np.inf], cj, [np.inf]]))
        
    return boundaries

def get_cell_key(ind: Individual, boundaries: list[np.ndarray]) -> tuple[int, ...]:
    """Return the grid cell coordinate key (tuple of integers) for an individual."""
    sub_indices = []
    for j, cost_j in enumerate(ind.obj):
        idx = np.where(cost_j < boundaries[j])[0][0] - 1
        sub_indices.append(idx)
    return tuple(sub_indices)

def build_grid_cells(archive: list[Individual], boundaries: list[np.ndarray]) -> dict[tuple[int, ...], list[int]]:
    """Group archive indices into occupied grid cells."""
    cells = {}
    for idx, ind in enumerate(archive):
        key = get_cell_key(ind, boundaries)
        cells.setdefault(key, []).append(idx)
    return cells

def select_from_archive(archive: list[Individual], cells: dict[tuple[int, ...], list[int]], beta: float) -> Individual:
    """Select one individual from the archive using Pareto Envelope selection."""
    cell_keys = list(cells.keys())
    cell_sizes = np.array([len(cells[k]) for k in cell_keys])
    
    p = 1.0 / (cell_sizes ** beta)
    p = p / np.sum(p)
    
    selected_cell_idx = roulette_wheel_selection(p)
    selected_cell_key = cell_keys[selected_cell_idx]
    
    members = cells[selected_cell_key]
    selected_member_idx = np.random.choice(members)
    return archive[selected_member_idx]

def pesa2(global_config) -> None:
    """Run the PESA-II algorithm."""
    nPop = global_config.N
    nArchive = 100  # Default archive size
    nGrid = 7
    inflation_factor = 0.1
    beta_deletion = 1.0
    beta_selection = 2.0

    pCrossover = 0.5
    nCrossover = 2 * int(np.round(pCrossover * nPop / 2))
    pMutation = 1.0 - pCrossover
    nMutation = nPop - nCrossover

    gamma_cross = 0.15
    h_mut = 0.3
    VarMin = global_config.lower
    VarMax = global_config.upper
    D = global_config.D

    # Initialize population and archive
    pop = global_config.initialization()
    archive: list[Individual] = []

    # Main Loop
    while global_config.not_termination(archive if archive else pop):
        # Determine domination on pop
        pop = determine_domination(pop)
        ndpop = [ind.copy() for ind in pop if not ind.is_dominated]
        
        # Merge with archive and filter dominated
        archive = archive + ndpop
        archive = determine_domination(archive)
        archive = [ind for ind in archive if not ind.is_dominated]

        # Truncate archive if full
        if len(archive) > nArchive:
            extra = len(archive) - nArchive
            for _ in range(extra):
                boundaries = get_grid_boundaries(archive, nGrid, inflation_factor)
                cells = build_grid_cells(archive, boundaries)
                
                cell_keys = list(cells.keys())
                cell_sizes = np.array([len(cells[k]) for k in cell_keys])
                
                # Deletion probability: cell size ^ beta_deletion
                p_del = cell_sizes ** beta_deletion
                p_del = p_del / np.sum(p_del)
                
                del_cell_idx = roulette_wheel_selection(p_del)
                del_cell_key = cell_keys[del_cell_idx]
                
                members = cells[del_cell_key]
                del_member_idx = np.random.choice(members)
                archive.pop(del_member_idx)

        # Skip crossover/mutation on last iteration
        if global_config.evaluated >= global_config.evaluation:
            continue

        # Crossover & Mutation for next generation
        boundaries = get_grid_boundaries(archive, nGrid, inflation_factor)
        cells = build_grid_cells(archive, boundaries)
        
        offspring_decs = []
        
        # Crossover
        for _ in range(nCrossover // 2):
            p1 = select_from_archive(archive, cells, beta_selection)
            p2 = select_from_archive(archive, cells, beta_selection)
            alpha = np.random.uniform(-gamma_cross, 1.0 + gamma_cross, size=D)
            c1_dec = alpha * p1.dec + (1.0 - alpha) * p2.dec
            c2_dec = alpha * p2.dec + (1.0 - alpha) * p1.dec
            offspring_decs.append(c1_dec)
            offspring_decs.append(c2_dec)

        # Mutation
        sigma_mut = h_mut * (VarMax - VarMin)
        for _ in range(nMutation):
            p = select_from_archive(archive, cells, beta_selection)
            y_dec = p.dec + sigma_mut * np.random.randn(D)
            offspring_decs.append(y_dec)

        # Evaluate offspring
        offspring_decs = np.vstack(offspring_decs)
        pop = create_individuals(offspring_decs, global_config.problem, global_config)
