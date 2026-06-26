"""MOEAD-ABC: Multi-Objective Artificial Bee Colony Algorithm based on
Decomposition.

Reference:
    A Multiobjective Artificial Bee Colony Algorithm based on Decomposition,
    11th International Conference on Evolutionary Computation Theory and
    Applications, 2019.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist

from uniform_point import uniform_point
from roulette_wheel import roulette_wheel_selection
from individual import (
    Individual,
    create_individuals,
    pop_decs,
    pop_objs,
)


def moeadabc(global_config) -> None:
    """Run the MOEAD-ABC algorithm.

    Parameters
    ----------
    global_config : GlobalConfig
        Experiment controller that holds population size, problem instance,
        evaluation budget, etc.
    """
    agg_type = global_config.agg_type  # aggregation function type (1–4)

    # ------------------------------------------------------------------
    # Generate weight vectors
    # ------------------------------------------------------------------
    W, global_config.N = uniform_point(global_config.N, global_config.M)
    N = global_config.N
    T = 20  # neighbourhood size

    # ------------------------------------------------------------------
    # Detect neighbours of each weight vector
    # ------------------------------------------------------------------
    dist_matrix = cdist(W, W)
    B = np.argsort(dist_matrix, axis=1)[:, :T]  # (N, T) — 0-based indices

    # ------------------------------------------------------------------
    # Generate random initial population
    # ------------------------------------------------------------------
    population = global_config.initialization()  # list[Individual]
    all_objs = pop_objs(population)
    Z = np.min(all_objs, axis=0)  # ideal point

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    while global_config.not_termination(population):
        for i in range(N):
            # Neighbourhood indices (into population)
            P = B[i]  # shape (T,)

            # --- Fitness-proportionate selection of source bee ----------
            all_objs = pop_objs(population)
            neighbour_objs = np.vstack([population[j].obj for j in P])

            # Scaled Tchebycheff value for each neighbour
            Zmax = np.max(all_objs, axis=0)
            denom = Zmax - Z
            denom[denom == 0] = 1e-10  # avoid division by zero
            te_neighbour = np.max(
                np.abs(neighbour_objs - Z) / denom / W[i],
                axis=1,
            )

            mean_cost = np.mean(te_neighbour)

            if mean_cost == 0:
                m = np.random.randint(0, T)
            else:
                F = np.exp(-te_neighbour / mean_cost)
                prob = F / np.sum(F)
                m = roulette_wheel_selection(prob)

            # --- Choose k randomly from neighbourhood, k ≠ m -----------
            order = np.random.permutation(T)
            while order[0] == m:
                order = np.random.permutation(T)
            k = B[i, order[0]]

            # --- ABC offspring generation ------------------------------
            phi = np.random.uniform(-1, 1, size=global_config.D)
            parent_m = population[B[i, m]]
            parent_k = population[k]
            offspring_dec = parent_m.dec + phi * (parent_m.dec - parent_k.dec)

            # --- Polynomial mutation -----------------------------------
            D = global_config.D
            mu = 20
            for ii in range(D):
                if np.random.rand() < 1.0 / D:
                    r = np.random.rand()
                    if r < 0.5:
                        delta = (2 * r) ** (1 / (1 + mu)) - 1
                    else:
                        delta = 1 - (2 * (1 - r)) ** (1 / (mu + 1))
                    offspring_dec[ii] += delta * (
                        global_config.upper[ii] - global_config.lower[ii]
                    )

            # --- Evaluate offspring ------------------------------------
            offspring_list = create_individuals(
                offspring_dec.reshape(1, -1),
                global_config.problem,
                global_config,
            )
            offspring = offspring_list[0]

            # --- Update ideal point ------------------------------------
            Z = np.minimum(Z, offspring.obj)

            # --- Update neighbours via aggregation function ------------
            neighbour_objs = np.vstack([population[j].obj for j in P])

            if agg_type == 1:
                # PBI approach
                normW = np.sqrt(np.sum(W[P] ** 2, axis=1))
                diff_old = neighbour_objs - Z
                diff_new = offspring.obj - Z
                normP = np.sqrt(np.sum(diff_old ** 2, axis=1))
                normO = np.sqrt(np.sum(diff_new ** 2))
                cosineP = np.sum(diff_old * W[P], axis=1) / (normW * normP + 1e-30)
                cosineO = np.sum(
                    np.tile(diff_new, (T, 1)) * W[P], axis=1
                ) / (normW * normO + 1e-30)
                g_old = normP * cosineP + 5 * normP * np.sqrt(
                    np.maximum(1 - cosineP ** 2, 0)
                )
                g_new = normO * cosineO + 5 * normO * np.sqrt(
                    np.maximum(1 - cosineO ** 2, 0)
                )

            elif agg_type == 2:
                # Tchebycheff approach
                g_old = np.max(
                    np.abs(neighbour_objs - Z) * W[P], axis=1
                )
                g_new = np.max(
                    np.tile(np.abs(offspring.obj - Z), (T, 1)) * W[P],
                    axis=1,
                )

            elif agg_type == 3:
                # Tchebycheff with normalisation
                all_objs_with_off = np.vstack([all_objs, offspring.obj])
                Zmax = np.max(all_objs_with_off, axis=0)
                denom = Zmax - Z
                denom[denom == 0] = 1e-10
                g_old = np.max(
                    np.abs(neighbour_objs - Z) / denom / W[P], axis=1
                )
                g_new = np.max(
                    np.tile(np.abs(offspring.obj - Z) / denom, (T, 1)) / W[P],
                    axis=1,
                )

            elif agg_type == 4:
                # Modified Tchebycheff approach
                g_old = np.max(
                    np.abs(neighbour_objs - Z) / W[P], axis=1
                )
                g_new = np.max(
                    np.tile(np.abs(offspring.obj - Z), (T, 1)) / W[P],
                    axis=1,
                )

            # Replace improved neighbours
            improved = np.where(g_old >= g_new)[0]
            for idx in improved:
                population[P[idx]] = offspring.copy()
