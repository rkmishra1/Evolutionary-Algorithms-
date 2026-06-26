"""Plotting utilities for multi-objective optimization results."""

import numpy as np
import matplotlib.pyplot as plt


def draw(data: np.ndarray, title: str = "", ax=None) -> None:
    """Display objective-space data.

    Supports 2-D scatter, 3-D scatter, and parallel-coordinate plots
    for ≥4 objectives.

    Parameters
    ----------
    data : np.ndarray, shape (n, m)
        Objective values to plot.
    title : str, optional
        Figure title.
    ax : matplotlib Axes, optional
        Axes to plot on.  If *None*, a new figure is created.
    """
    n, m = data.shape

    if m == 2:
        if ax is None:
            fig, ax = plt.subplots()
        ax.scatter(
            data[:, 0], data[:, 1],
            s=15, c="grey", edgecolors="dimgrey", linewidths=0.5,
        )
        ax.set_xlabel(r"$f_1$")
        ax.set_ylabel(r"$f_2$")

    elif m == 3:
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")
        ax.scatter(
            data[:, 0], data[:, 1], data[:, 2],
            s=20, c="grey", edgecolors="dimgrey", linewidths=0.5,
        )
        ax.set_xlabel(r"$f_1$")
        ax.set_ylabel(r"$f_2$")
        ax.set_zlabel(r"$f_3$")
        ax.view_init(elev=30, azim=135)

    else:
        # Parallel coordinates for many-objective (M > 3)
        if ax is None:
            fig, ax = plt.subplots()
        label = np.tile(np.concatenate([[0.99], np.arange(2, m), [m + 0.01]]), (n, 1))
        data_plot = data.copy()
        # Flip every other row for visual continuity
        data_plot[1::2, :] = data_plot[1::2, ::-1]
        label[1::2, :] = label[1::2, ::-1]
        for i in range(n):
            ax.plot(label[i], data_plot[i], color="grey", linewidth=0.8)
        ax.set_xlabel("Dimension No.")
        ax.set_ylabel("Value")
        ax.set_xlim(1, m)

    if title:
        ax.set_title(title)
    plt.tight_layout()
    plt.show()
