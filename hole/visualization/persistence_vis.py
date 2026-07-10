"""
Persistence visualizations including diagrams, barcodes, and dimensionality reduction.

This module provides functions for visualizing persistent homology results
and performing dimensionality reduction for data exploration.
"""

import warnings
from typing import List, Optional, Tuple, Union

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_persistence_barcode(
    persistence: List[Tuple],
    pts: int = 10,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
    figsize: tuple = (8, 5),
) -> plt.Axes:
    """
    Plot persistence barcode from persistence data.

    Parameters
    ----------
    persistence : list
        List of persistence pairs from GUDHI
    pts : int, optional
        Number of persistence points to plot
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates new figure and axes.
    title : str, optional
        Title for the plot
    figsize : tuple, optional
        Figure size if creating new figure

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes object containing the plot
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    sns.set_style("whitegrid")

    birth_times = [birth for _, (birth, death) in persistence[:pts]]
    death_times = [death for _, (birth, death) in persistence[:pts]]

    if not birth_times:
        ax.text(
            0.5,
            0.5,
            "No persistence data to plot",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        return ax

    min_birth = min(birth_times)
    max_death = max(d for d in death_times if d < float("inf"))
    delta = (max_death - min_birth) * 0.1
    infinity = max_death + delta
    axis_start = min_birth - delta
    axis_end = max_death + delta * 2

    dimensions = sorted(set(dim for dim, _ in persistence[:pts]))
    palette = sns.color_palette("Set1", n_colors=len(dimensions))
    color_map = {dim: palette[i] for i, dim in enumerate(dimensions)}

    for i, (dim, (birth, death)) in enumerate(persistence[:pts]):
        bar_length = (death - birth) if death != float("inf") else (infinity - birth)
        ax.barh(i, bar_length, left=birth, color=color_map[dim], alpha=0.7)

    legend_patches = [
        mpatches.Patch(color=color_map[dim], label=f"H{dim}") for dim in dimensions
    ]
    ax.legend(handles=legend_patches, loc="best", fontsize=10)

    if title is None:
        title = "Persistence Barcode"
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Filtration Value", fontsize=12)
    ax.set_ylabel("Features", fontsize=12)
    ax.set_yticks([])
    ax.invert_yaxis()

    if birth_times:
        ax.set_xlim((axis_start, axis_end))

    return ax


def plot_persistence_diagram(
    persistence: List[Tuple],
    pts: int = 10,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
    figsize: tuple = (6, 6),
) -> plt.Axes:
    """
    Plot persistence diagram from persistence data.

    Parameters
    ----------
    persistence : list
        List of persistence pairs from GUDHI
    pts : int, optional
        Number of persistence points to plot
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates new figure and axes.
    title : str, optional
        Title for the plot
    figsize : tuple, optional
        Figure size if creating new figure

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes object containing the plot
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    sns.set_style("whitegrid")

    birth_times = [birth for _, (birth, death) in persistence[:pts]]
    death_times = [death for _, (birth, death) in persistence[:pts]]

    if not birth_times:
        ax.text(
            0.5,
            0.5,
            "No persistence data to plot",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        return ax

    min_birth = min(birth_times)
    max_death = max(d for d in death_times if d < float("inf"))

    delta = (max_death - min_birth) * 0.1
    infinity = max_death + 3 * delta
    axis_end = max_death + delta
    axis_start = min_birth - delta

    dimensions = sorted(set(dim for dim, _ in persistence[:pts]))
    palette = sns.color_palette("Set1", n_colors=len(dimensions))
    color_map = {dim: palette[i] for i, dim in enumerate(dimensions)}

    x = [birth for (dim, (birth, death)) in persistence[:pts]]
    y = [
        death if death != float("inf") else infinity
        for (dim, (birth, death)) in persistence[:pts]
    ]
    c = [color_map[dim] for (dim, (birth, death)) in persistence[:pts]]

    sizes = [
        20 + 80 * ((death - birth) / (max(1e-5, max_death - min_birth)))
        for (_, (birth, death)) in persistence[:pts]
    ]
    ax.scatter(x, y, s=sizes, alpha=0.7, c=c, edgecolors="k")

    # Diagonal line
    ax.fill_between(
        [axis_start, axis_end],
        [axis_start, axis_end],
        axis_start,
        color="lightgrey",
        alpha=0.5,
    )

    # Handle infinite death times
    if any(death == float("inf") for (_, (birth, death)) in persistence[:pts]):
        ax.scatter(
            [min_birth],
            [infinity],
            s=150,
            color="black",
            marker="*",
            label="Infinite Death",
        )
        ax.plot(
            [axis_start, axis_end],
            [infinity, infinity],
            linewidth=1.0,
            color="k",
            alpha=0.6,
        )

        yt = np.array(ax.get_yticks())
        yt = yt[yt < axis_end]  # Avoid out-of-bounds y-ticks
        yt = np.append(yt, infinity)
        ytl = ["%.3f" % e for e in yt]
        ytl[-1] = r"$+\infty$"
        ax.set_yticks(yt)
        ax.set_yticklabels(ytl)

    ax.legend(
        handles=[
            mpatches.Patch(color=color_map[dim], label=f"H{dim}") for dim in dimensions
        ],
        title="Dimension",
        loc="lower right",
    )

    ax.set_xlabel("Birth", fontsize=12)
    ax.set_ylabel("Death", fontsize=12)

    if title is None:
        title = "Persistence Diagram"
    ax.set_title(title, fontsize=12)

    ax.set_xlim(axis_start, axis_end)
    ax.set_ylim(min_birth, infinity + delta / 2)

    return ax


def plot_dimensionality_reduction(
    data: Union[np.ndarray, tuple],
    method: str = "pca",
    labels: Optional[np.ndarray] = None,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
    figsize: tuple = (8, 6),
    point_size: float = 50,
    alpha: float = 0.7,
    show_legend: bool = True,
    class_names: Optional[dict] = None,
    metric: str = "euclidean",
    precomputed: Union[bool, str] = "auto",
    **kwargs,
) -> plt.Axes:
    """
    Plot dimensionality reduction visualization.

    Parameters
    ----------
    data : np.ndarray or tuple
        Input data. Can be:
        - 2D array of features for dimensionality reduction
        - Distance matrix (will be converted using MDS)
        - Tuple of (x, y) coordinates for direct plotting
    method : str, optional
        Dimensionality reduction method: 'pca', 'mds', 'tsne', 'umap', or
        'phate' (PHATE recommended for neural-network latent spaces).
    labels : np.ndarray, optional
        Labels for coloring points
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates new figure and axes.
    title : str, optional
        Title for the plot
    figsize : tuple, optional
        Figure size if creating new figure
    point_size : float, optional
        Size of scatter points
    alpha : float, optional
        Alpha transparency for points
    show_legend : bool, optional
        Whether to show legend
    metric : str, optional
        Distance metric used when projecting a feature matrix ('euclidean',
        'cosine', ...). Ignored when ``data`` is a distance matrix.
    precomputed : bool or "auto", optional
        Whether ``data`` is a precomputed distance matrix. "auto" falls back
        to the square/symmetric/zero-diagonal heuristic in
        :func:`hole.projections.project`; pass True explicitly when you know.
    **kwargs : dict
        Additional plotting arguments

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes object containing the plot
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    # Handle different data input types
    if isinstance(data, tuple) and len(data) == 2:
        # Direct coordinates provided
        coords_2d = np.column_stack(data)
    else:
        # Need dimensionality reduction
        coords_2d = _perform_dimensionality_reduction(
            data, method, metric=metric, precomputed=precomputed
        )

    # Generate colors based on labels
    if labels is not None:
        unique_labels = sorted(set(labels))
        n_colors = len(unique_labels)

        # Use tab20 as unified palette for consistency with blob visualizations
        cmap = plt.cm.tab20
        
        label_to_color = {}
        for i, label in enumerate(unique_labels):
            if label == -1:
                # Special gray color for noise
                label_to_color[label] = (0.5, 0.5, 0.5, 1.0)
            else:
                label_to_color[label] = cmap(label % cmap.N)

        point_colors = [label_to_color[label] for label in labels]
    else:
        point_colors = "blue"

    # Create scatter plot
    ax.scatter(
        coords_2d[:, 0],
        coords_2d[:, 1],
        c=point_colors,
        s=point_size,
        alpha=alpha,
        edgecolors="white",
        linewidth=0.5,
        **kwargs,
    )

    # Add legend if labels provided
    if show_legend and labels is not None:
        legend_elements = []
        for label in unique_labels:
            legend_elements.append(
                plt.Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor=label_to_color[label],
                    markeredgecolor="white",
                    markersize=8,
                    label=class_names[label] if class_names and label in class_names else f"Class {label}",
                )
            )
        ax.legend(
            handles=legend_elements,
            title="Labels",
            loc="upper left",
            frameon=True,
            fancybox=True,
            shadow=False,
            framealpha=0.8,
        )

    # Styling - match blob visualization aesthetics
    ax.set_xlabel(f"{method.upper()} Component 1", fontsize=14)
    ax.set_ylabel(f"{method.upper()} Component 2", fontsize=14)

    if title is None:
        title = f"{method.upper()}"
    ax.set_title(title, fontsize=16, fontweight="bold", pad=20)

    # Clean aesthetics - remove ticks and grids
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    ax.set_facecolor("white")

    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.tight_layout()
    return ax


def _perform_dimensionality_reduction(
    data: np.ndarray, method: str = "pca", n_components: int = 2, random_state: int = 42,
    metric: str = "euclidean", precomputed: Union[bool, str] = "auto", **kwargs,
) -> np.ndarray:
    """
    Perform dimensionality reduction on data.

    Delegates to :mod:`hole.projections`, which provides a single ``project``
    entry point over PCA, MDS, t-SNE, UMAP and PHATE, handles both feature
    matrices and precomputed distance matrices, and degrades gracefully when an
    optional backend (``umap-learn`` / ``phate``) is not installed.

    Parameters
    ----------
    data : np.ndarray
        Input data: a feature matrix ``(n_samples, n_features)`` or a precomputed
        symmetric distance matrix ``(n_samples, n_samples)`` (auto-detected).
    method : str
        Method to use: ``'pca'``, ``'mds'``, ``'tsne'``, ``'umap'`` or
        ``'phate'``. PHATE is recommended for neural-network latent spaces.
    n_components : int
        Number of components for the output.
    random_state : int
        Random state for reproducibility.
    metric : str
        Distance metric for non-precomputed data ('euclidean', 'cosine', ...).
    **kwargs
        Extra method-specific arguments forwarded to the underlying estimator
        (e.g. PHATE ``knn``/``decay``/``t``, UMAP ``n_neighbors``/``min_dist``,
        t-SNE ``perplexity``).

    Returns
    -------
    np.ndarray
        Reduced data ``(n_samples, n_components)``.
    """
    from ..projections import project, METHODS

    if method.lower() not in METHODS:
        raise ValueError(
            f"Unknown method: {method}. Use one of {METHODS}."
        )
    # With precomputed="auto", `project` detects a distance matrix (square +
    # symmetric + zero diagonal); callers that *know* (e.g. HOLEVisualizer built
    # from a distance matrix) pass precomputed=True so a slightly asymmetric or
    # nonzero-diagonal matrix is never silently treated as a feature matrix.
    return np.asarray(
        project(
            data,
            method=method,
            n_components=n_components,
            metric=metric,
            precomputed=precomputed,
            random_state=random_state,
            **kwargs,
        )
    )
