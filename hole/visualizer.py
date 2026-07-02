"""
Main HOLEVisualizer class - the primary interface for HOLE visualization.

This class automatically computes persistent homology when initialized and provides
access to all visualization functions for homological observation of latent embeddings.
"""

from typing import Any, Callable, Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger

from .core.distance_metrics import distance_matrix

# Import from core and utils
from .core.persistence import (
    compute_cluster_evolution,
    compute_persistence,
    extract_death_thresholds,
    select_meaningful_thresholds,
)

# Import high-quality visualization classes
from .visualization import (
    BlobVisualizer,
    ComponentEvolutionVisualizer,
    PersistenceDendrogram,
)

# Import visualization functions
from .visualization.persistence_vis import (
    plot_dimensionality_reduction,
    plot_persistence_barcode,
    plot_persistence_diagram,
)


class HOLEVisualizer:
    """
    Main class for HOLE visualization - Homological Observation of Latent Embeddings.

    This class takes point clouds and automatically computes persistent homology,
    then provides access to various visualization methods.

    Parameters
    ----------
    point_cloud : np.ndarray, optional
        Input point cloud data of shape (n_samples, n_features)
    distance_matrix : np.ndarray, optional
        Precomputed distance matrix of shape (n_samples, n_samples)
    distance_metric : str or callable, optional
        Distance metric to use if computing from point cloud.
        Can be 'euclidean', 'manhattan', 'cosine', etc. or a callable.
        Default is 'euclidean'.
    max_dimension : int, optional
        Maximum dimension for persistence computation. Default is 1.
    max_edge_length : float, optional
        Maximum edge length for Rips complex. Default is np.inf.

    Attributes
    ----------
    point_cloud : np.ndarray
        The input point cloud data
    distance_matrix : np.ndarray
        The distance matrix used for computations
    persistence : list
        Computed persistence pairs
    n_points : int
        Number of points in the dataset
    """

    def __init__(
        self,
        point_cloud: Optional[np.ndarray] = None,
        distance_matrix_input: Optional[np.ndarray] = None,
        distance_metric: Union[str, Callable] = "euclidean",
        max_dimension: int = 1,
        max_edge_length: float = np.inf,
    ):
        # Validate inputs
        if point_cloud is None and distance_matrix_input is None:
            raise ValueError("Must provide either point_cloud or distance_matrix")

        if point_cloud is not None and distance_matrix_input is not None:
            raise ValueError("Provide only one of point_cloud or distance_matrix")

        # Validate point cloud
        if point_cloud is not None:
            if not isinstance(point_cloud, np.ndarray):
                raise TypeError("point_cloud must be a numpy array")
            if point_cloud.ndim != 2:
                raise ValueError(
                    "point_cloud must be 2D array of shape (n_samples, n_features)"
                )
            if point_cloud.shape[0] < 2:
                raise ValueError("point_cloud must have at least 2 samples")

        # Validate distance matrix
        if distance_matrix_input is not None:
            if not isinstance(distance_matrix_input, np.ndarray):
                raise TypeError("distance_matrix_input must be a numpy array")
            if distance_matrix_input.ndim != 2:
                raise ValueError("distance_matrix_input must be 2D")
            if distance_matrix_input.shape[0] != distance_matrix_input.shape[1]:
                raise ValueError("distance_matrix_input must be square")
            if distance_matrix_input.shape[0] < 2:
                raise ValueError("distance_matrix_input must be at least 2x2")
            if not np.allclose(distance_matrix_input, distance_matrix_input.T):
                logger.warning(
                    "Distance matrix is not symmetric, this may cause issues"
                )
            if np.any(np.diag(distance_matrix_input) != 0):
                logger.warning(
                    "Distance matrix diagonal is not zero, this may cause issues"
                )

        # Validate other parameters
        if not isinstance(max_dimension, int) or max_dimension < 0:
            raise ValueError("max_dimension must be a non-negative integer")
        if max_dimension > 2:
            logger.warning("max_dimension > 2 may be computationally expensive")
        if not (isinstance(max_edge_length, (int, float)) and max_edge_length > 0):
            raise ValueError("max_edge_length must be a positive number")

        # Store inputs
        self.point_cloud = point_cloud
        self.distance_metric = distance_metric
        self.max_dimension = max_dimension
        self.max_edge_length = max_edge_length

        # Compute or store distance matrix
        if distance_matrix_input is not None:
            self.distance_matrix = distance_matrix_input
            self.n_points = distance_matrix_input.shape[0]
        else:
            logger.info("Computing distance matrix...")
            self.distance_matrix = distance_matrix(point_cloud, metric=distance_metric)
            self.n_points = len(point_cloud)

        # Compute persistent homology automatically
        logger.info("Computing persistent homology...")
        self.persistence = compute_persistence(
            self.distance_matrix,
            max_dimension=max_dimension,
            max_edge_length=max_edge_length,
        )
        logger.info(f"Computed persistence with {len(self.persistence)} features")

        # Store additional computed data for visualizations
        self._cluster_evolution = None
        self._death_thresholds = None

    def _compute_cluster_evolution(self, max_thresholds: int = 8):
        """Compute cluster evolution data for flow visualizations."""
        if self._cluster_evolution is not None:
            return self._cluster_evolution

        logger.info("Computing cluster evolution...")

        # Extract death thresholds for 0-dimensional features
        self._death_thresholds = extract_death_thresholds(self.persistence, dimension=0)

        # Select meaningful thresholds
        selected_thresholds = select_meaningful_thresholds(
            self._death_thresholds, max_thresholds
        )

        # Compute cluster evolution
        self._cluster_evolution = compute_cluster_evolution(
            self.distance_matrix, selected_thresholds
        )

        return self._cluster_evolution

    # visualization methods
    def get_blob_visualizer(self, **kwargs):
        """Get a BlobVisualizer instance."""
        return BlobVisualizer(**kwargs)

    def get_cluster_flow_visualizer(self, **kwargs):
        """Get a ComponentEvolutionVisualizer instance."""
        return ComponentEvolutionVisualizer(**kwargs)

    def get_persistence_dendrogram_visualizer(self, **kwargs):
        """Get a PersistenceDendrogram visualizer instance."""
        return PersistenceDendrogram(**kwargs)

    # Persistence visualization methods (legacy support)
    def plot_persistence_diagram(self, ax=None, title=None, pts=10, **kwargs):
        """
        Plot persistence diagram.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, creates new axes.
        title : str, optional
            Title for the plot
        pts : int, optional
            Number of persistence points to plot
        **kwargs : dict
            Additional arguments passed to plot_persistence_diagram

        Returns
        -------
        ax : matplotlib.axes.Axes
            The axes object containing the plot
        """
        return plot_persistence_diagram(
            self.persistence, ax=ax, title=title, pts=pts, **kwargs
        )

    def plot_persistence_barcode(self, ax=None, title=None, pts=10, **kwargs):
        """
        Plot persistence barcode.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, creates new axes.
        title : str, optional
            Title for the plot
        pts : int, optional
            Number of persistence points to plot
        **kwargs : dict
            Additional arguments passed to plot_persistence_barcode

        Returns
        -------
        ax : matplotlib.axes.Axes
            The axes object containing the plot
        """
        return plot_persistence_barcode(
            self.persistence, ax=ax, title=title, pts=pts, **kwargs
        )

    def plot_dimensionality_reduction(
        self, method="pca", ax=None, true_labels=None, title=None, **kwargs
    ):
        """
        Plot dimensionality reduction visualization.

        Parameters
        ----------
        method : str, optional
            Method for dimensionality reduction: 'pca', 'mds', 'tsne', 'umap',
            or 'phate' (PHATE recommended for neural-network latent spaces).
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, creates new axes.
        true_labels : np.ndarray, optional
            True labels for coloring points
        title : str, optional
            Title for the plot
        **kwargs : dict
            Additional arguments passed to plot_dimensionality_reduction

        Returns
        -------
        ax : matplotlib.axes.Axes
            The axes object containing the plot
        """
        data = (
            self.point_cloud if self.point_cloud is not None else self.distance_matrix
        )
        return plot_dimensionality_reduction(
            data, method=method, ax=ax, labels=true_labels, title=title, **kwargs
        )
