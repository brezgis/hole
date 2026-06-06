"""
HOLE: Homological Observation of Latent Embeddings

A library for topological analysis and
visualization of deep learning representations.
"""

import sys

from loguru import logger

# Core functionality
from . import core, utils, visualization


def configure_logging(level: str = "INFO", colorize: bool = True) -> int:
    """Add a default loguru sink for HOLE log output.

    Libraries should not mutate the consumer's logging setup at import time,
    so this is opt-in. Call from application code (e.g. example scripts).

    Returns the loguru handler id (pass to `logger.remove(id)` to detach).
    """
    return logger.add(
        sys.stdout,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        ),
        level=level,
        colorize=colorize,
    )

# Import commonly used functions to top level for convenience
from .core.distance_metrics import (
    chebyshev_distance,
    cosine_distance,
    density_normalized_distance,
    euclidean_distance,
    geodesic_distances,
    mahalanobis_distance,
    manhattan_distance,
)
from .core.mst_processor import MSTProcessor
from .core.persistence import (
    compute_cluster_evolution,
    compute_persistence,
    compute_persistence_statistics,
    extract_death_thresholds,
    select_meaningful_thresholds,
    track_cluster_flows,
)
from .visualization.cluster_flow import (
    ClusterFlowAnalyzer,
    analyze_activation_flows,
)
from .visualization.layer_flow import (
    LayerEvolutionAnalyzer,
    analyze_layer_flows,
    cluster_to_k,
)
from .visualization.heatmap_dendrograms import (
    PersistenceDendrogram,
    analyze_activation_persistence,
)
from .visualization.scatter_hull import BlobVisualizer
from .visualizer import HOLEVisualizer

__version__ = "0.1.0"
__license__ = "GPL-3.0-or-later"
__copyright__ = "Copyright 2024, HOLE Development Team"

__all__ = [
    # Main classes
    "HOLEVisualizer",
    "MSTProcessor",
    "ClusterFlowAnalyzer",
    "LayerEvolutionAnalyzer",
    "BlobVisualizer",
    "PersistenceDendrogram",
    # Distance metrics
    "euclidean_distance",
    "cosine_distance",
    "mahalanobis_distance",
    "manhattan_distance",
    "chebyshev_distance",
    "geodesic_distances",
    "density_normalized_distance",
    # Persistence primitives
    "compute_persistence",
    "extract_death_thresholds",
    "compute_cluster_evolution",
    "select_meaningful_thresholds",
    "track_cluster_flows",
    "compute_persistence_statistics",
    # High-level analysis drivers
    "analyze_activation_flows",
    "analyze_layer_flows",
    "cluster_to_k",
    "analyze_activation_persistence",
    # Logging helper
    "configure_logging",
    # Submodules
    "core",
    "utils",
    "visualization",
]


def get_version():
    """Return the version of HOLE."""
    return __version__


def get_info():
    """Return basic information about HOLE."""
    return {
        "name": "HOLE",
        "version": __version__,
        "description": "Homological Observation of Latent Embeddings",
        "author": "Sudhanva M Athreya, University of Utah",
        "license": __license__,
    }
