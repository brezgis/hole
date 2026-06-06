"""
Visualization package for HOLE - Homological Observation of Latent Embeddings.

This package provides high-quality visualization classes and functions for analyzing 
point clouds, distance matrices, and persistence diagrams. The main HOLEVisualizer 
class is located in the parent hole package.
"""

# Distance functions are now in core
from ..core.distance_metrics import distance_matrix, euclidean
from .cluster_flow import ClusterFlowAnalyzer, ComponentEvolutionVisualizer
from .layer_flow import (
    LayerEvolutionAnalyzer,
    analyze_layer_flows,
    cluster_to_k,
)

# Persistence plotting functions — use these directly, or use
# HOLEVisualizer.plot_persistence_diagram(...) for the class-style API.
from .persistence_vis import (
    plot_dimensionality_reduction,
    plot_persistence_barcode,
    plot_persistence_diagram,
)
from .heatmap_dendrograms import PersistenceDendrogram
from .scatter_hull import BlobVisualizer

__all__ = [
    # Main visualization classes (no confusing aliases)
    "BlobVisualizer",
    "ComponentEvolutionVisualizer",
    "ClusterFlowAnalyzer",
    "LayerEvolutionAnalyzer",
    "analyze_layer_flows",
    "cluster_to_k",
    "PersistenceDendrogram",
    # Plotting functions
    "plot_persistence_diagram",
    "plot_persistence_barcode",
    "plot_dimensionality_reduction",
    # Utility functions
    "euclidean",
    "distance_matrix",
]
