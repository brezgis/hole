"""
Blob Visualization for Cluster Separation Analysis

This module provides visualization of cluster separation at specific distance thresholds
from persistent homology cluster evolution. Shows t-SNE, PCA, and MDS plots with
nodes colored by true labels and convex hulls around cluster assignments.
"""

import os
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from matplotlib.patches import Polygon
from scipy.spatial import ConvexHull

from ..config import DEFAULT_RANDOM_STATE


def get_label_color(label: int, n_classes: int = 10, shade: int = 0):
    """Get a consistent color for a class label. Used across all visualizations.

    Uses tab10 for ≤10 classes, tab20 for ≤20, evenly spaced on gist_ncar beyond.
    Label -1 is always gray (noise).

    Args:
        label: Class label index.
        n_classes: Total number of classes.
        shade: 0 = base color, 1+ = progressively darker variants for
               distinguishing multiple clusters with the same dominant class.
    """
    if label == -1:
        return (0.5, 0.5, 0.5, 1.0)
    if n_classes <= 10:
        rgba = plt.cm.tab10(label % 10)
    elif n_classes <= 20:
        rgba = plt.cm.tab20(label % 20)
    else:
        rgba = plt.cm.gist_ncar(label / n_classes)
    if shade > 0:
        # Darken by 15% per shade level, floor at 30% brightness
        factor = max(0.3, 1.0 - 0.15 * shade)
        rgba = (rgba[0] * factor, rgba[1] * factor, rgba[2] * factor, rgba[3])
    return rgba


class BlobVisualizer:
    """
    Visualizes cluster separation at specific persistent homology thresholds.
    Creates t-SNE, PCA, and MDS plots with nodes colored by true labels
    and convex hulls around cluster assignments.
    """

    def __init__(
        self,
        figsize: Tuple[int, int] = (12, 10),
        dpi: int = 300,
        alpha_hull: float = 0.3,
        class_names: Optional[Dict[int, str]] = None,
        shared_cluster_colors: Optional[List] = None,
        outlier_percentage: float = 0.1,
        show_contours: bool = True,
    ):
        """
        Initialize the blob visualizer.

        Args:
            figsize: Figure size for plots
            dpi: DPI for saved plots
            alpha_hull: Alpha transparency for convex hulls
            class_names: Optional dictionary mapping class indices to names
            shared_cluster_colors: Optional shared color list for consistency across visualizations
            outlier_percentage: Percentage of points to consider as outliers (0.0-1.0)
            show_contours: Whether to show contour lines inside blobs
        """
        self.figsize = figsize
        self.dpi = dpi
        self.alpha_hull = alpha_hull
        self.outlier_percentage = outlier_percentage
        self.show_contours = show_contours

        # Class names - use provided or generic defaults
        self.class_names = class_names or {i: f"Class_{i}" for i in range(10)}

        # shared_cluster_colors kept for API compat but ignored
        self.cluster_colors = shared_cluster_colors

    def _get_class_color(self, class_id: int, n_classes: int):
        """Get consistent color for a class across all visualizations."""
        return get_label_color(class_id, n_classes)

    def _compute_convex_hull(
        self, points: np.ndarray, padding_factor: float = 0.15
    ) -> Optional[np.ndarray]:
        """
        Compute expanded convex hull for a set of points with padding for better visual coverage.

        Args:
            points: 2D points array
            padding_factor: Factor to expand the hull outward (0.15 = 15% expansion)

        Returns:
            Expanded hull vertices or None if not enough points
        """
        if len(points) < 3:
            return None

        try:
            hull = ConvexHull(points)
            hull_vertices = points[hull.vertices]

            # Calculate centroid of the cluster
            centroid = np.mean(points, axis=0)

            # Expand each vertex outward from the centroid
            expanded_vertices = []
            for vertex in hull_vertices:
                # Vector from centroid to vertex
                direction = vertex - centroid
                # Expand outward by padding_factor
                expanded_vertex = vertex + direction * padding_factor
                expanded_vertices.append(expanded_vertex)

            return np.array(expanded_vertices)

        except Exception as e:
            logger.warning(f"Could not compute convex hull: {e}")
            return None

    def _compute_smooth_hull(
        self,
        points: np.ndarray,
        padding_factor: float = 0.15,
        smoothing_factor: float = 0.3,
    ) -> Optional[np.ndarray]:
        """
        Compute a smooth, rounded hull around points using interpolation.

        Args:
            points: 2D points array
            padding_factor: Factor to expand the hull outward
            smoothing_factor: How much to round the corners (0.0 = sharp, 1.0 = very round)

        Returns:
            Smooth hull vertices or None if not enough points
        """
        if len(points) < 3:
            return None

        try:
            # First get the convex hull
            hull = ConvexHull(points)
            hull_vertices = points[hull.vertices]
            centroid = np.mean(points, axis=0)

            # Expand vertices outward
            expanded_vertices = []
            for vertex in hull_vertices:
                direction = vertex - centroid
                expanded_vertex = vertex + direction * padding_factor
                expanded_vertices.append(expanded_vertex)

            expanded_vertices = np.array(expanded_vertices)

            # Create smooth boundary by interpolating between vertices with curves
            smooth_points = []
            n_vertices = len(expanded_vertices)
            n_interpolation = 20  # Points between each pair of vertices

            for i in range(n_vertices):
                current = expanded_vertices[i]
                next_vertex = expanded_vertices[(i + 1) % n_vertices]
                # prev_vertex = expanded_vertices[(i - 1) % n_vertices]

                # Add the current vertex
                smooth_points.append(current)

                # Create curved transition to next vertex
                # Use bezier-like curve with control points
                control_distance = (
                    np.linalg.norm(next_vertex - current) * smoothing_factor
                )

                # Control point: move from current vertex towards next, but also slightly outward
                direction_to_next = (next_vertex - current) / np.linalg.norm(
                    next_vertex - current
                )
                direction_outward = (current - centroid) / np.linalg.norm(
                    current - centroid
                )
                control_point = (
                    current
                    + direction_to_next * control_distance
                    + direction_outward * control_distance * 0.3
                )

                # Interpolate curve from current to next vertex
                for j in range(1, n_interpolation):
                    t = j / n_interpolation
                    # Quadratic Bezier curve: (1-t)²P₀ + 2(1-t)tP₁ + t²P₂
                    curve_point = (
                        (1 - t) ** 2 * current
                        + 2 * (1 - t) * t * control_point
                        + t**2 * next_vertex
                    )
                    smooth_points.append(curve_point)

            return np.array(smooth_points)

        except Exception as e:
            logger.warning(f"    Warning: Could not compute smooth hull: {e}")
            # Fallback to regular convex hull
            return self._compute_convex_hull(points, padding_factor)

    def _create_blob_boundary(
        self, points: np.ndarray, method: str = "smooth", padding_factor: float = 0.15
    ) -> Optional[np.ndarray]:
        """
        Create blob boundary around points.

        Args:
            points: 2D points array
            method: 'convex' for convex hull, 'circle' for circular boundary, 'smooth' for rounded convex hull
            padding_factor: Expansion factor for better coverage

        Returns:
            Boundary points or None
        """
        if len(points) < 2:
            return None

        if method == "convex":
            return self._compute_convex_hull(points, padding_factor=padding_factor)
        elif method == "circle":
            # Create circular boundary around points
            center = np.mean(points, axis=0)
            distances = np.sqrt(np.sum((points - center) ** 2, axis=1))
            radius = np.max(distances) * (1.2 + padding_factor)  # Add padding

            # Generate circle points
            theta = np.linspace(0, 2 * np.pi, 50)
            circle_points = np.column_stack(
                [center[0] + radius * np.cos(theta), center[1] + radius * np.sin(theta)]
            )
            return circle_points
        elif method == "smooth":
            return self._compute_smooth_hull(points, padding_factor=padding_factor)

        return None

    def _detect_outliers(
        self, 
        points: np.ndarray, 
        cluster_labels: np.ndarray, 
        true_labels: np.ndarray,
        percentage: float = 0.1
    ) -> np.ndarray:
        """
        Detect outlier classes based on percentage threshold within persistent homology clusters.
        Classes that make up less than the specified percentage of points within each cluster
        are considered outlier classes and will be plotted as scatter points.
        
        Args:
            points: 2D points array
            cluster_labels: Persistent homology cluster assignments
            true_labels: True class labels
            percentage: Percentage threshold (0.0-1.0). Classes below this percentage 
                       within each cluster are considered outliers.
            
        Returns:
            Boolean array indicating outlier class points
        """
        if percentage <= 0:
            return np.zeros(len(points), dtype=bool)
            
        outlier_mask = np.zeros(len(points), dtype=bool)
        unique_clusters = np.unique(cluster_labels)
        
        for cluster_id in unique_clusters:
            if cluster_id == -1:  # Skip noise points
                continue
                
            cluster_mask = cluster_labels == cluster_id
            cluster_true_labels = true_labels[cluster_mask]
            cluster_size = len(cluster_true_labels)
            
            if cluster_size < 2:
                continue
                
            # Get class counts within this cluster
            unique_classes, class_counts = np.unique(cluster_true_labels, return_counts=True)
            
            # Find classes that are below the percentage threshold
            for class_id, count in zip(unique_classes, class_counts):
                class_percentage = count / cluster_size
                
                if class_percentage < percentage:
                    # This class is an outlier class in this cluster
                    cluster_indices = np.where(cluster_mask)[0]
                    class_mask_in_cluster = cluster_true_labels == class_id
                    outlier_class_indices = cluster_indices[class_mask_in_cluster]
                    outlier_mask[outlier_class_indices] = True
            
        return outlier_mask

    def _create_contour_grid_by_class(
        self, 
        points: np.ndarray, 
        labels: np.ndarray,
        hull_points: np.ndarray, 
        resolution: int = 50
    ) -> Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Create grids for contour plotting within the hull boundary, separated by class.
        
        Args:
            points: Cluster points
            labels: True class labels for the points
            hull_points: Hull boundary points
            resolution: Grid resolution
            
        Returns:
            Dictionary mapping class_id -> (X, Y, Z) grid coordinates and density values
        """
        from matplotlib.path import Path
        from scipy.stats import gaussian_kde
        
        # Create bounding box around hull
        x_min, x_max = hull_points[:, 0].min(), hull_points[:, 0].max()
        y_min, y_max = hull_points[:, 1].min(), hull_points[:, 1].max()
        
        # Add some padding
        padding = 0.1
        x_range = x_max - x_min
        y_range = y_max - y_min
        x_min -= x_range * padding
        x_max += x_range * padding
        y_min -= y_range * padding
        y_max += y_range * padding
        
        # Create grid
        x = np.linspace(x_min, x_max, resolution)
        y = np.linspace(y_min, y_max, resolution)
        X, Y = np.meshgrid(x, y)
        
        # Create path from hull for masking
        hull_path = Path(hull_points)
        
        # Create contour grids for each class
        class_grids = {}
        unique_classes = np.unique(labels)
        
        for class_id in unique_classes:
            class_mask = labels == class_id
            class_points = points[class_mask]
            
            if len(class_points) >= 2:
                try:
                    kde = gaussian_kde(class_points.T)
                    positions = np.vstack([X.ravel(), Y.ravel()])
                    Z = kde(positions).reshape(X.shape)
                except:
                    # Fallback: simple distance-based density
                    Z = np.zeros_like(X)
                    for point in class_points:
                        distances = np.sqrt((X - point[0])**2 + (Y - point[1])**2)
                        Z += np.exp(-distances**2 / (2 * 0.5**2))  # Gaussian fallback
                
                # Mask points outside hull
                grid_points = np.column_stack([X.ravel(), Y.ravel()])
                inside_hull = hull_path.contains_points(grid_points)
                Z_masked = Z.copy()
                Z_masked[~inside_hull.reshape(X.shape)] = np.nan
                
                # Only store if we have valid density data
                if not np.all(np.isnan(Z_masked)):
                    class_grids[class_id] = (X, Y, Z_masked)
        
        return class_grids

    def _plot_dimensionality_reduction(
        self,
        activations: np.ndarray,
        y_true: np.ndarray,
        cluster_labels: np.ndarray,
        method: str,
        threshold: float,
        title_prefix: str,
        save_path: str,
    ) -> plt.Figure:
        """
        Create a dimensionality reduction plot with cluster blobs.

        Args:
            activations: Input activation data
            y_true: True class labels
            cluster_labels: Cluster assignments at threshold
            method: 'pca', 'mds', 'tsne', 'umap', or 'phate'
            threshold: Distance threshold value
            title_prefix: Prefix for plot title
            save_path: Path to save the plot

        Returns:
            matplotlib Figure object
        """
        logger.info(f"      Creating {method.upper()} plot...")

        # Apply dimensionality reduction via the shared projections module, which
        # supports pca / mds / tsne / umap / phate (PHATE recommended for NN
        # latent spaces) and degrades gracefully when an optional backend is
        # missing.
        from ..projections import project, METHODS

        if method.lower() not in METHODS:
            raise ValueError(f"Unknown method: {method}. Use one of {METHODS}.")

        reduced = project(
            activations,
            method=method,
            n_components=2,
            random_state=DEFAULT_RANDOM_STATE,
        )
        reduced_data = np.asarray(reduced)
        method_name = method.upper()

        # PCA gets informative variance-labelled axes; everything else gets
        # generic component labels.
        if method.lower() == "pca" and getattr(reduced, "reducer", None) is not None:
            explained_var = reduced.reducer.explained_variance_ratio_
            xlabel = f"PC1 ({explained_var[0]:.2%} variance)"
            ylabel = f"PC2 ({explained_var[1]:.2%} variance)"
        else:
            xlabel = f"{method_name} 1"
            ylabel = f"{method_name} 2"

        # Create plot
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # First, draw cluster boundaries and contours
        _hull_shade_counter = {}  # tracks shade per dominant label
        unique_clusters = np.unique(cluster_labels)
        for cluster_id in unique_clusters:
            if cluster_id == -1:  # Skip noise points for hull drawing
                continue

            cluster_mask = cluster_labels == cluster_id
            cluster_points = reduced_data[cluster_mask]

            if len(cluster_points) >= 3:  # Need at least 3 points for convex hull
                # Color hull by dominant true label in cluster
                cluster_true = y_true[cluster_mask]
                dominant = np.bincount(cluster_true.astype(int)).argmax()
                n_true_labels_local = len(set(y_true))
                shade = _hull_shade_counter.get(dominant, 0)
                _hull_shade_counter[dominant] = shade + 1
                cluster_color = get_label_color(dominant, n_true_labels_local, shade=shade)

                # Create smooth hull
                hull_points = self._create_blob_boundary(
                    cluster_points, method="smooth"
                )
                if hull_points is not None:
                    # Draw filled hull with better border styling
                    hull_polygon = Polygon(
                        hull_points,
                        alpha=self.alpha_hull,
                        facecolor=cluster_color,
                        edgecolor="black",
                        linewidth=2.5,
                        linestyle="-",
                        zorder=1,
                    )
                    ax.add_patch(hull_polygon)

                    # Add contour lines inside the blob if enabled, colored by class
                    if self.show_contours and len(cluster_points) >= 5:
                        try:
                            # Get true labels for points in this cluster
                            cluster_true_labels = y_true[cluster_mask]
                            
                            # Create class-based contour grids
                            class_grids = self._create_contour_grid_by_class(
                                cluster_points, cluster_true_labels, hull_points
                            )
                            
                            # Plot contours for each class with class-specific colors
                            for class_id, (X, Y, Z) in class_grids.items():
                                # Get unified class color
                                class_color = self._get_class_color(class_id, n_true_labels)
                                
                                # Plot contours for this class
                                contours = ax.contour(X, Y, Z, levels=4, colors=[class_color], 
                                                    alpha=0.9, linewidths=2.0, zorder=2)
                                
                        except Exception as e:
                            logger.warning(f"        Warning: Could not create contours for cluster {cluster_id}: {e}")

                    # Add cluster label
                    center = np.mean(cluster_points, axis=0)
                    ax.text(
                        center[0],
                        center[1],
                        f"C{cluster_id}",
                        fontsize=11,
                        fontweight="bold",
                        color="black",
                        ha="center",
                        va="center",
                        zorder=3,
                        bbox=dict(
                            boxstyle="round,pad=0.4",
                            facecolor="white",
                            alpha=0.9,
                            edgecolor="black",
                            linewidth=1,
                        ),
                    )

        # Use unified color palette for true labels
        unique_true_labels = sorted(set(y_true))
        n_true_labels = len(unique_true_labels)

        # Draw all points as scatter ONLY if contours are disabled
        if not self.show_contours:
            blob_colors = []
            for label in y_true:
                blob_colors.append(self._get_class_color(label, n_true_labels))
            
            ax.scatter(
                reduced_data[:, 0],
                reduced_data[:, 1],
                c=blob_colors,
                s=50,
                alpha=0.7,
                edgecolors="white",
                linewidth=0.5,
                zorder=3,
                marker='o'
            )

        # Customize plot
        ax.set_xlabel(xlabel, fontsize=14)
        ax.set_ylabel(ylabel, fontsize=14)
        # ax.set_title(f"{title_prefix} - {method_name} - Death Threshold {threshold:.4f}",
        #             fontsize=16, fontweight='bold', pad=20)
        ax.set_title(
            f"{method_name} - Death Threshold {threshold:.4f}",
            fontsize=16,
            fontweight="bold",
            pad=20,
        )

        # Clean aesthetics - remove ticks and grids
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        ax.set_facecolor("white")

        for spine in ax.spines.values():
            spine.set_visible(False)

        # Add legend for cluster hulls
        cluster_info = []
        for cluster_id in unique_clusters:
            if cluster_id != -1:
                count = np.sum(cluster_labels == cluster_id)
                cluster_info.append(f"Cluster {cluster_id}: {count} points")

        if cluster_info:
            legend_text = "\n".join(cluster_info[:10])  # Show first 10 clusters
            ax.text(
                0.02,
                0.98,
                legend_text,
                transform=ax.transAxes,
                fontsize=9,
                verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8),
            )

        plt.tight_layout()

        # Save plot
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight", facecolor="white")
        logger.info(f"        Saved: {save_path}")

        return fig

    def analyze_blob_separation(
        self,
        activations: np.ndarray,
        y_true: np.ndarray,
        cluster_evolution: Dict,
        output_dir: str,
        model_name: str,
        condition_name: str,
        layer_name: str,
        distance_metric: str = "Euclidean",
    ) -> Dict:
        """
        Analyze cluster separation at middle and 4th stage thresholds.

        Args:
            activations: Activation data
            y_true: True class labels
            cluster_evolution: Cluster evolution data from ClusterFlowAnalyzer
            output_dir: Output directory for plots
            model_name: Model name
            condition_name: Condition name (e.g., 'gaussian', 'inference')
            layer_name: Layer name
            distance_metric: Distance metric used ('Euclidean', 'Mahalanobis', etc.)

        Returns:
            Dictionary with analysis results
        """
        logger.info(f"    Analyzing blob separation for {layer_name} - {distance_metric}")

        # Extract thresholds and labels from cluster evolution
        components_ = cluster_evolution["components_"]
        labels_ = cluster_evolution["labels_"]

        if distance_metric not in components_ or distance_metric not in labels_:
            logger.warning(
                f"      Warning: {distance_metric} not found in cluster evolution data"
            )
            return {}

        # Get all thresholds (sorted)
        thresholds = sorted([float(t) for t in components_[distance_metric].keys()])

        if len(thresholds) < 4:
            logger.warning(f"      Warning: Need at least 4 thresholds, got {len(thresholds)}")
            return {}

        # Select middle (3rd) and 4th stage thresholds
        # In 5-stage evolution: Stage 1=True, Stage 2=thresholds[0], Stage 3=thresholds[1],
        # Stage 4=thresholds[2], Stage 5=thresholds[3]
        middle_threshold = thresholds[1]  # 3rd stage (index 1)
        fourth_threshold = thresholds[2]  # 4th stage (index 2)

        logger.info(
            f"      Selected thresholds: Middle={middle_threshold:.4f}, Fourth={fourth_threshold:.4f}"
        )

        # Get cluster labels for both thresholds
        middle_labels = labels_[distance_metric][str(middle_threshold)]
        fourth_labels = labels_[distance_metric][str(fourth_threshold)]

        # Clean names for file paths
        clean_layer_name = layer_name.replace("/", "_").replace(".", "_")

        # Create title prefix
        # if condition_name.lower() in ['inference', 'clean']:
        #     if model_name.lower() == 'original':
        #         title_prefix = f"{distance_metric} - {layer_name}"
        #     else:
        #         title_prefix = f" {model_name.replace('_', ' ').title()} - {distance_metric} - {layer_name}"
        # else:
        #     if model_name.lower() == 'original':
        #         title_prefix = f"{condition_name.replace('_', ' ').title()} - {distance_metric} - {layer_name}"
        #     else:
        #         title_prefix = f" {model_name.replace('_', ' ').title()} - {condition_name.replace('_', ' ').title()} - {distance_metric} - {layer_name}"
        title_prefix = ""
        results = {}

        # Analyze both thresholds
        for stage_name, threshold, cluster_labels in [
            ("middle", middle_threshold, middle_labels),
            ("fourth", fourth_threshold, fourth_labels),
        ]:
            logger.info(f"      Processing {stage_name} stage (threshold={threshold:.4f})...")

            stage_results = {}

            # Create plots for each dimensionality reduction method
            for method in ["pca"]:
                save_path = os.path.join(
                    output_dir,
                    f"{model_name}_{condition_name}_{clean_layer_name}_{distance_metric}_{threshold:.4f}_{method}.png",
                )

                try:
                    fig = self._plot_dimensionality_reduction(
                        activations,
                        y_true,
                        cluster_labels,
                        method,
                        threshold,
                        title_prefix,
                        save_path,
                    )
                    stage_results[method] = {
                        "figure": fig,
                        "save_path": save_path,
                        "threshold": threshold,
                    }
                    plt.close(fig)  # Free memory

                except Exception as e:
                    logger.error(f"        Error creating {method} plot: {e}")
                    stage_results[method] = None

            # Calculate cluster statistics
            unique_clusters = np.unique(cluster_labels)
            n_clusters = len(unique_clusters)
            cluster_sizes = [np.sum(cluster_labels == c) for c in unique_clusters]

            stage_results["statistics"] = {
                "n_clusters": n_clusters,
                "cluster_sizes": cluster_sizes,
                "unique_clusters": unique_clusters.tolist(),
                "threshold": threshold,
            }

            results[stage_name] = stage_results

            logger.info(
                f"        {stage_name.title()} stage: {n_clusters} clusters, sizes: {cluster_sizes}"
            )

        return results

    def plot_pca_with_cluster_hulls(
        self,
        points: np.ndarray,
        true_labels: np.ndarray,
        threshold: float,
        save_path: Optional[str] = None,
        title: Optional[str] = None,
        metric: str = "euclidean",
        class_names: Optional[dict] = None,
        method: str = "pca",
    ) -> plt.Figure:
        """
        Create a 2D projection plot with points colored by true labels and convex
        hulls for clusters at threshold.

        Clustering always happens in the original feature space; ``method`` only
        controls the 2D canvas the hulls are drawn on.

        Args:
            points: Input data points (n_samples, n_features)
            true_labels: True class labels for coloring points
            threshold: Distance threshold for clustering
            save_path: Optional path to save the plot
            title: Optional title for the plot
            metric: Distance metric for clustering ('euclidean', 'cosine', etc.)
            class_names: Optional map label -> display name
            method: Projection for the 2D view: 'pca' (default), 'mds', 'tsne',
                'umap', or 'phate' (PHATE recommended for NN latent spaces)

        Returns:
            matplotlib Figure object
        """
        from matplotlib.patches import Polygon
        from scipy.spatial import ConvexHull
        from sklearn.cluster import AgglomerativeClustering

        from ..projections import project, METHODS

        if method.lower() not in METHODS:
            raise ValueError(f"Unknown method: {method}. Use one of {METHODS}.")

        # Project to 2D via the shared projections module.
        reduced = project(
            points, method=method, n_components=2,
            random_state=DEFAULT_RANDOM_STATE,
        )
        points_2d = np.asarray(reduced)
        method_name = method.upper()

        # Get cluster assignments at threshold
        if metric == "mahalanobis":
            # Mahalanobis requires a precomputed distance matrix for
            # AgglomerativeClustering because sklearn's linkage tree
            # cannot accept metric_params.
            from sklearn.metrics import pairwise_distances
            VI = np.linalg.pinv(np.cov(points, rowvar=False))
            dist_mat = pairwise_distances(points, metric="mahalanobis", VI=VI)
            clustering = AgglomerativeClustering(
                n_clusters=None, distance_threshold=threshold,
                linkage="single", metric="precomputed",
            )
            cluster_labels = clustering.fit_predict(dist_mat)
        else:
            clustering = AgglomerativeClustering(
                n_clusters=None, distance_threshold=threshold,
                linkage="single", metric=metric,
            )
            cluster_labels = clustering.fit_predict(points)

        # Create the visualization
        fig, ax = plt.subplots(1, 1, figsize=self.figsize)

        # Get unique class labels for unified color palette
        unique_class_labels = sorted(set(true_labels))
        n_classes = len(unique_class_labels)

        # Hull colors derived from dominant true label in each cluster
        # (computed per-cluster below)

        # Detect outliers first (class-based outliers)
        outlier_mask = self._detect_outliers(points_2d, cluster_labels, true_labels, self.outlier_percentage)
        
        # Plot convex hulls and contours for each cluster
        _hull_shade_counter = {}  # tracks shade per dominant label
        unique_clusters = np.unique(cluster_labels)
        for cluster_id in unique_clusters:
            cluster_mask = cluster_labels == cluster_id
            cluster_points_2d = points_2d[cluster_mask]

            if len(cluster_points_2d) >= 3:  # Need at least 3 points for ConvexHull
                try:
                    # Create smooth hull using the existing method
                    hull_points = self._create_blob_boundary(cluster_points_2d, method="smooth")
                    
                    if hull_points is not None:
                        # Create polygon for the blob with distinct color per cluster
                        # Color hull by its dominant true label, shade duplicates
                        cluster_true = true_labels[cluster_mask]
                        dominant = np.bincount(cluster_true).argmax()
                        shade = _hull_shade_counter.get(dominant, 0)
                        _hull_shade_counter[dominant] = shade + 1
                        base_rgba = get_label_color(dominant, n_classes, shade=shade)
                        cluster_color = (*base_rgba[:3], self.alpha_hull)
                        polygon = Polygon(
                            hull_points,
                            alpha=self.alpha_hull,
                            facecolor=cluster_color,
                            edgecolor="darkblue",
                            linewidth=2,
                            linestyle="-",
                        )
                        ax.add_patch(polygon)
                        
                        # Add contour lines inside the blob if enabled, colored by class
                        if self.show_contours and len(cluster_points_2d) >= 5:
                            try:
                                # Get true labels for points in this cluster
                                cluster_true_labels = true_labels[cluster_mask]
                                
                                # Only create contours for non-outlier classes in this cluster
                                cluster_outlier_mask = outlier_mask[cluster_mask]
                                non_outlier_mask = ~cluster_outlier_mask
                                
                                if np.any(non_outlier_mask):
                                    non_outlier_points = cluster_points_2d[non_outlier_mask]
                                    non_outlier_labels = cluster_true_labels[non_outlier_mask]
                                    
                                    # Create class-based contour grids for non-outlier classes only
                                    class_grids = self._create_contour_grid_by_class(
                                        non_outlier_points, non_outlier_labels, hull_points
                                    )
                                    
                                    # Plot contours for each non-outlier class with class-specific colors
                                    for class_id, (X, Y, Z) in class_grids.items():
                                        # Get unified class color
                                        class_color = self._get_class_color(class_id, n_classes)
                                        
                                        # Plot contours for this class
                                        contours = ax.contour(X, Y, Z, levels=4, colors=[class_color], 
                                                            alpha=0.9, linewidths=2.0, zorder=2)
                                    
                            except Exception as e:
                                logger.warning(f"        Warning: Could not create contours for cluster {cluster_id}: {e}")
                        
                except Exception:
                    # Fallback for degenerate cases
                    pass

        # Plot non-outlier blob points as scatter ONLY if contours are disabled
        if not self.show_contours:
            non_outlier_mask = ~outlier_mask
            blob_points = points_2d[non_outlier_mask]
            blob_labels = true_labels[non_outlier_mask]
            
            for class_id in np.unique(blob_labels):
                class_mask = blob_labels == class_id
                if np.any(class_mask):
                    class_label = class_names[class_id] if class_names and class_id in class_names else f'Class {class_id}'
                    ax.scatter(
                        blob_points[class_mask, 0],
                        blob_points[class_mask, 1],
                        c=[self._get_class_color(class_id, n_classes)],
                        s=80,
                        alpha=0.7,
                        edgecolors="white",
                        linewidth=0.5,
                        zorder=3,
                        label=class_label,
                    )

        # Plot outliers as scatter points colored by TRUE classes
        outlier_points = points_2d[outlier_mask]
        outlier_labels = true_labels[outlier_mask]
        
        for class_id in np.unique(outlier_labels):
            class_mask = outlier_labels == class_id
            if np.any(class_mask):
                ax.scatter(
                    outlier_points[class_mask, 0],
                    outlier_points[class_mask, 1],
                    c=[self._get_class_color(class_id, n_classes)],
                    s=120,
                    alpha=0.9,
                    edgecolors="black",
                    linewidth=1.0,
                    zorder=4,
                    label=f'{class_names[class_id]} (outlier)' if class_names and class_id in class_names else f'Class {class_id} Outliers'
                )

        # Set labels and title - BIGGER fonts for paper. PCA gets informative
        # variance-labelled axes; other methods get generic component labels.
        if method.lower() == "pca" and getattr(reduced, "reducer", None) is not None:
            evr = reduced.reducer.explained_variance_ratio_
            ax.set_xlabel(f"PC1 ({evr[0]:.1%} variance)", fontsize=14)
            ax.set_ylabel(f"PC2 ({evr[1]:.1%} variance)", fontsize=14)
        else:
            ax.set_xlabel(f"{method_name} 1", fontsize=14)
            ax.set_ylabel(f"{method_name} 2", fontsize=14)

        if title is None:
            n_outliers = np.sum(outlier_mask)
            title = f"HOLE Blob Visualization: {method_name} + Contours + Outliers (Threshold: {threshold:.3f})"
        ax.set_title(title, fontsize=16, fontweight="bold", pad=20)

        # Clean aesthetics - remove ticks and grids
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        ax.set_facecolor("white")

        for spine in ax.spines.values():
            spine.set_visible(False)

        # Add legend for class labels
        handles, labels_list = ax.get_legend_handles_labels()
        if handles:
            ax.legend(loc='upper right', fontsize=10, frameon=True, fancybox=True, shadow=False, framealpha=0.8)

        plt.tight_layout()

        # Save if path provided
        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
            logger.info(f"Saved blob visualization: {save_path}")

        return fig


def analyze_activation_blobs(
    activation_file: str,
    output_dir: str,
    model_name: str,
    condition_name: str,
    true_labels: Optional[np.ndarray] = None,
    max_points: int = 100,
    distance_metrics: List[str] = None,
    class_names: Optional[Dict[int, str]] = None,
) -> Dict:
    """
    Analyze blob separation for activations at middle and 4th stage thresholds.

    Args:
        activation_file: Path to activation .npy file
        output_dir: Output directory for blob visualizations
        model_name: Model name
        condition_name: Condition name
        true_labels: True class labels
        max_points: Maximum points to use for analysis
        distance_metrics: List of distance metrics to analyze
        class_names: Optional dictionary mapping class indices to names

    Returns:
        Dictionary with blob analysis results
    """
    logger.info(f"Analyzing blob separation for {model_name} - {condition_name}")

    if distance_metrics is None:
        distance_metrics = [
            "Euclidean",
            "Mahalanobis",
            "Cosine",
            "Density_Normalized_Euclidean",
            "Density_Normalized_Mahalanobis",
        ]

    # Load activations
    try:
        all_activations = np.load(activation_file, allow_pickle=True).item()
        if not isinstance(all_activations, dict):
            logger.warning(f"Warning: Expected dictionary, got {type(all_activations)}")
            return {}
    except Exception as e:
        logger.error(f"Error loading {activation_file}: {e}")
        return {}

    if true_labels is None:
        logger.warning("No true labels provided, skipping blob analysis")
        return {}

    # Import required modules
    from ..core.mst_processor import MSTProcessor
    from .cluster_flow import ClusterFlowAnalyzer

    from ..core.distance_metrics import (
        cosine_distance,
        density_normalized_distance,
        distance_matrix,
        mahalanobis_distance,
    )

    # Initialize blob visualizer
    blob_viz = BlobVisualizer(
        figsize=(14, 10), dpi=300, alpha_hull=0.3, class_names=class_names
    )

    # Create output directory
    blob_output_dir = os.path.join(output_dir, "blob_vis")
    os.makedirs(blob_output_dir, exist_ok=True)

    results = {}

    # Process each layer
    for layer_name, activation_data in all_activations.items():
        logger.info(f"  Processing layer: {layer_name}")

        # Handle activation shapes
        if len(activation_data.shape) == 3:
            pc = activation_data[:, 0, :]  # Use class token
        elif len(activation_data.shape) == 2:
            pc = activation_data
        else:
            continue

        # Subsample if needed
        if pc.shape[0] > max_points:
            indices = np.random.choice(pc.shape[0], max_points, replace=False)
            pc = pc[indices]
            layer_labels = true_labels[indices] if true_labels is not None else None
        else:
            layer_labels = true_labels

        if layer_labels is None:
            continue

        # Initialize MST processor for distance calculations
        mst_obj = MSTProcessor()

        try:
            # Compute distance matrices
            X_pca = mst_obj.pca_utils(pc)

            distance_matrices = {}
            if "Euclidean" in distance_metrics:
                distance_matrices["Euclidean"] = distance_matrix(pc)
            if "Mahalanobis" in distance_metrics:
                distance_matrices["Mahalanobis"] = mahalanobis_distance(X_pca)
            if "Cosine" in distance_metrics:
                distance_matrices["Cosine"] = cosine_distance(pc)
            if "Density_Normalized_Euclidean" in distance_metrics:
                base_euclid = distance_matrices.get("Euclidean", distance_matrix(pc))
                distance_matrices[
                    "Density_Normalized_Euclidean"
                ] = density_normalized_distance(X_pca, base_euclid, k=5)
            if "Density_Normalized_Mahalanobis" in distance_metrics:
                base_maha = distance_matrices.get(
                    "Mahalanobis", mahalanobis_distance(X_pca)
                )
                distance_matrices[
                    "Density_Normalized_Mahalanobis"
                ] = density_normalized_distance(X_pca, base_maha, k=5)

            layer_results = {}

            # Process each distance metric
            for dist_name, dist_matrix in distance_matrices.items():
                logger.info(f"    Processing {dist_name} distance metric...")

                try:
                    # Compute cluster evolution
                    analyzer = ClusterFlowAnalyzer(dist_matrix, max_thresholds=4)
                    cluster_evolution = analyzer.compute_cluster_evolution(
                        layer_labels, metric_name=dist_name
                    )

                    # Analyze blob separation
                    blob_results = blob_viz.analyze_blob_separation(
                        pc,
                        layer_labels,
                        cluster_evolution,
                        blob_output_dir,
                        model_name,
                        condition_name,
                        layer_name,
                        dist_name,
                    )

                    layer_results[dist_name] = blob_results

                except Exception as e:
                    logger.error(f"      Error processing {dist_name}: {e}")
                    continue

            results[layer_name] = layer_results

        except Exception as e:
            logger.error(f"    Error processing layer {layer_name}: {e}")
            continue

    return results


def run_blob_analysis_on_results(
    results_dir: str = "results_compression",
    max_points: int = 100,
    distance_metrics: List[str] = None,
    class_names: Optional[Dict[int, str]] = None,
) -> None:
    """
    Run blob analysis on all activation files in results directory.

    Args:
        results_dir: Directory containing model results
        max_points: Maximum points per analysis
        distance_metrics: List of distance metrics to analyze
        class_names: Optional dictionary mapping class indices to names
    """
    # Resolve to an absolute path once so nothing below depends on the process
    # working directory (and so the label-file search never has to escape via
    # cwd-relative "../" hops).
    results_dir = os.path.abspath(os.path.expanduser(results_dir))
    logger.info(f"Running blob analysis on {results_dir}...")

    if distance_metrics is None:
        distance_metrics = [
            "Euclidean",
            "Mahalanobis",
            "Cosine",
            "Density_Normalized_Euclidean",
            "Density_Normalized_Mahalanobis",
        ]

    # Load true labels. Search only WITHIN the (absolute) results_dir and its
    # parent's canonical "results/original" layout -- all resolved to absolute
    # paths, no cwd-relative fallbacks.
    parent = os.path.dirname(results_dir)
    possible_paths = [
        os.path.join(results_dir, "test_labels.npy"),
        os.path.join(results_dir, "original", "true_labels.npy"),
        os.path.join(parent, "results", "original", "true_labels.npy"),
    ]
    labels_file = None
    for path in possible_paths:
        if os.path.exists(path):
            labels_file = path
            break

    if labels_file and os.path.exists(labels_file):
        true_labels = np.load(labels_file)
        logger.info(f"Loaded true labels from {labels_file}")
    else:
        logger.warning("Warning: True labels not found in any expected location")
        logger.info(f"Searched: {[labels_file]}")
        return

    # Process each model directory
    for model_name in os.listdir(results_dir):
        model_path = f"{results_dir}/{model_name}"
        if os.path.isdir(model_path) and model_name not in [
            "visualizations",
            "model_stats",
            "tda_analysis",
            "persistence_dendrograms",
            "flow_visualization",
            "blob_vis",
        ]:
            activations_dir = f"{model_path}/activations"
            if os.path.exists(activations_dir):
                logger.info(f"Processing {model_name}...")

                # Process each activation file
                activation_files = [
                    f
                    for f in os.listdir(activations_dir)
                    if f.endswith("_all_layers.npy")
                ]

                for activation_file in activation_files:
                    condition_name = activation_file.replace("_all_layers.npy", "")
                    activation_path = f"{activations_dir}/{activation_file}"

                    analyze_activation_blobs(
                        activation_path,
                        results_dir,
                        model_name,
                        condition_name,
                        true_labels,
                        max_points,
                        distance_metrics,
                        class_names,
                    )


if __name__ == "__main__":
    print(
        "Blob Visualization: Cluster separation analysis at persistent homology thresholds"
    )
    print("Usage:")
    print("  from vis.blob_vis import BlobVisualizer, analyze_activation_blobs")
    print("  run_blob_analysis_on_results('results_compression')")
