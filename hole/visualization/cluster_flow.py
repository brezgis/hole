"""
Flow Visualization for Persistent Homology Cluster Evolution

This module provides Sankey diagrams and stacked bar charts to show how clusters
evolve through different death thresholds in persistent homology filtration.
Based on the reference ComponentEvolutionVisualizer implementation.
"""

import os
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from matplotlib.patches import FancyBboxPatch

# Import distance functions and shared persistence helpers from core
from ..core.distance_metrics import (
    cosine_distance,
    density_normalized_distance,
    distance_matrix,
    mahalanobis_distance,
)
from ..core.persistence import compute_cluster_evolution as _compute_cluster_evolution


class _UnionFind:
    """Minimal union-find (disjoint-set) with path compression + union by rank.

    Used to sweep single-linkage/MST merges incrementally: instead of rebuilding
    a graph and recomputing connected components at every threshold (O(n^2)
    each), we start from n singletons and union one MST edge at a time as the
    threshold rises, so advancing to the next threshold is near-O(1) amortised.
    """

    __slots__ = ("parent", "rank", "n_comp")

    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n
        self.n_comp = n

    def find(self, x: int) -> int:
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a: int, b: int) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        self.n_comp -= 1
        return True

    def labels(self) -> np.ndarray:
        """Consecutive integer labels (0..k-1) for the current partition."""
        roots = [self.find(i) for i in range(len(self.parent))]
        remap = {}
        out = np.empty(len(roots), dtype=int)
        for i, r in enumerate(roots):
            cid = remap.get(r)
            if cid is None:
                cid = len(remap)
                remap[r] = cid
            out[i] = cid
        return out


class ClusterFlowAnalyzer:
    """Analyzes cluster evolution through persistent homology filtration."""

    def __init__(self, distance_matrix: np.ndarray, max_thresholds: int = 8):
        """
        Initialize with distance matrix.

        Args:
            distance_matrix: 2D symmetric distance matrix
            max_thresholds: Maximum number of thresholds to analyze
        """
        self.distance_matrix = distance_matrix
        self.n_points = distance_matrix.shape[0]
        self.max_thresholds = max_thresholds

        # Will be computed
        self.persistence = None
        self.death_thresholds = None
        self.cluster_evolution = None

        # MST cache: single-linkage merge structure, computed once and reused for
        # every threshold query (see _ensure_mst / _labels_at / _sweep_best).
        self._mst_w = None  # sorted MST edge weights (== H0 death thresholds)
        self._mst_r = None  # edge endpoints, aligned with _mst_w
        self._mst_c = None

    # ------------------------------------------------------------------ #
    # MST-backed connectivity (fast, exact for any symmetric distance)
    # ------------------------------------------------------------------ #
    def _ensure_mst(self):
        """Compute (once) the MST of the distance graph and sort its edges.

        Connected components of the "distance <= t" graph are identical to those
        of the MST restricted to edges with weight <= t -- a minimax-path
        property that holds for ANY symmetric non-negative weight matrix, metric
        or not. So the sorted MST edge weights ARE the H0 death thresholds
        (single-linkage merge heights), and we never need to build the full Rips
        complex or a dense per-threshold graph.
        """
        if self._mst_w is not None:
            return
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import minimum_spanning_tree

        D = np.array(self.distance_matrix, dtype=float, copy=True)
        n = self.n_points
        # Exact-zero off-diagonal distances (duplicate/near-duplicate points --
        # ubiquitous in LLM latents) would be dropped by the sparse
        # representation and those points would never merge. Bump them to a tiny
        # positive epsilon so they become the earliest (cheapest) merges.
        off = ~np.eye(n, dtype=bool)
        pos = D[off & (D > 0)]
        eps = (pos.min() * 1e-6) if pos.size else 1e-12
        zero_off = off & (D <= 0)
        if zero_off.any():
            D[zero_off] = eps

        mst = minimum_spanning_tree(csr_matrix(D)).tocoo()
        w, r, c = mst.data, mst.row, mst.col
        order = np.argsort(w, kind="mergesort")  # stable, ascending
        self._mst_w = w[order]
        self._mst_r = r[order].astype(int)
        self._mst_c = c[order].astype(int)

    def _all_merge_thresholds(self) -> List[float]:
        """Distinct single-linkage merge heights, ascending (the H0 deaths)."""
        self._ensure_mst()
        if self._mst_w is None or len(self._mst_w) == 0:
            return []
        return sorted(set(self._mst_w.tolist()))

    def _labels_at(self, threshold: float) -> np.ndarray:
        """Cluster labels at ``threshold`` via union-find over MST edges <= t."""
        self._ensure_mst()
        uf = _UnionFind(self.n_points)
        k = int(np.searchsorted(self._mst_w, threshold, side="right"))
        for i in range(k):
            uf.union(int(self._mst_r[i]), int(self._mst_c[i]))
        return uf.labels()

    def _n_clusters_at(self, threshold: float) -> int:
        """Number of connected components at ``threshold`` (cheap)."""
        self._ensure_mst()
        k = int(np.searchsorted(self._mst_w, threshold, side="right"))
        # each distinct merge reduces the component count by one
        uf = _UnionFind(self.n_points)
        for i in range(k):
            uf.union(int(self._mst_r[i]), int(self._mst_c[i]))
        return uf.n_comp

    def _sweep_best(self, thresholds, score_fn, maximize=True):
        """Scan ``thresholds`` (ascending) with ONE incremental union-find.

        For each threshold we advance the union-find over any MST edges that have
        become active, then score the current partition. Because the union-find
        is never rebuilt, the whole sweep is ~O(n alpha(n) + L * score_cost)
        instead of O(L * n^2) -- this is the "jump ahead between merge events"
        optimisation. Returns (best_threshold, best_score).
        """
        self._ensure_mst()
        thresholds = list(thresholds)
        if not thresholds:
            return None, None
        uf = _UnionFind(self.n_points)
        w, r, c = self._mst_w, self._mst_r, self._mst_c
        m = len(w)
        ei = 0
        best_t, best_s = thresholds[0], (-np.inf if maximize else np.inf)
        for t in thresholds:
            while ei < m and w[ei] <= t:
                uf.union(int(r[ei]), int(c[ei]))
                ei += 1
            s = score_fn(uf.labels())
            if (maximize and s > best_s) or (not maximize and s < best_s):
                best_s, best_t = s, t
        return best_t, best_s

    def compute_cluster_evolution(
        self, true_labels: Optional[np.ndarray] = None,
        filter_small_clusters: bool = False,
        min_cluster_size: int = 10,
        metric_name: str = "Euclidean",
    ) -> Dict:
        """
        Compute cluster evolution through different death thresholds.
        Returns data in the format expected by ComponentEvolutionVisualizer.

        Args:
            true_labels: Optional true labels for comparison
            filter_small_clusters: If True, remove datapoints from clusters with size <= min_cluster_size at middle threshold
            min_cluster_size: Minimum cluster size threshold for filtering
            metric_name: Name of the distance metric (used as dictionary key)

        Returns:
            Dictionary containing components_ and labels_ in the expected format
        """
        logger.info("Computing single-linkage merge thresholds (via MST)...")

        # The H0 death thresholds of the Rips filtration are exactly the
        # single-linkage merge heights, i.e. the MST edge weights. Deriving them
        # from the MST avoids building the full Rips simplex tree over all
        # O(n^2) edges just to read off H0 -- a large speedup on the filtering
        # step. self.persistence is kept populated (as H0 birth/death pairs) so
        # the public attribute still means something.
        all_thresholds = self._all_merge_thresholds()
        self.persistence = [(0, (0.0, t)) for t in all_thresholds]
        logger.info(f"Found {len(all_thresholds)} total death thresholds")

        # Select 4 specific thresholds for meaningful 5-stage evolution
        selected_thresholds = self._select_meaningful_thresholds(
            all_thresholds, true_labels
        )

        logger.info(f"Selected thresholds: {[f'{t:.4f}' for t in selected_thresholds]}")

        # Initialize components_ and labels_ dictionaries
        components_ = {metric_name: {}}
        labels_ = {metric_name: {}}
        
        # Delegate per-threshold connected-component computation to the shared
        # helper in core.persistence so the two implementations don't drift.
        evolution = _compute_cluster_evolution(self.distance_matrix, selected_thresholds)
        all_cluster_labels = {
            str(threshold): evolution[threshold]["labels"]
            for threshold in selected_thresholds
        }
        
        # Filter small clusters if requested
        filter_mask = np.ones(self.n_points, dtype=bool)
        if filter_small_clusters and len(selected_thresholds) >= 3:
            # Use middle threshold (stage 3 - index 1 in selected_thresholds)
            middle_threshold = str(selected_thresholds[1])
            middle_labels = all_cluster_labels[middle_threshold]
            
            # Count cluster sizes at middle threshold
            cluster_sizes = Counter(middle_labels)
            
            # Identify points in small clusters
            small_clusters = {cid for cid, count in cluster_sizes.items() if count <= min_cluster_size}
            filter_mask = np.array([label not in small_clusters for label in middle_labels])
            
            n_filtered = np.sum(~filter_mask)
            logger.info(f"Filtering {n_filtered} points from {len(small_clusters)} small clusters (size <= {min_cluster_size})")
        
        # Apply filtering and store results
        for threshold in selected_thresholds:
            threshold_str = str(threshold)
            cluster_labels = all_cluster_labels[threshold_str]
            
            if filter_small_clusters:
                # Keep only filtered points
                filtered_labels = cluster_labels[filter_mask]
                
                # Renumber clusters to be consecutive
                unique_clusters = sorted(set(filtered_labels))
                cluster_map = {old_id: new_id for new_id, old_id in enumerate(unique_clusters)}
                filtered_labels = np.array([cluster_map[label] for label in filtered_labels])
                
                components_[metric_name][threshold_str] = len(unique_clusters)
                labels_[metric_name][threshold_str] = filtered_labels
            else:
                components_[metric_name][threshold_str] = len(set(cluster_labels))
                labels_[metric_name][threshold_str] = cluster_labels
        
        # Also filter true_labels if filtering is enabled
        filtered_true_labels = true_labels
        if filter_small_clusters and true_labels is not None:
            filtered_true_labels = true_labels[filter_mask]

        return {
            "components_": components_,
            "labels_": labels_,
            "true_labels": filtered_true_labels,
        }

    def _select_meaningful_thresholds(
        self, all_thresholds: List[float], true_labels: Optional[np.ndarray] = None
    ) -> List[float]:
        """
        Select 4 meaningful thresholds for 5-stage visualization:
        Stage 1: True labels (not a threshold)
        Stage 2: Initial clusters (very small threshold - many clusters)
        Stage 3: Clusters similar to true labels (threshold where clusters roughly match the true classes)
        Stage 4: Intermediate merging (between similar and final)
        Stage 5: Final single cluster

        Robust to duplicate/tied thresholds: the four semantic picks may collide
        (e.g. when many pairwise distances are equal), so after de-duplicating we
        BACKFILL from the remaining thresholds to always return as many distinct
        stages as are available (up to 4). This prevents the downstream Sankey /
        stacked-bar plots -- which require >= 4 stages -- from silently failing
        with "Need >= 4 stages" whenever two picks happen to coincide.
        """
        # De-duplicate up front so all reasoning below is over distinct values.
        all_thresholds = sorted(set(all_thresholds))
        if len(all_thresholds) < 4:
            logger.warning(
                f"Only {len(all_thresholds)} distinct threshold(s) available; "
                "using all of them."
            )
            return all_thresholds

        # Stage 2: Initial clusters - smallest threshold (many small clusters)
        initial_threshold = all_thresholds[0]

        # Stage 5: Final single cluster. Skip-ahead: the whole graph is connected
        # once the last MST edge activates, so the answer is simply the largest
        # merge height -- no per-threshold connectivity scan needed.
        final_threshold = all_thresholds[-1]
        if self._n_clusters_at(final_threshold) != 1:
            # Disconnected distance graph (e.g. inf distances): fall back to the
            # threshold that yields the fewest components.
            final_threshold = min(
                all_thresholds, key=lambda t: self._n_clusters_at(t)
            )

        # Stage 3: threshold whose clustering best matches the true labels.
        similar_threshold = self._find_similar_to_true_labels(
            all_thresholds, true_labels
        )

        # Stage 4: intermediate threshold between "similar" and "final".
        intermediate_candidates = [
            t for t in all_thresholds if similar_threshold < t < final_threshold
        ]
        if intermediate_candidates:
            intermediate_threshold = intermediate_candidates[
                len(intermediate_candidates) // 2
            ]
        else:
            target = (similar_threshold + final_threshold) / 2
            intermediate_threshold = min(
                all_thresholds, key=lambda x: abs(x - target)
            )

        # De-duplicate the semantic picks, preserving their meaningful order.
        selected = []
        for t in (initial_threshold, similar_threshold,
                  intermediate_threshold, final_threshold):
            if t not in selected:
                selected.append(t)

        # Backfill to 4 distinct stages when picks collided, drawing evenly from
        # the thresholds strictly between the smallest and largest already-picked
        # values so the added stages are informative rather than clumped.
        target_n = min(4, len(all_thresholds))
        if len(selected) < target_n:
            pool = [t for t in all_thresholds if t not in selected]
            # order pool by distance to the centre of the current spread so the
            # gaps get filled first
            lo, hi = min(selected), max(selected)
            mid = 0.5 * (lo + hi)
            pool.sort(key=lambda t: abs(t - mid))
            for t in pool:
                if len(selected) >= target_n:
                    break
                selected.append(t)

        selected = sorted(selected)

        logger.info("Selected thresholds breakdown:")
        stage_names = ["Initial (many clusters)", "Similar to true labels",
                       "Intermediate merging", "Final single cluster"]
        for name, t in zip(stage_names, selected):
            logger.info(f"  {name}: {t:.4f}")

        return selected

    def _find_similar_to_true_labels(
        self, all_thresholds: List[float], true_labels: Optional[np.ndarray] = None
    ) -> float:
        """
        Find threshold where clusters best match true labels - where data points
        are grouped together with their original class labels (with some outliers).
        Uses clustering purity/homogeneity to find best match.
        """
        if not all_thresholds:
            return 0.0

        if true_labels is None:
            # Fallback when no labels are available: aim for a moderate, dataset
            # -agnostic number of clusters rather than the fully-merged extreme.
            # Uses the incremental MST sweep (no dense per-threshold graphs).
            target_clusters = 10
            best_threshold, _ = self._sweep_best(
                all_thresholds,
                score_fn=lambda labels: abs(len(set(labels.tolist())) - target_clusters),
                maximize=False,
            )
            return best_threshold if best_threshold is not None else all_thresholds[0]

        logger.info("Finding threshold where data points cluster with their true labels...")

        true_labels = np.asarray(true_labels)
        # Densify true labels to 0..c-1 once so scoring can use a contingency
        # matrix instead of per-cluster np.unique loops (much faster per sweep).
        uniq_true = np.unique(true_labels)
        true_dense = np.searchsorted(uniq_true, true_labels)
        n_true = len(uniq_true)
        n_total = len(true_dense)

        def _combined(cluster_labels):
            # Build the cluster x true-class contingency table in one vectorised
            # pass. Purity = sum of per-cluster maxima / N (each cluster's
            # majority true class); homogeneity = sum of per-true-class maxima / N
            # (each class's majority cluster). Prioritise purity, keep some
            # homogeneity.
            k = int(cluster_labels.max()) + 1 if len(cluster_labels) else 0
            if k == 0 or n_total == 0:
                return 0.0
            contingency = np.zeros((k, n_true), dtype=np.int64)
            np.add.at(contingency, (cluster_labels, true_dense), 1)
            purity = contingency.max(axis=1).sum() / n_total
            homogeneity = contingency.max(axis=0).sum() / n_total
            return 0.7 * purity + 0.3 * homogeneity

        best_threshold, best_score = self._sweep_best(
            all_thresholds, score_fn=_combined, maximize=True
        )
        if best_threshold is None:
            best_threshold = all_thresholds[len(all_thresholds) // 3]
            best_score = 0.0

        logger.info(f"    Best threshold: {best_threshold:.4f} (score: {best_score:.3f})")
        return best_threshold

    def _calculate_purity(
        self, true_labels: np.ndarray, cluster_labels: np.ndarray
    ) -> float:
        """
        Calculate clustering purity: for each cluster, what fraction belongs to the most common true class.
        High purity means clusters contain mostly points from the same true class.
        """
        if len(true_labels) != len(cluster_labels):
            return 0.0

        total_correct = 0
        total_points = len(true_labels)

        # For each cluster, find the most common true label
        unique_clusters = np.unique(cluster_labels)

        for cluster_id in unique_clusters:
            # Get all points in this cluster
            cluster_mask = cluster_labels == cluster_id
            cluster_true_labels = true_labels[cluster_mask]

            if len(cluster_true_labels) > 0:
                # Find most common true label in this cluster
                unique_labels, counts = np.unique(
                    cluster_true_labels, return_counts=True
                )
                max_count = np.max(counts)
                total_correct += max_count

        purity = total_correct / total_points if total_points > 0 else 0.0
        return purity

    def _calculate_homogeneity(
        self, true_labels: np.ndarray, cluster_labels: np.ndarray
    ) -> float:
        """
        Calculate clustering homogeneity: for each true class, what fraction is in the most common cluster.
        High homogeneity means each true class is mostly in one cluster.
        """
        if len(true_labels) != len(cluster_labels):
            return 0.0

        total_correct = 0
        total_points = len(true_labels)

        # For each true class, find the most common cluster
        unique_true_labels = np.unique(true_labels)

        for true_label in unique_true_labels:
            # Get all points with this true label
            true_mask = true_labels == true_label
            true_cluster_labels = cluster_labels[true_mask]

            if len(true_cluster_labels) > 0:
                # Find most common cluster for this true label
                unique_clusters, counts = np.unique(
                    true_cluster_labels, return_counts=True
                )
                max_count = np.max(counts)
                total_correct += max_count

        homogeneity = total_correct / total_points if total_points > 0 else 0.0
        return homogeneity


class ComponentEvolutionVisualizer:
    """
    A class for visualizing component evolution through death thresholds.
    """

    def __init__(self, components_, labels_, class_names=None):
        """
        Initialize the component evolution visualizer.

        Args:
            components_: Dictionary of components at each threshold
            labels_: Dictionary of labels at each threshold
            class_names: Optional dictionary mapping class indices to names
        """
        self.components_ = components_
        self.labels_ = labels_
        # Initialize with None - will be created when needed
        self.color_mapping = None

        # Default class names if none provided
        self.class_names = class_names or {
            0: "Cluster_0",
            1: "Cluster_1",
            2: "Cluster_2",
            3: "Cluster_3",
            4: "Cluster_4",
            5: "Cluster_5",
            6: "Cluster_6",
            7: "Cluster_7",
            8: "Cluster_8",
            9: "Cluster_9",
        }

    def _create_color_mapping(self, key, thresholds, original_labels=None):
        """Create a consistent color mapping for all components across all thresholds.

        Uses composite string keys to avoid collision between true label IDs and cluster IDs.
        True labels: 'L{id}' (e.g. 'L0', 'L1')
        Clusters: 'T{threshold_idx}_C{cluster_id}' (e.g. 'T0_C5', 'T1_C3')
        """
        from .scatter_hull import get_label_color

        n_true_classes = len(set(original_labels) - {-1}) if original_labels is not None else 0

        # Build full color mapping
        color_mapping = {}

        # True labels always get the base shade, with 'L' prefix
        if original_labels is not None:
            for label in sorted(set(original_labels) - {-1}):
                color_mapping[f'L{label}'] = get_label_color(label, n_true_classes)

        # Filtration clusters: color by dominant true label, shade duplicates PER threshold
        for threshold_idx, threshold in enumerate(thresholds):
            threshold_str = str(threshold)
            if threshold_str not in self.labels_[key]:
                continue
            cluster_labels = self.labels_[key][threshold_str]

            # Group clusters by dominant label at this threshold
            dominant_map = {}  # cluster_id -> dominant_label
            for cluster_id in set(cluster_labels):
                if cluster_id == -1:
                    continue  # Noise handled separately
                if original_labels is not None:
                    mask = cluster_labels == cluster_id
                    dominant_map[cluster_id] = Counter(original_labels[mask]).most_common(1)[0][0]

            # Shade counter resets per threshold
            label_counter = defaultdict(int)
            for cluster_id in sorted(dominant_map.keys()):
                dom_label = dominant_map[cluster_id]
                shade = label_counter[dom_label]
                # Use 'T{idx}_C{id}' format to avoid collision with true labels
                color_mapping[f'T{threshold_idx}_C{cluster_id}'] = get_label_color(dom_label, n_true_classes, shade=shade)
                label_counter[dom_label] += 1

        color_mapping['noise'] = (0.5, 0.5, 0.5, 1.0)
        logger.debug(f"Created color mapping for {len(color_mapping)} component ids")
        return color_mapping

    def plot_sankey(
        self,
        key,
        original_labels=None,
        ax=None,
        title=None,
        gray_second_layer=True,
        show_true_labels_text=True,
        show_filtration_text=True,
        stage_order=None,
        stage_labels=None,
        x_axis_label="Cluster Evolution Stages",
    ):
        """
        Create an (N+1)-stage Sankey diagram. Stage 0 is always the true
        labels; the remaining stages are taken from the data.

        Two calling conventions are supported:

        * Legacy (intra-layer filtration): pass nothing extra. The stages are
          the numeric death thresholds stored in ``components_[key]``, sorted by
          value, and displayed as ``f"{t:.4f}"``. Requires >= 4 thresholds, as
          before.
        * Cross-layer evolution: pass ``stage_order`` as an explicit ordered
          list of the string keys in ``labels_[key]`` (e.g. layer names). The
          x-axis then represents model depth rather than filtration scale, and
          ``stage_labels`` (optional) gives the per-stage display text.

        Flows between consecutive stages are co-membership counts over the same
        N points, so this renders splits as well as merges.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(18, 10))

        if key not in self.components_ or key not in self.labels_:
            ax.text(
                0.5,
                0.5,
                f"No data found for key: {key}",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=14,
            )
            return ax

        # Check if we have original labels
        has_original = original_labels is not None
        if not has_original:
            ax.text(
                0.5,
                0.5,
                "No true labels provided for 5-stage visualization",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=14,
            )
            return ax

        # Determine the ordered stages. Legacy callers pass numeric filtration
        # thresholds (sorted by value); cross-layer callers pass an explicit
        # stage_order of arbitrary string keys (e.g. layer names) so the x-axis
        # represents model depth rather than filtration scale.
        if stage_order is None:
            stage_keys = sorted(self.components_[key].keys(), key=float)
            stage_disp = [f"{float(k):.4f}" for k in stage_keys]
            min_stages = 4
        else:
            stage_keys = [str(s) for s in stage_order]
            stage_disp = (
                [str(s) for s in stage_labels]
                if stage_labels is not None
                else [str(s) for s in stage_keys]
            )
            min_stages = 1

        if len(stage_keys) < min_stages:
            ax.text(
                0.5,
                0.5,
                f"Need >= {min_stages} stages, got {len(stage_keys)}",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=14,
            )
            return ax

        # Create consistent color mapping across all stages
        self.color_mapping = self._create_color_mapping(
            key, stage_keys, original_labels
        )

        # Stage 0 is always the true labels; the rest come from the data.
        stage_names = ["True Labels"] + stage_disp
        n_stages = len(stage_names)

        # Evenly space stages across the canvas. Reduces to the legacy
        # 0.1, 0.3, 0.5, 0.7, 0.9 layout when n_stages == 5.
        x_positions = list(np.linspace(0.1, 0.9, n_stages))

        # Track node positions for each stage
        node_positions = {}
        flows = []

        # STAGE 1: True labels
        logger.debug("Creating Stage 1: True labels")
        original_counts = Counter(original_labels)
        total_points = len(original_labels)

        y_start, y_end = 0.1, 0.9
        total_height = y_end - y_start
        current_y = y_start
        node_positions[0] = {}

        for comp_id in sorted(original_counts.keys()):
            count = original_counts[comp_id]
            height = (count / total_points) * total_height

            node_positions[0][comp_id] = {
                "x": x_positions[0],
                "y": current_y + height / 2,
                "height": height,
                "count": count,
                "color": self.color_mapping.get(f'L{comp_id}', (0.5, 0.5, 0.5, 1.0)),
                "y_start": current_y,
                "y_end": current_y + height,
            }
            current_y += height

        # STAGES 1..N: Process each data stage (threshold or layer)
        for stage_idx, stage_key in enumerate(stage_keys):
            actual_stage = stage_idx + 1
            threshold_str = stage_key

            if threshold_str not in self.labels_[key]:
                continue

            logger.debug(f"Creating Stage {actual_stage + 1}: {stage_disp[stage_idx]}")

            labels_at_threshold = self.labels_[key][threshold_str]
            component_counts = Counter(labels_at_threshold)

            # Calculate positions for this stage
            current_y = y_start
            node_positions[actual_stage] = {}

            for comp_id in sorted(component_counts.keys()):
                count = component_counts[comp_id]
                height = (count / total_points) * total_height

                # Use gray for second layer (first filtration stage) if requested
                if (gray_second_layer and actual_stage == 1):
                    color = (0.7, 0.7, 0.7, 1.0)  # Light gray
                elif comp_id == -1:
                    color = (0.5, 0.5, 0.5, 1.0)  # Noise
                else:
                    # Look up with threshold-specific key
                    color = self.color_mapping.get(f'T{stage_idx}_C{comp_id}', (0.3, 0.3, 0.3, 1.0))

                node_positions[actual_stage][comp_id] = {
                    "x": x_positions[actual_stage],
                    "y": current_y + height / 2,
                    "height": height,
                    "count": count,
                    "color": color,
                    "y_start": current_y,
                    "y_end": current_y + height,
                }
                current_y += height

        # Calculate flows between consecutive stages
        for stage in range(n_stages - 1):
            from_stage = stage
            to_stage = stage + 1

            # Get labels for both stages
            if from_stage == 0:
                labels1 = original_labels
            else:
                stage_pos = from_stage - 1
                if stage_pos < len(stage_keys):
                    labels1 = self.labels_[key][stage_keys[stage_pos]]
                else:
                    continue

            stage_pos = to_stage - 1
            if stage_pos < len(stage_keys):
                labels2 = self.labels_[key][stage_keys[stage_pos]]
            else:
                continue

            # Calculate flows
            flow_mapping = defaultdict(lambda: defaultdict(int))
            for point_idx, (comp1, comp2) in enumerate(zip(labels1, labels2)):
                flow_mapping[comp1][comp2] += 1

            # Create flow objects
            for comp1, comp2_dict in flow_mapping.items():
                for comp2, count in comp2_dict.items():
                    if (
                        comp1 in node_positions[from_stage]
                        and comp2 in node_positions[to_stage]
                    ):
                        flows.append(
                            {
                                "from_stage": from_stage,
                                "to_stage": to_stage,
                                "from_comp": comp1,
                                "to_comp": comp2,
                                "count": count,
                                "from_node": node_positions[from_stage][comp1],
                                "to_node": node_positions[to_stage][comp2],
                            }
                        )

        # Draw flows (behind nodes)
        max_flow = max([f["count"] for f in flows]) if flows else 1

        for flow in flows:
            thickness = max(0.003, min(0.04, (flow["count"] / max_flow) * 0.06))

            from_node = flow["from_node"]
            to_node = flow["to_node"]

            # Connection points
            x1 = from_node["x"] + 0.01
            y1 = from_node["y"]
            x2 = to_node["x"] - 0.01
            y2 = to_node["y"]

            # Create smooth bezier curve
            control_distance = (x2 - x1) * 0.3
            cx1 = x1 + control_distance
            cx2 = x2 - control_distance

            # Generate curve points
            n_points = 20
            t_values = np.linspace(0, 1, n_points)

            curve_x, curve_y = [], []
            for t in t_values:
                bx = (
                    (1 - t) ** 3 * x1
                    + 3 * (1 - t) ** 2 * t * cx1
                    + 3 * (1 - t) * t**2 * cx2
                    + t**3 * x2
                )
                by = (
                    (1 - t) ** 3 * y1
                    + 3 * (1 - t) ** 2 * t * y1
                    + 3 * (1 - t) * t**2 * y2
                    + t**3 * y2
                )
                curve_x.append(bx)
                curve_y.append(by)

            # Create flow polygon
            upper_y = [y + thickness / 2 for y in curve_y]
            lower_y = [y - thickness / 2 for y in curve_y]

            flow_x = curve_x + curve_x[::-1]
            flow_y = upper_y + lower_y[::-1]

            ax.fill(
                flow_x,
                flow_y,
                color=from_node["color"],
                alpha=0.6,
                edgecolor="none",
                zorder=1,
            )

        # Draw nodes (on top of flows)
        for stage_idx in range(n_stages):
            if stage_idx not in node_positions:
                continue

            for comp_id, node in node_positions[stage_idx].items():
                # Draw node rectangle - make wider for better text visibility
                rect = FancyBboxPatch(
                    (node["x"] - 0.012, node["y_start"]),
                    0.024,
                    node["height"],
                    boxstyle="round,pad=0.001",
                    facecolor=node["color"],
                    edgecolor="black",
                    linewidth=0.8,
                    alpha=0.9,
                    zorder=2,
                )
                ax.add_patch(rect)

                # Add labels based on flags
                if (
                    node["height"] > 0.015
                ):  # Slightly lower threshold for better visibility
                    show_text = False
                    if stage_idx == 0:
                        # Stage 1: True labels - controlled by show_true_labels_text
                        show_text = show_true_labels_text
                        if comp_id in self.class_names:
                            label_text = self.class_names[comp_id]
                            font_size = 7 if len(label_text) > 6 else 8
                        else:
                            label_text = f"{comp_id}"
                            font_size = 8
                    else:
                        # Other stages: Filtration stages - controlled by show_filtration_text
                        show_text = show_filtration_text
                        label_text = f"{comp_id}"
                        font_size = 8

                    if show_text:
                        ax.text(
                            node["x"],
                            node["y"],
                            label_text,
                            ha="center",
                            va="center",
                            fontsize=font_size,
                            fontweight="bold",
                            color="white",
                            rotation=0,
                        )

        # Add stage labels with threshold values - BIGGER text for paper
        for i, stage_name in enumerate(stage_names):
            ax.text(
                x_positions[i],
                0.05,
                stage_name,
                ha="center",
                va="center",
                fontsize=14,  # Bigger for paper
                rotation=0,
                fontweight="normal",  # Remove bold as requested
            )

        # Formatting
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel(x_axis_label, fontsize=16)  # Bigger for paper
        ax.set_ylabel("Component Size (Normalized)", fontsize=16)  # Bigger for paper

        if title:
            ax.set_title(title, fontsize=18, pad=20)

        # Remove ticks and spines
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

        # Add legend for class names
        if self.class_names and 0 in node_positions:
            from matplotlib.patches import Patch
            legend_handles = []
            for comp_id in sorted(node_positions[0].keys()):
                name = self.class_names.get(comp_id, f"Class {comp_id}")
                color = node_positions[0][comp_id]["color"]
                legend_handles.append(Patch(facecolor=color, edgecolor="black", linewidth=0.5, label=name))
            ax.legend(handles=legend_handles, loc="upper right", fontsize=10,
                      frameon=True, fancybox=True, shadow=False, framealpha=0.8)

        return ax

    def plot_stacked_bars(
        self,
        key,
        original_labels=None,
        ax=None,
        title=None,
        gray_second_layer=True,
        show_true_labels_text=True,
        show_filtration_text=True,
        stage_order=None,
        stage_labels=None,
        x_axis_label="Cluster Evolution Stages",
    ):
        """
        Create an (N+1)-stage stacked bar chart (true labels + N data stages).

        Supports the same two calling conventions as :meth:`plot_sankey`:
        legacy numeric thresholds (when ``stage_order`` is omitted) or an
        explicit ``stage_order`` of string keys for cross-layer evolution.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(16, 10))

        if key not in self.components_ or key not in self.labels_:
            ax.text(
                0.5,
                0.5,
                f"No data found for key: {key}",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=14,
            )
            return ax

        # Check if we have original labels
        has_original = original_labels is not None
        if not has_original:
            ax.text(
                0.5,
                0.5,
                "No true labels provided for 5-stage visualization",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=14,
            )
            return ax

        # Determine ordered stages (see plot_sankey for the two conventions).
        if stage_order is None:
            stage_keys = sorted(self.components_[key].keys(), key=float)
            stage_disp = [f"{float(k):.4f}" for k in stage_keys]
            min_stages = 4
        else:
            stage_keys = [str(s) for s in stage_order]
            stage_disp = (
                [str(s) for s in stage_labels]
                if stage_labels is not None
                else [str(s) for s in stage_keys]
            )
            min_stages = 1

        if len(stage_keys) < min_stages:
            ax.text(
                0.5,
                0.5,
                f"Need >= {min_stages} stages, got {len(stage_keys)}",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=14,
            )
            return ax

        # Create consistent color mapping if not already created
        if self.color_mapping is None:
            self.color_mapping = self._create_color_mapping(
                key, stage_keys, original_labels
            )

        # Layout: true labels at 0, a small gap, then consecutive data stages.
        # Reduces to the legacy [0, 1.5, 2, 3, 4, 5] layout for 4 stages.
        stage_names = ["True Labels", ""] + stage_disp
        x_positions = np.array([0, 1.5] + [2.0 + i for i in range(len(stage_keys))])
        bar_width = 1.0  # Make bars stick together by using full width

        # Process each stage
        stage_data = []

        # Stage 1: True labels
        original_counts = Counter(original_labels)
        stage_data.append(("True Labels", original_counts))

        # Stage 2: Empty separator (white bar)
        stage_data.append(("", {}))

        # Stages 3..: Each data stage
        for stage_key, disp in zip(stage_keys, stage_disp):
            if stage_key in self.labels_[key]:
                labels_at_threshold = self.labels_[key][stage_key]
                component_counts = Counter(labels_at_threshold)
                stage_data.append((disp, component_counts))

        # Create stacked bars
        for stage_idx, (stage_name, component_counts) in enumerate(stage_data):
            if stage_idx == 1:  # Empty separator - just skip, no visible bar
                continue

            if not component_counts:
                continue

            # Stack components in a single bar
            bottom = 0
            for comp_id in sorted(component_counts.keys()):
                count = component_counts[comp_id]

                # Use gray for second layer (initial PH clusters) if requested
                if (
                    gray_second_layer and stage_idx == 2
                ):  # Second threshold layer (first filtration stage)
                    color = (0.7, 0.7, 0.7, 1.0)  # Light gray
                elif stage_idx == 0:
                    color = self.color_mapping.get(f'L{comp_id}', (0.5, 0.5, 0.5, 1.0))
                else:
                    threshold_idx = stage_idx - 2
                    color = self.color_mapping.get(f'T{threshold_idx}_C{comp_id}', (0.3, 0.3, 0.3, 1.0))

                # Create bar segment
                ax.bar(
                    x_positions[stage_idx],
                    count,
                    bottom=bottom,
                    width=bar_width,
                    color=color,
                    alpha=0.85,
                    edgecolor="black",
                    linewidth=0.5,
                )

                # Add component label if significant and flags allow
                if (
                    count > sum(component_counts.values()) * 0.04
                ):  # Show labels for >4% of total
                    show_text = False
                    if stage_idx == 0:
                        # Stage 1: True labels - controlled by show_true_labels_text
                        show_text = show_true_labels_text
                        label_text = f"{comp_id}"
                        font_size = (
                            12 if len(label_text) <= 6 else 10
                        )  # Bigger for paper
                    elif stage_idx > 1:  # Skip empty separator stage (stage_idx == 1)
                        # Filtration stages - controlled by show_filtration_text
                        show_text = show_filtration_text
                        label_text = f"{comp_id}"
                        font_size = 12  # Bigger for paper

                    if show_text:
                        ax.text(
                            x_positions[stage_idx],
                            bottom + count / 2,
                            label_text,
                            ha="center",
                            va="center",
                            fontsize=font_size,
                            fontweight="normal",  # Remove bold as requested
                            color="white",
                            rotation=0,
                        )

                bottom += count

        # Customize plot - BIGGER fonts for paper
        ax.set_xlabel(x_axis_label, fontsize=16)  # Bigger for paper
        ax.set_ylabel("Component Size", fontsize=16)  # Bigger for paper
        # ax.set_title(
        #     f"Stacked Bar Chart - {title if title else key}", fontsize=18, pad=20  # Bigger for paper
        # )

        # Set x-axis labels with threshold values (skip the empty separator)
        ax.set_xticks(
            [x_positions[0]] + list(x_positions[2:])
        )  # Skip separator position
        ax.set_xticklabels(
            [stage_names[0]] + stage_names[2:], fontsize=14, rotation=0
        )  # Bigger for paper

        # Set x-axis limits to avoid any weird spacing around the invisible separator
        ax.set_xlim(-0.5, float(x_positions[-1]) + 0.5)

        # Remove all visual elements except the bars
        ax.set_yticks([])
        ax.grid(False)  # Explicitly turn off grid
        for spine in ax.spines.values():
            spine.set_visible(False)

        # Add legend for class names
        if self.class_names and original_labels is not None:
            from matplotlib.patches import Patch
            unique_labels = sorted(set(original_labels))
            legend_handles = []
            for comp_id in unique_labels:
                name = self.class_names.get(comp_id, f"Class {comp_id}")
                color = self.color_mapping.get(f'L{comp_id}', (0.5, 0.5, 0.5, 1.0))
                legend_handles.append(Patch(facecolor=color, edgecolor="black", linewidth=0.5, label=name))
            ax.legend(handles=legend_handles, loc="upper right", fontsize=10,
                      frameon=True, fancybox=True, shadow=False, framealpha=0.8)

        return ax


class FlowVisualizer:
    """
    High-level flow visualization class that wraps ComponentEvolutionVisualizer
    with more user-friendly interface.
    """

    def __init__(
        self,
        figsize: Tuple[int, int] = (20, 12),
        dpi: int = 800,
        class_names: Optional[Dict[int, str]] = None,
    ):
        """
        Initialize the flow visualizer.

        Args:
            figsize: Figure size for plots
            dpi: DPI for saved plots
            class_names: Optional dictionary mapping class indices to names
        """
        self.figsize = figsize
        self.dpi = dpi
        self.class_names = class_names

    def plot_sankey_flow(
        self,
        cluster_evolution: Dict,
        save_path: Optional[str] = None,
        # title: str = "5-Stage Cluster Evolution",
        title: str = None,
        show_true_labels_text: bool = True,
        show_filtration_text: bool = True,
        stage_order: Optional[List] = None,
        stage_labels: Optional[List] = None,
        x_axis_label: str = "Cluster Evolution Stages",
        gray_second_layer: bool = True,
    ) -> plt.Figure:
        """
        Plot a Sankey diagram showing cluster evolution using ComponentEvolutionVisualizer.

        Pass ``stage_order``/``stage_labels`` to render cross-layer evolution
        (x-axis = model depth) instead of intra-layer filtration.

        Args:
            cluster_evolution: Dictionary from ClusterFlowAnalyzer.compute_cluster_evolution()
            save_path: Path to save the figure
            title: Title for the plot
            show_true_labels_text: Whether to show text labels in true labels blocks
            show_filtration_text: Whether to show text labels in filtration stage blocks

        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        plt.tight_layout()

        # Extract components and labels
        components_ = cluster_evolution.get("components_", {})
        labels_ = cluster_evolution.get("labels_", {})
        true_labels = cluster_evolution.get("true_labels", None)

        # Create visualizer
        visualizer = ComponentEvolutionVisualizer(
            components_, labels_, self.class_names
        )

        # Plot Sankey diagram
        for key in components_.keys():
            visualizer.plot_sankey(
                key,
                true_labels,
                ax,
                title,
                show_true_labels_text=show_true_labels_text,
                show_filtration_text=show_filtration_text,
                stage_order=stage_order,
                stage_labels=stage_labels,
                x_axis_label=x_axis_label,
                gray_second_layer=gray_second_layer,
            )
            break  # avoid plotting all distance metrics in one plot

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight", facecolor="white")
            logger.info(f"Saved Sankey flow diagram: {save_path}")

        return fig

    def plot_stacked_bar_evolution(
        self,
        cluster_evolution: Dict,
        save_path: Optional[str] = None,
        title: str = "5-Stage Cluster Evolution",
        show_true_labels_text: bool = True,
        show_filtration_text: bool = True,
        stage_order: Optional[List] = None,
        stage_labels: Optional[List] = None,
        x_axis_label: str = "Cluster Evolution Stages",
        gray_second_layer: bool = True,
    ) -> plt.Figure:
        """
        Plot a stacked bar chart showing cluster evolution using ComponentEvolutionVisualizer

        Args:
            cluster_evolution: Dictionary from ClusterFlowAnalyzer.compute_cluster_evolution()
            save_path: Path to save the figure
            title: Title for the plot
            show_true_labels_text: Whether to show text labels in true labels blocks
            show_filtration_text: Whether to show text labels in filtration stage blocks

        Returns:
            matplotlib Figure object
        """
        # fig, ax = plt.subplots(figsize=(self.figsize[0], self.figsize[1] * 0.6))
        fig, ax = plt.subplots(figsize=self.figsize)
        plt.tight_layout()

        # Extract components and labels
        components_ = cluster_evolution.get("components_", {})
        labels_ = cluster_evolution.get("labels_", {})
        true_labels = cluster_evolution.get("true_labels", None)

        # Create visualizer
        visualizer = ComponentEvolutionVisualizer(
            components_, labels_, self.class_names
        )

        # Plot stacked bars
        for key in components_.keys():
            visualizer.plot_stacked_bars(
                key,
                true_labels,
                ax,
                title,
                show_true_labels_text=show_true_labels_text,
                show_filtration_text=show_filtration_text,
                stage_order=stage_order,
                stage_labels=stage_labels,
                x_axis_label=x_axis_label,
                gray_second_layer=gray_second_layer,
            )
            break  # so that we only plot one distance metric in one plot

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight", facecolor="white")
            logger.info(f"Saved stacked bar evolution chart: {save_path}")

        return fig


def analyze_activation_flows(
    activation_file: str,
    output_dir: str,
    model_name: str,
    condition_name: str,
    true_labels: Optional[np.ndarray] = None,
    max_points: int = 100,
    max_thresholds: int = 6,
    class_names: Optional[Dict[int, str]] = None,
    distance_metrics: Optional[List[str]] = None,
) -> Dict:
    """
    Analyze cluster flow evolution for activations using 5-stage ComponentEvolutionVisualizer..

    Args:
        activation_file: Path to activation .npy file
        output_dir: Output directory for flow visualizations
        model_name: Model name
        condition_name: Condition name
        true_labels: True class labels
        max_points: Maximum points to use for analysis
        max_thresholds: Maximum number of thresholds to plot
        class_names: Optional dictionary mapping class indices to names
        distance_metrics: List of distance metrics to compute. If None, computes all 5.

    Returns:
        Dictionary with flow analysis results
    """
    logger.info(f"Analyzing cluster flow evolution for {model_name} - {condition_name}")

    # Load activations
    try:
        all_activations = np.load(activation_file, allow_pickle=True).item()
        if not isinstance(all_activations, dict):
            logger.warning(f"Expected dictionary, got {type(all_activations)}")
            return {}
    except Exception as e:
        logger.error(f"Error loading {activation_file}: {e}")
        return {}

    # Create output directory
    flow_output_dir = os.path.join(output_dir, f"{model_name}_{condition_name}")
    os.makedirs(flow_output_dir, exist_ok=True)

    # Initialize flow visualizer
    flow_viz = FlowVisualizer(figsize=(40, 20), class_names=class_names)

    # Default class names if none provided
    if class_names is None:
        class_names = {i: f"Class_{i}" for i in range(10)}

    # Import required modules
    from ..core.mst_processor import MSTProcessor

    results = {}

    # Process each layer
    for layer_name, activation_data in all_activations.items():
        logger.info(f"Processing layer: {layer_name}")

        # Handle different activation shapes
        if len(activation_data.shape) == 3:
            # [batch_size, seq_len, hidden_dim] - use class token
            pc = activation_data[:, 0, :]
        elif len(activation_data.shape) == 2:
            # [batch_size, hidden_dim] - already flattened
            pc = activation_data
        else:
            logger.warning(f"Unexpected shape {activation_data.shape}, skipping layer {layer_name}")
            continue

        # Subsample if too many points
        if pc.shape[0] > max_points:
            indices = np.random.choice(pc.shape[0], max_points, replace=False)
            pc = pc[indices]
            layer_labels = true_labels[indices] if true_labels is not None else None
        else:
            layer_labels = true_labels

        if layer_labels is None:
            logger.warning(f"No labels for {layer_name}, skipping...")
            continue

        # Clean layer name for filename
        clean_layer_name = layer_name.replace("/", "_").replace(".", "_")

        # Initialize MST processor for distance calculations
        mst_obj = MSTProcessor()

        try:
            # Compute distance matrices
            X_pca = mst_obj.pca_utils(pc)

            distance_matrices = {}
            if distance_metrics is None or "Euclidean" in distance_metrics:
                distance_matrices["Euclidean"] = distance_matrix(pc)
            if distance_metrics is None or "Mahalanobis" in distance_metrics:
                distance_matrices["Mahalanobis"] = mahalanobis_distance(X_pca)
            if distance_metrics is None or "Cosine" in distance_metrics:
                distance_matrices["Cosine"] = cosine_distance(pc)

            # Add density normalized versions
            if distance_metrics is None or "Density_Normalized_Euclidean" in distance_metrics:
                base_euclid = distance_matrices.get("Euclidean", distance_matrix(pc))
                distance_matrices[
                    "Density_Normalized_Euclidean"
                ] = density_normalized_distance(X_pca, base_euclid, k=5)
            if distance_metrics is None or "Density_Normalized_Mahalanobis" in distance_metrics:
                base_maha = distance_matrices.get("Mahalanobis", mahalanobis_distance(X_pca))
                distance_matrices[
                    "Density_Normalized_Mahalanobis"
                ] = density_normalized_distance(X_pca, base_maha, k=5)

            layer_results = {}

            # Process each distance metric
            for dist_name, dist_matrix in distance_matrices.items():
                logger.info(f"Processing {dist_name} distance metric...")

                try:
                    # Create title
                    if condition_name.lower() in ["inference", "clean"]:
                        if model_name.lower() == "original":
                            title_prefix = f"ViT Model - {dist_name} - {layer_name}"
                        else:
                            title_prefix = f"ViT {model_name.replace('_', ' ').title()} - {dist_name} - {layer_name}"
                    else:
                        if model_name.lower() == "original":
                            title_prefix = f"ViT Model - {condition_name.replace('_', ' ').title()} - {dist_name} - {layer_name}"
                        else:
                            title_prefix = f"ViT {model_name.replace('_', ' ').title()} - {condition_name.replace('_', ' ').title()} - {dist_name} - {layer_name}"

                    # Compute cluster evolution
                    analyzer = ClusterFlowAnalyzer(
                        dist_matrix, max_thresholds=max_thresholds
                    )
                    cluster_evolution = analyzer.compute_cluster_evolution(
                        layer_labels, metric_name=dist_name
                    )

                    # Save Sankey diagram
                    sankey_path = os.path.join(
                        flow_output_dir, f"{clean_layer_name}_{dist_name}_sankey.png"
                    )
                    sankey_fig = flow_viz.plot_sankey_flow(
                        cluster_evolution,
                        save_path=sankey_path,
                        # title=f"Sankey Diagram - {title_prefix}",
                    )
                    plt.close(sankey_fig)

                    # Save stacked bar chart
                    bars_path = os.path.join(
                        flow_output_dir,
                        f"{clean_layer_name}_{dist_name}_stacked_bars.png",
                    )
                    bars_fig = flow_viz.plot_stacked_bar_evolution(
                        cluster_evolution,
                        save_path=bars_path,
                        title=f"Stacked Bar Chart - {title_prefix}",
                    )
                    plt.close(bars_fig)

                    # Store results
                    layer_results[dist_name] = {
                        "cluster_evolution": cluster_evolution,
                        "sankey_path": sankey_path,
                        "bars_path": bars_path,
                    }

                except Exception as e:
                    logger.error(f"Error processing {dist_name}: {e}")
                    continue

            results[layer_name] = layer_results

        except Exception as e:
            logger.error(f"Error processing layer {layer_name}: {e}")
            continue

    return results


if __name__ == "__main__":
    logger.info("Flow Visualization: 5-stage component evolution through persistent homology")
    logger.info(
        "Stages: True Labels → Initial Clusters → Similar to True → Intermediate → Final Cluster"
    )
