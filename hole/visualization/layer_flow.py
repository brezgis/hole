"""
Cross-Layer Cluster Evolution for Persistent Homology.

Where :mod:`hole.visualization.cluster_flow` shows how clusters evolve through
filtration thresholds *within* a single point cloud (one layer), this module
shows how clusters evolve *across* the layers of a model. The x-axis becomes
**model depth** rather than filtration scale -- e.g. for a 32-layer LLM you
might inspect layers ``[0, 8, 16, 24, 31]`` and watch the class structure
emerge, split, and merge as you descend the stack.

The construction reuses the intra-layer Sankey machinery wholesale, because the
flow accounting only needs one invariant: **the same N input examples produce an
activation at every layer, so a point keeps its identity across layers**. A flow
between two adjacent stages is just the count of points that were in cluster A at
the shallower layer and cluster B at the deeper one. Unlike single-linkage
filtration (which only merges), depth-to-depth flows render both merges *and*
splits -- which is exactly the signal of representations disentangling.

Per-layer clustering uses a *fixed target-k* cut of the single-linkage
(minimum-spanning-tree) hierarchy: for each layer we pick the filtration
threshold whose connected components number ``k`` (default ``k`` = number of true
classes). Using the same ``k`` at every layer keeps the stages comparable, so
"early layers are tangled, deep layers separate" emerges honestly instead of
being forced by a per-layer purity-matched threshold.
"""

import os
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
from loguru import logger
from scipy.sparse.csgraph import minimum_spanning_tree

from ..core.distance_metrics import (
    chebyshev_distance,
    cosine_distance,
    euclidean_distance,
    mahalanobis_distance,
    manhattan_distance,
)
from ..core.persistence import compute_cluster_evolution as _compute_cluster_evolution
from .cluster_flow import FlowVisualizer

# Map friendly metric names to the core distance-matrix builders. Keys double as
# the dictionary key under which results are stored and as figure-name tokens.
METRIC_FUNCS = {
    "Euclidean": euclidean_distance,
    "Cosine": cosine_distance,
    "Manhattan": manhattan_distance,
    "Chebyshev": chebyshev_distance,
    "Mahalanobis": mahalanobis_distance,
}


def cluster_to_k(distance_matrix: np.ndarray, k: int) -> tuple:
    """Cut the single-linkage hierarchy of ``distance_matrix`` into ``k`` clusters.

    For 0-dimensional persistent homology of a Rips filtration, the death events
    are exactly the minimum-spanning-tree edge weights: each finite death merges
    two components, so after including the ``m`` smallest MST edges there are
    ``n - m`` connected components. To obtain ``k`` clusters we therefore choose
    the threshold equal to the ``(n - k)``-th smallest MST edge weight and label
    points by connected components at that threshold. This is exact, needs no
    GUDHI call, and can always reach any ``k`` in ``[1, n]``.

    Parameters
    ----------
    distance_matrix : np.ndarray
        Symmetric ``(n, n)`` distance matrix.
    k : int
        Desired number of clusters (clamped to ``[1, n]``).

    Returns
    -------
    (threshold, labels) : (float, np.ndarray)
        The chosen filtration threshold and the integer cluster label per point.
        Note: ties in MST edge weights can yield slightly fewer than ``k``
        clusters (several merges happen at one threshold); this is reported via
        the returned ``labels`` and is itself informative.
    """
    n = distance_matrix.shape[0]
    k = int(max(1, min(k, n)))

    # MST edge weights, ascending -- the single-linkage merge (death) thresholds.
    mst = minimum_spanning_tree(distance_matrix).toarray()
    weights = np.sort(mst[mst > 0.0])

    if k >= n or weights.size == 0:
        threshold = 0.0  # every point is its own cluster
    elif k <= 1:
        threshold = float(weights[-1])  # one merge short of fully connected
    else:
        # Need (n - k) merges -> include the (n - k) smallest MST edges.
        idx = min(max(n - k - 1, 0), weights.size - 1)
        threshold = float(weights[idx])

    labels = _compute_cluster_evolution(distance_matrix, [threshold])[threshold][
        "labels"
    ]
    return threshold, np.asarray(labels)


class LayerEvolutionAnalyzer:
    """Compute cluster evolution across the layers of a model.

    Produces the same ``components_`` / ``labels_`` structure consumed by
    :class:`~hole.visualization.cluster_flow.ComponentEvolutionVisualizer`, but
    keyed by **layer** instead of filtration threshold, plus an explicit ordered
    stage list so the renderer lays depth out left-to-right.
    """

    def __init__(
        self,
        embeddings: Dict[str, np.ndarray],
        true_labels: np.ndarray,
        layers: Optional[Sequence[Union[int, str]]] = None,
        metric: str = "Euclidean",
        n_clusters_per_layer: Optional[int] = None,
        max_points: Optional[int] = None,
        layer_key_fmt: str = "layer_{}",
        stage_label_fmt: str = "L{}",
        random_state: int = 0,
    ):
        """
        Parameters
        ----------
        embeddings : dict
            ``{layer_key: ndarray(N, D)}`` -- one already-pooled vector per
            example per layer. Rows MUST be aligned across layers (example ``i``
            is the same row at every layer); this is what makes cross-layer flow
            meaningful.
        true_labels : np.ndarray
            ``(N,)`` class label per example, used to anchor colours and set the
            default ``k``.
        layers : sequence of int or str, optional
            The layers to use as stages, in display order. Integers are mapped to
            keys via ``layer_key_fmt`` (e.g. ``8 -> "layer_8"``); strings are used
            verbatim. ``None`` uses every key in ``embeddings`` (sorted by the
            trailing integer when present, else lexicographically).
        metric : str
            One of :data:`METRIC_FUNCS`.
        n_clusters_per_layer : int, optional
            Target ``k`` per layer. Defaults to the number of distinct true labels.
        max_points : int, optional
            If set and ``N`` exceeds it, subsample to this many points **once** and
            reuse the same indices for every layer (preserving alignment).
        layer_key_fmt : str
            Format string mapping an integer layer index to its ``embeddings`` key.
        stage_label_fmt : str
            Format string for the per-stage display label from the layer index
            (or key, when layers are passed as strings).
        random_state : int
            Seed for the one-shot subsample.
        """
        if metric not in METRIC_FUNCS:
            raise ValueError(
                f"Unknown metric '{metric}'. Choose from {sorted(METRIC_FUNCS)}"
            )

        self.embeddings = embeddings
        self.true_labels = np.asarray(true_labels)
        self.metric = metric
        self.layer_key_fmt = layer_key_fmt
        self.stage_label_fmt = stage_label_fmt
        self.max_points = max_points
        self.random_state = random_state

        self.n_clusters_per_layer = (
            int(n_clusters_per_layer)
            if n_clusters_per_layer is not None
            else len(set(self.true_labels.tolist()))
        )

        self._resolve_stages(layers)

    def _resolve_stages(self, layers):
        """Resolve the requested layers into ordered (key, display-label) pairs."""
        if layers is None:
            keys = list(self.embeddings.keys())

            def _natural_key(key):
                tail = key.rsplit("_", 1)[-1]
                return (0, int(tail)) if tail.isdigit() else (1, key)

            keys = sorted(keys, key=_natural_key)
            self.stage_keys = keys
            self.stage_labels = [
                self.stage_label_fmt.format(k.rsplit("_", 1)[-1]) for k in keys
            ]
        else:
            stage_keys, stage_labels = [], []
            for item in layers:
                if isinstance(item, (int, np.integer)):
                    key = self.layer_key_fmt.format(int(item))
                    label = self.stage_label_fmt.format(int(item))
                else:
                    key = str(item)
                    tail = key.rsplit("_", 1)[-1]
                    label = self.stage_label_fmt.format(tail if tail.isdigit() else key)
                if key not in self.embeddings:
                    raise KeyError(
                        f"Layer key '{key}' not found in embeddings. "
                        f"Available: {list(self.embeddings.keys())[:8]}..."
                    )
                stage_keys.append(key)
                stage_labels.append(label)
            self.stage_keys = stage_keys
            self.stage_labels = stage_labels

    def _subsample_indices(self, n_points: int) -> Optional[np.ndarray]:
        """Choose subsample indices once, shared across all layers."""
        if self.max_points is None or n_points <= self.max_points:
            return None
        rng = np.random.default_rng(self.random_state)
        return np.sort(rng.choice(n_points, self.max_points, replace=False))

    def compute(self) -> Dict:
        """Run the per-layer clustering and assemble the evolution payload.

        Returns
        -------
        dict
            ``components_``, ``labels_`` (both keyed ``{metric: {layer_key: ...}}``),
            ``true_labels``, ``stage_order`` and ``stage_labels`` -- ready to hand
            to :class:`FlowVisualizer` with ``stage_order``/``stage_labels`` set.
        """
        metric_fn = METRIC_FUNCS[self.metric]

        # Determine N from the first requested layer and pick shared subsample.
        first = self.embeddings[self.stage_keys[0]]
        n_points = first.shape[0]
        idx = self._subsample_indices(n_points)

        true_labels = self.true_labels
        if idx is not None:
            true_labels = true_labels[idx]
            logger.info(
                f"Subsampled {self.max_points}/{n_points} points (shared across layers)"
            )

        components_ = {self.metric: {}}
        labels_ = {self.metric: {}}

        for key in self.stage_keys:
            pc = np.asarray(self.embeddings[key])
            if idx is not None:
                pc = pc[idx]

            dist = metric_fn(pc)
            threshold, cluster_labels = cluster_to_k(dist, self.n_clusters_per_layer)

            n_clusters = len(set(cluster_labels.tolist()))
            components_[self.metric][key] = n_clusters
            labels_[self.metric][key] = cluster_labels
            logger.info(
                f"  {key}: k_target={self.n_clusters_per_layer} -> "
                f"{n_clusters} clusters @ threshold {threshold:.4f}"
            )

        return {
            "components_": components_,
            "labels_": labels_,
            "true_labels": true_labels,
            "stage_order": list(self.stage_keys),
            "stage_labels": list(self.stage_labels),
        }


def analyze_layer_flows(
    embeddings: Union[str, Dict[str, np.ndarray]],
    output_dir: str,
    true_labels: np.ndarray,
    layers: Optional[Sequence[Union[int, str]]] = None,
    model_name: str = "model",
    condition_name: str = "clean",
    metrics: Sequence[str] = ("Euclidean",),
    n_clusters_per_layer: Optional[int] = None,
    max_points: Optional[int] = None,
    class_names: Optional[Dict[int, str]] = None,
    layer_key_fmt: str = "layer_{}",
    stage_label_fmt: str = "L{}",
    figsize: tuple = (24, 12),
    dpi: int = 300,
    random_state: int = 0,
) -> Dict:
    """Render cross-layer cluster-evolution Sankey + stacked-bar figures.

    Mirrors :func:`~hole.visualization.cluster_flow.analyze_activation_flows`, but
    the x-axis is **model depth**: one figure per (model, condition, metric)
    spanning the chosen ``layers``.

    Parameters
    ----------
    embeddings : dict or str
        ``{layer_key: ndarray(N, D)}`` of aligned per-layer embeddings, or a path
        to a ``.npy`` file containing such a dict (``np.load(..., allow_pickle=True).item()``).
    output_dir : str
        Base directory; figures land under ``output_dir/{model}_{condition}/``.
    true_labels : np.ndarray
        ``(N,)`` class labels.
    layers : sequence of int or str, optional
        Layers to use as stages (integers map via ``layer_key_fmt``). ``None``
        uses all layers in ``embeddings``.
    metrics : sequence of str
        Distance metrics to render, each producing its own figure pair.
    n_clusters_per_layer : int, optional
        Target ``k`` per layer (default = number of true classes).
    max_points : int, optional
        One-shot subsample cap shared across layers.
    class_names : dict, optional
        ``{class_id: name}`` for the true-label legend.

    Returns
    -------
    dict
        ``{metric: {"cluster_evolution": ..., "sankey_path": ..., "bars_path": ...}}``.
    """
    import matplotlib.pyplot as plt

    if isinstance(embeddings, str):
        embeddings = np.load(embeddings, allow_pickle=True).item()
    if not isinstance(embeddings, dict):
        raise TypeError(
            f"embeddings must be a dict or path to a .npy dict, got {type(embeddings)}"
        )

    out_dir = os.path.join(output_dir, f"{model_name}_{condition_name}")
    os.makedirs(out_dir, exist_ok=True)

    flow_viz = FlowVisualizer(figsize=figsize, dpi=dpi, class_names=class_names)
    results = {}

    for metric in metrics:
        logger.info(f"Cross-layer flow: {model_name}/{condition_name} [{metric}]")
        try:
            analyzer = LayerEvolutionAnalyzer(
                embeddings,
                true_labels,
                layers=layers,
                metric=metric,
                n_clusters_per_layer=n_clusters_per_layer,
                max_points=max_points,
                layer_key_fmt=layer_key_fmt,
                stage_label_fmt=stage_label_fmt,
                random_state=random_state,
            )
            evolution = analyzer.compute()
        except Exception as e:  # noqa: BLE001 - report and skip this metric
            logger.error(f"Failed to compute layer evolution for {metric}: {e}")
            continue

        stage_order = evolution["stage_order"]
        stage_labels = evolution["stage_labels"]

        sankey_path = os.path.join(out_dir, f"{metric}_layerflow_sankey.png")
        sankey_fig = flow_viz.plot_sankey_flow(
            evolution,
            save_path=sankey_path,
            title=f"{model_name} {condition_name} - {metric} - cluster flow across layers",
            stage_order=stage_order,
            stage_labels=stage_labels,
            x_axis_label="Model Depth (layer)",
            gray_second_layer=False,  # every layer is a depth stage, not a filtration step
        )
        plt.close(sankey_fig)

        bars_path = os.path.join(out_dir, f"{metric}_layerflow_stacked_bars.png")
        bars_fig = flow_viz.plot_stacked_bar_evolution(
            evolution,
            save_path=bars_path,
            title=f"{model_name} {condition_name} - {metric} - cluster sizes across layers",
            stage_order=stage_order,
            stage_labels=stage_labels,
            x_axis_label="Model Depth (layer)",
            gray_second_layer=False,  # every layer is a depth stage, not a filtration step
        )
        plt.close(bars_fig)

        results[metric] = {
            "cluster_evolution": evolution,
            "sankey_path": sankey_path,
            "bars_path": bars_path,
        }

    return results
