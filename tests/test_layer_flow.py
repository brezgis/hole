"""Tests for the fork's additions: cross-layer cluster evolution
(``hole.visualization.layer_flow``) and the dendrogram class-color band
(``PersistenceDendrogram.plot_dendrogram(class_labels=...)``).

These cover the behaviours surfaced by the adversarial review: the tie-robust
``cluster_to_k`` cut, the N-alignment guards, input validation, natural-sort of
layer keys, and the band's color consistency (including the noise label -1).
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

import hole
from hole import LayerEvolutionAnalyzer, analyze_layer_flows, cluster_to_k
from hole.visualization.heatmap_dendrograms import PersistenceDendrogram
from hole.visualization.scatter_hull import get_label_color


# --------------------------------------------------------------------------- #
# cluster_to_k
# --------------------------------------------------------------------------- #
def _well_separated(n_clusters, per=4, dim=6, seed=0):
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(n_clusters, dim)) * 30.0
    labels = np.repeat(np.arange(n_clusters), per)
    X = centers[labels] + rng.normal(size=(n_clusters * per, dim)) * 0.05
    return X, labels


def test_cluster_to_k_distinct_is_exact():
    X, _ = _well_separated(5, per=6)
    D = hole.euclidean_distance(X)
    for k in (1, 2, 3, 5, 10, X.shape[0]):
        _, lab = cluster_to_k(D, k)
        assert len(set(lab.tolist())) == k


def test_cluster_to_k_tight_pairs_ties():
    """Regression: tied/near-tied MST weights must not collapse the cut.

    The old threshold-based cut returned ~4 clusters for k=9 on 10 tight pairs;
    the fcluster maxclust cut returns the requested k.
    """
    rng = np.random.default_rng(1)
    centers = rng.normal(size=(10, 6)) * 20.0
    X = np.repeat(centers, 2, axis=0) + rng.normal(size=(20, 6)) * 0.01
    D = hole.euclidean_distance(X)
    for k in (9, 10, 11):
        _, lab = cluster_to_k(D, k)
        assert len(set(lab.tolist())) == k


def test_cluster_to_k_extremes_and_clamping():
    X, _ = _well_separated(3, per=4)
    D = hole.euclidean_distance(X)
    n = D.shape[0]
    # k <= 1 -> single cluster; k >= n (and beyond) -> all singletons
    assert len(set(cluster_to_k(D, 1)[1].tolist())) == 1
    assert len(set(cluster_to_k(D, 0)[1].tolist())) == 1
    assert len(set(cluster_to_k(D, n)[1].tolist())) == n
    assert len(set(cluster_to_k(D, n + 50)[1].tolist())) == n


def test_cluster_to_k_tiny_inputs():
    assert cluster_to_k(np.zeros((1, 1)), 3)[1].tolist() == [0]
    thr, lab = cluster_to_k(np.array([[0.0, 2.0], [2.0, 0.0]]), 2)
    assert len(set(lab.tolist())) == 2


# --------------------------------------------------------------------------- #
# LayerEvolutionAnalyzer: validation + bookkeeping
# --------------------------------------------------------------------------- #
def _layers(n_layers=4, n=24, dim=8, n_classes=3, seed=0):
    """Aligned per-layer embeddings whose separation grows with depth."""
    rng = np.random.default_rng(seed)
    labels = np.repeat(np.arange(n_classes), n // n_classes)
    centers = rng.normal(size=(n_classes, dim)) * 6.0
    noise = rng.normal(size=(labels.size, dim)) * 2.0
    emb = {
        f"layer_{i}": (sep * centers[labels] + noise).astype(np.float32)
        for i, sep in enumerate(np.linspace(0.0, 1.0, n_layers))
    }
    return emb, labels


def test_analyzer_rejects_misaligned_layer_n():
    emb = {"layer_0": np.random.randn(20, 5), "layer_1": np.random.randn(18, 5)}
    with pytest.raises(ValueError, match="aligned"):
        LayerEvolutionAnalyzer(emb, np.zeros(20), layers=[0, 1]).compute()


def test_analyzer_rejects_label_length_mismatch():
    emb = {"layer_0": np.random.randn(20, 5)}
    with pytest.raises(ValueError, match="true_labels"):
        LayerEvolutionAnalyzer(emb, np.zeros(7), layers=[0]).compute()


def test_analyzer_rejects_empty_embeddings():
    with pytest.raises(ValueError):
        LayerEvolutionAnalyzer({}, np.zeros(3))


def test_analyzer_rejects_empty_layers():
    emb, labels = _layers()
    with pytest.raises(ValueError):
        LayerEvolutionAnalyzer(emb, labels, layers=[])


def test_analyzer_unknown_layer_key():
    emb, labels = _layers()
    with pytest.raises(KeyError):
        LayerEvolutionAnalyzer(emb, labels, layers=[99])


def test_analyzer_unknown_metric():
    emb, labels = _layers()
    with pytest.raises(ValueError):
        LayerEvolutionAnalyzer(emb, labels, layers=[0], metric="bogus")


def test_analyzer_natural_sort_layer_keys():
    rng = np.random.default_rng(0)
    emb = {f"layer_{i}": rng.normal(size=(6, 4)) for i in (2, 10, 1)}
    a = LayerEvolutionAnalyzer(emb, np.arange(6) % 2)  # layers=None -> all, sorted
    assert a.stage_keys == ["layer_1", "layer_2", "layer_10"]
    assert a.stage_labels == ["L1", "L2", "L10"]


def test_analyzer_default_k_is_num_classes():
    emb, labels = _layers(n_classes=3)
    a = LayerEvolutionAnalyzer(emb, labels, layers=[0, 1])
    assert a.n_clusters_per_layer == 3


def test_analyzer_compute_payload_aligned():
    emb, labels = _layers(n_layers=4, n=24, n_classes=3)
    ev = LayerEvolutionAnalyzer(emb, labels, layers=[0, 2, 3]).compute()
    assert list(ev["stage_order"]) == ["layer_0", "layer_2", "layer_3"]
    assert ev["true_labels"].shape[0] == 24
    for key in ev["stage_order"]:
        assert ev["labels_"]["Euclidean"][key].shape[0] == 24  # one label per point
        assert ev["components_"]["Euclidean"][key] >= 1


def test_analyzer_subsample_shared_across_layers():
    emb, labels = _layers(n_layers=3, n=30, n_classes=3)
    ev = LayerEvolutionAnalyzer(
        emb, labels, layers=[0, 1, 2], max_points=12, random_state=0
    ).compute()
    assert ev["true_labels"].shape[0] == 12
    for key in ev["stage_order"]:
        assert ev["labels_"]["Euclidean"][key].shape[0] == 12


def test_analyze_layer_flows_end_to_end(tmp_path):
    emb, labels = _layers(n_layers=5, n=30, n_classes=3)
    res = analyze_layer_flows(
        emb,
        output_dir=str(tmp_path),
        true_labels=labels,
        layers=[0, 2, 4],
        metrics=("Euclidean", "Cosine"),
    )
    assert set(res) == {"Euclidean", "Cosine"}
    for r in res.values():
        assert (tmp_path / "model_clean" / "Euclidean_layerflow_sankey.png").parent.exists()
        assert __import__("os").path.exists(r["sankey_path"])
        assert __import__("os").path.exists(r["bars_path"])


# --------------------------------------------------------------------------- #
# Dendrogram class-color band
# --------------------------------------------------------------------------- #
def _dendro(points):
    pd = PersistenceDendrogram(points=np.asarray(points, dtype=float))
    pd.build_linkage_matrix_from_persistence()
    return pd


def _band_facecolors_by_leaf(pd, class_labels):
    """Return the band patch facecolor for each leaf, in dendrogram leaf order."""
    plt.close("all")
    result = pd.plot_dendrogram(class_labels=class_labels, show_legend=False)
    fig = plt.gcf()
    band_ax = fig.axes[1]  # [dendrogram, band]
    colors = [tuple(p.get_facecolor()) for p in band_ax.patches]
    plt.close(fig)
    return result["leaves"], colors


def test_dendrogram_band_noise_label_is_gray():
    rng = np.random.default_rng(0)
    pts = np.vstack([rng.normal(0, 0.1, (4, 3)), rng.normal(5, 0.1, (4, 3))])
    class_labels = np.array([0, 0, 0, 0, 1, 1, 1, -1])
    leaves, colors = _band_facecolors_by_leaf(_dendro(pts), class_labels)
    gray = (0.5, 0.5, 0.5, 1.0)
    # the leaf whose class is -1 must be gray
    for leaf_pos, leaf_idx in enumerate(leaves):
        if class_labels[leaf_idx] == -1:
            assert colors[leaf_pos] == gray
            break
    else:
        pytest.fail("no -1 leaf found")


def test_dendrogram_band_colors_match_raw_label():
    """Non-contiguous labels must color by raw class id (matching get_label_color),
    so the band is consistent with scatter/cluster_flow plots."""
    rng = np.random.default_rng(1)
    cids = [2, 5, 9]
    pts = np.vstack([rng.normal(c * 5, 0.1, (3, 3)) for c in cids])
    class_labels = np.repeat(cids, 3)
    leaves, colors = _band_facecolors_by_leaf(_dendro(pts), class_labels)
    n_classes = 3
    for leaf_pos, leaf_idx in enumerate(leaves):
        cid = int(class_labels[leaf_idx])
        expected = tuple(get_label_color(cid, n_classes=max(n_classes, 2)))
        assert colors[leaf_pos] == pytest.approx(expected)


def test_dendrogram_default_off_unchanged():
    rng = np.random.default_rng(2)
    pts = rng.normal(size=(8, 3))
    plt.close("all")
    result = _dendro(pts).plot_dendrogram()  # no class_labels
    assert "leaves" in result
    assert len(plt.gcf().axes) == 1  # single axis, no band
    plt.close("all")
