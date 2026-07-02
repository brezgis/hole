"""Tests for the cluster-flow threshold selection: duplicate-threshold
robustness and the MST-based fast path.

Covers two fixes:

1. **Duplicate/tied thresholds.**  When several of the four "meaningful" stage
   picks coincide (common with tied pairwise distances), selection used to
   collapse to < 4 stages and the Sankey / stacked-bar plots would bail out with
   "Need >= 4 stages".  Selection now backfills to 4 distinct stages whenever
   that many distinct thresholds exist.

2. **MST fast path.**  Connected components of the ``distance <= t`` graph equal
   those of the MST thresholded at ``t`` for any symmetric distance matrix.  The
   analyzer now sweeps thresholds with an incremental union-find over the MST
   instead of rebuilding a dense graph per threshold.  We check the labels match
   a brute-force connected-components computation exactly.
"""

import os
import sys

import numpy as np
import pytest
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hole.visualization.cluster_flow import ClusterFlowAnalyzer  # noqa: E402


def _labels_bruteforce(D, t):
    """Reference connected-component labels of the full distance<=t graph."""
    n = D.shape[0]
    adj = (D <= t) & ~np.eye(n, dtype=bool)
    _, labels = connected_components(csr_matrix(adj), directed=False)
    return labels


def _same_partition(a, b):
    """True if two label arrays induce the same partition (ids may differ)."""
    a, b = np.asarray(a), np.asarray(b)
    return len({(x, y) for x, y in zip(a, b)}) == len(set(a.tolist())) == len(set(b.tolist()))


def _blobs(n_per=25, centers=4, dim=6, spread=0.15, seed=0):
    rng = np.random.default_rng(seed)
    cs = rng.normal(scale=6.0, size=(centers, dim))
    return np.ascontiguousarray(np.vstack([c + spread * rng.standard_normal((n_per, dim)) for c in cs]))


def _distance_matrix(x):
    from scipy.spatial.distance import squareform, pdist
    return squareform(pdist(x))


def test_mst_labels_match_bruteforce():
    """MST-thresholded labels == full-graph connected components, exactly."""
    D = _distance_matrix(_blobs())
    an = ClusterFlowAnalyzer(D)
    thresholds = an._all_merge_thresholds()
    # sample a spread of thresholds
    for t in np.percentile(thresholds, [5, 25, 50, 75, 95]):
        got = an._labels_at(t)
        ref = _labels_bruteforce(D, t)
        assert _same_partition(got, ref), f"partition mismatch at t={t:.4f}"
        assert an._n_clusters_at(t) == len(set(ref.tolist()))


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_selection_always_returns_four_distinct_when_available(seed):
    """The invariant that fixes the bug: >=4 distinct thresholds available ->
    exactly 4 distinct ascending stages selected (never fewer)."""
    D = _distance_matrix(_blobs(seed=seed))
    labels = np.repeat(np.arange(4), 25)
    an = ClusterFlowAnalyzer(D)
    all_thr = an._all_merge_thresholds()
    assert len(set(all_thr)) >= 4  # blobs give many distinct float thresholds

    selected = an._select_meaningful_thresholds(all_thr, labels)
    assert len(selected) == len(set(selected)), "selected thresholds must be distinct"
    assert len(selected) == 4, f"expected 4 stages, got {len(selected)}: {selected}"
    assert selected == sorted(selected), "stages must be ascending"


def test_backfill_when_semantic_picks_collide():
    """Force the collision that used to break selection: if the label-similar
    pick coincides with the initial pick, backfill must still yield 4 distinct
    stages (previously this dropped to 3 and the plots failed)."""
    D = _distance_matrix(_blobs(seed=7))
    labels = np.repeat(np.arange(4), 25)
    an = ClusterFlowAnalyzer(D)
    all_thr = an._all_merge_thresholds()

    # Stub the "similar to true labels" pick to collide with the initial (=min)
    # threshold, guaranteeing a duplicate among the semantic picks.
    an._find_similar_to_true_labels = lambda thr, tl: thr[0]

    selected = an._select_meaningful_thresholds(all_thr, labels)
    assert len(selected) == len(set(selected)) == 4, (
        f"backfill should recover 4 distinct stages, got {selected}"
    )
    assert selected == sorted(selected)


def test_selection_handles_few_thresholds_gracefully():
    """With < 4 distinct thresholds, return all of them (no crash, no padding)."""
    # three collinear points -> only two distinct merge heights
    pts = np.array([[0.0, 0.0], [1.0, 0.0], [3.0, 0.0]])
    D = _distance_matrix(pts)
    an = ClusterFlowAnalyzer(D)
    all_thr = an._all_merge_thresholds()
    selected = an._select_meaningful_thresholds(all_thr, None)
    assert selected == sorted(set(selected))
    assert len(selected) <= 3


def test_full_evolution_runs_and_has_distinct_stages():
    """End-to-end compute_cluster_evolution produces >= 4 keyed stages."""
    D = _distance_matrix(_blobs())
    labels = np.repeat(np.arange(4), 25)
    an = ClusterFlowAnalyzer(D)
    out = an.compute_cluster_evolution(true_labels=labels, metric_name="Euclidean")
    stages = out["labels_"]["Euclidean"]
    assert len(stages) >= 4, f"expected >=4 distinct stages, got {len(stages)}"
    # every stage labels all points
    for lab in stages.values():
        assert len(lab) == D.shape[0]


def test_duplicate_points_do_not_break_mst():
    """Exact duplicate rows (distance 0) must still merge, not vanish."""
    base = _blobs(n_per=10, centers=3)
    D = _distance_matrix(np.vstack([base, base[:5]]))  # 5 exact duplicates
    an = ClusterFlowAnalyzer(D)
    thr = an._all_merge_thresholds()
    assert len(thr) > 0
    # at the largest threshold everything is one cluster
    assert an._n_clusters_at(thr[-1]) == 1


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
