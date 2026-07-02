"""Unified low-dimensional projection for topological-data-analysis pipelines.

A single entry point -- :func:`project` -- maps a high-dimensional point cloud
(or a precomputed pairwise-distance matrix) down to a handful of coordinates for
plotting or downstream geometry, using any of:

    - ``"pca"``    principal component analysis            (scikit-learn, always available)
    - ``"mds"``    (metric) multidimensional scaling        (scikit-learn, always available)
    - ``"tsne"``   t-distributed stochastic neighbor embed. (openTSNE if present, else scikit-learn)
    - ``"umap"``   uniform manifold approximation           (``pip install umap-learn``)
    - ``"phate"``  potential of heat-diffusion affinity     (``pip install phate``)

PHATE is the recommended default for neural-network latent spaces: it preserves
both local and global manifold structure and tends to reveal continuous
trajectories / branches that t-SNE and UMAP fragment.  See Moon et al.,
"Visualizing structure and transitions in high-dimensional biological data",
Nature Biotechnology 2019.

The module has only two hard dependencies (numpy + scikit-learn); ``umap``,
``phate`` and ``openTSNE`` are optional and only imported when their method is
requested, so importing this file never fails on a minimal install.  Ask a
method that is not installed and you get an :class:`ImportError` that names the
exact ``pip`` package to install.

Every method accepts either a feature matrix ``(n_samples, n_features)`` or a
precomputed square distance matrix (set ``precomputed=True`` or leave it at the
default ``"auto"`` and let the symmetry heuristic decide).  This matters for the
effective-resistance / spectral distances used elsewhere in these tools, which
are naturally produced as ``(n, n)`` matrices rather than coordinates.

Example
-------
>>> from projections import project
>>> emb2d = project(hidden_states, method="phate", metric="cosine")
>>> emb2d.shape
(n_tokens, 2)
"""

from __future__ import annotations

import warnings
from typing import Optional, Sequence, Union

import numpy as np

__all__ = [
    "project",
    "available_methods",
    "METHODS",
    "RECOMMENDED_METHOD",
    "ProjectionResult",
]

# The canonical method names this module understands.  PCA and MDS ship with
# scikit-learn so they are always usable; the rest are optional.
METHODS = ("pca", "mds", "tsne", "umap", "phate")

# What we suggest for LLM / NN latent spaces.  PHATE first, per the diffusion
# geometry argument above.
RECOMMENDED_METHOD = "phate"

# pip package that provides each optional backend, used to build helpful errors.
_PIP_PACKAGE = {
    "umap": "umap-learn",
    "phate": "phate",
    "tsne": "openTSNE (optional; scikit-learn is used as a fallback)",
}


class ProjectionResult(np.ndarray):
    """A projected embedding that also carries the fitted reducer.

    Behaves exactly like the ``(n_samples, n_components)`` ndarray it wraps, so
    existing code (``result[:, 0]``, ``result.shape``, ...) is unaffected, but
    ``result.reducer`` and ``result.method`` are available for callers that want
    to reuse the transform or read out method-specific attributes (e.g. PCA's
    ``explained_variance_ratio_``).
    """

    def __new__(cls, array, method=None, reducer=None):
        obj = np.asarray(array).view(cls)
        obj.method = method
        obj.reducer = reducer
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.method = getattr(obj, "method", None)
        self.reducer = getattr(obj, "reducer", None)


def available_methods() -> dict:
    """Return ``{method_name: bool}`` for whether each backend can run here.

    Handy for building UI dropdowns or skipping methods in a sweep without
    triggering an :class:`ImportError`.
    """
    status = {}
    for m in METHODS:
        try:
            _resolve_backend(m)
            status[m] = True
        except ImportError:
            status[m] = False
    return status


def _resolve_backend(method: str):
    """Import and return whatever object the method needs, or raise ImportError.

    Returns an opaque marker per method; the real construction happens in
    :func:`project`.  This exists so :func:`available_methods` can probe support
    cheaply and uniformly.
    """
    method = method.lower()
    if method in ("pca", "mds", "tsne"):
        # scikit-learn is a hard dependency, so these always resolve.  For tsne
        # we *prefer* openTSNE but do not require it, hence it lives here.
        import sklearn  # noqa: F401

        return method
    if method == "umap":
        try:
            import umap  # noqa: F401
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError(
                "The 'umap' projection needs the umap-learn package. "
                "Install it with:  pip install umap-learn"
            ) from exc
        return method
    if method == "phate":
        try:
            import phate  # noqa: F401
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError(
                "The 'phate' projection needs the phate package. "
                "Install it with:  pip install phate"
            ) from exc
        return method
    raise ValueError(
        f"Unknown projection method {method!r}. Choose one of {METHODS}."
    )


def _as_2d_float(x) -> np.ndarray:
    """Coerce torch tensors / lists / arrays to a contiguous float64 2D array.

    Mirrors the embedding-ingestion helper in the eff-ph integration layer so
    hidden states straight off a GPU model (CUDA / bf16 / fp16, which do not
    convert to numpy directly) are handled the same way everywhere.
    """
    if hasattr(x, "detach") and hasattr(x, "cpu"):  # torch tensor
        try:
            import torch

            x = x.detach().to("cpu", torch.float32).numpy()
        except Exception:
            x = x.detach().cpu().float().numpy()

    x = np.ascontiguousarray(np.asarray(x, dtype="float64"))
    if x.ndim != 2:
        raise ValueError(
            f"Expected a 2D array (n_samples, n_features) or (n, n) distance "
            f"matrix; got shape {x.shape}."
        )
    return x


def _looks_like_distance_matrix(x: np.ndarray) -> bool:
    """Square, symmetric, zero (or near-zero) diagonal => treat as distances."""
    n, m = x.shape
    if n != m or n < 2:
        return False
    if not np.allclose(x, x.T, atol=1e-8, rtol=1e-5):
        return False
    # A feature matrix that happens to be square and symmetric (rare) usually
    # has a nonzero diagonal; a distance matrix has a ~zero diagonal.
    return bool(np.allclose(np.diag(x), 0.0, atol=1e-8))


def _safe_perplexity(n_samples: int, requested: Optional[float]) -> float:
    """t-SNE perplexity must be < n_samples; clamp to a sane range."""
    hi = max(2.0, (n_samples - 1) / 3.0)
    base = 30.0 if requested is None else float(requested)
    return float(min(base, hi, max(5.0, hi)) if requested is None else min(base, hi))


def project(
    X,
    method: str = RECOMMENDED_METHOD,
    n_components: int = 2,
    metric: str = "euclidean",
    precomputed: Union[bool, str] = "auto",
    random_state: Optional[int] = 42,
    return_reducer: bool = False,
    **kwargs,
) -> ProjectionResult:
    """Project ``X`` to ``n_components`` dimensions with the chosen method.

    Parameters
    ----------
    X : array-like or torch.Tensor
        Either a feature matrix ``(n_samples, n_features)`` or a precomputed
        pairwise-distance matrix ``(n_samples, n_samples)``.
    method : {"pca", "mds", "tsne", "umap", "phate"}
        Projection algorithm. Defaults to ``"phate"`` -- the recommended choice
        for NN/LLM latent spaces.
    n_components : int
        Output dimensionality (2 for plotting, 3 for interactive views, more for
        downstream geometry).
    metric : str
        Distance metric used to build neighbourhoods from a *feature* matrix
        (e.g. ``"euclidean"``, ``"cosine"``, ``"correlation"``). Cosine is often
        the right choice for high-dimensional contextual embeddings. Ignored when
        the input is already a distance matrix.
    precomputed : bool or "auto"
        Whether ``X`` is a distance matrix. ``"auto"`` (default) inspects ``X``:
        square + symmetric + zero diagonal is treated as precomputed distances.
    random_state : int or None
        Seed for the stochastic methods (t-SNE, UMAP, PHATE, MDS init).
    return_reducer : bool
        If True, also return the fitted reducer object as a second value.
    **kwargs
        Passed straight through to the underlying estimator, so any
        method-specific knob (PHATE ``knn``/``decay``/``t``, UMAP
        ``n_neighbors``/``min_dist``, t-SNE ``perplexity``, ...) is reachable.

    Returns
    -------
    ProjectionResult
        An ``(n_samples, n_components)`` array subclass; ``.reducer`` and
        ``.method`` attributes carry the fitted estimator and method name. If
        ``return_reducer=True``, returns ``(embedding, reducer)`` instead.
    """
    method = method.lower()
    _resolve_backend(method)  # fail fast with a helpful message if unavailable

    X = _as_2d_float(X)

    if precomputed == "auto":
        is_precomputed = _looks_like_distance_matrix(X)
    else:
        is_precomputed = bool(precomputed)
        if is_precomputed and X.shape[0] != X.shape[1]:
            raise ValueError(
                f"precomputed=True but X is not square: shape {X.shape}."
            )

    n_samples = X.shape[0]
    if n_components >= n_samples:
        raise ValueError(
            f"n_components ({n_components}) must be < n_samples ({n_samples})."
        )

    dispatch = {
        "pca": _project_pca,
        "mds": _project_mds,
        "tsne": _project_tsne,
        "umap": _project_umap,
        "phate": _project_phate,
    }
    embedding, reducer = dispatch[method](
        X,
        n_components=n_components,
        metric=metric,
        is_precomputed=is_precomputed,
        random_state=random_state,
        **kwargs,
    )

    result = ProjectionResult(
        np.ascontiguousarray(embedding, dtype="float64"),
        method=method,
        reducer=reducer,
    )
    if return_reducer:
        return result, reducer
    return result


# --------------------------------------------------------------------------- #
# Per-method implementations. Each returns (embedding, fitted_reducer).
# --------------------------------------------------------------------------- #
def _project_pca(X, n_components, metric, is_precomputed, random_state, **kwargs):
    from sklearn.decomposition import PCA

    if is_precomputed:
        # PCA needs coordinates, not distances. Classical (metric) MDS recovers a
        # coordinate embedding from the distance matrix, which is the honest
        # analogue; warn so the caller knows the swap happened.
        warnings.warn(
            "method='pca' was given a precomputed distance matrix; PCA needs "
            "coordinates, so classical MDS is used to embed the distances "
            "instead.",
            stacklevel=3,
        )
        return _project_mds(
            X, n_components, metric, is_precomputed, random_state, **kwargs
        )

    reducer = PCA(n_components=n_components, random_state=random_state, **kwargs)
    return reducer.fit_transform(X), reducer


def _project_mds(X, n_components, metric, is_precomputed, random_state, **kwargs):
    from sklearn.manifold import MDS
    from sklearn.metrics import pairwise_distances

    n_init = kwargs.pop("n_init", 4)
    max_iter = kwargs.pop("max_iter", 300)
    if is_precomputed:
        reducer = MDS(
            n_components=n_components,
            dissimilarity="precomputed",
            random_state=random_state,
            n_init=n_init,
            max_iter=max_iter,
            normalized_stress="auto",
            **kwargs,
        )
        return reducer.fit_transform(X), reducer

    # For a feature matrix with a non-euclidean metric, precompute distances so
    # the metric is actually honoured (sklearn MDS only knows euclidean natively).
    if metric not in ("euclidean", "l2"):
        d = pairwise_distances(X, metric=metric)
        reducer = MDS(
            n_components=n_components,
            dissimilarity="precomputed",
            random_state=random_state,
            n_init=n_init,
            max_iter=max_iter,
            normalized_stress="auto",
            **kwargs,
        )
        return reducer.fit_transform(d), reducer

    reducer = MDS(
        n_components=n_components,
        random_state=random_state,
        n_init=n_init,
        max_iter=max_iter,
        normalized_stress="auto",
        **kwargs,
    )
    return reducer.fit_transform(X), reducer


def _project_tsne(X, n_components, metric, is_precomputed, random_state, **kwargs):
    """t-SNE via openTSNE when available (faster, precomputed-affinity aware),
    otherwise scikit-learn's TSNE."""
    perplexity = _safe_perplexity(X.shape[0], kwargs.pop("perplexity", None))

    # Prefer openTSNE if installed.
    try:
        from openTSNE import TSNE as OpenTSNE

        metric_arg = "precomputed" if is_precomputed else metric
        reducer = OpenTSNE(
            n_components=n_components,
            perplexity=perplexity,
            metric=metric_arg,
            random_state=random_state,
            **kwargs,
        )
        embedding = reducer.fit(X)
        return np.asarray(embedding), reducer
    except ImportError:
        pass

    from sklearn.manifold import TSNE

    if is_precomputed:
        reducer = TSNE(
            n_components=n_components,
            perplexity=perplexity,
            metric="precomputed",
            init="random",  # 'pca' init is invalid for precomputed distances
            random_state=random_state,
            **kwargs,
        )
    else:
        reducer = TSNE(
            n_components=n_components,
            perplexity=perplexity,
            metric=metric,
            random_state=random_state,
            **kwargs,
        )
    return reducer.fit_transform(X), reducer


def _project_umap(X, n_components, metric, is_precomputed, random_state, **kwargs):
    import umap

    metric_arg = "precomputed" if is_precomputed else metric
    # UMAP's n_neighbors must be < n_samples; clamp the default gracefully.
    n_neighbors = kwargs.pop("n_neighbors", 15)
    n_neighbors = int(min(n_neighbors, max(2, X.shape[0] - 1)))
    reducer = umap.UMAP(
        n_components=n_components,
        metric=metric_arg,
        n_neighbors=n_neighbors,
        random_state=random_state,
        **kwargs,
    )
    return reducer.fit_transform(X), reducer


def _project_phate(X, n_components, metric, is_precomputed, random_state, **kwargs):
    import phate

    # PHATE reads the input mode from knn_dist: a metric name for feature
    # matrices, or 'precomputed_distance' for an (n, n) distance matrix.
    if is_precomputed:
        knn_dist = "precomputed_distance"
    else:
        knn_dist = kwargs.pop("knn_dist", metric)

    # knn must be < n_samples; PHATE errors otherwise on tiny inputs.
    knn = kwargs.pop("knn", 5)
    knn = int(min(knn, max(2, X.shape[0] - 2)))

    verbose = kwargs.pop("verbose", 0)
    reducer = phate.PHATE(
        n_components=n_components,
        knn=knn,
        knn_dist=knn_dist,
        random_state=random_state,
        verbose=verbose,
        **kwargs,
    )
    return reducer.fit_transform(X), reducer
