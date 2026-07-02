# Setup with uv

This fork migrated from Poetry to a PEP 621 `pyproject.toml`, so
[uv](https://docs.astral.sh/uv/) manages the environment directly (no more
Poetry dependency-resolution pain).

```bash
uv venv                                   # create .venv (Python >=3.9)
source .venv/bin/activate
uv pip install -e '.[projections]'        # HOLE + PCA/UMAP/t-SNE/PHATE
```

Extras:

| extra          | what it adds                                                          |
|----------------|----------------------------------------------------------------------|
| `projections`  | PHATE + UMAP + openTSNE (PCA/MDS/t-SNE already come with the base)    |
| `hooks`        | torch/transformers/timm/hydra for extracting model activations       |
| `dev`          | pytest, black, isort, flake8, mypy, pre-commit                       |

`plot_dimensionality_reduction(..., method="phate")` (and `"umap"`, `"tsne"`,
`"pca"`, `"mds"`) route through `hole.projections`; methods whose backend is not
installed raise a clear `ImportError` naming the pip package.

## Quick check

```bash
uv run python -m pytest tests/test_cluster_flow_thresholds.py -q
```
