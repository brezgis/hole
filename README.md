# HOLE — extended fork (projections, cross-layer flow, faster cluster-flow)

This is a fork of [**FoxHound0x00/hole**](https://github.com/FoxHound0x00/hole) — *HOLE: Homological Observation of Latent Embeddings* (Sudhanva M. Athreya & Paul Rosen). The original README is preserved unchanged below; this fork keeps all of it and adds:

- **Cross-layer cluster evolution.** Track how latent clusters emerge, split, and merge *across a model's layers* (x-axis = depth, e.g. LLM layers `[0, 8, 16, 24, 31]`), not just through one layer's filtration — via `analyze_layer_flows` / `LayerEvolutionAnalyzer` and a depth-axis Sankey.
- **A projections module.** Unified `hole/projections.py` covering PCA / MDS / t-SNE / **UMAP** / **PHATE** (precomputed-distance aware, graceful backend fallback; PHATE recommended for NN latent spaces). `plot_dimensionality_reduction` now supports `umap`/`phate` (was pca/tsne/mds only).
- **Faster, fixed cluster-flow.** Fixes a duplicate-threshold bug that made the Sankey / stacked-bar plots bail out with "need ≥ 4 stages", and rebuilds the threshold scan as an incremental union-find sweep over the MST: **~13× faster** end-to-end (1447 ms → 112 ms at n = 500), identical clustering.
- **Visualization + packaging.** An optional color band under the dendrogram (map point class → color for large point clouds), now accepting numpy-array labels and coloring by raw class id so noise (`-1`) stays gray; Poetry → PEP 621 / uv packaging with `projections`, `hooks`, and `dev` extras.

### Quick start (this fork)
```
git clone https://github.com/brezgis/hole.git
cd hole
pip install -e .                 # core
pip install -e '.[projections]'  # adds the UMAP + PHATE projection backends
```
See `SETUP.md` for the uv-based setup and the full list of extras.

---

# HOLE: Homological Observation of Latent Embeddings

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

A Python library for topological data analysis and visualization of deep learning representations using persistent homology.

## Features

- **Persistent Homology Computation** - Compute topological features of point clouds and distance matrices
- **Multiple Distance Metrics** - Euclidean, cosine, Manhattan, Mahalanobis, and geodesic distances
- **Rich Visualizations** - Persistence diagrams, barcodes, dimensionality reduction, and cluster flow analysis
- **Easy-to-Use API** - Simple interface for both beginners and advanced users
- **Comprehensive Examples** - From basic usage to advanced topological analysis

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/FoxHound0x00/hole
cd hole

# Install with pip
pip install -e .

# Or install with poetry
poetry install
```

### Basic Usage

```python
import hole
from sklearn.datasets import make_blobs

# Generate sample data
points, labels = make_blobs(n_samples=50, centers=3, random_state=42)

# Create HOLE visualizer
viz = hole.HOLEVisualizer(point_cloud=points)

# Generate visualizations
viz.plot_persistence_diagram()
viz.plot_dimensionality_reduction(method="pca", true_labels=labels)
```

### Distance Metrics

```python
import hole

# Compute different distance matrices
euclidean_dist = hole.euclidean_distance(points)
cosine_dist = hole.cosine_distance(points)
manhattan_dist = hole.manhattan_distance(points)

# Use with visualizer
viz = hole.HOLEVisualizer(distance_matrix_input=euclidean_dist)
```

## Examples

The `examples/` directory contains several examples:

- **`core/hole_example.py`** - Basic usage showcasing core HOLE visualizations (recommended starting point)
- **`core/distance_metrics.py`** - Comprehensive analysis across different distance metrics and data structures
- **`nlp/`** - BERT NER example with HOLE topological analysis
- **`modeling/`** - Vision model fine-tuning and inference scripts

```bash
cd examples/core
python hole_example.py
```

## API Reference

### Main Classes

- **`HOLEVisualizer`** - Main interface for topological analysis and visualization
- **`MSTProcessor`** - Minimum spanning tree analysis
- **`ClusterFlowAnalyzer`** - Cluster evolution through filtration
- **`BlobVisualizer`** - Scatter plot visualizations with convex hulls
- **`PersistenceDendrogram`** - Hierarchical clustering with persistence

### Distance Functions

- `euclidean_distance(X)` - Euclidean distance matrix
- `cosine_distance(X)` - Cosine distance matrix  
- `manhattan_distance(X)` - Manhattan (L1) distance matrix
- `mahalanobis_distance(X)` - Mahalanobis distance matrix
- `geodesic_distances(X)` - Geodesic distance matrix

## Dependencies

- Python 3.8+
- NumPy
- SciPy  
- Matplotlib
- Seaborn
- Scikit-learn
- GUDHI (for persistent homology)
- NetworkX

## Development

### Running Tests

```bash
pytest tests/ -v
```

### Code Quality

The codebase follows Python best practices:
- Proper error handling and logging
- Comprehensive input validation
- Type hints and documentation
- Modular, maintainable design

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

## Citation

If you use HOLE in your research, please cite:

```bibtex
@software{hole2024,
  title={HOLE: Homological Observation of Latent Embeddings},
  author={Sudhanva M Athreya and Paul Rosen},
  year={2024},
  url={https://github.com/your-username/hole}
}
```

## Authors

- **Sudhanva M Athreya** - University of Utah
- **Paul Rosen** - University of Utah

## Acknowledgments

- Built using the GUDHI library for persistent homology computations
- Inspired by topological data analysis research in machine learning interpretability
