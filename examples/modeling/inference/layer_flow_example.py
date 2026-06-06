"""
Cross-layer cluster-evolution example.

Shows how class structure in a model's latent space emerges, splits, and merges
as you move *through the layers* (x-axis = model depth), rather than through the
filtration of a single layer.

The inputs are aligned per-layer embeddings: a dict ``{f"layer_{i}": (N, D)}``
where row ``j`` is the same example at every layer (e.g. produced by your
embedding extractor with mean- or last-token pooling). Here we synthesise such a
dict where the classes start tangled and separate with depth.

Run:
    python examples/modeling/inference/layer_flow_example.py
"""

import numpy as np

import hole
from hole import analyze_layer_flows

hole.configure_logging("INFO")


def make_synthetic_layers(n_per_class=40, n_classes=4, dim=32, n_layers=12, seed=0):
    """Aligned embeddings whose class separation grows linearly with depth."""
    rng = np.random.default_rng(seed)
    labels = np.repeat(np.arange(n_classes), n_per_class)
    centers = rng.normal(size=(n_classes, dim)) * 6.0
    fixed_noise = rng.normal(size=(labels.size, dim)) * 2.0  # per-point => identity kept

    embeddings = {}
    for li, sep in enumerate(np.linspace(0.0, 1.0, n_layers)):
        embeddings[f"layer_{li}"] = (sep * centers[labels] + fixed_noise).astype(
            np.float32
        )
    return embeddings, labels


def main():
    embeddings, labels = make_synthetic_layers()

    # Pick the layers you want as stages -- the same pattern as a per-model config:
    #   for model in models: for metric in metrics: analyze_layer_flows(..., layers=LAYERS)
    LAYERS = [0, 3, 6, 9, 11]

    results = analyze_layer_flows(
        embeddings,
        output_dir="layer_flow_outputs",
        true_labels=labels,
        layers=LAYERS,
        model_name="toy_llm",
        condition_name="clean",
        metrics=("Euclidean", "Cosine"),
        # n_clusters_per_layer defaults to the number of true classes
        class_names={0: "class_0", 1: "class_1", 2: "class_2", 3: "class_3"},
    )

    for metric, r in results.items():
        print(f"[{metric}] sankey -> {r['sankey_path']}")
        print(f"[{metric}] bars   -> {r['bars_path']}")
        print(f"[{metric}] clusters/layer: {r['cluster_evolution']['components_'][metric]}")


if __name__ == "__main__":
    main()
