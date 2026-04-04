import json
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx


def load_json_files(folder_path):
    folder = Path(folder_path)
    models = []

    for file in folder.rglob("*.json"):
        try:
            with open(file, "r", encoding="utf-8") as f:
                data = json.load(f)

            if not isinstance(data, dict):
                continue

            data["source_file"] = str(file)
            models.append(data)

        except Exception as e:
            print(f"Skipping {file}: {e}")

    return models


def normalize_features(features):
    if not isinstance(features, list):
        return []

    out = []
    seen = set()

    for x in features:
        x = str(x).strip()
        if x and x not in seen:
            seen.add(x)
            out.append(x)

    return out


def extract_quality_scores(quality_dict):
    result = {}

    if not isinstance(quality_dict, dict):
        return result

    for attr, value in quality_dict.items():
        if not isinstance(value, dict):
            continue

        score = None
        for k in ["score", "value", "fuzzy_score"]:
            if k in value:
                score = value[k]
                break

        if score is None and len(value) == 1:
            score = next(iter(value.values()))

        result[str(attr).strip()] = score

    return result


def build_knowledge_graph(sample_models, excluded_features=None, max_features_per_model=None):
    """
    excluded_features: set/list of feature names to remove from the graph
    max_features_per_model: optional cap to reduce clutter
    """
    if excluded_features is None:
        excluded_features = set()
    excluded_features = {str(x).strip().lower() for x in excluded_features}

    G = nx.Graph()

    for model in sample_models:
        model_name = model.get("modelID", "unknown_model")
        features = normalize_features(model.get("Features", []))
        quality_scores = extract_quality_scores(model.get("Quality", {}))

        if max_features_per_model is not None:
            features = features[:max_features_per_model]

        # remove unwanted features
        features = [f for f in features if f.strip().lower() not in excluded_features]

        model_node = f"model::{model_name}"
        G.add_node(model_node, node_type="model", label=model_name)

        for feature in features:
            feature_node = f"feature::{feature}"
            G.add_node(feature_node, node_type="feature", label=feature)
            G.add_edge(model_node, feature_node, relation="has_feature", edge_text="has")

        for quality_attr, score in quality_scores.items():
            quality_node = f"quality::{quality_attr}"
            G.add_node(quality_node, node_type="quality", label=quality_attr)
            G.add_edge(
                model_node,
                quality_node,
                relation="has_quality",
                score=score
            )

    return G


def make_layered_positions(G, model_spacing=8.0, vertical_spacing=1.8):
    """
    Manual layout:
    - models in the center, spaced far apart vertically
    - features on the left
    - qualities on the right
    """
    pos = {}

    model_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "model"]
    feature_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "feature"]
    quality_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "quality"]

    # Center models, spaced out more
    if len(model_nodes) == 1:
        model_y = [0]
    else:
        start = (len(model_nodes) - 1) * model_spacing / 2
        model_y = [start - i * model_spacing for i in range(len(model_nodes))]

    for node, y in zip(model_nodes, model_y):
        pos[node] = (0, y)

    # Features on the left
    feature_start = (len(feature_nodes) - 1) * vertical_spacing / 2 if feature_nodes else 0
    for i, node in enumerate(sorted(feature_nodes)):
        y = feature_start - i * vertical_spacing
        pos[node] = (-6, y)

    # Qualities on the right
    quality_start = (len(quality_nodes) - 1) * vertical_spacing / 2 if quality_nodes else 0
    for i, node in enumerate(sorted(quality_nodes)):
        y = quality_start - i * vertical_spacing
        pos[node] = (6, y)

    return pos

import math
import random

import math

import math
from collections import defaultdict

def radial_cluster_layout(
    G,
    model_spacing=20,
    feature_base_radius=5.5,
    quality_base_radius=8.5,
    nodes_per_feature_ring=6,
    nodes_per_quality_ring=5,
    ring_gap=2.2,
):
    """
    Radial layout with:
    - models spaced apart
    - features on inner rings
    - qualities on outer rings
    - shared nodes positioned via centroid instead of averaging
    """

    model_nodes = sorted(
        n for n, d in G.nodes(data=True)
        if d.get("node_type") == "model"
    )

    # model centers
    pos = {m: (i * model_spacing, 0) for i, m in enumerate(model_nodes)}

    # store candidate positions
    proposals = defaultdict(list)

    def propose(nodes, center, base_radius, nodes_per_ring):

        cx, cy = center

        for idx, node in enumerate(nodes):

            ring = idx // nodes_per_ring
            ring_index = idx % nodes_per_ring

            radius = base_radius + ring * ring_gap

            ring_nodes = nodes[ring*nodes_per_ring:(ring+1)*nodes_per_ring]
            n_ring = len(ring_nodes)

            if n_ring == 1:
                angle = 0
            else:
                angle = 2 * math.pi * ring_index / n_ring

            x = cx + radius * math.cos(angle)
            y = cy + radius * math.sin(angle)

            proposals[node].append((x, y))

    for model in model_nodes:

        cx, cy = pos[model]

        features = sorted(
            n for n in G.neighbors(model)
            if G.nodes[n].get("node_type") == "feature"
        )

        qualities = sorted(
            n for n in G.neighbors(model)
            if G.nodes[n].get("node_type") == "quality"
        )

        propose(features, (cx, cy), feature_base_radius, nodes_per_feature_ring)
        propose(qualities, (cx, cy), quality_base_radius, nodes_per_quality_ring)

    # resolve final positions
    for node, points in proposals.items():

        xs = [p[0] for p in points]
        ys = [p[1] for p in points]

        x = sum(xs) / len(xs)
        y = sum(ys) / len(ys)

        # small vertical offset for shared nodes
        if len(points) > 1:
            y += 0.5

        pos[node] = (x, y)

    return pos
def draw_knowledge_graph(
    G,
    save_path=None,
    model_color="#5DA5DA",
    feature_color="#60BD68",
    quality_color="#F17CB0",
    model_shape="o",
    feature_shape="s",
    quality_shape="D",
    model_size=5200,
    feature_size=1800,
    quality_size=2200,
    font_size=9,
    edge_font_size=8,
    model_spacing=8.0,
    vertical_spacing=1.8,
):
    plt.figure(figsize=(24, 12))

    pos = radial_cluster_layout(
    G,
    model_spacing=22,
    feature_base_radius=5.0,
    quality_base_radius=8.0,
    nodes_per_feature_ring=7,
    nodes_per_quality_ring=5,
    ring_gap=2.0,
)

    model_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "model"]
    feature_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "feature"]
    quality_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "quality"]

    labels = {n: d.get("label", n) for n, d in G.nodes(data=True)}

    # Draw nodes
    nx.draw_networkx_nodes(
        G, pos,
        nodelist=model_nodes,
        node_color=model_color,
        node_shape=model_shape,
        node_size=model_size,
        alpha=0.95
    )

    nx.draw_networkx_nodes(
        G, pos,
        nodelist=feature_nodes,
        node_color=feature_color,
        node_shape=feature_shape,
        node_size=feature_size,
        alpha=0.9
    )

    nx.draw_networkx_nodes(
        G, pos,
        nodelist=quality_nodes,
        node_color=quality_color,
        node_shape=quality_shape,
        node_size=quality_size,
        alpha=0.9
    )

    # Split edges
    feature_edges = []
    quality_edges = []

    for u, v, d in G.edges(data=True):
        if d.get("relation") == "has_feature":
            feature_edges.append((u, v))
        elif d.get("relation") == "has_quality":
            quality_edges.append((u, v))

    nx.draw_networkx_edges(
        G, pos,
        edgelist=feature_edges,
        width=1.6,
        alpha=0.7
    )

    nx.draw_networkx_edges(
        G, pos,
        edgelist=quality_edges,
        width=1.8,
        alpha=0.8,
        style="dashed"
    )

    nx.draw_networkx_labels(
        G, pos,
        labels=labels,
        font_size=font_size
    )

    # Feature edges get "has"
    edge_labels = {}
    for u, v, d in G.edges(data=True):
        if d.get("relation") == "has_feature":
            edge_labels[(u, v)] = "has"
        elif d.get("relation") == "has_quality":
            edge_labels[(u, v)] = str(d.get("score"))

    nx.draw_networkx_edge_labels(
        G, pos,
        edge_labels=edge_labels,
        font_size=edge_font_size
    )

    # No title, no legend
    plt.axis("off")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    folder_path = "HF-Models-T7-U"

    all_models = load_json_files(folder_path)
    print("Loaded models:", len(all_models))

    wanted_files = {
        "lllyasviel__ControlNet.json",
        "black-forest-labs__FLUX.1-dev.json",
        "Lightricks__LTX-Video.json",
    }

    sample_models = [
        m for m in all_models
        if Path(m.get("source_file", "")).name in wanted_files
    ]

    print("\nSelected models:")
    for m in sample_models:
        print("-", m.get("modelID"))

    excluded_features = {
        "12 billion parameter rectified flow transformer",
        "personal identifiable information",
        "annen",
    }

    G = build_knowledge_graph(
        sample_models,
        excluded_features=excluded_features,
        max_features_per_model=20,
    )

    draw_knowledge_graph(
        G,
        save_path="13-GRAPHS/Knowledge_Graph/sample_model_knowledge_graph.png",
        model_spacing=22.0,      # increase if model names still too close
        vertical_spacing=1.5,
        model_color="#5795D7",
        feature_color="#3E8D33",
        quality_color="#EB797B",
        model_shape="o",
        feature_shape="s",
        quality_shape="D",
    )
