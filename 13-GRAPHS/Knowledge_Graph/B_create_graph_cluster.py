import json
from pathlib import Path
from collections import defaultdict
import math

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
    Builds graph with:
    - cluster chain: assigned_modality -> task -> family_root -> model
    - model -> feature
    - model -> quality
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

        features = [f for f in features if f.strip().lower() not in excluded_features]

        model_node = f"model::{model_name}"
        G.add_node(model_node, node_type="model", label=model_name)

        # ---------------------------
        # Cluster chain
        # Clusters -> assigned_modality -> task -> family_root -> model
        # ---------------------------
        clusters = model.get("Clusters", {})
        if isinstance(clusters, dict):
            assigned_modality = str(clusters.get("assigned_modality", "")).strip()
            task = str(clusters.get("task", "")).strip()
            family_root = str(clusters.get("family_root", "")).strip()

            prev_node = None

            if assigned_modality:
                modality_node = f"cluster_modality::{assigned_modality}"
                G.add_node(
                    modality_node,
                    node_type="cluster_modality",
                    label=assigned_modality
                )
                prev_node = modality_node

            if task:
                task_node = f"cluster_task::{task}"
                G.add_node(
                    task_node,
                    node_type="cluster_task",
                    label=task
                )
                if prev_node is not None:
                    G.add_edge(prev_node, task_node, relation="cluster_flow")
                prev_node = task_node

            if family_root:
                family_node = f"cluster_family::{family_root}"
                G.add_node(
                    family_node,
                    node_type="cluster_family",
                    label=family_root
                )
                if prev_node is not None:
                    G.add_edge(prev_node, family_node, relation="cluster_flow")
                prev_node = family_node

            if prev_node is not None:
                G.add_edge(prev_node, model_node, relation="belongs_to_cluster")

        # ---------------------------
        # Features
        # ---------------------------
        for feature in features:
            feature_node = f"feature::{feature}"
            G.add_node(feature_node, node_type="feature", label=feature)
            G.add_edge(model_node, feature_node, relation="has_feature", edge_text="has")

        # ---------------------------
        # Qualities
        # ---------------------------
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


def radial_cluster_layout(
    G,
    model_spacing=22,
    feature_base_radius=5.0,
    quality_base_radius=8.0,
    nodes_per_feature_ring=7,
    nodes_per_quality_ring=5,
    ring_gap=2.0,
):
    """
    Layout:
    - cluster chain on the left:
        modality -> task -> family -> model
    - models spaced horizontally
    - features around models on inner rings
    - qualities around models on outer rings
    """

    pos = {}

    model_nodes = sorted(
        n for n, d in G.nodes(data=True)
        if d.get("node_type") == "model"
    )
    modality_nodes = sorted(
        n for n, d in G.nodes(data=True)
        if d.get("node_type") == "cluster_modality"
    )
    task_nodes = sorted(
        n for n, d in G.nodes(data=True)
        if d.get("node_type") == "cluster_task"
    )
    family_nodes = sorted(
        n for n, d in G.nodes(data=True)
        if d.get("node_type") == "cluster_family"
    )

    # Place models
    model_centers = {}
    for i, m in enumerate(model_nodes):
        x, y = i * model_spacing, 0
        pos[m] = (x, y)
        model_centers[m] = (x, y)

    # Initial vertical placement for cluster columns
    def place_vertical(nodes, x, y_gap=4.0):
        if not nodes:
            return
        start = (len(nodes) - 1) * y_gap / 2
        for i, node in enumerate(nodes):
            pos[node] = (x, start - i * y_gap)

    place_vertical(modality_nodes, x=-30, y_gap=6.0)
    place_vertical(task_nodes, x=-22, y_gap=6.0)
    place_vertical(family_nodes, x=-14, y_gap=6.0)

    # Refine y positions by averaging neighbor y positions
    def average_y(neighbors):
        ys = [pos[n][1] for n in neighbors if n in pos]
        return sum(ys) / len(ys) if ys else 0

    for node in task_nodes:
        neigh = list(G.neighbors(node))
        pos[node] = (-22, average_y(neigh))

    for node in family_nodes:
        neigh = list(G.neighbors(node))
        pos[node] = (-14, average_y(neigh))

    for node in modality_nodes:
        neigh = list(G.neighbors(node))
        pos[node] = (-30, average_y(neigh))

    # Radial proposals for features and qualities
    proposals = defaultdict(list)

    def propose(nodes, center, base_radius, nodes_per_ring):
        cx, cy = center

        for idx, node in enumerate(nodes):
            ring = idx // nodes_per_ring
            ring_index = idx % nodes_per_ring
            radius = base_radius + ring * ring_gap

            ring_nodes = nodes[ring * nodes_per_ring:(ring + 1) * nodes_per_ring]
            n_ring = len(ring_nodes)

            angle = 0 if n_ring == 1 else 2 * math.pi * ring_index / n_ring

            x = cx + radius * math.cos(angle)
            y = cy + radius * math.sin(angle)
            proposals[node].append((x, y))

    for model in model_nodes:
        cx, cy = model_centers[model]

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

    for node, points in proposals.items():
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        x = sum(xs) / len(xs)
        y = sum(ys) / len(ys)
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
    cluster_modality_color="#C7A0E8",
    cluster_task_color="#F3C567",
    cluster_family_color="#9ED9D8",
    model_shape="o",
    feature_shape="s",
    quality_shape="D",
    cluster_shape="o",
    model_size=7000,              # was 5200
    feature_size=2600,            # was 1800
    quality_size=3000,            # was 2200
    cluster_modality_size=3200,   # was 2600
    cluster_task_size=3400,       # was 2800
    cluster_family_size=3600,     # was 3000
    font_size=12,        # was 9
    edge_font_size=10   # was 8
):
    plt.figure(figsize=(28, 14))

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
    cluster_modality_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "cluster_modality"]
    cluster_task_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "cluster_task"]
    cluster_family_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "cluster_family"]

    labels = {n: d.get("label", n) for n, d in G.nodes(data=True)}

    # Draw nodes
    nx.draw_networkx_nodes(
        G, pos,
        nodelist=cluster_modality_nodes,
        node_color=cluster_modality_color,
        node_shape=cluster_shape,
        node_size=cluster_modality_size,
        alpha=0.95
    )

    nx.draw_networkx_nodes(
        G, pos,
        nodelist=cluster_task_nodes,
        node_color=cluster_task_color,
        node_shape=cluster_shape,
        node_size=cluster_task_size,
        alpha=0.95
    )

    nx.draw_networkx_nodes(
        G, pos,
        nodelist=cluster_family_nodes,
        node_color=cluster_family_color,
        node_shape=cluster_shape,
        node_size=cluster_family_size,
        alpha=0.95
    )

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
    cluster_edges = []
    feature_edges = []
    quality_edges = []

    for u, v, d in G.edges(data=True):
        if d.get("relation") in {"cluster_flow", "belongs_to_cluster"}:
            cluster_edges.append((u, v))
        elif d.get("relation") == "has_feature":
            feature_edges.append((u, v))
        elif d.get("relation") == "has_quality":
            quality_edges.append((u, v))

    nx.draw_networkx_edges(
        G, pos,
        edgelist=cluster_edges,
        width=2.2,
        alpha=0.85
    )

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

    # Only show quality scores as edge labels
    edge_labels = {}
    for u, v, d in G.edges(data=True):
        if d.get("relation") == "has_quality":
            edge_labels[(u, v)] = str(d.get("score"))

    nx.draw_networkx_edge_labels(
        G, pos,
        edge_labels=edge_labels,
        font_size=edge_font_size
    )

    plt.axis("off")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


if __name__ == "__main__":
    folder_path = "HF-Models-T7-U"

    all_models = load_json_files(folder_path)
    print("Loaded models:", len(all_models))

    wanted_files = {
        "deepseek-ai__DeepSeek-R1.json",
        "unsloth__DeepSeek-R1-GGUF.json",
        "cognitivecomputations__DeepSeek-R1-AWQ.json",
    }

    sample_models = [
        m for m in all_models
        if Path(m.get("source_file", "")).name in wanted_files
    ]

    print("\nSelected models:")
    for m in sample_models:
        print("-", m.get("modelID"))

        clusters = m.get("Clusters", {})
        if isinstance(clusters, dict):
            print("  assigned_modality:", clusters.get("assigned_modality"))
            print("  task:", clusters.get("task"))
            print("  family_root:", clusters.get("family_root"))

    excluded_features = {
        "12 billion parameter rectified flow transformer",
        "personal identifiable information",
        "annen",
    }

    G = build_knowledge_graph(
        sample_models,
        excluded_features=excluded_features,
        max_features_per_model=10,
    )

    draw_knowledge_graph(
        G,
        save_path="13-GRAPHS/Knowledge_Graph/sample_model_knowledge_graph_cluster.png",
        model_color="#5795D7",
        feature_color="#3E8D33",
        quality_color="#EB797B",
        cluster_modality_color="#C7A0E8",
        cluster_task_color="#F3C567",
        cluster_family_color="#9ED9D8",
        model_shape="o",
        feature_shape="s",
        quality_shape="D",
        cluster_shape="o",
    )
    