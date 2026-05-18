import json
from pathlib import Path
from collections import defaultdict, Counter
import math

import matplotlib.pyplot as plt
import networkx as nx


# -----------------------------------------------------------------------------
# Data loading helpers
# -----------------------------------------------------------------------------
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


# -----------------------------------------------------------------------------
# Feature / quality normalization
# -----------------------------------------------------------------------------
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


# -----------------------------------------------------------------------------
# Feature selection logic
# -----------------------------------------------------------------------------
def _normalize_text_set(values):
    if values is None:
        return None
    return {str(v).strip().lower() for v in values if str(v).strip()}


def select_top_features_for_models(
    sample_models,
    acceptable_features=None,
    excluded_features=None,
    top_x_features_per_model=None,
):
    """
    Select features per model using these rules:
    1. Only keep features from acceptable_features when that list is provided.
    2. Remove excluded_features.
    3. Rank shared features higher: a feature tied to multiple models is prioritized.
    4. Preserve original per-model order as the tie-breaker.

    Returns:
        dict[model_name] -> ordered selected feature list
    """
    acceptable_lc = _normalize_text_set(acceptable_features)
    excluded_lc = _normalize_text_set(excluded_features) or set()

    model_feature_lists = {}
    feature_presence_count = Counter()

    for model in sample_models:
        model_name = model.get("modelID", "unknown_model")
        features = normalize_features(model.get("Features", []))

        filtered = []
        seen_lc = set()
        for idx, feature in enumerate(features):
            feature_lc = feature.lower()
            if feature_lc in excluded_lc:
                continue
            if acceptable_lc is not None and feature_lc not in acceptable_lc:
                continue
            if feature_lc in seen_lc:
                continue
            seen_lc.add(feature_lc)
            filtered.append((idx, feature))

        model_feature_lists[model_name] = filtered

        for _, feature in filtered:
            feature_presence_count[feature.lower()] += 1

    selected = {}
    for model_name, indexed_features in model_feature_lists.items():
        ranked = sorted(
            indexed_features,
            key=lambda item: (
                -feature_presence_count[item[1].lower()],  # shared across models first
                item[0],  # original order next
                item[1].lower(),
            ),
        )

        if top_x_features_per_model is not None:
            ranked = ranked[:top_x_features_per_model]

        selected[model_name] = [feature for _, feature in ranked]

    return selected


# -----------------------------------------------------------------------------
# Graph build
# -----------------------------------------------------------------------------
def build_knowledge_graph(
    sample_models,
    excluded_features=None,
    max_features_per_model=None,
    acceptable_features=None,
    preselected_features_by_model=None,
):
    """
    Builds graph with:
    - clusters positioned above models
    - model -> feature
    - model -> quality

    Feature selection options:
    - acceptable_features: keep only these features (case-insensitive)
    - max_features_per_model: take top X selected features per model
    - shared features are prioritized when ranking
    - preselected_features_by_model: optional explicit precomputed selection
    """
    if excluded_features is None:
        excluded_features = set()

    excluded_features_lc = {str(x).strip().lower() for x in excluded_features}

    if preselected_features_by_model is None:
        preselected_features_by_model = select_top_features_for_models(
            sample_models,
            acceptable_features=acceptable_features,
            excluded_features=excluded_features_lc,
            top_x_features_per_model=max_features_per_model,
        )

    G = nx.Graph()

    for model in sample_models:
        model_name = model.get("modelID", "unknown_model")
        features = preselected_features_by_model.get(model_name, [])
        quality_scores = extract_quality_scores(model.get("Quality", {}))

        model_node = f"model::{model_name}"
        G.add_node(model_node, node_type="model", label=model_name)

        clusters = model.get("Clusters", {})
        if isinstance(clusters, dict):
            assigned_modality = str(clusters.get("assigned_modality", "")).strip().capitalize()
            task = str(clusters.get("task", "")).strip().capitalize()
            family_root = str(clusters.get("family_root", "")).strip().capitalize()

            modality_node = None
            task_node = None
            family_node = None

            if assigned_modality:
                modality_node = f"cluster_modality::{assigned_modality}"
                G.add_node(modality_node, node_type="cluster_modality", label=assigned_modality)

            if task:
                task_node = f"cluster_task::{task}"
                G.add_node(task_node, node_type="cluster_task", label=task)

            if family_root:
                family_node = f"cluster_family::{family_root}"
                G.add_node(family_node, node_type="cluster_family", label=family_root)

            if modality_node is not None and task_node is not None:
                G.add_edge(modality_node, task_node, relation="cluster_flow", cluster_level="modality_to_task")

            if task_node is not None and family_node is not None:
                G.add_edge(task_node, family_node, relation="cluster_flow", cluster_level="task_to_family")

            if family_node is not None:
                G.add_edge(family_node, model_node, relation="belongs_to_cluster", cluster_level="family")
            elif task_node is not None:
                G.add_edge(task_node, model_node, relation="belongs_to_cluster", cluster_level="task")
            elif modality_node is not None:
                G.add_edge(modality_node, model_node, relation="belongs_to_cluster", cluster_level="modality")

        for feature in features:
            if feature.strip().lower() in excluded_features_lc:
                continue
            feature_node = f"feature::{feature}"
            G.add_node(feature_node, node_type="feature", label=feature)
            G.add_edge(model_node, feature_node, relation="has_feature", edge_text="has")

        for quality_attr, score in quality_scores.items():
            quality_node = f"quality::{quality_attr}"
            G.add_node(quality_node, node_type="quality", label=quality_attr)
            G.add_edge(model_node, quality_node, relation="has_quality", score=score)

    return G


# -----------------------------------------------------------------------------
# Layout
# -----------------------------------------------------------------------------
def radial_cluster_layout(
    G,
    model_spacing=22,
    feature_base_radius=5.0,
    quality_base_radius=8.5,
    nodes_per_feature_ring=7,
    nodes_per_quality_ring=5,
    ring_gap=2.0,
    cluster_y_levels=None,
):
    """
    Layout:
    - clusters above models in 3 horizontal rows:
        modality (top), task (middle), family (lower top)
    - models in one row
    - features around models on inner rings
    - qualities around models on outer rings
    """
    if cluster_y_levels is None:
        cluster_y_levels = {
        "cluster_modality": 7,
        "cluster_task": 7,
        "cluster_family": 4,
    }

    pos = {}

    model_nodes = sorted(
        n for n, d in G.nodes(data=True)
        if d.get("node_type") == "model"
    )

    model_centers = {}
    for i, m in enumerate(model_nodes):
        x, y = i * model_spacing, 0
        pos[m] = (x, y)
        model_centers[m] = (x, y)

    cluster_types = ["cluster_modality", "cluster_task", "cluster_family"]

    for cluster_type in cluster_types:
        y = cluster_y_levels[cluster_type]
        cluster_nodes = sorted(
            n for n, d in G.nodes(data=True)
            if d.get("node_type") == cluster_type
        )

        # fixed horizontal offsets per cluster type
        cluster_x_offset = {
            "cluster_modality": -6,
            "cluster_task": +5,
            "cluster_family": +6,
        }

        for node in cluster_nodes:
            attached_models = [nbr for nbr in G.neighbors(node) if G.nodes[nbr].get("node_type") == "model"]
            if attached_models:
                x = sum(model_centers[m][0] for m in attached_models) / len(attached_models)
            else:
                x = 0

            # apply simple offset to avoid overlap
            x += cluster_x_offset.get(cluster_type, 0)

            pos[node] = (x, y)

        # spread overlapping cluster nodes that share the same x on the same row
        grouped = defaultdict(list)
        for node in cluster_nodes:
            grouped[round(pos[node][0], 2)].append(node)

        for _, nodes_here in grouped.items():
            if len(nodes_here) <= 1:
                continue
            nodes_here = sorted(nodes_here)
            start = -((len(nodes_here) - 1) * 3.0) / 2
            horizontal_spacing = 8.0

            start = -((len(nodes_here) - 1) * horizontal_spacing) / 2
            for i, node in enumerate(nodes_here):
                x, _ = pos[node]
                pos[node] = (x + start + i * horizontal_spacing, y)

    proposals = defaultdict(list)

    def propose(nodes, center, base_radius, nodes_per_ring, angle_offset=0.0):
        cx, cy = center

        for idx, node in enumerate(nodes):
            ring = idx // nodes_per_ring
            ring_index = idx % nodes_per_ring
            radius = base_radius + ring * ring_gap

            ring_nodes = nodes[ring * nodes_per_ring:(ring + 1) * nodes_per_ring]
            n_ring = len(ring_nodes)

            angle = angle_offset if n_ring == 1 else angle_offset + 2 * math.pi * ring_index / n_ring

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

        # Slight angle offsets keep labels away from the direct vertical cluster->model lines
        propose(features, (cx, cy), feature_base_radius, nodes_per_feature_ring, angle_offset=math.pi / 8)
        propose(qualities, (cx, cy), quality_base_radius, nodes_per_quality_ring, angle_offset=math.pi / 10)

    for node, points in proposals.items():
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        x = sum(xs) / len(xs)
        y = sum(ys) / len(ys)
        if len(points) > 1:
            y += 0.6
        pos[node] = (x, y)

    return pos


# -----------------------------------------------------------------------------
# Drawing
# -----------------------------------------------------------------------------
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
    model_size=12000,
    feature_size=4500,
    quality_size=5200,
    cluster_modality_size=6000,
    cluster_task_size=6500,
    cluster_family_size=7000,
    font_size = 18,
    edge_font_size = 14,
    figsize=(28, 16),
):
    plt.figure(figsize=figsize)

    pos = radial_cluster_layout(
    G,
    model_spacing=22,
    feature_base_radius=5.0,
    quality_base_radius=8.5,
    nodes_per_feature_ring=7,
    nodes_per_quality_ring=5,
    ring_gap=2.0,
    cluster_y_levels={
        "cluster_modality": 8,
        "cluster_family": 8,
        "cluster_task": 8,
        },
    )

    model_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "model"]
    feature_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "feature"]
    quality_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "quality"]
    cluster_modality_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "cluster_modality"]
    cluster_task_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "cluster_task"]
    cluster_family_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "cluster_family"]

    labels = {n: d.get("label", n) for n, d in G.nodes(data=True)}

    nx.draw_networkx_nodes(
        G, pos,
        nodelist=cluster_modality_nodes,
        node_color=cluster_modality_color,
        node_shape=cluster_shape,
        node_size=cluster_modality_size,
        alpha=0.98,
        linewidths=1.2,
        edgecolors="black",
    )

    nx.draw_networkx_nodes(
        G, pos,
        nodelist=cluster_task_nodes,
        node_color=cluster_task_color,
        node_shape=cluster_shape,
        node_size=cluster_task_size,
        alpha=0.98,
        linewidths=1.2,
        edgecolors="black",
    )

    nx.draw_networkx_nodes(
        G, pos,
        nodelist=cluster_family_nodes,
        node_color=cluster_family_color,
        node_shape=cluster_shape,
        node_size=cluster_family_size,
        alpha=0.98,
        linewidths=1.2,
        edgecolors="black",
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

    cluster_edges = []
    feature_edges = []
    quality_edges = []

    for u, v, d in G.edges(data=True):
        if d.get("relation") in {"belongs_to_cluster", "cluster_flow"}:
            cluster_edges.append((u, v))
        elif d.get("relation") == "has_feature":
            feature_edges.append((u, v))
        elif d.get("relation") == "has_quality":
            quality_edges.append((u, v))

    nx.draw_networkx_edges(
        G, pos,
        edgelist=cluster_edges,
        width=2.8,
        alpha=0.95,
        min_source_margin=20,
        min_target_margin=20,
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
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
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

    # -----------------------------------------------------------------
    # USER-CONTROLLED FEATURE FILTERING
    # Replace acceptable_features with your own list.
    # Only features from this list can appear on the graph.
    # Shared features across multiple instances are prioritized first.
    # Then top_x_features_per_model are kept for each instance.
    # -----------------------------------------------------------------
    acceptable_features = [
            "deepseek r1",
            "llama",
            "reasoning task",
            "large scale reinforcement learning",
            "distillation",
            "dense model",
            "language mixing",
            "cold start data",
            "openai compatible api",
            "very large language model (vllm)",
            "open sourced deepseek r1 zero",
            "4 bit format",
            "sharegpt chatml / vicuna template",
            "average weighted quantization (awq)",
            "fp8",
            "float16",
            "full context length",
            "gpu memory utilization",
            "80 gb gpus",
            "python",
            "pytorch"
    ]

    top_x_features_per_model = 10

    selected_features_by_model = select_top_features_for_models(
        sample_models,
        acceptable_features=acceptable_features if acceptable_features else None,
        excluded_features=excluded_features,
        top_x_features_per_model=top_x_features_per_model,
    )

    print("\nSelected features by model:")
    for model_name, feature_list in selected_features_by_model.items():
        print(f"- {model_name}")
        for feature in feature_list:
            print(f"    • {feature}")

    G = build_knowledge_graph(
        sample_models,
        excluded_features=excluded_features,
        max_features_per_model=top_x_features_per_model,
        acceptable_features=acceptable_features if acceptable_features else None,
        preselected_features_by_model=selected_features_by_model,
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
