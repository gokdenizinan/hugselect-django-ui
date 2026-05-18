#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Level 1 domain assignment for foundation models
-----------------------------------------------

Goal:
- Assign each model to a broad Level 1 class such as:
  text, image, audio, video, multimodal, tabular, time-series, graph
- No GPT calls
- Uses model description + metadata
- Supports sentence-transformers embeddings, with TF-IDF fallback

Implemented behavior:
1. Per-item score normalization across domains before rule blending
2. Toggleable modality routing
3. Modality routing uses pipeline_tag only
4. Modality routing acts only as a tie-breaker when top-1 vs top-2 confidence is small
5. Added explanation outputs: matched keywords and boost sources
6. Removed all secondary-label handling

Expected input CSV columns (flexible, only model_id/name + description are strongly recommended):
- model_id
- model_name
- description / short_description
- tags
- pipeline_tag
- base_model
- library_name
- task
- modality

Outputs:
- level1_domain_assignments.csv
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from rdflib import Graph
from rdflib.namespace import RDF, RDFS, SKOS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize


# ----------------------------
# Helpers
# ----------------------------


def safe_str(x) -> str:
    if x is None:
        return ""
    if isinstance(x, float) and pd.isna(x):
        return ""
    return str(x).strip()



def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", safe_str(text)).strip()



def slugify(text: str) -> str:
    text = normalize_space(text).lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")



def parse_listish(value) -> List[str]:
    """
    Turns values like:
    - "['nlp', 'text-generation']"
    - '["nlp","text-generation"]'
    - "nlp, text-generation"
    - "nlp|text-generation"
    into a list of strings.
    """
    s = safe_str(value)
    if not s:
        return []

    if (s.startswith("[") and s.endswith("]")) or (s.startswith("(") and s.endswith(")")):
        try:
            parsed = json.loads(s.replace("'", '"'))
            if isinstance(parsed, list):
                return [normalize_space(x) for x in parsed if normalize_space(x)]
        except Exception:
            pass

    parts = re.split(r"[|,;/]", s)
    return [normalize_space(p) for p in parts if normalize_space(p)]



def zscore_rows(mat: np.ndarray) -> np.ndarray:
    """Z-score normalize each item across domains."""
    mat = np.asarray(mat, dtype=float)
    means = mat.mean(axis=1, keepdims=True)
    stds = mat.std(axis=1, keepdims=True)
    stds[stds == 0] = 1.0
    return (mat - means) / stds



def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


# ----------------------------
# TTL parsing
# ----------------------------


def _gather_labelled_nodes(g: Graph):
    nodes = set(g.subjects(RDF.type, SKOS.Concept))
    for s in g.subjects(SKOS.prefLabel, None):
        nodes.add(s)
    for s in g.subjects(SKOS.altLabel, None):
        nodes.add(s)
    for s in g.subjects(RDFS.label, None):
        nodes.add(s)
    return nodes



def _best_label(g: Graph, node) -> str:
    for p in (SKOS.prefLabel, SKOS.altLabel, RDFS.label):
        for lbl in g.objects(node, p):
            val = safe_str(lbl)
            if val:
                return val
    iri = str(node)
    return iri.rsplit("/", 1)[-1].rsplit("#", 1)[-1]



def all_paths_from_ttl(ttl_path: Path) -> List[List[str]]:
    g = Graph()
    try:
        g.parse(str(ttl_path), format="turtle")
    except Exception:
        g.parse(str(ttl_path))

    nodes = _gather_labelled_nodes(g)
    if not nodes:
        return []

    parents = {n: [p for p in g.objects(n, SKOS.broader) if p in nodes] for n in nodes}
    children = {}
    for n, ps in parents.items():
        for p in ps:
            children.setdefault(p, []).append(n)

    roots = [n for n in nodes if not parents.get(n)]
    if not roots:
        all_parents = set()
        for ps in parents.values():
            all_parents.update(ps)
        roots = list(nodes - all_parents)

    paths = []

    def dfs(cur, trail):
        lbl = _best_label(g, cur)
        trail2 = trail + [lbl]
        ch = children.get(cur, [])
        if not ch:
            paths.append(trail2)
        else:
            for nxt in sorted(ch, key=lambda z: _best_label(g, z).lower()):
                dfs(nxt, trail2)

    for r in sorted(roots, key=lambda z: _best_label(g, z).lower()):
        dfs(r, [])

    return paths


# ----------------------------
# Broad domain config
# ----------------------------


def default_broad_domains() -> Dict[str, Dict]:
    """
    Broad domains aligned to the requested Level 1 classes.
    """
    return {
        "text": {
            "aliases": [
                "text", "nlp", "language", "llm", "bert", "gpt",
                "token", "sequence", "transformer", "dialogue",
                "chat", "qa", "summarization", "translation",
                "text classification", "token classification",
                "text generation", "question answering", "fill mask",
                "sentence similarity"
            ],
            "path_keywords": [
                "natural language processing",
                "language processing",
                "computational linguistics",
                "text mining",
                "information retrieval"
            ],
        },
        "image": {
            "aliases": [
                "image", "vision", "cv", "pixel", "segmentation",
                "object detection", "classification", "cnn",
                "resnet", "yolo", "vit", "image classification",
                "image segmentation", "image to image", "depth estimation",
                "image feature extraction", "ocr"
            ],
            "path_keywords": [
                "computer vision",
                "image processing",
                "pattern recognition",
                "visual",
                "graphics"
            ],
        },
        "audio": {
            "aliases": [
                "audio", "speech", "voice", "asr", "tts",
                "wav", "sound", "acoustic", "speech recognition",
                "automatic speech recognition", "text to speech",
                "audio classification", "audio to audio"
            ],
            "path_keywords": [
                "speech",
                "audio",
                "acoustics",
                "signal processing"
            ],
        },
        "video": {
            "aliases": [
                "video", "temporal", "frame sequence",
                "video classification", "action recognition"
            ],
            "path_keywords": [
                "video",
                "multimedia",
                "pattern recognition",
                "signal processing"
            ],
        },
        "multimodal": {
            "aliases": [
                "multimodal", "vision-language", "vlm",
                "text-image", "image-text", "cross-modal",
                "clip", "blip", "vqa", "visual question answering",
                "image to text", "document question answering", "text to image"
            ],
            "path_keywords": [
                "multimodal",
                "multimedia",
                "data fusion",
                "human computer interaction"
            ],
        },
        "tabular": {
            "aliases": [
                "tabular", "structured data", "csv",
                "dataframe", "feature vector", "table", "tables"
            ],
            "path_keywords": [
                "data mining",
                "database",
                "structured data",
                "data analysis"
            ],
        },
        "time-series": {
            "aliases": [
                "time series", "forecasting", "temporal data",
                "sequence prediction", "time-series forecasting"
            ],
            "path_keywords": [
                "time series",
                "forecasting",
                "prediction",
                "temporal"
            ],
        },
        "graph": {
            "aliases": [
                "graph", "gnn", "node", "edge",
                "graph neural network", "knowledge graph"
            ],
            "path_keywords": [
                "graph",
                "networks",
                "knowledge representation",
                "relational"
            ],
        },
        "Other / Unclear": {
            "aliases": [],
            "path_keywords": [],
        },
    }


PRIMARY_DOMAINS = [
    "text",
    "image",
    "audio",
    "video",
    "multimodal",
    "tabular",
    "time-series",
    "graph",
]


LEVEL_1_MODALITY = {
    "text": {
        "keywords": [
            "text", "nlp", "language", "llm", "bert", "gpt",
            "token", "sequence", "transformer", "dialogue",
            "chat", "qa", "summarization", "translation"
        ],
        "pipeline_tags": [
            "text-classification",
            "token-classification",
            "text-generation",
            "summarization",
            "translation",
            "question-answering",
            "fill-mask",
            "sentence-similarity"
        ]
    },
    "image": {
        "keywords": [
            "image", "vision", "cv", "pixel", "segmentation",
            "object detection", "classification", "cnn",
            "resnet", "yolo", "vit"
        ],
        "pipeline_tags": [
            "image-classification",
            "object-detection",
            "image-segmentation",
            "image-to-image",
            "depth-estimation",
            "image-feature-extraction"
        ]
    },
    "audio": {
        "keywords": [
            "audio", "speech", "voice", "asr", "tts",
            "wav", "sound", "acoustic", "speech recognition"
        ],
        "pipeline_tags": [
            "automatic-speech-recognition",
            "text-to-speech",
            "audio-classification",
            "audio-to-audio"
        ]
    },
    "video": {
        "keywords": [
            "video", "temporal", "frame sequence",
            "video classification", "action recognition"
        ],
        "pipeline_tags": [
            "video-classification"
        ]
    },
    "multimodal": {
        "keywords": [
            "multimodal", "vision-language", "vlm",
            "text-image", "image-text", "cross-modal",
            "clip", "blip", "vqa", "visual question answering"
        ],
        "pipeline_tags": [
            "image-to-text",
            "visual-question-answering",
            "document-question-answering",
            "text-to-image"
        ]
    },
    "tabular": {
        "keywords": [
            "tabular", "structured data", "csv",
            "dataframe", "feature vector"
        ],
        "pipeline_tags": []
    },
    "time-series": {
        "keywords": [
            "time series", "forecasting", "temporal data",
            "sequence prediction"
        ],
        "pipeline_tags": [
            "time-series-forecasting"
        ]
    },
    "graph": {
        "keywords": [
            "graph", "gnn", "node", "edge",
            "graph neural network"
        ],
        "pipeline_tags": []
    }
}


# ----------------------------
# IEEE path indexing
# ----------------------------


def build_taxonomy_path_strings(ttl_path: Path) -> List[str]:
    raw_paths = all_paths_from_ttl(ttl_path)
    out = []
    for seq in raw_paths:
        seq = [normalize_space(x) for x in seq if normalize_space(x)]
        if seq:
            out.append(" > ".join(seq))
    return sorted(set(out))



def find_matching_ieee_paths(
    all_taxonomy_paths: List[str],
    broad_domains: Dict[str, Dict],
) -> Dict[str, List[str]]:
    domain_to_paths = {}
    for domain, cfg in broad_domains.items():
        kws = [k.lower() for k in cfg.get("path_keywords", []) if normalize_space(k)]
        matches = []
        for path in all_taxonomy_paths:
            p = path.lower()
            if any(kw in p for kw in kws):
                matches.append(path)
        domain_to_paths[domain] = matches
    return domain_to_paths


# ----------------------------
# Text construction
# ----------------------------


def build_full_text(row: pd.Series) -> str:
    name = safe_str(row.get("model_name", row.get("model_id", "")))
    model_id = safe_str(row.get("model_id", ""))
    description = safe_str(row.get("short_description", row.get("description", "")))
    pipeline_tag = safe_str(row.get("pipeline_tag", ""))
    base_model = safe_str(row.get("base_model", ""))
    library_name = safe_str(row.get("library_name", ""))
    model_type = safe_str(row.get("model_type", ""))
    modality = safe_str(row.get("modality", ""))
    task = safe_str(row.get("task", ""))
    tags = parse_listish(row.get("tags", ""))

    pieces = [
        f"model name: {name}",
        f"model id: {model_id}",
        f"description: {description}",
        f"pipeline tag: {pipeline_tag}",
        f"task: {task}",
        f"modality: {modality}",
        f"model type: {model_type}",
        f"library: {library_name}",
        f"base model: {base_model}",
    ]
    if tags:
        pieces.append("tags: " + ", ".join(tags))

    return normalize_space(" | ".join([p for p in pieces if normalize_space(p)]))


# ----------------------------
# Modality routing (pipeline_tag only)
# ----------------------------


def detect_modality_route_from_pipeline_tag(row: pd.Series) -> Tuple[Optional[str], List[str], List[str]]:
    pipeline_tag = safe_str(row.get("pipeline_tag", "")).lower()
    routing_reasons: List[str] = []
    matched_keywords: List[str] = []

    if not pipeline_tag:
        return None, matched_keywords, routing_reasons

    matched_domains = []
    for domain, spec in LEVEL_1_MODALITY.items():
        exact_hits = [tag for tag in spec.get("pipeline_tags", []) if pipeline_tag == tag.lower()]
        partial_hits = [tag for tag in spec.get("pipeline_tags", []) if tag.lower() in pipeline_tag and pipeline_tag != tag.lower()]
        if exact_hits or partial_hits:
            matched_domains.append(domain)
            matched_keywords.extend(exact_hits + partial_hits)

    matched_keywords = sorted(set(matched_keywords))

    if len(matched_domains) == 1:
        routing_reasons.append("pipeline_tag matched configured modality tag")
        return matched_domains[0], matched_keywords, routing_reasons

    if len(matched_domains) > 1:
        routing_reasons.append("pipeline_tag matched multiple modality tags; no routing applied")

    return None, matched_keywords, routing_reasons


# ----------------------------
# Rule boosts
# ----------------------------


def rule_boosts(
    row: pd.Series,
    raw_scores: Dict[str, float],
    broad_domains: Dict[str, Dict],
) -> Tuple[Dict[str, float], Dict[str, List[str]], Dict[str, List[str]]]:
    """
    Smarter metadata-based boosts.
    - boosts scale with confidence headroom to avoid overwhelming strong embeddings
    - returns explanations (matched keywords + boost sources)
    """
    boosts = {k: 0.0 for k in broad_domains.keys()}
    matched_keywords: Dict[str, List[str]] = {k: [] for k in broad_domains.keys()}
    boost_sources: Dict[str, List[str]] = {k: [] for k in broad_domains.keys()}

    text = " ".join([
        safe_str(row.get("pipeline_tag", "")),
        safe_str(row.get("model_type", "")),
        safe_str(row.get("modality", "")),
        safe_str(row.get("task", "")),
        safe_str(row.get("short_description", row.get("description", ""))),
        safe_str(row.get("model_name", row.get("model_id", ""))),
        " ".join(parse_listish(row.get("tags", ""))),
    ]).lower()

    for domain, cfg in broad_domains.items():
        hits = []
        for alias in cfg.get("aliases", []):
            alias_l = alias.lower()
            if alias_l and alias_l in text:
                hits.append(alias_l)

        if hits:
            headroom = max(0.0, 1.0 - clamp(raw_scores.get(domain, 0.0), -1.0, 1.0))
            hit_bonus = min(0.30, 0.04 * len(set(hits)) + 0.08 * headroom)
            boosts[domain] += hit_bonus
            matched_keywords[domain].extend(sorted(set(hits)))
            boost_sources[domain].append(f"Alias hits ({len(set(hits))}) scaled by headroom={headroom:.3f}")

    handcrafted = {
        "text": ["bert", "roberta", "t5", "gpt", "llama", "mistral", "translation", "summarization"],
        "image": ["vit", "resnet", "yolo", "segment", "image", "vision", "diffusion", "sdxl"],
        "audio": ["whisper", "wav2vec", "speech", "audio", "asr", "tts"],
        "video": ["video", "temporal", "frame", "action recognition"],
        "multimodal": ["multimodal", "vision-language", "vlm", "image-text", "text-image", "clip", "blip", "vqa"],
        "tabular": ["tabular", "structured data", "csv", "dataframe", "feature vector"],
        "time-series": ["time series", "forecasting", "temporal data", "sequence prediction"],
        "graph": ["graph", "gnn", "node", "edge", "knowledge graph"],
    }

    for domain, keywords in handcrafted.items():
        hits = [kw for kw in keywords if kw in text]
        if hits:
            headroom = max(0.0, 1.0 - clamp(raw_scores.get(domain, 0.0), -1.0, 1.0))
            bonus = min(0.25, 0.03 * len(set(hits)) + 0.06 * headroom)
            boosts[domain] += bonus
            matched_keywords[domain].extend(sorted(set(hits)))
            boost_sources[domain].append(f"Handcrafted hints ({len(set(hits))}) scaled by headroom={headroom:.3f}")

    for domain in broad_domains.keys():
        matched_keywords[domain] = sorted(set(matched_keywords[domain]))

    return boosts, matched_keywords, boost_sources


# ----------------------------
# Encoders
# ----------------------------


class TextEncoder:
    def fit(self, texts: List[str]):
        raise NotImplementedError

    def encode(self, texts: List[str]) -> np.ndarray:
        raise NotImplementedError


class TfidfTextEncoder(TextEncoder):
    def __init__(self):
        self.vec = TfidfVectorizer(
            max_features=50000,
            ngram_range=(1, 3),
            token_pattern=r"(?u)\b[\w\-]+\b",
            lowercase=True,
        )

    def fit(self, texts: List[str]):
        self.mat = normalize(self.vec.fit_transform(texts))
        return self

    def encode(self, texts: List[str]) -> np.ndarray:
        return normalize(self.vec.transform(texts))


class SentenceTransformerEncoder(TextEncoder):
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)

    def fit(self, texts: List[str]):
        return self

    def encode(self, texts: List[str]) -> np.ndarray:
        emb = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
        return np.asarray(emb)



def get_encoder(encoder_name: str) -> TextEncoder:
    if encoder_name == "sbert":
        try:
            return SentenceTransformerEncoder()
        except Exception:
            print("Warning: sentence-transformers unavailable; falling back to TF-IDF.")
            return TfidfTextEncoder()
    return TfidfTextEncoder()


# ----------------------------
# Domain prototype building
# ----------------------------


def build_domain_prototypes(
    broad_domains: Dict[str, Dict],
    domain_to_ieee_paths: Dict[str, List[str]],
) -> Dict[str, str]:
    prototypes = {}
    for domain, cfg in broad_domains.items():
        aliases = cfg.get("aliases", [])
        paths = domain_to_ieee_paths.get(domain, [])
        pieces = [
            f"domain: {domain}",
            "aliases: " + ", ".join(aliases) if aliases else "",
            "ieee paths: " + " || ".join(paths[:300]) if paths else "",
        ]
        prototypes[domain] = normalize_space(" ".join([p for p in pieces if p]))
    return prototypes


# ----------------------------
# Main assignment
# ----------------------------


def assign_domains(
    df: pd.DataFrame,
    ttl_path: Path,
    encoder_name: str = "sbert",
    min_confidence_margin: float = 0.03,
    use_modality_routing: bool = True,
    routing_tie_break_margin: float = 0.08,
) -> pd.DataFrame:
    broad_domains = default_broad_domains()
    taxonomy_paths = build_taxonomy_path_strings(ttl_path)
    domain_to_ieee_paths = find_matching_ieee_paths(taxonomy_paths, broad_domains)
    prototypes = build_domain_prototypes(broad_domains, domain_to_ieee_paths)

    df = df.copy()
    df["full_text"] = df.apply(build_full_text, axis=1)

    encoder = get_encoder(encoder_name)
    domain_names = list(broad_domains.keys())
    proto_texts = [prototypes[d] for d in domain_names]
    all_fit_texts = proto_texts + df["full_text"].fillna("").tolist()
    encoder.fit(all_fit_texts)

    proto_emb = encoder.encode(proto_texts)
    item_emb = encoder.encode(df["full_text"].fillna("").tolist())

    raw_sim_matrix = cosine_similarity(item_emb, proto_emb)
    normalized_sim_matrix = zscore_rows(raw_sim_matrix)

    rows = []
    for row_idx, (_, row) in enumerate(df.iterrows()):
        raw_scores = {domain_names[j]: float(raw_sim_matrix[row_idx, j]) for j in range(len(domain_names))}
        normalized_scores = {domain_names[j]: float(normalized_sim_matrix[row_idx, j]) for j in range(len(domain_names))}

        boosts, matched_by_domain, boost_sources = rule_boosts(row, normalized_scores, broad_domains)
        final_scores = {k: normalized_scores[k] + boosts.get(k, 0.0) for k in normalized_scores.keys()}

        routed_domain = None
        routing_keywords: List[str] = []
        routing_reasons: List[str] = []
        routing_applied = False

        if use_modality_routing:
            routed_domain, routing_keywords, routing_reasons = detect_modality_route_from_pipeline_tag(row)
        else:
            routing_reasons.append("modality routing disabled")

        primary_ranking = sorted(
            [(d, final_scores[d]) for d in PRIMARY_DOMAINS],
            key=lambda x: x[1],
            reverse=True,
        )
        top1_domain, top1_score = primary_ranking[0]
        top2_domain, top2_score = primary_ranking[1]

        margin = top1_score - top2_score
        assigned = top1_domain if margin >= min_confidence_margin else "Other / Unclear"

        if use_modality_routing and routed_domain and routed_domain in {top1_domain, top2_domain}:
            if margin < routing_tie_break_margin:
                if routed_domain == top2_domain:
                    assigned = top2_domain
                    routing_applied = True
                    routing_reasons.append(
                        f"pipeline_tag tie-break selected top2 over top1 because margin={margin:.6f} < {routing_tie_break_margin:.6f}"
                    )
                else:
                    if assigned == "Other / Unclear":
                        assigned = top1_domain
                        routing_applied = True
                        routing_reasons.append(
                            f"pipeline_tag tie-break resolved low-confidence case in favor of top1 because margin={margin:.6f} < {routing_tie_break_margin:.6f}"
                        )
            else:
                routing_reasons.append(
                    f"pipeline_tag match ignored because margin={margin:.6f} >= {routing_tie_break_margin:.6f}"
                )
        elif use_modality_routing and routed_domain and routed_domain not in {top1_domain, top2_domain}:
            routing_reasons.append("pipeline_tag match ignored because routed domain was not in top2 candidates")

        explanations = {
            "matched_keywords_by_domain": {k: v for k, v in matched_by_domain.items() if v},
            "boost_sources_by_domain": {k: v for k, v in boost_sources.items() if v},
            "modality_routing_enabled": use_modality_routing,
            "modality_routing_applied": routing_applied,
            "modality_routing_keywords": routing_keywords,
            "modality_routing_reasons": routing_reasons,
        }

        rows.append({
            "model_id": safe_str(row.get("model_id", "")),
            "model_name": safe_str(row.get("model_name", row.get("model_id", ""))),
            "description": safe_str(row.get("short_description", row.get("description", ""))),
            "assigned_domain": assigned,
            "top1_domain": top1_domain,
            "top1_score": round(top1_score, 6),
            "top2_domain": top2_domain,
            "top2_score": round(top2_score, 6),
            "score_margin": round(margin, 6),
            "routed_domain": routed_domain or "",
            "routing_applied": routing_applied,
            "raw_scores_json": json.dumps(raw_scores, ensure_ascii=False),
            "normalized_scores_json": json.dumps(normalized_scores, ensure_ascii=False),
            "boosts_json": json.dumps(boosts, ensure_ascii=False),
            "all_scores_json": json.dumps(final_scores, ensure_ascii=False),
            "explanations_json": json.dumps(explanations, ensure_ascii=False),
            "base_model": safe_str(row.get("basemodels", "")),
        })

    out = pd.DataFrame(rows)
    return out.sort_values(["assigned_domain", "top1_score"], ascending=[True, False])


# ----------------------------
# CLI
# ----------------------------
if __name__ == "__main__":
    input_csv = "7-CLUSTERING_MODELS/hf_models.csv"
    ttl_path = "7-CLUSTERING_MODELS/WorkflowInputs/ieee-taxonomy-2025.ttl"
    output_csv = "7-CLUSTERING_MODELS/level1_domain_assignments_improved.csv"

    encoder = "sbert"   # or "tfidf"
    min_margin = 0.03
    use_modality_routing = True
    routing_tie_break_margin = 0.08

    df = pd.read_csv(input_csv)

    result = assign_domains(
        df=df,
        ttl_path=Path(ttl_path),
        encoder_name=encoder,
        min_confidence_margin=min_margin,
        use_modality_routing=use_modality_routing,
        routing_tie_break_margin=routing_tie_break_margin,
    )

    result.to_csv(output_csv, index=False)
    print(f"Done! Saved to {output_csv}")
