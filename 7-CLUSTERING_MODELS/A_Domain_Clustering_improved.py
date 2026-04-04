#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Level 1 domain assignment for foundation models
-----------------------------------------------

Goal:
- Assign each model to a broad IEEE-style domain such as:
  NLP, Computer Vision, Speech/Audio, Multimodal, RL, etc.
- No GPT calls
- Uses model description + metadata
- Supports sentence-transformers embeddings, with TF-IDF fallback

Implemented improvements:
1. Per-item score normalization across domains before rule blending
3. Modality-first routing before final domain assignment
4. Smarter rule boosts scaled by domain confidence headroom
5. Generative AI handled as a secondary label rather than primary domain
6. Added explanation outputs: matched keywords and boost sources

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
    Curated broad domains for Level 1.

    Generative AI is retained as a secondary label only.
    It is excluded from primary domain ranking to avoid conflicts with
    NLP / CV / Multimodal / Audio primary modality assignments.
    """
    return {
        "Natural Language Processing": {
            "aliases": [
                "nlp", "natural language processing", "language model",
                "text generation", "text classification", "question answering",
                "summarization", "token classification", "translation",
                "masked language modeling", "causal language modeling"
            ],
            "path_keywords": [
                "natural language processing",
                "language processing",
                "information retrieval",
                "text mining",
                "computational linguistics"
            ],
        },
        "Computer Vision": {
            "aliases": [
                "computer vision", "vision", "image", "object detection",
                "segmentation", "image classification", "ocr",
                "image generation", "diffusion", "vision transformer"
            ],
            "path_keywords": [
                "computer vision",
                "image processing",
                "pattern recognition",
                "visual",
                "graphics"
            ],
        },
        "Speech and Audio": {
            "aliases": [
                "speech", "audio", "automatic speech recognition", "asr",
                "text to speech", "tts", "speaker recognition",
                "audio classification", "music"
            ],
            "path_keywords": [
                "speech",
                "audio",
                "acoustics",
                "signal processing"
            ],
        },
        "Multimodal AI": {
            "aliases": [
                "multimodal", "vision-language", "vlm", "image-text",
                "audio-text", "video-text", "cross-modal"
            ],
            "path_keywords": [
                "multimedia",
                "multimodal",
                "human computer interaction",
                "data fusion"
            ],
        },
        "Reinforcement Learning": {
            "aliases": [
                "reinforcement learning", "rl", "agent", "policy optimization",
                "decision making", "control", "robot learning"
            ],
            "path_keywords": [
                "reinforcement learning",
                "control",
                "decision"
            ],
        },
        "Time Series and Forecasting": {
            "aliases": [
                "time series", "forecasting", "sequence forecasting",
                "temporal prediction"
            ],
            "path_keywords": [
                "time series",
                "forecasting",
                "prediction"
            ],
        },
        "Graph and Relational Learning": {
            "aliases": [
                "graph", "gnn", "graph neural network",
                "knowledge graph", "relational learning"
            ],
            "path_keywords": [
                "graph",
                "knowledge representation",
                "networks"
            ],
        },
        "Scientific and Domain-specific AI": {
            "aliases": [
                "biology", "chemistry", "medical", "healthcare",
                "scientific", "materials", "geospatial", "climate",
                "protein", "genomics", "legal", "finance"
            ],
            "path_keywords": [
                "bioinformatics",
                "medical",
                "healthcare",
                "scientific"
            ],
        },
        "Generative AI": {
            "aliases": [
                "generative ai", "generative model", "diffusion model",
                "text generation", "image generation", "video generation",
                "llm", "foundation model"
            ],
            "path_keywords": [
                "generative",
                "machine learning",
                "artificial intelligence"
            ],
            "secondary_only": True,
        },
        "Other / Unclear": {
            "aliases": [],
            "path_keywords": [],
        },
    }


PRIMARY_DOMAINS = [
    "Natural Language Processing",
    "Computer Vision",
    "Speech and Audio",
    "Multimodal AI",
    "Reinforcement Learning",
    "Time Series and Forecasting",
    "Graph and Relational Learning",
    "Scientific and Domain-specific AI",
]


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
# Modality-first routing
# ----------------------------


def modality_config() -> Dict[str, Dict[str, List[str]]]:
    return {
        "text": {
            "keywords": [
                "text", "nlp", "language", "translation", "summarization",
                "question answering", "qa", "token classification", "causal language modeling",
                "masked language modeling", "chat", "llm"
            ],
            "domains": ["Natural Language Processing"],
        },
        "vision": {
            "keywords": [
                "image", "vision", "visual", "object detection", "segmentation",
                "ocr", "diffusion", "image classification", "video", "vision transformer"
            ],
            "domains": ["Computer Vision"],
        },
        "audio": {
            "keywords": [
                "audio", "speech", "asr", "tts", "acoustic", "speaker recognition",
                "wav2vec", "whisper", "music"
            ],
            "domains": ["Speech and Audio"],
        },
        "multimodal": {
            "keywords": [
                "multimodal", "vision-language", "vlm", "image-text",
                "audio-text", "video-text", "cross-modal", "text-image", "text-audio"
            ],
            "domains": ["Multimodal AI"],
        },
        "rl": {
            "keywords": [
                "reinforcement learning", "agent", "policy", "control", "decision making", "robot learning"
            ],
            "domains": ["Reinforcement Learning"],
        },
        "graph": {
            "keywords": ["graph", "gnn", "knowledge graph", "relational learning"],
            "domains": ["Graph and Relational Learning"],
        },
        "time_series": {
            "keywords": ["time series", "forecasting", "temporal prediction", "sequence forecasting"],
            "domains": ["Time Series and Forecasting"],
        },
        "scientific": {
            "keywords": [
                "bio", "biology", "chemistry", "protein", "genomics", "medical", "healthcare",
                "materials", "geospatial", "climate", "finance", "legal"
            ],
            "domains": ["Scientific and Domain-specific AI"],
        },
    }



def detect_modality_route(row: pd.Series) -> Tuple[Optional[str], List[str], List[str]]:
    text = " ".join([
        safe_str(row.get("pipeline_tag", "")),
        safe_str(row.get("model_type", "")),
        safe_str(row.get("modality", "")),
        safe_str(row.get("task", "")),
        safe_str(row.get("short_description", row.get("description", ""))),
        safe_str(row.get("model_name", row.get("model_id", ""))),
        " ".join(parse_listish(row.get("tags", ""))),
    ]).lower()

    cfg = modality_config()
    hits_by_bucket: Dict[str, List[str]] = {}
    for bucket, spec in cfg.items():
        hits = [kw for kw in spec["keywords"] if kw in text]
        if hits:
            hits_by_bucket[bucket] = hits

    routing_reasons: List[str] = []
    matched_keywords: List[str] = sorted({kw for hits in hits_by_bucket.values() for kw in hits})

    if "multimodal" in hits_by_bucket:
        routing_reasons.append("multimodal keyword hit")
        return "Multimodal AI", matched_keywords, routing_reasons

    modality_count = sum(1 for k in ["text", "vision", "audio"] if k in hits_by_bucket)
    if modality_count >= 2:
        routing_reasons.append("multiple core modalities detected")
        return "Multimodal AI", matched_keywords, routing_reasons

    for bucket in ["vision", "audio", "text", "rl", "graph", "time_series", "scientific"]:
        if bucket in hits_by_bucket:
            routing_reasons.append(f"{bucket} modality/task keywords matched")
            return cfg[bucket]["domains"][0], matched_keywords, routing_reasons

    return None, matched_keywords, routing_reasons


# ----------------------------
# Rule boosts
# ----------------------------


def generative_secondary_score(row: pd.Series, raw_scores: Dict[str, float], broad_domains: Dict[str, Dict]) -> Tuple[float, List[str], List[str]]:
    text = " ".join([
        safe_str(row.get("pipeline_tag", "")),
        safe_str(row.get("model_type", "")),
        safe_str(row.get("short_description", row.get("description", ""))),
        safe_str(row.get("model_name", row.get("model_id", ""))),
        " ".join(parse_listish(row.get("tags", ""))),
    ]).lower()

    gen_aliases = [a.lower() for a in broad_domains["Generative AI"].get("aliases", [])]
    matched = [a for a in gen_aliases if a and a in text]
    sources: List[str] = []

    base = float(raw_scores.get("Generative AI", 0.0))
    headroom = max(0.0, 1.0 - clamp(base, -1.0, 1.0))
    bonus = min(0.30, 0.06 * len(set(matched)) + 0.08 * headroom if matched else 0.0)

    if matched:
        sources.append(f"Generative AI aliases: {', '.join(sorted(set(matched)))}")

    return base + bonus, sorted(set(matched)), sources



def rule_boosts(row: pd.Series, raw_scores: Dict[str, float], broad_domains: Dict[str, Dict]) -> Tuple[Dict[str, float], Dict[str, List[str]], Dict[str, List[str]]]:
    """
    Smarter metadata-based boosts.
    Improvement #4:
    - boosts scale with confidence headroom to avoid overwhelming strong embeddings
    Improvement #6:
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
        if cfg.get("secondary_only"):
            continue
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
        "Natural Language Processing": ["bert", "roberta", "t5", "gpt", "llama", "mistral", "translation", "summarization"],
        "Computer Vision": ["vit", "resnet", "yolo", "segment", "image", "vision", "diffusion", "sdxl"],
        "Speech and Audio": ["whisper", "wav2vec", "speech", "audio", "asr", "tts"],
        "Multimodal AI": ["multimodal", "vision-language", "vlm", "image-text", "video-text"],
        "Reinforcement Learning": ["reinforcement learning", "policy", "agent", "control"],
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
            token_pattern=r"(?u)[\w\-]+",
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

        routed_domain, routing_keywords, routing_reasons = detect_modality_route(row)

        primary_ranking = sorted(
            [(d, final_scores[d]) for d in PRIMARY_DOMAINS],
            key=lambda x: x[1],
            reverse=True,
        )
        top1_domain, top1_score = primary_ranking[0]
        top2_domain, top2_score = primary_ranking[1]

        margin = top1_score - top2_score
        assigned = top1_domain if margin >= min_confidence_margin else "Other / Unclear"

        # Improvement #3: modality-first routing can override when clearly present.
        if routed_domain and routed_domain in PRIMARY_DOMAINS:
            assigned = routed_domain
            if routed_domain != top1_domain:
                routing_reasons.append(f"routing override over similarity winner {top1_domain}")

        # Improvement #5: Generative AI is emitted as a secondary label only.
        generative_score, generative_hits, gen_sources = generative_secondary_score(row, normalized_scores, broad_domains)
        secondary_labels = []
        if generative_hits or generative_score > 0.35:
            secondary_labels.append("Generative AI")

        explanations = {
            "matched_keywords_by_domain": {k: v for k, v in matched_by_domain.items() if v},
            "boost_sources_by_domain": {k: v for k, v in boost_sources.items() if v},
            "modality_routing_keywords": routing_keywords,
            "modality_routing_reasons": routing_reasons,
            "generative_secondary_keywords": generative_hits,
            "generative_secondary_sources": gen_sources,
        }

        rows.append({
            "model_id": safe_str(row.get("model_id", "")),
            "model_name": safe_str(row.get("model_name", row.get("model_id", ""))),
            "description": safe_str(row.get("short_description", row.get("description", ""))),
            "assigned_domain": assigned,
            "secondary_labels": json.dumps(secondary_labels, ensure_ascii=False),
            "top1_domain": top1_domain,
            "top1_score": round(top1_score, 6),
            "top2_domain": top2_domain,
            "top2_score": round(top2_score, 6),
            "score_margin": round(margin, 6),
            "routed_domain": routed_domain or "",
            "raw_scores_json": json.dumps(raw_scores, ensure_ascii=False),
            "normalized_scores_json": json.dumps(normalized_scores, ensure_ascii=False),
            "boosts_json": json.dumps(boosts, ensure_ascii=False),
            "all_scores_json": json.dumps(final_scores, ensure_ascii=False),
            "explanations_json": json.dumps(explanations, ensure_ascii=False),
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

    df = pd.read_csv(input_csv)

    result = assign_domains(
        df=df,
        ttl_path=Path(ttl_path),
        encoder_name=encoder,
        min_confidence_margin=min_margin,
    )

    result.to_csv(output_csv, index=False)
    print(f"Done! Saved to {output_csv}")
