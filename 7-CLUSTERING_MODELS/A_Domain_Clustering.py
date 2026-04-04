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

Expected input CSV columns (flexible, only model_id/name + description are strongly recommended):
- model_id
- model_name
- description
- tags
- pipeline_tag
- base_model
- library_name
- task
- modality

You can rename/adapt in `build_full_text()` if your columns differ.

Outputs:
- level1_domain_assignments.csv
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional

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

    # JSON-ish list
    if (s.startswith("[") and s.endswith("]")) or (s.startswith("(") and s.endswith(")")):
        try:
            parsed = json.loads(s.replace("'", '"'))
            if isinstance(parsed, list):
                return [normalize_space(x) for x in parsed if normalize_space(x)]
        except Exception:
            pass

    # fallback split
    parts = re.split(r"[|,;/]", s)
    return [normalize_space(p) for p in parts if normalize_space(p)]


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

    You should edit this list for your use case.
    Each domain has:
    - aliases: labels / keywords
    - path_keywords: terms to match against IEEE paths
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
        },
        "Other / Unclear": {
            "aliases": [],
            "path_keywords": [],
        },
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
    """
    Match each broad domain to the subset of IEEE paths whose path text
    contains one of the configured path_keywords.
    """
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
    """
    Build one text string per model from description + metadata.
    Adjust these field names to match your actual Hugging Face export.
    """
    name = safe_str(row.get("model_name", row.get("model_id", "")))
    model_id = safe_str(row.get("model_id", ""))
    description = safe_str(row.get("short_description", ""))
    pipeline_tag = safe_str(row.get("pipeline_tag", ""))
    base_model = safe_str(row.get("base_model", ""))
    library_name = safe_str(row.get("library_name", ""))
    model_type = safe_str(row.get("model_type", ""))
    tags = parse_listish(row.get("tags", ""))

    pieces = [
        f"model name: {name}",
        f"model id: {model_id}",
        f"description: {description}",
        f"pipeline tag: {pipeline_tag}",
        f"model type: {model_type}",
        f"library: {library_name}",
        f"base model: {base_model}",
    ]
    if tags:
        pieces.append("tags: " + ", ".join(tags))

    return normalize_space(" | ".join([p for p in pieces if normalize_space(p)]))


# ----------------------------
# Rule boosts
# ----------------------------

def rule_boosts(row: pd.Series, broad_domains: Dict[str, Dict]) -> Dict[str, float]:
    """
    Cheap metadata-based boosts.
    """
    boosts = {k: 0.0 for k in broad_domains.keys()}

    text = " ".join([
        safe_str(row.get("pipeline_tag", "")),
        safe_str(row.get("model_type", "")),
        safe_str(row.get("short_description", "")),
        safe_str(row.get("model_name", row.get("model_id", ""))),
        " ".join(parse_listish(row.get("tags", ""))),
    ]).lower()

    for domain, cfg in broad_domains.items():
        for alias in cfg.get("aliases", []):
            alias_l = alias.lower()
            if alias_l and alias_l in text:
                boosts[domain] += 0.08

    # simple handcrafted hints
    if any(x in text for x in ["bert", "roberta", "t5", "gpt", "llama", "mistral", "translation", "summarization"]):
        boosts["Natural Language Processing"] += 0.12

    if any(x in text for x in ["vit", "resnet", "yolo", "segment", "image", "vision", "diffusion", "sdxl"]):
        boosts["Computer Vision"] += 0.12

    if any(x in text for x in ["whisper", "wav2vec", "speech", "audio", "asr", "tts"]):
        boosts["Speech and Audio"] += 0.12

    if any(x in text for x in ["multimodal", "vision-language", "vlm", "image-text", "video-text"]):
        boosts["Multimodal AI"] += 0.15

    if any(x in text for x in ["reinforcement learning", "policy", "agent", "control"]):
        boosts["Reinforcement Learning"] += 0.15

    return boosts


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
        # no training needed
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
    """
    Create one prototype text per domain by combining:
    - domain name
    - aliases
    - matched IEEE path strings
    """
    prototypes = {}
    for domain, cfg in broad_domains.items():
        aliases = cfg.get("aliases", [])
        paths = domain_to_ieee_paths.get(domain, [])

        pieces = [
            f"domain: {domain}",
            "aliases: " + ", ".join(aliases) if aliases else "",
            "ieee paths: " + " || ".join(paths[:300]) if paths else "",  # cap to avoid giant strings
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

    # build model texts
    df = df.copy()
    df["full_text"] = df.apply(build_full_text, axis=1)

    # build encoder
    encoder = get_encoder(encoder_name)

    proto_texts = [prototypes[d] for d in broad_domains.keys()]
    all_fit_texts = proto_texts + df["full_text"].fillna("").tolist()

    encoder.fit(all_fit_texts)

    proto_emb = encoder.encode(proto_texts)
    item_emb = encoder.encode(df["full_text"].fillna("").tolist())

    sims = cosine_similarity(item_emb, proto_emb)
    domain_names = list(broad_domains.keys())

    rows = []
    for i, row in df.iterrows():
        raw_scores = {domain_names[j]: float(sims[i, j]) for j in range(len(domain_names))}
        boosts = rule_boosts(row, broad_domains)
        final_scores = {k: raw_scores[k] + boosts.get(k, 0.0) for k in raw_scores.keys()}

        ranked = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
        top1_domain, top1_score = ranked[0]
        top2_domain, top2_score = ranked[1]

        margin = top1_score - top2_score
        assigned = top1_domain if margin >= min_confidence_margin else "Other / Unclear"

        rows.append({
            "model_id": safe_str(row.get("model_id", "")),
            "model_name": safe_str(row.get("model_name", row.get("model_id", ""))),
            "description": safe_str(row.get("short_description", "")),
            "assigned_domain": assigned,
            "top1_domain": top1_domain,
            "top1_score": round(top1_score, 6),
            "top2_domain": top2_domain,
            "top2_score": round(top2_score, 6),
            "score_margin": round(margin, 6),
            "all_scores_json": json.dumps(final_scores, ensure_ascii=False),
        })

    out = pd.DataFrame(rows)
    return out.sort_values(["assigned_domain", "top1_score"], ascending=[True, False])


# ----------------------------
# CLI
# ----------------------------
if __name__ == "__main__":
    # === EDIT THESE PATHS ===
    input_csv = "7-CLUSTERING_MODELS/hf_models.csv"
    ttl_path = "7-CLUSTERING_MODELS/WorkflowInputs/ieee-taxonomy-2025.ttl"
    output_csv = "7-CLUSTERING_MODELS/level1_domain_assignments.csv"

    # Optional settings
    encoder = "sbert"   # or "tfidf"
    min_margin = 0.03

    import pandas as pd

    df = pd.read_csv(input_csv)
    # df = df.iloc[:10]

    result = assign_domains(
        df=df,
        ttl_path=Path(ttl_path),
        encoder_name=encoder,
        min_confidence_margin=min_margin,
    )

    result.to_csv(output_csv, index=False)

    print(f"Done! Saved to {output_csv}")