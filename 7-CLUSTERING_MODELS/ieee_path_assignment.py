#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
IEEE path assignment for foundation models
-----------------------------------------

Goal:
- Assign each model to the most relevant IEEE taxonomy path(s)
- Uses model description + metadata
- No GPT calls
- Supports sentence-transformers embeddings, with TF-IDF fallback

Behavior:
1. Builds a single text representation from description + metadata
2. Embeds every IEEE path string and every model text
3. Scores model-to-path similarity
4. Adds lightweight lexical boosts from exact/partial token overlaps
5. Returns the top IEEE path, top-k candidates, confidence margin, and explanations

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
- model_type

Outputs:
- ieee_path_assignments.csv
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

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



def parse_listish(value) -> List[str]:
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
    mat = np.asarray(mat, dtype=float)
    means = mat.mean(axis=1, keepdims=True)
    stds = mat.std(axis=1, keepdims=True)
    stds[stds == 0] = 1.0
    return (mat - means) / stds



def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))



def tokenize(text: str) -> List[str]:
    text = normalize_space(text).lower()
    return re.findall(r"[a-z0-9][a-z0-9\-_/]+|[a-z0-9]", text)


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



def build_taxonomy_path_strings(ttl_path: Path) -> List[str]:
    raw_paths = all_paths_from_ttl(ttl_path)
    out = []
    for seq in raw_paths:
        seq = [normalize_space(x) for x in seq if normalize_space(x)]
        if seq:
            out.append(" > ".join(seq))
    return sorted(set(out))


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
            token_pattern=r"(?u)\b[\w\-/]+\b",
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
# IEEE path text + rule boosts
# ----------------------------


STOPWORDS = {
    "and", "or", "of", "for", "to", "the", "a", "an", "in", "on", "with",
    "by", "from", "using", "based", "system", "systems", "methods", "method",
    "models", "model", "learning", "machine", "artificial", "intelligence",
}


PIPELINE_TAG_SYNONYMS: Dict[str, List[str]] = {
    "text-generation": ["natural language processing", "language models", "text generation"],
    "text2text-generation": ["natural language processing", "text generation", "sequence to sequence learning"],
    "fill-mask": ["natural language processing", "language models"],
    "question-answering": ["question answering", "natural language processing"],
    "document-question-answering": ["document analysis", "question answering", "optical character recognition"],
    "visual-question-answering": ["computer vision", "question answering", "multimodal interaction"],
    "text-classification": ["text categorization", "natural language processing"],
    "token-classification": ["named entity recognition", "natural language processing", "sequence labeling"],
    "sentence-similarity": ["semantic similarity", "natural language processing", "information retrieval"],
    "text-retrieval": ["information retrieval", "search engines"],
    "text-ranking": ["information retrieval", "ranking"],
    "translation": ["machine translation", "natural language processing"],
    "summarization": ["automatic summarization", "natural language processing"],
    "image-classification": ["computer vision", "image classification"],
    "zero-shot-image-classification": ["computer vision", "image classification"],
    "object-detection": ["computer vision", "object detection"],
    "zero-shot-object-detection": ["computer vision", "object detection"],
    "image-segmentation": ["computer vision", "image segmentation"],
    "depth-estimation": ["computer vision", "depth estimation", "stereo vision"],
    "keypoint-detection": ["computer vision", "pose estimation"],
    "image-feature-extraction": ["computer vision", "feature extraction"],
    "mask-generation": ["computer vision", "image segmentation"],
    "image-to-text": ["image captioning", "computer vision", "multimodal interaction"],
    "image-text-to-text": ["computer vision", "natural language processing", "multimodal interaction"],
    "text-to-image": ["image generation", "computer vision", "generative models"],
    "image-to-image": ["image processing", "computer vision"],
    "unconditional-image-generation": ["image generation", "computer vision", "generative models"],
    "image-to-video": ["video generation", "computer vision", "generative models"],
    "text-to-video": ["video generation", "computer vision", "generative models"],
    "video-to-video": ["video processing", "video generation"],
    "video-classification": ["video analysis", "computer vision", "action recognition"],
    "video-text-to-text": ["video analysis", "natural language processing", "multimodal interaction"],
    "automatic-speech-recognition": ["speech recognition", "audio processing"],
    "text-to-speech": ["speech synthesis", "audio processing"],
    "audio-classification": ["audio classification", "audio processing"],
    "audio-to-audio": ["audio processing", "speech enhancement"],
    "audio-text-to-text": ["speech recognition", "natural language processing", "audio processing"],
    "voice-activity-detection": ["speech processing", "audio processing"],
    "tabular-classification": ["classification", "structured data", "data mining"],
    "tabular-regression": ["regression analysis", "structured data", "data mining"],
    "table-question-answering": ["question answering", "structured data"],
    "table-to-text": ["data-to-text generation", "structured data", "natural language processing"],
    "time-series-forecasting": ["time series analysis", "forecasting"],
    "graph-ml": ["graph theory", "graph neural networks", "knowledge graphs"],
    "reinforcement-learning": ["reinforcement learning", "control systems"],
    "robotics": ["robotics", "autonomous systems"],
    "text-to-audio": ["audio generation", "audio processing", "generative models"],
    "text-to-3d": ["3d reconstruction", "computer graphics", "generative models"],
    "image-to-3d": ["3d reconstruction", "computer vision"],
    "any-to-any": ["multimodal interaction", "generative models"],
}


FAMILY_HINTS: Dict[str, List[str]] = {
    "bert": ["language models", "natural language processing"],
    "gpt": ["language models", "text generation"],
    "llama": ["language models", "text generation"],
    "t5": ["natural language processing", "text generation"],
    "mistral": ["language models", "text generation"],
    "whisper": ["speech recognition", "audio processing"],
    "wav2vec": ["speech recognition", "audio processing"],
    "yolo": ["object detection", "computer vision"],
    "sam": ["image segmentation", "computer vision"],
    "clip": ["multimodal interaction", "computer vision", "natural language processing"],
    "blip": ["multimodal interaction", "computer vision", "natural language processing"],
    "sdxl": ["image generation", "generative models"],
    "stable diffusion": ["image generation", "diffusion models", "generative models"],
    "diffusion": ["diffusion models", "generative models"],
}


def build_augmented_model_text(row: pd.Series) -> str:
    full_text = build_full_text(row)
    pipeline_tag = safe_str(row.get("pipeline_tag", "")).lower()
    model_name = safe_str(row.get("model_name", row.get("model_id", ""))).lower()
    base_model = safe_str(row.get("base_model", "")).lower()
    tags = [t.lower() for t in parse_listish(row.get("tags", ""))]

    extras: List[str] = []

    if pipeline_tag in PIPELINE_TAG_SYNONYMS:
        extras.extend(PIPELINE_TAG_SYNONYMS[pipeline_tag])

    combined_for_family = " | ".join([model_name, base_model, " ".join(tags), full_text.lower()])
    for hint, synonyms in FAMILY_HINTS.items():
        if hint in combined_for_family:
            extras.extend(synonyms)

    if extras:
        full_text += " | inferred taxonomy hints: " + ", ".join(sorted(set(extras)))

    return normalize_space(full_text)



def build_path_text(path: str) -> str:
    segments = [normalize_space(x) for x in path.split(">") if normalize_space(x)]
    leaf = segments[-1] if segments else path
    leaf_tokens = tokenize(leaf)
    extras = [t for t in leaf_tokens if t not in STOPWORDS]
    parts = [
        f"ieee taxonomy path: {path}",
        f"leaf concept: {leaf}",
    ]
    if extras:
        parts.append("leaf keywords: " + ", ".join(extras))
    return normalize_space(" | ".join(parts))



def lexical_boost(model_text: str, path: str) -> Tuple[float, List[str]]:
    model_l = model_text.lower()
    path_segments = [normalize_space(x).lower() for x in path.split(">") if normalize_space(x)]
    matched: List[str] = []
    boost = 0.0

    for seg in path_segments:
        if len(seg) >= 4 and seg in model_l:
            matched.append(seg)
            boost += 0.10

    path_tokens = [t for t in tokenize(path) if len(t) >= 4 and t not in STOPWORDS]
    token_hits = sorted({tok for tok in path_tokens if tok in model_l})
    if token_hits:
        matched.extend(token_hits)
        boost += min(0.18, 0.03 * len(token_hits))

    return min(0.28, boost), sorted(set(matched))


# ----------------------------
# Main assignment
# ----------------------------


def assign_ieee_paths(
    df: pd.DataFrame,
    ttl_path: Path,
    encoder_name: str = "sbert",
    min_confidence_margin: float = 0.03,
    top_k: int = 5,
) -> pd.DataFrame:
    taxonomy_paths = build_taxonomy_path_strings(ttl_path)
    if not taxonomy_paths:
        raise ValueError(f"No IEEE taxonomy paths could be parsed from: {ttl_path}")

    df = df.copy()
    df["full_text"] = df.apply(build_augmented_model_text, axis=1)

    path_texts = [build_path_text(path) for path in taxonomy_paths]

    encoder = get_encoder(encoder_name)
    encoder.fit(path_texts + df["full_text"].fillna("").tolist())

    path_emb = encoder.encode(path_texts)
    item_emb = encoder.encode(df["full_text"].fillna("").tolist())

    raw_sim_matrix = cosine_similarity(item_emb, path_emb)
    normalized_sim_matrix = zscore_rows(raw_sim_matrix)

    rows = []
    for row_idx, (_, row) in enumerate(df.iterrows()):
        raw_scores = raw_sim_matrix[row_idx]
        norm_scores = normalized_sim_matrix[row_idx].copy()

        boosts: List[float] = []
        matched_terms_per_path: List[List[str]] = []
        for path in taxonomy_paths:
            boost, matched = lexical_boost(df.iloc[row_idx]["full_text"], path)
            boosts.append(boost)
            matched_terms_per_path.append(matched)

        boosts_arr = np.asarray(boosts, dtype=float)
        final_scores = norm_scores + boosts_arr

        ranked_idx = np.argsort(final_scores)[::-1]
        top_idx = ranked_idx[0]
        second_idx = ranked_idx[1] if len(ranked_idx) > 1 else ranked_idx[0]

        assigned_path = taxonomy_paths[top_idx]
        top1_score = float(final_scores[top_idx])
        top2_score = float(final_scores[second_idx])
        margin = top1_score - top2_score
        low_confidence = margin < min_confidence_margin

        top_candidates = []
        for idx in ranked_idx[:top_k]:
            top_candidates.append({
                "path": taxonomy_paths[idx],
                "score": round(float(final_scores[idx]), 6),
                "raw_similarity": round(float(raw_scores[idx]), 6),
                "normalized_similarity": round(float(norm_scores[idx]), 6),
                "lexical_boost": round(float(boosts_arr[idx]), 6),
                "matched_terms": matched_terms_per_path[idx],
            })

        rows.append({
            "model_id": safe_str(row.get("model_id", "")),
            "model_name": safe_str(row.get("model_name", row.get("model_id", ""))),
            "description": safe_str(row.get("short_description", row.get("description", ""))),
            "assigned_ieee_path": assigned_path,
            "top1_ieee_path": taxonomy_paths[top_idx],
            "top1_score": round(top1_score, 6),
            "top2_ieee_path": taxonomy_paths[second_idx],
            "top2_score": round(top2_score, 6),
            "score_margin": round(margin, 6),
            "low_confidence": low_confidence,
            "top_k_candidates_json": json.dumps(top_candidates, ensure_ascii=False),
            "full_text_for_matching": df.iloc[row_idx]["full_text"],
        })

    out = pd.DataFrame(rows)
    return out.sort_values(["top1_score", "score_margin"], ascending=[False, False])


# ----------------------------
# CLI
# ----------------------------

if __name__ == "__main__":
    input_csv = "7-CLUSTERING_MODELS/hf_models.csv"
    ttl_path = "7-CLUSTERING_MODELS/WorkflowInputs/ieee-taxonomy-2025.ttl"
    output_csv = "7-CLUSTERING_MODELS/ieee_path_assignments.csv"

    encoder = "sbert"   # or "tfidf"
    min_margin = 0.03
    top_k = 5

    df = pd.read_csv(input_csv)

    result = assign_ieee_paths(
        df=df,
        ttl_path=Path(ttl_path),
        encoder_name=encoder,
        min_confidence_margin=min_margin,
        top_k=top_k,
    )

    result.to_csv(output_csv, index=False)
    print(f"Done! Saved to {output_csv}")
