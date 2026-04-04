#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Memory-optimized IEEE path assignment for foundation models.

What changed vs the original version:
- Never materializes the full model x path similarity matrix
- Encodes models in batches
- Stores embeddings as float32
- Uses top-k selection per batch
- Optional path prefiltering using lexical overlap to shrink candidate paths
- Writes output incrementally if desired

This is intended for large catalogs such as ~70k Hugging Face models.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from rdflib import Graph
from rdflib.namespace import RDF, RDFS, SKOS
from sklearn.feature_extraction.text import TfidfVectorizer
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

STOPWORDS = {
    "and", "or", "of", "for", "to", "the", "a", "an", "in", "on", "with",
    "by", "from", "using", "based", "system", "systems", "methods", "method",
    "models", "model", "learning", "machine", "artificial", "intelligence",
}


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



def build_augmented_model_text(row: pd.Series) -> str:
    full_text = build_full_text(row)
    pipeline_tag = safe_str(row.get("pipeline_tag", "")).lower()
    model_name = safe_str(row.get("model_name", row.get("model_id", ""))).lower()
    base_model = safe_str(row.get("base_model", "")).lower()
    tags = [t.lower() for t in parse_listish(row.get("tags", ""))]

    extras: List[str] = []
    if pipeline_tag in PIPELINE_TAG_SYNONYMS:
        extras.extend(PIPELINE_TAG_SYNONYMS[pipeline_tag])

    combined = " | ".join([model_name, base_model, " ".join(tags), full_text.lower()])
    for hint, synonyms in FAMILY_HINTS.items():
        if hint in combined:
            extras.extend(synonyms)

    if extras:
        full_text += " | inferred taxonomy hints: " + ", ".join(sorted(set(extras)))
    return normalize_space(full_text)



def build_path_text(path: str) -> str:
    segments = [normalize_space(x) for x in path.split(">") if normalize_space(x)]
    leaf = segments[-1] if segments else path
    leaf_tokens = [t for t in tokenize(leaf) if t not in STOPWORDS]
    parts = [f"ieee taxonomy path: {path}", f"leaf concept: {leaf}"]
    if leaf_tokens:
        parts.append("leaf keywords: " + ", ".join(leaf_tokens))
    return normalize_space(" | ".join(parts))


# ----------------------------
# Encoders
# ----------------------------

class TextEncoder:
    def fit(self, texts: List[str]):
        raise NotImplementedError

    def encode(self, texts: List[str], batch_size: int = 128):
        raise NotImplementedError


class TfidfTextEncoder(TextEncoder):
    def __init__(self):
        self.vec = TfidfVectorizer(
            max_features=40000,
            ngram_range=(1, 2),
            token_pattern=r"(?u)\b[\w\-/]+\b",
            lowercase=True,
            dtype=np.float32,
        )

    def fit(self, texts: List[str]):
        self.vec.fit(texts)
        return self

    def encode(self, texts: List[str], batch_size: int = 128):
        return normalize(self.vec.transform(texts), copy=False)


class SentenceTransformerEncoder(TextEncoder):
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)

    def fit(self, texts: List[str]):
        return self

    def encode(self, texts: List[str], batch_size: int = 128):
        emb = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return emb.astype(np.float32, copy=False)



def get_encoder(encoder_name: str) -> TextEncoder:
    if encoder_name == "sbert":
        try:
            return SentenceTransformerEncoder()
        except Exception:
            print("Warning: sentence-transformers unavailable; falling back to TF-IDF.")
            return TfidfTextEncoder()
    return TfidfTextEncoder()


# ----------------------------
# Scoring helpers
# ----------------------------


def lexical_boost(model_text: str, path: str) -> Tuple[float, List[str]]:
    model_l = model_text.lower()
    path_segments = [normalize_space(x).lower() for x in path.split(">") if normalize_space(x)]
    matched: List[str] = []
    boost = 0.0

    for seg in path_segments:
        if len(seg) >= 4 and seg in model_l:
            matched.append(seg)
            boost += 0.08

    path_tokens = [t for t in tokenize(path) if len(t) >= 4 and t not in STOPWORDS]
    token_hits = sorted({tok for tok in path_tokens if tok in model_l})
    if token_hits:
        matched.extend(token_hits)
        boost += min(0.12, 0.02 * len(token_hits))

    return min(0.20, boost), sorted(set(matched))



def build_path_term_index(paths: List[str]) -> Dict[str, List[int]]:
    index: Dict[str, List[int]] = {}
    for i, path in enumerate(paths):
        terms = sorted(set([t for t in tokenize(path) if len(t) >= 4 and t not in STOPWORDS]))
        for term in terms:
            index.setdefault(term, []).append(i)
    return index



def candidate_path_indices(model_text: str, path_term_index: Dict[str, List[int]], total_paths: int,
                           min_candidates: int = 150, max_candidates: int = 1200) -> np.ndarray:
    terms = [t for t in tokenize(model_text) if len(t) >= 4 and t not in STOPWORDS]
    hits: List[int] = []
    for t in set(terms):
        hits.extend(path_term_index.get(t, []))

    if not hits:
        return np.arange(total_paths, dtype=np.int32)

    uniq, counts = np.unique(np.asarray(hits, dtype=np.int32), return_counts=True)
    order = np.argsort(counts)[::-1]
    cand = uniq[order]

    if cand.shape[0] < min_candidates:
        return np.arange(total_paths, dtype=np.int32)
    return cand[:max_candidates]



def dense_similarity(batch_emb: np.ndarray, path_emb: np.ndarray) -> np.ndarray:
    return batch_emb @ path_emb.T


# ----------------------------
# Main assignment
# ----------------------------


def assign_ieee_paths_memory_optimized(
    df: pd.DataFrame,
    ttl_path: Path,
    encoder_name: str = "sbert",
    min_confidence_margin: float = 0.03,
    top_k: int = 5,
    batch_size: int = 128,
    use_candidate_prefilter: bool = True,
) -> pd.DataFrame:
    taxonomy_paths = build_taxonomy_path_strings(ttl_path)
    if not taxonomy_paths:
        raise ValueError(f"No IEEE taxonomy paths could be parsed from: {ttl_path}")

    df = df.copy()
    df["full_text"] = df.apply(build_augmented_model_text, axis=1)

    path_texts = [build_path_text(path) for path in taxonomy_paths]
    encoder = get_encoder(encoder_name)
    encoder.fit(path_texts + df["full_text"].fillna("").tolist())

    path_emb = encoder.encode(path_texts, batch_size=batch_size)
    if not isinstance(path_emb, np.ndarray):
        path_emb = path_emb.toarray().astype(np.float32, copy=False)
    else:
        path_emb = path_emb.astype(np.float32, copy=False)

    path_term_index = build_path_term_index(taxonomy_paths) if use_candidate_prefilter else {}

    rows = []
    texts = df["full_text"].fillna("").tolist()

    for start in range(0, len(df), batch_size):
        end = min(start + batch_size, len(df))
        batch_df = df.iloc[start:end]
        batch_texts = texts[start:end]
        batch_emb = encoder.encode(batch_texts, batch_size=batch_size)

        if not isinstance(batch_emb, np.ndarray):
            batch_emb = batch_emb.toarray().astype(np.float32, copy=False)
        else:
            batch_emb = batch_emb.astype(np.float32, copy=False)

        for local_i, (_, row) in enumerate(batch_df.iterrows()):
            model_text = batch_texts[local_i]

            if use_candidate_prefilter:
                cand_idx = candidate_path_indices(model_text, path_term_index, total_paths=len(taxonomy_paths))
            else:
                cand_idx = np.arange(len(taxonomy_paths), dtype=np.int32)

            sims = dense_similarity(batch_emb[local_i:local_i+1], path_emb[cand_idx]).reshape(-1)
            if sims.shape[0] == 0:
                continue

            # row-wise normalization without storing a full matrix
            sim_mean = float(sims.mean())
            sim_std = float(sims.std())
            if sim_std == 0.0:
                sim_std = 1.0
            norm_sims = (sims - sim_mean) / sim_std

            boosts = np.zeros_like(norm_sims, dtype=np.float32)
            matched_terms_per_local: List[List[str]] = []
            for j, pidx in enumerate(cand_idx):
                boost, matched = lexical_boost(model_text, taxonomy_paths[pidx])
                boosts[j] = boost
                matched_terms_per_local.append(matched)

            final_scores = norm_sims + boosts
            local_rank = np.argsort(final_scores)[::-1]
            best_local = local_rank[0]
            second_local = local_rank[1] if local_rank.shape[0] > 1 else local_rank[0]

            top1_idx = int(cand_idx[best_local])
            top2_idx = int(cand_idx[second_local])
            top1_score = float(final_scores[best_local])
            top2_score = float(final_scores[second_local])
            margin = top1_score - top2_score
            low_confidence = margin < min_confidence_margin

            top_candidates = []
            for lid in local_rank[:top_k]:
                global_idx = int(cand_idx[lid])
                top_candidates.append({
                    "path": taxonomy_paths[global_idx],
                    "score": round(float(final_scores[lid]), 6),
                    "raw_similarity": round(float(sims[lid]), 6),
                    "normalized_similarity": round(float(norm_sims[lid]), 6),
                    "lexical_boost": round(float(boosts[lid]), 6),
                    "matched_terms": matched_terms_per_local[lid],
                })

            rows.append({
                "model_id": safe_str(row.get("model_id", "")),
                "model_name": safe_str(row.get("model_name", row.get("model_id", ""))),
                "description": safe_str(row.get("short_description", row.get("description", ""))),
                "assigned_ieee_path": taxonomy_paths[top1_idx],
                "top1_ieee_path": taxonomy_paths[top1_idx],
                "top1_score": round(top1_score, 6),
                "top2_ieee_path": taxonomy_paths[top2_idx],
                "top2_score": round(top2_score, 6),
                "score_margin": round(margin, 6),
                "low_confidence": low_confidence,
                "candidate_pool_size": int(cand_idx.shape[0]),
                "top_k_candidates_json": json.dumps(top_candidates, ensure_ascii=False),
            })

        print(f"Processed rows {start}..{end-1} / {len(df)-1}")

    out = pd.DataFrame(rows)
    return out.sort_values(["top1_score", "score_margin"], ascending=[False, False])


if __name__ == "__main__":
    input_csv = "7-CLUSTERING_MODELS/hf_models.csv"
    ttl_path = "7-CLUSTERING_MODELS/WorkflowInputs/ieee-taxonomy-2025.ttl"
    output_csv = "7-CLUSTERING_MODELS/ieee_path_assignments_memory_optimized.csv"

    encoder = "sbert"  # or "tfidf"
    min_margin = 0.03
    top_k = 5
    batch_size = 96

    df = pd.read_csv(input_csv)
    result = assign_ieee_paths_memory_optimized(
        df=df,
        ttl_path=Path(ttl_path),
        encoder_name=encoder,
        min_confidence_margin=min_margin,
        top_k=top_k,
        batch_size=batch_size,
        use_candidate_prefilter=True,
    )
    result.to_csv(output_csv, index=False)
    print(f"Done! Saved to {output_csv}")
