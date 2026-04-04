#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
family_discovery_with_relations.py

Purpose
-------
Level 2 family assignment for large Hugging Face model inventories.

This script:
1. Matches known families using a small seed catalog
2. Extracts candidate families automatically from model names/base_model
3. Promotes frequent candidates into discovered families
4. Infers parent-child family relations
5. Outputs model-level assignments and family relations

Input CSV columns expected (flexible, but these help):
- model_id
- model_name
- description
- assigned_modality
- tags
- pipeline_tag
- base_model
- library_name
- task
- modality

Outputs
-------
- family_assignments.csv
- family_relations.csv
- discovered_family_candidates.csv
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ============================================================
# Utilities
# ============================================================

def safe_str(x) -> str:
    if x is None:
        return ""
    if isinstance(x, float) and pd.isna(x):
        return ""
    return str(x).strip()


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", safe_str(text)).strip()


def lower_clean(text: str) -> str:
    return normalize_space(text).lower()


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


def strip_org_prefix(name: str) -> str:
    name = safe_str(name)
    if "/" in name:
        return name.split("/")[-1]
    return name


def normalize_model_name(name: str) -> str:
    s = strip_org_prefix(name).lower()
    s = s.replace("_", "-")
    s = s.replace(".", "-")
    s = re.sub(r"[^a-z0-9\-\+]+", "-", s)
    s = re.sub(r"-+", "-", s).strip("-")
    return s


def title_case_slug(s: str) -> str:
    s = safe_str(s).replace("_", " ").replace("-", " ")
    s = re.sub(r"\s+", " ", s).strip()
    if not s:
        return ""
    return " ".join(w.upper() if len(w) <= 3 else w.capitalize() for w in s.split())


def seq_sim(a: str, b: str) -> float:
    return SequenceMatcher(None, safe_str(a).lower(), safe_str(b).lower()).ratio()


STOP_TOKENS = {
    "base", "large", "small", "medium", "tiny", "mini", "micro", "nano",
    "xl", "xxl", "xlarge", "xxlarge",
    "chat", "instruct", "instruction", "sft", "rlhf",
    "hf", "fp16", "fp32", "int4", "int8", "gguf", "awq", "gptq",
    "v1", "v2", "v3", "v4", "v5",
    "1b", "2b", "3b", "4b", "7b", "8b", "9b", "13b", "14b", "20b", "30b", "34b", "40b", "70b",
    "1-5", "2-1", "2-0",
    "uncased", "cased",
    "model", "models",
    "text", "image", "vision", "audio",
    "classifier", "classification", "generation", "generator",
    "finetuned", "fine", "tuned",
    "adapter", "lora", "merged",
    "checkpoint",
}


CHILD_HINTS = {
    "llama-2": ("LLaMA", "LLaMA 2"),
    "llama-3": ("LLaMA", "LLaMA 3"),
    "code-llama": ("LLaMA", "CodeLlama"),
    "codellama": ("LLaMA", "CodeLlama"),
    "tinyllama": ("LLaMA", "TinyLLaMA"),
    "stable-diffusion-xl": ("Stable Diffusion", "SDXL"),
    "sdxl": ("Stable Diffusion", "SDXL"),
    "stable-diffusion-3": ("Stable Diffusion", "SD3"),
    "stable-diffusion-2": ("Stable Diffusion", "SD 2.x"),
    "stable-diffusion-1": ("Stable Diffusion", "SD 1.x"),
    "flan-t5": ("T5", "FLAN-T5"),
    "mt5": ("T5", "mT5"),
    "byt5": ("T5", "ByT5"),
    "xlm-roberta": ("RoBERTa", "XLM-RoBERTa"),
    "deberta-v2": ("DeBERTa", "DeBERTa v2"),
    "deberta-v3": ("DeBERTa", "DeBERTa v3"),
    "mixtral": ("Mistral", "Mixtral"),
    "qwen2": ("Qwen", "Qwen2"),
    "qwen2-5": ("Qwen", "Qwen2.5"),
    "qwen-2": ("Qwen", "Qwen2"),
    "qwen-2-5": ("Qwen", "Qwen2.5"),
    "phi-2": ("Phi", "Phi-2"),
    "phi-3": ("Phi", "Phi-3"),
    "phi-4": ("Phi", "Phi-4"),
    "dinov2": ("ViT", "DINOv2"),
    "mobile-sam": ("SAM", "MobileSAM"),
}


# ============================================================
# Seed catalog
# ============================================================

def seed_family_catalog() -> Dict[str, Dict]:
    """
    Small curated seed catalog.
    This is intentionally not exhaustive.
    """
    return {
        "BERT": {
            "aliases": ["bert", "bert-base", "bert-large"],
            "parent": "",
            "domain_hints": ["Natural Language Processing"],
        },
        "RoBERTa": {
            "aliases": ["roberta", "xlm-roberta", "xlmr"],
            "parent": "BERT",
            "domain_hints": ["Natural Language Processing"],
        },
        "DistilBERT": {
            "aliases": ["distilbert"],
            "parent": "BERT",
            "domain_hints": ["Natural Language Processing"],
        },
        "DeBERTa": {
            "aliases": ["deberta", "deberta-v2", "deberta-v3"],
            "parent": "BERT",
            "domain_hints": ["Natural Language Processing"],
        },
        "ALBERT": {
            "aliases": ["albert"],
            "parent": "BERT",
            "domain_hints": ["Natural Language Processing"],
        },
        "BioBERT": {
            "aliases": ["biobert", "clinicalbert", "pubmedbert", "bluebert"],
            "parent": "BERT",
            "domain_hints": ["Natural Language Processing", "Scientific and Domain-specific AI"],
        },
        "SciBERT": {
            "aliases": ["scibert"],
            "parent": "BERT",
            "domain_hints": ["Natural Language Processing", "Scientific and Domain-specific AI"],
        },
        "T5": {
            "aliases": ["t5", "flan-t5", "mt5", "byt5"],
            "parent": "",
            "domain_hints": ["Natural Language Processing"],
        },
        "BART": {
            "aliases": ["bart", "mbart"],
            "parent": "",
            "domain_hints": ["Natural Language Processing"],
        },
        "GPT": {
            "aliases": ["gpt", "gpt2", "gpt-j", "gpt-neo", "gpt-neox"],
            "parent": "",
            "domain_hints": ["Natural Language Processing", "Generative AI"],
        },
        "LLaMA": {
            "aliases": ["llama", "llama-2", "llama-3", "code-llama", "codellama", "tinyllama"],
            "parent": "",
            "domain_hints": ["Natural Language Processing", "Generative AI"],
        },
        "Mistral": {
            "aliases": ["mistral", "mixtral"],
            "parent": "",
            "domain_hints": ["Natural Language Processing", "Generative AI"],
        },
        "Falcon": {
            "aliases": ["falcon"],
            "parent": "",
            "domain_hints": ["Natural Language Processing", "Generative AI"],
        },
        "BLOOM": {
            "aliases": ["bloom", "bloomz"],
            "parent": "",
            "domain_hints": ["Natural Language Processing", "Generative AI"],
        },
        "Qwen": {
            "aliases": ["qwen", "qwen2", "qwen2.5"],
            "parent": "",
            "domain_hints": ["Natural Language Processing", "Generative AI"],
        },
        "Phi": {
            "aliases": ["phi", "phi-2", "phi-3", "phi-4"],
            "parent": "",
            "domain_hints": ["Natural Language Processing", "Generative AI"],
        },
        "Gemma": {
            "aliases": ["gemma", "codegemma", "recurrentgemma"],
            "parent": "",
            "domain_hints": ["Natural Language Processing", "Generative AI"],
        },
        "Stable Diffusion": {
            "aliases": ["stable-diffusion", "stable-diffusion-xl", "sdxl", "sd-1-5", "sd-2", "latent-diffusion"],
            "parent": "",
            "domain_hints": ["Computer Vision", "Generative AI"],
        },
        "ControlNet": {
            "aliases": ["controlnet"],
            "parent": "Stable Diffusion",
            "domain_hints": ["Computer Vision", "Generative AI"],
        },
        "ViT": {
            "aliases": ["vit", "vision-transformer", "deit", "beit", "dino", "dinov2"],
            "parent": "",
            "domain_hints": ["Computer Vision"],
        },
        "CLIP": {
            "aliases": ["clip", "openclip", "siglip"],
            "parent": "",
            "domain_hints": ["Computer Vision", "Multimodal AI"],
        },
        "YOLO": {
            "aliases": ["yolo", "yolov5", "yolov8", "yolos"],
            "parent": "",
            "domain_hints": ["Computer Vision"],
        },
        "SAM": {
            "aliases": ["segment-anything", "sam", "mobile-sam"],
            "parent": "",
            "domain_hints": ["Computer Vision"],
        },
        "ResNet": {
            "aliases": ["resnet", "resnext"],
            "parent": "",
            "domain_hints": ["Computer Vision"],
        },
        "U-Net": {
            "aliases": ["unet", "u-net"],
            "parent": "",
            "domain_hints": ["Computer Vision"],
        },
        "Whisper": {
            "aliases": ["whisper", "faster-whisper"],
            "parent": "",
            "domain_hints": ["Speech and Audio", "Generative AI"],
        },
        "wav2vec": {
            "aliases": ["wav2vec", "wav2vec2", "hubert", "xls-r"],
            "parent": "",
            "domain_hints": ["Speech and Audio"],
        },
        "SpeechT5": {
            "aliases": ["speecht5"],
            "parent": "",
            "domain_hints": ["Speech and Audio"],
        },
        "AudioLDM": {
            "aliases": ["audioldm"],
            "parent": "",
            "domain_hints": ["Speech and Audio", "Generative AI"],
        },
        "BLIP": {
            "aliases": ["blip", "blip-2"],
            "parent": "",
            "domain_hints": ["Multimodal AI"],
        },
        "LLaVA": {
            "aliases": ["llava", "llava-next"],
            "parent": "LLaMA",
            "domain_hints": ["Multimodal AI"],
        },
        "IDEFICS": {
            "aliases": ["idefics"],
            "parent": "",
            "domain_hints": ["Multimodal AI"],
        },
        "Kosmos": {
            "aliases": ["kosmos"],
            "parent": "",
            "domain_hints": ["Multimodal AI"],
        },
        "ESM": {
            "aliases": ["esm", "esm2", "proteinbert", "protbert"],
            "parent": "",
            "domain_hints": ["Scientific and Domain-specific AI"],
        },
        "AlphaFold-like": {
            "aliases": ["alphafold", "openfold", "esmfold"],
            "parent": "",
            "domain_hints": ["Scientific and Domain-specific AI"],
        },
        "GraphSAGE": {
            "aliases": ["graphsage"],
            "parent": "",
            "domain_hints": ["Graph and Relational Learning"],
        },
        "GAT": {
            "aliases": ["gat", "graph-attention-network"],
            "parent": "",
            "domain_hints": ["Graph and Relational Learning"],
        },
        "GCN": {
            "aliases": ["gcn", "graph-convolutional-network"],
            "parent": "",
            "domain_hints": ["Graph and Relational Learning"],
        },
        "TimeGPT": {
            "aliases": ["timegpt"],
            "parent": "",
            "domain_hints": ["Time Series and Forecasting"],
        },
        "PatchTST": {
            "aliases": ["patchtst"],
            "parent": "",
            "domain_hints": ["Time Series and Forecasting"],
        },
        "TimesFM": {
            "aliases": ["timesfm"],
            "parent": "",
            "domain_hints": ["Time Series and Forecasting"],
        },
    }


def build_alias_to_family(catalog: Dict[str, Dict]) -> Dict[str, str]:
    alias_to_family = {}
    for fam, cfg in catalog.items():
        alias_to_family[normalize_model_name(fam)] = fam
        for alias in cfg.get("aliases", []):
            alias_to_family[normalize_model_name(alias)] = fam
    return alias_to_family


# ============================================================
# Candidate extraction
# ============================================================

def tokens_from_name(s: str) -> List[str]:
    s = normalize_model_name(s)
    if not s:
        return []
    return [t for t in s.split("-") if t]


def remove_trailing_noise(tokens: List[str]) -> List[str]:
    out = []
    for tok in tokens:
        if tok in STOP_TOKENS:
            continue
        if re.fullmatch(r"\d+", tok):
            continue
        if re.fullmatch(r"v\d+", tok):
            continue
        if re.fullmatch(r"\d+x\d+", tok):
            continue
        out.append(tok)
    return out


def extract_candidate_from_name(name: str) -> str:
    """
    Heuristic root extraction from model name.
    """
    nm = normalize_model_name(name)
    toks = remove_trailing_noise(tokens_from_name(nm))

    if not toks:
        return ""

    joined = "-".join(toks)

    special_patterns = [
        r"stable-diffusion(?:-[a-z0-9]+)?",
        r"latent-diffusion",
        r"segment-anything",
        r"vision-transformer",
        r"graph-attention-network",
        r"graph-convolutional-network",
        r"decision-transformer",
        r"xlm-roberta",
        r"faster-whisper",
        r"code-llama",
    ]
    for pat in special_patterns:
        m = re.search(pat, joined)
        if m:
            return m.group(0)

    # Prefer first 1-2 meaningful tokens
    if len(toks) >= 2:
        first2 = f"{toks[0]}-{toks[1]}"
        if first2 in CHILD_HINTS:
            return first2

    return toks[0]


def extract_candidate_root(row: pd.Series) -> str:
    """
    Priority:
    1. base_model
    2. model_name
    3. model_id
    """
    for field in ["base_model", "model_name", "model_id"]:
        candidate = extract_candidate_from_name(safe_str(row.get(field, "")))
        if candidate:
            return candidate
    return ""


def infer_child_from_name(name: str) -> Tuple[str, str]:
    """
    Returns (parent_root, child_family) if strong child pattern found.
    """
    nm = normalize_model_name(name)
    for key, val in CHILD_HINTS.items():
        if key in nm:
            return val
    return ("", "")


# ============================================================
# Known family matching
# ============================================================

@dataclass
class FamilyMatch:
    family_root: str = ""
    family_child: str = ""
    method: str = ""
    confidence: float = 0.0
    evidence: Optional[Dict] = None


def build_row_text_for_match(row: pd.Series) -> str:
    pieces = [
        safe_str(row.get("model_id", "")),
        safe_str(row.get("model_name", "")),
        safe_str(row.get("short_description", "")),
        safe_str(row.get("pipeline_tag", "")),
        safe_str(row.get("library_name", "")),
        safe_str(row.get("model_type", "")),
        safe_str(row.get("base_model", "")),
        " ".join(parse_listish(row.get("tags", ""))),
    ]
    return lower_clean(" | ".join([p for p in pieces if p]))


def known_family_match(row: pd.Series, catalog: Dict[str, Dict], alias_to_family: Dict[str, str]) -> FamilyMatch:
    domain = safe_str(row.get("assigned_modality", ""))
    model_name = safe_str(row.get("model_name", row.get("model_id", "")))
    model_id = safe_str(row.get("model_id", ""))
    base_model = safe_str(row.get("base_model", ""))
    text = build_row_text_for_match(row)

    fields = [base_model, model_name, model_id]
    norm_fields = [normalize_model_name(f) for f in fields if safe_str(f)]

    # 1. strong child-family hints
    for f in norm_fields:
        parent, child = infer_child_from_name(f)
        if parent:
            return FamilyMatch(
                family_root=parent,
                family_child=child,
                method="known_child_pattern",
                confidence=0.97,
                evidence={"matched_on": f}
            )

    scores = defaultdict(float)

    for fam, cfg in catalog.items():
        domain_hints = cfg.get("domain_hints", [])
        domain_bonus = 0.08 if (not domain_hints or domain in domain_hints) else 0.0

        aliases = [normalize_model_name(a) for a in cfg.get("aliases", [])] + [normalize_model_name(fam)]
        aliases = list(sorted(set([a for a in aliases if a])))

        for alias in aliases:
            if not alias:
                continue

            # strongest: exact or contained in base_model
            if normalize_model_name(base_model) and alias in normalize_model_name(base_model):
                scores[fam] += 0.75 + domain_bonus

            # strong: model name
            if normalize_model_name(model_name) and alias in normalize_model_name(model_name):
                scores[fam] += 0.45 + domain_bonus

            # weaker: anywhere in metadata text
            if alias in text:
                scores[fam] += 0.18 + domain_bonus

    if not scores:
        return FamilyMatch()

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    fam1, score1 = ranked[0]
    fam2, score2 = ranked[1] if len(ranked) > 1 else ("", 0.0)
    margin = score1 - score2

    if score1 >= 0.75:
        method = "known_base_model_or_strong_alias"
    elif score1 >= 0.40:
        method = "known_name_alias"
    elif score1 >= 0.20:
        method = "known_metadata_alias"
    else:
        return FamilyMatch()

    if score1 < 0.22 or margin < 0.05:
        return FamilyMatch()

    parent = catalog.get(fam1, {}).get("parent", "")
    family_root = parent if parent else fam1
    family_child = fam1 if parent else ""

    return FamilyMatch(
        family_root=family_root,
        family_child=family_child,
        method=method,
        confidence=round(min(score1, 0.99), 6),
        evidence={"top_score": score1, "margin": margin}
    )


# ============================================================
# Candidate promotion / canonicalization
# ============================================================

def canonicalize_candidate(
    cand: str,
    alias_to_family: Dict[str, str],
    discovered_roots: Optional[Dict[str, str]] = None,
) -> Tuple[str, str]:
    """
    Returns (canonical_root, source)
    """
    c = normalize_model_name(cand)
    if not c:
        return ("", "")

    if c in alias_to_family:
        return (alias_to_family[c], "seed_alias_exact")

    # child hint to parent
    if c in CHILD_HINTS:
        return (CHILD_HINTS[c][0], "child_hint_parent")

    # close to seed aliases
    best_alias = ""
    best_family = ""
    best_score = 0.0
    for alias, fam in alias_to_family.items():
        score = seq_sim(c, alias)
        if score > best_score:
            best_alias = alias
            best_family = fam
            best_score = score
    if best_score >= 0.90:
        return (best_family, "seed_alias_fuzzy")

    if discovered_roots:
        if c in discovered_roots:
            return (discovered_roots[c], "discovered_exact")
        best_root = ""
        best_score = 0.0
        for k, v in discovered_roots.items():
            score = seq_sim(c, k)
            if score > best_score:
                best_root = v
                best_score = score
        if best_score >= 0.92:
            return (best_root, "discovered_fuzzy")

    return ("", "")


def promote_discovered_candidates(
    df: pd.DataFrame,
    alias_to_family: Dict[str, str],
    min_count_global: int = 8,
    min_count_per_domain: int = 4,
) -> pd.DataFrame:
    """
    Builds discovered family candidates from extracted roots.
    """
    work = df.copy()
    work["candidate_root_raw"] = work.apply(extract_candidate_root, axis=1)
    work["candidate_root_norm"] = work["candidate_root_raw"].apply(normalize_model_name)

    rows = []
    domain_counts = (
        work.groupby(["assigned_modality", "candidate_root_norm"])
        .size()
        .reset_index(name="domain_count")
    )
    global_counts = (
        work.groupby("candidate_root_norm")
        .size()
        .reset_index(name="global_count")
    )

    merged = domain_counts.merge(global_counts, on="candidate_root_norm", how="left")

    for _, r in merged.iterrows():
        cand = safe_str(r["candidate_root_norm"])
        if not cand:
            continue

        global_count = int(r["global_count"])
        domain_count = int(r["domain_count"])
        domain = safe_str(r["assigned_modality"])

        if global_count < min_count_global and domain_count < min_count_per_domain:
            continue

        canonical_root, source = canonicalize_candidate(cand, alias_to_family, discovered_roots=None)

        if canonical_root:
            label = canonical_root
            promoted = False
        else:
            label = title_case_slug(cand)
            promoted = True

        parent, child = infer_child_from_name(cand)

        rows.append({
            "assigned_modality": domain,
            "candidate_root_norm": cand,
            "candidate_global_count": global_count,
            "candidate_domain_count": domain_count,
            "canonical_or_discovered_root": label,
            "canonicalization_source": source if source else "new_discovered_root",
            "is_new_discovered_root": promoted,
            "suggested_parent_root": parent,
            "suggested_child_family": child,
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    out = out.sort_values(
        ["is_new_discovered_root", "candidate_global_count", "candidate_domain_count"],
        ascending=[False, False, False]
    ).drop_duplicates(
        subset=["assigned_modality", "candidate_root_norm", "canonical_or_discovered_root"]
    )
    return out


# ============================================================
# Family relations
# ============================================================

def build_family_relations(
    catalog: Dict[str, Dict],
    discovered_df: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    # Seed relations
    for fam, cfg in catalog.items():
        parent = safe_str(cfg.get("parent", ""))
        if parent:
            rows.append({
                "parent_family": parent,
                "child_family": fam,
                "relation_source": "seed_catalog",
                "assigned_modality": "|".join(cfg.get("domain_hints", [])),
            })

    # Child-hint relations from discovered candidates
    if discovered_df is not None and not discovered_df.empty:
        for _, r in discovered_df.iterrows():
            root = safe_str(r["canonical_or_discovered_root"])
            parent = safe_str(r.get("suggested_parent_root", ""))
            child = safe_str(r.get("suggested_child_family", ""))
            domain = safe_str(r.get("assigned_modality", ""))

            if parent and child:
                rows.append({
                    "parent_family": parent,
                    "child_family": child,
                    "relation_source": "discovered_child_hint",
                    "assigned_modality": domain,
                })

            # If discovered root itself looks like a child of a known parent
            cand = safe_str(r.get("candidate_root_norm", ""))
            p2, c2 = infer_child_from_name(cand)
            if p2 and c2:
                rows.append({
                    "parent_family": p2,
                    "child_family": c2,
                    "relation_source": "candidate_pattern",
                    "assigned_modality": domain,
                })

            # If the discovered root label is not the same as parent and a parent exists
            if parent and root and root != parent and root != child:
                rows.append({
                    "parent_family": parent,
                    "child_family": root,
                    "relation_source": "discovered_root_parent_inference",
                    "assigned_modality": domain,
                })

    if not rows:
        return pd.DataFrame(columns=["parent_family", "child_family", "relation_source", "assigned_modality"])

    rel = pd.DataFrame(rows).drop_duplicates()

    # Remove invalid self-links
    rel = rel[rel["parent_family"] != rel["child_family"]].copy()

    # Sort
    rel = rel.sort_values(["parent_family", "child_family", "relation_source"])
    return rel


# ============================================================
# Final assignment
# ============================================================

def build_discovered_root_lookup(discovered_df: pd.DataFrame) -> Dict[str, str]:
    lookup = {}
    if discovered_df is None or discovered_df.empty:
        return lookup

    for _, r in discovered_df.iterrows():
        cand = normalize_model_name(r["candidate_root_norm"])
        root = safe_str(r["canonical_or_discovered_root"])
        if cand and root:
            lookup[cand] = root
    return lookup


def assign_with_discovery(
    df: pd.DataFrame,
    catalog: Dict[str, Dict],
    discovered_df: pd.DataFrame,
) -> pd.DataFrame:
    alias_to_family = build_alias_to_family(catalog)
    discovered_lookup = build_discovered_root_lookup(discovered_df)

    rel_hint_map = {}
    if discovered_df is not None and not discovered_df.empty:
        for _, r in discovered_df.iterrows():
            cand = normalize_model_name(r["candidate_root_norm"])
            rel_hint_map[cand] = {
                "parent": safe_str(r.get("suggested_parent_root", "")),
                "child": safe_str(r.get("suggested_child_family", "")),
                "root": safe_str(r.get("canonical_or_discovered_root", "")),
            }

    results = []

    for _, row in df.iterrows():
        rowd = row.to_dict()

        # 1) seed catalog known-family matching
        km = known_family_match(row, catalog, alias_to_family)
        if km.family_root:
            rowd["family_root"] = km.family_root
            rowd["family_child"] = km.family_child
            rowd["assignment_method"] = km.method
            rowd["family_confidence"] = km.confidence
            rowd["candidate_root_raw"] = extract_candidate_root(row)
            rowd["candidate_root_norm"] = normalize_model_name(rowd["candidate_root_raw"])
            results.append(rowd)
            continue

        # 2) candidate extraction
        cand_raw = extract_candidate_root(row)
        cand_norm = normalize_model_name(cand_raw)

        # child pattern direct
        p, c = infer_child_from_name(cand_norm)
        if p:
            rowd["family_root"] = p
            rowd["family_child"] = c
            rowd["assignment_method"] = "candidate_child_pattern"
            rowd["family_confidence"] = 0.90
            rowd["candidate_root_raw"] = cand_raw
            rowd["candidate_root_norm"] = cand_norm
            results.append(rowd)
            continue

        # 3) canonicalize to seed/discovered root
        root, source = canonicalize_candidate(cand_norm, alias_to_family, discovered_lookup)

        if root:
            hint = rel_hint_map.get(cand_norm, {})
            child = safe_str(hint.get("child", ""))
            parent = safe_str(hint.get("parent", ""))

            if parent and not child and root != parent:
                child = root
                root = parent

            rowd["family_root"] = root
            rowd["family_child"] = child
            rowd["assignment_method"] = f"candidate_{source}"
            rowd["family_confidence"] = 0.78 if "discovered" in source else 0.84
            rowd["candidate_root_raw"] = cand_raw
            rowd["candidate_root_norm"] = cand_norm
            results.append(rowd)
            continue

        # 4) unresolved fallback
        rowd["family_root"] = "Other / Unclear"
        rowd["family_child"] = ""
        rowd["assignment_method"] = "unresolved"
        rowd["family_confidence"] = 0.0
        rowd["candidate_root_raw"] = cand_raw
        rowd["candidate_root_norm"] = cand_norm
        results.append(rowd)

    out = pd.DataFrame(results)
    return out


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", required=True, help="CSV with Level 1 domains already assigned")
    parser.add_argument("--out_assignments", default="family_assignments.csv")
    parser.add_argument("--out_relations", default="family_relations.csv")
    parser.add_argument("--out_candidates", default="discovered_family_candidates.csv")
    parser.add_argument("--min_count_global", type=int, default=8)
    parser.add_argument("--min_count_per_domain", type=int, default=4)
    args = parser.parse_args([
    "--input_csv", "7-CLUSTERING_MODELS/model_classification_improved_more.csv",
    "--out_assignments", "7-CLUSTERING_MODELS/clusters/family_assignments.csv",
    "--out_relations", "7-CLUSTERING_MODELS/clusters/family_relations.csv",
    "--out_candidates", "7-CLUSTERING_MODELS/clusters/discovered_family_candidates.csv"
])

    input_csv = Path(args.input_csv)
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    df = pd.read_csv(input_csv)
    print("loaded csv!")
    required = ["assigned_modality"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    catalog = seed_family_catalog()
    alias_to_family = build_alias_to_family(catalog)

    discovered_df = promote_discovered_candidates(
        df=df,
        alias_to_family=alias_to_family,
        min_count_global=args.min_count_global,
        min_count_per_domain=args.min_count_per_domain,
    )

    assignments = assign_with_discovery(
        df=df,
        catalog=catalog,
        discovered_df=discovered_df,
    )

    relations = build_family_relations(
        catalog=catalog,
        discovered_df=discovered_df,
    )

    # Reorder columns
    preferred = [
        "model_id",
        "model_name",
        "assigned_modality",
        "family_root",
        "family_child",
        "assignment_method",
        "family_confidence",
        "candidate_root_raw",
        "candidate_root_norm",
        "base_model",
        "pipeline_tag",
        "library_name",
        "model_type",
        "tags",
        "short_description",
    ]
    existing = [c for c in preferred if c in assignments.columns]
    others = [c for c in assignments.columns if c not in existing]
    assignments = assignments[existing + others]

    assignments.to_csv(args.out_assignments, index=False)
    relations.to_csv(args.out_relations, index=False)

    if discovered_df is None or discovered_df.empty:
        pd.DataFrame(columns=[
            "assigned_modality",
            "candidate_root_norm",
            "candidate_global_count",
            "candidate_domain_count",
            "canonical_or_discovered_root",
            "canonicalization_source",
            "is_new_discovered_root",
            "suggested_parent_root",
            "suggested_child_family",
        ]).to_csv(args.out_candidates, index=False)
    else:
        discovered_df.to_csv(args.out_candidates, index=False)

    print(f"Saved assignments: {args.out_assignments} ({len(assignments)} rows)")
    print(f"Saved relations:   {args.out_relations} ({len(relations)} rows)")
    print(f"Saved candidates:  {args.out_candidates} ({0 if discovered_df is None else len(discovered_df)} rows)")


if __name__ == "__main__":
    print("doing stuff")
    main()