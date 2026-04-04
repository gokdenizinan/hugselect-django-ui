#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
improved_family_discovery_with_relations.py

Goal
----
Cluster foundation models into meaningful model families with higher precision.

Main improvements over the earlier version
------------------------------------------
1. No more "first surviving token" fallback for family extraction.
2. Strong validation layer to reject nonsense candidate families.
3. Discovery runs only on unresolved rows, not the full dataset.
4. Candidate promotion requires multiple signals beyond raw frequency.
5. Better separation of family / child / variant attributes.
6. More conservative fuzzy matching.
7. Optional coherence checks for discovered families.

Expected outputs
----------------
- family_assignments.csv
- family_relations.csv
- discovered_family_candidates.csv
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from statistics import median
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


# ============================================================
# Noise / stop words / variants
# ============================================================

STOP_TOKENS = {
    # generic size/version
    "base", "large", "small", "medium", "tiny", "mini", "micro", "nano",
    "xl", "xxl", "xlarge", "xxlarge",
    "v1", "v2", "v3", "v4", "v5",
    "preview", "latest", "experimental",

    # tuning / alignment / serving
    "chat", "instruct", "instruction", "sft", "rlhf", "dpo",
    "aligned", "alignment", "uncensored", "reasoning",
    "tool", "function", "functioncall",

    # quantization / format
    "hf", "fp16", "fp32", "bf16", "int4", "int8", "gguf", "awq", "gptq", "exl2",

    # sizes
    "1b", "2b", "3b", "4b", "5b", "6b", "7b", "8b", "9b", "10b",
    "11b", "12b", "13b", "14b", "15b", "20b", "22b", "27b", "30b",
    "32b", "34b", "40b", "65b", "70b", "72b", "110b",

    # data/task/modality
    "model", "models", "text", "image", "vision", "audio", "video",
    "multimodal", "vl", "vlm",
    "classifier", "classification", "generation", "generator",
    "embedding", "embeddings", "embed", "retrieval", "reranker", "ranker",
    "reward", "detector", "segmentation", "captioning", "forecasting",

    # finetune artifacts
    "finetuned", "fine", "tuned", "adapter", "lora", "merged",
    "checkpoint", "weights", "pretrained", "distilled", "moe", "dense",

    # org-ish tokens often seen in names
    "meta", "facebook", "microsoft", "google", "openai", "nvidia", "huggingface",

    # language/domain descriptors that often create fake families
    "multilingual", "english", "japanese", "chinese", "arabic",
    "medical", "finance", "legal", "code", "general",
}

BAD_FAMILY_TOKENS = set(STOP_TOKENS) | {
    "other", "unclear", "new", "best", "fast", "faster", "pro", "plus",
}

NEGATIVE_HINTS = {
    "chat": 0.05,
    "instruct": 0.05,
    "classifier": 0.10,
    "classification": 0.10,
    "embedding": 0.10,
    "reranker": 0.10,
    "ranker": 0.10,
    "awq": 0.08,
    "gptq": 0.08,
    "gguf": 0.08,
    "fp16": 0.06,
    "int8": 0.06,
    "int4": 0.06,
    "lora": 0.08,
    "adapter": 0.08,
}


# ============================================================
# Parent-child hints
# ============================================================

CHILD_HINTS = {
    "llama-2": ("LLaMA", "LLaMA 2"),
    "llama-3": ("LLaMA", "LLaMA 3"),
    "llama-3-1": ("LLaMA", "LLaMA 3.1"),
    "llama-3-2": ("LLaMA", "LLaMA 3.2"),
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
# Explicit family patterns
# Pattern-first extraction is the main precision upgrade
# ============================================================

FAMILY_PATTERNS = [
    r"(llama(?:-[0-9](?:-[0-9])?)?)",
    r"(code-llama|codellama|tinyllama)",
    r"(mistral|mixtral)",
    r"(qwen(?:[0-9](?:-[0-9])?)?)",
    r"(qwen-[0-9](?:-[0-9])?)",
    r"(phi(?:-[0-9])?)",
    r"(gemma|codegemma|recurrentgemma|pali(?:-)?gemma)",
    r"(stable-diffusion(?:-[a-z0-9]+)?)",
    r"(sdxl|sd-?1-?5|sd-?2(?:-[a-z0-9]+)?|sd-?3(?:-[a-z0-9]+)?)",
    r"(controlnet)",
    r"(bert|distilbert|roberta|xlm-roberta|deberta|albert|scibert|biobert|pubmedbert|clinicalbert)",
    r"(t5|flan-t5|mt5|byt5|bart|mbart)",
    r"(gpt2|gpt-j|gpt-neo|gpt-neox|gpt)",
    r"(clip|openclip|siglip|vit|vision-transformer|deit|beit|dino|dinov2)",
    r"(yolo(?:v[0-9]+)?|yolos)",
    r"(sam|segment-anything|mobile-sam)",
    r"(resnet|resnext|unet|u-net)",
    r"(whisper|faster-whisper|wav2vec2?|hubert|xls-r|speecht5|audioldm)",
    r"(blip(?:-2)?|llava(?:-next)?|idefics|kosmos)",
    r"(esm2?|proteinbert|protbert|openfold|alphafold|esmfold)",
    r"(graphsage|graph-attention-network|gat|graph-convolutional-network|gcn)",
    r"(timegpt|patchtst|timesfm)",
]


# ============================================================
# Seed catalog
# ============================================================

def seed_family_catalog() -> Dict[str, Dict]:
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
            "aliases": ["llama", "llama-2", "llama-3", "llama-3.1", "llama-3.2", "code-llama", "codellama", "tinyllama"],
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
            "aliases": ["qwen", "qwen2", "qwen2.5", "qwen-2", "qwen-2-5"],
            "parent": "",
            "domain_hints": ["Natural Language Processing", "Generative AI"],
        },
        "Phi": {
            "aliases": ["phi", "phi-2", "phi-3", "phi-4"],
            "parent": "",
            "domain_hints": ["Natural Language Processing", "Generative AI"],
        },
        "Gemma": {
            "aliases": ["gemma", "codegemma", "recurrentgemma", "paligemma"],
            "parent": "",
            "domain_hints": ["Natural Language Processing", "Generative AI", "Multimodal AI"],
        },
        "Stable Diffusion": {
            "aliases": ["stable-diffusion", "stable-diffusion-xl", "sdxl", "sd-1-5", "sd-2", "sd-3", "latent-diffusion"],
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
# Attribute parsing
# ============================================================

def parse_attributes(name: str) -> Dict[str, str]:
    nm = normalize_model_name(name)
    out = {
        "size_variant": "",
        "tuning_variant": "",
        "quantization_variant": "",
        "domain_variant": "",
    }

    m = re.search(r"\b(\d+(?:-\d+)?b)\b", nm)
    if m:
        out["size_variant"] = m.group(1)

    for tok in ["chat", "instruct", "base", "sft", "rlhf", "dpo"]:
        if re.search(rf"\b{re.escape(tok)}\b", nm):
            out["tuning_variant"] = tok
            break

    for tok in ["awq", "gptq", "gguf", "fp16", "fp32", "bf16", "int8", "int4", "exl2"]:
        if re.search(rf"\b{re.escape(tok)}\b", nm):
            out["quantization_variant"] = tok
            break

    for tok in ["multilingual", "medical", "legal", "finance", "code"]:
        if re.search(rf"\b{re.escape(tok)}\b", nm):
            out["domain_variant"] = tok
            break

    return out


# ============================================================
# Candidate extraction
# ============================================================

def tokens_from_name(s: str) -> List[str]:
    s = normalize_model_name(s)
    if not s:
        return []
    return [t for t in s.split("-") if t]


def remove_noise_tokens(tokens: List[str]) -> List[str]:
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
        if re.fullmatch(r"\d+(?:-\d+)?b", tok):
            continue
        out.append(tok)
    return out


def infer_child_from_name(name: str) -> Tuple[str, str]:
    nm = normalize_model_name(name)
    if not nm:
        return ("", "")

    # existing exact hints first
    for key, val in CHILD_HINTS.items():
        if key in nm:
            return val

    # stronger regex normalization
    _, parent, child = normalize_child_family_candidate(nm)
    if parent:
        return (parent, child)

    return ("", "")


def extract_candidate_from_name(name: str) -> str:
    """
    Precision-first extraction:
    - pattern-first
    - child hint aware
    - no arbitrary fallback to first token
    """
    nm = normalize_model_name(name)
    if not nm:
        return ""

    parent, child = infer_child_from_name(nm)
    if parent and child:
        for key in CHILD_HINTS:
            if key in nm:
                return key

    for pat in FAMILY_PATTERNS:
        m = re.search(pat, nm)
        if m:
            return m.group(1)

    toks = remove_noise_tokens(tokens_from_name(nm))
    joined = "-".join(toks)

    # very conservative special patterns
    for pat in [
        r"stable-diffusion(?:-[a-z0-9]+)?",
        r"latent-diffusion",
        r"segment-anything",
        r"vision-transformer",
        r"graph-attention-network",
        r"graph-convolutional-network",
        r"xlm-roberta",
        r"faster-whisper",
    ]:
        m = re.search(pat, joined)
        if m:
            return m.group(0)

    return ""


def is_valid_family_candidate(cand: str, alias_to_family: Optional[Dict[str, str]] = None) -> bool:
    c = normalize_model_name(cand)
    if looks_like_child_not_root(c):
        return False
    if not c:
        return False

    if alias_to_family and c in alias_to_family:
        return True

    if c in CHILD_HINTS:
        return True

    toks = [t for t in c.split("-") if t]
    if not toks:
        return False

    if len(c) < 3:
        return False

    if re.fullmatch(r"[0-9\-\.]+", c):
        return False

    if any(tok in BAD_FAMILY_TOKENS for tok in toks):
        return False

    if all(re.fullmatch(r"(v\d+|\d+|\d+(?:-\d+)?b)", t) for t in toks):
        return False

    # reject generic single-token roots unless seeded
    if len(toks) == 1 and toks[0] in {
        "text", "image", "audio", "vision", "base", "chat", "instruct", "model"
    }:
        return False

    return True


def extract_candidate_root(row: pd.Series, alias_to_family: Dict[str, str]) -> str:
    """
    Priority:
    1. base_model
    2. model_name
    3. model_id

    But only return valid family candidates.
    """
    for field in ["base_models", "model_name", "model_id"]:
        raw = safe_str(row.get(field, ""))
        cand = extract_candidate_from_name(raw)
        if cand and is_valid_family_candidate(cand, alias_to_family):
            return cand
    return ""

def normalize_child_family_candidate(cand: str) -> Tuple[str, str, str]:
    """
    Returns (normalized_candidate, parent_root, child_family)

    If cand looks like a child/version of a known family, normalize it here.
    Otherwise returns (cand, "", "")
    """
    c = normalize_model_name(cand)
    if not c:
        return ("", "", "")

    # -------- Stable Diffusion variants --------
    sd_patterns = [
        (r"^sd[- ]?1[- ]?5$", "Stable Diffusion", "SD 1.5"),
        (r"^sd15$", "Stable Diffusion", "SD 1.5"),
        (r"^stable-diffusion[- ]?1[- ]?5$", "Stable Diffusion", "SD 1.5"),

        (r"^sd[- ]?2[- ]?1$", "Stable Diffusion", "SD 2.1"),
        (r"^sd21$", "Stable Diffusion", "SD 2.1"),
        (r"^stable-diffusion[- ]?2[- ]?1$", "Stable Diffusion", "SD 2.1"),

        (r"^sd[- ]?3$", "Stable Diffusion", "SD 3"),
        (r"^stable-diffusion[- ]?3$", "Stable Diffusion", "SD 3"),

        (r"^sd[- ]?3[- ]?5$", "Stable Diffusion", "SD 3.5"),
        (r"^sd35$", "Stable Diffusion", "SD 3.5"),
        (r"^stable-diffusion[- ]?3[- ]?5$", "Stable Diffusion", "SD 3.5"),

        (r"^sdxl$", "Stable Diffusion", "SDXL"),
        (r"^stable-diffusion-xl$", "Stable Diffusion", "SDXL"),
    ]

    for pat, parent, child in sd_patterns:
        if re.fullmatch(pat, c):
            return (c, parent, child)

    # -------- LLaMA variants --------
    llama_patterns = [
        (r"^llama[- ]?1$", "LLaMA", "LLaMA 1"),
        (r"^llama[- ]?2$", "LLaMA", "LLaMA 2"),
        (r"^llama[- ]?3$", "LLaMA", "LLaMA 3"),
        (r"^llama[- ]?3[- ]?1$", "LLaMA", "LLaMA 3.1"),
        (r"^llama[- ]?3[- ]?2$", "LLaMA", "LLaMA 3.2"),
        (r"^code[- ]?llama$", "LLaMA", "CodeLlama"),
        (r"^codellama$", "LLaMA", "CodeLlama"),
        (r"^tinyllama$", "LLaMA", "TinyLLaMA"),
    ]

    for pat, parent, child in llama_patterns:
        if re.fullmatch(pat, c):
            return (c, parent, child)

    # -------- Qwen variants --------
    qwen_patterns = [
        (r"^qwen[- ]?2$", "Qwen", "Qwen2"),
        (r"^qwen2$", "Qwen", "Qwen2"),
        (r"^qwen[- ]?2[- ]?5$", "Qwen", "Qwen2.5"),
        (r"^qwen2[- ]?5$", "Qwen", "Qwen2.5"),
    ]

    for pat, parent, child in qwen_patterns:
        if re.fullmatch(pat, c):
            return (c, parent, child)

    # -------- Phi variants --------
    phi_patterns = [
        (r"^phi[- ]?2$", "Phi", "Phi-2"),
        (r"^phi[- ]?3$", "Phi", "Phi-3"),
        (r"^phi[- ]?4$", "Phi", "Phi-4"),
    ]

    for pat, parent, child in phi_patterns:
        if re.fullmatch(pat, c):
            return (c, parent, child)

    return (c, "", "")

# ============================================================
# Matching
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
        safe_str(row.get("base_models", "")),
        " ".join(parse_listish(row.get("tags", ""))),
    ]
    return lower_clean(" | ".join([p for p in pieces if p]))


def score_negative_hints(text: str) -> float:
    penalty = 0.0
    for tok, val in NEGATIVE_HINTS.items():
        if re.search(rf"\b{re.escape(tok)}\b", text):
            penalty += val
    return min(penalty, 0.25)


def known_family_match(row: pd.Series, catalog: Dict[str, Dict], alias_to_family: Dict[str, str]) -> FamilyMatch:
    domain = safe_str(row.get("assigned_modality", ""))
    model_name = safe_str(row.get("model_name", row.get("model_id", "")))
    model_id = safe_str(row.get("model_id", ""))
    base_model = safe_str(row.get("base_models", ""))
    text = build_row_text_for_match(row)

    fields = [base_model, model_name, model_id]
    norm_fields = [normalize_model_name(f) for f in fields if safe_str(f)]

    # strong child-family hints
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
        domain_bonus = 0.15 if (not domain_hints or domain in domain_hints) else 0.0

        aliases = [normalize_model_name(a) for a in cfg.get("aliases", [])] + [normalize_model_name(fam)]
        aliases = sorted(set([a for a in aliases if a]))

        for alias in aliases:
            if normalize_model_name(base_model) and alias in normalize_model_name(base_model):
                scores[fam] += 0.85 + domain_bonus

            if normalize_model_name(model_name) and alias in normalize_model_name(model_name):
                scores[fam] += 0.50 + domain_bonus

            if alias in text:
                scores[fam] += 0.18 + domain_bonus

    if not scores:
        return FamilyMatch()

    penalty = score_negative_hints(text)
    for fam in list(scores.keys()):
        scores[fam] = max(0.0, scores[fam] - penalty)

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    fam1, score1 = ranked[0]
    fam2, score2 = ranked[1] if len(ranked) > 1 else ("", 0.0)
    margin = score1 - score2

    if score1 >= 0.85:
        method = "known_base_model_or_strong_alias"
    elif score1 >= 0.45:
        method = "known_name_alias"
    elif score1 >= 0.25:
        method = "known_metadata_alias"
    else:
        return FamilyMatch()

    if score1 < 0.28 or margin < 0.08:
        return FamilyMatch()

    parent = catalog.get(fam1, {}).get("parent", "")
    family_root = parent if parent else fam1
    family_child = fam1 if parent else ""

    return FamilyMatch(
        family_root=family_root,
        family_child=family_child,
        method=method,
        confidence=round(min(score1, 0.99), 6),
        evidence={"top_score": score1, "margin": margin, "penalty": penalty}
    )


# ============================================================
# Canonicalization
# ============================================================

def fuzzy_threshold(s: str) -> float:
    n = len(s)
    if n <= 4:
        return 0.985
    if n <= 7:
        return 0.96
    return 0.93


def canonicalize_candidate(
    cand: str,
    alias_to_family: Dict[str, str],
    discovered_roots: Optional[Dict[str, str]] = None,
) -> Tuple[str, str]:
    c = normalize_model_name(cand)
    if not c:
        return ("", "")

    if c in alias_to_family:
        return (alias_to_family[c], "seed_alias_exact")

    if c in CHILD_HINTS:
        return (CHILD_HINTS[c][0], "child_hint_parent")

    best_family = ""
    best_score = 0.0
    for alias, fam in alias_to_family.items():
        score = seq_sim(c, alias)
        if score > best_score:
            best_family = fam
            best_score = score
    if best_score >= fuzzy_threshold(c):
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
        if best_score >= max(0.96, fuzzy_threshold(c)):
            return (best_root, "discovered_fuzzy")

    return ("", "")


# ============================================================
# Discovery statistics / coherence
# ============================================================

def token_positions_for_candidate(series: pd.Series, cand: str) -> List[int]:
    positions = []
    root_token = cand.split("-")[0]
    for x in series.fillna("").astype(str):
        toks = normalize_model_name(x).split("-")
        if root_token in toks:
            positions.append(toks.index(root_token))
    return positions


def candidate_base_model_hit_rate(df: pd.DataFrame, cand: str) -> float:
    if df.empty:
        return 0.0
    vals = df["base_models"].fillna("").astype(str).map(normalize_model_name)
    return float(vals.str.contains(re.escape(cand), regex=True).mean())


def candidate_neighbor_coherence(series: pd.Series, cand: str) -> float:
    """
    Measures how consistently nearby tokens appear around the candidate.
    Higher is better.
    """
    left = []
    right = []
    root_tok = cand.split("-")[0]

    for x in series.fillna("").astype(str):
        toks = normalize_model_name(x).split("-")
        for i, tok in enumerate(toks):
            if tok == root_tok:
                if i > 0:
                    left.append(toks[i - 1])
                if i < len(toks) - 1:
                    right.append(toks[i + 1])

    def top_share(vals: List[str]) -> float:
        vals = [v for v in vals if v and v not in STOP_TOKENS]
        if not vals:
            return 0.0
        c = Counter(vals)
        return c.most_common(1)[0][1] / max(1, len(vals))

    return max(top_share(left), top_share(right))


def candidate_is_heterogeneous(
    subset: pd.DataFrame,
    catalog: Dict[str, Dict],
    alias_to_family: Dict[str, str],
) -> bool:
    """
    If rows mapped to this candidate already belong to many unrelated seed families,
    the candidate is probably an attribute, not a family.
    """
    roots = []
    for _, row in subset.iterrows():
        km = known_family_match(row, catalog, alias_to_family)
        if km.family_root:
            roots.append(km.family_root)
    return len(set(roots)) >= 4


# ============================================================
# Promotion / discovery
# ============================================================

def promote_discovered_candidates(
    df_unresolved: pd.DataFrame,
    catalog: Dict[str, Dict],
    alias_to_family: Dict[str, str],
    min_count_global: int = 12,
    min_count_per_domain: int = 5,
    min_base_model_rate: float = 0.35,
    max_median_position: float = 2.0,
    min_coherence: float = 0.30,
) -> pd.DataFrame:
    """
    Discover new family roots only from unresolved rows.
    This is much safer than mining from the whole dataset.
    """
    work = df_unresolved.copy()
    if work.empty:
        return pd.DataFrame()

    work["candidate_root_raw"] = work.apply(lambda r: extract_candidate_root(r, alias_to_family), axis=1)
    work["candidate_root_norm"] = work["candidate_root_raw"].apply(normalize_model_name)
    work = work[work["candidate_root_norm"] != ""].copy()

    if work.empty:
        return pd.DataFrame()

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
        domain = safe_str(r["assigned_modality"])
        global_count = int(r["global_count"])
        domain_count = int(r["domain_count"])

        if not cand:
            continue

        if not is_valid_family_candidate(cand, alias_to_family):
            continue

        if global_count < min_count_global and domain_count < min_count_per_domain:
            continue

        subset = work[work["candidate_root_norm"] == cand].copy()
        if subset.empty:
            continue

        if candidate_is_heterogeneous(subset, catalog, alias_to_family):
            continue

        base_rate = candidate_base_model_hit_rate(subset, cand)
        positions = token_positions_for_candidate(subset["model_name"], cand)
        median_pos = median(positions) if positions else 999
        coherence = candidate_neighbor_coherence(subset["model_name"], cand)

        # Require more than raw frequency
        if base_rate < min_base_model_rate and median_pos > max_median_position:
            continue

        if coherence < min_coherence and global_count < (min_count_global * 2):
            continue

        
        cand_norm2, forced_parent, forced_child = normalize_child_family_candidate(cand)
        if forced_parent:
            rows.append({
                "assigned_modality": domain,
                "candidate_root_norm": cand,
                "candidate_global_count": global_count,
                "candidate_domain_count": domain_count,
                "base_model_hit_rate": round(base_rate, 4),
                "median_name_token_position": round(float(median_pos), 4) if median_pos != 999 else 999.0,
                "neighbor_coherence": round(coherence, 4),
                "canonical_or_discovered_root": forced_parent,
                "canonicalization_source": "forced_child_parent_mapping",
                "is_new_discovered_root": False,
                "suggested_parent_root": forced_parent,
                "suggested_child_family": forced_child,
            })
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
            "base_model_hit_rate": round(base_rate, 4),
            "median_name_token_position": round(float(median_pos), 4) if median_pos != 999 else 999.0,
            "neighbor_coherence": round(coherence, 4),
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
        ["is_new_discovered_root", "candidate_global_count", "candidate_domain_count", "base_model_hit_rate"],
        ascending=[False, False, False, False]
    ).drop_duplicates(
        subset=["assigned_modality", "candidate_root_norm", "canonical_or_discovered_root"]
    )

    return out


# ============================================================
# Relations
# ============================================================

def build_family_relations(
    catalog: Dict[str, Dict],
    discovered_df: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for fam, cfg in catalog.items():
        parent = safe_str(cfg.get("parent", ""))
        if parent:
            rows.append({
                "parent_family": parent,
                "child_family": fam,
                "relation_source": "seed_catalog",
                "assigned_modality": "|".join(cfg.get("domain_hints", [])),
            })

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

            cand = safe_str(r.get("candidate_root_norm", ""))
            p2, c2 = infer_child_from_name(cand)
            if p2 and c2:
                rows.append({
                    "parent_family": p2,
                    "child_family": c2,
                    "relation_source": "candidate_pattern",
                    "assigned_modality": domain,
                })

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
    rel = rel[rel["parent_family"] != rel["child_family"]].copy()
    rel = rel.sort_values(["parent_family", "child_family", "relation_source"])
    return rel


# ============================================================
# Assignment
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

        attrs = parse_attributes(" ".join([
            safe_str(row.get("model_name", "")),
            safe_str(row.get("model_id", "")),
            safe_str(row.get("base_models", "")),
        ]))
        rowd.update(attrs)

        # 1) Known family matching
        km = known_family_match(row, catalog, alias_to_family)
        if km.family_root:
            rowd["family_root"] = km.family_root
            rowd["family_child"] = km.family_child
            rowd["assignment_method"] = km.method
            rowd["family_confidence"] = km.confidence
            rowd["candidate_root_raw"] = extract_candidate_root(row, alias_to_family)
            rowd["candidate_root_norm"] = normalize_model_name(rowd["candidate_root_raw"])
            results.append(rowd)
            continue

        # 2) Candidate extraction
        cand_raw = extract_candidate_root(row, alias_to_family)
        cand_norm = normalize_model_name(cand_raw)

        # 3) Direct child-pattern
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

        # 4) Canonicalize to known/discovered
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
            rowd["family_confidence"] = 0.76 if "discovered" in source else 0.84
            rowd["candidate_root_raw"] = cand_raw
            rowd["candidate_root_norm"] = cand_norm
            results.append(rowd)
            continue

        # 5) Unresolved
        rowd["family_root"] = "Other / Unclear"
        rowd["family_child"] = ""
        rowd["assignment_method"] = "unresolved"
        rowd["family_confidence"] = 0.0
        rowd["candidate_root_raw"] = cand_raw
        rowd["candidate_root_norm"] = cand_norm
        results.append(rowd)

    return pd.DataFrame(results)


def prune_bad_discovered_roots(discovered_df: pd.DataFrame) -> pd.DataFrame:
    if discovered_df is None or discovered_df.empty:
        return discovered_df

    bad_root_patterns = [
        r"^Sd\d+.*$",
        r"^SD\d+.*$",
        r"^Llama \d+.*$",
        r"^Phi[- ]?\d+.*$",
        r"^Qwen[- ]?\d+.*$",
    ]

    keep_rows = []
    for _, row in discovered_df.iterrows():
        root = safe_str(row.get("canonical_or_discovered_root", ""))
        if any(re.fullmatch(p, root) for p in bad_root_patterns):
            # keep only if it's actually being used as a child mapping, not a root
            if safe_str(row.get("suggested_parent_root", "")):
                keep_rows.append(True)
            else:
                keep_rows.append(False)
        else:
            keep_rows.append(True)

    return discovered_df.loc[keep_rows].copy()

# ============================================================
# Two-stage pipeline
# ============================================================

def stage1_known_assignments(df: pd.DataFrame, catalog: Dict[str, Dict]) -> pd.DataFrame:
    alias_to_family = build_alias_to_family(catalog)
    rows = []

    for _, row in df.iterrows():
        km = known_family_match(row, catalog, alias_to_family)
        rows.append({
            "model_id": safe_str(row.get("model_id", "")),
            "known_family_root": km.family_root,
            "known_family_child": km.family_child,
            "known_match_method": km.method,
            "known_match_confidence": km.confidence,
        })

    return pd.DataFrame(rows)

def looks_like_child_not_root(c: str) -> bool:
    c = normalize_model_name(c)
    child_like_patterns = [
        r"^sd\d+$",
        r"^sd[- ]?\d[- ]?\d$",
        r"^sd[- ]?\d(?:[- ]?\d)?$",
        r"^llama[- ]?\d(?:[- ]?\d)?$",
        r"^qwen[- ]?\d(?:[- ]?\d)?$",
        r"^phi[- ]?\d$",
    ]
    return any(re.fullmatch(p, c) for p in child_like_patterns)

# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", required=True, help="CSV with Level 1 modalities already assigned")
    parser.add_argument("--out_assignments", default="family_assignments.csv")
    parser.add_argument("--out_relations", default="family_relations.csv")
    parser.add_argument("--out_candidates", default="discovered_family_candidates.csv")
    parser.add_argument("--min_count_global", type=int, default=12)
    parser.add_argument("--min_count_per_domain", type=int, default=5)
    parser.add_argument("--min_base_model_rate", type=float, default=0.35)
    parser.add_argument("--max_median_position", type=float, default=2.0)
    parser.add_argument("--min_coherence", type=float, default=0.30)

    args = parser.parse_args([
    "--input_csv", "7-CLUSTERING_MODELS/level1_domain_assignments_improved.csv",
    "--out_assignments", "7-CLUSTERING_MODELS/clusters_improved/family_assignments.csv",
    "--out_relations", "7-CLUSTERING_MODELS/clusters_improved/family_relations.csv",
    "--out_candidates", "7-CLUSTERING_MODELS/clusters_improved/discovered_family_candidates.csv"
])

    input_csv = Path(args.input_csv)
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    df = pd.read_csv(input_csv)
    print("uploaded csv")
    if "assigned_modality" not in df.columns:
        raise ValueError("Missing required column: assigned_modality")

    catalog = seed_family_catalog()
    alias_to_family = build_alias_to_family(catalog)

    # Stage 1: known assignments only
    known_stage = stage1_known_assignments(df, catalog)
    df_work = df.copy()

    if "model_id" in df_work.columns and "model_id" in known_stage.columns:
        df_work = df_work.merge(known_stage, on="model_id", how="left")
    else:
        # fallback index merge
        known_stage = known_stage.reset_index(drop=True)
        df_work = df_work.reset_index(drop=True)
        df_work = pd.concat([df_work, known_stage], axis=1)

    unresolved_mask = df_work["known_family_root"].fillna("").eq("")
    df_unresolved = df_work[unresolved_mask].copy()

    print(f"Loaded rows: {len(df)}")
    print(f"Known-family resolved in stage 1: {len(df) - len(df_unresolved)}")
    print(f"Unresolved rows for discovery: {len(df_unresolved)}")

    # Stage 2: discover families only from unresolved rows
    discovered_df = promote_discovered_candidates(
        df_unresolved=df_unresolved,
        catalog=catalog,
        alias_to_family=alias_to_family,
        min_count_global=args.min_count_global,
        min_count_per_domain=args.min_count_per_domain,
        min_base_model_rate=args.min_base_model_rate,
        max_median_position=args.max_median_position,
        min_coherence=args.min_coherence,
    )
    discovered_df = prune_bad_discovered_roots(discovered_df)

    assignments = assign_with_discovery(
        df=df,
        catalog=catalog,
        discovered_df=discovered_df,
    )

    relations = build_family_relations(
        catalog=catalog,
        discovered_df=discovered_df,
    )

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
        "size_variant",
        "tuning_variant",
        "quantization_variant",
        "domain_variant",
        "base_models",
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
            "base_model_hit_rate",
            "median_name_token_position",
            "neighbor_coherence",
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
    print("doing stuf")
    main()