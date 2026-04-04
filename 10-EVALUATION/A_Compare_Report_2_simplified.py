#!/usr/bin/env python3
"""
Simplified comparison report

Input CSV expected columns (at minimum):
    ModelID
    Pipeline_Features
    Instruct_Features
    ChatGPT_Features
    Gemini_Features
    United_Features
(Counts/Descriptions columns can exist; they are ignored by the core evaluation.)

Outputs:
  1) overall_metrics.csv
     - one row per comparator vs Pipeline_Features with global TP/FP/FN + precision/recall/F1 + IoU.

  2) per_model_details.csv
     - long format: one row per model per comparator per feature decision (TP/FP/FN)
       including match_variant_used + match_reason (+ score) for TP rows.
"""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import List, Optional, Tuple, Dict

import pandas as pd

# -----------------------------
# Normalization & helpers
# -----------------------------

def _norm_np(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

PAREN_RE = re.compile(r"^(.*?)\s*\(([^()]+)\)\s*$")

def extract_np_variants(s: str) -> List[str]:
    """
    If phrase has a single (...) suffix, return both outside and inside parts as variants.
    Example: "General Game Playing (GGP)" -> ["General Game Playing", "GGP"]
    Else: [s]
    """
    if not s:
        return []
    s = str(s).strip()
    m = PAREN_RE.match(s)
    if not m:
        return [s]
    outside, inside = m.group(1).strip(), m.group(2).strip()
    out: List[str] = []
    if outside:
        out.append(outside)
    if inside:
        out.append(inside)
    return out

def is_acronym(np: str) -> bool:
    if not isinstance(np, str):
        return False
    if not (2 <= len(np) <= 7):
        return False
    if not np.isalpha():
        return False
    if sum(c.isupper() for c in np) < 2:
        return False
    if np.istitle():
        return False
    return True

def ensure_list(x) -> List[str]:
    """Parse list-like cells that might be lists, NaN, JSON-ish strings, or single strings."""
    if x is None:
        return []
    if isinstance(x, float) and pd.isna(x):
        return []
    if isinstance(x, list):
        return [str(v) for v in x if str(v).strip()]
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return []
        # try JSON list first
        if (s.startswith("[") and s.endswith("]")) or (s.startswith("(") and s.endswith(")")):
            try:
                v = ast.literal_eval(s)
                if isinstance(v, list):
                    return [str(t) for t in v if str(t).strip()]
                return [str(v)] if str(v).strip() else []
            except Exception:
                pass
        # allow single label strings
        return [s]
    # iterable fallback
    try:
        return [str(v) for v in list(x) if str(v).strip()]
    except Exception:
        return [str(x)] if str(x).strip() else []

def dedup_keep_order(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for it in items:
        if it is None:
            continue
        s = str(it).strip()
        if not s:
            continue
        if s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out

def partial_containment_match(a_raw: str, b_raw: str, *, min_contained_len: int = 5) -> bool:
    """True if normalized(a) contains normalized(b) or vice versa, requiring contained string length >= min_contained_len."""
    a = _norm_np(a_raw)
    b = _norm_np(b_raw)
    if not a or not b:
        return False
    short, long = (a, b) if len(a) <= len(b) else (b, a)
    if len(short) < min_contained_len:
        return False
    return short in long

# -----------------------------
# Matching (exact -> similarity -> partial containment)
# -----------------------------

def match_pair(pred_raw: str, truth_raw: str, *, sim_threshold: float = 0.90) -> Tuple[bool, float, str, str]:
    """
    Returns (matched?, score, reason, variant_used)
    - Tries variants of pred (outside/inside parentheses)
    - Exact match on normalized strings
    - Similarity match (SequenceMatcher ratio) >= threshold
    - If still not matched: partial containment (min 5 chars) as last resort
    """
    best = (False, 0.0, "no_match", "")
    for pv in extract_np_variants(pred_raw):
        p = _norm_np(pv)
        t = _norm_np(truth_raw)
        if not p or not t:
            continue

        if p == t:
            return True, 1.0, "exact_norm", pv

        score = SequenceMatcher(None, p, t).ratio()
        if score >= sim_threshold:
            # guard against acronym weirdness by requiring exact/sim already did it; if acronym and low similarity, we won't match anyway
            return True, float(score), f"sim>={sim_threshold}", pv

        # last-resort partial containment (avoid acronym-only tricks)
        if not is_acronym(pred_raw) and not is_acronym(truth_raw):
            if partial_containment_match(pv, truth_raw, min_contained_len=5):
                # score left at 0.0 because it is a heuristic match
                best = (True, 0.0, "partial_containment", pv)

    return best

@dataclass(frozen=True)
class Match:
    i: int
    j: int
    score: float
    reason: str
    variant: str

def one_to_one_align(preds: List[str], truths: List[str], *, sim_threshold: float = 0.90) -> Tuple[List[Match], List[int], List[int]]:
    """
    Greedy one-to-one alignment:
      - build all acceptable candidate matches
      - sort: exact > sim > partial, then score desc
      - pick matches without reusing indices
    """
    preds = list(preds)
    truths = list(truths)

    candidates: List[Match] = []
    for i, p in enumerate(preds):
        for j, t in enumerate(truths):
            ok, score, reason, variant = match_pair(p, t, sim_threshold=sim_threshold)
            if ok:
                candidates.append(Match(i=i, j=j, score=score, reason=reason, variant=variant))

    def rank(reason: str) -> int:
        if reason == "exact_norm":
            return 3
        if reason.startswith("sim>="):
            return 2
        if reason == "partial_containment":
            return 1
        return 0

    candidates.sort(key=lambda m: (rank(m.reason), m.score), reverse=True)

    used_i, used_j = set(), set()
    matches: List[Match] = []
    for m in candidates:
        if m.i in used_i or m.j in used_j:
            continue
        used_i.add(m.i)
        used_j.add(m.j)
        matches.append(m)

    unmatched_preds = [i for i in range(len(preds)) if i not in used_i]
    unmatched_truths = [j for j in range(len(truths)) if j not in used_j]
    return matches, unmatched_preds, unmatched_truths

# -----------------------------
# Reporting
# -----------------------------

def compare_pipeline_to_column(
    df: pd.DataFrame,
    *,
    model_id_col: str,
    pipeline_col: str,
    other_col: str,
    sim_threshold: float = 0.90,
) -> Tuple[Dict[str, float], List[Dict[str, object]]]:
    """
    Returns:
      overall_metrics dict
      per_model_detail_rows list of dicts (TP/FP/FN rows)
    """
    TP = FP = FN = 0
    n_models = 0

    detail_rows: List[Dict[str, object]] = []

    for _, row in df.iterrows():
        n_models += 1
        model_id = row[model_id_col]

        preds = dedup_keep_order(ensure_list(row[pipeline_col]))
        truths = dedup_keep_order(ensure_list(row[other_col]))

        matches, unmatched_preds, unmatched_truths = one_to_one_align(preds, truths, sim_threshold=sim_threshold)

        TP_i = len(matches)
        FP_i = len(unmatched_preds)
        FN_i = len(unmatched_truths)

        TP += TP_i
        FP += FP_i
        FN += FN_i

        # TP rows (include matching metadata)
        for m in matches:
            detail_rows.append({
                "ModelID": model_id,
                "compare_to": other_col,
                "decision": "TP",
                "pipeline_feature": preds[m.i],
                "other_feature": truths[m.j],
                "match_variant_used": m.variant,
                "match_reason": m.reason,
                "match_score": float(m.score),
            })

        # FP rows
        for i in unmatched_preds:
            detail_rows.append({
                "ModelID": model_id,
                "compare_to": other_col,
                "decision": "FP",
                "pipeline_feature": preds[i],
                "other_feature": "",
                "match_variant_used": "",
                "match_reason": "unmatched_pipeline_feature",
                "match_score": "",
            })

        # FN rows
        for j in unmatched_truths:
            detail_rows.append({
                "ModelID": model_id,
                "compare_to": other_col,
                "decision": "FN",
                "pipeline_feature": "",
                "other_feature": truths[j],
                "match_variant_used": "",
                "match_reason": "missing_from_pipeline",
                "match_score": "",
            })

    precision = TP / (TP + FP) if (TP + FP) else 0.0
    recall = TP / (TP + FN) if (TP + FN) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    iou = TP / (TP + FP + FN) if (TP + FP + FN) else 0.0

    overall = {
        "pipeline_col": pipeline_col,
        "compare_to": other_col,
        "n_models": int(n_models),
        "TP": int(TP),
        "FP": int(FP),
        "FN": int(FN),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,  # TP/(TP+FP+FN)
        "sim_threshold": float(sim_threshold),
    }
    return overall, detail_rows

def build_reports(
    input_csv: str,
    output_metrics_csv: str,
    output_details_csv: str,
    *,
    model_id_col: str = "ModelID",
    pipeline_col: str = "Pipeline_Features",
    compare_cols: Optional[List[str]] = None,
    sim_threshold: float = 0.90,
):
    df = pd.read_csv(input_csv)

    # keep only the columns we need (ignore extras if present)
    if compare_cols is None:
        compare_cols = [
            "Instruct_Features",
            "ChatGPT_Features",
            "Gemini_Features",
            "United_Features",
        ]

    needed = [model_id_col, pipeline_col] + compare_cols
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in input CSV: {missing}")

    metrics_rows: List[Dict[str, object]] = []
    details_rows: List[Dict[str, object]] = []

    for other_col in compare_cols:
        overall, detail = compare_pipeline_to_column(
            df,
            model_id_col=model_id_col,
            pipeline_col=pipeline_col,
            other_col=other_col,
            sim_threshold=sim_threshold,
        )
        metrics_rows.append(overall)
        details_rows.extend(detail)

    metrics_df = pd.DataFrame(metrics_rows)
    # nicer numeric formatting (still machine-readable)
    for c in ["precision", "recall", "f1", "iou"]:
        metrics_df[c] = metrics_df[c].astype(float).round(6)

    details_df = pd.DataFrame(details_rows)
    # order deterministically by model then comparator then decision
    decision_order = {"TP": 0, "FP": 1, "FN": 2}
    details_df["_decision_rank"] = details_df["decision"].map(decision_order).fillna(9).astype(int)
    details_df = details_df.sort_values(["ModelID", "compare_to", "_decision_rank"]).drop(columns=["_decision_rank"])

    metrics_df.to_csv(output_metrics_csv, index=False)
    details_df.to_csv(output_details_csv, index=False)

    return metrics_df, details_df



def run_comparison(
    input_path,
    output_metrics,
    output_details,
    sim_threshold=0.90,
):
    build_reports(
        input_csv=input_path,
        output_metrics_csv=output_metrics,
        output_details_csv=output_details,
        sim_threshold=sim_threshold,
    )

if __name__ == "__main__":
    run_comparison(
        input_path="10-EVALUATION/enriched_ffs/enriched_ffs_eval_T5_united.csv",
        output_metrics="10-EVALUATION/enriched_ffs/enriched_overall_metrics.csv",
        output_details="10-EVALUATION/enriched_ffs/enriched_details.csv",
        sim_threshold=0.90,
    )