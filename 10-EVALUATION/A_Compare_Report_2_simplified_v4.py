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


# Optional conservative fuzzy matcher (matches original behavior)
try:
    from rapidfuzz import fuzz  # type: ignore
except Exception:
    fuzz = None  # conservative fuzzy fallback disabled if rapidfuzz not installed


def np_alnum_len(s: str) -> int:
    return len(re.sub(r"[^A-Za-z0-9]", "", s or ""))


def conservative_threshold_np(s: str, base: float = 0.90) -> float:
    """
    Returns a float threshold for similarity in [0,1].
    Conservative for short phrases, slightly looser for long phrases.
    """
    L = np_alnum_len(s)
    if L <= 4:
        return max(base, 0.97)
    if L <= 7:
        return max(base, 0.95)
    if L <= 12:
        return base
    if L <= 20:
        return min(base, 0.89)
    return min(base, 0.88)


def match_pair_exact_then_sim(
    left_raw: str,
    right_raw: str,
    *,
    base_sim_threshold: float = 0.90,
    allow_conservative_fuzzy: bool = True,
) -> Tuple[bool, float, str]:
    """
    Returns (match?, score 0..1, reason) following the original logic:
      1) exact match on normalized strings
      2) SequenceMatcher similarity >= base_sim_threshold
      3) optional conservative fuzzy fallback (rapidfuzz) with dynamic cutoff
    """
    l_norm = _norm_np(left_raw)
    r_norm = _norm_np(right_raw)

    if not l_norm or not r_norm:
        return False, 0.0, "empty"

    # 1) exact (after normalization)
    if l_norm == r_norm:
        return True, 1.0, "exact_norm"

    # 2) similarity threshold (SequenceMatcher)
    sim_score = SequenceMatcher(None, l_norm, r_norm).ratio()
    if sim_score >= base_sim_threshold:
        return True, float(sim_score), "sim>=threshold"

    # 3) conservative fuzzy fallback
    if not allow_conservative_fuzzy or fuzz is None:
        return False, float(sim_score), "no_match"

    # Acronym/tech: do NOT do fuzzy-syn merges here
    if is_acronym(left_raw) or is_acronym(right_raw):
        return False, float(sim_score), "blocked_acronym_or_tech"

    L = max(np_alnum_len(left_raw), np_alnum_len(right_raw))
    scorer = fuzz.WRatio if L <= 12 else fuzz.ratio

    thr = conservative_threshold_np(left_raw, base=base_sim_threshold)  # 0..1
    cutoff = int(round(thr * 100))  # rapidfuzz uses 0..100

    rf_score = float(scorer(l_norm, r_norm))  # 0..100
    if rf_score >= cutoff:
        return True, rf_score / 100.0, f"conservative_fuzzy>=({cutoff})"

    return False, max(float(sim_score), rf_score / 100.0), f"conservative_fuzzy<({cutoff})"


def match_pair_with_abbrev_and_loosen(
    pred_raw: str,
    truth_raw: str,
    *,
    base_sim_threshold: float = 0.90,
    allow_conservative_fuzzy: bool = True,
    loosen: bool = False,
    partial_min_len: int = 5,
) -> Tuple[bool, float, str, str]:
    """
    Returns: (matched?, best_score, reason, variant_used)

    reason can be:
      - exact_norm
      - sim>=threshold
      - conservative_fuzzy>=...
      - partial_containment
    """
    pred_variants = extract_np_variants(pred_raw)  # outside + inside parentheses if present

    best_ok = False
    best_score = 0.0
    best_reason = ""
    best_variant = ""

    for pv in pred_variants:
        ok, score, reason = match_pair_exact_then_sim(
            pv,
            truth_raw,
            base_sim_threshold=base_sim_threshold,
            allow_conservative_fuzzy=allow_conservative_fuzzy,
        )

        # 4) loosened partial containment (only if strict didn't match)
        if not ok and loosen:
            if partial_containment_match(pv, truth_raw, min_contained_len=partial_min_len):
                ok = True
                score = 0.0
                reason = "partial_containment"

        if not ok:
            continue

        def rank(r: str) -> int:
            if r == "exact_norm":
                return 4
            if r.startswith("sim"):
                return 3
            if r.startswith("conservative_fuzzy"):
                return 2
            if r == "partial_containment":
                return 1
            return 0

        if (not best_ok) or (rank(reason) > rank(best_reason)) or (
            rank(reason) == rank(best_reason) and float(score) > float(best_score)
        ):
            best_ok, best_score, best_reason, best_variant = True, float(score), reason, pv

    return best_ok, best_score, best_reason, best_variant



def abbreviate_reason(reason: str) -> str:
    """Short abbreviations for reasons in the details CSV."""
    if not reason:
        return ""
    if reason == "exact_norm":
        return "EN"
    if reason.startswith("sim"):
        return "SIM"
    if reason.startswith("conservative_fuzzy>="):
        return "CF+"
    if reason.startswith("conservative_fuzzy<"):
        return "CF-"
    if reason == "partial_containment":
        return "PC"
    if reason == "blocked_acronym_or_tech":
        return "BLK"
    if reason == "empty":
        return "EMP"
    if reason == "no_match":
        return "NM"
    return reason[:8]


@dataclass(frozen=True)
class Match:
    i: int
    j: int
    score: float
    reason: str
    variant: str


def one_to_one_align(
    preds: List[str],
    truths: List[str],
    *,
    base_sim_threshold: float = 0.90,
    allow_conservative_fuzzy: bool = True,
    loosen: bool = False,
    partial_min_len: int = 5,
) -> Tuple[List[Match], List[int], List[int]]:
    """
    Greedy one-to-one alignment (matches original style):
      - build all acceptable candidate matches using match_pair_with_abbrev_and_loosen
      - sort: exact > sim > conservative_fuzzy > partial, then score desc
      - pick matches without reusing indices
    """
    preds = list(preds)
    truths = list(truths)

    candidates: List[Match] = []
    for i, p in enumerate(preds):
        for j, t in enumerate(truths):
            ok, score, reason, variant = match_pair_with_abbrev_and_loosen(
                p,
                t,
                base_sim_threshold=base_sim_threshold,
                allow_conservative_fuzzy=allow_conservative_fuzzy,
                loosen=loosen,
                partial_min_len=partial_min_len,
            )
            if ok:
                candidates.append(Match(i=i, j=j, score=float(score), reason=reason, variant=variant))

    def rank(reason: str) -> int:
        if reason == "exact_norm":
            return 4
        if reason.startswith("sim"):
            return 3
        if reason.startswith("conservative_fuzzy"):
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
    base_sim_threshold: float = 0.90,
    allow_conservative_fuzzy: bool = True,
    loosen: bool = False,
    partial_min_len: int = 5,
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

        matches, unmatched_preds, unmatched_truths = one_to_one_align(
            preds,
            truths,
            base_sim_threshold=base_sim_threshold,
            allow_conservative_fuzzy=allow_conservative_fuzzy,
            loosen=loosen,
            partial_min_len=partial_min_len,
        )

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
        "base_sim_threshold": float(base_sim_threshold),
        "allow_conservative_fuzzy": bool(allow_conservative_fuzzy),
        "loosen": bool(loosen),
        "partial_min_len": int(partial_min_len),
    }
    return overall, detail_rows


def build_wide_details(
    df: pd.DataFrame,
    *,
    model_id_col: str,
    pipeline_col: str,
    compare_cols: List[str],
    base_sim_threshold: float = 0.90,
    allow_conservative_fuzzy: bool = True,
    loosen: bool = False,
    partial_min_len: int = 5,
    include_variant_cols: bool = False,
    include_similarity_cols: bool = False,
    include_reason_cols: bool = False,
) -> pd.DataFrame:
    """
    Builds a *wide* details table per ModelID with columns for each comparator.

    Output rows are grouped per model:
      - First: one row per pipeline feature, showing matched features in each comparator column.
      - Then: 'UNMATCH' rows listing pipeline-only features and unmatched comparator features side-by-side.
    """
    # Comparator -> short prefix for column names
    prefix_map = {
        "Instruct_Features": "I",
        "ChatGPT_Features": "C",
        "Gemini_Features": "G",
        "United_Features": "U",
    }
    # Fallback prefixes for unexpected column names
    def _prefix(col: str) -> str:
        return prefix_map.get(col, re.sub(r'[^A-Za-z0-9]+', '', col)[:8] or "Other")

    rows: List[Dict[str, object]] = []

    for _, r in df.iterrows():
        model_id = r[model_id_col]
        preds = dedup_keep_order(ensure_list(r[pipeline_col]))

        # Align pipeline to each comparator independently
        per_col = {}
        for col in compare_cols:
            truths = dedup_keep_order(ensure_list(r[col]))
            matches, unmatched_preds_idx, unmatched_truths_idx = one_to_one_align(
                preds,
                truths,
                base_sim_threshold=base_sim_threshold,
                allow_conservative_fuzzy=allow_conservative_fuzzy,
                loosen=loosen,
                partial_min_len=partial_min_len,
            )
            # map pred index -> Match
            m_by_i = {m.i: m for m in matches}
            per_col[col] = {
                "truths": truths,
                "m_by_i": m_by_i,
                "unmatched_truths_idx": unmatched_truths_idx,
            }

        # Matched section: one row per pipeline feature
        pipeline_only: List[str] = []
        for i, pfeat in enumerate(preds):
            row = {"ModelID": model_id, "row_type": "MATCH", "Pipeline_feature": pfeat}

            any_match = False
            for col in compare_cols:
                pr = _prefix(col)
                truths = per_col[col]["truths"]
                m = per_col[col]["m_by_i"].get(i)
                if m is None:
                    row[col] = ""
                    if include_reason_cols:
                        row[f"{pr}_reason"] = ""
                    if include_variant_cols:
                        row[f"{pr}_variant"] = ""
                    if include_similarity_cols:
                        row[f"{pr}_sim"] = ""
                else:
                    any_match = True
                    row[col] = truths[m.j]
                    if include_reason_cols:
                        row[f"{pr}_reason"] = abbreviate_reason(m.reason)
                    if include_variant_cols:
                        row[f"{pr}_variant"] = m.variant
                    if include_similarity_cols:
                        row[f"{pr}_sim"] = float(m.score)

            if not any_match:
                pipeline_only.append(pfeat)
            if any_match:
                rows.append(row)

        # Unmatched section: list pipeline-only + each comparator's remaining truths, side-by-side
        unmatched_lists = {"Pipeline_feature": pipeline_only}
        for col in compare_cols:
            pr = _prefix(col)
            truths = per_col[col]["truths"]
            unmatched_truths = [truths[j] for j in per_col[col]["unmatched_truths_idx"]]
            unmatched_lists[col] = unmatched_truths

        max_len = max((len(v) for v in unmatched_lists.values()), default=0)
        for k in range(max_len):
            row = {"ModelID": model_id, "row_type": "UNMATCH", "Pipeline_feature": ""}
            # pipeline-only
            if k < len(unmatched_lists["Pipeline_feature"]):
                row["Pipeline_feature"] = unmatched_lists["Pipeline_feature"][k]
            # others
            for col in compare_cols:
                pr = _prefix(col)
                feats = unmatched_lists.get(col, [])
                row[col] = feats[k] if k < len(feats) else ""
                if include_reason_cols:
                    row[f"{pr}_reason"] = ""  # not applicable in unmatched rows
                if include_variant_cols:
                    row[f"{pr}_variant"] = ""
                if include_similarity_cols:
                    row[f"{pr}_sim"] = ""
            rows.append(row)

    # Rename *_reason -> smaller 'reason' prefix already short; keep as is.
    details_df = pd.DataFrame(rows)

    # Ensure stable column order
    base_cols = ["ModelID", "row_type", "Pipeline_feature"]
    other_cols = []
    for col in compare_cols:
        pr = _prefix(col)
        other_cols.append(col)
        if include_reason_cols:
            other_cols.append(f"{pr}_reason")
        if include_variant_cols:
            other_cols.append(f"{pr}_variant")
        if include_similarity_cols:
            other_cols.append(f"{pr}_sim")

    ordered = [c for c in base_cols + other_cols if c in details_df.columns] +               [c for c in details_df.columns if c not in set(base_cols + other_cols)]
    return details_df[ordered]


def build_reports(
    input_csv: str,
    output_metrics_csv: str,
    output_details_csv: str,
    *,
    model_id_col: str = "ModelID",
    pipeline_col: str = "Pipeline_Features",
    compare_cols: Optional[List[str]] = None,
    base_sim_threshold: float = 0.90,
    allow_conservative_fuzzy: bool = True,
    loosen: bool = False,
    partial_min_len: int = 5,
    include_variant_cols: bool = False,
    include_similarity_cols: bool = False,
    include_reason_cols: bool = False,
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

    for other_col in compare_cols:
        overall, _detail_long = compare_pipeline_to_column(
            df,
            model_id_col=model_id_col,
            pipeline_col=pipeline_col,
            other_col=other_col,
            base_sim_threshold=base_sim_threshold,
            allow_conservative_fuzzy=allow_conservative_fuzzy,
            loosen=loosen,
            partial_min_len=partial_min_len,
        )
        metrics_rows.append(overall)

    metrics_df = pd.DataFrame(metrics_rows)
    # nicer numeric formatting (still machine-readable)
    for c in ["precision", "recall", "f1", "iou"]:
        metrics_df[c] = metrics_df[c].astype(float).round(6)
    
    metrics_df = metrics_df.drop(columns=["base_sim_threshold","allow_conservative_fuzzy","loosen","partial_min_len"])

    # Wide details table (per model, side-by-side comparisons)
    details_df = build_wide_details(
        df,
        model_id_col=model_id_col,
        pipeline_col=pipeline_col,
        compare_cols=compare_cols,
        base_sim_threshold=base_sim_threshold,
        allow_conservative_fuzzy=allow_conservative_fuzzy,
        loosen=loosen,
        partial_min_len=partial_min_len,
        include_variant_cols=include_variant_cols,
        include_similarity_cols=include_similarity_cols,
        include_reason_cols=include_reason_cols,
    )

    details_df.loc[details_df["ModelID"] == details_df["ModelID"].shift(), "ModelID"] = ""
    # Save outputs
    import os

    out_metrics_dir = os.path.dirname(os.path.abspath(output_metrics_csv))
    out_details_dir = os.path.dirname(os.path.abspath(output_details_csv))
    if out_metrics_dir:
        os.makedirs(out_metrics_dir, exist_ok=True)
    if out_details_dir:
        os.makedirs(out_details_dir, exist_ok=True)

    metrics_df.to_csv(output_metrics_csv, index=False)
    details_df.to_csv(output_details_csv, index=False)

    return metrics_df, details_df


# ==============================
# CONFIG – EDIT THESE PATHS/FLAGS
# ==============================

FILTERED = True  
SAMPLED = True

FILL = "_filtered" if FILTERED else "_united"
FILL_SAM = "_50" if SAMPLED else ""
INPUT_CSV = f"10-EVALUATION/enriched_ffs/enriched_ffs_eval_T5{FILL}{FILL_SAM}.csv"
OUTPUT_METRICS_CSV = f"10-EVALUATION/enriched_ffs/enriched_overall_metrics{FILL}{FILL_SAM}.csv"
OUTPUT_DETAILS_CSV = f"10-EVALUATION/enriched_ffs/enriched_details{FILL}{FILL_SAM}.csv"

BASE_SIM_THRESHOLD = 0.50
ALLOW_CONSERVATIVE_FUZZY = True   # set False to disable rapidfuzz conservative fuzzy fallback
LOOSEN = True                    # set True to enable partial containment matching
PARTIAL_MIN_LEN = 3              # minimum contained length for loosened partial containment

INCLUDE_VARIANT_COLS = False      # optional: include per-source variant column (used for parentheses split)
INCLUDE_SIMILARITY_COLS = False   # optional: include per-source similarity score column
INCLUDE_READON_COLS = False         # optional: include per-source reason column (abbreviated)

COMPARE_COLS = [
    "Instruct_Features",
    "ChatGPT_Features",
    "Gemini_Features",
    "United_Features",
]


if __name__ == "__main__":
    # Run directly from this file (no terminal args needed)
    build_reports(
        input_csv=INPUT_CSV,
        output_metrics_csv=OUTPUT_METRICS_CSV,
        output_details_csv=OUTPUT_DETAILS_CSV,
        compare_cols=COMPARE_COLS,
        base_sim_threshold=BASE_SIM_THRESHOLD,
        allow_conservative_fuzzy=ALLOW_CONSERVATIVE_FUZZY,
        loosen=LOOSEN,
        partial_min_len=PARTIAL_MIN_LEN,
        include_variant_cols=INCLUDE_VARIANT_COLS,
        include_similarity_cols=INCLUDE_SIMILARITY_COLS,
        include_reason_cols=INCLUDE_READON_COLS,
    )

    print("Done.")
    print(f"Metrics: {OUTPUT_METRICS_CSV}")
    print(f"Details: {OUTPUT_DETAILS_CSV}")