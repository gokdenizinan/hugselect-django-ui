from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
import pandas as pd
import random
from typing import List, Tuple, Dict, Optional


# ----------------------------
# Config
# ----------------------------

@dataclass
class MatchConfig:
    sim_threshold: float = 0.50          # similarity threshold for fuzzy match
    normalize_lower: bool = True
    strip: bool = True

# ----------------------------
# Helpers: parsing + matching
# ----------------------------

def _norm_token(s: str, cfg: MatchConfig) -> str:
    if s is None:
        return ""
    s = str(s)
    if cfg.strip:
        s = s.strip()
    if cfg.normalize_lower:
        s = s.lower()
    return s

def split_features(cell) -> List[str]:
    """
    Robust-ish splitter for common formats:
      - list-like strings: "['a', 'b']"
      - comma separated: "a, b"
      - semicolon separated: "a; b"
      - newline separated
    Adjust if your format is different.
    """
    if cell is None or (isinstance(cell, float) and pd.isna(cell)):
        return []
    s = str(cell).strip()
    if not s:
        return []

    # list-like
    if s.startswith("[") and s.endswith("]"):
        # very light parsing: remove brackets, split on commas
        inner = s[1:-1].strip()
        if not inner:
            return []
        parts = [p.strip().strip("'").strip('"') for p in inner.split(",")]
        return [p for p in parts if p]

    # delimiter guessing
    if "\n" in s:
        parts = [p.strip() for p in s.splitlines()]
        return [p for p in parts if p]
    if ";" in s:
        parts = [p.strip() for p in s.split(";")]
        return [p for p in parts if p]
    if "," in s:
        parts = [p.strip() for p in s.split(",")]
        return [p for p in parts if p]

    return [s]

def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()

def greedy_match(
    predicted: List[str],
    truth: List[str],
    cfg: MatchConfig
) -> Tuple[List[dict], List[str], List[str]]:
    """
    Greedy 1-1 matching.
    Returns:
      - matches: list of dicts {pred, truth, sim}
      - fps: predicted unmatched
      - fns: truth unmatched
    """
    pred = predicted[:]
    tru = truth[:]
    used_truth = set()
    matches = []

    for p in pred:
        best_j = None
        best_sim = -1.0
        for j, t in enumerate(tru):
            if j in used_truth:
                continue
            sim = similarity(p, t)
            if sim > best_sim:
                best_sim = sim
                best_j = j
        if best_j is not None and best_sim >= cfg.sim_threshold:
            used_truth.add(best_j)
            matches.append({"pred": p, "truth": tru[best_j], "sim": round(best_sim, 3)})

    matched_pred = {m["pred"] for m in matches}
    matched_truth = {m["truth"] for m in matches}

    fps = [p for p in pred if p not in matched_pred]
    fns = [t for t in tru if t not in matched_truth]
    return matches, fps, fns


# ----------------------------
# Metrics
# ----------------------------

def compute_metrics(tp: int, fp: int, fn: int) -> Dict[str, float]:
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec  = tp / (tp + fn) if (tp + fn) else 0.0
    f1   = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
    jacc = tp / (tp + fp + fn) if (tp + fp + fn) else 0.0
    return {
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "jaccard": round(jacc, 4),
    }


# ----------------------------
# Sampling logic
# ----------------------------

def sample_with_overlap(
    ids2: List[str],
    ids5: List[str],
    n: int,
    seed: int
) -> Tuple[List[str], List[str], List[str]]:
    """
    Picks n IDs for set2 and n IDs for set5.
    Forces overlap IDs into both samples (as many as possible, up to n).
    Returns (sample2, sample5, overlap_used)
    """
    rng = random.Random(seed)
    s2 = list(dict.fromkeys(ids2))
    s5 = list(dict.fromkeys(ids5))
    overlap = sorted(set(s2).intersection(set(s5)))

    overlap_used = overlap[:n]  # include as many as fit

    def fill(base_ids: List[str], forced: List[str]) -> List[str]:
        remaining = [i for i in base_ids if i not in forced]
        rng.shuffle(remaining)
        out = forced[:] + remaining
        return out[:n]

    sample2 = fill(s2, overlap_used)
    sample5 = fill(s5, overlap_used)
    return sample2, sample5, overlap_used


# ----------------------------
# Main report builder
# ----------------------------

def build_report_for_feature_set(
    pred_csv_2: str,
    pred_csv_5: str,
    strained_truth_csv_2: str,
    strained_truth_csv_5: str,
    out_csv: str,
    *,
    model_id_col: str = "ModelID",
    pred_col: str = "Features",          # adjust if your column is "Features"
    truth_col: str = "truth",
    strained_col_out_name: str = "truth_strained",
    n_samples: int = 10,
    seed: int = 7,
    cfg: MatchConfig = MatchConfig(),
) -> None:
    # Load
    df2 = pd.read_csv(pred_csv_2)
    df5 = pd.read_csv(pred_csv_5)

    # Load strained truth and join it in
    st2 = pd.read_csv(strained_truth_csv_2)[[model_id_col, truth_col]].rename(columns={truth_col: strained_col_out_name})
    st5 = pd.read_csv(strained_truth_csv_5)[[model_id_col, truth_col]].rename(columns={truth_col: strained_col_out_name})

    df2 = df2.merge(st2, on=model_id_col, how="left")
    df5 = df5.merge(st5, on=model_id_col, how="left")

    # Sample model IDs with overlap forced in
    ids2 = df2[model_id_col].astype(str).tolist()
    ids5 = df5[model_id_col].astype(str).tolist()
    sample2, sample5, overlap_used = sample_with_overlap(ids2, ids5, n_samples, seed)

    # Subset to samples
    df2s = df2[df2[model_id_col].astype(str).isin(sample2)].copy()
    df5s = df5[df5[model_id_col].astype(str).isin(sample5)].copy()

    # For consistent ordering in output
    order2 = {mid: i for i, mid in enumerate(sample2)}
    order5 = {mid: i for i, mid in enumerate(sample5)}
    df2s["_ord"] = df2s[model_id_col].astype(str).map(order2)
    df5s["_ord"] = df5s[model_id_col].astype(str).map(order5)
    df2s = df2s.sort_values("_ord").drop(columns=["_ord"])
    df5s = df5s.sort_values("_ord").drop(columns=["_ord"])

    # Expand into report rows
    report_rows = []

    def add_eval_block(df: pd.DataFrame, eval_name: str):
        nonlocal report_rows

        # Overall counts over *sampled* models (you can change to full df if you want)
        total_tp = total_fp = total_fn = 0

        for _, r in df.iterrows():
            mid = str(r[model_id_col])

            pred_list = [_norm_token(x, cfg) for x in split_features(r.get(pred_col))]
            truth_list = [_norm_token(x, cfg) for x in split_features(r.get(truth_col))]
            strained_list = [_norm_token(x, cfg) for x in split_features(r.get(strained_col_out_name))]

            # remove empties
            pred_list = [x for x in pred_list if x]
            truth_list = [x for x in truth_list if x]
            strained_list = [x for x in strained_list if x]

            matches, fps, fns = greedy_match(pred_list, truth_list, cfg)

            tp = len(matches)
            fp = len(fps)
            fn = len(fns)
            total_tp += tp
            total_fp += fp
            total_fn += fn
            m = compute_metrics(tp, fp, fn)

            # A "header" row for the model (holds per-model metrics)
            report_rows.append({
                "eval_set": eval_name,
                "modelID": mid,
                "row_type": "MODEL_SUMMARY",
                "pred_feature": "",
                "truth_feature": "",
                "truth_strained_feature": "",
                "matched_pred": "",
                "matched_truth": "",
                "match_sim": "",
                "TP": "",
                "FP": "",
                "FN": "",
                "model_precision": m["precision"],
                "model_recall": m["recall"],
                "model_f1": m["f1"],
                "model_jaccard": m["jaccard"],
            })

            # TP rows (aligned items shown next to each other)
            for mm in matches:
                report_rows.append({
                    "eval_set": eval_name,
                    "modelID": mid,
                    "row_type": "TP",
                    "pred_feature": mm["pred"],
                    "truth_feature": mm["truth"],
                    "truth_strained_feature": "",  # we’ll show strained on its own block below
                    "matched_pred": mm["pred"],
                    "matched_truth": mm["truth"],
                    "match_sim": mm["sim"],
                    "TP": "x",
                    "FP": "",
                    "FN": "",
                    "model_precision": "",
                    "model_recall": "",
                    "model_f1": "",
                    "model_jaccard": "",
                })

            # FP rows
            for p in fps:
                report_rows.append({
                    "eval_set": eval_name,
                    "modelID": mid,
                    "row_type": "FP",
                    "pred_feature": p,
                    "truth_feature": "",
                    "truth_strained_feature": "",
                    "matched_pred": "",
                    "matched_truth": "",
                    "match_sim": "",
                    "TP": "",
                    "FP": "x",
                    "FN": "",
                    "model_precision": "",
                    "model_recall": "",
                    "model_f1": "",
                    "model_jaccard": "",
                })

            # FN rows
            for t in fns:
                report_rows.append({
                    "eval_set": eval_name,
                    "modelID": mid,
                    "row_type": "FN",
                    "pred_feature": "",
                    "truth_feature": t,
                    "truth_strained_feature": "",
                    "matched_pred": "",
                    "matched_truth": "",
                    "match_sim": "",
                    "TP": "",
                    "FP": "",
                    "FN": "x",
                    "model_precision": "",
                    "model_recall": "",
                    "model_f1": "",
                    "model_jaccard": "",
                })

            # Strained truth listing (row-by-row, no TP/FP/FN against it unless you want a second scoring pass)
            for st in strained_list:
                report_rows.append({
                    "eval_set": eval_name,
                    "modelID": mid,
                    "row_type": "STRAINED_TRUTH",
                    "pred_feature": "",
                    "truth_feature": "",
                    "truth_strained_feature": st,
                    "matched_pred": "",
                    "matched_truth": "",
                    "match_sim": "",
                    "TP": "",
                    "FP": "",
                    "FN": "",
                    "model_precision": "",
                    "model_recall": "",
                    "model_f1": "",
                    "model_jaccard": "",
                })

        overall = compute_metrics(total_tp, total_fp, total_fn)
        # Put an overall summary row at the end of the block (or you can move it to the top)
        report_rows.append({
            "eval_set": eval_name,
            "modelID": "",
            "row_type": "OVERALL_METRICS_SAMPLED",
            "pred_feature": "",
            "truth_feature": "",
            "truth_strained_feature": "",
            "matched_pred": "",
            "matched_truth": "",
            "match_sim": "",
            "TP": total_tp,
            "FP": total_fp,
            "FN": total_fn,
            "model_precision": overall["precision"],
            "model_recall": overall["recall"],
            "model_f1": overall["f1"],
            "model_jaccard": overall["jaccard"],
        })

    add_eval_block(df2s, "EVAL_2")
    add_eval_block(df5s, "EVAL_5")

    # Add one more row describing overlap forced in (for transparency)
    report_rows.append({
        "eval_set": "",
        "modelID": "",
        "row_type": f"OVERLAP_FORCED_IN ({len(overlap_used)}): " + ", ".join(overlap_used),
        "pred_feature": "",
        "truth_feature": "",
        "truth_strained_feature": "",
        "matched_pred": "",
        "matched_truth": "",
        "match_sim": "",
        "TP": "",
        "FP": "",
        "FN": "",
        "model_precision": "",
        "model_recall": "",
        "model_f1": "",
        "model_jaccard": "",
    })

    out = pd.DataFrame(report_rows)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)


# ----------------------------
# Example usage
# ----------------------------
if __name__ == "__main__":
    # Adjust pred_col to "Features" if that’s your actual column name
    pred_col_name = "Features"  # or "Features"

    # Feature-set T
    build_report_for_feature_set(
        pred_csv_2="10-EVALUATION/model_ffs_eval_T2.csv",
        pred_csv_5="10-EVALUATION/model_ffs_eval_T5.csv",
        strained_truth_csv_2="10-EVALUATION/model_ffs_eval_T2_strained.csv",
        strained_truth_csv_5="10-EVALUATION/model_ffs_eval_T5_strained.csv",
        out_csv="10-EVALUATION/reports/report_T_samples.csv",
        pred_col=pred_col_name,
        n_samples=10,
        seed=7,
        cfg=MatchConfig(sim_threshold=0.50),
    )

    # Feature-set G
    build_report_for_feature_set(
        pred_csv_2="10-EVALUATION/model_ffs_eval_G2.csv",
        pred_csv_5="10-EVALUATION/model_ffs_eval_G5.csv",
        strained_truth_csv_2="10-EVALUATION/model_ffs_eval_G2_strained.csv",
        strained_truth_csv_5="10-EVALUATION/model_ffs_eval_G5_strained.csv",
        out_csv="10-EVALUATION/reports/report_G_samples.csv",
        pred_col=pred_col_name,
        n_samples=10,
        seed=7,
        cfg=MatchConfig(sim_threshold=0.50),
    )

    # Feature-set old
    build_report_for_feature_set(
        pred_csv_2="10-EVALUATION/model_ffs_eval_old2.csv",
        pred_csv_5="10-EVALUATION/model_ffs_eval_old5.csv",
        strained_truth_csv_2="10-EVALUATION/model_ffs_eval_old2_strained.csv",
        strained_truth_csv_5="10-EVALUATION/model_ffs_eval_old5_strained.csv",
        out_csv="10-EVALUATION/reports/report_old_samples.csv",
        pred_col=pred_col_name,
        n_samples=10,
        seed=7,
        cfg=MatchConfig(sim_threshold=0.50),
    )
