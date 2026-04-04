import os
import json
import pandas as pd
from collections import Counter, defaultdict

from A_Compare_Report_2 import (
    ensure_list,
    one_to_one_align,
    _norm_np,
)

# Assumes these already exist from your code:
# - ensure_list
# - one_to_one_align
# - _norm_np


def _canon(s: str) -> str:
    """Canonical key for grouping features across minor formatting differences."""
    return _norm_np(s or "")

def _top_pairs(counter: Counter, k: int = 10):
    return counter.most_common(k)

def build_worst_feature_examples_csv(
    *,
    csv_in: str,
    csv_out: str,
    model_id_col: str = "modelID",
    pred_col: str = "Features",
    truth_col: str = "truth",
    # matching knobs
    base_sim_threshold: float = 0.90,
    allow_conservative_fuzzy: bool = True,
    loosen: bool = True,
    partial_min_len: int = 5,
    # report knobs
    top_n: int = 100,
    top_k_confusions: int = 10,
):
    """
    Creates a CSV focused on:
      - worst predicted features (FP-heavy vs TP)
      - worst truth features (FN-heavy)
    Includes examples of what they matched to (confusions).
    """

    df = pd.read_csv(csv_in)

    # --- Counters / aggregations ---
    # We store both a canonical key (normalized) and representative display strings.
    pred_display = {}   # canon -> most common raw form
    truth_display = {}  # canon -> most common raw form

    pred_occ = Counter()   # predicted occurrences (deduped per row)
    truth_occ = Counter()  # truth occurrences (deduped per row)

    pred_TP = Counter()
    pred_FP = Counter()

    truth_TP = Counter()
    truth_FN = Counter()

    # Confusion maps
    pred_to_truth = defaultdict(Counter)  # pred_canon -> Counter(truth_canon)
    truth_to_pred = defaultdict(Counter)  # truth_canon -> Counter(pred_canon)

    # Example modelIDs for quick inspection
    pred_fp_examples = defaultdict(list)   # pred_canon -> list of modelIDs where FP happened
    pred_tp_examples = defaultdict(list)   # pred_canon -> list of modelIDs where TP happened
    truth_fn_examples = defaultdict(list)  # truth_canon -> list of modelIDs where FN happened
    truth_tp_examples = defaultdict(list)  # truth_canon -> list of modelIDs where TP happened

    def _record_rep(map_rep: dict, canon: str, raw: str):
        # keep the first seen raw form unless we want “most common”; simple is fine
        if canon and canon not in map_rep and raw:
            map_rep[canon] = raw

    # --- Iterate rows and align ---
    for _, row in df.iterrows():
        mid = str(row.get(model_id_col, ""))

        preds_raw = list(dict.fromkeys(ensure_list(row.get(pred_col))))
        truth_raw = list(dict.fromkeys(ensure_list(row.get(truth_col))))

        # track occurrences (per-row deduped)
        for p in preds_raw:
            pc = _canon(p)
            if pc:
                pred_occ[pc] += 1
                _record_rep(pred_display, pc, p)

        for t in truth_raw:
            tc = _canon(t)
            if tc:
                truth_occ[tc] += 1
                _record_rep(truth_display, tc, t)

        # align (1:1)
        matches, unmatched_preds_idx, unmatched_truth_idx = one_to_one_align(
            preds_raw,
            truth_raw,
            base_sim_threshold=base_sim_threshold,
            allow_conservative_fuzzy=allow_conservative_fuzzy,
            loosen=loosen,
            partial_min_len=partial_min_len,
        )

        # matched pairs
        for m in matches:
            p = preds_raw[m.i]
            t = truth_raw[m.j]
            pc, tc = _canon(p), _canon(t)
            if not pc or not tc:
                continue

            pred_TP[pc] += 1
            truth_TP[tc] += 1

            pred_to_truth[pc][tc] += 1
            truth_to_pred[tc][pc] += 1

            if len(pred_tp_examples[pc]) < 20:
                pred_tp_examples[pc].append(mid)
            if len(truth_tp_examples[tc]) < 20:
                truth_tp_examples[tc].append(mid)

        # predicted unmatched = FP (this is what you called TN)
        for i in unmatched_preds_idx:
            p = preds_raw[i]
            pc = _canon(p)
            if not pc:
                continue
            pred_FP[pc] += 1
            if len(pred_fp_examples[pc]) < 20:
                pred_fp_examples[pc].append(mid)

        # truth unmatched = FN
        for j in unmatched_truth_idx:
            t = truth_raw[j]
            tc = _canon(t)
            if not tc:
                continue
            truth_FN[tc] += 1
            if len(truth_fn_examples[tc]) < 20:
                truth_fn_examples[tc].append(mid)

    # --- Build predicted feature “worst” table ---
    pred_rows = []
    for pc in pred_occ:
        tp = pred_TP[pc]
        fp = pred_FP[pc]
        occ = pred_occ[pc]

        # "hurts accuracy" proxies:
        # - FP/TP ratio: high means the feature is often wrong vs right
        # - FP rate: high means usually wrong when it appears
        # Add +1 smoothing to avoid divide-by-zero explosions but still rank well.
        fp_tp_ratio = fp / max(tp, 1)
        fp_rate = fp / max(occ, 1)

        # top truth matches (confusions)
        top_truth = _top_pairs(pred_to_truth[pc], k=top_k_confusions)
        top_truth_display = [
            {"truth": truth_display.get(tc, tc), "count": c}
            for tc, c in top_truth
        ]

        pred_rows.append({
            "row_type": "WORST_PREDICTED",
            "feature": pred_display.get(pc, pc),
            "feature_key": pc,
            "pred_occurrences": int(occ),
            "TP": int(tp),
            "FP": int(fp),
            "FP_over_TP": round(fp_tp_ratio, 3),
            "FP_rate": round(fp_rate, 3),
            "top_matched_truth": json.dumps(top_truth_display, ensure_ascii=False),
            "example_modelIDs_FP": json.dumps(pred_fp_examples.get(pc, []), ensure_ascii=False),
            "example_modelIDs_TP": json.dumps(pred_tp_examples.get(pc, []), ensure_ascii=False),
        })

    # sort: worst first
    # Primary: FP_over_TP, Secondary: FP count, Tertiary: FP_rate, then occurrences
    pred_rows.sort(key=lambda r: (r["FP"], r["FP_over_TP"], r["FP_rate"]), reverse=True)
    pred_rows = pred_rows[:top_n]

    # --- Build truth feature “worst” table ---
    truth_rows = []
    for tc in truth_occ:
        fn = truth_FN[tc]
        tp = truth_TP[tc]
        occ = truth_occ[tc]
        fn_rate = fn / max(occ, 1)

        top_pred = _top_pairs(truth_to_pred[tc], k=top_k_confusions)
        top_pred_display = [
            {"pred": pred_display.get(pc, pc), "count": c}
            for pc, c in top_pred
        ]

        truth_rows.append({
            "row_type": "WORST_TRUTH",
            "feature": truth_display.get(tc, tc),
            "feature_key": tc,
            "truth_occurrences": int(occ),
            "TP": int(tp),
            "FN": int(fn),
            "FN_rate": round(fn_rate, 3),
            "top_matched_predictions": json.dumps(top_pred_display, ensure_ascii=False),
            "example_modelIDs_FN": json.dumps(truth_fn_examples.get(tc, []), ensure_ascii=False),
            "example_modelIDs_TP": json.dumps(truth_tp_examples.get(tc, []), ensure_ascii=False),
        })

    # sort: worst first
    # Primary: FN count, Secondary: FN_rate, Tertiary: occurrences (to prioritize impactful misses)
    truth_rows.sort(key=lambda r: (r["FN"], r["FN_rate"], r["truth_occurrences"]), reverse=True)
    truth_rows = truth_rows[:top_n]

    # --- Add a compact parameters header row (no extra columns explosion) ---
    params = {
        "csv_in": csv_in,
        "model_id_col": model_id_col,
        "pred_col": pred_col,
        "truth_col": truth_col,
        "base_sim_threshold": base_sim_threshold,
        "allow_conservative_fuzzy": allow_conservative_fuzzy,
        "loosen": loosen,
        "partial_min_len": partial_min_len,
        "top_n": top_n,
        "top_k_confusions": top_k_confusions,
        "note": "TN is not defined for set matching; FP is used for 'unmatched predicted features'.",
    }
    header = [{
        "row_type": "PARAMETERS",
        "parameters": json.dumps(params, ensure_ascii=False)
    }]

    out_df = pd.DataFrame(header + pred_rows + truth_rows)

    out_df.to_csv(csv_out, index=False)
    return out_df


if __name__ == "__main__":
    # Example usage (adjust paths):
    build_worst_feature_examples_csv(
        csv_in="10-EVALUATION/model_ffs_eval_T5.csv",
        csv_out="10-EVALUATION/results_samples/WORST_FEATURES_T5.csv",
        model_id_col="modelID",
        pred_col="Features",
        truth_col="truth",
        base_sim_threshold=0.50,
        allow_conservative_fuzzy=True,
        loosen=True,
        partial_min_len=3,
        top_n=100,
        top_k_confusions=10,
    )