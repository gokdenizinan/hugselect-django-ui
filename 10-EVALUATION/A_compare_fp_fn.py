import json
import pandas as pd
from collections import Counter

from A_Compare_Report_2 import (
    ensure_list,
    one_to_one_align,
    _norm_np,
)
# Requires:
# - ensure_list
# - one_to_one_align
# - _norm_np


def _canon(s: str) -> str:
    return _norm_np(s or "")


def build_worst_feature_rate_lists(
    *,
    csv_in: str,
    json_out_pred: str,
    json_out_truth: str,
    json_out_best_pred: str = None,
    json_out_best_truth: str = None,
    model_id_col: str = "ModelID",
    pred_col: str = "Pipeline_Features",
    truth_col: str = "United_Features",
    base_sim_threshold: float = 0.90,
    allow_conservative_fuzzy: bool = True,
    loosen: bool = True,
    partial_min_len: int = 5,
    top_n: int = 100,
    min_occurrences: int = 2,
    min_rate: float = 0.5,
):
    df = pd.read_csv(csv_in)

    pred_occ = Counter()
    pred_TP = Counter()
    pred_FP = Counter()

    truth_occ = Counter()
    truth_TP = Counter()
    truth_FN = Counter()

    pred_display = {}
    truth_display = {}

    for _, row in df.iterrows():
        preds = list(dict.fromkeys(ensure_list(row.get(pred_col))))
        truths = list(dict.fromkeys(ensure_list(row.get(truth_col))))

        # count occurrences
        for p in preds:
            pc = _canon(p)
            if pc:
                pred_occ[pc] += 1
                pred_display.setdefault(pc, p)

        for t in truths:
            tc = _canon(t)
            if tc:
                truth_occ[tc] += 1
                truth_display.setdefault(tc, t)

        # align
        matches, unmatched_preds, unmatched_truth = one_to_one_align(
            preds,
            truths,
            base_sim_threshold=base_sim_threshold,
            allow_conservative_fuzzy=allow_conservative_fuzzy,
            loosen=loosen,
            partial_min_len=partial_min_len,
        )

        # matched
        for m in matches:
            p = preds[m.i]
            t = truths[m.j]
            pc = _canon(p)
            tc = _canon(t)

            if pc:
                pred_TP[pc] += 1
            if tc:
                truth_TP[tc] += 1

        # unmatched predicted → FP
        for i in unmatched_preds:
            pc = _canon(preds[i])
            if pc:
                pred_FP[pc] += 1

        # unmatched truth → FN
        for j in unmatched_truth:
            tc = _canon(truths[j])
            if tc:
                truth_FN[tc] += 1

    # -------------------------
    # Build worst predicted list
    # -------------------------
    worst_pred = []
    for pc in pred_occ:
        tp = pred_TP[pc]
        fp = pred_FP[pc]
        occ = pred_occ[pc]

        if occ == 0:
            continue

        fp_rate = fp / occ

        worst_pred.append({
            "feature": pred_display.get(pc, pc),
            "occurrences": int(occ),
            "TP": int(tp),
            "FP": int(fp),
            "FP_rate": round(fp_rate, 4),
        })

    # sort by worst FP rate first, then volume
    best_pred = [
        x for x in worst_pred
        if x["occurrences"] > min_occurrences and x["FP_rate"] <= min_rate
    ]
        
    worst_pred = [
    x for x in worst_pred
    if x["occurrences"] > min_occurrences and x["FP_rate"] > min_rate
    ]

    best_pred.sort(
        key=lambda x: (x["FP_rate"], x["TP"]),
        reverse=False
    )

    worst_pred.sort(
        key=lambda x: (x["FP_rate"], x["FP"]),
        reverse=True
    )


    # -------------------------
    # Build worst truth list
    # -------------------------
    worst_truth = []
    for tc in truth_occ:
        tp = truth_TP[tc]
        fn = truth_FN[tc]
        occ = truth_occ[tc]

        if occ == 0:
            continue

        fn_rate = fn / occ

        worst_truth.append({
            "feature": truth_display.get(tc, tc),
            "occurrences": int(occ),
            "TP": int(tp),
            "FN": int(fn),
            "FN_rate": round(fn_rate, 4),
        })

    # sort by worst FN rate first, then volume
    best_truth = [
        x for x in worst_truth
        if x["occurrences"] > min_occurrences and x["FN_rate"] <= min_rate
    ]

    worst_truth = [
        x for x in worst_truth
        if x["occurrences"] > min_occurrences and x["FN_rate"] > min_rate
    ]


    best_truth.sort(
        key=lambda x: (x["FN_rate"], x["TP"]),
        reverse=False
    )

    worst_truth.sort(
        key=lambda x: (x["FN_rate"], x["FN"]),
        reverse=True
    )

    if top_n:
        worst_truth = worst_truth[:top_n]
        worst_pred = worst_pred[:top_n]

    # -------------------------
    # Save JSONs
    # -------------------------
    with open(json_out_pred, "w", encoding="utf-8") as f:
        json.dump(worst_pred, f, indent=2, ensure_ascii=False)

    with open(json_out_truth, "w", encoding="utf-8") as f:
        json.dump(worst_truth, f, indent=2, ensure_ascii=False)
    
    with open(json_out_best_pred, "w", encoding="utf-8") as f:
        json.dump(best_pred, f, indent=2, ensure_ascii=False)
    
    with open(json_out_best_truth, "w", encoding="utf-8") as f:
        json.dump(best_truth, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(worst_pred)} worst predicted features → {json_out_pred}")
    print(f"Saved {len(worst_truth)} worst truth features → {json_out_truth}")
    
    print(f"Saved {len(best_pred)} best predicted features → {json_out_best_pred}")
    print(f"Saved {len(best_truth)} best truth features → {json_out_best_truth}")

if __name__ == "__main__":
    SAMPLE = "T5"
    build_worst_feature_rate_lists(
        truth_col= "Instruct_Features", # bunu united yap
        csv_in=f"10-EVALUATION/enriched_ffs/enriched_ffs_eval_{SAMPLE}_united_50.csv",
        json_out_pred=f"10-EVALUATION/results_samples/worst_predicted_fp_{SAMPLE}.json",
        json_out_truth=f"10-EVALUATION/results_samples/worst_truth_fn_{SAMPLE}.json",
        json_out_best_pred=f"10-EVALUATION/results_samples/best_predicted_tp_{SAMPLE}.json",
        json_out_best_truth=f"10-EVALUATION/results_samples/best_truth_tp_{SAMPLE}.json",
        base_sim_threshold=0.50,
        allow_conservative_fuzzy=True,
        loosen=True,
        partial_min_len=3,
        top_n=None,
        min_occurrences=0,
        min_rate=0.7
    )