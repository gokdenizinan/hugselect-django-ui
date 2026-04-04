import json
import pandas as pd
from collections import Counter, defaultdict
from statistics import mean, pstdev

from A_Compare_Report_2 import ensure_list, one_to_one_align, _norm_np


def _canon(s: str) -> str:
    return _norm_np(s or "")


def _safe_div(a: int, b: int) -> float:
    return (a / b) if b else 0.0


def build_balanced_feature_lists(
    *,
    csv_in: str,
    truth_cols: dict,  # {"United":"United_Features", "Chat":"Chat_Features", ...}
    pred_col: str = "Pipeline_Features",
    json_out_fp_remove: str = "balanced_fp_remove.json",
    json_out_fn_add: str = "balanced_fn_add.json",
    json_out_debug: str = None,
    base_sim_threshold: float = 0.90,
    allow_conservative_fuzzy: bool = True,
    loosen: bool = True,
    partial_min_len: int = 5,
    min_occurrences: int = 2,
    top_fn: int = None,
    top_fp: int = None,
    # balance knobs:
    alpha_balance: float = 0.35,   # penalize variance across truth sets
    beta_weakset: float = 0.35,    # prioritize improving weakest truth set
):
    """
    Outputs:
      - balanced_fp_remove: ranked predicted features to REMOVE from pipeline_f (FP-heavy across sets, balanced)
      - balanced_fn_add: ranked truth features to ADD to pipeline_f (FN-heavy across sets, balanced)
    """

    df = pd.read_csv(csv_in)
    truth_keys = list(truth_cols.keys())

    # per truth-set counters
    pred_occ = Counter()
    pred_display = {}

    truth_occ = {k: Counter() for k in truth_keys}
    truth_display = {k: {} for k in truth_keys}

    pred_TP = {k: Counter() for k in truth_keys}
    pred_FP = {k: Counter() for k in truth_keys}

    truth_TP = {k: Counter() for k in truth_keys}
    truth_FN = {k: Counter() for k in truth_keys}

    # Track per-row IoU so we can identify weakest set (optional but useful)
    # We approximate IoU at row level using exact string match after alignment decisions.
    # (Not perfect, but directional and consistent with your alignment.)
    iou_sums = {k: 0.0 for k in truth_keys}
    iou_counts = {k: 0 for k in truth_keys}

    for _, row in df.iterrows():
        preds = list(dict.fromkeys(ensure_list(row.get(pred_col))))
        # count predicted occurrences (shared for all truth sets)
        for p in preds:
            pc = _canon(p)
            if pc:
                pred_occ[pc] += 1
                pred_display.setdefault(pc, p)

        # do alignment separately for each truth set
        for k in truth_keys:
            truths = list(dict.fromkeys(ensure_list(row.get(truth_cols[k]))))

            for t in truths:
                tc = _canon(t)
                if tc:
                    truth_occ[k][tc] += 1
                    truth_display[k].setdefault(tc, t)

            matches, unmatched_preds, unmatched_truth = one_to_one_align(
                preds,
                truths,
                base_sim_threshold=base_sim_threshold,
                allow_conservative_fuzzy=allow_conservative_fuzzy,
                loosen=loosen,
                partial_min_len=partial_min_len,
            )

            # matched => TP for both sides
            matched_pred_idx = set()
            matched_truth_idx = set()

            for m in matches:
                p = preds[m.i]
                t = truths[m.j]
                pc = _canon(p)
                tc = _canon(t)
                if pc:
                    pred_TP[k][pc] += 1
                if tc:
                    truth_TP[k][tc] += 1
                matched_pred_idx.add(m.i)
                matched_truth_idx.add(m.j)

            # unmatched predicted => FP vs this truth set
            for i in unmatched_preds:
                pc = _canon(preds[i])
                if pc:
                    pred_FP[k][pc] += 1

            # unmatched truth => FN vs this truth set
            for j in unmatched_truth:
                tc = _canon(truths[j])
                if tc:
                    truth_FN[k][tc] += 1

            # row-level IoU estimate from alignment result
            tp = len(matches)
            fp = len(unmatched_preds)
            fn = len(unmatched_truth)
            denom = tp + fp + fn
            iou_val = (tp / denom) if denom else 1.0
            iou_sums[k] += iou_val
            iou_counts[k] += 1

    # Identify weakest truth set by avg IoU (to target balancing)
    avg_ious = {k: (iou_sums[k] / iou_counts[k]) if iou_counts[k] else 0.0 for k in truth_keys}
    weakest_k = min(avg_ious, key=avg_ious.get) if truth_keys else None

    # -------------------------
    # Build balanced FP-removal list
    # -------------------------
    fp_remove = []
    for pc, occ in pred_occ.items():
        if occ <= min_occurrences:
            continue

        fp_rates = {}
        for k in truth_keys:
            fp = pred_FP[k][pc]
            fp_rates[k] = _safe_div(fp, occ)

        rates = list(fp_rates.values())
        m = mean(rates) if rates else 0.0
        s = pstdev(rates) if len(rates) > 1 else 0.0
        weak = fp_rates.get(weakest_k, m) if weakest_k else m

        # score: high mean FP bad + high variance bad + especially bad in weakest set
        score = m + alpha_balance * s + beta_weakset * weak

        fp_remove.append({
            "feature": pred_display.get(pc, pc),
            "occurrences": int(occ),
            "fp_rate_mean": round(m, 6),
            "fp_rate_std": round(s, 6),
            "fp_rate_weakset": round(weak, 6),
            "fp_rates": {k: round(v, 6) for k, v in fp_rates.items()},
            "score": round(score, 6),
        })

    fp_remove.sort(key=lambda x: (x["score"], x["occurrences"]), reverse=True)
    if top_fp:
        fp_remove = fp_remove[:top_fp]

    # -------------------------
    # Build balanced FN-add list
    # -------------------------
    # We union truth features across all sets; use per-set occurrences for FN rate denominators
    all_truth_features = set()
    for k in truth_keys:
        all_truth_features |= set(truth_occ[k].keys())

    fn_add = []
    for tc in all_truth_features:
        fn_rates = {}
        occs = {}

        for k in truth_keys:
            occ = truth_occ[k][tc]
            if occ <= 0:
                continue
            occs[k] = occ
            fn = truth_FN[k][tc]
            fn_rates[k] = _safe_div(fn, occ)

        if not fn_rates:
            continue

        rates = list(fn_rates.values())
        m = mean(rates)
        s = pstdev(rates) if len(rates) > 1 else 0.0
        weak = fn_rates.get(weakest_k, m) if weakest_k else m

        score = m + alpha_balance * s + beta_weakset * weak

        # pick a nice display name from any set that has it
        disp = None
        for k in truth_keys:
            if tc in truth_display[k]:
                disp = truth_display[k][tc]
                break
        disp = disp or tc

        fn_add.append({
            "feature": disp,
            "fn_rate_mean": round(m, 6),
            "fn_rate_std": round(s, 6),
            "fn_rate_weakset": round(weak, 6),
            "fn_rates": {k: round(v, 6) for k, v in fn_rates.items()},
            "occurrences": {k: int(occs[k]) for k in occs},
            "score": round(score, 6),
        })

    fn_add.sort(key=lambda x: x["score"], reverse=True)
    if top_fn:
        fn_add = fn_add[:top_fn]

    # -------------------------
    # Save JSONs
    # -------------------------
    with open(json_out_fp_remove, "w", encoding="utf-8") as f:
        json.dump(fp_remove, f, indent=2, ensure_ascii=False)

    with open(json_out_fn_add, "w", encoding="utf-8") as f:
        json.dump(fn_add, f, indent=2, ensure_ascii=False)

    if json_out_debug:
        dbg = {
            "avg_ious": {k: round(v, 6) for k, v in avg_ious.items()},
            "weakest_set": weakest_k,
            "counts": {
                "fp_remove": len(fp_remove),
                "fn_add": len(fn_add),
            },
        }
        with open(json_out_debug, "w", encoding="utf-8") as f:
            json.dump(dbg, f, indent=2, ensure_ascii=False)

        # Extract feature values
    annan = [d["feature"] for d in fp_remove if "feature" in d]

    # Save to JSON
    with open("10-EVALUATION/results_samples/balanced_fp_remove_list.json", "w", encoding="utf-8") as f:
        json.dump(annan, f, indent=2, ensure_ascii=False)

        # Extract feature values
    annen = [d["feature"] for d in fn_add if "feature" in d]

    # Save to JSON
    with open("10-EVALUATION/results_samples/balanced_fn_add_list.json", "w", encoding="utf-8") as f:
        json.dump(annen, f, indent=2, ensure_ascii=False)

    print(f"Weakest set by avg IoU: {weakest_k} (avg IoU={avg_ious.get(weakest_k):.4f})")
    print(f"Saved {len(fp_remove)} balanced FP removals → {json_out_fp_remove}")
    print(f"Saved {len(fn_add)} balanced FN additions → {json_out_fn_add}")
    if json_out_debug:
        print(f"Saved debug → {json_out_debug}")

if __name__ == "__main__":
    SAMPLE = "T5"
    build_balanced_feature_lists(
        
        # csv_in=f"10-EVALUATION/enriched_ffs/enriched_ffs_eval_{SAMPLE}_united_50.csv",
        csv_in=f"10-EVALUATION/enriched_ffs/enriched_ffs_eval_{SAMPLE}_filtered_50.csv",
        pred_col="Pipeline_Features",
        truth_cols={
            "United": "United_Features",
        },
        json_out_fp_remove=f"10-EVALUATION/results_samples/balanced_fp_remove_{SAMPLE}.json",
        json_out_fn_add=f"10-EVALUATION/results_samples/balanced_fn_add_{SAMPLE}.json",
        json_out_debug=f"10-EVALUATION/results_samples/balanced_debug_{SAMPLE}.json",
        base_sim_threshold=0.50,
        allow_conservative_fuzzy=True,
        loosen=True,
        partial_min_len=3,
        min_occurrences=0,
        top_fn=25,
        top_fp=30,
        alpha_balance=0.35,
        beta_weakset=0.35,
    )