import json
import pandas as pd
from collections import defaultdict
from tqdm import tqdm

from A_Compare_Report_2 import ensure_list, one_to_one_align, _norm_np


def _canon(s: str) -> str:
    return _norm_np(s or "")


def _dedup_keep_order(xs):
    return list(dict.fromkeys(xs))


def _apply_rules_to_preds(preds, remove_set_canon, add_set_display):
    # remove by canonical match
    kept = []
    for p in preds:
        pc = _canon(p)
        if pc and pc in remove_set_canon:
            continue
        kept.append(p)

    # add missing (display strings), avoid duplicates by canon
    existing_canon = {_canon(x) for x in kept if _canon(x)}
    for a in add_set_display:
        ac = _canon(a)
        if ac and ac not in existing_canon:
            kept.append(a)
            existing_canon.add(ac)

    return _dedup_keep_order(kept)


def _row_iou_via_alignment(preds, truths, **align_kwargs) -> float:
    matches, unmatched_preds, unmatched_truth = one_to_one_align(preds, truths, **align_kwargs)
    tp = len(matches)
    fp = len(unmatched_preds)
    fn = len(unmatched_truth)
    denom = tp + fp + fn
    return (tp / denom) if denom else 1.0


def evaluate_objective(
    df,
    truth_cols: dict,     # {"Instruct": "...", "Gemini": "...", "Chat": "...", "United": "..."}
    pred_col: str,
    remove_set_canon: set,
    add_set_display: list,
    lam_balance: float,
    align_kwargs: dict,
    pbar=None,              # NEW

):
    # mean IoU per truth set (average across rows)
    sums = {k: 0.0 for k in truth_cols}
    n = 0

    for _, row in df.iterrows():
        raw_preds = _dedup_keep_order(ensure_list(row.get(pred_col)))
        preds = _apply_rules_to_preds(raw_preds, remove_set_canon, add_set_display)

        for k, col in truth_cols.items():
            truths = _dedup_keep_order(ensure_list(row.get(col)))
            sums[k] += _row_iou_via_alignment(preds, truths, **align_kwargs)

        n += 1
    
    if pbar is not None:
        pbar.update(1) 

    avgs = {k: (sums[k] / n if n else 0.0) for k in truth_cols}
    vals = list(avgs.values())

    mean_iou = sum(vals) / len(vals) if vals else 0.0
    # population std
    var = sum((v - mean_iou) ** 2 for v in vals) / (len(vals) if vals else 1)
    std = var ** 0.5

    obj = mean_iou - lam_balance * std
    return obj, avgs


def optimize_rules_for_balanced_iou(
    *,
    csv_in: str,
    truth_cols: dict,
    pred_col: str = "Pipeline_Features",
    json_out_rules: str = "iou_rules.json",
    json_out_history: str = "iou_opt_history.json",
    # alignment settings (reuse yours)
    base_sim_threshold: float = 0.50,
    allow_conservative_fuzzy: bool = True,
    loosen: bool = True,
    partial_min_len: int = 3,
    # optimization controls
    lam_balance: float = 0.30,   # higher = stronger equalization across sets
    max_steps: int = 50,
    candidate_pool: int = 400,   # limit expensive sims to top candidates
):
    df = pd.read_csv(csv_in)

    align_kwargs = dict(
        base_sim_threshold=base_sim_threshold,
        allow_conservative_fuzzy=allow_conservative_fuzzy,
        loosen=loosen,
        partial_min_len=partial_min_len,
    )

    # --- Build cheap global candidate lists (FP-ish and FN-ish) once ---
    # We approximate candidates using unmatcheds across ALL truth sets (union of evidence).
    fp_counts = defaultdict(int)   # predicted feature canon -> count unmatched vs any truth
    occ_counts = defaultdict(int)  # predicted feature canon -> occurrence in preds
    disp_pred = {}                # canon -> display

    fn_counts = defaultdict(int)   # truth feature canon -> count unmatched vs preds
    truth_occ = defaultdict(int)   # truth feature canon -> occurrences in truths (across all sets)
    disp_truth = {}               # canon -> display

    for _, row in df.iterrows():
        preds = _dedup_keep_order(ensure_list(row.get(pred_col)))
        for p in preds:
            pc = _canon(p)
            if pc:
                occ_counts[pc] += 1
                disp_pred.setdefault(pc, p)

        for k, col in truth_cols.items():
            truths = _dedup_keep_order(ensure_list(row.get(col)))

            for t in truths:
                tc = _canon(t)
                if tc:
                    truth_occ[tc] += 1
                    disp_truth.setdefault(tc, t)

            matches, unmatched_preds, unmatched_truth = one_to_one_align(preds, truths, **align_kwargs)

            for i in unmatched_preds:
                pc = _canon(preds[i])
                if pc:
                    fp_counts[pc] += 1

            for j in unmatched_truth:
                tc = _canon(truths[j])
                if tc:
                    fn_counts[tc] += 1

    # rank candidates (simple rates)
    fp_ranked = sorted(
        fp_counts.keys(),
        key=lambda pc: (fp_counts[pc] / (occ_counts[pc] or 1), fp_counts[pc]),
        reverse=True
    )
    fn_ranked = sorted(
        fn_counts.keys(),
        key=lambda tc: (fn_counts[tc] / (truth_occ[tc] or 1), fn_counts[tc]),
        reverse=True
    )

    fp_ranked = fp_ranked[:candidate_pool]
    fn_ranked = fn_ranked[:candidate_pool]

    # --- Greedy balanced optimization over global REMOVE/ADD rules ---
    remove_set = set()   # canon strings
    add_list = []        # display strings (stored as disp_truth[canon])

    history = []

    best_obj, best_avgs = evaluate_objective(
        df, truth_cols, pred_col, remove_set, add_list, lam_balance, align_kwargs
    )
    history.append({"step": 0, "move": None, "objective": best_obj, "avg_ious": best_avgs})

    num_candidates = len(fp_ranked) + len(fn_ranked)

    # worst-case: each step tries all candidates
    total_evals = 1 + (max_steps * num_candidates)

    with tqdm(total=total_evals, desc="Global Optimization Progress") as pbar:

        best_obj, best_avgs = evaluate_objective(
            df, truth_cols, pred_col,
            remove_set, add_list,
            lam_balance, align_kwargs,
            pbar=pbar
        )

        for step in range(1, max_steps + 1):

            best_move = None
            best_move_obj = best_obj
            best_move_avgs = best_avgs

            # Try removals
            for pc in fp_ranked:
                if pc in remove_set:
                    continue

                new_remove = set(remove_set)
                new_remove.add(pc)

                obj, avgs = evaluate_objective(
                    df, truth_cols, pred_col,
                    new_remove, add_list,
                    lam_balance, align_kwargs,
                    pbar=pbar
                )

                if obj > best_move_obj + 1e-12:
                    best_move_obj, best_move_avgs = obj, avgs
                    best_move = ("remove", pc)

            # Try additions
            for tc in fn_ranked:
                if tc in {_canon(x) for x in add_list}:
                    continue

                new_add = list(add_list)
                new_add.append(disp_truth.get(tc, tc))

                obj, avgs = evaluate_objective(
                    df, truth_cols, pred_col,
                    remove_set, new_add,
                    lam_balance, align_kwargs,
                    pbar=pbar
                )

                if obj > best_move_obj + 1e-12:
                    best_move_obj, best_move_avgs = obj, avgs
                    best_move = ("add", tc)

            if best_move is None:
                break

            kind, code = best_move
            if kind == "remove":
                remove_set.add(code)
            else:
                add_list.append(disp_truth.get(code, code))

            best_obj, best_avgs = best_move_obj, best_move_avgs
        history.append({
            "step": step,
            "move": {"type": kind, "feature": (disp_pred.get(code, code) if kind == "remove" else disp_truth.get(code, code))},
            "objective": best_obj,
            "avg_ious": best_avgs
        })

    rules = {
        "remove_features": [disp_pred.get(pc, pc) for pc in sorted(remove_set)],
        "add_features": add_list,
        "final_objective": best_obj,
        "final_avg_ious": best_avgs,
        "lam_balance": lam_balance,
    }

    with open(json_out_rules, "w", encoding="utf-8") as f:
        json.dump(rules, f, indent=2, ensure_ascii=False)

    with open(json_out_history, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

    print(f"Saved rules → {json_out_rules}")
    print(f"Saved optimization history → {json_out_history}")
    print("Final avg IoUs:", best_avgs)
    print("Final objective:", best_obj)


if __name__ == "__main__":
    SAMPLE = "T5"
    optimize_rules_for_balanced_iou(
        csv_in=f"10-EVALUATION/enriched_ffs/enriched_ffs_eval_{SAMPLE}_united_50.csv",
        pred_col="Pipeline_Features",
        truth_cols={
            "Instruct": "Instruct_Features",
            "Gemini": "Gemini_Features",
            "Chat": "ChatGPT_Features",
            "United": "United_Features",
        },
        json_out_rules=f"10-EVALUATION/results_samples/iou_rules_{SAMPLE}.json",
        json_out_history=f"10-EVALUATION/results_samples/iou_history_{SAMPLE}.json",
        base_sim_threshold=0.50,
        allow_conservative_fuzzy=True,
        loosen=True,
        partial_min_len=3,
        lam_balance=0.30,
        max_steps=40,
        candidate_pool=300,
    )