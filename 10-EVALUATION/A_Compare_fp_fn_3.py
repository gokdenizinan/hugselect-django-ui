import json
import pandas as pd
from collections import defaultdict
from statistics import mean, pstdev

from A_Compare_Report_2 import ensure_list, _norm_np  # only need these for fast version


def _canon(s: str) -> str:
    return _norm_np(s or "")


def _dedup_keep_order(xs):
    return list(dict.fromkeys(xs))


def _row_to_bitset(row_list, feat2idx):
    b = 0
    for x in row_list:
        c = _canon(x)
        if not c:
            continue
        j = feat2idx.get(c)
        if j is not None:
            b |= (1 << j)
    return b


def quick_optimize_global_rules_bitset(
    *,
    csv_in: str,
    truth_cols: dict,         # {"Instruct": "...", "Gemini": "...", "Chat": "...", "United": "..."}
    pred_col: str = "Pipeline_Features",
    json_out_rules: str = "10-Evaluation/results_samples/fast_rules.json",
    json_out_history: str = "10-Evaluation/results_samples/fast_history.json",
    lam_balance: float = 0.10,
    max_steps: int = 200,
    candidate_pool: int = 3000,     # shortlist size for removes and adds
    eval_sample_rows: int = 2000,  # use sample for inner-loop speed; set None for full
    full_eval_every: int = 10,     # do full eval occasionally to avoid drift
    random_seed: int = 0,
):
    df = pd.read_csv(csv_in)
    keys = list(truth_cols.keys())

    # --- Build universe of canonical features ---
    feat2display = {}
    all_feats = set()

    def ingest_list(xs):
        for x in xs:
            c = _canon(x)
            if c:
                all_feats.add(c)
                feat2display.setdefault(c, x)

    for _, row in df.iterrows():
        ingest_list(ensure_list(row.get(pred_col)))
        for k, col in truth_cols.items():
            ingest_list(ensure_list(row.get(col)))

    all_feats = sorted(all_feats)
    feat2idx = {c: i for i, c in enumerate(all_feats)}

    # --- Build per-row bitsets ---
    P_base = []
    T = {k: [] for k in keys}

    for _, row in df.iterrows():
        preds = _dedup_keep_order(ensure_list(row.get(pred_col)))
        P_base.append(_row_to_bitset(preds, feat2idx))

        for k, col in truth_cols.items():
            truths = _dedup_keep_order(ensure_list(row.get(col)))
            T[k].append(_row_to_bitset(truths, feat2idx))

    n_rows = len(P_base)

    # choose evaluation row indices for fast inner loop
    if eval_sample_rows is not None and eval_sample_rows < n_rows:
        sample = df.sample(n=eval_sample_rows, random_state=random_seed).index.tolist()
    else:
        sample = list(range(n_rows))

    # --- candidate shortlists (cheap frequency-based) ---
    # Remove candidates: features often in preds but rarely in truths (FP-ish)
    pred_count = defaultdict(int)
    truth_count_any = defaultdict(int)

    for i in range(n_rows):
        pb = P_base[i]
        # iterate bits by shifting (fast enough for moderate bit counts); for huge, you can store sets too
        x = pb
        while x:
            lsb = x & -x
            j = (lsb.bit_length() - 1)
            pred_count[j] += 1
            x -= lsb

        for k in keys:
            tb = T[k][i]
            y = tb
            while y:
                lsb = y & -y
                j = (lsb.bit_length() - 1)
                truth_count_any[j] += 1
                y -= lsb

    # score: predicted frequently, but appears in truths infrequently → remove candidate
    remove_candidates = sorted(
        pred_count.keys(),
        key=lambda j: (pred_count[j] / (truth_count_any[j] + 1), pred_count[j]),
        reverse=True,
    )[:candidate_pool]

# -------------------------------------------
# NEW ADD CANDIDATE CONSTRUCTION (Section 3)
# -------------------------------------------

    fn_row_counts = defaultdict(int)
    truth_row_counts = defaultdict(int)

    for i in range(n_rows):
        # union truth across sets
        t_union = 0
        for k in keys:
            t_union |= T[k][i]

        # count total truth appearances
        truth_bits = t_union
        while truth_bits:
            lsb = truth_bits & -truth_bits
            j = lsb.bit_length() - 1
            truth_row_counts[j] += 1
            truth_bits -= lsb

        # count missing from pipeline (FN)
        missing = t_union & ~P_base[i]
        miss_bits = missing
        while miss_bits:
            lsb = miss_bits & -miss_bits
            j = lsb.bit_length() - 1
            fn_row_counts[j] += 1
            miss_bits -= lsb

    add_candidates = sorted(
        fn_row_counts.keys(),
        key=lambda j: (fn_row_counts[j] / (truth_row_counts[j] or 1), fn_row_counts[j]),
        reverse=True,
    )[:candidate_pool]

    # --- evaluation ---
    def eval_obj(remove_mask, add_mask, rows):
        avg = {}
        for k in keys:
            ious = []
            Tk = T[k]
            for i in rows:
                P = (P_base[i] & ~remove_mask) | add_mask
                inter = (P & Tk[i]).bit_count()
                uni = (P | Tk[i]).bit_count()
                ious.append(inter / uni if uni else 1.0)
            avg[k] = sum(ious) / len(ious) if ious else 0.0

        vals = list(avg.values())
        m = mean(vals) if vals else 0.0
        s = pstdev(vals) if len(vals) > 1 else 0.0
        return (m - lam_balance * s), avg

    remove_mask = 0
    add_mask = 0

    best_obj, best_avg = eval_obj(remove_mask, add_mask, sample)
    history = [{"step": 0, "move": None, "objective": best_obj, "avg_ious": best_avg, "mode": "sample"}]

    for step in range(1, max_steps + 1):
        best_move = None
        best_move_obj = best_obj
        best_move_avg = best_avg

        # try removals
        for j in remove_candidates:
            bit = 1 << j
            if remove_mask & bit:
                continue
            obj, avg = eval_obj(remove_mask | bit, add_mask, sample)
            if obj > best_move_obj + 1e-15:
                best_move_obj, best_move_avg = obj, avg
                best_move = ("remove", j)

        # try additions
        for j in add_candidates:
            bit = 1 << j
            if add_mask & bit:
                continue
            obj, avg = eval_obj(remove_mask, add_mask | bit, sample)
            if obj > best_move_obj + 1e-15:
                best_move_obj, best_move_avg = obj, avg
                best_move = ("add", j)

        if best_move is None:
            break

        kind, j = best_move
        bit = 1 << j
        if kind == "remove":
            remove_mask |= bit
        else:
            add_mask |= bit

        best_obj, best_avg = best_move_obj, best_move_avg
        history.append({
            "step": step,
            "move": {"type": kind, "feature": feat2display.get(all_feats[j], all_feats[j])},
            "objective": best_obj,
            "avg_ious": best_avg,
            "mode": "sample",
        })

        # occasional full evaluation to keep it honest
        if full_eval_every and (step % full_eval_every == 0):
            full_obj, full_avg = eval_obj(remove_mask, add_mask, list(range(n_rows)))
            history.append({
                "step": step,
                "move": {"type": "eval_full", "feature": None},
                "objective": full_obj,
                "avg_ious": full_avg,
                "mode": "full",
            })

    # unpack rules
    remove_features = []
    add_features = []

    # extract bits
    rm = remove_mask
    while rm:
        lsb = rm & -rm
        j = (lsb.bit_length() - 1)
        remove_features.append(feat2display.get(all_feats[j], all_feats[j]))
        rm -= lsb

    am = add_mask
    while am:
        lsb = am & -am
        j = (lsb.bit_length() - 1)
        add_features.append(feat2display.get(all_feats[j], all_feats[j]))
        am -= lsb

    rules = {
        "remove_features": sorted(remove_features),
        "add_features": sorted(add_features),
        "lam_balance": lam_balance,
        "max_steps": max_steps,
        "candidate_pool": candidate_pool,
        "eval_sample_rows": eval_sample_rows,
        "full_eval_every": full_eval_every,
        "note": "Fast optimizer uses exact canonical matching (no fuzzy alignment). Validate/refine with fuzzy aligner afterward.",
    }

    with open(json_out_rules, "w", encoding="utf-8") as f:
        json.dump(rules, f, indent=2, ensure_ascii=False)

    with open(json_out_history, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

    print(f"Saved fast rules → {json_out_rules}")
    print(f"Saved history → {json_out_history}")


if __name__ == "__main__":
    SAMPLE = "T5"
    quick_optimize_global_rules_bitset(
        csv_in=f"10-EVALUATION/enriched_ffs/enriched_ffs_eval_{SAMPLE}_united_50.csv",
        pred_col="Pipeline_Features",
        truth_cols={
            "Instruct": "Instruct_Features",
            "Gemini": "Gemini_Features",
            "Chat": "ChatGPT_Features",
            "United": "United_Features",
        },
        lam_balance=0.30,
        max_steps=60,
        candidate_pool=800,
        eval_sample_rows=2000,  # increase if you want more accuracy
        full_eval_every=10,
    )