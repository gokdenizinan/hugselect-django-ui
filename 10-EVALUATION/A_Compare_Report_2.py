
import re
from dataclasses import dataclass
from collections import Counter
from difflib import SequenceMatcher
import json
from typing import Any, Iterable, List, Tuple, Dict, Optional, Sequence, Set


def _mode(values: Iterable[Any]) -> Any:
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    c = Counter(vals)
    return c.most_common(1)[0][0]


def _norm_np(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _sim(a: str, b: str) -> float:
    return SequenceMatcher(None, _norm_np(a), _norm_np(b)).ratio()


LEADING_SYMBOL_RE = re.compile(r"^[\W_]+")  # non-word chars at beginning

def strip_leading_symbols(s: str) -> str:
    return LEADING_SYMBOL_RE.sub("", s).strip()


def clean_np_from_chunk(chunk) -> str:
    """
    Expects `chunk` to be an iterable of tokens with `.pos_` and `.text` (spaCy-like).
    Drops leading determiners.
    """
    tokens = []
    seen_content = False
    for tok in chunk:
        if not seen_content and tok.pos_ == "DET":
            continue
        seen_content = True
        tokens.append(tok.text)
    return " ".join(tokens)


def normalize_np_from_chunk(chunk) -> str:
    text = clean_np_from_chunk(chunk)
    text = strip_leading_symbols(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def is_acronym(np: str) -> bool:
    if not (2 <= len(np) <= 7):
        return False
    if not np.isalpha():
        return False
    # At least 2 uppercase letters
    if sum(c.isupper() for c in np) < 2:
        return False
    # Avoid normal TitleCase words like "Dog"
    if np.istitle():
        return False
    return True


# -----------------------------
# Matching logic
# -----------------------------

def dynamic_syn_threshold(a: str, b: str, base: float = 0.90) -> float:
    """
    When synonyms are involved, you usually want to be stricter for short strings
    and can be slightly more forgiving for longer strings.

    This is a sane default curve (tweak as you like):
      - very short (<=4 chars): 0.97
      - short (5-7): 0.95
      - medium (8-12): base (0.90)
      - long (13-20): 0.89
      - very long (21+): 0.88
    """
    la = len(_norm_np(a))
    lb = len(_norm_np(b))
    L = max(la, lb)

    if L <= 4:
        return max(base, 0.97)
    if L <= 7:
        return max(base, 0.95)
    if L <= 12:
        return base
    if L <= 20:
        return min(base, 0.89)
    return min(base, 0.88)


def _apply_synonyms(norm_text: str, synonym_map: Dict[str, str]) -> str:
    """
    Simple synonym normalization: replace tokens according to synonym_map.
    Example map: {"car": "automobile", "tv": "television"}
    """
    if not synonym_map:
        return norm_text
    toks = norm_text.split()
    toks = [synonym_map.get(t, t) for t in toks]
    return " ".join(toks)


def exact_or_similar_match(
    left_raw: str,
    right_raw: str,
    *,
    sim_threshold: float = 0.90,
    synonym_map: Optional[Dict[str, str]] = None,
) -> Tuple[bool, float, str]:
    """
    Returns: (is_match, score_used, reason)

    Matching order:
      1) exact match after _norm_np
      2) similarity match >= sim_threshold
      3) optional synonym-assisted similarity match (non-acronyms only),
         using a dynamic threshold based on string length.
    """
    l_norm = _norm_np(left_raw)
    r_norm = _norm_np(right_raw)

    if not l_norm or not r_norm:
        return False, 0.0, "empty"

    # 1) exact match after normalization
    if l_norm == r_norm:
        return True, 1.0, "exact_norm"

    # 2) similarity match
    score = SequenceMatcher(None, l_norm, r_norm).ratio()
    if score >= sim_threshold:
        return True, score, "sim"

    # Acronym guard: do NOT do synonym tricks for acronyms
    if is_acronym(left_raw) or is_acronym(right_raw):
        return False, score, "acronym_no_syn"

    # 3) synonym-assisted similarity (optional)
    if synonym_map:
        l_syn = _apply_synonyms(l_norm, synonym_map)
        r_syn = _apply_synonyms(r_norm, synonym_map)

        # If synonyms changed anything, re-score and use a dynamic threshold
        if (l_syn != l_norm) or (r_syn != r_norm):
            syn_score = SequenceMatcher(None, l_syn, r_syn).ratio()
            th = dynamic_syn_threshold(left_raw, right_raw, base=sim_threshold)
            if syn_score >= th:
                return True, syn_score, f"syn_sim>=th({th:.2f})"
            return False, syn_score, f"syn_sim<th({th:.2f})"

    return False, score, "no_match"

def np_alnum_len(s: str) -> int:
    # similar to your _np_alnum_len idea
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

from difflib import SequenceMatcher

try:
    from rapidfuzz import fuzz
except Exception:
    fuzz = None  # conservative fuzzy fallback disabled if rapidfuzz not installed


def match_pair_exact_then_sim(
    left_raw: str,
    right_raw: str,
    *,
    base_sim_threshold: float = 0.90,
    allow_conservative_fuzzy: bool = True,
) -> tuple[bool, float, str]:
    """
    Returns (match?, score 0..1, reason)
    - Exact match is score=1.0
    - Similarity is SequenceMatcher ratio on normalized strings
    - Conservative fuzzy uses rapidfuzz like your merge (optional)
    """
    l_norm = _norm_np(left_raw)
    r_norm = _norm_np(right_raw)

    if not l_norm or not r_norm:
        return False, 0.0, "empty"

    # 1) exact (after normalization)
    if l_norm == r_norm:
        return True, 1.0, "exact_norm"

    # 2) required: if similarity >= 0.90 it counts as match
    sim_score = SequenceMatcher(None, l_norm, r_norm).ratio()
    if sim_score >= base_sim_threshold:
        return True, sim_score, "sim>=threshold"

    # 3) conservative fuzzy fallback (your "synonym matcher" style)
    if not allow_conservative_fuzzy or fuzz is None:
        return False, sim_score, "no_match"

    # Acronym/tech: do NOT do fuzzy-syn merges here
    if is_acronym(left_raw) or is_acronym(right_raw):
        return False, sim_score, "blocked_acronym_or_tech"

    # Choose scorer like your code: safer for short strings
    L = max(np_alnum_len(left_raw), np_alnum_len(right_raw))
    scorer = fuzz.WRatio if L <= 12 else fuzz.ratio

    thr = conservative_threshold_np(left_raw, base=base_sim_threshold)  # float 0..1
    cutoff = int(round(thr * 100))  # rapidfuzz uses 0..100

    rf_score = scorer(l_norm, r_norm)  # 0..100
    if rf_score >= cutoff:
        return True, rf_score / 100.0, f"conservative_fuzzy>=({cutoff})"

    return False, max(sim_score, rf_score / 100.0), f"conservative_fuzzy<({cutoff})"


def match_pair_with_abbrev_and_loosen(
    pred_raw: str,
    truth_raw: str,
    *,
    base_sim_threshold: float = 0.90,
    allow_conservative_fuzzy: bool = True,
    loosen: bool = False,
    partial_min_len: int = 5,
) -> tuple[bool, float, str, str]:
    """
    Returns: (matched?, best_score, reason, variant_used)

    reason can be:
      - exact_norm
      - sim>=0.90
      - conservative_fuzzy>=...
      - partial_containment
    """
    pred_variants = extract_np_variants(pred_raw)  # outside + inside parentheses if present

    best_ok = False
    best_score = 0.0
    best_reason = ""
    best_variant = ""

    for pv in pred_variants:
        # 1-3) your existing strict logic
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
                score = 0.0  # we can set a nominal score for sorting; see below
                reason = "partial_containment"

        if not ok:
            continue

        # rank reason types: exact > sim > conservative_fuzzy > partial
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

        # pick best by (rank, score)
        if (not best_ok) or (rank(reason) > rank(best_reason)) or (
            rank(reason) == rank(best_reason) and score > best_score
        ):
            best_ok, best_score, best_reason, best_variant = ok, score, reason, pv

    return best_ok, best_score, best_reason, best_variant


from dataclasses import dataclass
from typing import Sequence

@dataclass(frozen=True)
class Match:
    i: int
    j: int
    score: float
    reason: str
    variant: str  # which part matched

def one_to_one_align(
    preds,
    acts,
    *,
    base_sim_threshold: float = 0.90,
    allow_conservative_fuzzy: bool = True,
    loosen: bool = False,
    partial_min_len: int = 5,
) -> tuple[list[Match], list[int], list[int]]:
    candidates: list[Match] = []

    for i, p in enumerate(preds):
        for j, a in enumerate(acts):
            ok, score, reason, variant_used = match_pair_with_abbrev_and_loosen(
                p,
                a,
                base_sim_threshold=base_sim_threshold,
                allow_conservative_fuzzy=allow_conservative_fuzzy,
                loosen=loosen,
                partial_min_len=partial_min_len,
            )


            if ok:
                candidates.append(Match(i=i, j=j, score=score, reason=reason, variant=variant_used))

    # Prefer exact, then higher score
    def reason_rank(reason: str) -> int:
        if reason == "exact_norm":
            return 4
        if reason.startswith("sim"):
            return 3
        if reason.startswith("conservative_fuzzy"):
            return 2
        if reason == "partial_containment":
            return 1
        return 0

    # Prefer exact > sim > fuzzy > partial, then higher score
    def sort_key(m: Match):
        return (reason_rank(m.reason), m.score)

    candidates.sort(key=sort_key, reverse=True)


    candidates.sort(key=sort_key, reverse=True)

    used_i, used_j = set(), set()
    matches: list[Match] = []
    for m in candidates:
        if m.i in used_i or m.j in used_j:
            continue
        used_i.add(m.i)
        used_j.add(m.j)
        matches.append(m)

    unmatched_preds = [i for i in range(len(preds)) if i not in used_i]
    unmatched_acts  = [j for j in range(len(acts)) if j not in used_j]
    return matches, unmatched_preds, unmatched_acts


@dataclass(frozen=True)
class PairMatch:
    i: int
    j: int
    score: float
    reason: str

import ast
import pandas as pd


def ensure_list(x):
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return []
    if isinstance(x, list):
        return x
    if isinstance(x, str):
        s = x.strip()
        # stringified list like "['Stanza']" or '["Stanza"]'
        if s.startswith("[") and s.endswith("]"):
            try:
                v = ast.literal_eval(s)
                return v if isinstance(v, list) else [v]
            except Exception:
                pass
        # allow single label strings
        return [s] if s else []
    try:
        return list(x)
    except TypeError:
        return [x]


# -----------------------------
# New-format match wrappers
# -----------------------------

PAREN_RE = re.compile(r"^(.*?)\s*\(([^()]+)\)\s*$")


def extract_np_variants(s: str) -> list[str]:
    """
    Returns matchable variants for a noun phrase.
    Example:
      "General Game Playing Machine Learning (GGML)"
        -> ["General Game Playing Machine Learning", "GGML"]

    If no parentheses are present, returns [s].
    """
    if not s:
        return []

    s = s.strip()
    m = PAREN_RE.match(s)
    if not m:
        return [s]

    outside, inside = m.group(1).strip(), m.group(2).strip()
    variants = []
    if outside:
        variants.append(outside)
    if inside:
        variants.append(inside)

    return variants


def find_matches_one_to_one(actual, predicted, *, base_sim_threshold=0.90, allow_conservative_fuzzy=True):
    """
    Returns list of tuples: (pred_str, act_str, score, reason)
    Enforces one-to-one matching by calling one_to_one_align(preds, acts).
    """
    preds = list(predicted)
    acts = list(actual)

    matches, _, _ = one_to_one_align(
        preds,
        acts,
        base_sim_threshold=base_sim_threshold,
        allow_conservative_fuzzy=allow_conservative_fuzzy,
        loosen=True,          # ✅ enable partial matching
        partial_min_len=5,    # avoid tiny junk matches
    )

    out = []
    for m in matches:
        out.append((preds[m.i], acts[m.j], m.score, m.reason))
    return out


def evaluate_list_labels(
    df: pd.DataFrame,
    actual_col: str,
    predicted_col: str,
    print_matches: bool = True,
    *,
    base_sim_threshold: float = 0.90,
    allow_conservative_fuzzy: bool = True,
):
    """
    Label-level precision/recall/F1 computed with one-to-one alignment.
    Row-level accuracy = hit if at least one match exists in the row (same as your old logic).
    """
    total_rows = len(df)
    hit_rows = 0

    total_predicted = 0
    total_actual = 0
    total_correct = 0  # number of predicted labels that matched something (1:1)

    for idx, row in df.iterrows():
        actual_list = ensure_list(row[actual_col])
        predicted_list = ensure_list(row[predicted_col])

        # keep original strings, but avoid duplicates the same way you did
        actual = list(dict.fromkeys(actual_list))
        predicted = list(dict.fromkeys(predicted_list))

        total_predicted += len(predicted)
        total_actual += len(actual)

        matches = find_matches_one_to_one(
            actual,
            predicted,
            base_sim_threshold=base_sim_threshold,
            allow_conservative_fuzzy=allow_conservative_fuzzy,
        )

        # count correct predictions (each pred can appear at most once in matches)
        correct_preds = {p for p, _, _, _ in matches}
        total_correct += len(correct_preds)

        if matches:
            hit_rows += 1

            if print_matches:
                print(f"\nRow {idx}")
                print("  Matches:")
                for p, a, score, reason in matches:
                    print(f"    ✓ '{p}'  ↔  '{a}'   ({score:.3f}, {reason})")

    accuracy = hit_rows / total_rows if total_rows else 0
    precision = total_correct / total_predicted if total_predicted else 0
    recall = total_correct / total_actual if total_actual else 0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0

    return {
        "accuracy_row_level": accuracy,
        "precision_label_level": precision,
        "recall_label_level": recall,
        "f1_label_level": f1,
    }


def exact_subset_accuracy(df: pd.DataFrame, actual_col: str, predicted_col: str):
    """Row is correct if all predicted labels are contained in actual labels (exact *raw* string match)."""
    correct = 0
    for _, row in df.iterrows():
        actual = set(ensure_list(row[actual_col]))
        predicted = set(ensure_list(row[predicted_col]))
        if predicted.issubset(actual):
            correct += 1
    return correct / len(df) if len(df) else 0


def jaccard_score(df: pd.DataFrame, actual_col: str, predicted_col: str):
    scores = []
    for _, row in df.iterrows():
        actual = set(ensure_list(row[actual_col]))
        predicted = set(ensure_list(row[predicted_col]))
        union = len(actual | predicted)
        scores.append(len(actual & predicted) / union if union else 0)
    return sum(scores) / len(scores) if len(scores) else 0


def evaluate_overlap_metrics(
    df,
    actual_col,
    predicted_col,
    *,
    base_sim_threshold: float = 0.90,
    allow_conservative_fuzzy: bool = True,
    loosen: bool = True,
    partial_min_len: int = 5,
):
    """
    Global feature-level overlap metrics (TP/FP/FN) using one-to-one alignment.
    TP = number of matched predicted items (since each pred matches at most one act)
    FP = unmatched predicted
    FN = unmatched actual
    """
    TP = FP = FN = 0

    for _, row in df.iterrows():
        actual_list = ensure_list(row[actual_col])
        predicted_list = ensure_list(row[predicted_col])

        actual = list(dict.fromkeys(actual_list))
        predicted = list(dict.fromkeys(predicted_list))

        matches, unmatched_preds, unmatched_acts = one_to_one_align(
            predicted,
            actual,
            base_sim_threshold=base_sim_threshold,
            allow_conservative_fuzzy=allow_conservative_fuzzy,
            loosen=loosen,          # ✅ enable partial matching
            partial_min_len=partial_min_len,    # avoid tiny junk matches
        )

        TP += len(matches)
        FP += len(unmatched_preds)
        FN += len(unmatched_acts)

    precision = TP / (TP + FP) if (TP + FP) else 0
    recall = TP / (TP + FN) if (TP + FN) else 0
    overlap_accuracy = TP / (TP + FP + FN) if (TP + FP + FN) else 0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0

    return {
        "overlap_accuracy": overlap_accuracy,
        "overlap_precision": precision,
        "overlap_recall": recall,
        "overlap_f1": f1,
        "TP": TP,
        "FP": FP,
        "FN": FN,
    }

def partial_containment_match(a_raw: str, b_raw: str, *, min_contained_len: int = 5) -> bool:
    """
    True if normalized(a) contains normalized(b) or vice versa,
    requiring the contained string to be at least `min_contained_len`
    (to avoid junk matches like 'ai', 'ml', etc.).
    """
    a = _norm_np(a_raw)
    b = _norm_np(b_raw)
    if not a or not b:
        return False

    short, long = (a, b) if len(a) <= len(b) else (b, a)
    if len(short) < min_contained_len:
        return False

    return short in long


def add_row_accuracy_with_examples_and_save(
    df: pd.DataFrame,
    actual_col: str,
    predicted_col: str,
    output_csv: str,
    acc_col: str = "row_accuracy",
    *,
    base_sim_threshold: float = 0.90,
    allow_conservative_fuzzy: bool = True,
    loosen: bool = True,
    partial_min_len: int = 5,
) -> pd.DataFrame:
    """
    Row accuracy = Jaccard/IoU = TP / (TP + FP + FN) computed from one-to-one matches.

    Adds:
      - row_TP, row_FP, row_FN
      - tp_examples, fp_examples, fn_examples
    """

    def compute_row(actual_cell, predicted_cell):
        actual_list = ensure_list(actual_cell)
        predicted_list = ensure_list(predicted_cell)

        actual = list(dict.fromkeys(actual_list))
        predicted = list(dict.fromkeys(predicted_list))

        matches, unmatched_preds, unmatched_acts = one_to_one_align(
            predicted,
            actual,
            base_sim_threshold=base_sim_threshold,
            allow_conservative_fuzzy=allow_conservative_fuzzy,
            loosen=loosen,          # ✅ enable partial matching
            partial_min_len=partial_min_len,    # avoid tiny junk matches
        )

        TP = len(matches)
        FP = len(unmatched_preds)
        FN = len(unmatched_acts)

        denom = TP + FP + FN
        acc = TP / denom if denom else 0.0

        tp_examples = sorted([predicted[m.i] for m in matches])
        fp_examples = sorted([predicted[i] for i in unmatched_preds])
        fn_examples = sorted([actual[j] for j in unmatched_acts])

        return acc, TP, FP, FN, tp_examples, fp_examples, fn_examples

    out = df.copy()
    rows = out.apply(lambda r: compute_row(r[actual_col], r[predicted_col]), axis=1)

    out[acc_col] = [float(r[0]) for r in rows]
    out[acc_col] = out[acc_col].map(lambda x: f"{x:.3f}")

    out["row_TP"] = [r[1] for r in rows]
    out["row_FP"] = [r[2] for r in rows]
    out["row_FN"] = [r[3] for r in rows]

    out["tp_examples"] = [r[4] for r in rows]
    out["fp_examples"] = [r[5] for r in rows]
    out["fn_examples"] = [r[6] for r in rows]

    out.to_csv(output_csv, index=False)
    return out

def debug_row_detailed(
    df: pd.DataFrame,
    actual_col: str,
    predicted_col: str,
    row_idx: int = 1,
    *,
    base_sim_threshold: float = 0.90,
    allow_conservative_fuzzy: bool = True,
    loosen: bool = True,
    partial_min_len: int = 5,
):
    """
    Prints a detailed diagnostic view of a single row:
      - TP / FP / FN (counts + examples)
      - accuracy (IoU), precision, recall
      - all matches with score + reason
      - highlights matches caused by conservative fuzzy ("synonym") logic
    """
    row = df.iloc[row_idx]

    actual = list(dict.fromkeys(ensure_list(row[actual_col])))
    predicted = list(dict.fromkeys(ensure_list(row[predicted_col])))

    matches, unmatched_preds, unmatched_acts = one_to_one_align(
        predicted,
        actual,
        base_sim_threshold=base_sim_threshold,
        allow_conservative_fuzzy=allow_conservative_fuzzy,
        loosen=loosen,          # ✅ enable partial matching
        partial_min_len=partial_min_len,    # avoid tiny junk matches
    )

    TP = len(matches)
    FP = len(unmatched_preds)
    FN = len(unmatched_acts)

    precision = TP / (TP + FP) if (TP + FP) else 0.0
    recall = TP / (TP + FN) if (TP + FN) else 0.0
    accuracy = TP / (TP + FP + FN) if (TP + FP + FN) else 0.0

    print("=" * 80)
    print(f"ROW {row_idx} — DETAILED MATCH DEBUG")
    print("=" * 80)

    print("\nACTUAL (truth):")
    for a in actual:
        print(f"  - {a}")

    print("\nPREDICTED:")
    for p in predicted:
        print(f"  - {p}")

    print("\nMATCHES (one-to-one):")
    if not matches:
        print("  (none)")
    else:
        for m in matches:
            p = predicted[m.i]
            a = actual[m.j]
            tag = "🧠 synonym/fuzzy" if m.reason.startswith("conservative_fuzzy") else ""
            print(
                f"  ✓ '{p}'  ↔  '{a}'"
                f" | via='{m.variant}'"
                f" | score={m.score:.3f}"
                f" | reason={m.reason} {tag}"
            )

    print("\nTRUE POSITIVES (TP):", TP)
    for m in matches:
        print(f"  ✓ {predicted[m.i]}")

    print("\nFALSE POSITIVES (FP):", FP)
    for i in unmatched_preds:
        print(f"  ✗ {predicted[i]}")

    print("\nFALSE NEGATIVES (FN):", FN)
    for j in unmatched_acts:
        print(f"  ✗ {actual[j]}")

    print("\nROW METRICS:")
    print(f"  Accuracy (IoU): {accuracy:.3f}")
    print(f"  Precision:      {precision:.3f}")
    print(f"  Recall:         {recall:.3f}")

    print("\nSYNONYM / CONSERVATIVE FUZZY MATCHES:")
    syn_matches = [
        m for m in matches if m.reason.startswith("conservative_fuzzy")
    ]
    if not syn_matches:
        print("  (none)")
    else:
        for m in syn_matches:
            print(
                f"  🧠 '{predicted[m.i]}' ↔ '{actual[m.j]}'"
                f" | score={m.score:.3f}"
            )

    print("=" * 80)
import os
import json
import random
import pandas as pd
from typing import Dict, List, Tuple, Optional

# -----------------------------
# Helpers for display + metrics
# -----------------------------

def _to_json_list_str(x) -> str:
    """Stable display for list-like cells."""
    lst = ensure_list(x)
    return json.dumps(lst, ensure_ascii=False)

def compute_alignment_details(
    predicted_list: List[str],
    truth_list: List[str],
    *,
    base_sim_threshold: float = 0.90,
    allow_conservative_fuzzy: bool = True,
    loosen: bool = True,
    partial_min_len: int = 5,
):
    """
    Returns:
      matches: list[Match] (with i,j,score,reason,variant)
      TP, FP, FN counts
      tp_pred, fp_pred, fn_truth lists
      match_pairs: list of dict rows (pred, truth, reason, score, variant)
    """
    predicted = list(dict.fromkeys(predicted_list))
    truth = list(dict.fromkeys(truth_list))

    matches, unmatched_preds, unmatched_truth = one_to_one_align(
        predicted,
        truth,
        base_sim_threshold=base_sim_threshold,
        allow_conservative_fuzzy=allow_conservative_fuzzy,
        loosen=loosen,
        partial_min_len=partial_min_len,
    )

    TP = len(matches)
    FP = len(unmatched_preds)
    FN = len(unmatched_truth)

    tp_pred = [predicted[m.i] for m in matches]
    fp_pred = [predicted[i] for i in unmatched_preds]
    fn_truth = [truth[j] for j in unmatched_truth]

    match_pairs = []
    for m in matches:
        match_pairs.append({
            "pred_feature": predicted[m.i],
            "truth_feature": truth[m.j],
            "match_variant_used": m.variant,
            "match_reason": m.reason,
            "match_score": float(m.score),
        })

    return {
        "matches": matches,
        "TP": TP, "FP": FP, "FN": FN,
        "tp_pred": tp_pred, "fp_pred": fp_pred, "fn_truth": fn_truth,
        "match_pairs": match_pairs,
        "predicted_dedup": predicted,
        "truth_dedup": truth,
    }

def pr_iou_from_counts(TP: int, FP: int, FN: int) -> Tuple[float, float, float]:
    precision = TP / (TP + FP) if (TP + FP) else 0.0
    recall = TP / (TP + FN) if (TP + FN) else 0.0
    iou = TP / (TP + FP + FN) if (TP + FP + FN) else 0.0  # "row accuracy" / Jaccard
    return precision, recall, iou

def try_load_strained_truth(base_path: str) -> Optional[pd.DataFrame]:
    """
    If base_path is '.../model_ffs_eval_T2.csv'
    try '.../model_ffs_eval_T2_strained.csv'
    """
    root, ext = os.path.splitext(base_path)
    strained_path = f"{root}_strained{ext}"
    if os.path.exists(strained_path):
        return pd.read_csv(strained_path)
    return None

def sample_ids(
    ids: List[str],
    k: int,
    rng: random.Random
) -> List[str]:
    ids = list(dict.fromkeys(ids))
    if len(ids) <= k:
        return ids
    return rng.sample(ids, k)

# -----------------------------
# Main builder
# -----------------------------

def build_sample_review_csv_for_set(
    *,
    set_name: str,                # "T", "G", "old"
    path_eval_2: str,             # e.g. "10-EVALUATION/model_ffs_eval_T2.csv"
    path_eval_5: str,             # e.g. "10-EVALUATION/model_ffs_eval_T5.csv"
    output_csv: str,
    model_id_col: str = "modelID",
    pred_col: str = "Features",
    truth_col: str = "truth",
    seed: int = 1337,
    n_random_each: int = 10,
    n_overlap: int = 10,
    base_sim_threshold: float = 0.90,
    allow_conservative_fuzzy: bool = True,
    loosen: bool = True,
    partial_min_len: int = 5,
):
    rng = random.Random(seed)

    df2 = pd.read_csv(path_eval_2)
    df5 = pd.read_csv(path_eval_5)

    # strained truth (optional)
    df2s = try_load_strained_truth(path_eval_2)
    df5s = try_load_strained_truth(path_eval_5)

    strained_col_name = "truth_strained"
    if df2s is not None and truth_col in df2s.columns:
        df2 = df2.merge(
            df2s[[model_id_col, truth_col]].rename(columns={truth_col: strained_col_name}),
            on=model_id_col, how="left"
        )
    else:
        df2[strained_col_name] = None

    if df5s is not None and truth_col in df5s.columns:
        df5 = df5.merge(
            df5s[[model_id_col, truth_col]].rename(columns={truth_col: strained_col_name}),
            on=model_id_col, how="left"
        )
    else:
        df5[strained_col_name] = None

    ids2 = df2[model_id_col].astype(str).tolist()
    ids5 = df5[model_id_col].astype(str).tolist()

    overlap_ids = sorted(set(ids2) & set(ids5))
    sampled_overlap = sample_ids(overlap_ids, min(n_overlap, len(overlap_ids)), rng) if overlap_ids else []

    sampled2 = sample_ids(ids2, n_random_each, rng)
    sampled5 = sample_ids(ids5, n_random_each, rng)

    # Build “review rows” as a long CSV with row_type to make it readable in Sheets
    out_rows: List[Dict[str, object]] = []

    def add_model_block(df: pd.DataFrame, eval_tag: str, model_id: str):
        sub = df[df[model_id_col].astype(str) == str(model_id)]
        if sub.empty:
            return

        row = sub.iloc[0]
        pred_list = ensure_list(row.get(pred_col))
        truth_list = ensure_list(row.get(truth_col))
        strained_truth_list = ensure_list(row.get(strained_col_name))
        strained_set_norm = {_norm_np(t) for t in strained_truth_list}


        # Full-truth alignment
        full = compute_alignment_details(
            pred_list, truth_list,
            base_sim_threshold=base_sim_threshold,
            allow_conservative_fuzzy=allow_conservative_fuzzy,
            loosen=loosen,
            partial_min_len=partial_min_len,
        )
        p_full, r_full, iou_full = pr_iou_from_counts(full["TP"], full["FP"], full["FN"])

        # Strained-truth alignment (if available)
        strained = compute_alignment_details(
            pred_list, strained_truth_list,
            base_sim_threshold=base_sim_threshold,
            allow_conservative_fuzzy=allow_conservative_fuzzy,
            loosen=loosen,
            partial_min_len=partial_min_len,
        ) if strained_truth_list else {
            "TP": 0, "FP": 0, "FN": 0,
            "tp_pred": [], "fp_pred": [], "fn_truth": [],
            "match_pairs": []
        }
        p_s, r_s, iou_s = pr_iou_from_counts(strained["TP"], strained["FP"], strained["FN"])

        # 1) SUMMARY ROW
        out_rows.append({
            "set": set_name,
            "eval_tag": eval_tag,
            "row_type": "MODEL_SUMMARY",
            model_id_col: model_id,

            "predicted_features": _to_json_list_str(pred_list),
            "truth_features": _to_json_list_str(truth_list),
            "truth_strained_features": _to_json_list_str(strained_truth_list) if strained_truth_list else "",

            "TP_full": full["TP"], "FP_full": full["FP"], "FN_full": full["FN"],
            "precision_full": round(p_full, 3),
            "recall_full": round(r_full, 3),
            "iou_full": round(iou_full, 3),

            "TP_strained": strained["TP"], "FP_strained": strained["FP"], "FN_strained": strained["FN"],
            "precision_strained": round(p_s, 3),
            "recall_strained": round(r_s, 3),
            "iou_strained": round(iou_s, 3),
        })

        # 2) MATCH PAIRS (side-by-side)
        for mp in full["match_pairs"]:
            truth_feature = mp["truth_feature"]
            truth_in_strained = (
                "x" if _norm_np(truth_feature) in strained_set_norm else ""
            )

            out_rows.append({
                "set": set_name,
                "eval_tag": eval_tag,
                "row_type": "MATCH_PAIR_FULL",
                model_id_col: model_id,

                "pred_feature": mp["pred_feature"],
                "pred_TP_x": "x",
                "pred_FP_x": "",
                "truth_feature": truth_feature,
                "truth_TP_x": "x",
                "truth_FN_x": "",
                "truth_in_strained_x": truth_in_strained,

                "match_variant_used": mp["match_variant_used"],
                "match_reason": mp["match_reason"],
                "match_score": round(mp["match_score"], 3),
            })


        # 3) FP rows
        for f in full["fp_pred"]:
            out_rows.append({
                "set": set_name,
                "eval_tag": eval_tag,
                "row_type": "FP_FULL",
                model_id_col: model_id,

                "pred_feature": f,
                "pred_TP_x": "",
                "pred_FP_x": "x",
                "truth_feature": "",
                "truth_TP_x": "",
                "truth_FN_x": "",
            })

        # 4) FN rows
        for f in full["fn_truth"]:
            truth_in_strained = (
                "x" if _norm_np(f) in strained_set_norm else ""
            )

            out_rows.append({
                "set": set_name,
                "eval_tag": eval_tag,
                "row_type": "FN_FULL",
                model_id_col: model_id,

                "pred_feature": "",
                "pred_TP_x": "",
                "pred_FP_x": "",
                "truth_feature": f,
                "truth_TP_x": "",
                "truth_FN_x": "x",
                "truth_in_strained_x": truth_in_strained,
            })
    

        # # 5) OPTIONAL: strained match pairs (if you want them visible too)
        # # Uncomment if needed.
        # for mp in strained.get("match_pairs", []):
        #     out_rows.append({
        #         "set": set_name,
        #         "eval_tag": eval_tag,
        #         "row_type": "MATCH_PAIR_STRAINED",
        #         model_id_col: model_id,
        #         "pred_feature": mp["pred_feature"],
        #         "truth_feature": mp["truth_feature"],
        #         "match_reason": mp["match_reason"],
        #         "match_score": round(mp["match_score"], 3),
        #     })

    # ---------- Build blocks ----------
    # Eval “5” FIRST
    out_rows.append({
        "set": set_name,
        "eval_tag": "5",
        "row_type": "SECTION_HEADER",
        "note": f"Random sample: {n_random_each} models from eval=5"
    })
    for mid in sampled5:
        add_model_block(df5, "5", str(mid))

    # Eval “2” SECOND
    out_rows.append({
        "set": set_name,
        "eval_tag": "2",
        "row_type": "SECTION_HEADER",
        "note": f"Random sample: {n_random_each} models from eval=2"
    })
    for mid in sampled2:
        add_model_block(df2, "2", str(mid))


    # Overlap
    out_rows.append({
        "set": set_name, "eval_tag": "OVERLAP",
        "row_type": "SECTION_HEADER",
        "note": f"Overlapping modelIDs between eval=2 and eval=5 (sampled up to {n_overlap})"
    })
    for mid in sampled_overlap:
        # include both blocks (so you can compare truth=2 vs truth=5 for same modelID)
        add_model_block(df2, "2_overlap", str(mid))
        add_model_block(df5, "5_overlap", str(mid))

    out_df = pd.DataFrame(out_rows)

    # ---------- Overall metrics across sampled models (FULL truth only) ----------
    # Aggregate only summary rows for FULL truth
    # ---------- Overall metrics across sampled models ----------

    def totals_over_dataframe(df: pd.DataFrame) -> Dict[str, int]:
        """
        Micro-aggregate TP/FP/FN across ALL rows in df for:
        - full truth
        - strained truth
        """
        TP_full = FP_full = FN_full = 0
        TP_s = FP_s = FN_s = 0

        for _, row in df.iterrows():
            pred_list = ensure_list(row.get(pred_col))
            truth_list = ensure_list(row.get(truth_col))
            strained_truth_list = ensure_list(row.get("truth_strained", None))

            full = compute_alignment_details(
                pred_list, truth_list,
                base_sim_threshold=base_sim_threshold,
                allow_conservative_fuzzy=allow_conservative_fuzzy,
                loosen=loosen,
                partial_min_len=partial_min_len,
            )
            TP_full += full["TP"]; FP_full += full["FP"]; FN_full += full["FN"]

            if strained_truth_list:
                s = compute_alignment_details(
                    pred_list, strained_truth_list,
                    base_sim_threshold=base_sim_threshold,
                    allow_conservative_fuzzy=allow_conservative_fuzzy,
                    loosen=loosen,
                    partial_min_len=partial_min_len,
                )
                TP_s += s["TP"]; FP_s += s["FP"]; FN_s += s["FN"]

        return {
            "TP_full": int(TP_full), "FP_full": int(FP_full), "FN_full": int(FN_full),
            "TP_strained": int(TP_s), "FP_strained": int(FP_s), "FN_strained": int(FN_s),
        }

    summaries = out_df[out_df["row_type"] == "MODEL_SUMMARY"].copy()

    overall_blocks = []

    # ---------- Overall metrics (BOTH: ALL models and SAMPLED models) ----------
    summaries = out_df[out_df["row_type"] == "MODEL_SUMMARY"].copy()

    overall_blocks = []

    def add_overall_row(scope: str, label: str, TP_full: int, FP_full: int, FN_full: int, TP_s: int, FP_s: int, FN_s: int):
        p_full, r_full, iou_full = pr_iou_from_counts(TP_full, FP_full, FN_full)
        p_s, r_s, iou_s = pr_iou_from_counts(TP_s, FP_s, FN_s)

        overall_blocks.append({
            "set": set_name,
            "eval_tag": f"{scope}_{label}",   # 👈 combine both pieces
            "row_type": "OVERALL_METRICS",

            "TP_full": int(TP_full), "FP_full": int(FP_full), "FN_full": int(FN_full),
            "precision_full": round(p_full, 3),
            "recall_full": round(r_full, 3),
            "accuracy_full": round(iou_full, 3),   # IoU/Jaccard accuracy

            "TP_strained": int(TP_s), "FP_strained": int(FP_s), "FN_strained": int(FN_s),
            "precision_strained": round(p_s, 3),
            "recall_strained": round(r_s, 3),
            "accuracy_strained": round(iou_s, 3),  # IoU/Jaccard accuracy
        })

    # ---- (A) ALL MODELS metrics from the input CSVs ----
    tot2 = totals_over_dataframe(df2)
    tot5 = totals_over_dataframe(df5)

    add_overall_row("ALL_MODELS", "EVAL_5",  tot5["TP_full"], tot5["FP_full"], tot5["FN_full"],
                                    tot5["TP_strained"], tot5["FP_strained"], tot5["FN_strained"])
    
    # Per eval set (all models)
    add_overall_row("ALL_MODELS", "EVAL_2",  tot2["TP_full"], tot2["FP_full"], tot2["FN_full"],
                                    tot2["TP_strained"], tot2["FP_strained"], tot2["FN_strained"])


    # Combined (all models from both files; not de-duplicated across evals)
    add_overall_row("ALL_MODELS", "EVAL_2+5",
                    tot2["TP_full"] + tot5["TP_full"],
                    tot2["FP_full"] + tot5["FP_full"],
                    tot2["FN_full"] + tot5["FN_full"],
                    tot2["TP_strained"] + tot5["TP_strained"],
                    tot2["FP_strained"] + tot5["FP_strained"],
                    tot2["FN_strained"] + tot5["FN_strained"])

    # ---- (B) SAMPLED metrics (based on MODEL_SUMMARY rows) ----
    def totals_from_summaries(df_sum: pd.DataFrame) -> Dict[str, int]:
        return {
            "TP_full": int(df_sum["TP_full"].sum()),
            "FP_full": int(df_sum["FP_full"].sum()),
            "FN_full": int(df_sum["FN_full"].sum()),
            "TP_strained": int(df_sum["TP_strained"].sum()),
            "FP_strained": int(df_sum["FP_strained"].sum()),
            "FN_strained": int(df_sum["FN_strained"].sum()),
        }

    sam_all = totals_from_summaries(summaries)
    sam_2 = totals_from_summaries(summaries[summaries["eval_tag"].astype(str).str.startswith("2")])
    sam_5 = totals_from_summaries(summaries[summaries["eval_tag"].astype(str).str.startswith("5")])

    add_overall_row("SAMPLED", "EVAL_5", sam_5["TP_full"], sam_5["FP_full"], sam_5["FN_full"],
                                    sam_5["TP_strained"], sam_5["FP_strained"], sam_5["FN_strained"])
    add_overall_row("SAMPLED", "EVAL_2", sam_2["TP_full"], sam_2["FP_full"], sam_2["FN_full"],
                                    sam_2["TP_strained"], sam_2["FP_strained"], sam_2["FN_strained"])

    add_overall_row("SAMPLED", "EVAL_2+5",
                    sam_all["TP_full"], sam_all["FP_full"], sam_all["FN_full"],
                    sam_all["TP_strained"], sam_all["FP_strained"], sam_all["FN_strained"])

    overall_rows = pd.DataFrame(overall_blocks)

    # ---------- Parameter metadata row ----------
    # ---------- Parameter metadata row (compact) ----------
    param_dict = {
        "model_id_col": model_id_col,
        "pred_col": pred_col,
        "truth_col": truth_col,
        "seed": seed,
        "n_random_each": n_random_each,
        "n_overlap": n_overlap,
        "base_sim_threshold": base_sim_threshold,
        "allow_conservative_fuzzy": allow_conservative_fuzzy,
        "loosen": loosen,
        "partial_min_len": partial_min_len,
    }

    param_row = pd.DataFrame([{
        "row_type": "PARAMETERS",
        "set": set_name,
        "parameters": json.dumps(param_dict)
    }])

    # Put overall metrics at TOP
    out_df = pd.concat([param_row, overall_rows, out_df], ignore_index=True)

    out_df.to_csv(output_csv, index=False)
    return out_df
    


if __name__ == "__main__":
    # ---- Edit these paths to match your folder structure ----
    BASE = "10-EVALUATION/"

    # Feature sets:
    #   T:   model_ffs_eval_T2.csv   + model_ffs_eval_T5.csv
    #   G:   model_ffs_eval_G2.csv   + model_ffs_eval_G5.csv
    #   old: model_ffs_eval_old2.csv + model_ffs_eval_old5.csv

    configs = [
        {
            "set_name": "TT",
            "path_eval_2": os.path.join(BASE, "models_ffs/model_ffs_eval_TT2.csv"),
            "path_eval_5": os.path.join(BASE, "models_ffs/model_ffs_eval_TT5.csv"),
            "output_csv": os.path.join(BASE, "results/SAMPLES_TT_review.csv"),
        },
        #     "set_name": "T",
        #     "path_eval_2": os.path.join(BASE, "model_ffs_eval_T2.csv"),
        #     "path_eval_5": os.path.join(BASE, "model_ffs_eval_T5.csv"),
        #     "output_csv": os.path.join(BASE, "results/SAMPLES_T_review.csv"),
        # },
        # {
        #     "set_name": "G",
        #     "path_eval_2": os.path.join(BASE, "model_ffs_eval_G2.csv"),
        #     "path_eval_5": os.path.join(BASE, "model_ffs_eval_G5.csv"),
        #     "output_csv": os.path.join(BASE, "results/SAMPLES_G_review.csv"),
        # },
        # {
        #     "set_name": "old",
        #     "path_eval_2": os.path.join(BASE, "model_ffs_eval_old2.csv"),
        #     "path_eval_5": os.path.join(BASE, "model_ffs_eval_old5.csv"),
        #     "output_csv": os.path.join(BASE, "results/SAMPLES_old_review.csv"),
        # },
    ]

    for cfg in configs:
        build_sample_review_csv_for_set(
            **cfg,
            model_id_col="ModelID",
            pred_col="Features",
            truth_col="truth",
            seed=1337,
            n_random_each=10,
            n_overlap=13,
            # You can tune these:
            base_sim_threshold=0.50,           # (you used 0.50 in your debug main)
            allow_conservative_fuzzy=True,
            loosen=True,
            partial_min_len=3,
        )

    print("✅ Sample review CSVs written:")
    for cfg in configs:
        print(" -", cfg["output_csv"])
