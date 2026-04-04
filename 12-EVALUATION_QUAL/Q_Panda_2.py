import pandas as pd
import json
import numpy as np


with open("6-REVIEW_SENTIMENT_ANALYSIS/llm_quality_mapping/quality_mapping_output_AB50_all_expanded.json", "r") as f:
    data = json.load(f)


def proportional_reduce(series, amount, floor):
    """
    Reduce `amount` from a Series proportionally to current values,
    but never let any entry go below `floor`.
    Returns a float Series.
    """
    s = series.astype(float).copy()
    remaining = float(amount)

    while remaining > 1e-9:
        donors = s[s > floor]
        if donors.empty:
            break

        reducible = (donors - floor).sum()
        if reducible <= 1e-9:
            break

        take = min(remaining, reducible)
        weights = donors / donors.sum()
        reduction = weights * take

        # cap reduction so we don't go below floor
        max_reduction = donors - floor
        reduction = np.minimum(reduction, max_reduction)

        s.loc[donors.index] -= reduction
        remaining -= reduction.sum()

    return s


def proportional_increase_with_cap(series, amount, cap, weights):
    """
    Increase entries in `series` proportionally to `weights`,
    but never let any entry exceed `cap`.
    Returns a float Series.
    """
    s = series.astype(float).copy()
    remaining = float(amount)

    while remaining > 1e-9:
        receivers = s[s < cap]
        if receivers.empty:
            break

        room = (cap - receivers).sum()
        if room <= 1e-9:
            break

        give = min(remaining, room)

        receiver_weights = weights.loc[receivers.index].astype(float)
        if receiver_weights.sum() == 0:
            receiver_weights = pd.Series(1.0, index=receivers.index)

        receiver_weights = receiver_weights / receiver_weights.sum()
        increase = receiver_weights * give

        max_increase = cap - receivers
        increase = np.minimum(increase, max_increase)

        s.loc[receivers.index] += increase
        remaining -= increase.sum()

    return s


def round_to_total(float_series, total):
    """
    Round a float Series to integers while preserving exact total.
    Uses largest remainder method.
    """
    floored = np.floor(float_series).astype(int)
    diff = total - floored.sum()

    remainders = float_series - floored
    if diff > 0:
        add_idx = remainders.sort_values(ascending=False).index[:diff]
        floored.loc[add_idx] += 1
    elif diff < 0:
        sub_idx = remainders.sort_values(ascending=True).index[:abs(diff)]
        floored.loc[sub_idx] -= 1

    return floored.astype(int)


def rebalance_category_targets(base_counts, total_size=500, min_count=30, ideal_count=50):
    """
    Rebalance proportional targets:
    - hard floor at min_count
    - soft boost toward ideal_count
    - take rows away from larger categories proportionally
    """
    counts = base_counts.astype(float).copy()

    # --- Step 1: hard floor to minimum ---
    below_min = counts[counts < min_count]
    if not below_min.empty:
        counts.loc[below_min.index] = min_count

        # keep total fixed by reducing bigger categories proportionally
        extra_needed = counts.sum() - total_size
        if extra_needed > 0:
            counts = proportional_reduce(counts, amount=extra_needed, floor=min_count)

    # --- Step 2: soft boost toward ideal (only if there are donors above ideal) ---
    donors = counts[counts > ideal_count]
    receivers = counts[counts < ideal_count]

    if not donors.empty and not receivers.empty:
        available_to_redistribute = (donors - ideal_count).sum()
        needed_for_ideal = (ideal_count - receivers).sum()
        move_amount = min(available_to_redistribute, needed_for_ideal)

        if move_amount > 0:
            # reduce donors down toward ideal
            donor_part = counts.loc[donors.index]
            counts.loc[donors.index] = proportional_reduce(
                donor_part,
                amount=move_amount,
                floor=ideal_count
            )

            # give that amount to receivers, weighted by original base counts
            receiver_part = counts.loc[receivers.index]
            receiver_weights = base_counts.loc[receivers.index].astype(float)
            counts.loc[receivers.index] = proportional_increase_with_cap(
                receiver_part,
                amount=move_amount,
                cap=ideal_count,
                weights=receiver_weights
            )

    # final integer targets summing exactly to total_size
    counts = round_to_total(counts, total_size)

    return counts.sort_values(ascending=False)


def create_attribute_sample_with_boosted_categories(
    data_dict,
    existing_csv_path,
    fixed_n=60,
    total_sample_size=500,
    min_count=30,
    ideal_count=50,
    output_path="12-EVALUATION_QUAL/attribute_sample_3.csv",
    random_state=42,
):
    """
    Keep the first `fixed_n` rows from an existing CSV.
    Fill the rest so final category distribution:
    - starts from original proportions
    - boosts low categories to min/ideal where possible
    - reduces larger categories proportionally
    - keeps total = total_sample_size
    """

    rng = np.random.RandomState(random_state)

    df = pd.DataFrame.from_dict(data_dict, orient="index").reset_index(drop=True)

    df = df[["model_id", "reviews", "Primary_Category", "Rationale"]].copy()

    df = df.dropna(subset=["Primary_Category"])

    # REMOVE categories with only 1 example
    category_counts = df["Primary_Category"].value_counts()
    valid_categories = category_counts[category_counts > 10].index
    df = df[df["Primary_Category"].isin(valid_categories)].copy()

    # existing sampled csv
    existing_df = pd.read_csv(existing_csv_path)
    existing_df = existing_df[["model_id", "reviews", "Primary_Category", "Rationale"]].copy()
    existing_df = existing_df.dropna(subset=["Primary_Category"])

    fixed_df = existing_df.head(fixed_n).copy()
    if len(fixed_df) < fixed_n:
        raise ValueError(f"Existing CSV has only {len(existing_df)} rows, cannot keep first {fixed_n} rows.")

    if fixed_n > total_sample_size:
        raise ValueError("fixed_n cannot be larger than total_sample_size.")

    # remove fixed rows from candidate pool
    remaining_pool = df[~df["model_id"].isin(fixed_df["model_id"])].copy()

    # original proportional counts over total sample size
    proportions = df["Primary_Category"].value_counts(normalize=True).sort_index()
    base_targets = (proportions * total_sample_size)
    base_targets = round_to_total(base_targets, total_sample_size)

    # boosted/rebalanced final targets
    final_targets = rebalance_category_targets(
        base_targets,
        total_size=total_sample_size,
        min_count=min_count,
        ideal_count=ideal_count
    ).sort_index()

    # how many already fixed
    fixed_counts = fixed_df["Primary_Category"].value_counts().reindex(final_targets.index, fill_value=0)

    # initial required additional rows per category
    needed_counts = (final_targets - fixed_counts).clip(lower=0)

    sampled_parts = []
    used_ids = set(fixed_df["model_id"])

    # first pass: sample what each category ideally needs
    for cat in final_targets.index:
        group = remaining_pool[
            (remaining_pool["Primary_Category"] == cat) &
            (~remaining_pool["model_id"].isin(used_ids))
        ]
        n_needed = int(needed_counts.get(cat, 0))
        if n_needed > 0 and len(group) > 0:
            n_take = min(len(group), n_needed)
            part = group.sample(n=n_take, random_state=random_state)
            sampled_parts.append(part)
            used_ids.update(part["model_id"].tolist())

    sampled_rest = pd.concat(sampled_parts, ignore_index=True) if sampled_parts else pd.DataFrame(columns=df.columns)

    # second pass: top up until total_sample_size is reached
    # priority = categories still furthest below target
    while len(fixed_df) + len(sampled_rest) < total_sample_size:
        current_df = pd.concat([fixed_df, sampled_rest], ignore_index=True)
        current_counts = current_df["Primary_Category"].value_counts().reindex(final_targets.index, fill_value=0)
        deficits = (final_targets - current_counts).sort_values(ascending=False)

        took_any = False

        for cat, deficit in deficits.items():
            if deficit <= 0:
                continue

            available = remaining_pool[
                (remaining_pool["Primary_Category"] == cat) &
                (~remaining_pool["model_id"].isin(used_ids))
            ]

            if len(available) == 0:
                continue

            take_one = available.sample(n=1, random_state=rng.randint(0, 1_000_000))
            sampled_rest = pd.concat([sampled_rest, take_one], ignore_index=True)
            used_ids.update(take_one["model_id"].tolist())
            took_any = True

            if len(fixed_df) + len(sampled_rest) >= total_sample_size:
                break

        # if no category with deficit has remaining rows, fill from any leftover rows
        if not took_any:
            leftover = remaining_pool[~remaining_pool["model_id"].isin(used_ids)]
            if leftover.empty:
                raise ValueError("Not enough remaining rows to reach the requested total sample size.")
            take_one = leftover.sample(n=1, random_state=rng.randint(0, 1_000_000))
            sampled_rest = pd.concat([sampled_rest, take_one], ignore_index=True)
            used_ids.update(take_one["model_id"].tolist())

    # keep first 60 at top, shuffle only the newly sampled part
    sampled_rest = sampled_rest.sample(frac=1, random_state=random_state).reset_index(drop=True)
    final_df = pd.concat([fixed_df, sampled_rest], ignore_index=True)

    final_df.to_csv(output_path, index=False)

    print(f"Saved {len(final_df)} rows to {output_path}")

    print("\nOriginal proportional targets:")
    print(base_targets.sort_values(ascending=False))

    print("\nRebalanced final targets:")
    print(final_targets.sort_values(ascending=False))

    print("\nFixed first rows distribution:")
    print(fixed_df["Primary_Category"].value_counts().sort_values(ascending=False))

    print("\nFinal distribution in saved CSV:")
    print(final_df["Primary_Category"].value_counts().sort_values(ascending=False))

    return final_df, final_targets


sample_df, final_targets = create_attribute_sample_with_boosted_categories(
    data_dict=data,
    existing_csv_path="12-EVALUATION_QUAL/attribute_sample_2.csv",
    fixed_n=60,
    total_sample_size=500,
    min_count=30,
    ideal_count=30,
    output_path="12-EVALUATION_QUAL/quality_sample_A0.csv",
    random_state=42,
)

import json
import pandas as pd

IN_CSV = "12-EVALUATION_QUAL/quality_sample_A0.csv"
MAPPING_JSON = "6-REVIEW_SENTIMENT_ANALYSIS/llm_check_meaning/preprocessing_sentiment.json"  # <-- change this
OUT_CSV = "12-EVALUATION_QUAL/quality_sample_A00.csv"

# 1) load sampled/output csv
df = pd.read_csv(IN_CSV)

# 2) load mapping json (list of dicts)
with open(MAPPING_JSON, "r", encoding="utf-8") as f:
    mapping = json.load(f)

map_df = pd.DataFrame(mapping)

# sanity: keep only needed columns & drop duplicates
map_df = map_df[["model_id", "processed", "original"]].dropna()
map_df = map_df.drop_duplicates(subset=["model_id", "processed"], keep="first")

# 3) figure out which column in df contains the processed review text
# change this to your actual column name in the csv:
PROCESSED_COL = "reviews"  # <-- change if needed (e.g. "text", "content", etc.)

if PROCESSED_COL not in df.columns:
    raise ValueError(f"Couldn't find '{PROCESSED_COL}' in CSV columns: {list(df.columns)}")

# 4) merge
out = df.merge(
    map_df,
    how="left",
    left_on=["model_id", PROCESSED_COL],
    right_on=["model_id", "processed"],
)

# optional: drop the mapping key col from the right side
out = out.drop(columns=["processed"])

# 5) report match rate
missing = out["original"].isna().sum()
print(f"Rows: {len(out)} | Missing originals: {missing}")

# 6) save
out.to_csv(OUT_CSV, index=False)
print(f"Wrote: {OUT_CSV}")
