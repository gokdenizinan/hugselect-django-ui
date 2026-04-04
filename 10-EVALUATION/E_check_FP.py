import pandas as pd
from collections import Counter
import ast

# EN COK GORULEN FPLER LISTESI MANUEL OLARAK CIKARMAK ICIN
# FN HALINIDI YAPABILIRSIN
def parse_list_cell(cell):
    if pd.isna(cell):
        return []
    if isinstance(cell, list):
        return cell
    cell = str(cell).strip()

    try:
        parsed = ast.literal_eval(cell)
        if isinstance(parsed, list):
            return parsed
    except Exception:
        pass

    return [x.strip() for x in cell.split(",") if x.strip()]


# Load CSVs
df1 = pd.read_csv("10-EVALUATION/model_ffs_eval_T2_relax.csv")
df2 = pd.read_csv("10-EVALUATION/model_ffs_eval_T2_relax.csv")

# Counters
fp_counter = Counter()
tp_counter = Counter()

for df in [df1, df2]:
    df["fp_examples"] = df["fp_examples"].apply(parse_list_cell)
    df["tp_examples"] = df["tp_examples"].apply(parse_list_cell)

    for items in df["fp_examples"]:
        fp_counter.update(items)

    for items in df["tp_examples"]:
        tp_counter.update(items)

# Union of all items seen in FP or TP
all_items = set(fp_counter) | set(tp_counter)

# Build final table
rows = []
for item in all_items:
    rows.append({
        "item": item,
        "fp_count": fp_counter.get(item, 0),
        "tp_count": tp_counter.get(item, 0),
    })

result_df = pd.DataFrame(rows)

# Optional: sort by FP count (descending)
result_df = result_df.sort_values("fp_count", ascending=False).head(50)

print(result_df)
