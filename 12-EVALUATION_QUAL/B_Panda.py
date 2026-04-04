
import json
import re
import copy

def exract_output():
    with open("10-EVALUATION/lmm_ffs/output_B1.json") as f:
        a50_all = json.load(f)


    def merge_output_json(record):
        new_record = copy.deepcopy(record)
        output = record.get("output", "")

        # Match ```json ... ``` OR ``` ... ```
        match = re.search(
            r"```(?:json)?\s*([\s\S]*?)```",
            output,
            re.IGNORECASE
        )

        if match:
            json_str = match.group(1).strip()
            try:
                parsed = json.loads(json_str)
                new_record.update(parsed)
            except json.JSONDecodeError:
                # JSON inside code block wasn’t valid – optionally log or ignore
                pass

        return new_record

    # Apply to every record in your dictionary
    a50_all_expanded = {}

    for key, record in a50_all.items():
        a50_all_expanded[key] = merge_output_json(record)

    # Keep all items except those with mode_2 == -1 AND Sentiment == "Neutral"
    a50_all_expanded_filtered = {
        key: val
        for key, val in a50_all_expanded.items()
        if not (val.get("old_mode_2") == -1 and val.get("old_Sentiment") == "Neutral")
    }
    with open("10-EVALUATION/lmm_ffs/output_B1_exanded.json", "w", encoding="utf-8") as f:
        json.dump(a50_all_expanded_filtered, f, indent=2, ensure_ascii=False)

