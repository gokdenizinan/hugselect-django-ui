import json
import string
from typing import Dict, Any, List, Tuple, Set

INPUT_PATH = "2-NP_EXTRACTION/NP_comfy_GGG.json"
OUT_LLM_PATH = "4-LLM_FEATURE_ORGANIZATION/abbreviation_to_llm_GG.json"
OUT_PROCESSED_PATH = "4-LLM_FEATURE_ORGANIZATION/NP_GG_processed.json"


def create_name(dic: Dict[str, Any]) -> Tuple[Set[str], Set[str], Set[str]]:
    """
    Returns:
      name_set: model names (part after 'author/')
      author_set: authors (part before '/')
      modelid_set: raw model_id strings
    """
    name_set, author_set, modelid_set = set(), set(), set()

    for _, value in dic.items():
        ids = value.get("model_id", [])
        if isinstance(ids, str):
            ids = [ids]

        for mid in ids:
            if not isinstance(mid, str):
                continue
            modelid_set.add(mid)
            if "/" in mid:
                author, name = mid.split("/", 1)
                author_set.add(author)
                name_set.add(name)

    return name_set, author_set, modelid_set


def check_name(feature: str, name_list: Set[str]) -> bool:
    f = (feature or "").strip().lower()
    return any(f == (n or "").strip().lower() for n in name_list)


def is_acronym(np: str) -> bool:
    if not isinstance(np, str):
        return False
    if not (2 <= len(np) <= 7):
        return False
    if not np.isalpha():
        return False
    if sum(c.isupper() for c in np) < 2:
        return False
    if np.istitle():
        return False
    return True


def contains_expansion(acronym: str, sentences) -> Set[str]:
    """
    Looks for expansions matching acronym initials in the provided sentences.
    Returns a set of Title-Cased expansions, or {"(-)"} if none found.
    """
    if isinstance(sentences, str):
        sentences = [sentences]
    if not sentences:
        return {"(-)"}

    expansions = set()
    n = len(acronym)

    for sentence in sentences:
        if not isinstance(sentence, str):
            continue
        initial_words = sentence.split()
        # keep punctuation mostly, but remove parentheses that often surround expansions
        words = [w.replace("(", "").replace(")", "") for w in initial_words]
        words = [w for w in words if w]

        for i in range(len(words) - n + 1):
            window = words[i:i + n]
            initials = "".join(w[0].upper() for w in window if w)
            if initials == acronym.upper():
                expansions.add(" ".join(window).title())

    return expansions if expansions else {"(-)"}


def main():
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        np_info = json.load(f)

    # IMPORTANT: correct order
    name_list, author_list, modelid_list = create_name(np_info)

    kelimeler = []
    cok_basarili = 0  # kept for compatibility; now means "found exactly one expansion in sentences"
    basarili = 0      # means "used some expansion but multiple candidates or fallback"
    basarisiz = 0     # no expansion found

    for key, value in np_info.items():
        feature = value.get("noun_phrase", "")

        # base flags (still valid with new format)
        value["base_author"] = "yes" if check_name(feature, author_list) else "no"
        value["base_model"] = "yes" if check_name(feature, name_list) else "no"
        value["base_modelID"] = "yes" if check_name(feature, modelid_list) else "no"

        if not is_acronym(feature):
            continue

        expansions = contains_expansion(feature, value.get("sentence", []))

        # If exactly one real expansion
        if len(expansions) == 1:
            exp = next(iter(expansions))
            if exp != "(-)":
                np_open = f"{exp} ({feature})"
                value["noun_phrase"] = np_open
                kelimeler.append(np_open)
                cok_basarili += 1
            else:
                # no expansion found anywhere → send to LLM with context
                ctx = value.get("sentence", [])
                ctx_preview = ctx[0] if isinstance(ctx, list) and ctx else str(ctx)
                s = f"{feature}(_)" + f" context: {ctx_preview}"
                value["noun_phrase"] = s
                kelimeler.append(s)
                basarisiz += 1

        else:
            # multiple expansions OR mixture of (-) and real ones
            real = sorted([e for e in expansions if e != "(-)"])
            if real:
                base = "/".join(real) + f" ({feature})"
                # include a short context snippet for the LLM if you want
                ctx = value.get("sentence", [])
                ctx_preview = ctx[0] if isinstance(ctx, list) and ctx else str(ctx)
                with_ctx = base + f" context: {ctx_preview}"
                value["noun_phrase"] = with_ctx
                kelimeler.append(with_ctx)
                basarili += 1
            else:
                ctx = value.get("sentence", [])
                ctx_preview = ctx[0] if isinstance(ctx, list) and ctx else str(ctx)
                s = f"{feature}(_)" + f" context: {ctx_preview}"
                value["noun_phrase"] = s
                kelimeler.append(s)
                basarisiz += 1

    # Build abbreviation_to_llm: keep only unresolved or ambiguous ones
    abbreviation_to_llm = {}
    for i, kelime in enumerate(kelimeler):
        if "(_)" in kelime or "/" in kelime:
            abbreviation_to_llm[f"abbreviation_{i}"] = kelime

    with open(OUT_LLM_PATH, "w", encoding="utf-8") as f:
        json.dump(abbreviation_to_llm, f, indent=2, ensure_ascii=False)

    with open(OUT_PROCESSED_PATH, "w", encoding="utf-8") as f:
        json.dump(np_info, f, indent=2, ensure_ascii=False)

    print(f"Wrote: {OUT_LLM_PATH}")
    print(f"Wrote: {OUT_PROCESSED_PATH}")
    print("acronyms processed:", len(kelimeler))
    print("exactly one expansion:", cok_basarili)
    print("multiple/other expansions used:", basarili)
    print("no expansion:", basarisiz)


if __name__ == "__main__":
    print("doing stuff")
    main()
    print("finished")
