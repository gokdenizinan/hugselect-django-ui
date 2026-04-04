# yapilacaklar:
# elimde NP_G5 VAR
# bundaki npler arasinda kac tanesi bir onceki llmlere sokulmus ? 
# Nerden kontrol etcem? NP_checked kontorl edilmis npler
# NP_info_global_united NP_info_global_suan_fin bunlar abbreviationli
# bunarda bazen sikinti olabilir PSNSR (PSNR) neden boyle arastir
# "Connectionist Temporal Classification (CTC)" buna ne yapacaz aq
# bunlardan npleri al ve g5 ile karsilastir
# A_filter_descriptions abbreviation listesi veriyor.... bu da E_abbreviation_LLMine giriyor buda abbreviation add e giriyor
# Z_NP_LLM ile kalanlari filtrele
# kac tanesinin abbreviationa ihtiyaci var ? kac tanesinin abbreviation acilimi var? 
# elimde 40k feature var recallim 0.80 (chatgptnin feature selectioni bok gibidir belki (manuel olarak featurelari secebilir misin?))

import json

import re

def extract_np_variants(noun_phrase):
    """
    Turns:
    'Multimodal Sentiment Analysis Pipeline (MSP)'
    into:
    {'multimodal sentiment analysis pipeline', 'msp'}
    """
    noun_phrase = noun_phrase.strip()

    variants = set()

    # Lowercase normalized
    base = noun_phrase.lower()
    variants.add(base)

    # If it has an acronym in parentheses
    m = re.search(r"(.*?)\s*\(([^)]+)\)", noun_phrase)
    if m:
        full = m.group(1).strip().lower()
        acronym = m.group(2).strip().lower()
        variants.add(full)
        variants.add(acronym)

    # If it is already an acronym, keep it
    if noun_phrase.isupper() and len(noun_phrase) <= 10:
        variants.add(noun_phrase.lower())

    return variants

def build_np_variant_set(d):
    np_set = set()
    for v in d.values():
        np = v.get("noun_phrase")
        if np:
            np_set |= extract_np_variants(np)
    return np_set

def count_matching_noun_phrases(d1, d2):
    np1 = build_np_variant_set(d1)
    np2 = build_np_variant_set(d2)
    return len(np1 & np2)


def merge_dicts_dedup_by_noun_phrase(dict1, dict2):
    merged = {}
    seen_noun_phrases = set()

    # Iterate in order: dict1 first, then dict2
    for source in (dict1, dict2):
        for key, subdict in source.items():
            noun = subdict.get("noun_phrase")

            # Skip entries without noun_phrase
            if noun is None:
                continue

            # If we haven't seen this noun_phrase yet, keep it
            if noun not in seen_noun_phrases:
                seen_noun_phrases.add(noun)
                merged[key] = subdict

    return merged


with open("4-LLM_FEATURE_ORGANIZATION/NP_O.json", "r") as f:
    DICT_1 = json.load(f)

with open("2-NP_EXTRACTION/NP_X5.json", "r") as f:
    DICT_G = json.load(f)


print(len(DICT_1))
print(len(DICT_G))
print(count_matching_noun_phrases(DICT_1, DICT_G))

OUT_NP_LIST_JSON = "2-NP_EXTRACTION/NP_X5_list.json"
phrase_list = [entry["noun_phrase"]
for entry in DICT_G.values()
if entry.get("noun_phrase")
]

with open(OUT_NP_LIST_JSON, "w", encoding="utf-8") as f:
    json.dump(phrase_list, f, ensure_ascii=False, indent=2)
