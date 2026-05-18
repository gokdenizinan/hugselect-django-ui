import re
from typing import Any, Dict, List, Tuple, Optional

# ---- helpers to label features nicely ----

# SCORING HELPER
_WEIGHT_RE = re.compile(r"weight\((.*?)\s+in\s+\d+\)")  # grabs "Metadata.tags:image-classification"
_FVF_RE = re.compile(r"field value function:\s*(.*)")
_CONST_RE = re.compile(r"ConstantScore\((.*)\)\^([0-9.]+)")

def _simplify_feature(desc: str) -> str:
    """Turn ES description strings into compact feature labels."""
    m = _WEIGHT_RE.search(desc)
    if m:
        return m.group(1)

    m = _FVF_RE.search(desc)
    if m:
        # Example: log1p(doc['Metadata.likes'].value?:0.0 * factor=1.0)
        return m.group(1).strip()

    m = _CONST_RE.search(desc)
    if m:
        return f"ConstantScore^{m.group(2)}"

    # fallback: keep it short-ish
    return desc.strip()


def _find_child(desc_contains: str, details: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    for d in details:
        if desc_contains in (d.get("description") or ""):
            return d
    return None


def _extract_score_contributors(expl: Dict[str, Any]) -> List[Tuple[str, float]]:
    """
    Returns (feature, contribution_value) pairs:
    - Uses internal node values for real contributions (BM25 weight nodes, ConstantScore, product-of function nodes)
    - Skips BM25 parameter leaves (idf/tf/k1/b/N/n/avgdl/etc.)
    """
    out: List[Tuple[str, float]] = []
    if not expl:
        return out

    desc = expl.get("description", "") or ""
    value = float(expl.get("value", 0.0) or 0.0)
    details = expl.get("details") or []

    # 1) Atomic BM25 term contribution: stop at "weight(...)[PerFieldSimilarity]" node
    # This node's value is the clause score contribution (e.g. 2.2 or similar).
    if "weight(" in desc and "PerFieldSimilarity" in desc:
        out.append((_simplify_feature(desc), value))
        return out

    # 2) ConstantScore contributions
    if desc.startswith("ConstantScore(") or "ConstantScore(" in desc:
        out.append((_simplify_feature(desc), value))
        return out

    # 3) Collapse "product of" (typically field_value_factor * weight)
    # ES often shows:
    #   product of:
    #     field value function: log1p(...)
    #     weight
    # The real contribution is the product node's value.
    if desc.strip().startswith("product of") and details:
        fvf = _find_child("field value function:", details)
        w = _find_child("weight", details)

        if fvf:
            label = _simplify_feature(fvf.get("description", "field value function"))
            if w:
                wv = float(w.get("value", 0.0) or 0.0)
                label = f"{label} * weight({wv:g})"
            out.append((label, value))
            return out

        # Other "product of" cases (script score, decay, etc.) – still treat as atomic
        out.append((_simplify_feature(desc), value))
        return out

    # 4) Skip parameter-only leaves (idf/tf internals, BM25 params, maxBoost)
    # If it’s a leaf and looks like a parameter, ignore it.
    if not details:
        junk_markers = (
            "computed as", " from:", "idf", "tf,", "k1", "b,", "avgdl", "dl,",
            "freq", "N,", "n,", "maxBoost"
        )
        if any(j in desc for j in junk_markers):
            return out
        # Otherwise, keep unknown leaf (rare, but can happen)
        out.append((_simplify_feature(desc), value))
        return out

    # 5) Structural nodes: sum/min/max/etc. → recurse
    for d in details:
        out.extend(_extract_score_contributors(d))

    return out

# DIAGNOSTIC HELPER

def _extract_bool_query(final_query: Dict[str, Any]) -> Dict[str, Any] | None:
    """
    Pulls out the bool query from:
    query -> function_score -> query -> bool
    """
    try:
        return final_query["query"]["function_score"]["query"]["bool"]
    except Exception:
        return None

def _build_no_hits_diagnostics(bool_q: Dict[str, Any]) -> Dict[str, Any]:
    def as_list(x):
        if not x:
            return []
        return x if isinstance(x, list) else [x]

    musts = as_list(bool_q.get("must"))
    filters = as_list(bool_q.get("filter"))
    shoulds = as_list(bool_q.get("should"))
    must_nots = as_list(bool_q.get("must_not"))

    filters_aggs = {}

    def add(kind: str, i: int, clause: Dict[str, Any]):
        filters_aggs[f"{kind}[{i}]"] = clause

    for i, c in enumerate(musts):
        add("must", i, c)
    for i, c in enumerate(filters):
        add("filter", i, c)
    for i, c in enumerate(shoulds):
        add("should", i, c)
    for i, c in enumerate(must_nots):
        add("must_not_matches", i, c)

    return {
        "size": 0,
        "track_total_hits": True,
        "query": {"match_all": {}},
        "aggs": {
            "clauses": {
                "filters": {
                    "filters": filters_aggs
                },
                "aggs": {
                    "examples": {
                        "top_hits": {
                            "size": 2,
                            "_source": True
                        }
                    }
                }
            },
            "required_all": {
                "filter": {
                    "bool": {
                        "must": musts + filters
                    }
                }
            }
        }
    }
def _format_no_hits_diagnostics(resp: Dict[str, Any]) -> Dict[str, Any]:
    buckets = resp["aggregations"]["clauses"]["buckets"]
    required_all = resp["aggregations"]["required_all"]["doc_count"]

    blocking_required = []
    matching_optional = []

    for k, b in buckets.items():
        if b["doc_count"] == 0 and (k.startswith("must[") or k.startswith("filter[")):
            blocking_required.append(k)
        if b["doc_count"] > 0 and k.startswith("should["):
            matching_optional.append(k)

    return {
        "required_all_doc_count": required_all,
        "blocking_required_clauses": blocking_required,
        "optional_clauses_matching_docs": matching_optional,
        "clause_match_counts": {k: b["doc_count"] for k, b in buckets.items()}
    }


# DIAGNOSTIC RECURSIVE
from typing import Any, Dict, List

def _is_bool_clause(q: Dict[str, Any]) -> bool:
    return isinstance(q, dict) and "bool" in q and isinstance(q["bool"], dict)

def _as_list(x):
    if not x:
        return []
    return x if isinstance(x, list) else [x]

def _leaf_label(q: Dict[str, Any]) -> str:
    # Make compact labels for common leaf queries
    if "term" in q:
        (field, val), = q["term"].items()
        return f"term({field}={val})"
    if "match" in q:
        (field, val), = q["match"].items()
        return f"match({field})"
    if "range" in q:
        (field, spec), = q["range"].items()
        return f"range({field})"
    # fallback
    return next(iter(q.keys()), "clause")

def _expand_clauses(clause: Dict[str, Any], path: str, out: Dict[str, Dict[str, Any]], depth: int = 0, max_depth: int = 6):
    """
    Add this clause and recursively add children if it is a bool clause.
    """
    out[path] = clause

    if depth >= max_depth:
        return

    if not _is_bool_clause(clause):
        return

    b = clause["bool"]
    for key in ("must", "filter", "should", "must_not"):
        items = _as_list(b.get(key))
        for i, child in enumerate(items):
            child_path = f"{path}.bool.{key}[{i}]"
            if _is_bool_clause(child):
                _expand_clauses(child, child_path, out, depth + 1, max_depth)
            else:
                out[f"{child_path}.{_leaf_label(child)}"] = child


def _build_no_hits_diagnostics_recursive(bool_q: Dict[str, Any]) -> Dict[str, Any]:
    filters_aggs: Dict[str, Any] = {}

    # Expand top-level bool children
    top = {"bool": bool_q}  # wrap so expander handles it uniformly
    _expand_clauses(top, "root", filters_aggs)

    # optional: remove the root wrapper entry if you don’t want it
    filters_aggs.pop("root", None)

    return {
        "size": 0,
        "track_total_hits": True,
        "query": {"match_all": {}},
        "aggs": {
            "clauses": {
                "filters": {"filters": filters_aggs}
            },
            "required_all": {
                "filter": {"bool": {"must": _as_list(bool_q.get("must")) + _as_list(bool_q.get("filter"))}}
            }
        }
    }
