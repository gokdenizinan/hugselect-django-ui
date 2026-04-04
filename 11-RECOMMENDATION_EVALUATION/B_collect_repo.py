from __future__ import annotations

import base64
import dataclasses
import os
import re
import time
from typing import Dict, Iterable, List, Optional, Set, Tuple
from urllib.parse import urlparse

import requests


# ----------------------------
# Regexes for HF evidence
# ----------------------------

RE_HF_URL = re.compile(
    r"https?://huggingface\.co/([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+)",
    re.IGNORECASE,
)

# Handles CarperAI__FIM-NeoX-1.3B
RE_ORG__MODEL = re.compile(r"\b([A-Za-z0-9_.-]+)__([A-Za-z0-9_.-]+)\b")

# ORG/MODEL might match other things; we'll gate it by context words nearby.
RE_ORG_MODEL = re.compile(r"\b([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+)\b")

HF_CONTEXT = re.compile(
    r"(huggingface|hugging\s+face|transformers|diffusers|from_pretrained|"
    r"hf_hub_download|snapshot_download|pipeline\s*\(|AutoModel|AutoTokenizer|"
    r"AutoProcessor|model_name_or_path|repo_id\s*=)",
    re.IGNORECASE,
)

# Common model-loading signatures (helpful for evidence even when model ID is local path)
MODEL_LOAD_CUES = re.compile(
    r"(from_pretrained\s*\(|hf_hub_download\s*\(|snapshot_download\s*\(|pipeline\s*\(|repo_id\s*=)",
    re.IGNORECASE,
)

# Library evidence (you can expand)
LIBRARY_PATTERNS = {
    "transformers": re.compile(r"\btransformers\b", re.IGNORECASE),
    "diffusers": re.compile(r"\bdiffusers\b", re.IGNORECASE),
    "accelerate": re.compile(r"\baccelerate\b", re.IGNORECASE),
    "huggingface_hub": re.compile(r"\bhuggingface_hub\b", re.IGNORECASE),
    "torch": re.compile(r"\b(torch|pytorch)\b", re.IGNORECASE),
    "tensorflow": re.compile(r"\btensorflow\b", re.IGNORECASE),
    "jax": re.compile(r"\bjax\b", re.IGNORECASE),
    "peft": re.compile(r"\bpeft\b", re.IGNORECASE),
    "trl": re.compile(r"\btrl\b", re.IGNORECASE),
    "datasets": re.compile(r"\bdatasets\b", re.IGNORECASE),
}


# ----------------------------
# File selection
# ----------------------------

ALLOWED_EXTS = {
    ".py", ".ipynb", ".md", ".txt",
    ".yaml", ".yml", ".json", ".toml",
    ".sh", ".bash", ".zsh",
    ".cfg", ".ini",
}
ALLOWED_FILENAMES = {
    "dockerfile", "makefile", "requirements.txt", "environment.yml", "environment.yaml",
    "setup.py", "pyproject.toml",
}
SKIP_DIR_PARTS = {".git", ".github", "venv", ".venv", "__pycache__", "node_modules", "dist", "build"}
MAX_FILE_BYTES = 750_000  # keep it safe/fast


# ----------------------------
# Data classes
# ----------------------------

@dataclasses.dataclass
class EvidenceLine:
    path: str
    line_no: int
    snippet: str


@dataclasses.dataclass
class RepoScanResult:
    repo: str  # "owner/name"
    default_branch: str
    scanned_files: int
    skipped_files: int

    hf_models: List[str]
    hf_evidence: List[EvidenceLine]

    library_hits: List[str]
    library_evidence: Dict[str, List[EvidenceLine]]

    model_load_evidence: List[EvidenceLine]  # from_pretrained etc.


# ----------------------------
# GitHub API client
# ----------------------------
import time
import requests

class GitHubClient:
    def __init__(self, token: str, sleep_s: float = 0.25):
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {token}",   # modern GitHub auth
            "Accept": "application/vnd.github+json",
            "User-Agent": "hf-case-study-scanner"
        })
        self.sleep_s = sleep_s

    # --------- RATE-LIMIT SAFE CORE ---------
    def get_json(self, url: str, params=None) -> dict:
        max_attempts = 8
        for attempt in range(max_attempts):
            r = self.session.get(url, params=params, timeout=30)

            if 200 <= r.status_code < 300:
                time.sleep(self.sleep_s)
                return r.json()

            if r.status_code in (403, 429):
                remaining = r.headers.get("X-RateLimit-Remaining")
                reset = r.headers.get("X-RateLimit-Reset")
                retry_after = r.headers.get("Retry-After")
                body = (r.text or "").lower()

                # Primary rate limit
                if remaining == "0" and reset and reset.isdigit():
                    wait = max(1, int(reset) - int(time.time()) + 2)
                    print(f"GitHub rate limit exhausted. Sleeping {wait}s")
                    time.sleep(wait)
                    continue

                # Secondary rate limit / abuse
                if "secondary rate limit" in body or "abuse detection" in body:
                    wait = int(retry_after) if retry_after and retry_after.isdigit() else min(180, 10 * (attempt + 1))
                    print(f"GitHub secondary rate limit. Sleeping {wait}s")
                    time.sleep(wait)
                    continue

                # Non-retryable 403 (bad token, SSO, forbidden)
                raise RuntimeError(
                    f"GitHub 403 forbidden for {url}\n"
                    f"Remaining={remaining} Reset={reset}\n"
                    f"Body={r.text[:300]}"
                )

            if r.status_code in (500, 502, 503, 504):
                wait = min(120, 2 ** attempt)
                print(f"GitHub {r.status_code} server error. Sleeping {wait}s")
                time.sleep(wait)
                continue

            raise RuntimeError(f"GitHub API error {r.status_code} for {url}: {r.text[:300]}")

        raise RuntimeError(f"GitHub API failed after retries: {url}")

    # --------- THESE MUST EXIST ---------
    def get_repo_info(self, owner: str, name: str) -> dict:
        return self.get_json(f"https://api.github.com/repos/{owner}/{name}")

    def list_tree_recursive(self, owner: str, name: str, branch: str):
        ref = self.get_json(f"https://api.github.com/repos/{owner}/{name}/git/ref/heads/{branch}")
        sha = ref["object"]["sha"]
        tree = self.get_json(
            f"https://api.github.com/repos/{owner}/{name}/git/trees/{sha}",
            params={"recursive": "1"}
        )
        return tree.get("tree", [])

    def fetch_file_contents(self, owner: str, name: str, path: str, ref: str):
        data = self.get_json(
            f"https://api.github.com/repos/{owner}/{name}/contents/{path}",
            params={"ref": ref},
        )

        enc = data.get("encoding", "")
        content = data.get("content", "")

        if enc == "base64" and content:
            import base64
            return base64.b64decode(content), enc

        # fallback: download_url
        dl = data.get("download_url")
        if dl:
            r = self.session.get(dl, timeout=30)
            r.raise_for_status()
            time.sleep(self.sleep_s)
            return r.content, "raw"

        return b"", enc

# ----------------------------
# Helpers
# ----------------------------

def parse_github_repo(url_or_slug: str) -> Optional[Tuple[str, str]]:
    """
    Accepts:
      - https://github.com/owner/repo
      - https://github.com/owner/repo/
      - owner/repo
    """
    s = url_or_slug.strip()
    if not s:
        return None
    if "github.com" in s:
        u = urlparse(s)
        parts = [p for p in u.path.split("/") if p]
        if len(parts) >= 2:
            return parts[0], parts[1].replace(".git", "")
        return None
    # slug
    if "/" in s:
        owner, repo = s.split("/", 1)
        return owner, repo
    return None


def should_skip_path(path: str) -> bool:
    parts = [p.lower() for p in path.split("/") if p]
    if any(p in SKIP_DIR_PARTS for p in parts[:-1]):
        return True
    return False


def is_allowed_file(path: str) -> bool:
    if should_skip_path(path):
        return False
    base = path.split("/")[-1]
    low = base.lower()

    if low in ALLOWED_FILENAMES:
        return True
    for ext in ALLOWED_EXTS:
        if low.endswith(ext):
            return True

    # allow README-like even without ext
    if low.startswith("readme"):
        return True
    return False


def normalize_hf_id(org: str, model: str) -> str:
    return f"{org.strip()}/{model.strip()}"


def extract_hf_models_and_evidence(text: str, path: str) -> Tuple[Set[str], List[EvidenceLine], List[EvidenceLine]]:
    """
    Returns:
      hf_ids, hf_evidence_lines, model_load_evidence_lines
    """
    hf_ids: Set[str] = set()
    hf_evidence: List[EvidenceLine] = []
    load_evidence: List[EvidenceLine] = []

    lines = text.splitlines()
    for idx, line in enumerate(lines, 1):
        # model load cues
        if MODEL_LOAD_CUES.search(line):
            load_evidence.append(EvidenceLine(path=path, line_no=idx, snippet=line.strip()[:300]))

        # HF URL
        for m in RE_HF_URL.finditer(line):
            mid = normalize_hf_id(m.group(1), m.group(2))
            hf_ids.add(mid)
            hf_evidence.append(EvidenceLine(path=path, line_no=idx, snippet=line.strip()[:300]))

        # ORG__MODEL
        for m in RE_ORG__MODEL.finditer(line):
            mid = normalize_hf_id(m.group(1), m.group(2))
            hf_ids.add(mid)
            hf_evidence.append(EvidenceLine(path=path, line_no=idx, snippet=line.strip()[:300]))

        # ORG/MODEL (only with HF context nearby)
        for m in RE_ORG_MODEL.finditer(line):
            # guard: require HF context in same line
            if not HF_CONTEXT.search(line):
                continue
            mid = normalize_hf_id(m.group(1), m.group(2))
            # Extra guard: avoid common false positives like "http://", "https://"
            if mid.lower().startswith(("http:/", "https:/")):
                continue
            hf_ids.add(mid)
            hf_evidence.append(EvidenceLine(path=path, line_no=idx, snippet=line.strip()[:300]))

    return hf_ids, hf_evidence, load_evidence


def extract_libraries_and_evidence(text: str, path: str) -> Tuple[Set[str], Dict[str, List[EvidenceLine]]]:
    hits: Set[str] = set()
    evidence: Dict[str, List[EvidenceLine]] = {k: [] for k in LIBRARY_PATTERNS.keys()}

    lines = text.splitlines()
    for idx, line in enumerate(lines, 1):
        for lib, pat in LIBRARY_PATTERNS.items():
            if pat.search(line):
                hits.add(lib)
                if len(evidence[lib]) < 5:
                    evidence[lib].append(EvidenceLine(path=path, line_no=idx, snippet=line.strip()[:300]))

    # drop empty evidence lists
    evidence = {k: v for k, v in evidence.items() if v}
    return hits, evidence


def decode_text(raw: bytes) -> Optional[str]:
    # Try utf-8, then latin-1 fallback
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        try:
            return raw.decode("latin-1")
        except UnicodeDecodeError:
            return None


# ----------------------------
# Main scanner
# ----------------------------

def scan_github_repo(
    repo_url_or_slug: str,
    token: str,
    ref: Optional[str] = None,
    max_files: int = 100,
    max_total_bytes: int = 1_000_000,
) -> RepoScanResult:
    """
    Scans a GitHub repo for HF model IDs and evidence.

    - token: GitHub personal access token (classic is fine)
    - ref: branch/commit; if None uses default branch
    - max_files: limit number of files fetched (keep it fast)
    - max_total_bytes: total bytes across fetched files (keep it fast)
    """
    parsed = parse_github_repo(repo_url_or_slug)
    if not parsed:
        raise ValueError(f"Cannot parse GitHub repo from: {repo_url_or_slug}")
    owner, name = parsed

    gh = GitHubClient(token=token)

    info = gh.get_repo_info(owner, name)
    default_branch = info.get("default_branch", "main")
    branch = ref or default_branch

    tree = gh.list_tree_recursive(owner, name, branch)

    # Filter to allowed files and manageable size
    candidates = []
    for node in tree:
        if node.get("type") != "blob":
            continue
        path = node.get("path") or ""
        size = node.get("size") or 0
        if not path or size <= 0:
            continue
        if size > MAX_FILE_BYTES:
            continue
        if not is_allowed_file(path):
            continue
        candidates.append((path, size))

    # sort smaller first (more likely configs/readmes; cheaper)
    candidates.sort(key=lambda x: x[1])

    scanned_files = 0
    skipped_files = 0
    total_bytes = 0

    hf_ids_all: Set[str] = set()
    hf_evidence_all: List[EvidenceLine] = []
    load_evidence_all: List[EvidenceLine] = []

    lib_hits_all: Set[str] = set()
    lib_evidence_all: Dict[str, List[EvidenceLine]] = {}

    for path, size in candidates:
        if scanned_files >= max_files:
            skipped_files += 1
            continue
        if total_bytes + size > max_total_bytes:
            skipped_files += 1
            continue

        raw, _enc = gh.fetch_file_contents(owner, name, path, branch)
        if not raw:
            skipped_files += 1
            continue

        total_bytes += len(raw)
        text = decode_text(raw)
        if text is None:
            skipped_files += 1
            continue

        scanned_files += 1

        hf_ids, hf_ev, load_ev = extract_hf_models_and_evidence(text, path)
        hf_ids_all |= hf_ids
        # If we already found HF models AND some model-loading calls, stop early
        if hf_ids_all and len(load_evidence_all) >= 3:
            break

        # keep evidence bounded
        if len(hf_evidence_all) < 60:
            hf_evidence_all.extend(hf_ev[: max(0, 60 - len(hf_evidence_all))])
        if len(load_evidence_all) < 40:
            load_evidence_all.extend(load_ev[: max(0, 40 - len(load_evidence_all))])

        libs, lib_ev = extract_libraries_and_evidence(text, path)
        lib_hits_all |= libs
        # merge evidence dict, cap each lib to 5 lines
        for lib, evs in lib_ev.items():
            lib_evidence_all.setdefault(lib, [])
            if len(lib_evidence_all[lib]) < 5:
                lib_evidence_all[lib].extend(evs[: max(0, 5 - len(lib_evidence_all[lib]))])

    return RepoScanResult(
        repo=f"{owner}/{name}",
        default_branch=default_branch,
        scanned_files=scanned_files,
        skipped_files=skipped_files,
        hf_models=sorted(hf_ids_all),
        hf_evidence=hf_evidence_all,
        library_hits=sorted(lib_hits_all),
        library_evidence=lib_evidence_all,
        model_load_evidence=load_evidence_all,
    )

import re
from urllib.parse import urlparse

TRAILING_JUNK = re.compile(r"[)\].,;:]+$")

def clean_github_url(url: str) -> str:
    """
    Remove trailing punctuation and normalize GitHub URLs.
    """
    if not url:
        return url
    u = url.strip()
    u = TRAILING_JUNK.sub("", u)

    # normalize to https://github.com/owner/repo
    if "github.com" in u:
        p = urlparse(u)
        parts = [x for x in p.path.split("/") if x]
        if len(parts) >= 2:
            return f"https://github.com/{parts[0]}/{parts[1]}"
    return u


import os, json, time

def save_checkpoint(data, path):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)   # atomic write


# ----------------------------
# Convenience: token from env
# ----------------------------

def scan_repo_with_env_token(repo_url_or_slug: str, ref: Optional[str] = None) -> RepoScanResult:
    token = os.getenv("GITHUB_TOKEN", "").strip()
    if not token:
        raise RuntimeError("Set GITHUB_TOKEN in your environment or .env file.")
    return scan_github_repo(repo_url_or_slug, token=token, ref=ref)


import json
import os

# IN_JSON = "11-RECOMMENDATION_EVALUATION/paper_model_2/hf_case_study_candidates_openalex_3F.json"
# OUT_JSON = "11-RECOMMENDATION_EVALUATION/paper_model_2/hf_case_study_candidates_openalex_with_repo_scans_3.json"

IN_JSON = "11-RECOMMENDATION_EVALUATION/MORE_PAPERS/Candidates_SS_Batch_all_Dedup.json"
OUT_JSON = "11-RECOMMENDATION_EVALUATION/MORE_PAPERS/Candidates_SS_Batch_all_RepoScans.json"
def main():
    token = os.getenv("GITHUB_TOKEN", "").strip()
    if not token:
        raise RuntimeError("Set GITHUB_TOKEN in your environment (.env).")

    if os.path.exists(OUT_JSON):
        print("Resuming from existing checkpoint:", OUT_JSON)
        data = json.load(open(OUT_JSON, "r", encoding="utf-8"))
    else:
        data = json.load(open(IN_JSON, "r", encoding="utf-8"))


    for idx, paper in enumerate(data, 1):
        # only enrich likely candidates
        if paper.get("decision") not in ("include", "review"):
            continue

        repos = paper.get("github_repos") or []

        # ✅ DO NOT wipe prior results
        paper.setdefault("repo_scans", [])

        # ✅ compute which repos are already done
        done = set()
        for x in paper["repo_scans"]:
            if not isinstance(x, dict):
                continue
            # prefer repo_url (cleaned URL we store), else repo slug
            if x.get("repo_url"):
                done.add(x["repo_url"])
            if x.get("repo"):
                done.add(x["repo"])

        for repo_url in repos:
            clean = clean_github_url(repo_url)

            # ✅ skip if already scanned
            if clean in done:
                continue

            try:
                result = scan_github_repo(token=token, repo_url_or_slug=clean)

                paper["repo_scans"].append({
                    "repo_url": clean,  # ✅ store for skip logic
                    "repo": result.repo,
                    "default_branch": result.default_branch,
                    "scanned_files": result.scanned_files,
                    "skipped_files": result.skipped_files,
                    "hf_models": result.hf_models,
                    "library_hits": result.library_hits,
                    "hf_evidence": [e.__dict__ for e in result.hf_evidence],
                    "model_load_evidence": [e.__dict__ for e in result.model_load_evidence],
                    "library_evidence": {k: [e.__dict__ for e in v] for k, v in result.library_evidence.items()},
                })

            except Exception as e:
                paper["repo_scans"].append({
                    "repo_url": clean,  # ✅ store cleaned url even on error
                    "error": str(e),
                })

            # ✅ checkpoint after each repo so nothing is lost
            save_checkpoint(data, OUT_JSON)

        # optional: print progress for your sanity
        print(f"[{idx}/{len(data)}] processed paper: {paper.get('title','')[:80]}")


    json.dump(data, open(OUT_JSON, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"Saved: {OUT_JSON}")

if __name__ == "__main__":
    os.environ["GITHUB_TOKEN"]= "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"
    main()
