# SoftwareX release evaluation

This directory is the reproducibility package for the HugSelect `v1.0.0`
release-level study, bundled unchanged with the functionally equivalent
`v1.0.1` submission release. It separates measured runs from protocol metadata
so that missing services cannot be mistaken for completed experiments.

## Protocol

- Twenty predefined natural-language scenarios are stored in `scenarios.json`.
- Each exact query is submitted once to each system in a fresh conversation.
- No follow-up prompt is used.
- For assistants, record the displayed product/model label, run time, browsing
  state, raw response, refusal/truncation state, and the first ten normalized
  `owner/model` identifiers.
- HugSelect runs record the release commit, index name/count, search mode, raw
  results, and elapsed time.
- A service that requires unavailable authentication is recorded as
  `unavailable`; it is never imputed or simulated.

## Commands

Run the locally available HugSelect basic-fallback path:

```bash
.venv/bin/python evaluation/softwarex_v1/run_hugselect.py \
  --output evaluation/softwarex_v1/runs/hugselect.json
```

Normalize and validate captured assistant results:

```bash
.venv/bin/python evaluation/softwarex_v1/analyze.py \
  --runs evaluation/softwarex_v1/runs \
  --output evaluation/softwarex_v1/analysis/results.json
```

`analyze.py` checks candidate existence against the public Hugging Face API and
reports identifier validity, exact-ID agreement with HugSelect, family-level
agreement, and bootstrap confidence intervals. Agreement is not correctness.

## Review schema

`reviews.json` contains independent reviewer judgments when available. Each
judgment records scenario, system, candidate, relevance (`0`/`1`), reviewer ID,
and optional note. Inter-reviewer agreement is calculated only when at least two
independent reviewers assessed the same items; the script otherwise reports it
as unavailable rather than fabricating a value.
