# HugSelect v1.0.0

This is the SoftwareX reproducibility release of HugSelect.

## Included

- Django search, results, model-detail, comparison, saved-comparison, reporting,
  and Decision Stress interfaces;
- explicit Must Have, Should Have, Could Have, and Won't Have requirements;
- base-model-family filtering for Llama, Mistral, Qwen, and Gemma;
- Gemini requirement extraction with one-time GPT-4.1 mini continuation after
  explicit Gemini quota exhaustion;
- 147 automated recommender tests;
- a 20-scenario cross-system evaluation package with raw captured responses;
- field-ablation results for the basic fallback path;
- deterministic Elasticsearch export/import tooling;
- Docker Compose configuration and complete installation documentation.

## Knowledge-base asset

`models_t7-2025-04-29.ndjson.gz` contains 69,000 model records. Verify it with
the accompanying SHA-256 file before import.

## Known limitations

- The knowledge base is a fixed snapshot whose latest recorded model
  modification is 29 April 2025.
- Model metadata can be missing or stale; names are not used to infer missing
  base-model lineage.
- Recorded license metadata is evidence and not legal advice.
- The normal feature-extraction path requires Gemini API access. GPT-4.1 mini is
  used only for the user-supplied-key continuation after Gemini quota
  exhaustion. Other extraction failures use the narrower basic fallback.
- Microsoft Copilot and DeepSeek web evaluations could not be run anonymously
  during the release study and are preserved as unavailable records.
- Independent human relevance review was not available during the automated
  run, so no inter-reviewer agreement is claimed.
