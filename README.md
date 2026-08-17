# HugSelect

HugSelect is an evidence-driven, multi-criteria decision-support application
for selecting models from a curated Hugging Face snapshot. It turns a natural-
language request and optional explicit MoSCoW priorities into a ranked,
explainable shortlist. Users can inspect model evidence, compare two or three
models, save comparisons in their browser session, export reports, and test
whether a choice changes under five Decision Stress configurations.

HugSelect supports research decisions; it does not establish an objectively
best model. Stored metadata is evidence rather than ground truth, recorded
licenses are not legal advice, and feature-match percentages are not benchmark
performance scores.

## Released knowledge base

The `v1.0.0` release uses Elasticsearch index `models_t7`, containing **69,000
searchable model records**. The metadata was collected through 29 April 2025.
The compressed release snapshot is published as:

`models_t7-2025-04-29.ndjson.gz`

Download it from the `v1.0.0` GitHub release before following either setup
method below. The release also provides a SHA-256 checksum and an index
manifest. The original collection and evidence-construction pipelines remain
in the numbered repository directories for audit and reconstruction.

## Quick start with Docker Compose

Requirements: Docker with Compose, at least 4 GB free memory, and the released
index snapshot.

```bash
git clone https://github.com/gokdenizinan/hugselect-django-ui.git
cd hugselect-django-ui
git checkout v1.0.0
cp .env.example .env
```

Set `GEMINI_API_KEY` and a unique `DJANGO_SECRET_KEY` in `.env`, then start the
services:

```bash
docker compose up -d elasticsearch
python tools/elasticsearch_snapshot.py import \
  --url http://localhost:9200 \
  --index models_t7 \
  --input /path/to/models_t7-2025-04-29.ndjson.gz
docker compose up --build web
```

Open <http://127.0.0.1:8000/search/>. Confirm the imported record count with:

```bash
curl -s http://localhost:9200/models_t7/_count
```

The count must be `69000`.

## Local installation

HugSelect `v1.0.0` is verified with Python 3.14.5, Django 6.0.7,
Elasticsearch server 7.17.29, and the pinned Python packages in
`requirements.txt`.

```bash
git clone https://github.com/gokdenizinan/hugselect-django-ui.git
cd hugselect-django-ui
git checkout v1.0.0
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Start Elasticsearch 7.17.29 at `http://localhost:9200`, then import the release
snapshot:

```bash
python tools/elasticsearch_snapshot.py import \
  --input /path/to/models_t7-2025-04-29.ndjson.gz
```

Configure and run Django:

```bash
export GEMINI_API_KEY="your-key"
export DJANGO_SECRET_KEY="a-long-random-value"
export DJANGO_DEBUG=true
export DJANGO_ALLOWED_HOSTS="localhost,127.0.0.1"
cd hugselect_web
python manage.py migrate
python manage.py check
python manage.py runserver
```

## API keys and privacy

- `GEMINI_API_KEY` enables the normal structured-requirement extraction path.
- OpenAI is used only after Gemini explicitly reports exhausted quota. The
  results page then asks the user for an OpenAI API key for a single retry.
- The OpenAI fallback defaults to `gpt-4.1-mini`. Set
  `OPENAI_FALLBACK_MODEL` only to intentionally override that deployment
  default.
- A submitted OpenAI key is not stored in Django session data or application
  logs. The retry uses the Responses API with `store=False`; the query is still
  transmitted to OpenAI for processing.
- Never place API keys in source code, screenshots, shell history, reports, or
  Git. Use environment variables or the one-time fallback form.
- If extraction fails for a reason other than Gemini quota exhaustion,
  HugSelect can use its narrower basic Elasticsearch fallback.

## Search and ranking

The normal path is:

```text
query + explicit priorities
→ Gemini requirement extraction
→ hard Must Have / Won't Have feasibility filters
→ Elasticsearch candidate retrieval
→ weighted match and restrained popularity scoring
→ explainable shortlist
```

Must Have criteria are Elasticsearch filters: missing or non-matching evidence
excludes a model. Won't Have criteria are `must_not` clauses: matching evidence
excludes a model, while missing evidence does not. Should Have and Could Have
criteria influence ranking without excluding a candidate.

For each requested value, HugSelect retains the best indexed match and computes
the displayed feature-match percentage as the matched weighted evidence divided
by possible weighted evidence. Exact matches receive factor `1.0`, normalized
gram matches `0.90`, and synonym matches `0.80 × synonym confidence`. Missing
soft evidence contributes zero to the numerator but remains in the denominator.
Popularity adds logarithmic likes (`0.8`) and 30-day downloads (`0.75`) within
a capped rank-function contribution of `70`. Quality dimensions are stored and
shown as evidence, but quality-dimension query expansion is disabled in the
released web configuration. Comparison ties are resolved by 30-day downloads,
then likes, then stable model ID ordering.

The exact feature weights and Decision Stress multipliers are documented in
`docs/ranking.md`.

## Decision Stress

Decision Stress re-scores already stored comparison evidence. It does not rerun
Gemini, Elasticsearch retrieval, or external benchmarks. The five configurations
are starting priorities, essential-first, preference-first, functional-first,
and strict essentials. Results show winners, ties, exclusions, category
contributions, and outright-win consistency.

## Tests

Run the exact release verification suite:

```bash
.venv/bin/python hugselect_web/manage.py test recommender -v 2
```

The `v1.0.0` release contains 147 automated recommender tests. The SoftwareX
evaluation package, scenario definitions, raw response schema, normalization,
and analysis scripts are under `evaluation/softwarex_v1/`.

## Index export and recovery

Create a new reproducible NDJSON snapshot:

```bash
python tools/elasticsearch_snapshot.py export \
  --output models_t7-2025-04-29.ndjson.gz
```

The export contains one metadata line with mapping/settings followed by one
line per document. Import validates the document count. See
`docs/knowledge-base.md` for format, checksum, and recovery details.

## License and support

HugSelect is released under the [MIT License](LICENSE).

Support contact: `siamak.farshidi@wur.nl`

## Citation

The immutable SoftwareX release DOI will be added here after the GitHub release
has been archived. Until then, cite the exact `v1.0.0` GitHub release rather
than a moving branch.
