# HugSelect Django Search MVP

This directory contains the Django web interface for HugSelect. The website accepts a natural-language request and recommends relevant Hugging Face models.

## Search flow

```text
Browser search form
→ Django view
→ Gemini requirement extraction
→ HugSelect feature bundle
→ Elasticsearch query and ranking
→ Result cards
```

If feature-based recommendation is unavailable, HugSelect automatically attempts a basic Elasticsearch search.
If Gemini specifically reports exhausted quota, the results page instead lets the user retry the same feature-based search with their own OpenAI API key.

## Requirements

Before starting the website, ensure that:

- the repository virtual environment exists;
- Elasticsearch 7.17.29 is installed;
- the Elasticsearch index `models_t7` exists;
- a Gemini API key is available.

## 1. Go to the repository root

```bash
cd /path/to/tez_master
```

## 2. Activate the virtual environment

```bash
source .venv/bin/activate
```

## 3. Start Elasticsearch

Open a separate terminal and run:

```bash
~/elasticsearch-7.17.29/bin/elasticsearch
```

Elasticsearch should be available at:

```text
http://localhost:9200
```

## 4. Confirm that the model index exists

```bash
curl "http://localhost:9200/_cat/indices/models_t7?v"
```

The output should contain:

```text
models_t7
```

## 5. Set the Gemini API key

Set the key in the terminal session used to start Django:

```bash
export GEMINI_API_KEY="your-api-key-here"
```

Never place a real API key in source code, documentation, screenshots, or Git history.

## 6. Start Django

From the repository root:

```bash
cd hugselect_web
python manage.py check
python manage.py runserver
```

## 7. Open the search page

Open:

```text
http://127.0.0.1:8000/search/
```

Example query:

```text
I need a popular text-generation model with an Apache-2.0 license, preferably recent and not gated.
```

## Search modes

### Feature-based recommendation

Gemini extracts structured requirements such as task, domain, language, license, library, functionality, and quality preferences.

HugSelect creates a feature bundle and uses the feature-aware query builder to rank models in Elasticsearch.

### Basic fallback search

If Gemini or feature extraction is unavailable, HugSelect attempts a simpler Elasticsearch text search.

When fallback succeeds:

- results are still displayed;
- the page shows a friendly warning;
- the search mode is shown as **Basic fallback search**;
- technical exception details are logged only in the Django terminal.

If both search methods fail, the browser displays a friendly error message.

### OpenAI continuation after Gemini quota exhaustion

When Gemini reports that its quota is exhausted, HugSelect keeps the query and structured requirements in the current session and asks for an OpenAI API key on the results page. The key is sent directly to the server for that single retry, is not saved in Django session data, and is not written to application logs. HugSelect uses the OpenAI Responses API with `store=False` and defaults to `gpt-4.1-mini`.

An optional deployment-level override can select another compatible model:

```bash
export OPENAI_FALLBACK_MODEL="gpt-4.1-mini"
```

The user needs OpenAI API access and available API credit; a ChatGPT subscription by itself does not supply API credit.

## Useful test queries

```text
I need a Turkish sentiment analysis model.
```

```text
I want an image generation model based on diffusion, preferably not gated.
```

```text
I need a popular text-generation model with an Apache-2.0 license, preferably recent and not gated.
```

## Troubleshooting

Check Elasticsearch:

```bash
curl http://localhost:9200
```

Check the model index:

```bash
curl "http://localhost:9200/_cat/indices?v"
```

Check whether the Gemini environment variable exists:

```bash
test -n "$GEMINI_API_KEY" && echo "GEMINI_API_KEY is set"
```
