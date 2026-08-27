#!/usr/bin/env bash
set -Eeuo pipefail

REPOSITORY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SNAPSHOT_PATH="${1:-}"
COMPOSE=(docker compose --project-directory "${REPOSITORY_ROOT}")
CREATED_ENV_FILE=false
TEMP_DIRECTORY=""

fail() {
  echo "Docker verification failed: $*" >&2
  exit 1
}

cleanup() {
  "${COMPOSE[@]}" down --volumes --remove-orphans >/dev/null 2>&1 || true
  if [[ "${CREATED_ENV_FILE}" == true ]]; then
    rm -f "${REPOSITORY_ROOT}/.env"
  fi
  if [[ -n "${TEMP_DIRECTORY}" ]]; then
    rm -rf "${TEMP_DIRECTORY}"
  fi
}

trap cleanup EXIT

[[ -n "${SNAPSHOT_PATH}" ]] || fail \
  "pass the released models_t7-2025-04-29.ndjson.gz path"
[[ -f "${SNAPSHOT_PATH}" ]] || fail \
  "snapshot not found: ${SNAPSHOT_PATH}"
command -v docker >/dev/null || fail "Docker is not installed"
docker compose version >/dev/null || fail "Docker Compose is unavailable"

SNAPSHOT_PATH="$(cd "$(dirname "${SNAPSHOT_PATH}")" && pwd)/$(basename "${SNAPSHOT_PATH}")"
TEMP_DIRECTORY="$(mktemp -d)"

if [[ ! -f "${REPOSITORY_ROOT}/.env" ]]; then
  cp "${REPOSITORY_ROOT}/.env.example" "${REPOSITORY_ROOT}/.env"
  CREATED_ENV_FILE=true
fi

cd "${REPOSITORY_ROOT}"
"${COMPOSE[@]}" down --volumes --remove-orphans
"${COMPOSE[@]}" build web
"${COMPOSE[@]}" up -d elasticsearch

for _ in {1..60}; do
  if curl -fsS http://127.0.0.1:9200/_cluster/health >/dev/null; then
    break
  fi
  sleep 2
done
curl -fsS http://127.0.0.1:9200/_cluster/health >/dev/null || \
  fail "Elasticsearch did not become healthy"

python3 tools/elasticsearch_snapshot.py import \
  --url http://127.0.0.1:9200 \
  --index models_t7 \
  --input "${SNAPSHOT_PATH}"

DOCUMENT_COUNT="$(
  curl -fsS http://127.0.0.1:9200/models_t7/_count |
    python3 -c 'import json, sys; print(json.load(sys.stdin)["count"])'
)"
[[ "${DOCUMENT_COUNT}" == "69000" ]] || fail \
  "restored ${DOCUMENT_COUNT} documents; expected 69000"

"${COMPOSE[@]}" up -d web
for _ in {1..60}; do
  if curl -fsS http://127.0.0.1:8000/search/ >/dev/null; then
    break
  fi
  sleep 2
done
curl -fsS http://127.0.0.1:8000/search/ >/dev/null || \
  fail "Django did not become healthy"

"${COMPOSE[@]}" exec -T web python manage.py shell -c \
  "from recommender.services import search_models_basic; results = search_models_basic('instruction-following text generation', limit=10); assert results, 'no candidates returned'; print('Representative retrieval:', results[0]['model_id'])"

curl -fsS \
  -c "${TEMP_DIRECTORY}/cookies.txt" \
  -o "${TEMP_DIRECTORY}/search.html" \
  http://127.0.0.1:8000/search/
CSRF_TOKEN="$(
  sed -n 's/.*name="csrfmiddlewaretoken" value="\([^"]*\)".*/\1/p' \
    "${TEMP_DIRECTORY}/search.html" | head -n 1
)"
[[ -n "${CSRF_TOKEN}" ]] || fail "could not read the search CSRF token"

curl -fsS -L \
  -b "${TEMP_DIRECTORY}/cookies.txt" \
  -c "${TEMP_DIRECTORY}/cookies.txt" \
  -e http://127.0.0.1:8000/search/ \
  --data-urlencode "csrfmiddlewaretoken=${CSRF_TOKEN}" \
  --data-urlencode "query=instruction-following text-generation model" \
  --data-urlencode "search_scope=all" \
  -o "${TEMP_DIRECTORY}/results.html" \
  http://127.0.0.1:8000/search/

grep -q "Search results" "${TEMP_DIRECTORY}/results.html" || \
  fail "representative POST did not reach the results page"
grep -Eq "Feature-based recommendation|Basic fallback search" \
  "${TEMP_DIRECTORY}/results.html" || \
  fail "results page did not report a search mode"

echo "Docker verification passed:"
echo "- web image built from Python 3.14.5"
echo "- Elasticsearch 7.17.29 started"
echo "- models_t7 restored with 69000 documents"
echo "- Django search page started"
echo "- representative Elasticsearch retrieval returned candidates"
echo "- CSRF-protected search POST reached the results page"
