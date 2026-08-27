# Elasticsearch knowledge-base reproduction

HugSelect `v1.0.1` uses index `models_t7` with 69,000 searchable documents.
The latest recorded model modification in the snapshot is 29 April 2025.

## Release files

The GitHub release publishes:

- `models_t7-2025-04-29.ndjson.gz` — mapping/settings metadata and documents;
- `models_t7-2025-04-29.ndjson.gz.sha256` — integrity checksum;
- `models_t7-2025-04-29-manifest.json` — index name, count, source period,
  export date, format version, and release commit.

Verify the downloaded archive before import:

```bash
sha256sum --check models_t7-2025-04-29.ndjson.gz.sha256
```

## Import

Start Elasticsearch 7.17.29 and run:

```bash
python tools/elasticsearch_snapshot.py import \
  --url http://localhost:9200 \
  --index models_t7 \
  --input models_t7-2025-04-29.ndjson.gz
```

Import refuses to overwrite an existing index unless `--replace` is supplied.
It creates the index from the archived mapping/settings, bulk-loads documents,
refreshes the index, and verifies the expected count.

Verify the restored server version and index count independently:

```bash
curl -s http://localhost:9200/
curl -s http://localhost:9200/models_t7/_count
```

The first response must identify Elasticsearch `7.17.29`; the second must
report `69000` documents.

## Export

```bash
python tools/elasticsearch_snapshot.py export \
  --url http://localhost:9200 \
  --index models_t7 \
  --output models_t7-2025-04-29.ndjson.gz
```

Each line after the header is independently parseable JSON with `_id` and
`_source`. This makes the snapshot auditable without proprietary tooling.

## Reconstruction from pipelines

The numbered directories at the repository root preserve the original model
filtering, feature extraction, review collection, quality-evidence mapping,
clustering, and evaluation stages. Those research scripts document provenance,
but the archived index is the normative input for reproducing the released web
application because rebuilding depends on mutable external sources.
