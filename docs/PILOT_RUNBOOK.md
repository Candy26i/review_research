# Pilot data-collection runbook

Pipeline: OpenReview parquet -> arXiv title match -> version chains (OAI-PMH) -> stratified
pilot manifest -> per-version e-print downloads. All scripts run from the repo root with
`python3 src/collect/<name>.py`. Network calls to arXiv are rate-limited to 1 request / 3 s
(arXiv terms), so runtime is roughly 3 s per network call.

## Smoke test (verified 2026-10-20, all steps passed)

```sh
python3 src/collect/load_iclr.py                                          # local, ~1 s, 34,612 rows
python3 src/collect/match_arxiv.py --years 2024 2025 --sample 400 --limit 12 --seed 42   # 12 queries, ~45 s
python3 src/collect/fetch_versions.py                                     # OAI calls for multi-version matches
python3 src/collect/build_pilot_manifest.py --target 4                    # local, instant
python3 src/collect/download_sources.py --max-papers 2                    # 4 files, ~18 MB
```

`--limit 12` caps work to the first 12 candidates of the deterministic stratified
400-sample (seed 42), so every smoke checkpoint row stays valid for the real run.
Re-running step 2 unchanged performs 0 new queries (verified).

## Real pilot run

```sh
python3 src/collect/load_iclr.py
python3 src/collect/match_arxiv.py --years 2024 2025 --sample 400 --limit 0 --seed 42   # 400 queries, ~25 min
python3 src/collect/fetch_versions.py                                     # 1 OAI call per multi-version match
python3 src/collect/build_pilot_manifest.py --target 50
python3 src/collect/download_sources.py --max-papers 0                    # all manifest papers, all versions
```

Run the network scripts sequentially, never concurrently (the rate limiter is per-process).
Optional: `match_arxiv.py --retry-none` re-queries previous misses; downstream, keep the
LAST row per `openreview_id`.

## Output files (one line per record)

- `data/processed/iclr_papers.jsonl` — `openreview_id, year, title, title_norm, authors[],
  decision_raw, decision (accepted|rejected|desk_rejected|withdrawn|...), scores[],
  n_reviews, mean_score (null if no scores), abstract`. OVERWRITTEN each run.
- `data/processed/arxiv_matches.jsonl` — `openreview_id, iclr_year, title, decision,
  mean_score, scores[], n_reviews, arxiv_id, arxiv_title, latest_version, published,
  updated, match_method (exact|fuzzy|none), match_score, author_overlap, date_plausible`.
  Append-only checkpoint on `openreview_id`.
- `data/processed/version_chains.jsonl` — match fields plus `n_versions, chain_source
  (oai|synthesized|oai_error:*), versions[] {v, date, size_kb, source_type, stage,
  stage_fine}` (stages from the ICLR year timeline). Checkpoint on `arxiv_id`; raw OAI XML
  cached in `data/raw/oai_xml/`.
- `data/processed/pilot_manifest.jsonl` — chain fields plus `strata {decision,
  version_bucket (2|3+), spans_stages}`; stratified round-robin selection, deterministic in
  `--seed`, superset across growing `--target`. OVERWRITTEN each run.
- `data/processed/downloads.jsonl` — `key (arxiv_idvN), arxiv_id, v, path, bytes, kind
  (gzip|pdf|other), content_type, stage`. Checkpoint on `key`; archives land in
  `data/raw/sources/<arxiv_id>/` and are never extracted here.

## Resume semantics

Everything is checkpointed; every script is safe to re-run. `match_arxiv`,
`fetch_versions` and `download_sources` append to their output JSONL and skip keys already
present (plus the XML cache for `fetch_versions`), so an interrupted run continues where it
stopped and a completed run does zero network calls. `load_iclr` and
`build_pilot_manifest` are pure-local and overwrite their outputs deterministically.
Scores are raw per-reviewer ratings; scales differ by year, so never compare `mean_score`
across years directly.
