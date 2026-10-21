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

## Processing (verified 2026-10-20, full 50-paper run)

```sh
python3 src/process/extract_sources.py    # 144 archives + 1 PDF fallback (1 network call), ~30 s
python3 src/process/latex_to_text.py      # LaTeX -> scrubbed markdown, all versions, ~40 s
python3 src/process/scrub_and_audit.py    # email/URL/leak-line scrub + identity-leak audit, ~3 s
python3 src/process/diff_versions.py      # churn metrics for adjacent version pairs, ~20 s
```

Archive contents are untrusted: extraction uses `tarfile` with `filter='data'` only, a
500 MB per-archive cap, and nothing extracted is ever executed or imported.

### Output files (one line per record)

- `data/processed/extracted.jsonl` — `key (arxiv_idvN), arxiv_id, v, dir, kind
  (tar|single_tex|pdf), n_files, tex_files[], has_bbl, bytes_extracted, ok, error`;
  trees land in `data/raw/extracted/<safe_id>/vN/`. Checkpoint on `key`.
- `data/processed/latex_log.jsonl` — `key, arxiv_id, v, main_tex, converter
  (pylatexenc|detex_fallback|pdf), n_inputs_flattened, missing_inputs[], finalcopy_flag,
  tex_title, n_chars, ok, error`; markdown in `data/processed/texts_raw/<safe_id>/vN.md`.
  Checkpoint on `key`; only `ok` extraction rows are converted.
- `data/processed/scrub_log.jsonl` — `key, arxiv_id, v, n_chars, n_words, est_tokens,
  removed {emails, urls, leak_lines}, leak_flags {author_surnames{}, short_surname_hits{},
  reviewer, rebuttal, iclr, neurips, arxiv, acknowledg, github.com}, sha256`; final texts
  in `data/processed/texts/<safe_id>/vN.md`. Checkpoint on `key`; audit counts only,
  removes nothing.
- `data/processed/version_diffs.jsonl` — `arxiv_id, v_from, v_to, stage_from, stage_to,
  ratio, churn, churn_class (cosmetic|minor|major), words_from, words_to, words_added,
  words_removed`. OVERWRITTEN each run (no checkpoint; the whole pilot diffs in seconds).

### Resume and surgical reset

The first three logs are append-only and keyed on `key = "<arxiv_id>v<v>"`: re-running
skips keys already present, so an interrupted run continues where it stopped and a
completed run is a no-op. `scrub_and_audit` reports a missing input file as
`missing input (not checkpointed, rerun later)` and does NOT checkpoint it.

To re-process specific keys after a module fix (surgical reset): (1) delete those keys'
rows from the affected log(s) — `latex_log.jsonl` and/or `scrub_log.jsonl`, (2) delete
their output files `texts_raw/<safe_id>/vN.md` / `texts/<safe_id>/vN.md`, (3) re-run the
affected step(s) — only the deleted keys are recomputed — and (4) re-run
`diff_versions.py`, which always rebuilds from scratch. When a conversion fix changes
output for only SOME versions of a paper, re-run every affected version of that paper, or
the version diff measures converter drift instead of author edits.

### Known data caveats (this pilot)

- `2301.12935` v2 was withdrawn by the authors on arXiv (1 KB stub, `source_type: "I"`,
  no source and no PDF exist — both 404). It stays `ok=false` in `extracted.jsonl`;
  142 of 143 manifest versions have text, and its two adjacent diff pairs are skipped.
- `downloads.jsonl` carries 2 extra versions (`2309.16319` v1/v2) that left the manifest
  after download; they are extracted/converted but never scrubbed or diffed.
- Diff pairs with `ratio < 0.30` were each inspected for this run: all 4 are genuine
  major rewrites (retitle/reframe `2307.04047`, restructures `2309.07915`/`2402.05525`,
  ICLR->ACL template resubmission `2411.19378`), not extraction inconsistencies.
