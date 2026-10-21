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

## E0 scoring (mock-verified 2026-10-20)

Scores every selected paper x version x rep with DeepSeek (`deepseek-chat`, OpenAI-compatible
`/chat/completions`, `response_format: json_object`), then analyzes whether the LLM rating
tracks arXiv versions. Mock end-to-end run (8 papers, 3 reps, 72 calls) verified: 100%
parsed_ok, text SHAs match `scrub_log.jsonl` (real scrubbed texts are what gets sent),
withdrawn `2301.12935` v2 skipped without error, re-run makes 0 new calls.

### Key setup

Put the API key in the repo-root `.env` (gitignored) — never commit or print it:

```
DEEPSEEK_API_KEY=sk-...
```

(`export DEEPSEEK_API_KEY=...` in the shell also works; the env var wins over `.env`.)

### Real runs

```sh
# pilot: 10 papers (round-robin across strata, deterministic, superset-stable)
python3 src/review/run_llm_review.py --sample-papers 10 --reps 3
python3 src/analysis/analyze_scores.py

# full: all 50 manifest papers (142 versions x 3 reps = 426 calls)
python3 src/review/run_llm_review.py --reps 3
python3 src/analysis/analyze_scores.py
```

Defaults: log `data/processed/reviews_log.jsonl`, raw bodies
`data/processed/reviews/<model>/`, report `results/e0_scores_summary.md`, per-paper rows
`results/e0_per_paper.jsonl`. `--limit N` caps calls per run for a cautious first ramp-up;
`--mock` exercises the whole pipeline with zero network calls.

### Approximate cost

```
cost ~= prompt_tokens/1e6 * $0.28 + completion_tokens/1e6 * $0.42   (deepseek-chat list price)
```

Prompt tokens per call ~= paper `est_tokens` (from `scrub_log.jsonl`) + ~1.2k prompt
overhead; completions are capped at 2k tokens (~250-600 typical). Estimated from
`scrub_log.jsonl`: pilot (10 papers, 3 reps) = 87 calls, ~1.6M prompt tokens, **~$0.5**;
full (50 papers, 3 reps) = 426 calls, ~7.4M prompt tokens, **~$2.2**. List price, ignoring
DeepSeek's cache discount (repeated prefixes across reps can come back cheaper as
`prompt_cache_hit_tokens`).

### Resume semantics

`reviews_log.jsonl` is an append-only checkpoint keyed on
`key = "<arxiv_id>v<v>|<model>|t<temperature>|r<rep>"`: re-running skips keys already
logged, an interrupted run continues where it stopped, and a completed run makes 0 calls.
Changing `--model` or `--temperature` creates new keys (a fresh grid), not overwrites.
Failed calls (HTTP error after retries, or unparseable after 1 parse re-request) ARE
checkpointed with `parsed_ok=false`; to retry them, delete those rows from the log and
re-run. Missing text files are skipped and never checkpointed. `analyze_scores.py` is
pure-local, reads only `parsed_ok` rows, and OVERWRITES both outputs each run.

## Multimodal arm (Gemini)

Anonymized two-arm pilot: the same scrubbed text, with and without the paper's own
figures, scored by Gemini through its OpenAI-compatible endpoint (verified live
2026-10-20 on 2408.06291v1: both arms parsed_ok).

### Key setup

Add to the repo-root `.env` (never commit it):

```
GEMINI_API_KEY=<your key>
```

Model note: `gemini-2.5-flash` returns HTTP 404 for newer API keys ("no longer
available to new users"); `gemini-3.8-flash` is in this key's `/models` list and is
what the live smoke used. If a command 404s, list models and substitute:
`GET https://generativelanguage.googleapis.com/v1beta/openai/models`.

### 1. Figure collection (one-time per paper set, pure-local)

```bash
# 10 pilot papers (the same deterministic round-robin selection as --sample-papers 10)
python3 src/process/collect_figures.py \
  --papers 2412.12089,2412.04139,2408.06291,2301.12935,2412.05276,2412.07759,2402.05525,2310.02953,2305.14122,2411.15216
# full corpus: drop --papers
```

Outputs `data/processed/figures/{safe_id}/v{v}/fig_NN.png` (RGB, longest side 1024 px,
document order, deduped, capped at 20) and the checkpoint log
`data/processed/figures_log.jsonl` (key `<arxiv_id>v<v>`; resumable). Figures whose
filename contains an identity tell (logo/author/affiliation/orcid/signature/headshot/
photo_) are skipped and logged — pilot-10 result: 30 versions, 423 PNGs, 0 filtered,
0 errors (2301.12935v2 has no LaTeX tree and gets a clean reason row).

### 2. Two-arm pilot (10 papers, 3 reps each)

```bash
# arm A: gemini, text-only (keys: ...|gemini-3.8-flash|t1.0|r<rep>)
python3 src/review/run_llm_review.py --sample-papers 10 --reps 3 \
  --model gemini-3.8-flash \
  --base-url https://generativelanguage.googleapis.com/v1beta/openai \
  --api-key-env GEMINI_API_KEY

# arm B: gemini + figures (keys: ...|gemini-3.8-flash|t1.0|img|r<rep>; log label gets '+figs')
python3 src/review/run_llm_review.py --sample-papers 10 --reps 3 \
  --model gemini-3.8-flash \
  --base-url https://generativelanguage.googleapis.com/v1beta/openai \
  --api-key-env GEMINI_API_KEY \
  --images-dir data/processed/figures/

python3 src/analysis/analyze_scores.py   # reports each model label separately
```

### Key-format note (checkpoint compatibility)

Text-only keys are byte-identical to the historical format
`<arxiv_id>v<v>|<model>|t<temperature>|r<rep>` — all 87 existing DeepSeek rows still
checkpoint-skip (verified). Multimodal tasks insert an `|img|` segment:
`<arxiv_id>v<v>|<model>|t<temperature>|img|r<rep>`, so the two arms never collide and
each resumes independently. A version with no figure dir runs text-only under the
text-only key even with `--images-dir` set.

### Approximate cost (pilot, both arms)

Live smoke (2408.06291v1, gemini-3.8-flash): text-only 15,773 prompt tokens; +figs
(3 images) 19,030 — ~1,086 image tokens per 1024-px figure. Pilot scale: 87 calls/arm;
text arm ~1.6M prompt tokens, +figs arm adds ~1.2M image tokens (~1,150 attached
images after the 12 MB/request base64 budget); completions ~0.07M/arm. At flash-class
list prices (order $0.3-0.5/M input, $2.5/M output) that is roughly **$1-2 for both
arms combined** — check the current price page before the full 50-paper run
(~5x pilot volume).
