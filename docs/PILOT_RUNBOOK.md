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

## Contamination / recognition probe (run 2026-10-21)

Can the model identify the anonymized pilot papers FROM MEMORY? One call per selected
version sends the scrubbed text with `src/review/prompts/recognition_v1.md` and logs the
model's claimed title/authors/venue/year/decision plus a 0..1 confidence, graded against
the pilot manifest and the ICLR corpus (`src/review/run_recognition_probe.py` docstring
has the grading rules).

```sh
python3 src/review/run_recognition_probe.py --which anchor   # default: 50 calls, one anchor
                                                             # version per paper, ~$0.23
# --which all  = every scrubbed version (142 calls)
# --mock       = no network, deterministic fake guesses through the same parse/grade path
```

Temperature is pinned to 0 (factual probe). Append-only checkpoint
`data/processed/recognition_log.jsonl` keyed on `<arxiv_id>v<v>|<model>`; re-running
makes 0 new calls. Headline numbers go in `docs/FINDINGS.md`
("Contamination / recognition probe").

## E4 probes (length-matched known-direction edits; generated + mock-verified 2026-10-21)

Do scores track content or document length? Six deterministic variants of each probe
paper's anchor version (latest pre_submission/during_review version, else v1 — the
`analyze_scores.py` anchor rule): `base` (verbatim), `shuffle_paragraphs` /
`corrupt_numbers` / `typos` (quality DOWN, length within +-1%), `pad_filler` (+25..35%
content-free filler), `drop_appendix` (<= -10%; skipped when the text has no
`## Appendix`). The paper subset reuses `run_llm_review.select_papers`, so the 25 probe
papers are the deterministic stratified round-robin prefix of the main study's
selection. Generated 2026-10-21: 148 variant texts (25 papers x 6 - 2 `no appendix
heading` skips), `base.md` byte-identical to the anchor texts, regeneration into a
scratch dir reproduced all 148 files hash-identically.

### Generation (pure-local, ~20 s)

```sh
python3 src/process/make_probes.py   # -> data/processed/probes/<safe_id>/<variant>.md
                                     #    + log data/processed/probes_log.jsonl
```

### Scoring + analysis

```sh
# mock e2e first (no network, same parse/log path; point --log/--out-dir at a tmp dir)
python3 src/review/run_probe_review.py --mock --reps 2 \
  --log /tmp/e4mock/log.jsonl --out-dir /tmp/e4mock/reviews

# real: deepseek-reasoner (all defaults), 148 probes x 3 reps = 444 calls
python3 src/review/run_probe_review.py --reps 3

# real: gemini text-only, 148 x 2 = 296 calls (key setup: "Multimodal arm" above)
python3 src/review/run_probe_review.py \
  --model gemini-3.8-flash \
  --base-url https://generativelanguage.googleapis.com/v1beta/openai \
  --api-key-env GEMINI_API_KEY --max-tokens 8000 --reps 2

python3 src/analysis/analyze_probes.py   # -> results/e4_probes_summary.md
                                         #    (every model in the log, separately)
```

Scoring uses the exact main-study rubric (`src/review/prompts/review_v1.md`) through
`run_llm_review.py`'s request/parse/validate machinery; mock e2e verified 2026-10-21
(296/296 parsed_ok, re-run makes 0 calls, analysis report renders with ns matching the
probe counts).

### Approximate cost

Probe texts average ~14.8k prompt tokens/call (pad +31% and drop -40..-67% roughly
balance around the ~16k paper mean). deepseek-reasoner, 444 calls ~ 6.6M prompt tokens:
the script's printed formula (deepseek-chat list, $0.28/$0.42 per M) gives **~$2.5**;
deepseek-reasoner's own list price plus reasoning-token completions realistically lands
around **$5-10** — check the current price page first. gemini-3.8-flash, 296 calls ~
4.4M prompt tokens at flash-class list prices (~$0.3-0.5/M in, ~$2.5/M out): roughly
**$2-3**.

### Resume semantics

All three stages are independently resumable:

- `data/processed/probes_log.jsonl` — append-only, keyed `<arxiv_id>|<variant>`;
  re-running `make_probes.py` skips logged keys (a completed run writes nothing).
  Variants are pure functions of (anchor text, `--seed`); to rebuild a paper, delete its
  rows from the log AND its `data/processed/probes/<safe_id>/` dir, then re-run.
- `data/processed/probe_reviews_log.jsonl` — append-only, keyed
  `<arxiv_id>|<variant>|<model>|t<temperature>|r<rep>`: an interrupted run continues
  where it stopped, a completed run makes 0 calls (verified), and a new `--model` /
  `--temperature` is a fresh key grid, not an overwrite. Failed calls ARE checkpointed
  with `parsed_ok=false` — delete those rows to retry. Probes-log rows with
  `skipped: true` are never scored. Raw bodies go under
  `data/processed/reviews/<model>-probes/`, so probe raws never mix with main-study raws.
- `analyze_probes.py` — pure-local; OVERWRITES `results/e4_probes_summary.md` each run
  and contextualizes effects against the E0 true-placebo noise floor parsed from
  `results/e0_scores_summary.md`.

## Fresh score-free corpus + contamination split

### Contamination split (pure-local; no API calls)

Splits every model's version-effect MixedLM by the E3 recognition probe and reports,
per model: group descriptives, the `rating ~ pos * recog + (1|paper)` interaction,
per-group `rating ~ pos + (1|paper)` slopes, and an accept/reject AUC anchor.

```bash
python3 src/analysis/analyze_contamination_split.py
# -> results/contamination_split.md          (split: recognized, 38/12 papers)

python3 src/analysis/analyze_contamination_split.py \
  --split authors_any_correct --out results/contamination_split_authors.md
# -> results/contamination_split_authors.md  (split: authors-recalled, 18/32 papers)
```

Inputs: `data/processed/reviews_log.jsonl` (parsed_ok rows),
`data/processed/recognition_log.jsonl` (one row per paper: `arxiv_id`,
`recognized`, `authors_any_correct`, `title_correct`, `confidence`),
`data/processed/pilot_manifest.jsonl`. Papers missing from the recognition log are
reported as `unprobed` and excluded from the tests. Both reports are OVERWRITTEN on
each run. Verified 2026-10-22: on the authors_any_correct split the version effect
survives in the authors-not-recalled group for all five arms.

### Fresh score-free corpus (live arXiv; search + OAI, 1 req / 3 s)

Collects post-training-cutoff multi-version papers (submittedDate window defaults to
2026-07-01..2026-09-20) with NO human review scores by construction — supports
version-tracking analyses only, round-robin across `--categories`.

```bash
# smoke (verified 2026-10-22: 4 rows, one per category; NOTE --limit-queries caps
# search+OAI combined and discovery pages are NOT cached, so the budget must cover
# ~1 search page per category PLUS one OAI call per new paper; --limit-queries 3
# exhausts on discovery alone and lands 0 rows)
python3 src/collect/collect_fresh_arxiv.py --target 4 --limit-queries 9

# full collection (orchestrator): no query cap; ~4-20 search pages + 30 OAI calls
python3 src/collect/collect_fresh_arxiv.py --target 30
```

Outputs (append-only, checkpointed on `arxiv_id`; reruns resume and already-kept
papers count toward `--target`):

- `data/processed/fresh_manifest.jsonl` — pilot_manifest-shaped rows: `arxiv_id`,
  `openreview_id` (= arxiv_id), `iclr_year: null`, `title`, `scores: []`,
  `mean_score/n_reviews` empty, `decision: "unknown"`, `strata`, `n_versions`,
  `versions` (`v`, `date`, `stage: "unknown"`, ...; parsed from cached arXivRaw OAI
  XML under `data/raw/oai_xml/`), `chain_source`, `categories`, `fresh: true`.
- `data/processed/fresh_papers.jsonl` — iclr_papers-shaped rows keyed
  `openreview_id = arxiv_id` (`title`, `title_norm`, `authors`, `abstract`, empty
  score/decision fields), which keeps the scrubber's author-audit join working.

Downstream stages (`download_sources`, `extract_sources`, `latex_to_text`,
`scrub_and_audit`, review runner) consume these files unchanged.

## E2 pairwise comparison (mock-verified 2026-10-22)

For every adjacent version pair (v_i, v_{i+1}) of the same paper whose scrubbed texts
both exist, `run_pairwise_review.py` shows the two versions as two anonymized
submissions (A/B) in ONE prompt and asks which is the STRONGER submission
(`src/review/prompts/pairwise_v1.md`; verdict JSON: `better` A/B/tie, `margin`
slight/clear/decisive, `confidence` 1-5). `--orders both` runs each pair twice (early
as A, then swapped) so position bias is measurable and removable; the logged
`picked_version` maps the positional verdict back onto early/later/tie.
`analyze_pairwise.py` then reports later-pick rates by churn class (incl. the
true-placebo subset), position bias + order consistency, the pairwise-vs-absolute
cross-tab against `reviews_log.jsonl` on the same pairs, and margin calibration.

Mock end-to-end verified 2026-10-22: 50 papers -> 91 pairs with both texts (93
adjacent pairs minus the 2 touching withdrawn `2301.12935` v2) x 2 orders = 182
calls, 182/182 parsed_ok, re-run makes 0 new calls; analysis ns all reconcile (91
distinct pairs, churn ns 50+20+112 = 182, placebo 34 + cosmetic-with-figs 16 = 50
cosmetic, section-3 join finds absolute scores for all 91 pairs, 0 excluded).

### Real runs

```sh
# mock e2e first (no network, same parse/log path; point --log/--out-dir at a tmp dir)
python3 src/review/run_pairwise_review.py --mock --orders both --reps 1 \
  --log /tmp/e2mock/log.jsonl --out-dir /tmp/e2mock/reviews

# real: deepseek-chat, 91 pairs x 2 orders = 182 calls
python3 src/review/run_pairwise_review.py --orders both --reps 1

# real: deepseek-reasoner, same 182 calls (raise the completion cap for reasoning tokens)
python3 src/review/run_pairwise_review.py --model deepseek-reasoner --max-tokens 16000 \
  --orders both --reps 1

python3 src/analysis/analyze_pairwise.py   # -> results/e2_pairwise_summary.md
                                           #    (every model in the log, separately)

# optional fresh-corpus variants (30 papers, 35 pairs x 2 orders = 70 calls/arm)
python3 src/review/run_pairwise_review.py --orders both --reps 1 \
  --manifest data/processed/fresh_manifest.jsonl --texts data/processed/fresh_texts/ \
  --log data/processed/fresh_pairwise_log.jsonl --out-dir data/processed/reviews_fresh/
python3 src/analysis/analyze_pairwise.py \
  --log data/processed/fresh_pairwise_log.jsonl \
  --diffs data/processed/fresh_version_diffs.jsonl \
  --manifest data/processed/fresh_manifest.jsonl \
  --reviews-log data/processed/fresh_reviews_log.jsonl \
  --out results/e2_fresh_pairwise_summary.md
```

Key setup is the same as E0 (`DEEPSEEK_API_KEY` in the repo-root `.env`). `--limit N`
caps calls per run for a cautious first ramp-up. Defaults: log
`data/processed/pairwise_log.jsonl`, raw bodies
`data/processed/reviews/<model>-pairwise/` (never mixed with E0/E4 raws).

### Approximate cost

Each call carries TWO full papers, so prompts are roughly twice E0's: ~35k input
tokens/call (mock char/4 estimate: mean ~33k, median ~28k, max ~65k; per-paper texts
hard-capped at `--max-chars-each` 190000 so two always fit in the 128K context).
Main corpus, per arm: 182 calls ~ 6.4M prompt tokens. deepseek-chat at list price
($0.28/M in, $0.42/M out, completions capped at 2k) = **~$2**. deepseek-reasoner at
its own list price plus reasoning-token completions (16k cap) realistically lands
around **$5-10** — check the current price page first. Fresh corpus is 70 calls/arm
(~38% of the main corpus): **~$0.8** chat / **~$2-4** reasoner. List prices ignore
DeepSeek's prompt-cache discount.

### Resume semantics

`pairwise_log.jsonl` is an append-only checkpoint keyed on
`key = "<arxiv_id>|v<v_from>v<v_to>|<order>|<model>|t<temperature>|r<rep>"`: re-running
skips keys already logged, an interrupted run continues where it stopped, and a
completed run makes 0 calls (mock-verified). The order token is part of the key, so an
interrupted `--orders both` run resumes per (pair, order), and a new `--model` /
`--temperature` is a fresh key grid, not an overwrite. `--orders random` draws one
seeded order per pair that is stable across runs/resumes (same `--seed`). Failed calls
(HTTP error after retries, or unparseable after 1 parse re-request) ARE checkpointed
with `parsed_ok=false` — delete those rows from the log to retry. Pairs with a missing
text file are skipped and never checkpointed. `analyze_pairwise.py` is pure-local,
keeps only `parsed_ok` rows, tolerates a torn final line from a concurrent append, and
OVERWRITES `--out` each run.

## ACL 2025 corpus (all of ACL 2025: long + short + Findings)

Scales the version-tracking design to every paper published at ACL 2025
(`2025.acl-long`, `2025.acl-short`, `2025.findings-acl`). The Anthology lists only
published papers, so every row is `decision: "accepted"` with NO review scores —
like the fresh corpus, this supports version-tracking analyses only. ACL rows carry
`year`/`iclr_year` **0** on purpose: 0 has no entry in `iclr_timeline.TIMELINES`, so
every version stage is labeled `unknown` (ACL papers must never pick up an ICLR
timeline). Two side effects of year 0 are cosmetic, not errors: `match_arxiv` prints
every match as `implausible dates` (its window becomes [-3, 1]), and
`strata.spans_stages` is always `false`.

Slice verified live 2026-10-22: collection parsed **3,086** papers from the three
volume `.bib` exports (1,602 long + 97 short + 1,387 findings); a 10-paper live match
slice matched 10/10 (9 exact, 1 fuzzy; 7/10 with `latest_version >= 2`) and the
re-run made 0 queries; one OAI chain + one synthesized v1-only chain produced an
`acl_manifest.jsonl` row whose schema is exactly `fresh_manifest.jsonl`'s keys plus
`venue` and `acl_id`, all stages `unknown`.

### Full run (end to end)

```sh
# 1) Paper lists: 3 Anthology volume fetches, cached under data/raw/anthology/
#    (re-run = 0 network calls). 3,086 rows -> data/processed/acl_papers.jsonl.
python3 src/collect/collect_acl.py

# 2) arXiv title matching: 3,086 queries at 1 req / 3 s -> ~2-3 h.
#    (--years 0 is explicit; the file contains only year-0 rows anyway.)
python3 src/collect/match_arxiv.py \
  --in data/processed/acl_papers.jsonl --out data/processed/acl_matches.jsonl \
  --decisions accepted --years 0

# 3) Version chains: one OAI call per match with latest_version >= 2 (v1-only rows
#    are synthesized without a call) -> ~1-1.5 h at the slice's 70% multi-version rate.
python3 src/collect/fetch_versions.py \
  --matches data/processed/acl_matches.jsonl --out data/processed/acl_chains.jsonl

# 4) Full score-free manifest: keeps every chain with n_versions >= 2, re-attaches
#    venue/acl_id. Pure-local; OVERWRITES data/processed/acl_manifest.jsonl.
python3 src/collect/build_acl_manifest.py

# 5) LaTeX sources, every version of every manifest paper (1 req / 3 s) -> ~3 h.
python3 src/collect/download_sources.py \
  --manifest data/processed/acl_manifest.jsonl --dest data/raw/sources/ \
  --log data/processed/acl_downloads.jsonl

# 6) Processing, acl_* logs/dirs mirroring the fresh corpus (extracted trees share
#    data/raw/extracted/, keyed by arxiv_id, exactly like the fresh run).
python3 src/process/extract_sources.py \
  --downloads data/processed/acl_downloads.jsonl \
  --manifest data/processed/acl_manifest.jsonl \
  --dest data/raw/extracted/ --log data/processed/acl_extracted.jsonl
python3 src/process/latex_to_text.py \
  --extracted data/processed/acl_extracted.jsonl \
  --out-dir data/processed/acl_texts_raw/ --log data/processed/acl_latex_log.jsonl
python3 src/process/scrub_and_audit.py \
  --in-dir data/processed/acl_texts_raw/ --manifest data/processed/acl_manifest.jsonl \
  --papers data/processed/acl_papers.jsonl --out-dir data/processed/acl_texts/ \
  --log data/processed/acl_scrub_log.jsonl
python3 src/process/diff_versions.py \
  --texts data/processed/acl_texts/ --manifest data/processed/acl_manifest.jsonl \
  --out data/processed/acl_version_diffs.jsonl

# 7) Review run: deepseek-chat, K=3 reps, ~10,000 calls (ramp up with --limit first,
#    e.g. --limit 50, inspect the log, then re-run without it to resume).
python3 src/review/run_llm_review.py --reps 3 --workers 4 \
  --manifest data/processed/acl_manifest.jsonl --texts data/processed/acl_texts/ \
  --log data/processed/acl_reviews_log.jsonl --out-dir data/processed/reviews_acl/

# 8) Analysis: pure-local, OVERWRITES both outputs.
python3 src/analysis/analyze_scores.py \
  --log data/processed/acl_reviews_log.jsonl \
  --manifest data/processed/acl_manifest.jsonl \
  --diffs data/processed/acl_version_diffs.jsonl \
  --scrub data/processed/acl_scrub_log.jsonl \
  --out results/acl_scores_summary.md --per-paper-out results/acl_per_paper.jsonl
```

### Approximate cost

Scaling the pilot's measured rate (~17.4k prompt tokens/call, completions capped at
2k): ~10,000 calls ~ 175M prompt tokens -> **~$50-55** at the deepseek-chat list price
($0.28/M in, $0.42/M out), before DeepSeek's prompt-cache discount. Check the current
price page before launching.

### Resume semantics

Every network stage is safe to re-run and an interrupted run continues where it
stopped. `collect_acl` caches the raw volume responses under `data/raw/anthology/`
and appends to `acl_papers.jsonl` checkpointed on `openreview_id` (= the anthology
ID), so a re-run makes 0 network calls. `match_arxiv` checkpoints on
`openreview_id` (verified on the slice: re-run = 0 queries; `--retry-none` re-queries
only `match_method: "none"` rows), `fetch_versions` on `arxiv_id` (plus the XML cache
under `data/raw/oai_xml/`), `download_sources` on `<arxiv_id>v<v>`.
`extract_sources` / `latex_to_text` / `scrub_and_audit` are append-only on
`key = "<arxiv_id>v<v>"` (surgical reset: see the pilot processing section).
`build_acl_manifest` and `diff_versions` are pure-local and rebuild from scratch, so
re-run them after any upstream change. `run_llm_review` checkpoints
`acl_reviews_log.jsonl` on `key = "<arxiv_id>v<v>|<model>|t<temperature>|r<rep>"`;
failed calls ARE checkpointed with `parsed_ok=false` (delete those rows to retry),
missing text files are skipped and never checkpointed. `analyze_scores` is
pure-local and OVERWRITES its outputs each run.

## E5 resubmission pairs (LLM deltas vs REAL human score deltas)

Ground-truth anchor: ICLR papers rejected in year N and resubmitted in year N+1..3
with human review scores in BOTH rounds. Per round the manifest carries the latest
arXiv version visible by that year's decision cutoff (stages `round1`/`round2`), so
the LLM's round2-minus-round1 score delta can be correlated against the TRUE human
delta (`human_delta_raw` only within one realized score scale — 2022-2025 are
mutually compatible — and `human_delta_z`, per-year z-scored, across all years).

Verified live 2026-10-23 on the real data: 36,113 parquet rows -> **184** candidate
pairs (round 1 rejected, gap 1-3y) -> 184 identity-passed -> **150** score-complete
(decided round 2, >=1 score both rounds) -> **29** arXiv-matched against the 463k-row
bulk harvest -> **7** version-selected manifest rows (exclusions: 121 no_bulk_match,
34 decision2_withdrawn, 15 no version before the round-1 decision, 7 unchanged
between rounds). Coverage is concentrated in 2024->2025 (24/50 matched, 48%) because
the bulk harvest only holds OAI datestamps >= 2024-06-01; growing n beyond this means
re-harvesting older arXiv records, not loosening the matcher.

### Full run (end to end)

```sh
# 1) Pair finder: pure-local (parquet + bulk harvest, 0 network calls).
#    OVERWRITES e5_pairs.jsonl (every candidate with exclude_reason),
#    e5_manifest.jsonl (fresh/ACL schema + e5 extras) and e5_papers.jsonl.
python3 src/collect/find_resubmissions.py

# 2) LaTeX sources: 7 papers x 2 versions = 14 downloads (1 req / 3 s, ~1 min).
python3 src/collect/download_sources.py \
  --manifest data/processed/e5_manifest.jsonl --dest data/raw/sources/ \
  --log data/processed/e5_downloads.jsonl

# 3) Processing, e5_* logs/dirs mirroring the ACL corpus (extracted trees share
#    data/raw/extracted/, keyed by arxiv_id+v, so reruns dedupe across corpora).
python3 src/process/extract_sources.py \
  --downloads data/processed/e5_downloads.jsonl \
  --manifest data/processed/e5_manifest.jsonl \
  --dest data/raw/extracted/ --log data/processed/e5_extracted.jsonl
python3 src/process/latex_to_text.py \
  --extracted data/processed/e5_extracted.jsonl \
  --out-dir data/processed/e5_texts_raw/ --log data/processed/e5_latex_log.jsonl
python3 src/process/scrub_and_audit.py \
  --in-dir data/processed/e5_texts_raw/ --manifest data/processed/e5_manifest.jsonl \
  --papers data/processed/e5_papers.jsonl --out-dir data/processed/e5_texts/ \
  --log data/processed/e5_scrub_log.jsonl

# 4) Review runs: BOTH models into the SAME log (the checkpoint key includes the
#    model, so the runs coexist and resume independently). K=3 reps x 14 texts
#    = 42 calls per model, 84 total.
python3 src/review/run_llm_review.py --reps 3 --workers 4 \
  --model deepseek-chat \
  --manifest data/processed/e5_manifest.jsonl --texts data/processed/e5_texts/ \
  --log data/processed/e5_reviews_log.jsonl --out-dir data/processed/reviews_e5/
python3 src/review/run_llm_review.py --reps 3 --workers 4 \
  --model deepseek-reasoner \
  --manifest data/processed/e5_manifest.jsonl --texts data/processed/e5_texts/ \
  --log data/processed/e5_reviews_log.jsonl --out-dir data/processed/reviews_e5/

# 5) Analysis: pure-local, one section per model in the log; OVERWRITES outputs.
#    (Defaults already point at the e5_* files and the ACL noise-floor summary.)
python3 src/analysis/analyze_e5.py --json-out results/e5_stats.json
```

### Approximate cost

84 calls at the pilot's measured ~17.4k prompt tokens/call ≈ 1.5M prompt tokens +
≤0.2M completion tokens (capped at 2k/call). The deepseek-chat half is ~$0.25 at
list price ($0.28/M in, $0.42/M out); deepseek-reasoner is pricier per token and
adds reasoning tokens, but at n=42 calls the whole run stays **~$1-2 total** before
the prompt-cache discount. Check the current price page before launching.

### Interpretation guardrails

`results/e5_summary.md` writes its own caveats; the two that matter most: (a) n=7 —
treat every statistic as directional until the pair base grows (the binding
constraint is bulk-harvest coverage, see funnel above); (b) match-coverage bias —
only papers touched on arXiv after 2024-06 can enter, over-representing actively
revised work. `human_delta` ground truth itself is noisy (different committees
across years), which attenuates correlations toward 0.
