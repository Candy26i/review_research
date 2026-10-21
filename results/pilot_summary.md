# P0 pilot dataset — collection summary (2026-10-20)

Pipeline run: ICLR 2024–2025, 400 candidates stratified by year × decision (seed 42).
Commands as in [docs/PILOT_RUNBOOK.md](../docs/PILOT_RUNBOOK.md); all steps checkpointed.

## Funnel

| Step | Count |
|---|---|
| ICLR papers loaded (2019–2025) | 34,612 |
| Candidates queried against arXiv (2024–25, stratified) | 400 |
| Matched to arXiv (all exact title matches, 0 fuzzy) | 250 (62.5%) |
| Matched with ≥2 arXiv versions | 170 eligible |
| **Pilot manifest (stratified, seed-stable)** | **50** |
| Version instances in manifest | 143 |
| Source archives downloaded (1.8 GB, all gzip) | 142/143 |

One version (2301.12935v2) has no downloadable source on arXiv (HTTP 404, likely
PDF-only submission) — the text-extraction stage needs a PDF fallback. Three of 400
match queries and one OAI fetch failed on transient network errors; a re-run will
retry them (checkpointed), which may grow the eligible pool slightly beyond 170.

## Key rates (useful for scaling to P1)

- **Match rate by decision:** accepted 76% (2024) / 83% (2025); rejected 37% (2024) / 50% (2025).
  Rejected papers are the bottleneck — plan candidate pools accordingly (~4× oversampling
  of rejected papers to hit a target).
- **Multi-version rate among matched:** accepted 73%, rejected 63%.
- Overall yield: 400 candidates → 170 eligible (42.5%).

## Manifest composition (n=50)

- Strata: accepted/2v 13, accepted/3+v 13, rejected/2v 12, rejected/3+v 12 (26 accepted / 24 rejected).
- Years: 2024 ×21, 2025 ×29. Versions per paper: 2 ×25, 3 ×14, 4 ×8, 6 ×2, 7 ×1.
- All 50 papers span ≥2 review stages across their versions.
- Version stages: pre_submission 31, during_review 46, post_decision 66.
- Adjacent-version stage transitions: during_review→post_decision 37 (revision straddling
  the decision — the highest-value pairs), pre_submission→during_review 13,
  pre_submission→post_decision 9, within-stage 34.
- Human ground truth: mean reviewer score 6.52 (accepted) vs 5.42 (rejected).

## Files (data/ is gitignored; the frozen manifest is committed deliberately)

- `data/processed/pilot_manifest.jsonl` — the 50-paper pilot sample (committed: arXiv
  matching depends on live API state, so the sample is frozen here for reproducibility).
- `data/processed/{iclr_papers,arxiv_matches,version_chains,downloads}.jsonl` — regenerate
  via the runbook.
- `data/raw/sources/<arxiv_id>/arXiv-<id>v<n>.gz` — per-version source archives (not in git).

## Text extraction & anonymization (done 2026-10-20)

- **142/142 obtainable versions converted** (all via pylatexenc; the withdrawn
  2301.12935v2 has neither source nor PDF on arXiv and is excluded — its paper keeps v1/v3).
  Corpus: **1.34 M words ≈ 2.29 M tokens**, median 59 k chars/version.
- Pipeline: safe tar extraction (`filter='data'`, 500 MB cap) → `\input`-flattening →
  LaTeX-level anonymization (author blocks, acknowledgments, camera-ready toggles,
  template keyword lines) → zero-arg macro expansion (62 k expansions; method names like
  "SAPO" would otherwise render as "()" — definitions are also collected from `.sty` files)
  → `\texttt`/`\verb` unwrapping (pylatexenc drops their content: inline code, tool names)
  → markdown conversion → text-level scrub (emails/URLs incl. scheme-less `github.com/org`
  links, venue/camera-ready/reviewer-thanks lines, header-region affiliation footnotes,
  layout residue) → leak audit.
- **Camera-ready metadata:** 97/144 versions carry a final-copy toggle in their source
  (`\iclrfinalcopy` etc.) — recorded in `latex_log.jsonl`, scrubbed from the text.
- **Leak audit of the final corpus:** 0 emails, 0 raw URLs/repo orgs, 0 venue-acceptance
  lines, 0 template keyword lines, 0 header affiliations; 11 docs have author-surname
  string hits (manually inspected by QA agents: common-word false positives and third-person
  self-citations); institution names in running prose (e.g. "labelers from KAIST") are
  counted in `leak_flags.institution_hits` but not removed — known limitation, to be
  handled by the E4 recognition probe.
- **Churn (91 adjacent pairs):** cosmetic 25 / minor 10 / major 56; mean similarity 0.78.
  By stage transition: pre_submission→during_review has the heaviest edits (mean churn
  0.46, n=13); during_review→post_decision averages 0.19 (n=37). Four pairs with
  similarity < 0.30 were each hand-verified as genuine rewrites (retitle/template change),
  not extraction bugs.
- Three QA inspectors swept all 142 files against author lists plus deep-read a sample;
  all systemic findings they raised (ICML template keyword leak, scheme-less repo URLs,
  affiliation footnotes, a stale-draft main-tex mis-pick that made one paper's versions
  identical, empty method-name macros) were fixed and the corpus fully regenerated.

## Next (E0 scoring pilot)

1. Review rubric + prompt (ICLR-style 1–10 + confidence, JSON output).
2. K-repeat scoring on a handful of papers to estimate variance → power analysis for the
   main study. Needs a model/API-key decision.
