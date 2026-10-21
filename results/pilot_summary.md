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

## E0 scoring pilot — three arms (done 2026-10-21)

10 papers × 29 versions × 3 reps per arm; same scrubbed texts, same rubric (ICLR scale
{1,3,5,6,8,10}); full tables in [e0_scores_summary.md](e0_scores_summary.md).

| metric | deepseek-chat (text) | gemini-3.8-flash (text) | gemini-3.8-flash (+figs) |
|---|---|---|---|
| mean Kendall τ (version↑score) | **-0.14** | **+0.64** | +0.44 (3 papers all-tied) |
| major-churn pairs ranked ↑ / ↓ | 50% / 50% | 87.5% / 12.5% | 62.5% / **0%** |
| cosmetic placebo unchanged | 33% | 33% | **67%** |
| test-retest SD / ICC(2,1) | 0.9 / 0.26 | 0.5 / 0.65 | **0.4 / 0.79** |
| signal-to-noise (version vs retest) | 0.68 | 1.36 | **1.54** |
| human-score Spearman (anchor) | 0.74 | 0.72 | 0.64 |
| accept/reject AUC (rating) | 0.79 | 0.83 | 0.77 |
| rating mean (human 6.0) | 4.2 harsh | 5.4 | 5.5 |

**Takeaways (pilot-sized, n=10):**
1. **Reviewer reliability is model-dependent, dramatically.** DeepSeek-chat cannot resolve
   revision-sized differences (τ≈0, noise>signal); Gemini flash tracks versions well
   (τ=0.64, 87.5% of major revisions ranked higher).
2. **Figures buy reliability, not discrimination.** The +figs arm is the most *consistent*
   (ICC 0.79, best placebo behavior, zero wrong-direction on major revisions) but ties more —
   it moves less, and when it moves it's right.
3. **Thinking confound to resolve:** gemini-3.8-flash reasons by default, deepseek-chat does
   not — "reasoning enables revision-tracking" is an alternative reading; deepseek-reasoner
   on the same papers would separate model-family from reasoning.
4. DeepSeek is harsh + compressed (mean 4.2, never uses 1/8/10); Gemini sits near the human
   mean. All arms discriminate accepted/rejected papers far above chance.

## E1 full-scale results — Gemini two arms, all 50 papers (2026-10-21)

426 (text) + 420 (+figs) parsed reviews, K=3, 142 versions; DeepSeek remains pilot-scale
(10 papers). Full tables in [e0_scores_summary.md](e0_scores_summary.md).

| metric (50 papers) | gemini text | gemini +figs | deepseek-chat (10 papers) |
|---|---|---|---|
| mixed-effects: rating points per revision step | **+0.28** (CI +0.21..+0.36, p<1e-4) | **+0.22** (p<1e-4) | +0.01 (p=0.90) |
| Page trend test p | 0.039 | 0.024 | 0.78 |
| mean Kendall τ (finite) / all-tied papers | 0.57 / 18 | 0.48 / 15 | -0.14 / 1 |
| mean per-paper Pearson (≥3 versions) | 0.43 | 0.42 | -0.13 |
| test-retest SD / ICC(2,1) | 0.4 / 0.81 | 0.4 / 0.81 | 0.9 / 0.26 |
| cosmetic placebo unchanged | 48% | 52% | 33% |
| human anchor Pearson / accept-reject AUC | 0.16 / 0.70 | 0.28 / 0.71 | 0.52 / 0.79 |

**What changed from the 10-paper pilot:**
1. **The version effect is now statistically solid** for Gemini: ~+0.25 rating points per
   revision step (p<1e-4, three test families agree). DeepSeek's null is equally solid.
2. **The pilot's figure advantage largely washed out**: at scale both Gemini arms have
   ICC 0.81; figures still give the best cosmetic-placebo behavior (52% unchanged) and a
   slightly smaller effect size — figures make the reviewer marginally more conservative.
3. **New at scale: between-paper validity is weak.** Human-score correlation dropped from
   0.7 (pilot) to 0.16–0.28, accept/reject AUC from 0.83 to ~0.70. Combined with the strong
   within-paper signal, the story inverts the naive expectation: **Gemini can see a paper
   improving, but is mediocre at telling good papers from bad ones** — and DeepSeek is the
   reverse (decent anchor correlation at pilot scale, blind to revisions). Also ~1/3 of
   papers get identical Gemini scores across all versions (all-tied).
4. Scale compression persists in all arms (nobody uses 1/8/10 much; humans do).

## Next

1. deepseek-reasoner arm (thinking hypothesis) on the same 50 papers.
2. E4: synthetic causal probes (inject known-direction edits), contamination probe,
   true-placebo subset (17 pairs with no text AND no figure changes).
3. E5 human-ground-truth anchoring (rebuttal pairs / resubmissions) and write-up.
