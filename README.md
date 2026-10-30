# review_research

**Are LLM reviewers reliable? Evidence from arXiv version histories.**

arXiv keeps every revision of a paper (v1, v2, v3 …). If LLM review scores are meaningful,
they should track this within-paper quality ordering. We test whether they do — and whether
the signal survives controls for length, cosmetic edits, camera-ready leakage, and
training-data contamination.

- 📋 Full experiment design: [docs/EXPERIMENT_DESIGN.md](docs/EXPERIMENT_DESIGN.md)
- 🛠️ Pilot data-collection runbook: [docs/PILOT_RUNBOOK.md](docs/PILOT_RUNBOOK.md)
- 🎯 Hypotheses: scores correlate positively with version (H1) but are far from monotonic (H2);
  test–retest noise rivals the revision signal (H3); superficial edits move scores as much as
  substantive ones (H4)
- 🗂️ Planned benchmark: scrubbed, diff-stratified arXiv version chains linked to ICLR/OpenReview
  human scores and decisions

## Status

- [x] Initial idea (slides)
- [x] Literature pass & experiment design
- [x] P0 pilot data: 50-paper stratified manifest + 142 per-version sources ([summary](results/pilot_summary.md))
- [x] P0 scoring pilot, 3 arms ([results](results/e0_scores_summary.md)): version-tracking is **model-dependent** — DeepSeek-chat τ=-0.14 (noise>signal) vs Gemini-flash τ=+0.64; adding figures maximizes reliability (ICC 0.79, best placebo behavior)
- [x] E1 main run, 4 arms × 50 papers × K=3 (1,359 reviews) — headline findings in [docs/FINDINGS.md](docs/FINDINGS.md):
  - **Completed 2×2** ({DeepSeek, Gemini} × {think off, on}, 50 papers each): both factors additive, model family larger — Gemini-no-think (+0.20/step) beats DeepSeek-think (+0.15); after length control only the Gemini arms stay significant
  - **Verbosity bias absorbs most of it**: after controlling for document length, only Gemini-text keeps an independent version effect (+0.14/step, p=0.001)
  - **Longitudinal ≠ cross-sectional**: all arms see papers improving yet separate accepted/rejected at only AUC ~0.7 (humans 0.88)
  - **Contamination probe**: for 36% of papers the model names a true author never present in the input (real memorization); zero evidence of decision memorization
- [x] E4 length-matched edit probes, both arms: **no verbosity bias** (adding 31% fluff *tanks* scores: −0.9/−2.2; cutting 40% length changes nothing — the length effect was content mediation), but **no arm detects substantive damage** (falsified result numbers pass unnoticed; paragraph shuffling even raises the reasoner's score)
- [x] 2×2 completed (deepseek-chat @ 50 papers; Gemini `reasoning_effort: none` arm) — 2,537 reviews total across six arms
- [x] Contamination closed by three lines: split analysis (effect survives on unmemorized papers, interactions negative), and **zero-recognition replication** on a fresh post-cutoff corpus (0/29 recognized; gemini +0.23/step p=0.016, reasoner +0.19/step p=0.0097)
- [x] E2 pairwise comparison: same pairs, same models — later-version pick rate jumps from 38.5%→**68.7%** (chat) and 50.5%→**76.9%** (reasoner, 92.9% on major revisions); LLM reviewers are comparators, not scorers
- [x] DeepSeek fill-ins: fresh×chat null (−0.02/step); E4×chat flagged **length-sensitive** — converging with the observational length-control result
- [x] **ACL 2025 scale replication**: 1,379 papers / 3,629 versions / 10,860 reviews ($17) — +0.10/step (p<1e-4) matching ICLR; survives length control at scale; dose-response cosmetic +0.05 < minor +0.10 < major +0.20
- [x] E5 human-delta anchoring (25 resubmission pairs via the Kaggle snapshot): **honest null** — LLM deltas do not track human cross-round deltas (agreement 32-36%), separating "perceives revision" (robust) from "matches human re-scoring" (no evidence)
- [x] Q2 frozen-splice pilot built ([runbook](docs/SPLICE_PILOT.md)): 822 papers / 5 conditions (true, micro, ident, v1self, xdoc), builder + portable tagger + paired-bootstrap analysis, adversarially reviewed; position-only baseline AUC 0.67 — awaiting 4090 training runs
- [x] **Frontier arm (Claude Opus 5.5, $56)**: version effect replicates (+0.14/step p<1e-4); most reliable scorer yet (ICC 0.87, 84% cosmetic ties) and closest to humans cross-sectionally (accept AUC 0.80 vs cheap arms' 0.57-0.71, humans 0.88); one security paper refused (cyber) in all reps
- [x] **E4 × frontier (~$55)**: substance-blindness is a capability gradient — Claude is the first arm where every length-preserved degradation is individually significant (corrupt_numbers −0.43, shuffle −0.23) and filler over-punishment shrinks (−0.49 vs −2.16), yet the pre-registered content verdict still doesn't fire and falsified numbers are still misattributed to presentation, not soundness
- [ ] Write-up
- [x] Collection pipeline: ICLR↔arXiv matching, version chains, stratified manifest, source downloads
- [ ] P1 dataset: version chains + edit strata + scrubbing
- [ ] P2 main runs (E1 monotonicity, E2 pairwise, E3 reliability)
- [ ] P3 controls (placebo pairs, causal probes, contamination)
- [ ] P4 human ground-truth anchoring (rebuttal & resubmission pairs)
- [ ] P5 write-up + benchmark release

## Repo layout (planned)

```
docs/        experiment design, notes
src/collect/   arXiv version chains, OpenReview matching
src/process/   text extraction, version diffing, scrubbing
src/review/    LLM review runners + prompts
src/analysis/  stats & figures
data/        (gitignored) raw & processed data
results/     scores, model outputs
```
