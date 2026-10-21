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
  - **Reasoning is the hinge**: same DeepSeek model, reasoning off → τ=-0.14; reasoning on → τ=+0.49. Gemini (reasons by default): τ=+0.57
  - **Verbosity bias absorbs most of it**: after controlling for document length, only Gemini-text keeps an independent version effect (+0.14/step, p=0.001)
  - **Longitudinal ≠ cross-sectional**: all arms see papers improving yet separate accepted/rejected at only AUC ~0.7 (humans 0.88)
  - **Contamination probe**: 70% exact-title recall despite scrubbing; zero evidence of decision memorization
- [ ] Complete the 2×2 (chat @ 50 papers, Gemini thinking-off); E4 length-matched causal probes; post-cutoff contamination split; E5 human-delta anchoring
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
