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
- [ ] P0 scoring pilot: rubric + variance estimate
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
