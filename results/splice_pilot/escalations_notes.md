# Splice pilot — pre-registered escalations (4090, 2026-10-29/30)

Headline runs (ctx1 / ctx0 / nosym, seed 0) were DEAD at pilot scale (see
`summary.md`). This file covers the two pre-registered escalations, the extra
runs they forced, and what can and cannot be concluded. Per-run bootstrap
tables: `summary_escalations.md`. Every number below that is not in a
`summary*.md` is reproduced by `src/analysis/splice_pilot_checks.py`
(output: `checks.md`). Raw logs: `train_*.log`.

## Status against the pre-registered rule

**DEAD at pilot scale; escalation 1 not run as registered; escalation 2 run
but inconclusive under the current training recipe.** All 13 runs have
delta point < 0 and delta CI-high < 0.05 (best: ctx0 +0.022, ctx1_len512
+0.013). The ctx0 gate (delta ≥ 0.05) is never approached (ctx0 −0.004,
ctx0_half −0.019). So no run would have triggered SCALE even taken alone.
What the escalations could not settle is whether the null is a *scale*
artifact (see escalation 2).

## All runs

| run | eval epoch | val by epoch | true | v1self | xdoc | micro | ident | logit sd | ρ(score, words) | delta [95% CI] |
|---|---|---|---|---|---|---|---|---|---|---|
| ctx1 (seed 0) | 1/2 | .629 / .638 | 0.651 | 0.637 | 0.652 | 0.570 | 0.428 | 0.53 | +0.38 | −0.017 [−0.042, +0.009] |
| ctx1_seed1 | 1/2 | .628 / .643 | 0.650 | 0.637 | 0.656 | 0.585 | 0.425 | 0.66 | +0.45 | −0.018 [−0.042, +0.007] |
| ctx1_seed2 | 1/2 | .583 / .596 | 0.592 | 0.590 | 0.576 | 0.545 | 0.420 | **0.0006** | +0.39 | −0.076 [−0.104, −0.047] |
| ctx1_len512 | 1/2 | .638 / .663 | 0.653 | 0.637 | 0.655 | 0.578 | 0.431 | 0.74 | +0.41 | −0.014 [−0.040, +0.013] |
| ctx0 (seed 0) | 1/2 | .615 / .668 | 0.664 | 0.639 | 0.647 | 0.596 | 0.429 | 0.69 | +0.62 | −0.004 [−0.030, +0.022] |
| ctx0_half | 1/2 | .576 / .633 | 0.649 | 0.633 | 0.621 | 0.601 | 0.425 | 0.66 | +0.70 | −0.019 [−0.044, +0.008] |
| ctx1_half (2 ep) | **0**/2 | .621 / .592 | 0.612 | 0.606 | 0.599 | 0.592 | 0.419 | **0.016** | +0.63 | −0.056 [−0.079, −0.031] |
| ctx1_half_ep4 | 1/4 | .573 / .577 / .500 / .459 | 0.537 | 0.551 | 0.507 | 0.573 | 0.414 | **0.034** | +0.75 | −0.131 [−0.160, −0.101] |
| ctx1_quarter (2 ep) | **0**/2 | .582 / .575 | 0.539 | 0.552 | 0.511 | 0.571 | 0.408 | **0.025** | +0.76 | −0.129 [−0.159, −0.100] |
| ctx1_quarter_ep8 | **0**/8 | .598 / .578 / .580 / .560 / .586 / .584 / .583 / .583 | 0.608 | 0.614 | 0.594 | 0.586 | 0.400 | **0.036** | +0.65 | −0.060 [−0.086, −0.034] |
| nosym (seed 0) | 1/2 | .443 / .479 | 0.505 | 0.488 | 0.496 | 0.509 | 0.486 | **0.0000** | +0.02 | −0.162 [−0.206, −0.120] |
| nosym_seed1 | **0**/2 | .581 / .484 | 0.603 | 0.614 | 0.616 | 0.547 | 0.444 | **0.020** | +0.30 | −0.065 [−0.094, −0.034] |
| nosym_seed2_lr1e5 | 1/2 | .571 / .632 | 0.632 | 0.632 | 0.629 | 0.569 | 0.412 | 0.65 | +0.50 | −0.036 [−0.059, −0.011] |

All runs: DeBERTa-v3-base, batch 16, max-len 384 (512 for len512), BCE with
pos_weight ≈ 33, bf16 autocast, lr 2e-5 except where named, best-val-epoch
checkpoint. "eval epoch" = the checkpoint actually scored (best val).
"logit sd" = population std of the raw test logits over all 64,429 rows.
Slot baselines (LR on position / position+surface, trained on train/true):
true 0.668 / 0.697, v1self 0.668 / 0.669, xdoc 0.659 / 0.682, micro 0.607 /
0.602, ident 0.436 / 0.434.

## Two training regimes (post-hoc, but the gap is bimodal)

Logit sd is bimodal: six runs at 0.53–0.74, seven at ≤ 0.036, nothing in
between. The low-sd runs are **not** degenerate rankers except two: nosym
seed 0 (54 distinct values spanning 2e-7 — a constant logit, AUC is float
noise) and ctx1_seed2 (range 0.006). The other five (all four sub-100% ctx1
runs and nosym_seed1) still rank all 64k rows (≈15.5k distinct scores, same
per-paper AUC spread as healthy runs) but with tiny output gain and a
ranking that is mostly paragraph length: Spearman(score, words) = +0.63 to
+0.76 vs +0.38 to +0.45 for the healthy ctx1 runs. Call them "low-gain /
length-ranking", not collapsed. In every one of them the best val epoch was
epoch 0 or 1 and later epochs *lowered* val AUC (ep4: .577 → .500 → .459).

The regime is confounded with everything that was varied — all 4/4 sub-100%
ctx1 runs, 2/3 nosym runs and 1/5 full-data unmasked runs are low-gain — so
"lr 2e-5 is unstable" is a hypothesis, not a finding: the only lr 1e-5 run
(nosym_seed2) is healthy, but it also differs in seed and ablation. An
lr-isolating pair (ctx1 seed 0 @ 1e-5; nosym seed 2 @ 2e-5) is the cheap next
test. Excluding low-gain runs is conservative for a DEAD call only.

## Verdict is DEAD on every healthy run, and the text signal is cheap-surface

- Six healthy runs: AUC_true 0.632–0.664, each ≈ v1self ≈ xdoc. The
  regime-B "foreign text" ceiling (xdoc) sits at its own slot baseline
  (0.652 vs pos-LR 0.659), so even wholly foreign paragraphs are not detected
  beyond position. No intro/related ≫ method/experiment novelty signature.
- There *is* a small, consistent within-model text signal: paired
  true − v1self = +0.014 [+0.000, +0.026] (ctx1), +0.013 [+0.001, +0.026]
  (seed1), +0.017 [+0.003, +0.030] (len512), +0.025 [+0.006, +0.043] (ctx0),
  +0.016 [−0.008, +0.040] (ctx0_half). The delta term that binds is the
  position LR, which uses metadata (para_idx, n_paras, section, kind, words)
  DeBERTa never sees, so a fair question is whether the model adds anything
  *on top of* those features.
- **Fusion check** (paper-grouped 5-fold CV on test/true, LR): position
  features alone 0.670; + any healthy model's logit 0.677–0.682; position +
  the seven cheap surface stats (digit/punct/caps fraction, type-token ratio,
  mean word length, cit/ref density) 0.695; + model logit **0.694–0.695**.
  The DeBERTa signal is fully absorbed by seven hand-made surface features.
  That is the strongest single argument against scaling: even as "one channel
  of an evidence-fusion detector" it currently contributes nothing.

## Escalation 1 — more context (`--max-len 512`)

Not a real test on this dataset: with CTX_WORDS = 50, ~2.9% of ctx1 inputs
(~3.3% of positives) exceed 384 tokens and ~0.9% exceed 512 (median 175,
p95 342). Result as expected: 0.653 vs 0.651, paired −0.003 [−0.016, +0.010].
**The registered escalation — rebuild with `CTX_WORDS = 100` — needs the raw
data and has to run on the Mac** (`build_splice_pilot.py`), then retrain here
(~20 min/run).

## Escalation 2 — data-size curve (`--train-frac`)

Added `--train-frac` (paper-level, deterministic sha256 on paper id; 0.25 ⊂
0.5; val/test untouched). Counts: 100% = 595 train papers / 48,890 rows /
1,436 pos; 50% = 311 / 25,472 / 751; 25% = 145 / 11,280 / 352; val fixed at
65 papers / 5,535 rows / 159 pos. (The runbook's "696 train papers" is stale:
train.jsonl holds 660 cond=true papers = 595 train + 65 val.)

Raw ctx1 curve (2 epochs): 25% 0.539 → 50% 0.612 → 100% 0.651; paired
ctx1 − ctx1_half = +0.039 [+0.017, +0.058], ctx1_half − ctx1_quarter =
+0.074 [+0.052, +0.097]. ctx0: 50% 0.649 → 100% 0.664, paired +0.015
[+0.001, +0.028]. **Nominally, AUC_true is still rising with data in both
contexts**, which by the pre-registered wording means the null cannot be
called a construct failure yet. But the curve is not clean:

1. The "step-matched" reruns (50% × 4 ep, 25% × 8 ep) did not match steps:
   best-val checkpointing picked epoch 1 (3,184 steps) and epoch 0 (705
   steps) respectively, because later epochs lowered val AUC. They scored
   0.537 and 0.608. So "more updates" was never actually tested, and the
   fixed-epoch confound (25% = 1,410 total steps vs 6,112) is unresolved.
2. Every sub-100% ctx1 run landed in the low-gain regime, while ctx0_half
   did not. Within ctx1 the curve therefore mixes "less data" with "regime
   change at this recipe"; the one clean pair (ctx0 vs ctx0_half, both
   healthy) shows a small rise of 0.015 on a single seed, with no ctx0 seed
   replicate to compare it against (ctx1 healthy seeds differ by 0.001;
   ctx1 seed 2 is low-gain at 0.592).
3. Even if the rise is real, it is a rise toward the *slot baseline*
   (0.668), not past it, and the fusion check says the extra AUC is
   cheap-surface. "Rising with data" and "contributes nothing beyond seven
   surface stats" are both true at pilot scale.

To make escalation 2 decisive: a stable recipe (see above), fixed-step or
last-epoch evaluation, 2–3 seeds per point, and report the fusion AUC per
point rather than raw AUC_true.

## nosym (digits masked)

Seed 0 was a constant-logit run and proves nothing. Seed 1 is low-gain
(0.603). Seed 2 at lr 1e-5 is healthy: true 0.632, v1self 0.632, xdoc 0.629;
paired true − v1self +0.000 [−0.013, +0.014]; ctx1 − nosym_seed2 = +0.019
[+0.005, +0.032]. That comparison is confounded (lr and seed differ, n = 1),
so the digit channel cannot be quantified from it; the defensible statement
is that the slot baseline's digit-fraction feature adds ~0 (pos+surface
without it: 0.696 vs 0.697 per the review), so the numeric channel is weak
either way. Runbook cheat-sheet updated to require a health check before
reading a nosym run.

## micro (runbook-mandated check, since micro > 0.5)

micro is 0.57–0.60 in every healthy run, but the slot baselines score micro
at 0.607 / 0.602 — above every model — so micro > 0.5 is entirely slot
edit-proneness. The top-20 ctx1-scored micro positives (checks.md §E) are
genuine light copyedits: word swaps ("also"→"moreover", "raking"→"ranking",
"Continuous"→"Compressed"), dropped URLs/bold markers, a citation added —
not extraction-drift artifacts. Scores are 0.55–0.60 logits, barely above the
run's median. Nothing to chase here.

## ident < 0.5 is explained (not a bug)

Ident documents are byte-for-byte the v1self documents (all 475 positives
and 12,304 negatives identical in text/prev/next/section/kind/words); ident
scores equal v1self scores at the same slot to float noise, and
AUC(ident labels, v1self scores) reproduces 0.4286 (ctx0) / 0.4275 (ctx1)
exactly. The ident positives are a random sample of the *unchanged-in-v2,
byte-equal* paragraphs, and that filter selects a biased subset: later in
the document (mean rel. pos 0.58 vs 0.50 for all slots / 0.39 for true
positives), shorter (63 words vs 74 / 86), more captions and appendix. The
slot baselines score test/ident at 0.436 / 0.434, i.e. the metadata alone
reproduces the sub-0.5 value. So ident < 0.5 is expected whenever the
model carries slot edit-proneness and the byte-equal filter is biased; only
ident ≫ 0.5 would flag a code-path leak. `docs/SPLICE_PILOT.md` corrected.

## What remains untested here

- CTX_WORDS = 100 rebuild (Mac) → retrain ctx1/ctx0 here.
- lr-isolating pair (ctx1 seed 0 @ lr 1e-5; nosym seed 2 @ lr 2e-5).
- Clean data curve: stable recipe, fixed-step eval, 2–3 seeds/point, fusion
  AUC per point. ctx0 seed replicates (the ctx0 gate currently rests on n=1).
