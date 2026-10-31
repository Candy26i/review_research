# attribution_xgen — cross-generator held-out test set (build notes)

2026-10-31, Mac. Purpose: the v0 detector was trained with a single generator
(deepseek-chat); this set tests it on generators it has never seen.

## Recipe

Claude rewrites of **exactly the same (paper, block, style) combos deepseek
rewrote on the test split** — the task sampler is seeded per paper and
independent of model, verified 2,192/2,192 overlap before the run. Same style
prompts, same acceptance gates, same builder, same cleaning. Only the
generator changes.

- `claude-haiku-5-5`: all 2,192 test-split combos
- `claude-sonnet-5-5`: seeded-random 600 of the same combos (generator-strength
  gradient)
- `--max-tokens 8000` (claude 5 thinking spends the max_tokens budget; at the
  old 2000 a truncation could slip the length gates — pre-run adversarial
  review finding, plus a new `stop_reason != 'end_turn'` error gate)

## Generation stats

| model | ok | length-window | unchanged | refusal | stop_reason |
|---|---|---|---|---|---|
| claude-haiku-5-5 | 1,959 | 209 | 23 | 1 | 0 |
| claude-sonnet-5-5 | 540 | 43 | 16 | 1 | 0 |

Rejection rate ~10.6% / 10.0% vs deepseek's 9.35% on the same combos —
comparable censoring. The haiku refusal ("The paragraph you've provided is
actually a figure caption…") was caught by the Claude-tuned hedge regex added
in the pre-run review (0 false hits on 69,802 human / 10,274 deepseek rows).
Tokens: 778k in / 888k out, ≈ $7.

## Dataset

`data/processed/attribution_xgen/test.jsonl`: 16,101 rows = 13,667 human +
2,434 ai (haiku 1,906 / sonnet 528) over 550 papers. train.jsonl is human-only
(ai rows exist only for test-split papers) — do not train on this dir.

- contested-label rows: **0** (v0 deepseek had 3) → the human side is one row
  LARGER than v0.2's test (13,667 vs 13,666): v0 dropped one human row whose
  revision deepseek reproduced verbatim. Negligible, but documented.
- magnitude shift: Claude edits are heavier — ratio bands ai [0.3,0.6)/[0.6,0.92)/[0.92,1.01)
  = 34%/57%/9% vs deepseek 15%/67%/19% (human 14%/43%/43%). Interpret via the
  band-sliced AUCs and the within-set refit HGB, not the pooled number alone.

## Reporting rules (pre-registered before seeing any eval number)

1. Headline = **per-generator AUC** (haiku on full coverage; sonnet separately).
   The pooled test AUC double-counts 600 shared combos across generators.
2. Quote margins against BOTH `surface_hgb` (transferred tree: apples-to-apples
   with the transferred DeBERTa) and `surface_hgb_refit_cv` (paper-grouped CV
   refit on this set: the honest cheap ceiling here).
3. F1 / balanced-acc are deepseek-val-threshold transfer numbers at a shifted
   class ratio — report AUC-family numbers for the generalization claim.
