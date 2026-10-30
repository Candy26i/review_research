# Attribution v0 — post-hoc adversarial QC of the DeBERTa baselines (2026-10-30, 4090)

Three independent probes (cheap-cue ceiling, error/band analysis, code-level
leakage) over `deberta` and `deberta_nfkc`, every non-clean finding re-derived
by a second agent from scratch. Nothing was refuted. Numbers below are the
re-derived ones.

## Headline survives

- test AUC 0.9895 [0.9873, 0.9917] (nfkc 0.9893); predictions align 1:1 with
  test.jsonl; paper split clean (0 paper / old_text / new_text overlap across
  splits); val-only threshold; paper-level bootstrap is correct; model input is
  only `old [SEP] new`; truncation symmetric (AUC on non-truncated rows 0.9894).
- Ratio/length are not the shortcut: ratio-stratified AUC (0.02-wide bins)
  0.971; within-class Spearman(score, ratio) has *opposite* signs (human +0.40,
  ai −0.60), i.e. the model reads the edit, not its size.
- Neither side alone is predictive: char-ngram LR on new_text 0.749, on
  old_text 0.533. The signal is in the (old, new) pair.
- No paper fingerprint: within-paper-centered score still AUC 0.947; paired
  same-old_text probe (645 human/ai pairs sharing identical old_text)
  P(score_ai > score_human) = 0.995.
- Unicode tells (em dash, straight quotes, en dash) are class-skewed but
  low-coverage; dropping every row with an introduced tell: 0.9895 → 0.9884.
  Consistent with the nfkc ablation (−0.0002).

## Corrections to how the result should be reported

1. **The +0.24 margin is against a linear baseline and overstates the gap.**
   The 9 surface features with HistGradientBoosting instead of LR give 0.836;
   adding 5 count deltas (word/char ratio, Δcit, Δref, Δdigits) gives
   **0.903 [0.895, 0.911]** (stable across seeds / hyper-params; RF 0.891;
   grouped-CV on train 0.904). Honest statement: DeBERTa 0.990 vs a fair
   cheap-feature ceiling ≈ 0.90, **margin ≈ +0.09**, not +0.24. Stacking
   DeBERTa + the 14 cheap features in paper-grouped CV = 0.989 vs 0.988 alone,
   so DeBERTa already subsumes the cheap signal. Recommend adding a tree
   baseline to `train_attribution.py`'s surface baseline.
2. **Prompt-induced confounds (generator side), real but not load-bearing:**
   - `<cit.>`/`<ref>` count changes in 21.1% of human rows vs 0.40% of ai
     rows (prompts say "keep every citation marker"); digit-count changes
     22.8% vs 3.8%. DeBERTa learned it (human FP rate 0.000 on cit/ref-changed
     rows vs 0.015 elsewhere, in every ratio band). Headline on the
     cit/ref-unchanged subset: 0.9869; also dropping digit-changed: 0.9850.
   - word-ratio is hard-clipped to [0.45, 1.70] for ai by the generator's
     rejection window; 6.7% of human rows fall outside. Rule alone AUC 0.533;
     in-window-only AUC 0.9887.
   - Restricting humans to ai-lookalikes (in-window AND cit/ref preserved,
     n_h = 10,342) vs all ai: **0.9864**. That is the number to quote as the
     confound-robust floor.
3. **Builder asymmetry (bug):** `--min-ratio 0.30` is applied to the human
   class only; 44 train / 12 test ai rows have ratio < 0.30 (one is a refusal
   that slipped the generator's error filter: "I'm not able to apply this
   edit…"). Negligible (AUC without them 0.9897) but should be fixed in
   `build_attribution_dataset.py` and the refusal filter in `make_ai_edits.py`
   widened beyond `stop_reason == 'refusal'`.
4. **False positives cluster in a few papers:** at threshold 3.02, 164/13,666
   human FPs, 42 of them from 5 papers (2111.12580, 1909.09505, 2206.11190,
   2010.10233, 2011.08225) whose whole v2 revision reads as professional
   copyedit (median human score near the boundary). Top-15 highest-scored
   human rows are genuine copyedits ("doesn't"→"does not", "firstly"→"first",
   contraction removal), not data errors. Open question for the write-up: is
   "edited by a copyeditor/Grammarly" distinguishable from "edited by an LLM",
   and are all pre-2022 v2s truly LLM-free (2206.11190 is June 2022).
5. **Hard band (ratio ≥ 0.92) is grammar-dominated:** 246/369 ai rows are
   style=grammar; per kind para 0.962, caption 0.992, bullet 0.890 (n_ai 16).
   Lowest-scored ai rows are 1–3-token grammar edits. All ai rows are one
   generator (deepseek-chat), so "ai" here means "deepseek-chat edit".

## Recommended before this goes in a paper

- Add HGB surface baseline + report margin vs it; report the lookalike-floor
  AUC (0.986) alongside the headline.
- Fix the asymmetric min-ratio filter and the refusal filter; rebuild.
- Second generator (another model family) for a held-out-generator test.
- A human-copyedit control (e.g. Grammarly-style or professional-edit pairs)
  to separate "LLM" from "polished".
