# attribution_v1 (final) — 6-generator edit-attribution dataset

Built 2026-11-06 on the Mac from seven generator logs + the pre-2022 human
revision pairs. Replaces the preliminary build of 621ad18 (same out-dir).

## Composition

| split | rows | human | ai | papers |
|---|---|---|---|---|
| train | 86,076 | 55,497 | 30,579 | 2,274 |
| test | 22,381 | 13,533 | 8,848 | 550 |

Paper overlap train∩test: **0** (split by `seed64("split|{arxiv_id}") % 5`).

Train classes (model, rows): deepseek-chat 7,963 · claude-haiku-5-5 7,768 ·
gpt-5.4-mini-2026-03-17 7,046 · Qwen/Qwen3-32B-AWQ 5,763 · gemini-3.8-flash
2,039 (partial: credits ran out at 23%; resumable) · human 55,497.

Test-only probe models (never trained on): claude-sonnet-5-5 500 ·
gpt-5.5-2026-04-23 545 · deepseek-reasoner 525. Predicted-class histograms on
probes = family-vs-checkpoint fingerprint evidence.

## New symmetric controls in this build (vs the preliminary one)

Trigger: PC's adversarial QC of the qwen test log
(`results/attribution_v0/qwen_test_qc.md`) + a Mac-side rescan of all
generators. Same precedent as the newline and $-math fixes: pipeline
artifacts are normalized/filtered symmetrically, behaviour is kept.

1. **Typography normalization in `clean()`** (NFKC + explicit map of curly
   quotes U+2018/19/1A/1B → `'`, U+201C–201F → `"`, en/em dash → `-`).
   The human side comes through detex (straight quotes); generators emit
   smart punctuation: curly-apostrophe introduction was qwen 12.0%,
   gpt-5.5 6.5%, gpt-5.4-mini 4.9% vs human 0.15% — a codepoint-level
   which-generator leak at ~100% precision. NFKC also folds the unicode
   sub/superscripts haiku introduced (1.2%). Post-build scan: **0 rows** in
   either split contain any of these codepoints. (The trainer's
   `--fold-unicode` ablation is now a no-op on this dataset.)
2. **Markup-introduction filter** (symmetric, both classes): rows whose
   new_text introduces `\cite{`, `\ref{`, `<sub>`, `<sup>`, `**`, `[cit.]`,
   or bare `<cit>` absent from old_text are dropped — these are reactions to
   OUR detex placeholder format, not real-world style (gpt-5.4-mini wrote
   bare `<cit>` in 3.3% of rows; qwen produced `[cit.]`/`\cite{}` forms).
   Dropped 483 human + 564 ai rows — both classes lose rows, so the filter
   itself creates no inverse leak.

3. **Round 2** (12-agent adversarial workflow on the round-1 build; every
   major finding independently re-derived, none refuted):
   - TYPO_MAP extended with the sibling codepoints round 1 missed: U+2212
     minus (ai-only, 649 introduced rows, 100%-precision tell — and NFKC
     *manufactures* it from U+207B superscript minus), U+2032 prime,
     U+2044 fraction slash (manufactured by NFKC from vulgar fractions),
     U+2010/2012/2015 dashes.
   - **Backtick fold** `` ` `` → `'` and `''` → `"`: detex passes LaTeX
     `` `...' `` quoting through as literal backticks — a HUMAN-side quote
     artifact (introduced 0.59% human vs 0.035% ai, 97.7%-precision human
     tell). Round 1's "symmetric" normalization had only erased the AI
     side's smart quotes.
   - MARKUP_INTRO extended with `\(`, `\[`, `\begin{` — the `$`-gate's
     LaTeX siblings (gpt-5.4-mini introduced `\(` in 2.9% of rows, 0 human);
     citation-mangle regex broadened (`<cit` w/o `.>`, `(cit.)`, `[cit]`).
   - Prompt-echo strip: leading `Figure/Table/Fig./Algorithm caption:` on
     new_text (sonnet wrote it on 26.7% of its caption rows; 0 human).
   - Qwen's LaTeX `~` non-breaking space before `<cit.>`/`<ref>` folded to a
     space (qwen-only, 100%-precision tell).
   - Human `ratio` recomputed on cleaned text so its provenance matches the
     ai class (it is a surface-baseline feature).
   Post-rebuild scan: all of the above at **0 occurrences** in both splits.

Other drops: 368 no-ops after cleaning, 51 math-markup asymmetries,
41 dupes, 24 contested-label, 1,059 model-contested (same (old,new) from two
generators — undecidable for which-generator).

## Known residual cues (kept deliberately — report, don't hide)

- **Citation-count deltas are behaviour, not filtered**: humans change
  `<cit.>` counts in 12.7% of rows (real revisions add/remove citations);
  AI rates are low (gpt-5.4-mini 3.3%, qwen ~4%, others <1%). Filtering
  both directions would cost 12.7% of the human class; filtering AI-only
  would create an inverse leak. The surface-feature baseline prices this
  cue into the honest margin instead.
- **Qwen censoring bias** (from PC QC): qwen rejects math-heavy paragraphs
  (33.5% reject at 10+ math symbols) and its `expand` style mostly exceeds
  the length ceiling (255 surviving rows total) — qwen survivors are less
  mathy/longer-sourced than other generators' on the same combos.
  Censoring-only separability ≈ 0.59–0.60 AUC qwen-vs-deepseek. For the
  confusion matrix, also report the all-generators-ok intersection slice
  and the surface baseline next to the DeBERTa numbers.
- **Human-sided math-unicode introductions** (review-confirmed, kept): detex
  renders human-added math as unicode (∈, Greek, ≤, ∑) while generators add
  math as LaTeX — and the `$`/`\(` gates drop those AI rows. Net effect:
  "introduces math-unicode → human" decides ~3.6% of test at ~93% precision.
  This is partly real behaviour (humans add new equations between versions;
  style-constrained rewrites don't add content) — same family as the
  citation-count residual. Filtering it would cost ~3.5% of the human class
  and systematically exclude math-adding edits; priced by the surface
  baseline instead. Candidate symmetric gate for v1.1.
- **Qwen drops `<cit.>` markers at ~10× other generators** (1.9% of its rows;
  15.6% under expand) — behaviour toward our placeholder, kept + reported.
- **Lexical tells are signal, not leaks** (for calibration): qwen introduces
  "enhance"/"significant(ly)" in ~3.4–3.7% of rows (next AI ≤0.3%);
  deepseek-reasoner introduces a semicolon into semicolon-free text in 13.8%
  of rows. These are exactly what which-generator attribution should learn.
- **`**` removal** (present in old, dropped in new) is not filtered — only
  introductions are. Presence is ~2% of rows across all classes. Note the
  MARKUP_INTRO `**` entry drops some legitimate human footnote-marker edits
  (label-blind, both classes).
- Edit magnitude per style differs by generator (qwen rewrites much harder:
  expand sim 0.32 vs deepseek 0.82) — real behaviour, legitimate signal,
  but the style+length+similarity baseline must be reported alongside.

## Reporting rules (pre-registered, carried over from xgen)

Headline numbers are per-generator, not pooled; always next to the
surface-HGB baseline refit on the same slice; AUC for generalization claims;
probe models are reported as predicted-class histograms + binary-collapse
AUC vs human, never as "accuracy" (they have no own class).
