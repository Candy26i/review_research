# Qwen3-32B test-split log — QC (2026-11-06, 4090)

Scope: `data/processed/ai_edits_qwen_log.jsonl` as committed in a2266fd
(2,199 rows; Qwen/Qwen3-32B-AWQ, vLLM 0.10.2, thinking on, `--max-tokens 6000`).
Three probes (accepted rows, rejections, code review); every medium/high
finding re-derived from scratch by a second agent. None was refuted; numbers
below are the re-derived ones. Comparators: deepseek ai rows (1,973) and human
rows (13,666) of `attribution_v0/test.jsonl`, claude rows of `attribution_xgen`.

## Coverage and yield

- Tasks cover all 1,973 v0 deepseek test combos (sampler replay: 2,817/2,818
  papers, `pre2022_reconcile.json`). 0 think-tag leaks, 0 refusals, all ok rows
  re-pass every generator gate.
- Gate yield 1,524/2,199 (69.3%): 348 math-markup asymmetry, 322 length
  window, 5 unchanged. After the builder's filters 1,415 survive (64.3%): the
  0.30 similarity floor removes 109 more (expand 43, rewrite 42, compress 19).
- Per-style gate yield: compress 80.7%, **expand 22.4%**, grammar 81.9%,
  polish 77.2%, rewrite 82.1%. Post-builder style mix
  344/52/378/325/316 vs deepseek 404/306/431/405/427.
- The log also holds 5 train-split rows from the smoke test (harmless; the
  builder assigns split by hash).

## Censoring bias (matters for the which-generator matrix)

1. **Math-heavy paragraphs are censored.** Qwen adds `$` to plain-unicode math
   in 15.8% of calls (haiku ~1.5%, deepseek ~0.03%); the reject rate rises from
   2.9% (no math symbols in old_text) to 33.5% (10+ symbols). Qwen survivors
   are less mathy than other generators' survivors on the same combos.
2. **Expand is close to unusable.** 245 of 424 expand outputs exceed the 1.7
   ceiling (median ratio 2.43); Qwen adds a roughly fixed amount of text, so
   only long paragraphs survive (Spearman(ratio, old_words) = −0.77). Effective
   expand yield after the builder: 52/424 (12.3%).
3. Censoring-only separability (style + old_text features, paper-grouped HGB)
   Qwen vs deepseek ≈ 0.59–0.60 AUC. Removable by intersecting combos that are
   ok for every generator (1,358 qwen∩deepseek) or by stratified resampling.

## Surface fingerprints in accepted rows (label leaks for which-generator)

| cue (introduced in new_text) | Qwen | deepseek | human | note |
|---|---|---|---|---|
| curly apostrophe U+2019 | 12.0% | 0.05% | 0.20% | claude 0%; expand 38.9%; survives `clean()` |
| em dash U+2014 | 10.4% | 4.4% | 0.44% | always unspaced; expand 49.5% |
| unicode sub/superscripts (x_1→x₁) | 4.1% | 1.3% | 0% | 16.7% of rows whose old has _ or ^ |
| `<cit.>` count decreased | 4.0% | 0.15% | — | 42 rows lose all citations; expand 21% |
| citation marker form altered (`<cit>`, `[cit.]`, `[1]`) | ~2% | 0.05% | — | 6 rows invent numeric citations |
| HTML `<sub>/<sup>`, markdown `**`/`*x*`, `\cite{}`/`\ref{}` | ~30 rows | ~0 | — | the math gate only looks at `$` |
| "This paragraph…/The figure…" openers on caption inputs | 0.9% | 0.3% | 0 | shared artifact, Qwen ~3× |
| we/our removed entirely | 12.8% | 3.8% | 3.0% | plus acronym expansion 6.8% vs 0.9%, some wrong |

Edit magnitude is also a strong generator cue and is real behaviour, not an
artifact: old~new word-sequence similarity on matched pairs, Qwen vs deepseek —
expand 0.32 vs 0.82, rewrite 0.42 vs 0.60, compress 0.53 vs 0.76; a
length/similarity-only classifier still reaches ≈0.895 AUC on the matched
intersection.

## Recommendations (not applied — builder/gate design decisions)

- Builder: normalise typographic quotes/apostrophes (and consider dashes and
  NFKC for sub/superscripts) symmetrically for all classes, or report an
  ablation; otherwise U+2019 alone flags one Qwen row in eight at ~100%
  precision.
- Builder or generator gate: reject rows whose `<cit.>`/`<ref>` counts differ
  or that introduce `<cit>`, `[cit.]`, `\cite`, `<sub>`, `**`.
- For the multiclass confusion matrix: build on the all-generators-ok
  intersection, or at least report a style+length+similarity baseline next to
  it, and treat Qwen expand as a separate caveat (or drop expand).
- Generator: log the rejected output text (`new_text` is null on rejected
  rows, so the 675 rejections cannot be audited); strip `Caption:`/bullet lead
  markers before sending.

## Code review of a2266fd (fixed in the follow-up commit)

- `latex_to_text.py`: a worker that died while idle left the pool broken for
  every later document (silent `detex_fallback`); submit is now retried on a
  fresh pool. `--shard` is validated. Not triggered in the committed run
  (2,848/2,848 keys, 4 fallbacks, same as expected).
- `write_text(newline=…)` needs Python ≥ 3.10; replaced by `open(...,
  newline=…)` so the Mac is unaffected on any version.
- `reconcile_pre2022_texts.py`: `unverifiable` (28 papers) mixes 26 papers
  with no classifiable picks and 2 train papers (2102.06311 and one more) that
  do get sampled but have no v0 ai row to check against (≤ 8 rows in the full
  run). `match` verifies only picks deepseek accepted (10,260 of 11,268).
