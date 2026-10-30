# E4: synthetic edit probes - content sensitivity vs length sensitivity

- review log: `data/processed/probe_reviews_log.jsonl` - 1599 rows, 444 parsed_ok rating rows used (model filter: claude-opus-5-5)
- thresholds (fixed upfront): content-sensitive iff mean degradation delta < -0.30 with 95% CI excluding 0; length-sensitive iff pad_filler delta > +0.15 or drop_appendix delta < -0.15 with 95% CI excluding 0

## Probe construction context (from the probes log)

| variant | n built | n skipped | mean length ratio vs base |
|---|---|---|---|
| base | 25 | 0 | 1.00 (n=25) |
| shuffle_paragraphs | 25 | 0 | 1.00 (n=25) |
| corrupt_numbers | 25 | 0 | 1.00 (n=25) |
| typos | 25 | 0 | 1.00 (n=25) |
| pad_filler | 25 | 0 | 1.31 (n=25) |
| drop_appendix | 23 | 2 (no appendix heading: 2) | 0.60 (n=23) |

## Model: claude-opus-5-5

- papers with a scored base: 25 (mean base rating 5.17) | reps per (paper, variant) cell: 3-3

### 1. Per-variant paired effects (variant mean minus the same paper's base mean, across reps)

| variant | n papers | mean delta | 95% CI | delta<0 | delta==0 | delta>0 | mean \|delta\| |
|---|---|---|---|---|---|---|---|
| shuffle_paragraphs | 25 | -0.23 | [-0.42, -0.04] | 28.0% | 72.0% | 0.0% | 0.23 |
| corrupt_numbers | 25 | -0.43 | [-0.72, -0.13] | 44.0% | 52.0% | 4.0% | 0.45 |
| typos | 25 | -0.20 | [-0.40, -0.00] | 24.0% | 76.0% | 0.0% | 0.20 |
| pad_filler | 25 | -0.49 | [-0.78, -0.20] | 52.0% | 44.0% | 4.0% | 0.52 |
| drop_appendix | 23 | +0.03 | [-0.25, +0.31] | 21.7% | 65.2% | 13.0% | 0.32 |

Subscore deltas (same pairing; n in parentheses = papers where both sides report that subscore):

| variant | d soundness | d presentation | d contribution | largest mover | expected | right subscore? |
|---|---|---|---|---|---|---|
| shuffle_paragraphs | -0.01 (n=25) | -0.81 (n=25) | -0.00 (n=25) | presentation | - | - |
| corrupt_numbers | -0.33 (n=25) | -0.65 (n=25) | +0.00 (n=25) | presentation | soundness | NO |
| typos | -0.01 (n=25) | -0.55 (n=25) | -0.00 (n=25) | presentation | presentation | YES |
| pad_filler | +0.07 (n=25) | -0.89 (n=25) | +0.01 (n=25) | presentation | - | - |
| drop_appendix | +0.09 (n=23) | +0.07 (n=23) | -0.01 (n=23) | soundness | - | - |
- typos -> presentation should move most: **YES**
- corrupt_numbers -> soundness should move most: **NO**

### 2. Mixed model: rating ~ C(variant, base reference) + (1|paper), rep-level rows

- n rep-level rows: 444 | papers: 25 | variants: 5 + base

| variant (vs base) | coef | 95% CI | p |
|---|---|---|---|
| corrupt_numbers | -0.43 | [-0.60, -0.25] | 0.0000 |
| drop_appendix | +0.02 | [-0.16, +0.20] | 0.8005 |
| pad_filler | -0.49 | [-0.67, -0.32] | 0.0000 |
| shuffle_paragraphs | -0.23 | [-0.40, -0.05] | 0.0108 |
| typos | -0.20 | [-0.37, -0.03] | 0.0245 |

### 3. The 2x2 verdict: content sensitivity vs length sensitivity

- content sensitivity (mean paired delta over length-preserved degradations shuffle_paragraphs, corrupt_numbers, typos): -0.28 (95% CI [-0.49, -0.08], n=25 papers) | threshold < -0.30 with CI excluding 0 -> not fired
- length sensitivity: pad_filler -0.49 (95% CI [-0.78, -0.20], n=25) vs > +0.15 -> not fired | drop_appendix +0.03 (95% CI [-0.25, +0.31], n=23) vs < -0.15 -> not fired
- pad-minus-drop asymmetry (papers with both, n=23): -0.54 (95% CI [-0.84, -0.24])

|  | length-sensitive: NO | length-sensitive: YES |
|---|---|---|
| content-sensitive: YES | genuine quality signal | quality signal + length bias |
| content-sensitive: NO | insensitive / noise-level **<= this model** | length-driven reviewer |

This model is NOT content-sensitive by the mechanical rule: the mean rating delta across the length-preserved degradations is -0.28 (95% CI [-0.49, -0.08], n=25), which does not cross the -0.30 threshold with a CI excluding 0. It is NOT length-sensitive: pad_filler moved ratings by -0.49 (95% CI [-0.78, -0.20]) against the +0.15 threshold and drop_appendix by +0.03 (95% CI [-0.25, +0.31]) against the -0.15 threshold, so neither trigger fired. Verdict: neither fired: effects are below both thresholds (noise-level at this sample size).

### 4. Noise context (E0/E1 true-placebo floor)

- no true-placebo line for this exact model; using the mean floor across the 5 e0 models: mean |delta| 0.36 (range 0.20-0.53)
- the floor is a between-adjacent-version quantity from real revisions, so ratios are context, not a significance test.

| effect | mean delta | \|effect\| / noise floor | exceeds floor? |
|---|---|---|---|
| content (mean degradation delta) | -0.28 | 0.8x | no |
| shuffle_paragraphs | -0.23 | 0.6x | no |
| corrupt_numbers | -0.43 | 1.2x | YES |
| typos | -0.20 | 0.6x | no |
| pad_filler | -0.49 | 1.4x | YES |
| drop_appendix | +0.03 | 0.1x | no |

