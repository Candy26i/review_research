# E5: LLM score deltas vs REAL human score deltas

- log: `data/processed/e5_reviews_log.jsonl` - 300 rows, 297 parsed_ok rating rows used
- pairs: `data/processed/e5_pairs.jsonl` - 184 rows, 25 includable, 159 excluded by the builder
- manifest: `data/processed/e5_manifest.jsonl` - 25 papers | noise floor: `results/acl_scores_summary.md`

## Model: deepseek-chat

### Coverage

- parsed_ok reviews: 150 | papers with both rounds scored: 25 | joined to a pair row: 25
- skipped: 0 paper(s) missing a scored round, 0 scored paper(s) absent from the manifest, 0 scored paper(s) without a pair row, 0 pair row(s) never scored by this model

### 1. Correlation: LLM delta vs human delta

| subset | n | Pearson r | p | Spearman rho | p |
|---|---|---|---|---|---|
| same_scale, human_delta_raw | 17 | -0.25 | 0.3271 | -0.28 | 0.2734 |
| all pairs, human_delta_z | 25 | -0.09 | 0.6631 | -0.13 | 0.5384 |

Direction table (human delta from human_delta_z, raw fallback; 0 human-flat pair(s) excluded):

| human \ LLM | LLM up | LLM flat | LLM down |
|---|---|---|---|
| human up | 7 | 3 | 5 |
| human down | 6 | 3 | 1 |

- sign agreement (up,up + down,down): 8/25 = 32.0% (Wilson 95% CI [17.2%, 51.6%]); an LLM-flat pair counts as disagreement

### 2. Magnitude

OLS llm_delta ~ human_delta (slope = LLM rating points per human rating point):

| subset | n | slope | 95% CI | p |
|---|---|---|---|---|
| same_scale, human_delta_raw | 17 | -0.09 | [-0.29, +0.10] | 0.3271 |
| all pairs, human_delta_z | 25 | -0.06 | [-0.34, +0.22] | 0.6631 |

Does the LLM delta separate eventual round-2 re-acceptance?

- mean llm_delta: re-accepted +0.11 (n=9) vs re-rejected +0.29 (n=16)
- ROC AUC(llm_delta -> re-accepted): 0.43 (n=9+16)

### 3. Baseline: overall LLM delta vs the placebo noise floor

- mean llm_delta over all 25 scored pairs: +0.23 (95% CI [-0.06, +0.52], one-sample t vs 0: p=0.1211) - resubmissions are revised to address reviews, so this should be positive
- TRUE-PLACEBO noise floor: mean |delta| 0.47 (mean delta +0.05, n=615 cosmetic pairs) -> |mean llm_delta| / floor = 0.48 (NOT above the floor)

### 4. Caveats (read before quoting)

1. n: 25 scored pairs (184 rows in the pairs file, 159 excluded by the builder [no_bulk_match: 50, version_no_version_before_round1_decision: 43, decision2_withdrawn: 34, version_unchanged_between_rounds: 32]; 0 includable pair(s) not scored by this model, 0 missing a scored round). Small-n statistics above carry their own guards.
2. Match-coverage bias: a pair survives only when its paper's arXiv record is in the bulk harvest (datestamp >= 2024-06-01; no_bulk_match alone removed 50 pair(s)), i.e. papers touched on arXiv recently - older or never-revised submissions cannot enter, so the sample over-represents actively revised work.
3. weak_identity: 0/25 scored pairs were matched across rounds on title similarity without a strong author confirmation - some may not be the same underlying paper.
4. same_scale: 17/25 scored pairs have both rounds on the same realized ICLR score scale; human_delta_raw is only interpretable there (2022-2025 are mutually compatible; other years are not), hence the z-scored column for the rest.
5. Human deltas are themselves noisy (reviewer turnover, different committees across years for resubmissions), which attenuates every correlation above toward 0.

## Model: deepseek-reasoner

### Coverage

- parsed_ok reviews: 147 | papers with both rounds scored: 25 | joined to a pair row: 25
- skipped: 0 paper(s) missing a scored round, 0 scored paper(s) absent from the manifest, 0 scored paper(s) without a pair row, 0 pair row(s) never scored by this model

### 1. Correlation: LLM delta vs human delta

| subset | n | Pearson r | p | Spearman rho | p |
|---|---|---|---|---|---|
| same_scale, human_delta_raw | 17 | -0.13 | 0.6299 | -0.10 | 0.7022 |
| all pairs, human_delta_z | 25 | +0.05 | 0.8255 | +0.04 | 0.8406 |

Direction table (human delta from human_delta_z, raw fallback; 0 human-flat pair(s) excluded):

| human \ LLM | LLM up | LLM flat | LLM down |
|---|---|---|---|
| human up | 7 | 3 | 5 |
| human down | 4 | 4 | 2 |

- sign agreement (up,up + down,down): 9/25 = 36.0% (Wilson 95% CI [20.2%, 55.5%]); an LLM-flat pair counts as disagreement

### 2. Magnitude

OLS llm_delta ~ human_delta (slope = LLM rating points per human rating point):

| subset | n | slope | 95% CI | p |
|---|---|---|---|---|
| same_scale, human_delta_raw | 17 | -0.05 | [-0.25, +0.16] | 0.6299 |
| all pairs, human_delta_z | 25 | +0.02 | [-0.20, +0.25] | 0.8255 |

Does the LLM delta separate eventual round-2 re-acceptance?

- mean llm_delta: re-accepted +0.15 (n=9) vs re-rejected +0.02 (n=16)
- ROC AUC(llm_delta -> re-accepted): 0.57 (n=9+16)

### 3. Baseline: overall LLM delta vs the placebo noise floor

- mean llm_delta over all 25 scored pairs: +0.07 (95% CI [-0.16, +0.30], one-sample t vs 0: p=0.5566) - resubmissions are revised to address reviews, so this should be positive
- TRUE-PLACEBO noise floor (model 'deepseek-reasoner' absent; using the only floor, 'deepseek-chat'): mean |delta| 0.47 (mean delta +0.05, n=615 cosmetic pairs) -> |mean llm_delta| / floor = 0.14 (NOT above the floor)

### 4. Caveats (read before quoting)

1. n: 25 scored pairs (184 rows in the pairs file, 159 excluded by the builder [no_bulk_match: 50, version_no_version_before_round1_decision: 43, decision2_withdrawn: 34, version_unchanged_between_rounds: 32]; 0 includable pair(s) not scored by this model, 0 missing a scored round). Small-n statistics above carry their own guards.
2. Match-coverage bias: a pair survives only when its paper's arXiv record is in the bulk harvest (datestamp >= 2024-06-01; no_bulk_match alone removed 50 pair(s)), i.e. papers touched on arXiv recently - older or never-revised submissions cannot enter, so the sample over-represents actively revised work.
3. weak_identity: 0/25 scored pairs were matched across rounds on title similarity without a strong author confirmation - some may not be the same underlying paper.
4. same_scale: 17/25 scored pairs have both rounds on the same realized ICLR score scale; human_delta_raw is only interpretable there (2022-2025 are mutually compatible; other years are not), hence the z-scored column for the rest.
5. Human deltas are themselves noisy (reviewer turnover, different committees across years for resubmissions), which attenuates every correlation above toward 0.

