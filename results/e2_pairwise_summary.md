# E2: Pairwise version discrimination

- pairwise log: `data/processed/pairwise_log.jsonl` - 364 rows, 364 parsed_ok
- diffs: `data/processed/version_diffs.jsonl` | manifest: `data/processed/pilot_manifest.jsonl` (50 papers)
- reviews log (absolute mode): `data/processed/reviews_log.jsonl` - models present: deepseek-chat, deepseek-reasoner, gemini-3.8-flash, gemini-3.8-flash+figs, gemini-3.8-flash-renone

## Model: deepseek-chat

- judgments used: 182 | distinct pairs: 91 | papers: 50

### 1. Later-pick rate

| subset | n | later-pick | 95% CI | tie | early |
|---|---|---|---|---|---|
| ALL | 182 | 68.7% | [61.6%, 75.0%] | 18.1% | 13.2% |
| cosmetic | 50 | 30.0% | [19.1%, 43.8%] | 66.0% | 4.0% |
| minor | 20 | 75.0% | [53.1%, 88.8%] | 0.0% | 25.0% |
| major | 112 | 84.8% | [77.0%, 90.3%] | 0.0% | 15.2% |
| TRUE PLACEBO (cosmetic & figs unchanged) | 34 | 14.7% | [6.4%, 30.1%] | 82.4% | 2.9% |
| cosmetic & figure changes | 16 | 62.5% | [38.6%, 81.5%] | 31.2% | 6.2% |

- TRUE-PLACEBO check (no real difference to find, so later-pick should ~= early-pick): later 14.7% vs early 2.9% (diff +11.8 pts, n=34, ties 82.4%)

### 2. Position bias

- P(pick label A): 52.3% [44.4%, 60.2%] over n=149 non-tie labeled judgments (33 tie(s) and 0 label-unknown excluded); no-bias reference is 50%
- pairs judged in both orders: 91 - same version both orders: 55 (60.4%) | flips with position: 18 (19.8%) | tie-involved/ambiguous: 18 (19.8%)
- one-order-only pairs (excluded from consistency): 0

### 3. Pairwise vs absolute scoring (same pairs)

| mode | n pairs | later | tie | early |
|---|---|---|---|---|
| pairwise (majority of judgments) | 91 | 60.4% | 36.3% | 3.3% |
| absolute (mean-rating direction) | 91 | 38.5% | 29.7% | 31.9% |

Cross-tab (rows = pairwise, columns = absolute):

| pairwise \ absolute | later | tie | earlier |
|---|---|---|---|
| later | 23 | 19 | 13 |
| tie | 10 | 8 | 15 |
| earlier | 2 | 0 | 1 |

- agreement on the same pairs: 32/91 (35.2%)
- judged pairs without absolute scores for both versions: 0 (excluded)

### 4. Margin calibration

| margin | n | later-pick | 95% CI | tie | early |
|---|---|---|---|---|---|
| slight | 139 | 61.2% | [52.9%, 68.8%] | 23.7% | 15.1% |
| clear | 43 | 93.0% | [81.4%, 97.6%] | 0.0% | 7.0% |

## Model: deepseek-reasoner

- judgments used: 182 | distinct pairs: 91 | papers: 50

### 1. Later-pick rate

| subset | n | later-pick | 95% CI | tie | early |
|---|---|---|---|---|---|
| ALL | 182 | 76.9% | [70.3%, 82.4%] | 17.0% | 6.0% |
| cosmetic | 50 | 40.0% | [27.6%, 53.8%] | 58.0% | 2.0% |
| minor | 20 | 80.0% | [58.4%, 91.9%] | 10.0% | 10.0% |
| major | 112 | 92.9% | [86.5%, 96.3%] | 0.0% | 7.1% |
| TRUE PLACEBO (cosmetic & figs unchanged) | 34 | 17.6% | [8.3%, 33.5%] | 82.4% | 0.0% |
| cosmetic & figure changes | 16 | 87.5% | [64.0%, 96.5%] | 6.2% | 6.2% |

- TRUE-PLACEBO check (no real difference to find, so later-pick should ~= early-pick): later 17.6% vs early 0.0% (diff +17.6 pts, n=34, ties 82.4%)

### 2. Position bias

- P(pick label A): 49.0% [41.2%, 56.9%] over n=151 non-tie labeled judgments (31 tie(s) and 0 label-unknown excluded); no-bias reference is 50%
- pairs judged in both orders: 91 - same version both orders: 69 (75.8%) | flips with position: 4 (4.4%) | tie-involved/ambiguous: 18 (19.8%)
- one-order-only pairs (excluded from consistency): 0

### 3. Pairwise vs absolute scoring (same pairs)

| mode | n pairs | later | tie | early |
|---|---|---|---|---|
| pairwise (majority of judgments) | 91 | 76.9% | 18.7% | 4.4% |
| absolute (mean-rating direction) | 91 | 50.5% | 25.3% | 24.2% |

Cross-tab (rows = pairwise, columns = absolute):

| pairwise \ absolute | later | tie | earlier |
|---|---|---|---|
| later | 40 | 19 | 11 |
| tie | 4 | 3 | 10 |
| earlier | 2 | 1 | 1 |

- agreement on the same pairs: 44/91 (48.4%)
- judged pairs without absolute scores for both versions: 0 (excluded)

### 4. Margin calibration

| margin | n | later-pick | 95% CI | tie | early |
|---|---|---|---|---|---|
| slight | 93 | 55.9% | [45.8%, 65.6%] | 32.3% | 11.8% |
| clear | 83 | 100.0% | [95.6%, 100.0%] | 0.0% | 0.0% |
| decisive | 6 | 83.3% | [43.6%, 97.0%] | 16.7% | 0.0% |

