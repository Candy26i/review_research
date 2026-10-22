# E0: LLM review-score analysis

- log: `data/processed/reviews_log.jsonl` - 2124 rows, 2124 parsed_ok rating rows used
- manifest: `data/processed/pilot_manifest.jsonl` (50 papers) | diffs: `data/processed/version_diffs.jsonl`

## Model: deepseek-chat

- parsed_ok reviews: 426 | papers scored: 50 | paper-versions scored: 142

### 1. Version tracking

- papers with >= 2 scored versions: 50
- mean Kendall tau-b (all-tied excluded): 0.09 (n=39)
- tau>0: 46.0% | tau==0 (ties, incl. all-tied): 26.0% | tau<0: 28.0% | all-tied papers: 11
- tau hist: [-1.0,-0.5):12 [-0.5,+0.0):2 [+0.0,+0.5):9 [+0.5,+1.0]:16 all-tied(nan):11

Adjacent-pair direction (mean rating of v_{i+1} vs v_i):

| churn_class | n | P(>) | P(=) | P(<) |
|---|---|---|---|---|
| ALL | 92 | 39.1% | 29.3% | 31.5% |
| cosmetic | 25 | 16.0% | 32.0% | 52.0% |
| minor | 10 | 30.0% | 30.0% | 40.0% |
| major | 56 | 50.0% | 28.6% | 21.4% |
| unknown | 1 | 100.0% | 0.0% | 0.0% |

Cosmetic pairs (should be '='): 8/25 tied (32.0%).

### 1b. Inferential version-effect tests

- bootstrap 95% CI for the mean finite tau (papers resampled with replacement, 10,000 draws, seed 42): mean 0.09, CI [-0.16, +0.34]
- per-paper correlation, version position vs mean rating (>= 3 versions, constant-rating papers excluded): mean Pearson 0.31 (n=23, 69.6% positive) | mean Spearman 0.28 (n=23)
- Page trend test (first 3 versions of the 25 papers with >= 3 versions; H1 = ratings increase along versions): L=308, p=0.1444
- mixed-effects regression rating ~ version_position + (1|paper) on rep-level ratings (n=426, 50 papers): +0.11 rating points per revision step (95% CI [+0.02, +0.20], p=0.0166)
- verbosity control rating ~ version_position + log_words + (1|paper) (n=426, 50 papers): version +0.01/step (95% CI [-0.09, +0.11], p=0.8490) | log_words +1.40 (95% CI [+0.76, +2.04], p=0.0000)
  - The version coefficient does NOT survive the length control (p=0.8490 >= 0.05 once log document length is adjusted for), consistent with verbosity bias.

### 1c. Where the effect lives

Adjacent-pair mean rating delta (v_to minus v_from) by stage transition (stages from the manifest):

| stage transition | n | mean delta | mean \|delta\| | share delta==0 |
|---|---|---|---|---|
| during_review -> post_decision | 37 | +0.00 | 0.52 | 35.1% |
| post_decision -> post_decision | 20 | -0.08 | 0.55 | 25.0% |
| pre_submission -> during_review | 13 | +0.28 | 0.59 | 30.8% |
| pre_submission -> post_decision | 9 | -0.04 | 0.78 | 11.1% |
| pre_submission -> pre_submission | 8 | +0.54 | 1.12 | 12.5% |
| during_review -> during_review | 5 | +0.20 | 0.33 | 60.0% |

Same deltas by churn_class (true placebo = cosmetic text churn AND unchanged figures):

| subset | n | mean delta | mean \|delta\| | share delta==0 |
|---|---|---|---|---|
| cosmetic | 25 | -0.32 | 0.56 | 32.0% |
| minor | 10 | -0.03 | 0.37 | 30.0% |
| major | 56 | +0.24 | 0.65 | 28.6% |
| unknown | 1 | +1.67 | 1.67 | 0.0% |
| TRUE PLACEBO (cosmetic & figs unchanged) | 17 | -0.31 | 0.43 | 41.2% |
| cosmetic & figure changes | 8 | -0.33 | 0.83 | 12.5% |

- TRUE-PLACEBO noise floor (text and figures both unchanged): n=17, mean |delta| 0.43, share with delta==0 41.2%. Cosmetic pairs with figure changes are reported separately (n=8); cosmetic pairs without a figs_any_change flag are in neither subset.
- rating delta vs churn: Spearman rho(churn, |delta|) = 0.11 (p=0.3049, n=91 adjacent pairs)

### 2. Reliability

- test-retest SD across reps (per paper-version, n=142): mean 0.7 | median 0.6
- ICC(2,1) (targets=paper-versions n=142, raters=rep index k=3, 0 incomplete target(s) dropped): 0.44
- signal vs noise: between-version SD of per-version means (within paper, n=50) mean 0.5 vs test-retest SD mean 0.7 -> ratio 0.64

### 3. Human anchor

- anchor version = latest pre_submission/during_review (fallback v1); papers with scored anchor: 50 (0 skipped: anchor version not scored)

| subset | n | Pearson r | Spearman rho |
|---|---|---|---|
| overall | 50 | 0.11 | 0.13 |
| accepted | 26 | -0.03 | 0.02 |
| rejected | 24 | 0.13 | 0.14 |

| accept/reject discrimination | n | ROC AUC |
|---|---|---|
| LLM mean rating (anchor) | 50 | 0.57 |
| LLM accept-vote share (anchor) | 50 | 0.61 |
| human mean_score (reference) | 50 | 0.88 |

### 4. Distributions

| score | LLM n | LLM % | human n | human % |
|---|---|---|---|---|
| 1 | 0 | 0.0% | 0 | 0.0% |
| 2 | 0 | 0.0% | 0 | 0.0% |
| 3 | 131 | 30.8% | 17 | 8.9% |
| 4 | 0 | 0.0% | 0 | 0.0% |
| 5 | 142 | 33.3% | 44 | 23.0% |
| 6 | 151 | 35.4% | 88 | 46.1% |
| 7 | 0 | 0.0% | 0 | 0.0% |
| 8 | 2 | 0.5% | 39 | 20.4% |
| 9 | 0 | 0.0% | 0 | 0.0% |
| 10 | 0 | 0.0% | 3 | 1.6% |

- LLM ratings: mean 4.8 SD 1.3 | human per-review scores: mean 6.0 SD 1.5 (scales differ by ICLR year)
- mean LLM rating (all scored versions/reps): accepted 4.9 (n=222) vs rejected 4.6 (n=204)

## Model: deepseek-reasoner

- parsed_ok reviews: 426 | papers scored: 50 | paper-versions scored: 142

### 1. Version tracking

- papers with >= 2 scored versions: 50
- mean Kendall tau-b (all-tied excluded): 0.49 (n=41)
- tau>0: 60.0% | tau==0 (ties, incl. all-tied): 28.0% | tau<0: 12.0% | all-tied papers: 9
- tau hist: [-1.0,-0.5):4 [-0.5,+0.0):2 [+0.0,+0.5):11 [+0.5,+1.0]:24 all-tied(nan):9

Adjacent-pair direction (mean rating of v_{i+1} vs v_i):

| churn_class | n | P(>) | P(=) | P(<) |
|---|---|---|---|---|
| ALL | 92 | 50.0% | 26.1% | 23.9% |
| cosmetic | 25 | 24.0% | 28.0% | 48.0% |
| minor | 10 | 40.0% | 40.0% | 20.0% |
| major | 56 | 64.3% | 21.4% | 14.3% |
| unknown | 1 | 0.0% | 100.0% | 0.0% |

Cosmetic pairs (should be '='): 7/25 tied (28.0%).

### 1b. Inferential version-effect tests

- bootstrap 95% CI for the mean finite tau (papers resampled with replacement, 10,000 draws, seed 42): mean 0.49, CI [+0.30, +0.67]
- per-paper correlation, version position vs mean rating (>= 3 versions, constant-rating papers excluded): mean Pearson 0.40 (n=23, 87.0% positive) | mean Spearman 0.38 (n=23)
- Page trend test (first 3 versions of the 25 papers with >= 3 versions; H1 = ratings increase along versions): L=314, p=0.0202
- mixed-effects regression rating ~ version_position + (1|paper) on rep-level ratings (n=426, 50 papers): +0.15 rating points per revision step (95% CI [+0.08, +0.23], p=0.0000)
- verbosity control rating ~ version_position + log_words + (1|paper) (n=426, 50 papers): version +0.07/step (95% CI [-0.01, +0.16], p=0.1045) | log_words +1.14 (95% CI [+0.51, +1.78], p=0.0004)
  - The version coefficient does NOT survive the length control (p=0.1045 >= 0.05 once log document length is adjusted for), consistent with verbosity bias.

### 1c. Where the effect lives

Adjacent-pair mean rating delta (v_to minus v_from) by stage transition (stages from the manifest):

| stage transition | n | mean delta | mean \|delta\| | share delta==0 |
|---|---|---|---|---|
| during_review -> post_decision | 37 | +0.29 | 0.52 | 27.0% |
| post_decision -> post_decision | 20 | -0.10 | 0.33 | 35.0% |
| pre_submission -> during_review | 13 | +0.13 | 0.44 | 23.1% |
| pre_submission -> post_decision | 9 | +0.63 | 0.63 | 22.2% |
| pre_submission -> pre_submission | 8 | +0.17 | 0.75 | 12.5% |
| during_review -> during_review | 5 | -0.20 | 0.60 | 20.0% |

Same deltas by churn_class (true placebo = cosmetic text churn AND unchanged figures):

| subset | n | mean delta | mean \|delta\| | share delta==0 |
|---|---|---|---|---|
| cosmetic | 25 | -0.16 | 0.45 | 28.0% |
| minor | 10 | +0.03 | 0.37 | 40.0% |
| major | 56 | +0.36 | 0.56 | 21.4% |
| unknown | 1 | +0.00 | 0.00 | 100.0% |
| TRUE PLACEBO (cosmetic & figs unchanged) | 17 | -0.18 | 0.53 | 23.5% |
| cosmetic & figure changes | 8 | -0.13 | 0.29 | 37.5% |

- TRUE-PLACEBO noise floor (text and figures both unchanged): n=17, mean |delta| 0.53, share with delta==0 23.5%. Cosmetic pairs with figure changes are reported separately (n=8); cosmetic pairs without a figs_any_change flag are in neither subset.
- rating delta vs churn: Spearman rho(churn, |delta|) = 0.15 (p=0.1461, n=91 adjacent pairs)

### 2. Reliability

- test-retest SD across reps (per paper-version, n=142): mean 0.6 | median 0.6
- ICC(2,1) (targets=paper-versions n=142, raters=rep index k=3, 0 incomplete target(s) dropped): 0.64
- signal vs noise: between-version SD of per-version means (within paper, n=50) mean 0.4 vs test-retest SD mean 0.6 -> ratio 0.65

### 3. Human anchor

- anchor version = latest pre_submission/during_review (fallback v1); papers with scored anchor: 50 (0 skipped: anchor version not scored)

| subset | n | Pearson r | Spearman rho |
|---|---|---|---|
| overall | 50 | 0.16 | 0.22 |
| accepted | 26 | -0.27 | -0.17 |
| rejected | 24 | 0.20 | 0.22 |

| accept/reject discrimination | n | ROC AUC |
|---|---|---|
| LLM mean rating (anchor) | 50 | 0.69 |
| LLM accept-vote share (anchor) | 50 | 0.66 |
| human mean_score (reference) | 50 | 0.88 |

### 4. Distributions

| score | LLM n | LLM % | human n | human % |
|---|---|---|---|---|
| 1 | 0 | 0.0% | 0 | 0.0% |
| 2 | 0 | 0.0% | 0 | 0.0% |
| 3 | 87 | 20.4% | 17 | 8.9% |
| 4 | 0 | 0.0% | 0 | 0.0% |
| 5 | 148 | 34.7% | 44 | 23.0% |
| 6 | 168 | 39.4% | 88 | 46.1% |
| 7 | 0 | 0.0% | 0 | 0.0% |
| 8 | 23 | 5.4% | 39 | 20.4% |
| 9 | 0 | 0.0% | 0 | 0.0% |
| 10 | 0 | 0.0% | 3 | 1.6% |

- LLM ratings: mean 5.1 SD 1.3 | human per-review scores: mean 6.0 SD 1.5 (scales differ by ICLR year)
- mean LLM rating (all scored versions/reps): accepted 5.5 (n=222) vs rejected 4.8 (n=204)

## Model: gemini-3.8-flash

- parsed_ok reviews: 426 | papers scored: 50 | paper-versions scored: 142

### 1. Version tracking

- papers with >= 2 scored versions: 50
- mean Kendall tau-b (all-tied excluded): 0.57 (n=32)
- tau>0: 50.0% | tau==0 (ties, incl. all-tied): 42.0% | tau<0: 8.0% | all-tied papers: 18
- tau hist: [-1.0,-0.5):3 [-0.5,+0.0):1 [+0.0,+0.5):6 [+0.5,+1.0]:22 all-tied(nan):18

Adjacent-pair direction (mean rating of v_{i+1} vs v_i):

| churn_class | n | P(>) | P(=) | P(<) |
|---|---|---|---|---|
| ALL | 92 | 40.2% | 44.6% | 15.2% |
| cosmetic | 25 | 24.0% | 48.0% | 28.0% |
| minor | 10 | 10.0% | 80.0% | 10.0% |
| major | 56 | 51.8% | 37.5% | 10.7% |
| unknown | 1 | 100.0% | 0.0% | 0.0% |

Cosmetic pairs (should be '='): 12/25 tied (48.0%).

### 1b. Inferential version-effect tests

- bootstrap 95% CI for the mean finite tau (papers resampled with replacement, 10,000 draws, seed 42): mean 0.57, CI [+0.35, +0.77]
- per-paper correlation, version position vs mean rating (>= 3 versions, constant-rating papers excluded): mean Pearson 0.43 (n=19, 78.9% positive) | mean Spearman 0.41 (n=19)
- Page trend test (first 3 versions of the 25 papers with >= 3 versions; H1 = ratings increase along versions): L=312, p=0.0385
- mixed-effects regression rating ~ version_position + (1|paper) on rep-level ratings (n=426, 50 papers): +0.28 rating points per revision step (95% CI [+0.21, +0.36], p=0.0000)
- verbosity control rating ~ version_position + log_words + (1|paper) (n=426, 50 papers): version +0.14/step (95% CI [+0.06, +0.23], p=0.0012) | log_words +1.95 (95% CI [+1.28, +2.62], p=0.0000)
  - The version coefficient SURVIVES the length control (+0.14/step, p=0.0012 after adjusting for log document length), so the version effect is not just verbosity bias.

### 1c. Where the effect lives

Adjacent-pair mean rating delta (v_to minus v_from) by stage transition (stages from the manifest):

| stage transition | n | mean delta | mean \|delta\| | share delta==0 |
|---|---|---|---|---|
| during_review -> post_decision | 37 | +0.19 | 0.44 | 48.6% |
| post_decision -> post_decision | 20 | +0.15 | 0.38 | 35.0% |
| pre_submission -> during_review | 13 | +0.46 | 0.56 | 53.8% |
| pre_submission -> post_decision | 9 | +0.96 | 0.96 | 33.3% |
| pre_submission -> pre_submission | 8 | +0.50 | 0.67 | 37.5% |
| during_review -> during_review | 5 | +0.53 | 0.53 | 60.0% |

Same deltas by churn_class (true placebo = cosmetic text churn AND unchanged figures):

| subset | n | mean delta | mean \|delta\| | share delta==0 |
|---|---|---|---|---|
| cosmetic | 25 | +0.04 | 0.33 | 48.0% |
| minor | 10 | +0.03 | 0.10 | 80.0% |
| major | 56 | +0.53 | 0.68 | 37.5% |
| unknown | 1 | +0.33 | 0.33 | 0.0% |
| TRUE PLACEBO (cosmetic & figs unchanged) | 17 | +0.04 | 0.31 | 47.1% |
| cosmetic & figure changes | 8 | +0.04 | 0.38 | 50.0% |

- TRUE-PLACEBO noise floor (text and figures both unchanged): n=17, mean |delta| 0.31, share with delta==0 47.1%. Cosmetic pairs with figure changes are reported separately (n=8); cosmetic pairs without a figs_any_change flag are in neither subset.
- rating delta vs churn: Spearman rho(churn, |delta|) = 0.19 (p=0.0761, n=91 adjacent pairs)

### 2. Reliability

- test-retest SD across reps (per paper-version, n=142): mean 0.4 | median 0.0
- ICC(2,1) (targets=paper-versions n=142, raters=rep index k=3, 0 incomplete target(s) dropped): 0.81
- signal vs noise: between-version SD of per-version means (within paper, n=50) mean 0.5 vs test-retest SD mean 0.4 -> ratio 1.27

### 3. Human anchor

- anchor version = latest pre_submission/during_review (fallback v1); papers with scored anchor: 50 (0 skipped: anchor version not scored)

| subset | n | Pearson r | Spearman rho |
|---|---|---|---|
| overall | 50 | 0.16 | 0.28 |
| accepted | 26 | -0.22 | -0.03 |
| rejected | 24 | 0.37 | 0.38 |

| accept/reject discrimination | n | ROC AUC |
|---|---|---|
| LLM mean rating (anchor) | 50 | 0.70 |
| LLM accept-vote share (anchor) | 50 | 0.58 |
| human mean_score (reference) | 50 | 0.88 |

### 4. Distributions

| score | LLM n | LLM % | human n | human % |
|---|---|---|---|---|
| 1 | 2 | 0.5% | 0 | 0.0% |
| 2 | 0 | 0.0% | 0 | 0.0% |
| 3 | 46 | 10.8% | 17 | 8.9% |
| 4 | 0 | 0.0% | 0 | 0.0% |
| 5 | 45 | 10.6% | 44 | 23.0% |
| 6 | 240 | 56.3% | 88 | 46.1% |
| 7 | 0 | 0.0% | 0 | 0.0% |
| 8 | 93 | 21.8% | 39 | 20.4% |
| 9 | 0 | 0.0% | 0 | 0.0% |
| 10 | 0 | 0.0% | 3 | 1.6% |

- LLM ratings: mean 6.0 SD 1.4 | human per-review scores: mean 6.0 SD 1.5 (scales differ by ICLR year)
- mean LLM rating (all scored versions/reps): accepted 6.3 (n=222) vs rejected 5.6 (n=204)

## Model: gemini-3.8-flash+figs

- parsed_ok reviews: 420 | papers scored: 49 | paper-versions scored: 140

### 1. Version tracking

- papers with >= 2 scored versions: 49
- mean Kendall tau-b (all-tied excluded): 0.48 (n=34)
- tau>0: 51.0% | tau==0 (ties, incl. all-tied): 34.7% | tau<0: 14.3% | all-tied papers: 15
- tau hist: [-1.0,-0.5):7 [-0.5,+0.0):0 [+0.0,+0.5):3 [+0.5,+1.0]:24 all-tied(nan):15

Adjacent-pair direction (mean rating of v_{i+1} vs v_i):

| churn_class | n | P(>) | P(=) | P(<) |
|---|---|---|---|---|
| ALL | 91 | 37.4% | 48.4% | 14.3% |
| cosmetic | 25 | 24.0% | 52.0% | 24.0% |
| minor | 10 | 20.0% | 50.0% | 30.0% |
| major | 55 | 47.3% | 45.5% | 7.3% |
| unknown | 1 | 0.0% | 100.0% | 0.0% |

Cosmetic pairs (should be '='): 13/25 tied (52.0%).

### 1b. Inferential version-effect tests

- bootstrap 95% CI for the mean finite tau (papers resampled with replacement, 10,000 draws, seed 42): mean 0.48, CI [+0.22, +0.72]
- per-paper correlation, version position vs mean rating (>= 3 versions, constant-rating papers excluded): mean Pearson 0.42 (n=16, 75.0% positive) | mean Spearman 0.41 (n=16)
- Page trend test (first 3 versions of the 25 papers with >= 3 versions; H1 = ratings increase along versions): L=314, p=0.0239
- mixed-effects regression rating ~ version_position + (1|paper) on rep-level ratings (n=420, 49 papers): +0.22 rating points per revision step (95% CI [+0.15, +0.30], p=0.0000)
- verbosity control rating ~ version_position + log_words + (1|paper) (n=420, 49 papers): version +0.06/step (95% CI [-0.03, +0.15], p=0.1822) | log_words +2.27 (95% CI [+1.58, +2.96], p=0.0000)
  - The version coefficient does NOT survive the length control (p=0.1822 >= 0.05 once log document length is adjusted for), consistent with verbosity bias.

### 1c. Where the effect lives

Adjacent-pair mean rating delta (v_to minus v_from) by stage transition (stages from the manifest):

| stage transition | n | mean delta | mean \|delta\| | share delta==0 |
|---|---|---|---|---|
| during_review -> post_decision | 37 | +0.32 | 0.49 | 43.2% |
| post_decision -> post_decision | 20 | +0.13 | 0.33 | 55.0% |
| pre_submission -> during_review | 13 | +0.41 | 0.41 | 61.5% |
| pre_submission -> post_decision | 8 | +0.75 | 0.92 | 25.0% |
| pre_submission -> pre_submission | 8 | +0.21 | 0.46 | 62.5% |
| during_review -> during_review | 5 | -0.33 | 0.33 | 40.0% |

Same deltas by churn_class (true placebo = cosmetic text churn AND unchanged figures):

| subset | n | mean delta | mean \|delta\| | share delta==0 |
|---|---|---|---|---|
| cosmetic | 25 | +0.03 | 0.35 | 52.0% |
| minor | 10 | -0.03 | 0.30 | 50.0% |
| major | 55 | +0.47 | 0.56 | 45.5% |
| unknown | 1 | +0.00 | 0.00 | 100.0% |
| TRUE PLACEBO (cosmetic & figs unchanged) | 17 | +0.08 | 0.31 | 52.9% |
| cosmetic & figure changes | 8 | -0.08 | 0.42 | 50.0% |

- TRUE-PLACEBO noise floor (text and figures both unchanged): n=17, mean |delta| 0.31, share with delta==0 52.9%. Cosmetic pairs with figure changes are reported separately (n=8); cosmetic pairs without a figs_any_change flag are in neither subset.
- rating delta vs churn: Spearman rho(churn, |delta|) = 0.15 (p=0.1555, n=90 adjacent pairs)

### 2. Reliability

- test-retest SD across reps (per paper-version, n=140): mean 0.4 | median 0.0
- ICC(2,1) (targets=paper-versions n=140, raters=rep index k=3, 0 incomplete target(s) dropped): 0.81
- signal vs noise: between-version SD of per-version means (within paper, n=49) mean 0.4 vs test-retest SD mean 0.4 -> ratio 1.12

### 3. Human anchor

- anchor version = latest pre_submission/during_review (fallback v1); papers with scored anchor: 49 (0 skipped: anchor version not scored)

| subset | n | Pearson r | Spearman rho |
|---|---|---|---|
| overall | 49 | 0.28 | 0.33 |
| accepted | 25 | -0.11 | 0.05 |
| rejected | 24 | 0.37 | 0.38 |

| accept/reject discrimination | n | ROC AUC |
|---|---|---|
| LLM mean rating (anchor) | 49 | 0.71 |
| LLM accept-vote share (anchor) | 49 | 0.60 |
| human mean_score (reference) | 49 | 0.88 |

### 4. Distributions

| score | LLM n | LLM % | human n | human % |
|---|---|---|---|---|
| 1 | 7 | 1.7% | 0 | 0.0% |
| 2 | 0 | 0.0% | 0 | 0.0% |
| 3 | 42 | 10.0% | 17 | 8.9% |
| 4 | 0 | 0.0% | 0 | 0.0% |
| 5 | 40 | 9.5% | 44 | 23.0% |
| 6 | 240 | 57.1% | 88 | 46.1% |
| 7 | 0 | 0.0% | 0 | 0.0% |
| 8 | 91 | 21.7% | 39 | 20.4% |
| 9 | 0 | 0.0% | 0 | 0.0% |
| 10 | 0 | 0.0% | 3 | 1.6% |

- LLM ratings: mean 6.0 SD 1.5 | human per-review scores: mean 6.0 SD 1.5 (scales differ by ICLR year)
- mean LLM rating (all scored versions/reps): accepted 6.4 (n=216) vs rejected 5.5 (n=204)

## Model: gemini-3.8-flash-renone

- parsed_ok reviews: 426 | papers scored: 50 | paper-versions scored: 142

### 1. Version tracking

- papers with >= 2 scored versions: 50
- mean Kendall tau-b (all-tied excluded): 0.45 (n=34)
- tau>0: 50.0% | tau==0 (ties, incl. all-tied): 36.0% | tau<0: 14.0% | all-tied papers: 16
- tau hist: [-1.0,-0.5):7 [-0.5,+0.0):0 [+0.0,+0.5):3 [+0.5,+1.0]:24 all-tied(nan):16

Adjacent-pair direction (mean rating of v_{i+1} vs v_i):

| churn_class | n | P(>) | P(=) | P(<) |
|---|---|---|---|---|
| ALL | 92 | 38.0% | 48.9% | 13.0% |
| cosmetic | 25 | 20.0% | 64.0% | 16.0% |
| minor | 10 | 40.0% | 30.0% | 30.0% |
| major | 56 | 44.6% | 46.4% | 8.9% |
| unknown | 1 | 100.0% | 0.0% | 0.0% |

Cosmetic pairs (should be '='): 16/25 tied (64.0%).

### 1b. Inferential version-effect tests

- bootstrap 95% CI for the mean finite tau (papers resampled with replacement, 10,000 draws, seed 42): mean 0.45, CI [+0.19, +0.69]
- per-paper correlation, version position vs mean rating (>= 3 versions, constant-rating papers excluded): mean Pearson 0.52 (n=17, 76.5% positive) | mean Spearman 0.53 (n=17)
- Page trend test (first 3 versions of the 25 papers with >= 3 versions; H1 = ratings increase along versions): L=316, p=0.0098
- mixed-effects regression rating ~ version_position + (1|paper) on rep-level ratings (n=426, 50 papers): +0.20 rating points per revision step (95% CI [+0.13, +0.26], p=0.0000)
- verbosity control rating ~ version_position + log_words + (1|paper) (n=426, 50 papers): version +0.09/step (95% CI [+0.01, +0.17], p=0.0214) | log_words +1.43 (95% CI [+0.83, +2.03], p=0.0000)
  - The version coefficient SURVIVES the length control (+0.09/step, p=0.0214 after adjusting for log document length), so the version effect is not just verbosity bias.

### 1c. Where the effect lives

Adjacent-pair mean rating delta (v_to minus v_from) by stage transition (stages from the manifest):

| stage transition | n | mean delta | mean \|delta\| | share delta==0 |
|---|---|---|---|---|
| during_review -> post_decision | 37 | +0.28 | 0.44 | 51.4% |
| post_decision -> post_decision | 20 | +0.20 | 0.40 | 50.0% |
| pre_submission -> during_review | 13 | +0.33 | 0.44 | 61.5% |
| pre_submission -> post_decision | 9 | +0.04 | 0.56 | 22.2% |
| pre_submission -> pre_submission | 8 | +0.21 | 0.21 | 50.0% |
| during_review -> during_review | 5 | +0.07 | 0.33 | 40.0% |

Same deltas by churn_class (true placebo = cosmetic text churn AND unchanged figures):

| subset | n | mean delta | mean \|delta\| | share delta==0 |
|---|---|---|---|---|
| cosmetic | 25 | +0.07 | 0.31 | 64.0% |
| minor | 10 | +0.07 | 0.40 | 30.0% |
| major | 56 | +0.33 | 0.47 | 46.4% |
| unknown | 1 | +0.33 | 0.33 | 0.0% |
| TRUE PLACEBO (cosmetic & figs unchanged) | 17 | +0.16 | 0.20 | 82.4% |
| cosmetic & figure changes | 8 | -0.13 | 0.54 | 25.0% |

- TRUE-PLACEBO noise floor (text and figures both unchanged): n=17, mean |delta| 0.20, share with delta==0 82.4%. Cosmetic pairs with figure changes are reported separately (n=8); cosmetic pairs without a figs_any_change flag are in neither subset.
- rating delta vs churn: Spearman rho(churn, |delta|) = 0.18 (p=0.0909, n=91 adjacent pairs)

### 2. Reliability

- test-retest SD across reps (per paper-version, n=142): mean 0.4 | median 0.0
- ICC(2,1) (targets=paper-versions n=142, raters=rep index k=3, 0 incomplete target(s) dropped): 0.78
- signal vs noise: between-version SD of per-version means (within paper, n=50) mean 0.4 vs test-retest SD mean 0.4 -> ratio 1.15

### 3. Human anchor

- anchor version = latest pre_submission/during_review (fallback v1); papers with scored anchor: 50 (0 skipped: anchor version not scored)

| subset | n | Pearson r | Spearman rho |
|---|---|---|---|
| overall | 50 | 0.22 | 0.36 |
| accepted | 26 | -0.12 | 0.02 |
| rejected | 24 | 0.36 | 0.58 |

| accept/reject discrimination | n | ROC AUC |
|---|---|---|
| LLM mean rating (anchor) | 50 | 0.69 |
| LLM accept-vote share (anchor) | 50 | 0.55 |
| human mean_score (reference) | 50 | 0.88 |

### 4. Distributions

| score | LLM n | LLM % | human n | human % |
|---|---|---|---|---|
| 1 | 0 | 0.0% | 0 | 0.0% |
| 2 | 0 | 0.0% | 0 | 0.0% |
| 3 | 37 | 8.7% | 17 | 8.9% |
| 4 | 0 | 0.0% | 0 | 0.0% |
| 5 | 54 | 12.7% | 44 | 23.0% |
| 6 | 252 | 59.2% | 88 | 46.1% |
| 7 | 0 | 0.0% | 0 | 0.0% |
| 8 | 83 | 19.5% | 39 | 20.4% |
| 9 | 0 | 0.0% | 0 | 0.0% |
| 10 | 0 | 0.0% | 3 | 1.6% |

- LLM ratings: mean 6.0 SD 1.3 | human per-review scores: mean 6.0 SD 1.5 (scales differ by ICLR year)
- mean LLM rating (all scored versions/reps): accepted 6.4 (n=222) vs rejected 5.6 (n=204)

