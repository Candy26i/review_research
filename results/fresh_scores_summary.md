# E0: LLM review-score analysis

- log: `data/processed/fresh_reviews_log.jsonl` - 583 rows, 583 parsed_ok rating rows used
- manifest: `data/processed/fresh_manifest.jsonl` (30 papers) | diffs: `data/processed/fresh_version_diffs.jsonl`

## Model: deepseek-reasoner

- parsed_ok reviews: 193 | papers scored: 29 | paper-versions scored: 65

### 1. Version tracking

- papers with >= 2 scored versions: 28
- mean Kendall tau-b (all-tied excluded): 0.36 (n=15)
- tau>0: 39.3% | tau==0 (ties, incl. all-tied): 46.4% | tau<0: 14.3% | all-tied papers: 13
- tau hist: [-1.0,-0.5):4 [-0.5,+0.0):0 [+0.0,+0.5):2 [+0.5,+1.0]:9 all-tied(nan):13

Adjacent-pair direction (mean rating of v_{i+1} vs v_i):

| churn_class | n | P(>) | P(=) | P(<) |
|---|---|---|---|---|
| ALL | 36 | 30.6% | 50.0% | 19.4% |
| cosmetic | 11 | 36.4% | 36.4% | 27.3% |
| minor | 7 | 28.6% | 57.1% | 14.3% |
| major | 17 | 23.5% | 58.8% | 17.6% |
| unknown | 1 | 100.0% | 0.0% | 0.0% |

Cosmetic pairs (should be '='): 4/11 tied (36.4%).

### 1b. Inferential version-effect tests

- bootstrap 95% CI for the mean finite tau (papers resampled with replacement, 10,000 draws, seed 42): mean 0.36, CI [-0.08, +0.76]
- per-paper correlation, version position vs mean rating (>= 3 versions, constant-rating papers excluded): mean Pearson 0.23 (n=4, 75.0% positive) | mean Spearman 0.16 (n=4)
- Page trend test (first 3 versions of the 7 papers with >= 3 versions; H1 = ratings increase along versions): L=86, p=0.3476
- mixed-effects regression rating ~ version_position + (1|paper) on rep-level ratings (n=190, 28 papers): +0.19 rating points per revision step (95% CI [+0.05, +0.34], p=0.0097)
- verbosity control rating ~ version_position + log_words + (1|paper): n/a (needs >= 3 papers, >= 2 positions, and scrub-log n_words with variation)

### 1c. Where the effect lives

Adjacent-pair mean rating delta (v_to minus v_from) by stage transition (stages from the manifest):

| stage transition | n | mean delta | mean \|delta\| | share delta==0 |
|---|---|---|---|---|
| unknown -> unknown | 36 | +0.19 | 0.42 | 50.0% |

Same deltas by churn_class (true placebo = cosmetic text churn AND unchanged figures):

| subset | n | mean delta | mean \|delta\| | share delta==0 |
|---|---|---|---|---|
| cosmetic | 11 | +0.18 | 0.36 | 36.4% |
| minor | 7 | +0.43 | 0.62 | 57.1% |
| major | 17 | +0.06 | 0.33 | 58.8% |
| unknown | 1 | +1.00 | 1.00 | 0.0% |
| TRUE PLACEBO (cosmetic & figs unchanged) | 8 | +0.21 | 0.38 | 50.0% |
| cosmetic & figure changes | 3 | +0.11 | 0.33 | 0.0% |

- TRUE-PLACEBO noise floor (text and figures both unchanged): n=8, mean |delta| 0.38, share with delta==0 50.0%. Cosmetic pairs with figure changes are reported separately (n=3); cosmetic pairs without a figs_any_change flag are in neither subset.
- rating delta vs churn: Spearman rho(churn, |delta|) = -0.01 (p=0.9450, n=35 adjacent pairs)

### 2. Reliability

- test-retest SD across reps (per paper-version, n=65): mean 0.3 | median 0.0
- ICC(2,1) (targets=paper-versions n=63, raters=rep index k=3, 2 incomplete target(s) dropped): 0.77
- signal vs noise: between-version SD of per-version means (within paper, n=28) mean 0.3 vs test-retest SD mean 0.3 -> ratio 0.89

### 3. Human anchor

- anchor version = latest pre_submission/during_review (fallback v1); papers with scored anchor: 29 (0 skipped: anchor version not scored)

| subset | n | Pearson r | Spearman rho |
|---|---|---|---|
| overall | 0 | n/a | n/a |
| accepted | 0 | n/a | n/a |
| rejected | 0 | n/a | n/a |

| accept/reject discrimination | n | ROC AUC |
|---|---|---|
| LLM mean rating (anchor) | 0 | n/a |
| LLM accept-vote share (anchor) | 0 | n/a |
| human mean_score (reference) | 0 | n/a |

### 4. Distributions

| score | LLM n | LLM % | human n | human % |
|---|---|---|---|---|
| 1 | 0 | 0.0% | 0 | n/a |
| 2 | 0 | 0.0% | 0 | n/a |
| 3 | 119 | 61.7% | 0 | n/a |
| 4 | 0 | 0.0% | 0 | n/a |
| 5 | 36 | 18.7% | 0 | n/a |
| 6 | 37 | 19.2% | 0 | n/a |
| 7 | 0 | 0.0% | 0 | n/a |
| 8 | 1 | 0.5% | 0 | n/a |
| 9 | 0 | 0.0% | 0 | n/a |
| 10 | 0 | 0.0% | 0 | n/a |

- LLM ratings: mean 4.0 SD 1.3 | human per-review scores: mean n/a SD n/a (scales differ by ICLR year)
- mean LLM rating (all scored versions/reps): accepted n/a (n=0) vs rejected n/a (n=0)

## Model: gemini-3.8-flash

- parsed_ok reviews: 195 | papers scored: 29 | paper-versions scored: 65

### 1. Version tracking

- papers with >= 2 scored versions: 28
- mean Kendall tau-b (all-tied excluded): 0.33 (n=15)
- tau>0: 32.1% | tau==0 (ties, incl. all-tied): 57.1% | tau<0: 10.7% | all-tied papers: 13
- tau hist: [-1.0,-0.5):3 [-0.5,+0.0):0 [+0.0,+0.5):4 [+0.5,+1.0]:8 all-tied(nan):13

Adjacent-pair direction (mean rating of v_{i+1} vs v_i):

| churn_class | n | P(>) | P(=) | P(<) |
|---|---|---|---|---|
| ALL | 36 | 33.3% | 44.4% | 22.2% |
| cosmetic | 11 | 0.0% | 72.7% | 27.3% |
| minor | 7 | 57.1% | 14.3% | 28.6% |
| major | 17 | 41.2% | 41.2% | 17.6% |
| unknown | 1 | 100.0% | 0.0% | 0.0% |

Cosmetic pairs (should be '='): 8/11 tied (72.7%).

### 1b. Inferential version-effect tests

- bootstrap 95% CI for the mean finite tau (papers resampled with replacement, 10,000 draws, seed 42): mean 0.33, CI [-0.07, +0.71]
- per-paper correlation, version position vs mean rating (>= 3 versions, constant-rating papers excluded): mean Pearson 0.42 (n=6, 83.3% positive) | mean Spearman 0.41 (n=6)
- Page trend test (first 3 versions of the 7 papers with >= 3 versions; H1 = ratings increase along versions): L=89, p=0.1173
- mixed-effects regression rating ~ version_position + (1|paper) on rep-level ratings (n=192, 28 papers): +0.23 rating points per revision step (95% CI [+0.04, +0.41], p=0.0158)
- verbosity control rating ~ version_position + log_words + (1|paper): n/a (needs >= 3 papers, >= 2 positions, and scrub-log n_words with variation)

### 1c. Where the effect lives

Adjacent-pair mean rating delta (v_to minus v_from) by stage transition (stages from the manifest):

| stage transition | n | mean delta | mean \|delta\| | share delta==0 |
|---|---|---|---|---|
| unknown -> unknown | 36 | +0.20 | 0.57 | 44.4% |

Same deltas by churn_class (true placebo = cosmetic text churn AND unchanged figures):

| subset | n | mean delta | mean \|delta\| | share delta==0 |
|---|---|---|---|---|
| cosmetic | 11 | -0.18 | 0.18 | 72.7% |
| minor | 7 | +0.24 | 0.43 | 14.3% |
| major | 17 | +0.37 | 0.84 | 41.2% |
| unknown | 1 | +1.33 | 1.33 | 0.0% |
| TRUE PLACEBO (cosmetic & figs unchanged) | 8 | -0.25 | 0.25 | 62.5% |
| cosmetic & figure changes | 3 | +0.00 | 0.00 | 100.0% |

- TRUE-PLACEBO noise floor (text and figures both unchanged): n=8, mean |delta| 0.25, share with delta==0 62.5%. Cosmetic pairs with figure changes are reported separately (n=3); cosmetic pairs without a figs_any_change flag are in neither subset.
- rating delta vs churn: Spearman rho(churn, |delta|) = 0.44 (p=0.0082, n=35 adjacent pairs)

### 2. Reliability

- test-retest SD across reps (per paper-version, n=65): mean 0.4 | median 0.0
- ICC(2,1) (targets=paper-versions n=65, raters=rep index k=3, 0 incomplete target(s) dropped): 0.89
- signal vs noise: between-version SD of per-version means (within paper, n=28) mean 0.3 vs test-retest SD mean 0.4 -> ratio 0.91

### 3. Human anchor

- anchor version = latest pre_submission/during_review (fallback v1); papers with scored anchor: 29 (0 skipped: anchor version not scored)

| subset | n | Pearson r | Spearman rho |
|---|---|---|---|
| overall | 0 | n/a | n/a |
| accepted | 0 | n/a | n/a |
| rejected | 0 | n/a | n/a |

| accept/reject discrimination | n | ROC AUC |
|---|---|---|
| LLM mean rating (anchor) | 0 | n/a |
| LLM accept-vote share (anchor) | 0 | n/a |
| human mean_score (reference) | 0 | n/a |

### 4. Distributions

| score | LLM n | LLM % | human n | human % |
|---|---|---|---|---|
| 1 | 9 | 4.6% | 0 | n/a |
| 2 | 0 | 0.0% | 0 | n/a |
| 3 | 34 | 17.4% | 0 | n/a |
| 4 | 0 | 0.0% | 0 | n/a |
| 5 | 38 | 19.5% | 0 | n/a |
| 6 | 77 | 39.5% | 0 | n/a |
| 7 | 0 | 0.0% | 0 | n/a |
| 8 | 37 | 19.0% | 0 | n/a |
| 9 | 0 | 0.0% | 0 | n/a |
| 10 | 0 | 0.0% | 0 | n/a |

- LLM ratings: mean 5.4 SD 1.8 | human per-review scores: mean n/a SD n/a (scales differ by ICLR year)
- mean LLM rating (all scored versions/reps): accepted n/a (n=0) vs rejected n/a (n=0)

## Model: gemini-3.8-flash-renone

- parsed_ok reviews: 195 | papers scored: 29 | paper-versions scored: 65

### 1. Version tracking

- papers with >= 2 scored versions: 28
- mean Kendall tau-b (all-tied excluded): 0.52 (n=15)
- tau>0: 42.9% | tau==0 (ties, incl. all-tied): 50.0% | tau<0: 7.1% | all-tied papers: 13
- tau hist: [-1.0,-0.5):2 [-0.5,+0.0):0 [+0.0,+0.5):4 [+0.5,+1.0]:9 all-tied(nan):13

Adjacent-pair direction (mean rating of v_{i+1} vs v_i):

| churn_class | n | P(>) | P(=) | P(<) |
|---|---|---|---|---|
| ALL | 36 | 38.9% | 44.4% | 16.7% |
| cosmetic | 11 | 18.2% | 81.8% | 0.0% |
| minor | 7 | 28.6% | 28.6% | 42.9% |
| major | 17 | 58.8% | 23.5% | 17.6% |
| unknown | 1 | 0.0% | 100.0% | 0.0% |

Cosmetic pairs (should be '='): 9/11 tied (81.8%).

### 1b. Inferential version-effect tests

- bootstrap 95% CI for the mean finite tau (papers resampled with replacement, 10,000 draws, seed 42): mean 0.52, CI [+0.15, +0.83]
- per-paper correlation, version position vs mean rating (>= 3 versions, constant-rating papers excluded): mean Pearson 0.43 (n=5, 100.0% positive) | mean Spearman 0.45 (n=5)
- Page trend test (first 3 versions of the 7 papers with >= 3 versions; H1 = ratings increase along versions): L=90, p=0.1173
- mixed-effects regression rating ~ version_position + (1|paper) on rep-level ratings (n=192, 28 papers): +0.28 rating points per revision step (95% CI [+0.07, +0.50], p=0.0104)
- verbosity control rating ~ version_position + log_words + (1|paper): n/a (needs >= 3 papers, >= 2 positions, and scrub-log n_words with variation)

### 1c. Where the effect lives

Adjacent-pair mean rating delta (v_to minus v_from) by stage transition (stages from the manifest):

| stage transition | n | mean delta | mean \|delta\| | share delta==0 |
|---|---|---|---|---|
| unknown -> unknown | 36 | +0.27 | 0.71 | 44.4% |

Same deltas by churn_class (true placebo = cosmetic text churn AND unchanged figures):

| subset | n | mean delta | mean \|delta\| | share delta==0 |
|---|---|---|---|---|
| cosmetic | 11 | +0.12 | 0.12 | 81.8% |
| minor | 7 | -0.10 | 0.57 | 28.6% |
| major | 17 | +0.53 | 1.20 | 23.5% |
| unknown | 1 | +0.00 | 0.00 | 100.0% |
| TRUE PLACEBO (cosmetic & figs unchanged) | 8 | +0.17 | 0.17 | 75.0% |
| cosmetic & figure changes | 3 | +0.00 | 0.00 | 100.0% |

- TRUE-PLACEBO noise floor (text and figures both unchanged): n=8, mean |delta| 0.17, share with delta==0 75.0%. Cosmetic pairs with figure changes are reported separately (n=3); cosmetic pairs without a figs_any_change flag are in neither subset.
- rating delta vs churn: Spearman rho(churn, |delta|) = 0.51 (p=0.0018, n=35 adjacent pairs)

### 2. Reliability

- test-retest SD across reps (per paper-version, n=65): mean 0.4 | median 0.0
- ICC(2,1) (targets=paper-versions n=65, raters=rep index k=3, 0 incomplete target(s) dropped): 0.87
- signal vs noise: between-version SD of per-version means (within paper, n=28) mean 0.4 vs test-retest SD mean 0.4 -> ratio 0.91

### 3. Human anchor

- anchor version = latest pre_submission/during_review (fallback v1); papers with scored anchor: 29 (0 skipped: anchor version not scored)

| subset | n | Pearson r | Spearman rho |
|---|---|---|---|
| overall | 0 | n/a | n/a |
| accepted | 0 | n/a | n/a |
| rejected | 0 | n/a | n/a |

| accept/reject discrimination | n | ROC AUC |
|---|---|---|
| LLM mean rating (anchor) | 0 | n/a |
| LLM accept-vote share (anchor) | 0 | n/a |
| human mean_score (reference) | 0 | n/a |

### 4. Distributions

| score | LLM n | LLM % | human n | human % |
|---|---|---|---|---|
| 1 | 14 | 7.2% | 0 | n/a |
| 2 | 0 | 0.0% | 0 | n/a |
| 3 | 37 | 19.0% | 0 | n/a |
| 4 | 0 | 0.0% | 0 | n/a |
| 5 | 39 | 20.0% | 0 | n/a |
| 6 | 72 | 36.9% | 0 | n/a |
| 7 | 0 | 0.0% | 0 | n/a |
| 8 | 33 | 16.9% | 0 | n/a |
| 9 | 0 | 0.0% | 0 | n/a |
| 10 | 0 | 0.0% | 0 | n/a |

- LLM ratings: mean 5.2 SD 1.9 | human per-review scores: mean n/a SD n/a (scales differ by ICLR year)
- mean LLM rating (all scored versions/reps): accepted n/a (n=0) vs rejected n/a (n=0)

