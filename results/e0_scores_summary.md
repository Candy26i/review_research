# E0: LLM review-score analysis

- log: `data/processed/reviews_log.jsonl` - 261 rows, 261 parsed_ok rating rows used
- manifest: `data/processed/pilot_manifest.jsonl` (50 papers) | diffs: `data/processed/version_diffs.jsonl`

## Model: deepseek-chat

- parsed_ok reviews: 87 | papers scored: 10 | paper-versions scored: 29

### 1. Version tracking

- papers with >= 2 scored versions: 10
- mean Kendall tau-b (all-tied excluded): -0.14 (n=9)
- tau>0: 40.0% | tau==0 (ties, incl. all-tied): 20.0% | tau<0: 40.0% | all-tied papers: 1
- tau hist: [-1.0,-0.5):4 [-0.5,+0.0):0 [+0.0,+0.5):3 [+0.5,+1.0]:2 all-tied(nan):1

Adjacent-pair direction (mean rating of v_{i+1} vs v_i):

| churn_class | n | P(>) | P(=) | P(<) |
|---|---|---|---|---|
| ALL | 19 | 36.8% | 15.8% | 47.4% |
| cosmetic | 9 | 11.1% | 33.3% | 55.6% |
| minor | 1 | 100.0% | 0.0% | 0.0% |
| major | 8 | 50.0% | 0.0% | 50.0% |
| unknown | 1 | 100.0% | 0.0% | 0.0% |

Cosmetic pairs (should be '='): 3/9 tied (33.3%).

### 2. Reliability

- test-retest SD across reps (per paper-version, n=29): mean 0.9 | median 1.2
- ICC(2,1) (targets=paper-versions n=29, raters=rep index k=3, 0 incomplete target(s) dropped): 0.26
- signal vs noise: between-version SD of per-version means (within paper, n=10) mean 0.6 vs test-retest SD mean 0.9 -> ratio 0.68

### 3. Human anchor

- anchor version = latest pre_submission/during_review (fallback v1); papers with scored anchor: 10 (0 skipped: anchor version not scored)

| subset | n | Pearson r | Spearman rho |
|---|---|---|---|
| overall | 10 | 0.52 | 0.74 |
| accepted | 6 | 0.84 | 0.86 |
| rejected | 4 | -0.54 | -0.26 |

| accept/reject discrimination | n | ROC AUC |
|---|---|---|
| LLM mean rating (anchor) | 10 | 0.79 |
| LLM accept-vote share (anchor) | 10 | 0.58 |
| human mean_score (reference) | 10 | 1.00 |

### 4. Distributions

| score | LLM n | LLM % | human n | human % |
|---|---|---|---|---|
| 1 | 0 | 0.0% | 0 | 0.0% |
| 2 | 0 | 0.0% | 0 | 0.0% |
| 3 | 43 | 49.4% | 17 | 8.9% |
| 4 | 0 | 0.0% | 0 | 0.0% |
| 5 | 30 | 34.5% | 44 | 23.0% |
| 6 | 14 | 16.1% | 88 | 46.1% |
| 7 | 0 | 0.0% | 0 | 0.0% |
| 8 | 0 | 0.0% | 39 | 20.4% |
| 9 | 0 | 0.0% | 0 | 0.0% |
| 10 | 0 | 0.0% | 3 | 1.6% |

- LLM ratings: mean 4.2 SD 1.2 | human per-review scores: mean 6.0 SD 1.5 (scales differ by ICLR year)
- mean LLM rating (all scored versions/reps): accepted 4.4 (n=48) vs rejected 3.9 (n=39)

## Model: gemini-3.8-flash

- parsed_ok reviews: 87 | papers scored: 10 | paper-versions scored: 29

### 1. Version tracking

- papers with >= 2 scored versions: 10
- mean Kendall tau-b (all-tied excluded): 0.64 (n=9)
- tau>0: 70.0% | tau==0 (ties, incl. all-tied): 20.0% | tau<0: 10.0% | all-tied papers: 1
- tau hist: [-1.0,-0.5):1 [-0.5,+0.0):0 [+0.0,+0.5):1 [+0.5,+1.0]:7 all-tied(nan):1

Adjacent-pair direction (mean rating of v_{i+1} vs v_i):

| churn_class | n | P(>) | P(=) | P(<) |
|---|---|---|---|---|
| ALL | 19 | 63.2% | 15.8% | 21.1% |
| cosmetic | 9 | 44.4% | 33.3% | 22.2% |
| minor | 1 | 0.0% | 0.0% | 100.0% |
| major | 8 | 87.5% | 0.0% | 12.5% |
| unknown | 1 | 100.0% | 0.0% | 0.0% |

Cosmetic pairs (should be '='): 3/9 tied (33.3%).

### 2. Reliability

- test-retest SD across reps (per paper-version, n=29): mean 0.5 | median 0.6
- ICC(2,1) (targets=paper-versions n=29, raters=rep index k=3, 0 incomplete target(s) dropped): 0.65
- signal vs noise: between-version SD of per-version means (within paper, n=10) mean 0.7 vs test-retest SD mean 0.5 -> ratio 1.36

### 3. Human anchor

- anchor version = latest pre_submission/during_review (fallback v1); papers with scored anchor: 10 (0 skipped: anchor version not scored)

| subset | n | Pearson r | Spearman rho |
|---|---|---|---|
| overall | 10 | 0.69 | 0.72 |
| accepted | 6 | 0.51 | 0.72 |
| rejected | 4 | 0.90 | 0.60 |

| accept/reject discrimination | n | ROC AUC |
|---|---|---|
| LLM mean rating (anchor) | 10 | 0.83 |
| LLM accept-vote share (anchor) | 10 | 0.88 |
| human mean_score (reference) | 10 | 1.00 |

### 4. Distributions

| score | LLM n | LLM % | human n | human % |
|---|---|---|---|---|
| 1 | 0 | 0.0% | 0 | 0.0% |
| 2 | 0 | 0.0% | 0 | 0.0% |
| 3 | 13 | 14.9% | 17 | 8.9% |
| 4 | 0 | 0.0% | 0 | 0.0% |
| 5 | 23 | 26.4% | 44 | 23.0% |
| 6 | 45 | 51.7% | 88 | 46.1% |
| 7 | 0 | 0.0% | 0 | 0.0% |
| 8 | 6 | 6.9% | 39 | 20.4% |
| 9 | 0 | 0.0% | 0 | 0.0% |
| 10 | 0 | 0.0% | 3 | 1.6% |

- LLM ratings: mean 5.4 SD 1.3 | human per-review scores: mean 6.0 SD 1.5 (scales differ by ICLR year)
- mean LLM rating (all scored versions/reps): accepted 5.8 (n=48) vs rejected 5.0 (n=39)

## Model: gemini-3.8-flash+figs

- parsed_ok reviews: 87 | papers scored: 10 | paper-versions scored: 29

### 1. Version tracking

- papers with >= 2 scored versions: 10
- mean Kendall tau-b (all-tied excluded): 0.44 (n=7)
- tau>0: 50.0% | tau==0 (ties, incl. all-tied): 30.0% | tau<0: 20.0% | all-tied papers: 3
- tau hist: [-1.0,-0.5):2 [-0.5,+0.0):0 [+0.0,+0.5):0 [+0.5,+1.0]:5 all-tied(nan):3

Adjacent-pair direction (mean rating of v_{i+1} vs v_i):

| churn_class | n | P(>) | P(=) | P(<) |
|---|---|---|---|---|
| ALL | 19 | 31.6% | 57.9% | 10.5% |
| cosmetic | 9 | 11.1% | 66.7% | 22.2% |
| minor | 1 | 0.0% | 100.0% | 0.0% |
| major | 8 | 62.5% | 37.5% | 0.0% |
| unknown | 1 | 0.0% | 100.0% | 0.0% |

Cosmetic pairs (should be '='): 6/9 tied (66.7%).

### 2. Reliability

- test-retest SD across reps (per paper-version, n=29): mean 0.4 | median 0.0
- ICC(2,1) (targets=paper-versions n=29, raters=rep index k=3, 0 incomplete target(s) dropped): 0.79
- signal vs noise: between-version SD of per-version means (within paper, n=10) mean 0.5 vs test-retest SD mean 0.4 -> ratio 1.54

### 3. Human anchor

- anchor version = latest pre_submission/during_review (fallback v1); papers with scored anchor: 10 (0 skipped: anchor version not scored)

| subset | n | Pearson r | Spearman rho |
|---|---|---|---|
| overall | 10 | 0.58 | 0.64 |
| accepted | 6 | 0.70 | 0.74 |
| rejected | 4 | 0.53 | 0.00 |

| accept/reject discrimination | n | ROC AUC |
|---|---|---|
| LLM mean rating (anchor) | 10 | 0.77 |
| LLM accept-vote share (anchor) | 10 | 0.73 |
| human mean_score (reference) | 10 | 1.00 |

### 4. Distributions

| score | LLM n | LLM % | human n | human % |
|---|---|---|---|---|
| 1 | 0 | 0.0% | 0 | 0.0% |
| 2 | 0 | 0.0% | 0 | 0.0% |
| 3 | 15 | 17.2% | 17 | 8.9% |
| 4 | 0 | 0.0% | 0 | 0.0% |
| 5 | 16 | 18.4% | 44 | 23.0% |
| 6 | 48 | 55.2% | 88 | 46.1% |
| 7 | 0 | 0.0% | 0 | 0.0% |
| 8 | 8 | 9.2% | 39 | 20.4% |
| 9 | 0 | 0.0% | 0 | 0.0% |
| 10 | 0 | 0.0% | 3 | 1.6% |

- LLM ratings: mean 5.5 SD 1.4 | human per-review scores: mean 6.0 SD 1.5 (scales differ by ICLR year)
- mean LLM rating (all scored versions/reps): accepted 5.9 (n=48) vs rejected 5.0 (n=39)

