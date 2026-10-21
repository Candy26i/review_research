# E0: LLM review-score analysis

- log: `data/processed/reviews_log.jsonl` - 933 rows, 933 parsed_ok rating rows used
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

### 1b. Inferential version-effect tests

- per-paper correlation, version position vs mean rating (>= 3 versions, constant-rating papers excluded): mean Pearson -0.13 (n=5, 40.0% positive) | mean Spearman -0.21 (n=5)
- Page trend test (first 3 versions of the 5 papers with >= 3 versions; H1 = ratings increase along versions): L=58, p=0.7816
- mixed-effects regression rating ~ version_position + (1|paper) on rep-level ratings (n=87, 10 papers): +0.01 rating points per revision step (95% CI [-0.20, +0.23], p=0.8950)

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

- per-paper correlation, version position vs mean rating (>= 3 versions, constant-rating papers excluded): mean Pearson 0.43 (n=19, 78.9% positive) | mean Spearman 0.41 (n=19)
- Page trend test (first 3 versions of the 25 papers with >= 3 versions; H1 = ratings increase along versions): L=312, p=0.0385
- mixed-effects regression rating ~ version_position + (1|paper) on rep-level ratings (n=426, 50 papers): +0.28 rating points per revision step (95% CI [+0.21, +0.36], p=0.0000)

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

- per-paper correlation, version position vs mean rating (>= 3 versions, constant-rating papers excluded): mean Pearson 0.42 (n=16, 75.0% positive) | mean Spearman 0.41 (n=16)
- Page trend test (first 3 versions of the 25 papers with >= 3 versions; H1 = ratings increase along versions): L=314, p=0.0239
- mixed-effects regression rating ~ version_position + (1|paper) on rep-level ratings (n=420, 49 papers): +0.22 rating points per revision step (95% CI [+0.15, +0.30], p=0.0000)

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

