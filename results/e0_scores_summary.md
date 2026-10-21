# E0: LLM review-score analysis

- log: `data/processed/reviews_log.jsonl` - 87 rows, 87 parsed_ok rating rows used
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

