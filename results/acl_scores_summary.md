# E0: LLM review-score analysis

- log: `data/processed/acl_reviews_log.jsonl` - 10860 rows, 10860 parsed_ok rating rows used
- manifest: `data/processed/acl_manifest.jsonl` (1379 papers) | diffs: `data/processed/acl_version_diffs.jsonl`

## Model: deepseek-chat

- parsed_ok reviews: 10860 | papers scored: 1376 | paper-versions scored: 3620

### 1. Version tracking

- papers with >= 2 scored versions: 1375
- mean Kendall tau-b (all-tied excluded): 0.21 (n=957)
- tau>0: 41.7% | tau==0 (ties, incl. all-tied): 34.3% | tau<0: 24.1% | all-tied papers: 418
- tau hist: [-1.0,-0.5):266 [-0.5,+0.0):65 [+0.0,+0.5):147 [+0.5,+1.0]:479 all-tied(nan):418

Adjacent-pair direction (mean rating of v_{i+1} vs v_i):

| churn_class | n | P(>) | P(=) | P(<) |
|---|---|---|---|---|
| ALL | 2244 | 36.6% | 38.6% | 24.8% |
| cosmetic | 863 | 31.1% | 41.9% | 27.0% |
| minor | 464 | 36.2% | 38.1% | 25.6% |
| major | 917 | 42.0% | 35.8% | 22.2% |

Cosmetic pairs (should be '='): 362/863 tied (41.9%).

### 1b. Inferential version-effect tests

- bootstrap 95% CI for the mean finite tau (papers resampled with replacement, 10,000 draws, seed 42): mean 0.21, CI [+0.16, +0.26]
- per-paper correlation, version position vs mean rating (>= 3 versions, constant-rating papers excluded): mean Pearson 0.27 (n=431, 71.0% positive) | mean Spearman 0.27 (n=431)
- Page trend test (first 3 versions of the 560 papers with >= 3 versions; H1 = ratings increase along versions): L=6918, p=0.0000
- mixed-effects regression rating ~ version_position + (1|paper) on rep-level ratings (n=10857, 1375 papers): +0.10 rating points per revision step (95% CI [+0.08, +0.12], p=0.0000)
- verbosity control rating ~ version_position + log_words + (1|paper) (n=10857, 1375 papers): version +0.07/step (95% CI [+0.05, +0.09], p=0.0000) | log_words +0.55 (95% CI [+0.46, +0.63], p=0.0000)
  - The version coefficient SURVIVES the length control (+0.07/step, p=0.0000 after adjusting for log document length), so the version effect is not just verbosity bias.

### 1c. Where the effect lives

Adjacent-pair mean rating delta (v_to minus v_from) by stage transition (stages from the manifest):

| stage transition | n | mean delta | mean \|delta\| | share delta==0 |
|---|---|---|---|---|
| unknown -> unknown | 2244 | +0.12 | 0.52 | 38.6% |

Same deltas by churn_class (true placebo = cosmetic text churn AND unchanged figures):

| subset | n | mean delta | mean \|delta\| | share delta==0 |
|---|---|---|---|---|
| cosmetic | 863 | +0.05 | 0.48 | 41.9% |
| minor | 464 | +0.10 | 0.52 | 38.1% |
| major | 917 | +0.20 | 0.57 | 35.8% |
| TRUE PLACEBO (cosmetic & figs unchanged) | 615 | +0.05 | 0.47 | 43.6% |
| cosmetic & figure changes | 239 | +0.04 | 0.51 | 36.4% |

- TRUE-PLACEBO noise floor (text and figures both unchanged): n=615, mean |delta| 0.47, share with delta==0 43.6%. Cosmetic pairs with figure changes are reported separately (n=239); cosmetic pairs without a figs_any_change flag are in neither subset.
- rating delta vs churn: Spearman rho(churn, |delta|) = 0.06 (p=0.0052, n=2244 adjacent pairs)

### 2. Reliability

- test-retest SD across reps (per paper-version, n=3620): mean 0.6 | median 0.6
- ICC(2,1) (targets=paper-versions n=3620, raters=rep index k=3, 0 incomplete target(s) dropped): 0.45
- signal vs noise: between-version SD of per-version means (within paper, n=1375) mean 0.4 vs test-retest SD mean 0.6 -> ratio 0.67

### 3. Human anchor

- anchor version = latest pre_submission/during_review (fallback v1); papers with scored anchor: 1376 (0 skipped: anchor version not scored)

| subset | n | Pearson r | Spearman rho |
|---|---|---|---|
| overall | 0 | n/a | n/a |
| accepted | 0 | n/a | n/a |
| rejected | 0 | n/a | n/a |

| accept/reject discrimination | n | ROC AUC |
|---|---|---|
| LLM mean rating (anchor) | 1376 | n/a |
| LLM accept-vote share (anchor) | 1376 | n/a |
| human mean_score (reference) | 0 | n/a |

### 4. Distributions

| score | LLM n | LLM % | human n | human % |
|---|---|---|---|---|
| 1 | 54 | 0.5% | 0 | n/a |
| 2 | 0 | 0.0% | 0 | n/a |
| 3 | 6701 | 61.7% | 0 | n/a |
| 4 | 0 | 0.0% | 0 | n/a |
| 5 | 2832 | 26.1% | 0 | n/a |
| 6 | 1265 | 11.6% | 0 | n/a |
| 7 | 0 | 0.0% | 0 | n/a |
| 8 | 8 | 0.1% | 0 | n/a |
| 9 | 0 | 0.0% | 0 | n/a |
| 10 | 0 | 0.0% | 0 | n/a |

- LLM ratings: mean 3.9 SD 1.2 | human per-review scores: mean n/a SD n/a (scales differ by ICLR year)
- mean LLM rating (all scored versions/reps): accepted 3.9 (n=10860) vs rejected n/a (n=0)

