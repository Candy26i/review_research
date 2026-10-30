# E0: LLM review-score analysis

- log: `data/processed/reviews_log.jsonl` - 2547 rows, 423 parsed_ok rating rows used (model filter: claude-opus-5-5)
- manifest: `data/processed/pilot_manifest.jsonl` (50 papers) | diffs: `data/processed/version_diffs.jsonl`

## Model: claude-opus-5-5

- parsed_ok reviews: 423 | papers scored: 50 | paper-versions scored: 141

### 1. Version tracking

- papers with >= 2 scored versions: 49
- mean Kendall tau-b (all-tied excluded): 0.67 (n=23)
- tau>0: 40.8% | tau==0 (ties, incl. all-tied): 53.1% | tau<0: 6.1% | all-tied papers: 26
- tau hist: [-1.0,-0.5):2 [-0.5,+0.0):1 [+0.0,+0.5):1 [+0.5,+1.0]:19 all-tied(nan):26

Adjacent-pair direction (mean rating of v_{i+1} vs v_i):

| churn_class | n | P(>) | P(=) | P(<) |
|---|---|---|---|---|
| ALL | 91 | 27.5% | 65.9% | 6.6% |
| cosmetic | 25 | 12.0% | 84.0% | 4.0% |
| minor | 10 | 20.0% | 80.0% | 0.0% |
| major | 55 | 34.5% | 56.4% | 9.1% |
| unknown | 1 | 100.0% | 0.0% | 0.0% |

Cosmetic pairs (should be '='): 21/25 tied (84.0%).

### 1b. Inferential version-effect tests

- bootstrap 95% CI for the mean finite tau (papers resampled with replacement, 10,000 draws, seed 42): mean 0.67, CI [+0.40, +0.89]
- per-paper correlation, version position vs mean rating (>= 3 versions, constant-rating papers excluded): mean Pearson 0.72 (n=11, 90.9% positive) | mean Spearman 0.72 (n=11)
- Page trend test (first 3 versions of the 25 papers with >= 3 versions; H1 = ratings increase along versions): L=314, p=0.0239
- mixed-effects regression rating ~ version_position + (1|paper) on rep-level ratings (n=420, 49 papers): +0.14 rating points per revision step (95% CI [+0.09, +0.18], p=0.0000)
- verbosity control rating ~ version_position + log_words + (1|paper) (n=420, 49 papers): version +0.05/step (95% CI [-0.00, +0.11], p=0.0736) | log_words +1.18 (95% CI [+0.72, +1.64], p=0.0000)
  - The version coefficient does NOT survive the length control (p=0.0736 >= 0.05 once log document length is adjusted for), consistent with verbosity bias.

### 1c. Where the effect lives

Adjacent-pair mean rating delta (v_to minus v_from) by stage transition (stages from the manifest):

| stage transition | n | mean delta | mean \|delta\| | share delta==0 |
|---|---|---|---|---|
| during_review -> post_decision | 36 | +0.18 | 0.27 | 66.7% |
| post_decision -> post_decision | 20 | +0.08 | 0.18 | 80.0% |
| pre_submission -> during_review | 13 | +0.18 | 0.28 | 53.8% |
| pre_submission -> post_decision | 9 | +0.26 | 0.41 | 44.4% |
| pre_submission -> pre_submission | 8 | +0.25 | 0.25 | 75.0% |
| during_review -> during_review | 5 | +0.20 | 0.20 | 60.0% |

Same deltas by churn_class (true placebo = cosmetic text churn AND unchanged figures):

| subset | n | mean delta | mean \|delta\| | share delta==0 |
|---|---|---|---|---|
| cosmetic | 25 | +0.09 | 0.15 | 84.0% |
| minor | 10 | +0.13 | 0.13 | 80.0% |
| major | 55 | +0.21 | 0.33 | 56.4% |
| unknown | 1 | +0.67 | 0.67 | 0.0% |
| TRUE PLACEBO (cosmetic & figs unchanged) | 17 | +0.08 | 0.16 | 88.2% |
| cosmetic & figure changes | 8 | +0.12 | 0.12 | 75.0% |

- TRUE-PLACEBO noise floor (text and figures both unchanged): n=17, mean |delta| 0.16, share with delta==0 88.2%. Cosmetic pairs with figure changes are reported separately (n=8); cosmetic pairs without a figs_any_change flag are in neither subset.
- rating delta vs churn: Spearman rho(churn, |delta|) = 0.35 (p=0.0009, n=90 adjacent pairs)

### 2. Reliability

- test-retest SD across reps (per paper-version, n=141): mean 0.2 | median 0.0
- ICC(2,1) (targets=paper-versions n=141, raters=rep index k=3, 0 incomplete target(s) dropped): 0.87
- signal vs noise: between-version SD of per-version means (within paper, n=49) mean 0.3 vs test-retest SD mean 0.2 -> ratio 1.44

### 3. Human anchor

- anchor version = latest pre_submission/during_review (fallback v1); papers with scored anchor: 49 (1 skipped: anchor version not scored)

| subset | n | Pearson r | Spearman rho |
|---|---|---|---|
| overall | 49 | 0.38 | 0.43 |
| accepted | 25 | 0.00 | 0.05 |
| rejected | 24 | 0.31 | 0.36 |

| accept/reject discrimination | n | ROC AUC |
|---|---|---|
| LLM mean rating (anchor) | 49 | 0.80 |
| LLM accept-vote share (anchor) | 49 | 0.79 |
| human mean_score (reference) | 49 | 0.88 |

### 4. Distributions

| score | LLM n | LLM % | human n | human % |
|---|---|---|---|---|
| 1 | 0 | 0.0% | 0 | 0.0% |
| 2 | 0 | 0.0% | 0 | 0.0% |
| 3 | 53 | 12.5% | 17 | 8.9% |
| 4 | 0 | 0.0% | 0 | 0.0% |
| 5 | 175 | 41.4% | 44 | 23.0% |
| 6 | 168 | 39.7% | 88 | 46.1% |
| 7 | 0 | 0.0% | 0 | 0.0% |
| 8 | 27 | 6.4% | 39 | 20.4% |
| 9 | 0 | 0.0% | 0 | 0.0% |
| 10 | 0 | 0.0% | 3 | 1.6% |

- LLM ratings: mean 5.3 SD 1.2 | human per-review scores: mean 6.0 SD 1.5 (scales differ by ICLR year)
- mean LLM rating (all scored versions/reps): accepted 5.9 (n=219) vs rejected 4.8 (n=204)

