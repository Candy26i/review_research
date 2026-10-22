# Contamination split: version effect by recognition

- reviews log: `data/processed/reviews_log.jsonl` - 2124 rows, 2124 parsed_ok rating rows
- recognition log: `data/processed/recognition_log.jsonl` - 50 rows, 50 papers with a boolean `recognized` flag
- split `recognized`: 38 recognized / 12 unrecognized; papers missing from the recognition log -> 'unprobed' (reported, excluded from tests)
- manifest: `data/processed/pilot_manifest.jsonl` (50 papers)

## Model: deepseek-chat

- parsed_ok reviews: 426 | papers scored: 50 | split field: `recognized`

### 1. Group sizes and descriptives

| group | papers | papers >=2 versions | rep rows | mean rating | mean tau (n) | all-tied | pairs | P(>) | P(=) | P(<) |
|---|---|---|---|---|---|---|---|---|---|---|
| recognized | 38 | 38 | 333 | 5.0 | 0.02 (30) | 8 | 73 | 35.6% | 31.5% | 32.9% |
| unrecognized | 12 | 12 | 93 | 4.0 | 0.32 (9) | 3 | 19 | 52.6% | 21.1% | 26.3% |
| unprobed | 0 | 0 | 0 | n/a | n/a (0) | 0 | 0 | n/a | n/a | n/a |

### 2. Version-effect tests (rep-level MixedLM, probed papers with >= 2 scored versions)

- interaction model rating ~ pos * recog + (1|paper), recog=1 for recognized: pos:recog = +0.08/step (95% CI [-0.15, +0.30], p=0.4979; n=426 rows, 50 papers)
  - baseline slope in the unrecognized group (the 'pos' main effect): +0.04/step (95% CI [-0.16, +0.24], p=0.6761)
- single-group model, recognized: rating ~ pos + (1|paper) = +0.12/step (95% CI [+0.02, +0.22], p=0.0158; n=333 rows, 38 papers)
- single-group model, unrecognized: rating ~ pos + (1|paper) = +0.05/step (95% CI [-0.17, +0.26], p=0.6697; n=93 rows, 12 papers)

### 3. Human anchor: accept/reject AUC of anchor-version mean rating

| group | n decided | accepted/rejected | ROC AUC |
|---|---|---|---|
| recognized | 38 | 21/17 | 0.64 |
| unrecognized | 12 | 5/7 | 0.29 |
| unprobed | 0 | 0/0 | n/a (<2 papers) |

### 4. Verdict

The version effect within the unrecognized group is +0.05/step (95% CI [-0.17, +0.26], p=0.6697), not significantly positive on its own. The pos:recog interaction is +0.08 (95% CI [-0.15, +0.30], p=0.4979), not significantly positive. Verdict: underpowered/indeterminate - unrecognized-group coefficient +0.05 (p=0.6697) is not significantly positive; interaction +0.08 (p=0.4979) is not significantly positive.

## Model: deepseek-reasoner

- parsed_ok reviews: 426 | papers scored: 50 | split field: `recognized`

### 1. Group sizes and descriptives

| group | papers | papers >=2 versions | rep rows | mean rating | mean tau (n) | all-tied | pairs | P(>) | P(=) | P(<) |
|---|---|---|---|---|---|---|---|---|---|---|
| recognized | 38 | 38 | 333 | 5.4 | 0.40 (30) | 8 | 73 | 46.6% | 27.4% | 26.0% |
| unrecognized | 12 | 12 | 93 | 4.2 | 0.74 (11) | 1 | 19 | 63.2% | 21.1% | 15.8% |
| unprobed | 0 | 0 | 0 | n/a | n/a (0) | 0 | 0 | n/a | n/a | n/a |

### 2. Version-effect tests (rep-level MixedLM, probed papers with >= 2 scored versions)

- interaction model rating ~ pos * recog + (1|paper), recog=1 for recognized: pos:recog = -0.05/step (95% CI [-0.23, +0.13], p=0.5887; n=426 rows, 50 papers)
  - baseline slope in the unrecognized group (the 'pos' main effect): +0.19/step (95% CI [+0.03, +0.36], p=0.0228)
- single-group model, recognized: rating ~ pos + (1|paper) = +0.14/step (95% CI [+0.06, +0.22], p=0.0007; n=333 rows, 38 papers)
- single-group model, unrecognized: rating ~ pos + (1|paper) = +0.19/step (95% CI [+0.02, +0.36], p=0.0257; n=93 rows, 12 papers)

### 3. Human anchor: accept/reject AUC of anchor-version mean rating

| group | n decided | accepted/rejected | ROC AUC |
|---|---|---|---|
| recognized | 38 | 21/17 | 0.76 |
| unrecognized | 12 | 5/7 | 0.43 |
| unprobed | 0 | 0/0 | n/a (<2 papers) |

### 4. Verdict

The version effect within the unrecognized group is +0.19/step (95% CI [+0.02, +0.36], p=0.0257), independently positive and significant. The pos:recog interaction is -0.05 (95% CI [-0.23, +0.13], p=0.5887), not significantly positive. Verdict: effect survives on unrecognized papers.

## Model: gemini-3.8-flash

- parsed_ok reviews: 426 | papers scored: 50 | split field: `recognized`

### 1. Group sizes and descriptives

| group | papers | papers >=2 versions | rep rows | mean rating | mean tau (n) | all-tied | pairs | P(>) | P(=) | P(<) |
|---|---|---|---|---|---|---|---|---|---|---|
| recognized | 38 | 38 | 333 | 6.3 | 0.49 (24) | 14 | 73 | 35.6% | 49.3% | 15.1% |
| unrecognized | 12 | 12 | 93 | 4.8 | 0.80 (8) | 4 | 19 | 57.9% | 26.3% | 15.8% |
| unprobed | 0 | 0 | 0 | n/a | n/a (0) | 0 | 0 | n/a | n/a | n/a |

### 2. Version-effect tests (rep-level MixedLM, probed papers with >= 2 scored versions)

- interaction model rating ~ pos * recog + (1|paper), recog=1 for recognized: pos:recog = -0.02/step (95% CI [-0.20, +0.16], p=0.8248; n=426 rows, 50 papers)
  - baseline slope in the unrecognized group (the 'pos' main effect): +0.30/step (95% CI [+0.13, +0.46], p=0.0004)
- single-group model, recognized: rating ~ pos + (1|paper) = +0.28/step (95% CI [+0.19, +0.36], p=<0.0001; n=333 rows, 38 papers)
- single-group model, unrecognized: rating ~ pos + (1|paper) = +0.29/step (95% CI [+0.14, +0.45], p=0.0002; n=93 rows, 12 papers)

### 3. Human anchor: accept/reject AUC of anchor-version mean rating

| group | n decided | accepted/rejected | ROC AUC |
|---|---|---|---|
| recognized | 38 | 21/17 | 0.81 |
| unrecognized | 12 | 5/7 | 0.29 |
| unprobed | 0 | 0/0 | n/a (<2 papers) |

### 4. Verdict

The version effect within the unrecognized group is +0.29/step (95% CI [+0.14, +0.45], p=0.0002), independently positive and significant. The pos:recog interaction is -0.02 (95% CI [-0.20, +0.16], p=0.8248), not significantly positive. Verdict: effect survives on unrecognized papers.

## Model: gemini-3.8-flash+figs

- parsed_ok reviews: 420 | papers scored: 49 | split field: `recognized`

### 1. Group sizes and descriptives

| group | papers | papers >=2 versions | rep rows | mean rating | mean tau (n) | all-tied | pairs | P(>) | P(=) | P(<) |
|---|---|---|---|---|---|---|---|---|---|---|
| recognized | 37 | 37 | 327 | 6.3 | 0.30 (25) | 12 | 72 | 31.9% | 50.0% | 18.1% |
| unrecognized | 12 | 12 | 93 | 4.6 | 0.98 (9) | 3 | 19 | 57.9% | 42.1% | 0.0% |
| unprobed | 0 | 0 | 0 | n/a | n/a (0) | 0 | 0 | n/a | n/a | n/a |

### 2. Version-effect tests (rep-level MixedLM, probed papers with >= 2 scored versions)

- interaction model rating ~ pos * recog + (1|paper), recog=1 for recognized: pos:recog = -0.16/step (95% CI [-0.34, +0.03], p=0.0943; n=420 rows, 49 papers)
  - baseline slope in the unrecognized group (the 'pos' main effect): +0.35/step (95% CI [+0.18, +0.51], p=<0.0001)
- single-group model, recognized: rating ~ pos + (1|paper) = +0.19/step (95% CI [+0.11, +0.27], p=<0.0001; n=327 rows, 37 papers)
- single-group model, unrecognized: rating ~ pos + (1|paper) = +0.35/step (95% CI [+0.15, +0.54], p=0.0004; n=93 rows, 12 papers)

### 3. Human anchor: accept/reject AUC of anchor-version mean rating

| group | n decided | accepted/rejected | ROC AUC |
|---|---|---|---|
| recognized | 37 | 20/17 | 0.84 |
| unrecognized | 12 | 5/7 | 0.26 |
| unprobed | 0 | 0/0 | n/a (<2 papers) |

### 4. Verdict

The version effect within the unrecognized group is +0.35/step (95% CI [+0.15, +0.54], p=0.0004), independently positive and significant. The pos:recog interaction is -0.16 (95% CI [-0.34, +0.03], p=0.0943), not significantly positive. Verdict: effect survives on unrecognized papers.

## Model: gemini-3.8-flash-renone

- parsed_ok reviews: 426 | papers scored: 50 | split field: `recognized`

### 1. Group sizes and descriptives

| group | papers | papers >=2 versions | rep rows | mean rating | mean tau (n) | all-tied | pairs | P(>) | P(=) | P(<) |
|---|---|---|---|---|---|---|---|---|---|---|
| recognized | 38 | 38 | 333 | 6.3 | 0.27 (25) | 13 | 73 | 30.1% | 56.2% | 13.7% |
| unrecognized | 12 | 12 | 93 | 4.8 | 0.96 (9) | 3 | 19 | 68.4% | 21.1% | 10.5% |
| unprobed | 0 | 0 | 0 | n/a | n/a (0) | 0 | 0 | n/a | n/a | n/a |

### 2. Version-effect tests (rep-level MixedLM, probed papers with >= 2 scored versions)

- interaction model rating ~ pos * recog + (1|paper), recog=1 for recognized: pos:recog = -0.35/step (95% CI [-0.51, -0.18], p=<0.0001; n=426 rows, 50 papers)
  - baseline slope in the unrecognized group (the 'pos' main effect): +0.47/step (95% CI [+0.33, +0.62], p=<0.0001)
- single-group model, recognized: rating ~ pos + (1|paper) = +0.13/step (95% CI [+0.06, +0.19], p=0.0002; n=333 rows, 38 papers)
- single-group model, unrecognized: rating ~ pos + (1|paper) = +0.47/step (95% CI [+0.29, +0.65], p=<0.0001; n=93 rows, 12 papers)

### 3. Human anchor: accept/reject AUC of anchor-version mean rating

| group | n decided | accepted/rejected | ROC AUC |
|---|---|---|---|
| recognized | 38 | 21/17 | 0.80 |
| unrecognized | 12 | 5/7 | 0.26 |
| unprobed | 0 | 0/0 | n/a (<2 papers) |

### 4. Verdict

The version effect within the unrecognized group is +0.47/step (95% CI [+0.29, +0.65], p=<0.0001), independently positive and significant. The pos:recog interaction is -0.35 (95% CI [-0.51, -0.18], p=<0.0001), not significantly positive. Verdict: effect survives on unrecognized papers.

