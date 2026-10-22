# Contamination split: version effect by recognition

- reviews log: `data/processed/reviews_log.jsonl` - 2124 rows, 2124 parsed_ok rating rows
- recognition log: `data/processed/recognition_log.jsonl` - 50 rows, 50 papers with a boolean `authors_any_correct` flag
- split `authors_any_correct`: 18 authors-recalled / 32 authors-not-recalled; papers missing from the recognition log -> 'unprobed' (reported, excluded from tests)
- manifest: `data/processed/pilot_manifest.jsonl` (50 papers)

## Model: deepseek-chat

- parsed_ok reviews: 426 | papers scored: 50 | split field: `authors_any_correct`

### 1. Group sizes and descriptives

| group | papers | papers >=2 versions | rep rows | mean rating | mean tau (n) | all-tied | pairs | P(>) | P(=) | P(<) |
|---|---|---|---|---|---|---|---|---|---|---|
| authors-recalled | 18 | 18 | 144 | 5.3 | 0.21 (13) | 5 | 30 | 36.7% | 36.7% | 26.7% |
| authors-not-recalled | 32 | 32 | 282 | 4.5 | 0.03 (26) | 6 | 62 | 40.3% | 25.8% | 33.9% |
| unprobed | 0 | 0 | 0 | n/a | n/a (0) | 0 | 0 | n/a | n/a | n/a |

### 2. Version-effect tests (rep-level MixedLM, probed papers with >= 2 scored versions)

- interaction model rating ~ pos * recog + (1|paper), recog=1 for authors-recalled: pos:recog = +0.02/step (95% CI [-0.20, +0.23], p=0.8684; n=426 rows, 50 papers)
  - baseline slope in the authors-not-recalled group (the 'pos' main effect): +0.11/step (95% CI [+0.01, +0.21], p=0.0316)
- single-group model, authors-recalled: rating ~ pos + (1|paper) = +0.13/step (95% CI [-0.03, +0.28], p=0.1026; n=144 rows, 18 papers)
- single-group model, authors-not-recalled: rating ~ pos + (1|paper) = +0.11/step (95% CI [+0.00, +0.22], p=0.0469; n=282 rows, 32 papers)

### 3. Human anchor: accept/reject AUC of anchor-version mean rating

| group | n decided | accepted/rejected | ROC AUC |
|---|---|---|---|
| authors-recalled | 18 | 12/6 | 0.79 |
| authors-not-recalled | 32 | 14/18 | 0.42 |
| unprobed | 0 | 0/0 | n/a (<2 papers) |

### 4. Verdict

The version effect within the authors-not-recalled group is +0.11/step (95% CI [+0.00, +0.22], p=0.0469), independently positive and significant. The pos:recog interaction is +0.02 (95% CI [-0.20, +0.23], p=0.8684), not significantly positive. Verdict: effect survives on authors-not-recalled papers.

## Model: deepseek-reasoner

- parsed_ok reviews: 426 | papers scored: 50 | split field: `authors_any_correct`

### 1. Group sizes and descriptives

| group | papers | papers >=2 versions | rep rows | mean rating | mean tau (n) | all-tied | pairs | P(>) | P(=) | P(<) |
|---|---|---|---|---|---|---|---|---|---|---|
| authors-recalled | 18 | 18 | 144 | 5.9 | 0.65 (12) | 6 | 30 | 46.7% | 40.0% | 13.3% |
| authors-not-recalled | 32 | 32 | 282 | 4.8 | 0.42 (29) | 3 | 62 | 51.6% | 19.4% | 29.0% |
| unprobed | 0 | 0 | 0 | n/a | n/a (0) | 0 | 0 | n/a | n/a | n/a |

### 2. Version-effect tests (rep-level MixedLM, probed papers with >= 2 scored versions)

- interaction model rating ~ pos * recog + (1|paper), recog=1 for authors-recalled: pos:recog = -0.03/step (95% CI [-0.20, +0.15], p=0.7499; n=426 rows, 50 papers)
  - baseline slope in the authors-not-recalled group (the 'pos' main effect): +0.16/step (95% CI [+0.08, +0.25], p=0.0001)
- single-group model, authors-recalled: rating ~ pos + (1|paper) = +0.13/step (95% CI [-0.00, +0.27], p=0.0536; n=144 rows, 18 papers)
- single-group model, authors-not-recalled: rating ~ pos + (1|paper) = +0.16/step (95% CI [+0.08, +0.25], p=0.0003; n=282 rows, 32 papers)

### 3. Human anchor: accept/reject AUC of anchor-version mean rating

| group | n decided | accepted/rejected | ROC AUC |
|---|---|---|---|
| authors-recalled | 18 | 12/6 | 0.83 |
| authors-not-recalled | 32 | 14/18 | 0.56 |
| unprobed | 0 | 0/0 | n/a (<2 papers) |

### 4. Verdict

The version effect within the authors-not-recalled group is +0.16/step (95% CI [+0.08, +0.25], p=0.0003), independently positive and significant. The pos:recog interaction is -0.03 (95% CI [-0.20, +0.15], p=0.7499), not significantly positive. Verdict: effect survives on authors-not-recalled papers.

## Model: gemini-3.8-flash

- parsed_ok reviews: 426 | papers scored: 50 | split field: `authors_any_correct`

### 1. Group sizes and descriptives

| group | papers | papers >=2 versions | rep rows | mean rating | mean tau (n) | all-tied | pairs | P(>) | P(=) | P(<) |
|---|---|---|---|---|---|---|---|---|---|---|
| authors-recalled | 18 | 18 | 144 | 6.8 | 0.74 (8) | 10 | 30 | 30.0% | 60.0% | 10.0% |
| authors-not-recalled | 32 | 32 | 282 | 5.5 | 0.51 (24) | 8 | 62 | 45.2% | 37.1% | 17.7% |
| unprobed | 0 | 0 | 0 | n/a | n/a (0) | 0 | 0 | n/a | n/a | n/a |

### 2. Version-effect tests (rep-level MixedLM, probed papers with >= 2 scored versions)

- interaction model rating ~ pos * recog + (1|paper), recog=1 for authors-recalled: pos:recog = -0.12/step (95% CI [-0.30, +0.05], p=0.1677; n=426 rows, 50 papers)
  - baseline slope in the authors-not-recalled group (the 'pos' main effect): +0.31/step (95% CI [+0.23, +0.40], p=<0.0001)
- single-group model, authors-recalled: rating ~ pos + (1|paper) = +0.19/step (95% CI [+0.06, +0.32], p=0.0040; n=144 rows, 18 papers)
- single-group model, authors-not-recalled: rating ~ pos + (1|paper) = +0.31/step (95% CI [+0.23, +0.40], p=<0.0001; n=282 rows, 32 papers)

### 3. Human anchor: accept/reject AUC of anchor-version mean rating

| group | n decided | accepted/rejected | ROC AUC |
|---|---|---|---|
| authors-recalled | 18 | 12/6 | 0.90 |
| authors-not-recalled | 32 | 14/18 | 0.53 |
| unprobed | 0 | 0/0 | n/a (<2 papers) |

### 4. Verdict

The version effect within the authors-not-recalled group is +0.31/step (95% CI [+0.23, +0.40], p=<0.0001), independently positive and significant. The pos:recog interaction is -0.12 (95% CI [-0.30, +0.05], p=0.1677), not significantly positive. Verdict: effect survives on authors-not-recalled papers.

## Model: gemini-3.8-flash+figs

- parsed_ok reviews: 420 | papers scored: 49 | split field: `authors_any_correct`

### 1. Group sizes and descriptives

| group | papers | papers >=2 versions | rep rows | mean rating | mean tau (n) | all-tied | pairs | P(>) | P(=) | P(<) |
|---|---|---|---|---|---|---|---|---|---|---|
| authors-recalled | 17 | 17 | 138 | 6.8 | 0.09 (11) | 6 | 29 | 27.6% | 48.3% | 24.1% |
| authors-not-recalled | 32 | 32 | 282 | 5.5 | 0.67 (23) | 9 | 62 | 41.9% | 48.4% | 9.7% |
| unprobed | 0 | 0 | 0 | n/a | n/a (0) | 0 | 0 | n/a | n/a | n/a |

### 2. Version-effect tests (rep-level MixedLM, probed papers with >= 2 scored versions)

- interaction model rating ~ pos * recog + (1|paper), recog=1 for authors-recalled: pos:recog = -0.23/step (95% CI [-0.40, -0.05], p=0.0118; n=420 rows, 49 papers)
  - baseline slope in the authors-not-recalled group (the 'pos' main effect): +0.28/step (95% CI [+0.19, +0.36], p=<0.0001)
- single-group model, authors-recalled: rating ~ pos + (1|paper) = +0.05/step (95% CI [-0.08, +0.19], p=0.4614; n=138 rows, 17 papers)
- single-group model, authors-not-recalled: rating ~ pos + (1|paper) = +0.28/step (95% CI [+0.19, +0.36], p=<0.0001; n=282 rows, 32 papers)

### 3. Human anchor: accept/reject AUC of anchor-version mean rating

| group | n decided | accepted/rejected | ROC AUC |
|---|---|---|---|
| authors-recalled | 17 | 11/6 | 0.89 |
| authors-not-recalled | 32 | 14/18 | 0.57 |
| unprobed | 0 | 0/0 | n/a (<2 papers) |

### 4. Verdict

The version effect within the authors-not-recalled group is +0.28/step (95% CI [+0.19, +0.36], p=<0.0001), independently positive and significant. The pos:recog interaction is -0.23 (95% CI [-0.40, -0.05], p=0.0118), not significantly positive. Verdict: effect survives on authors-not-recalled papers.

## Model: gemini-3.8-flash-renone

- parsed_ok reviews: 426 | papers scored: 50 | split field: `authors_any_correct`

### 1. Group sizes and descriptives

| group | papers | papers >=2 versions | rep rows | mean rating | mean tau (n) | all-tied | pairs | P(>) | P(=) | P(<) |
|---|---|---|---|---|---|---|---|---|---|---|
| authors-recalled | 18 | 18 | 144 | 6.8 | 0.11 (9) | 9 | 30 | 23.3% | 60.0% | 16.7% |
| authors-not-recalled | 32 | 32 | 282 | 5.6 | 0.58 (25) | 7 | 62 | 45.2% | 43.5% | 11.3% |
| unprobed | 0 | 0 | 0 | n/a | n/a (0) | 0 | 0 | n/a | n/a | n/a |

### 2. Version-effect tests (rep-level MixedLM, probed papers with >= 2 scored versions)

- interaction model rating ~ pos * recog + (1|paper), recog=1 for authors-recalled: pos:recog = -0.21/step (95% CI [-0.37, -0.05], p=0.0082; n=426 rows, 50 papers)
  - baseline slope in the authors-not-recalled group (the 'pos' main effect): +0.25/step (95% CI [+0.17, +0.32], p=<0.0001)
- single-group model, authors-recalled: rating ~ pos + (1|paper) = +0.03/step (95% CI [-0.10, +0.16], p=0.6139; n=144 rows, 18 papers)
- single-group model, authors-not-recalled: rating ~ pos + (1|paper) = +0.25/step (95% CI [+0.17, +0.32], p=<0.0001; n=282 rows, 32 papers)

### 3. Human anchor: accept/reject AUC of anchor-version mean rating

| group | n decided | accepted/rejected | ROC AUC |
|---|---|---|---|
| authors-recalled | 18 | 12/6 | 0.90 |
| authors-not-recalled | 32 | 14/18 | 0.52 |
| unprobed | 0 | 0/0 | n/a (<2 papers) |

### 4. Verdict

The version effect within the authors-not-recalled group is +0.25/step (95% CI [+0.17, +0.32], p=<0.0001), independently positive and significant. The pos:recog interaction is -0.21 (95% CI [-0.37, -0.05], p=0.0082), not significantly positive. Verdict: effect survives on authors-not-recalled papers.

