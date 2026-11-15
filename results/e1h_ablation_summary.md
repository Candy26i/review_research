# E1-H ablation: which part of the harness request lowers the scores?

- log: `data/processed/e1h_ablation_log.jsonl` — 994 attempts logged, 994 distinct runs

## What each arm did

| arm | usable runs | tool use | failed runs | runs with a retry | mean rating | rating 3 | accept | mean output tokens | median reasoning chars | reasoning opens with "We need" | input read from cache | mean cost |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| bare API call | 284 | 0 | 0 | 4 | 5.007 | 26.4% | 42.3% | 6,200 | 20,834 | 100.0% (n=284) | 100.0% of runs | ¥0.027 |
| bare + system prompt | 141 | 0 | 1 | 1 | 4.723 | 31.9% | 31.9% | 8,069 | 26,021 | 61.7% (n=141) | 100.0% of runs | ¥0.035 |
| bare + tool definitions | 141 | 0 | 1 | 1 | 4.617 | 33.3% | 24.1% | 7,394 | 19,170 | 40.4% (n=141) | 100.0% of runs | ¥0.033 |
| bare + runtime-context block | 142 | 0 | 0 | 0 | 4.887 | 31.7% | 40.8% | 7,787 | 26,002 | 100.0% (n=142) | 100.0% of runs | ¥0.034 |
| bare + all three (replay) | 141 | 0 | 1 | 1 | 4.546 | 39.0% | 28.4% | 11,387 | 36,494 | 67.4% (n=141) | 100.0% of runs | ¥0.050 |
| real harness | 141 | 0 | 1 | 13 | 4.660 | 34.8% | 27.0% | 10,256 | n/a | 60.3% (n=141) | 100.0% of runs | ¥0.045 |

Reasoning length is not available for the real harness (its event stream caps reasoning text); the opening is. "Failed runs" are runs still without a result.

Reasons of failed attempts inside API-arm runs (the real harness logs its own): bare: connection error: ConnectionError x4; sys: connection error: ConnectionError x1; tools: connection error: ConnectionError x1; replay: connection error: ConnectionError x1

## Level difference against the concurrent bare call

Single parts (primary):

- bare + system prompt minus bare: **-0.304** (95% CI [-0.466, -0.145], SE 0.083), sign-flip p = 0.0005 | papers 50 (below 26, above 10, equal 14) | paper-versions 141
- bare + tool definitions minus bare: **-0.419** (95% CI [-0.598, -0.249], SE 0.089), sign-flip p < 0.0001 | papers 50 (below 30, above 8, equal 12) | paper-versions 141
- bare + runtime-context block minus bare: **-0.110** (95% CI [-0.262, +0.040], SE 0.078), sign-flip p = 0.1660 | papers 50 (below 20, above 15, equal 15) | paper-versions 142
- Holm-adjusted p across the single parts: sys 0.0011, tools < 0.001, ctx 0.1660

Full effect in this session (reference):

- bare + all three (replay) minus bare: **-0.444** (95% CI [-0.587, -0.310], SE 0.071), sign-flip p < 0.0001 | papers 50 (below 35, above 5, equal 10) | paper-versions 141
- real harness minus bare: **-0.312** (95% CI [-0.486, -0.132], SE 0.092), sign-flip p = 0.0022 | papers 50 (below 30, above 10, equal 10) | paper-versions 141
- E1-H, for comparison: harness minus direct -0.483 (95% CI [-0.615, -0.359]); -0.57 and -0.33 in its two run sessions

Parts against each other and against the replay (for the reading rule):

- sys minus tools: **+0.106** (95% CI [-0.058, +0.274], SE 0.085), sign-flip p = 0.2242 | papers 50 (below 10, above 18, equal 22) | paper-versions 140
- sys minus ctx: **-0.174** (95% CI [-0.366, +0.007], SE 0.095), sign-flip p = 0.0697 | papers 50 (below 21, above 10, equal 19) | paper-versions 141
- tools minus ctx: **-0.300** (95% CI [-0.510, -0.088], SE 0.107), sign-flip p = 0.0084 | papers 50 (below 25, above 9, equal 16) | paper-versions 141
- replay minus sys: **-0.137** (95% CI [-0.315, +0.025], SE 0.087), sign-flip p = 0.1252 | papers 50 (below 17, above 12, equal 21) | paper-versions 140
- replay minus tools: **-0.031** (95% CI [-0.223, +0.157], SE 0.097), sign-flip p = 0.7625 | papers 50 (below 16, above 15, equal 19) | paper-versions 140
- replay minus ctx: **-0.334** (95% CI [-0.498, -0.171], SE 0.085), sign-flip p = 0.0003 | papers 50 (below 26, above 5, equal 19) | paper-versions 141

## Content check: replay against the real harness

- replay minus real harness: **-0.132** (95% CI [-0.312, +0.041], SE 0.091), sign-flip p = 0.1530 | papers 50 (below 16, above 14, equal 20) | paper-versions 140

## Additivity

- replay effect minus the sum of the three single-part effects: **+0.425** (95% CI [+0.100, +0.758], SE 0.171), sign-flip p = 0.0163 | papers 50 (below 15, above 24, equal 11) | paper-versions 139
  (negative: the parts together lower the rating more than their sum; only all-or-nothing departures are detectable at this sample size)

## Sensitivity checks

- without runs that had a retry inside the run (20 dropped):
    - sys - bare: -0.289 [-0.449, -0.132] (50 papers)
    - tools - bare: -0.430 [-0.608, -0.261] (50 papers)
    - ctx - bare: -0.120 [-0.270, +0.030] (50 papers)
    - replay - bare: -0.459 [-0.601, -0.327] (50 papers)
    - dsh - bare: -0.322 [-0.523, -0.128] (50 papers)
    - replay - dsh: -0.117 [-0.340, +0.116] (50 papers)
- cache state: every arm read its input from the cache in at least 95% of its runs; no restriction needed

Each line gives its own standard error. Version slopes are not analysed.
