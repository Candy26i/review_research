# E1-H: version-score slope with and without DeepSeek Harness

- logs: `data/processed/e1h_reviews_log.jsonl`, `data/processed/e1h_sub_reviews_log.jsonl` — 1308 attempts logged, 1278 distinct runs
- A direct call: 426 of 426 planned runs usable | attempts 426 | failed attempts 0
- B0 harness: 426 of 426 planned runs usable | attempts 433 | failed attempts 7
- B1 harness + forced subagent review: 419 of 426 planned runs usable (clean runs only) | attempts 449 | process verified 426 | of those needed a reminder 2, wrote a review before the report 5 | process failures 0 (0.0% of runs) | failed attempts (technical) 23, runs still without a result 0

## PRIMARY: B0 harness  minus  A direct call

- paper-versions usable in both arms: 142 of 142 planned | replicates (direct, harness) per cell: (3, 3): 142
- papers with >= 2 such versions: 50
- pooled slope (rating points per revision step):
    - direct: +0.184 (95% CI [+0.071, +0.302])
    - harness: +0.116 (95% CI [-0.011, +0.247])
    - **difference: -0.068 (95% CI [-0.200, +0.048]), sign-flip p = 0.2750**
    - weight concentration: the 3 heaviest papers carry 45%; two-version papers carry 9%
- unweighted mean of per-paper slopes:
    - direct: +0.205 (95% CI [+0.048, +0.362])
    - harness: +0.136 (95% CI [-0.004, +0.282])
    - difference: -0.068 (95% CI [-0.281, +0.143]), paired t(49) = -0.63, p = 0.5322 (10 of 50 differences are exactly zero)
    - SD of the per-paper differences 0.768: with 50 papers this test detects a difference of about 0.30 points per step at 80% power; anything smaller would likely look like no effect
- two-version papers (n = 25): pooled difference -0.067, unweighted -0.067
- papers with 3+ versions (n = 25): pooled difference -0.068, unweighted -0.070
- mean rating: direct 5.003, harness 4.520; level difference -0.483 (95% CI [-0.615, -0.359]), sign-flip p < 0.0001

## secondary: B1 harness + forced subagent review  minus  A direct call

- paper-versions usable in both arms: 142 of 142 planned | replicates (direct, harness_sub) per cell: (3, 3): 135, (3, 2): 7
- papers with >= 2 such versions: 50
- pooled slope (rating points per revision step):
    - direct: +0.184 (95% CI [+0.071, +0.302])
    - harness_sub: +0.094 (95% CI [-0.008, +0.187])
    - **difference: -0.090 (95% CI [-0.244, +0.060]), sign-flip p = 0.2907**
    - weight concentration: the 3 heaviest papers carry 45%; two-version papers carry 9%
- unweighted mean of per-paper slopes:
    - direct: +0.205 (95% CI [+0.048, +0.362])
    - harness_sub: +0.201 (95% CI [+0.056, +0.355])
    - difference: -0.003 (95% CI [-0.195, +0.205]), paired t(49) = -0.03, p = 0.9736 (11 of 50 differences are exactly zero)
    - SD of the per-paper differences 0.728: with 50 papers this test detects a difference of about 0.29 points per step at 80% power; anything smaller would likely look like no effect
- two-version papers (n = 25): pooled difference +0.067, unweighted +0.067
- papers with 3+ versions (n = 25): pooled difference -0.105, unweighted -0.074
- mean rating: direct 5.003, harness_sub 4.246; level difference -0.757 (95% CI [-0.908, -0.615]), sign-flip p < 0.0001

## secondary: B1 harness + forced subagent review  minus  B0 harness

- paper-versions usable in both arms: 142 of 142 planned | replicates (harness, harness_sub) per cell: (3, 3): 135, (3, 2): 7
- papers with >= 2 such versions: 50
- pooled slope (rating points per revision step):
    - harness: +0.116 (95% CI [-0.011, +0.247])
    - harness_sub: +0.094 (95% CI [-0.008, +0.187])
    - **difference: -0.023 (95% CI [-0.165, +0.113]), sign-flip p = 0.8085**
    - weight concentration: the 3 heaviest papers carry 45%; two-version papers carry 9%
- unweighted mean of per-paper slopes:
    - harness: +0.136 (95% CI [-0.004, +0.282])
    - harness_sub: +0.201 (95% CI [+0.056, +0.355])
    - difference: +0.065 (95% CI [-0.093, +0.227]), paired t(49) = 0.78, p = 0.4365 (13 of 50 differences are exactly zero)
    - SD of the per-paper differences 0.585: with 50 papers this test detects a difference of about 0.23 points per step at 80% power; anything smaller would likely look like no effect
- two-version papers (n = 25): pooled difference +0.133, unweighted +0.133
- papers with 3+ versions (n = 25): pooled difference -0.038, unweighted -0.004
- mean rating: harness 4.520, harness_sub 4.246; level difference -0.274 (95% CI [-0.378, -0.168]), sign-flip p < 0.0001

## Replicate noise

- direct: mean SD across replicates of the same paper-version 0.595 (n = 142 cells)
- harness: mean SD across replicates of the same paper-version 0.581 (n = 142 cells)
- harness_sub: mean SD across replicates of the same paper-version 0.540 (n = 142 cells)

## What each arm did

- A direct call: 426 usable runs | mean tokens in 17,323 out 6,133 | mean cost ¥0.042 per run (list price) | usage unknown in 0 runs | median latency 37s | runs with a retry inside the run 8
- B0 harness: 426 usable runs | mean tokens in 19,505 out 10,785 | mean cost ¥0.069 per run (list price) | usage unknown in 0 runs | median latency 62s | runs with a retry inside the run 19 | multi-step runs 0 | runs with tool/agent events 0
- B1 harness + forced subagent review: 419 usable runs | mean tokens in 172,363 out 25,949 | mean cost ¥0.195 per run (list price) | usage unknown in 0 runs | median latency 133s | runs with a retry inside the run 70 | mean share of the paper passed to the subagent 100.0% | runs that started extra subagents 0 | runs that wrote a review before the report 0

A larger slope means later versions are scored higher; on its own it does not show that the scores are more accurate.
