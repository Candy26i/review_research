# Frozen-splice pilot — summary

- position-only LR baseline (test true): AUC **0.668**
- position+surface LR baseline (test true): AUC **0.697**

## run: ctx1_len512

| condition | AUC | 95% CI | n | pos |
|---|---|---|---|---|
| true | 0.653 | [0.622, 0.682] | 12933 | 394 |
| v1self | 0.637 | [0.608, 0.665] | 12933 | 394 |
| micro | 0.578 | [0.548, 0.607] | 12851 | 455 |
| ident | 0.431 | [0.401, 0.462] | 12779 | 475 |
| xdoc | 0.655 | [0.625, 0.684] | 12933 | 394 |

Slices of cond=true (AUC):
- change_ratio [0.15,0.3): 0.657 (pos=82)
- change_ratio [0.3,0.45): 0.670 (pos=140)
- change_ratio [0.45,0.61): 0.638 (pos=172)
- kind=para: 0.646 (pos=345)
- kind=caption: 0.594 (pos=26)
- kind=bullet: 0.778 (pos=23)
- by section bucket (novelty signature if intro/related >> method/experiment):
  - abstract: 0.495 (pos=17)
  - intro: 0.573 (pos=63)
  - related: 0.595 (pos=37)
  - method: 0.549 (pos=51)
  - experiment: 0.597 (pos=90)
  - discussion: 0.623 (pos=15)
  - conclusion: 0.549 (pos=32)
  - other: 0.652 (pos=78)

**delta = AUC_true − max(AUC_v1self, pos-LR) = 0.653 − 0.668 = -0.014, 95% CI [-0.040, +0.013] (162 papers)** -> DEAD at pilot scale: run the pre-registered escalations (CTX 100, train-size curve) before accepting; then pivot Q2 to restricted channels  (ident sanity check: 0.431 — code-path only, vacuous for text channels by construction)
Reminder: SCALE additionally requires the ctx0 run at delta >= 0.05 (ctx1-only signal = seam artifact).

## run: ctx1_half

| condition | AUC | 95% CI | n | pos |
|---|---|---|---|---|
| true | 0.612 | [0.581, 0.643] | 12933 | 394 |
| v1self | 0.606 | [0.573, 0.635] | 12933 | 394 |
| micro | 0.592 | [0.565, 0.622] | 12851 | 455 |
| ident | 0.419 | [0.393, 0.445] | 12779 | 475 |
| xdoc | 0.599 | [0.571, 0.626] | 12933 | 394 |

Slices of cond=true (AUC):
- change_ratio [0.15,0.3): 0.588 (pos=82)
- change_ratio [0.3,0.45): 0.643 (pos=140)
- change_ratio [0.45,0.61): 0.599 (pos=172)
- kind=para: 0.606 (pos=345)
- kind=caption: 0.627 (pos=26)
- kind=bullet: 0.653 (pos=23)
- by section bucket (novelty signature if intro/related >> method/experiment):
  - abstract: 0.526 (pos=17)
  - intro: 0.505 (pos=63)
  - related: 0.535 (pos=37)
  - method: 0.550 (pos=51)
  - experiment: 0.580 (pos=90)
  - discussion: 0.611 (pos=15)
  - conclusion: 0.551 (pos=32)
  - other: 0.584 (pos=78)

**delta = AUC_true − max(AUC_v1self, pos-LR) = 0.612 − 0.668 = -0.056, 95% CI [-0.079, -0.031] (162 papers)** -> DEAD at pilot scale: run the pre-registered escalations (CTX 100, train-size curve) before accepting; then pivot Q2 to restricted channels  (ident sanity check: 0.419 — code-path only, vacuous for text channels by construction)
Reminder: SCALE additionally requires the ctx0 run at delta >= 0.05 (ctx1-only signal = seam artifact).

## run: ctx1_quarter

| condition | AUC | 95% CI | n | pos |
|---|---|---|---|---|
| true | 0.539 | [0.508, 0.569] | 12933 | 394 |
| v1self | 0.552 | [0.519, 0.581] | 12933 | 394 |
| micro | 0.571 | [0.544, 0.598] | 12851 | 455 |
| ident | 0.408 | [0.381, 0.434] | 12779 | 475 |
| xdoc | 0.511 | [0.483, 0.539] | 12933 | 394 |

Slices of cond=true (AUC):
- change_ratio [0.15,0.3): 0.494 (pos=82)
- change_ratio [0.3,0.45): 0.555 (pos=140)
- change_ratio [0.45,0.61): 0.546 (pos=172)
- kind=para: 0.531 (pos=345)
- kind=caption: 0.533 (pos=26)
- kind=bullet: 0.615 (pos=23)
- by section bucket (novelty signature if intro/related >> method/experiment):
  - abstract: 0.524 (pos=17)
  - intro: 0.510 (pos=63)
  - related: 0.561 (pos=37)
  - method: 0.497 (pos=51)
  - experiment: 0.526 (pos=90)
  - discussion: 0.534 (pos=15)
  - conclusion: 0.516 (pos=32)
  - other: 0.495 (pos=78)

**delta = AUC_true − max(AUC_v1self, pos-LR) = 0.539 − 0.668 = -0.129, 95% CI [-0.159, -0.100] (162 papers)** -> DEAD at pilot scale: run the pre-registered escalations (CTX 100, train-size curve) before accepting; then pivot Q2 to restricted channels  (ident sanity check: 0.408 — code-path only, vacuous for text channels by construction)
Reminder: SCALE additionally requires the ctx0 run at delta >= 0.05 (ctx1-only signal = seam artifact).

## run: ctx1_half_ep4

| condition | AUC | 95% CI | n | pos |
|---|---|---|---|---|
| true | 0.537 | [0.505, 0.568] | 12933 | 394 |
| v1self | 0.551 | [0.519, 0.580] | 12933 | 394 |
| micro | 0.573 | [0.544, 0.599] | 12851 | 455 |
| ident | 0.414 | [0.386, 0.440] | 12779 | 475 |
| xdoc | 0.507 | [0.479, 0.535] | 12933 | 394 |

Slices of cond=true (AUC):
- change_ratio [0.15,0.3): 0.484 (pos=82)
- change_ratio [0.3,0.45): 0.555 (pos=140)
- change_ratio [0.45,0.61): 0.547 (pos=172)
- kind=para: 0.531 (pos=345)
- kind=caption: 0.514 (pos=26)
- kind=bullet: 0.592 (pos=23)
- by section bucket (novelty signature if intro/related >> method/experiment):
  - abstract: 0.524 (pos=17)
  - intro: 0.512 (pos=63)
  - related: 0.570 (pos=37)
  - method: 0.497 (pos=51)
  - experiment: 0.526 (pos=90)
  - discussion: 0.532 (pos=15)
  - conclusion: 0.529 (pos=32)
  - other: 0.486 (pos=78)

**delta = AUC_true − max(AUC_v1self, pos-LR) = 0.537 − 0.668 = -0.131, 95% CI [-0.160, -0.101] (162 papers)** -> DEAD at pilot scale: run the pre-registered escalations (CTX 100, train-size curve) before accepting; then pivot Q2 to restricted channels  (ident sanity check: 0.414 — code-path only, vacuous for text channels by construction)
Reminder: SCALE additionally requires the ctx0 run at delta >= 0.05 (ctx1-only signal = seam artifact).

## run: ctx1_quarter_ep8

| condition | AUC | 95% CI | n | pos |
|---|---|---|---|---|
| true | 0.608 | [0.575, 0.638] | 12933 | 394 |
| v1self | 0.614 | [0.584, 0.643] | 12933 | 394 |
| micro | 0.586 | [0.556, 0.615] | 12851 | 455 |
| ident | 0.400 | [0.374, 0.427] | 12779 | 475 |
| xdoc | 0.594 | [0.565, 0.623] | 12933 | 394 |

Slices of cond=true (AUC):
- change_ratio [0.15,0.3): 0.606 (pos=82)
- change_ratio [0.3,0.45): 0.618 (pos=140)
- change_ratio [0.45,0.61): 0.600 (pos=172)
- kind=para: 0.603 (pos=345)
- kind=caption: 0.586 (pos=26)
- kind=bullet: 0.673 (pos=23)
- by section bucket (novelty signature if intro/related >> method/experiment):
  - abstract: 0.522 (pos=17)
  - intro: 0.514 (pos=63)
  - related: 0.552 (pos=37)
  - method: 0.532 (pos=51)
  - experiment: 0.564 (pos=90)
  - discussion: 0.592 (pos=15)
  - conclusion: 0.532 (pos=32)
  - other: 0.583 (pos=78)

**delta = AUC_true − max(AUC_v1self, pos-LR) = 0.608 − 0.668 = -0.060, 95% CI [-0.086, -0.034] (162 papers)** -> DEAD at pilot scale: run the pre-registered escalations (CTX 100, train-size curve) before accepting; then pivot Q2 to restricted channels  (ident sanity check: 0.400 — code-path only, vacuous for text channels by construction)
Reminder: SCALE additionally requires the ctx0 run at delta >= 0.05 (ctx1-only signal = seam artifact).

## run: ctx0_half

| condition | AUC | 95% CI | n | pos |
|---|---|---|---|---|
| true | 0.649 | [0.619, 0.679] | 12933 | 394 |
| v1self | 0.633 | [0.603, 0.662] | 12933 | 394 |
| micro | 0.601 | [0.571, 0.630] | 12851 | 455 |
| ident | 0.425 | [0.398, 0.453] | 12779 | 475 |
| xdoc | 0.621 | [0.591, 0.651] | 12933 | 394 |

Slices of cond=true (AUC):
- change_ratio [0.15,0.3): 0.603 (pos=82)
- change_ratio [0.3,0.45): 0.685 (pos=140)
- change_ratio [0.45,0.61): 0.643 (pos=172)
- kind=para: 0.645 (pos=345)
- kind=caption: 0.622 (pos=26)
- kind=bullet: 0.802 (pos=23)
- by section bucket (novelty signature if intro/related >> method/experiment):
  - abstract: 0.555 (pos=17)
  - intro: 0.556 (pos=63)
  - related: 0.650 (pos=37)
  - method: 0.604 (pos=51)
  - experiment: 0.607 (pos=90)
  - discussion: 0.675 (pos=15)
  - conclusion: 0.593 (pos=32)
  - other: 0.624 (pos=78)

**delta = AUC_true − max(AUC_v1self, pos-LR) = 0.649 − 0.668 = -0.019, 95% CI [-0.044, +0.008] (162 papers)** -> DEAD at pilot scale: run the pre-registered escalations (CTX 100, train-size curve) before accepting; then pivot Q2 to restricted channels  (ident sanity check: 0.425 — code-path only, vacuous for text channels by construction)
Reminder: SCALE additionally requires the ctx0 run at delta >= 0.05 (ctx1-only signal = seam artifact).

## run: ctx1_seed1

| condition | AUC | 95% CI | n | pos |
|---|---|---|---|---|
| true | 0.650 | [0.621, 0.680] | 12933 | 394 |
| v1self | 0.637 | [0.608, 0.666] | 12933 | 394 |
| micro | 0.585 | [0.557, 0.615] | 12851 | 455 |
| ident | 0.425 | [0.396, 0.454] | 12779 | 475 |
| xdoc | 0.656 | [0.626, 0.685] | 12933 | 394 |

Slices of cond=true (AUC):
- change_ratio [0.15,0.3): 0.645 (pos=82)
- change_ratio [0.3,0.45): 0.676 (pos=140)
- change_ratio [0.45,0.61): 0.632 (pos=172)
- kind=para: 0.643 (pos=345)
- kind=caption: 0.623 (pos=26)
- kind=bullet: 0.733 (pos=23)
- by section bucket (novelty signature if intro/related >> method/experiment):
  - abstract: 0.634 (pos=17)
  - intro: 0.543 (pos=63)
  - related: 0.555 (pos=37)
  - method: 0.581 (pos=51)
  - experiment: 0.600 (pos=90)
  - discussion: 0.630 (pos=15)
  - conclusion: 0.575 (pos=32)
  - other: 0.632 (pos=78)

**delta = AUC_true − max(AUC_v1self, pos-LR) = 0.650 − 0.668 = -0.018, 95% CI [-0.042, +0.007] (162 papers)** -> DEAD at pilot scale: run the pre-registered escalations (CTX 100, train-size curve) before accepting; then pivot Q2 to restricted channels  (ident sanity check: 0.425 — code-path only, vacuous for text channels by construction)
Reminder: SCALE additionally requires the ctx0 run at delta >= 0.05 (ctx1-only signal = seam artifact).

## run: ctx1_seed2

| condition | AUC | 95% CI | n | pos |
|---|---|---|---|---|
| true | 0.592 | [0.562, 0.622] | 12933 | 394 |
| v1self | 0.590 | [0.560, 0.622] | 12933 | 394 |
| micro | 0.545 | [0.515, 0.576] | 12851 | 455 |
| ident | 0.420 | [0.392, 0.450] | 12779 | 475 |
| xdoc | 0.576 | [0.541, 0.609] | 12933 | 394 |

Slices of cond=true (AUC):
- change_ratio [0.15,0.3): 0.583 (pos=82)
- change_ratio [0.3,0.45): 0.592 (pos=140)
- change_ratio [0.45,0.61): 0.596 (pos=172)
- kind=para: 0.584 (pos=345)
- kind=caption: 0.612 (pos=26)
- kind=bullet: 0.655 (pos=23)
- by section bucket (novelty signature if intro/related >> method/experiment):
  - abstract: 0.513 (pos=17)
  - intro: 0.515 (pos=63)
  - related: 0.602 (pos=37)
  - method: 0.550 (pos=51)
  - experiment: 0.523 (pos=90)
  - discussion: 0.548 (pos=15)
  - conclusion: 0.547 (pos=32)
  - other: 0.477 (pos=78)

**delta = AUC_true − max(AUC_v1self, pos-LR) = 0.592 − 0.668 = -0.076, 95% CI [-0.104, -0.047] (162 papers)** -> DEAD at pilot scale: run the pre-registered escalations (CTX 100, train-size curve) before accepting; then pivot Q2 to restricted channels  (ident sanity check: 0.420 — code-path only, vacuous for text channels by construction)
Reminder: SCALE additionally requires the ctx0 run at delta >= 0.05 (ctx1-only signal = seam artifact).

## run: nosym_seed1

| condition | AUC | 95% CI | n | pos |
|---|---|---|---|---|
| true | 0.603 | [0.572, 0.633] | 12933 | 394 |
| v1self | 0.614 | [0.585, 0.643] | 12933 | 394 |
| micro | 0.547 | [0.518, 0.575] | 12851 | 455 |
| ident | 0.444 | [0.417, 0.471] | 12779 | 475 |
| xdoc | 0.616 | [0.585, 0.647] | 12933 | 394 |

Slices of cond=true (AUC):
- change_ratio [0.15,0.3): 0.637 (pos=82)
- change_ratio [0.3,0.45): 0.607 (pos=140)
- change_ratio [0.45,0.61): 0.584 (pos=172)
- kind=para: 0.604 (pos=345)
- kind=caption: 0.612 (pos=26)
- kind=bullet: 0.520 (pos=23)
- by section bucket (novelty signature if intro/related >> method/experiment):
  - abstract: 0.533 (pos=17)
  - intro: 0.504 (pos=63)
  - related: 0.486 (pos=37)
  - method: 0.564 (pos=51)
  - experiment: 0.522 (pos=90)
  - discussion: 0.570 (pos=15)
  - conclusion: 0.518 (pos=32)
  - other: 0.573 (pos=78)

**delta = AUC_true − max(AUC_v1self, pos-LR) = 0.603 − 0.668 = -0.065, 95% CI [-0.094, -0.034] (162 papers)** -> DEAD at pilot scale: run the pre-registered escalations (CTX 100, train-size curve) before accepting; then pivot Q2 to restricted channels  (ident sanity check: 0.444 — code-path only, vacuous for text channels by construction)
Reminder: SCALE additionally requires the ctx0 run at delta >= 0.05 (ctx1-only signal = seam artifact).

## run: nosym_seed2_lr1e5

| condition | AUC | 95% CI | n | pos |
|---|---|---|---|---|
| true | 0.632 | [0.603, 0.660] | 12933 | 394 |
| v1self | 0.632 | [0.604, 0.661] | 12933 | 394 |
| micro | 0.569 | [0.538, 0.600] | 12851 | 455 |
| ident | 0.412 | [0.385, 0.439] | 12779 | 475 |
| xdoc | 0.629 | [0.600, 0.658] | 12933 | 394 |

Slices of cond=true (AUC):
- change_ratio [0.15,0.3): 0.633 (pos=82)
- change_ratio [0.3,0.45): 0.638 (pos=140)
- change_ratio [0.45,0.61): 0.627 (pos=172)
- kind=para: 0.626 (pos=345)
- kind=caption: 0.644 (pos=26)
- kind=bullet: 0.669 (pos=23)
- by section bucket (novelty signature if intro/related >> method/experiment):
  - abstract: 0.545 (pos=17)
  - intro: 0.526 (pos=63)
  - related: 0.555 (pos=37)
  - method: 0.552 (pos=51)
  - experiment: 0.575 (pos=90)
  - discussion: 0.602 (pos=15)
  - conclusion: 0.532 (pos=32)
  - other: 0.624 (pos=78)

**delta = AUC_true − max(AUC_v1self, pos-LR) = 0.632 − 0.668 = -0.036, 95% CI [-0.059, -0.011] (162 papers)** -> DEAD at pilot scale: run the pre-registered escalations (CTX 100, train-size curve) before accepting; then pivot Q2 to restricted channels  (ident sanity check: 0.412 — code-path only, vacuous for text channels by construction)
Reminder: SCALE additionally requires the ctx0 run at delta >= 0.05 (ctx1-only signal = seam artifact).
