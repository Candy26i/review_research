# Frozen-splice pilot — summary

- position-only LR baseline (test true): AUC **0.668**
- position+surface LR baseline (test true): AUC **0.697**

## run: ctx1

| condition | AUC | 95% CI | n | pos |
|---|---|---|---|---|
| true | 0.651 | [0.620, 0.680] | 12933 | 394 |
| v1self | 0.637 | [0.608, 0.666] | 12933 | 394 |
| micro | 0.570 | [0.539, 0.601] | 12851 | 455 |
| ident | 0.428 | [0.398, 0.458] | 12779 | 475 |
| xdoc | 0.652 | [0.623, 0.682] | 12933 | 394 |

Slices of cond=true (AUC):
- change_ratio [0.15,0.3): 0.639 (pos=82)
- change_ratio [0.3,0.45): 0.672 (pos=140)
- change_ratio [0.45,0.61): 0.639 (pos=172)
- kind=para: 0.644 (pos=345)
- kind=caption: 0.630 (pos=26)
- kind=bullet: 0.723 (pos=23)
- by section bucket (novelty signature if intro/related >> method/experiment):
  - abstract: 0.523 (pos=17)
  - intro: 0.573 (pos=63)
  - related: 0.617 (pos=37)
  - method: 0.553 (pos=51)
  - experiment: 0.595 (pos=90)
  - discussion: 0.642 (pos=15)
  - conclusion: 0.561 (pos=32)
  - other: 0.639 (pos=78)

**delta = AUC_true − max(AUC_v1self, pos-LR) = 0.651 − 0.668 = -0.017, 95% CI [-0.042, +0.009] (162 papers)** -> DEAD at pilot scale: run the pre-registered escalations (CTX 100, train-size curve) before accepting; then pivot Q2 to restricted channels  (ident sanity check: 0.428 — code-path only, vacuous for text channels by construction)
Reminder: SCALE additionally requires the ctx0 run at delta >= 0.05 (ctx1-only signal = seam artifact).

## run: ctx0

| condition | AUC | 95% CI | n | pos |
|---|---|---|---|---|
| true | 0.664 | [0.633, 0.694] | 12933 | 394 |
| v1self | 0.639 | [0.610, 0.668] | 12933 | 394 |
| micro | 0.596 | [0.565, 0.626] | 12851 | 455 |
| ident | 0.429 | [0.401, 0.457] | 12779 | 475 |
| xdoc | 0.647 | [0.616, 0.677] | 12933 | 394 |

Slices of cond=true (AUC):
- change_ratio [0.15,0.3): 0.645 (pos=82)
- change_ratio [0.3,0.45): 0.680 (pos=140)
- change_ratio [0.45,0.61): 0.659 (pos=172)
- kind=para: 0.656 (pos=345)
- kind=caption: 0.650 (pos=26)
- kind=bullet: 0.796 (pos=23)
- by section bucket (novelty signature if intro/related >> method/experiment):
  - abstract: 0.602 (pos=17)
  - intro: 0.560 (pos=63)
  - related: 0.663 (pos=37)
  - method: 0.615 (pos=51)
  - experiment: 0.616 (pos=90)
  - discussion: 0.659 (pos=15)
  - conclusion: 0.636 (pos=32)
  - other: 0.630 (pos=78)

**delta = AUC_true − max(AUC_v1self, pos-LR) = 0.664 − 0.668 = -0.004, 95% CI [-0.030, +0.022] (162 papers)** -> DEAD at pilot scale: run the pre-registered escalations (CTX 100, train-size curve) before accepting; then pivot Q2 to restricted channels  (ident sanity check: 0.429 — code-path only, vacuous for text channels by construction)
Reminder: SCALE additionally requires the ctx0 run at delta >= 0.05 (ctx1-only signal = seam artifact).

## run: nosym

| condition | AUC | 95% CI | n | pos |
|---|---|---|---|---|
| true | 0.505 | [0.473, 0.533] | 12933 | 394 |
| v1self | 0.488 | [0.459, 0.515] | 12933 | 394 |
| micro | 0.509 | [0.482, 0.536] | 12851 | 455 |
| ident | 0.486 | [0.459, 0.512] | 12779 | 475 |
| xdoc | 0.496 | [0.467, 0.523] | 12933 | 394 |

Slices of cond=true (AUC):
- change_ratio [0.15,0.3): 0.473 (pos=82)
- change_ratio [0.3,0.45): 0.505 (pos=140)
- change_ratio [0.45,0.61): 0.521 (pos=172)
- kind=para: 0.504 (pos=345)
- kind=caption: 0.536 (pos=26)
- kind=bullet: 0.486 (pos=23)
- by section bucket (novelty signature if intro/related >> method/experiment):
  - abstract: 0.434 (pos=17)
  - intro: 0.496 (pos=63)
  - related: 0.600 (pos=37)
  - method: 0.533 (pos=51)
  - experiment: 0.530 (pos=90)
  - discussion: 0.496 (pos=15)
  - conclusion: 0.474 (pos=32)
  - other: 0.473 (pos=78)

**delta = AUC_true − max(AUC_v1self, pos-LR) = 0.505 − 0.668 = -0.162, 95% CI [-0.206, -0.120] (162 papers)** -> DEAD at pilot scale: run the pre-registered escalations (CTX 100, train-size curve) before accepting; then pivot Q2 to restricted channels  (ident sanity check: 0.486 — code-path only, vacuous for text channels by construction)
Reminder: SCALE additionally requires the ctx0 run at delta >= 0.05 (ctx1-only signal = seam artifact).
