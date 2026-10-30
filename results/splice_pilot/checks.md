# Splice pilot - post-hoc checks

## A. Slot baselines (LR trained on train/true) per test condition

| cond | n | pos | position-only | position+surface |
|---|---|---|---|---|
| true | 12933 | 394 | 0.668 | 0.697 |
| v1self | 12933 | 394 | 0.668 | 0.669 |
| micro | 12851 | 455 | 0.607 | 0.602 |
| ident | 12779 | 475 | 0.436 | 0.434 |
| xdoc | 12933 | 394 | 0.659 | 0.682 |

## B. Per-run diagnostics

| run | best epoch (of) | val by epoch | logit sd | Spearman(score, words) on true | AUC_true | AUC_v1self | paired true-v1self gap [95% CI] |
|---|---|---|---|---|---|---|---|
| ctx0 | 1 (2) | 0.615 / 0.668 | 0.6912 | +0.62 | 0.664 | 0.639 | +0.025 [+0.006, +0.043] |
| ctx0_half | 1 (2) | 0.576 / 0.633 | 0.6559 | +0.70 | 0.649 | 0.633 | +0.016 [-0.008, +0.040] |
| ctx1 | 1 (2) | 0.629 / 0.638 | 0.5262 | +0.38 | 0.651 | 0.637 | +0.014 [+0.000, +0.026] |
| ctx1_half | 0 (2) | 0.621 / 0.592 | 0.0159 | +0.63 | 0.612 | 0.606 | +0.007 [-0.015, +0.029] |
| ctx1_half_ep4 | 1 (4) | 0.573 / 0.577 / 0.500 / 0.459 | 0.0339 | +0.75 | 0.537 | 0.551 | -0.014 [-0.044, +0.018] |
| ctx1_len512 | 1 (2) | 0.638 / 0.663 | 0.7383 | +0.41 | 0.653 | 0.637 | +0.017 [+0.003, +0.030] |
| ctx1_quarter | 0 (2) | 0.582 / 0.575 | 0.0254 | +0.76 | 0.539 | 0.552 | -0.013 [-0.043, +0.018] |
| ctx1_quarter_ep8 | 0 (8) | 0.598 / 0.578 / 0.580 / 0.560 / 0.586 / 0.584 / 0.583 / 0.583 | 0.0358 | +0.65 | 0.608 | 0.614 | -0.007 [-0.026, +0.015] |
| ctx1_seed1 | 1 (2) | 0.628 / 0.643 | 0.6559 | +0.45 | 0.650 | 0.637 | +0.013 [+0.001, +0.026] |
| ctx1_seed2 | 1 (2) | 0.583 / 0.596 | 0.0006 | +0.39 | 0.592 | 0.590 | +0.002 [-0.013, +0.015] |
| nosym | 1 (2) | 0.443 / 0.479 | 0.0000 | +0.02 | 0.505 | 0.488 | +0.017 [-0.021, +0.052] |
| nosym_seed1 | 0 (2) | 0.581 / 0.484 | 0.0204 | +0.30 | 0.603 | 0.614 | -0.010 [-0.023, +0.003] |
| nosym_seed2_lr1e5 | 1 (2) | 0.571 / 0.632 | 0.6454 | +0.50 | 0.632 | 0.632 | +0.000 [-0.013, +0.014] |

## C. Fusion check on test/true (paper-grouped 5-fold CV, LR, balanced)

Out-of-fold AUC; "+score" adds the run logit as one extra feature.

| run | pos | pos+score | pos+surf | pos+surf+score |
|---|---|---|---|---|
| ctx0 | 0.670 | 0.682 | 0.695 | 0.695 |
| ctx0_half | 0.670 | 0.681 | 0.695 | 0.695 |
| ctx1 | 0.670 | 0.678 | 0.695 | 0.695 |
| ctx1_half | 0.670 | 0.670 | 0.695 | 0.694 |
| ctx1_half_ep4 | 0.670 | 0.680 | 0.695 | 0.704 |
| ctx1_len512 | 0.670 | 0.677 | 0.695 | 0.694 |
| ctx1_quarter | 0.670 | 0.679 | 0.695 | 0.703 |
| ctx1_quarter_ep8 | 0.670 | 0.670 | 0.695 | 0.694 |
| ctx1_seed1 | 0.670 | 0.677 | 0.695 | 0.695 |
| ctx1_seed2 | 0.670 | 0.670 | 0.695 | 0.694 |
| nosym | 0.670 | 0.670 | 0.695 | 0.695 |
| nosym_seed1 | 0.670 | 0.671 | 0.695 | 0.695 |
| nosym_seed2_lr1e5 | 0.670 | 0.672 | 0.695 | 0.694 |

## D. Paired paper-bootstrap of AUC_true between runs

| A | B | AUC_A - AUC_B [95% CI] |
|---|---|---|
| ctx0 | ctx0_half | +0.0145 [+0.0012, +0.0283] |
| ctx1 | ctx1_half | +0.0386 [+0.0169, +0.0584] |
| ctx1 | ctx1_seed1 | +0.0006 [-0.0101, +0.0111] |
| ctx1 | ctx1_seed2 | +0.0589 [+0.0408, +0.0777] |
| ctx1 | ctx1_len512 | -0.0026 [-0.0160, +0.0099] |
| ctx1 | nosym_seed2_lr1e5 | +0.0185 [+0.0048, +0.0320] |
| ctx1_half | ctx1_quarter | +0.0737 [+0.0524, +0.0966] |

## E. micro condition

Slot baselines on micro: position-only 0.607, position+surface 0.602 (model micro AUCs are in summary*.md).

Top-20 ctx1-scored micro positives, word-level diff vs the v1 paragraph at the same slot:

- score +0.602 | ratio 0.9461 | Abstract | para | 6 edit(s): delete [LLM] -> []; delete [RL] -> []; replace [8B,] -> [8B / 3.2 3B,]
- score +0.590 | ratio 0.9552 | Introduction | para | 2 edit(s): replace [also,] -> [moreover,]; replace [mainly focus on inference stage, lacking] -> [lack]
- score +0.589 | ratio 0.9739 | Abstract | para | 2 edit(s): replace [remain significant limitations,] -> [remains a crucial limitation,]; delete [significantly] -> []
- score +0.588 | ratio 0.9792 | Analyzing Cross-Lingual A | para | 2 edit(s): replace [**Task-specific] -> [Task-specific]; replace [alignment:.**] -> [alignment:]
- score +0.583 | ratio 0.9597 | Visual Region-Based Layer | para | 1 edit(s): replace [LVLMs.] -> [LVLMs, serving as an initial exploration into LVLM-specific ]
- score +0.580 | ratio 0.9612 | Introduction | para | 3 edit(s): replace [LLaVA-1.5-7B] -> [LLaVA-1.5-7B, LLaVA-1.5-13B <cit.>]; replace [LLaVA-1.5-13B <cit.>,] -> [Bunny-Phi3-mini-4B-V,]; insert [] -> [varying]
- score +0.579 | ratio 0.9647 | More results | para | 3 edit(s): delete [**qplFlexora can Reduce Overfitting.**] -> []; insert [] -> [the sencond]; replace [study <ref>.] -> [study.]
- score +0.576 | ratio 0.9737 | Introduction | para | 3 edit(s): delete [basic,] -> []; replace [curtailing] -> [in mitigating]; replace [supervised fine-tuning, reinforcement learning,] -> [SFT, RL,]
- score +0.574 | ratio 0.9709 | Taxonomy | bullet | 2 edit(s): replace [multi-modal] -> [multimodal]; insert [] -> [audio-,]
- score +0.572 | ratio 0.9809 | Abstract | para | 1 edit(s): delete [Our code is available at [URL].] -> []
- score +0.571 | ratio 0.9886 | Conclusion | para | 2 edit(s): insert [] -> [classification]; insert [] -> [research]
- score +0.571 | ratio 0.9931 | Introduction | para | 1 edit(s): replace [interpretability.] -> [readability.]
- score +0.568 | ratio 0.9857 | Introduction | para | 1 edit(s): replace [raking] -> [ranking]
- score +0.566 | ratio 0.9398 | Introduction | bullet | 3 edit(s): replace [four] -> [9]; replace [health-related] -> [tabular]; delete [healthcare] -> []
- score +0.565 | ratio 0.9641 | Conclusion | para | 4 edit(s): delete [V^2R-Bench,] -> []; replace [framework] -> [framework, V^2R-Bench,]; insert [] -> [LVLM]
- score +0.561 | ratio 0.9585 | Abstract | para | 4 edit(s): insert [] -> [search engines,]; replace [noisy] -> [a sample of]; replace [expansion.] -> [expansion without fine-tuning.]
- score +0.561 | ratio 0.9873 | Introduction | para | 1 edit(s): replace [Continuous] -> [Compressed]
- score +0.556 | ratio 0.9877 | Evaluation Framework | para | 1 edit(s): replace [Moreover, context] -> [Context]
- score +0.555 | ratio 0.9825 | Conclusion | para | 2 edit(s): insert [] -> [retrieval-based]; delete [retrieval-based] -> []
- score +0.554 | ratio 0.9438 | Abstract | para | 3 edit(s): insert [] -> [and Arena-Hard]; replace [Llama-3-Instruct-8B.] -> [Llama-3-Instruct-8B, with the introduced hyperparameter fixe]; replace [effectiveness.] -> [effectiveness [Code available at: [URL]].]
