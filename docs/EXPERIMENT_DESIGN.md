# Are LLM Reviewers Reliable? Evidence from arXiv Version Histories

> Working title ideas: *ReviewLadder* / *VersionJudge* / *Climbing the arXiv Ladder: Do LLM Review Scores Track Paper Revisions?*

## 1. Research question

arXiv preserves every revision of a paper (v1, v2, v3 …). Authors revise papers mostly to
improve them — after reviewer feedback, rebuttals, or rejection-and-resubmission. This gives
us a large, free, naturally-occurring set of *ordered quality pairs*: for the same paper,
a later version should (usually) be at least as good as an earlier one.

**If an LLM reviewer is reliable, its score for a paper should track this ordering.**
If it cannot even tell v3 from v1 of the *same* paper — a far easier task than comparing
two different papers — then its absolute scores on new submissions deserve little trust.

### Hypotheses

- **H1 (positive correlation):** LLM scores correlate positively with version index within a paper.
- **H2 (weak monotonicity):** the relationship is noisy and non-linear; LLMs frequently rank an
  earlier version above a later one (as in the slide example: v1=3.5, v2=4, v3=3).
- **H3 (noise dominates):** test–retest variance of the LLM score on the *same* version is of the
  same order as the between-version signal, i.e. the reviewer cannot resolve revision-sized
  quality differences.
- **H4 (superficiality):** score changes are driven as much by superficial edits (length, polish,
  buzzwords, added limitations section) as by substantive ones (new experiments, fixed proofs).

H1–H2 are the slide hypotheses; H3–H4 are what makes the paper a *reliability study* rather
than a correlation study, and they connect directly to the LLM-as-a-judge literature.

## 2. Novelty and positioning

A literature pass (Oct 2026) found **no existing study that scores real arXiv version
histories with LLM reviewers**. Closest related work, and how we differ:

| Work | What it does | Difference from us |
|---|---|---|
| LLM-REVal ([arXiv 2510.12367](https://arxiv.org/abs/2510.12367)) | Simulated review/revision loop; scores rise ~+0.3 after revision | Simulated revisions by LLMs; we use *real* author revisions with real timelines |
| When Your Reviewer is an LLM ([arXiv 2509.09912](https://arxiv.org/abs/2509.09912)) | GPT vs human scores on 1,441 ICLR/NeurIPS papers; score inflation; prompt injection | Single version per paper; no within-paper longitudinal signal |
| Pairwise comparisons for LLM peer review ([arXiv 2506.11343](https://arxiv.org/abs/2506.11343)) | Argues pairwise > absolute scoring for review | We use *versions of the same paper* as the pairs — a controlled testbed they lack |
| APRES ([arXiv 2603.03142](https://arxiv.org/abs/2603.03142)) | Re-runs the NeurIPS consistency experiment with LLM committees | Consistency across committees, not across revisions |
| Identifying reliable metrics for scientific text revision ([ACL 2025](https://arxiv.org/abs/2506.04772)) | LLM judges on revision quality of *text fragments* | Fragment-level, instruction-following focus; we do full papers + review scores |
| DeepReviewer-14B ([arXiv 2503.08569](https://arxiv.org/abs/2503.08569)), CycleReviewer ([zhu-minjun/Researcher](https://github.com/zhu-minjun/Researcher)) | Train specialized reviewer models | They build reviewers; we build the *reliability benchmark* they should be tested on |

**Our contribution:** (1) the first version-longitudinal reliability test for LLM reviewers;
(2) a reusable benchmark dataset of scrubbed, diffed, strata-labeled arXiv version chains
linked to OpenReview ground truth; (3) a decomposition of what moves LLM scores
(substance vs. surface).

## 3. Experiments

### E0 — Pilot & calibration (week 1–2)
~50 ICLR-matched papers, one model. Goals: fix the rubric (ICLR-style 1–10 + confidence +
strengths/weaknesses JSON), verify text extraction quality, measure score variance to
power the main study. Decide K (repeats) from observed variance.

### E1 — Independent scoring (monotonicity)
Each version of each paper scored independently (shuffled order, K≈5 repeats, blinded —
see §6 scrubbing). Primary analysis: within-paper Kendall τ between score and version index;
mixed-effects model `score ~ version_index + log(length) + churn + (1 + version_index | paper)`.

### E2 — Pairwise discrimination
Show two versions of the same paper (anonymized, random A/B order): *which is better, which
would be accepted?* Swap order to measure position bias. Report accuracy vs. version order
overall and per edit-stratum. Comparing E1 vs E2 tests whether pairwise rescues the reviewer
(ties into the pairwise-review line of work).

### E3 — Test–retest reliability
Same version, K repeats × {temperature, paraphrased prompts, models}. Report ICC and
within-paper SD, and compare against (a) the between-version score delta, and (b) human
inter-reviewer disagreement from OpenReview (same-paper reviewer score SD) and the NeurIPS
consistency experiments. **Headline figure candidate:** signal-to-noise ratio of LLM
reviewers on revision-sized quality differences, with the human SNR beside it.

### E4 — Confound controls (the referee-proofing experiments)
1. **Cosmetic placebo:** version pairs with near-zero semantic diff (typo/metadata-only).
   Expected score delta ≈ 0; anything else is noise sensitivity.
2. **Synthetic causal probes:** take a final version and (a) inject typos, (b) delete the
   main experiment section, (c) paraphrase the abstract with hype words, (d) add a
   boilerplate limitations section. Known-direction quality edits → does the score move
   correctly and proportionally? This is the only *causal* test in the design.
3. **Length control:** later versions are longer (verbosity bias is well documented for
   LLM judges). Include length in the regression and run a truncated-input condition.
4. **Contamination control:** (a) ask the model to name title/venue/decision from the
   scrubbed text — measure recognition rate; (b) compare pre- vs post-training-cutoff
   papers (include 2025–2026 submissions); (c) check whether famous accepted papers get
   inflated v1 scores.

### E5 — Human ground-truth anchoring
Two gold subsets where *human* score deltas exist for version pairs:
- **Rebuttal pairs:** ICLR pre- vs post-rebuttal revisions, with before/after reviewer
  scores (the Re² dataset recovers pre-rebuttal scores that OpenReview overwrites).
- **Resubmission pairs:** rejected at ICLR year Y → resubmitted and accepted year Y+1
  (matched by title/authors across years; both rounds have full human scores).
Correlate LLM Δscore with human Δscore. This upgrades the claim from "scores rise with
version" to "scores track *human-judged* improvement".

### E6 — (Optional, hot) Gameability
Which moves the score more per token changed: substantive edits (new experiments, proofs)
or reviewer-pleasing edits (confident framing, promised code release, extra citations)?
Optionally reproduce the hidden-prompt-injection vulnerability on our own pipeline
(cf. [arXiv 2509.09912](https://arxiv.org/abs/2509.09912)). High press value; keep scoped.

### Interpretability (slides' Option I, revised)
Attention maps on 20k-token inputs are hard to read and attention ≠ explanation. Stronger
and model-agnostic: **section occlusion** — drop abstract / related work / experiments and
measure score change, i.e. what the reviewer actually relies on. Plus: verify the evidence
the model cites in its review text against the paper (hallucinated-evidence rate). Keep
attention visualization as a qualitative case study on one open model (e.g. Llama-3.1-8B /
Qwen2.5-7B — note: "Llama 7b" in the slides is outdated).

### Fine-tuning (slides' Option II)
Defer to phase 2 / a second paper. It competes directly with DeepReviewer-14B and
CycleReviewer-70B and needs the benchmark anyway — build the benchmark first; it then
becomes the evaluation suite for any fine-tuned reviewer, including ours.

## 4. Dataset pipeline

```
ICLR ground truth ──┐
(berenslab/iclr-dataset:                 ┌─> strata: cosmetic / minor / major
 36k papers, scores, decisions)          │   (latexdiff / token churn %)
                    ├─> arXiv match ──> version chains (arXiv API) ──> per-version text
OpenReview API  ────┘    (~77% for       (LaTeX source or ar5iv HTML   (scrubbed, see §6)
 (Re² for pre-rebuttal    accepted;       → markdown)
  scores)                 lower for rejected)
```

1. **Seed:** [berenslab/iclr-dataset](https://github.com/berenslab/iclr-dataset) (ICLR 2017–2025,
   decisions + reviewer scores) + OpenReview API for 2026; arXiv matching by title.
2. **Versions:** arXiv API `/abs` metadata gives version list + dates. Keep ≥2 versions for
   pairwise, ≥3 for monotonicity. Align version dates with the review timeline (submission
   deadline, rebuttal window, decision date) to label each version pre-submission /
   mid-review / post-decision.
3. **Text:** prefer arXiv LaTeX source (`e-print`) → pandoc/ar5iv → markdown. Text-only
   first (cheap, scrubbable); PDF-multimodal as a later robustness axis.
4. **Diff:** latexdiff or token-level diff → churn %, sections touched → edit strata.
5. **Sampling:** stratified over {accepted, rejected} × {year, incl. post-cutoff} × {churn}.
   Rejected papers with later revisions are especially valuable (improvement signal without
   acceptance leakage).

**Known pitfalls** (from the slides' challenge list, confirmed by the literature):
- OpenReview does **not** preserve the originally-submitted PDF for recent ICLR/NeurIPS —
  the arXiv chain is often the *only* public record of the reviewed version. (This is also a
  selling point of the dataset.)
- Camera-ready versions leak the answer ("Accepted at ICLR 2026", acknowledgments,
  de-anonymized authors) — scrubbing is mandatory, and post-decision versions must be
  analyzed as a separate stratum.
- Do **not** assume the last version is the accepted one; use the OpenReview decision.

## 5. Metrics & statistics

- **Monotonicity:** per-paper Kendall τ; fraction of adjacent pairs ordered correctly.
- **Discrimination:** pairwise accuracy (E2), with position-bias-corrected variant.
- **Reliability:** ICC(2,k); within-condition SD; SNR = between-version variance / test–retest variance.
- **Validity:** correlation of LLM Δscore with human Δscore (E5); accept/reject AUC as a sanity check.
- **Inference:** mixed-effects ordinal regression (papers as random effects), cluster-bootstrap CIs.
  Pre-register hypotheses + analysis plan (OSF) before the main run — cheap and very good optics.

## 6. Scrubbing / blinding protocol

Strip from every version before scoring: title page author block, acknowledgments, funding,
venue/decision mentions (regex + LLM pass), arXiv watermarks and version/date stamps, links
to code repos that reveal identity. Verify with the E4 contamination probe. Store scrubbed
text with a content hash so every score is traceable to an exact input.

## 7. Models & cost

- **Scorers (API):** 2–3 frontier models (e.g. GPT-5.x, Claude, Gemini) + 1–2 open-weight
  (Llama-3.1-8B, Qwen2.5-7B/72B) — open models enable the interpretability study and make
  results reproducible.
- **Cost reality check** (slides said ~$750 for 1k papers × 3 versions via GPT-4o PDF):
  text-only changes the picture. 20-page paper ≈ 15–25k tokens → 1,000 papers × 3 versions
  ≈ 60M input tokens ≈ low hundreds of dollars on a frontier model, ~$10–20 on a mini-class
  model; batch APIs give ~50% off. Strategy: mini-class model for K-repeat reliability runs,
  frontier models for single-pass main results on the stratified core set. A 4090 covers the
  open-weight runs.

## 8. Roadmap

| Phase | Weeks | Deliverable |
|---|---|---|
| P0 pilot | 1–2 | rubric, extraction QA, variance estimate, power analysis |
| P1 data | 2–5 | version-chain dataset + strata + scrubbing, released as the benchmark |
| P2 main runs | 4–7 | E1–E3 across models |
| P3 controls | 6–9 | E4 placebo/causal probes, contamination audit |
| P4 gold anchor | 8–10 | E5 rebuttal + resubmission subsets |
| P5 write-up | 10–12 | paper + open dataset + leaderboard page |

## 9. Hot topics this plugs into (and venue angles)

1. **Conferences are deploying LLM reviewers *now*** — the [AAAI-26 AI review pilot](https://arxiv.org/abs/2604.13940)
   generated AI reviews for ~23k submissions ([OpenReview announcement](https://openreview.net/forum/bok|openreview_hosts_recordbreaking_aaai_2026_conference_with_pioneering_ai_review_system),
   [AAAI FAQ](https://aaai.org/wp-content/uploads/2025/08/FAQ-for-the-AI-Assisted-Peer-Review-Process-Pilot-Program.pdf)),
   and ICML 2026 relaxed its AI-review policy. "Can these systems resolve revision-sized
   quality differences?" is exactly the question program chairs need answered.
2. **LLM-as-a-judge reliability** — score inflation, verbosity bias, position bias,
   sycophancy; our version chains are a clean new instrument for this literature.
3. **Review gameability & prompt injection** — hidden-instruction attacks in submitted PDFs
   ([arXiv 2509.09912](https://arxiv.org/abs/2509.09912)); E6 extends this to "which edits game the reviewer".
4. **Metascience / peer-review reform** — NeurIPS consistency experiments, now re-run with
   LLMs ([APRES](https://arxiv.org/abs/2603.03142)); we add the longitudinal axis nobody has.
5. **Benchmark contamination** — our pre/post-cutoff design doubles as a contamination study.

**Target venues:** ACL/EMNLP (NLP + metascience fit), NeurIPS Datasets & Benchmarks (the
version-chain benchmark), ICLR; workshops: SDP @ ACL, peer-review/metascience workshops.
TMLR is a good fallback with fast turnaround.

## 10. Open decisions from the slides — recommendations

| Question (slide 9–10) | Recommendation |
|---|---|
| Which venue / range | ICLR 2019–2026: fully public reviews incl. rejected papers, existing dataset, one rubric → avoids the "不同会议标准不一样" problem |
| Which topic | ML/AI only for the main study (LLM domain competence is highest; single rubric); physics/math as a generalization appendix at most |
| PDF or text | Text (LaTeX→markdown) for the main study; PDF-vision as one robustness condition later |
| Assume last version accepted? | **No.** Use OpenReview decisions; treat post-decision (camera-ready) versions as a separate stratum because they leak acceptance |
| How many papers | Pilot 50 → main 500–1,000 stratified (power analysis from pilot decides) |
| Only accepted papers? | No — stratify accepted/rejected; rejected-with-revisions is the most informative slice |
