# Frozen-splice pilot (Q2 feasibility probe)

**Question.** Given a single document, can a classifier localize real author
revisions — v2 paragraphs spliced into an otherwise-unchanged v1? This decides
whether "single-doc edit localization" deserves GPU-scale training, *before*
any H100 is committed.

**Design.** One version pair per paper (highest-churn adjacent pair, 0.02 ≤
churn ≤ 0.70), paragraphs aligned block-wise, pairs assigned globally by
descending word-level ratio with a moved-paragraph guard and a content-overlap
gate (post-review hardening: naive greedy pairing mislabeled ~17% of positives).
Five conditions:

| cond | what is spliced into v1 | measures |
|---|---|---|
| `true` | substantially rewritten v2 paragraphs (ratio 0.15–0.60), k≤3 | the task (regime C) |
| `micro` | near-identical v2 paragraphs (ratio ≥0.92) | regime-E rung; also the only (ambiguous) probe of extraction-drift artifacts |
| `ident` | v2 paragraphs byte-equal after whitespace collapse | code-path sanity check — **vacuous for text channels by construction** (positives are byte-identical replacements) |
| `v1self` | nothing (v1 untouched), positives = v1 slots of true splices | edit-proneness + position confound |
| `xdoc` | kind/length-matched rewritten-v2 paragraphs from **other** test papers, same slots | regime-B foreign-text ceiling; separates "editedness" from topical novelty |

Training uses only `true` docs (train split, paper-level 80/20, deduped by
arxiv_id across ACL/ICLR/fresh). The other conditions exist only for test
papers. All emitted text is whitespace-collapsed (line-wrap artifacts can't
differ between spliced and native blocks); adjacent splice slots are dropped
so no negative's context window contains spliced text.

**Pre-registered decision rule.**
`delta = AUC_true − max(AUC_v1self, AUC_position-only-LR)`, with a **paired
paper-level bootstrap CI** (all AUCs recomputed on each paper resample):

- delta CI low ≥ 0.05 **and** point ≥ 0.10 → SCALE: genuine single-doc edit
  signal → H100 training justified
- delta CI high < 0.05 → DEAD **at pilot scale** → run the two pre-registered
  escalations below; if still dead, pivot Q2 to restricted channels
  (LLM-insertion detection, citation anachronism, numeric inconsistency)
- otherwise → INCONCLUSIVE/WEAK: usable only as one channel of an
  evidence-fusion detector, or escalate

**Run hierarchy (pre-registered).** The ctx1 run is the headline gate, but a
SCALE verdict **additionally requires the ctx0 run's delta ≥ 0.05**. The model
sees paragraph + 50-word windows, and in spliced documents a v2 paragraph abuts
v1 context — a seam that does NOT exist in a genuinely revised document (there
the edited paragraph sits in its own coherent v2 context). `v1self` cannot
bound that channel (its docs contain no seams), so: ctx1 high / ctx0 ≈
baseline = seam detection, not editedness — do not scale on it.

**Escalations before accepting a null** (each is hours, not days):
1. rebuild with `CTX_WORDS = 100` (or train with `--max-len 512`) — rules out
   "the seam/coherence evidence needed more surrounding text";
2. train on 50% vs 100% of the 696 train papers — if AUC_true is still rising
   with data, the null is a scale artifact, not a construct failure.

## Step 1 — Mac (data factory, done once)

```bash
python3 src/process/build_splice_pilot.py
```

Outputs `data/processed/splice_pilot/{train,test}.jsonl + stats.json`.

## Step 2 — transfer to the 4090 box

Clone the repo on the PC, then copy the dataset over (it is gitignored):
`data/processed/splice_pilot/` → same path in the PC clone. Any transfer works
(scp / cloud drive / USB).

## Step 3 — PC (training, ~30–60 min per run on a 4090)

```bash
pip install torch transformers sentencepiece scikit-learn
```

```bash
python3 src/train/train_splice_tagger.py --data data/processed/splice_pilot --out results/splice_pilot/deberta_ctx1
```

Then the two ablations:

```bash
python3 src/train/train_splice_tagger.py --data data/processed/splice_pilot --out results/splice_pilot/deberta_ctx0 --context 0
```

```bash
python3 src/train/train_splice_tagger.py --data data/processed/splice_pilot --out results/splice_pilot/deberta_nosym --strip-symbols
```

- `--context 0`: paragraph alone (no seam visibility) — the editedness gate
- `--strip-symbols`: digits masked — removes the numeric/symbolic channel

The same script runs on the Mac (auto-selects MPS) if the PC is unavailable —
just slower (hours per run). **MPS caveat:** DeBERTa's backward pass is broken
on MPS (NaN from step 1; the script fails fast with a message) — on a Mac add
`--model roberta-base`. DeBERTa works normally on CUDA and CPU. Each run writes
`metrics.json` (per-condition AUC, visible immediately) and `predictions.jsonl`.

## Step 4 — analysis (either machine, sklearn only)

Copy `results/splice_pilot/*/predictions.jsonl` back if trained on the PC, then:

```bash
python3 src/analysis/analyze_splice_pilot.py --pred ctx1=results/splice_pilot/deberta_ctx1/predictions.jsonl ctx0=results/splice_pilot/deberta_ctx0/predictions.jsonl nosym=results/splice_pilot/deberta_nosym/predictions.jsonl
```

Writes `results/splice_pilot/summary.md`: per-condition AUC with paper-level
bootstrap CIs, the paired delta CI and verdict, LR baselines, and
change-ratio / kind / section-bucket slices.

## Interpretation cheat-sheet

- `true` high / `v1self` low / ctx0 confirms → real signal in the v2 text itself
- `true` ≈ `v1self` → the model only learned *where* edits tend to happen
- ctx1 high / ctx0 ≈ baseline → seam detection, not editedness — don't scale
- `true` ≈ `xdoc` → the model detects *foreign/novel* text generally, not
  revisions specifically; intro/related AUC ≫ method/experiment AUC in the
  section slice is the same novelty signature
- `micro` ≈ 0.5 → light copyedits carry no signal (expected, regime E); if
  `micro` is clearly above 0.5, manually diff the top-20 highest-scored micro
  positives to tell real micro-edit signal from extraction-drift artifact
- `ident` is a code-path sanity check only — its positives are byte-identical
  replacements, so no text/inference code path can separate them; it says
  nothing about other artifact channels. **≈0.5 is NOT guaranteed**: the ident
  document is byte-for-byte the `v1self` document, and its positives are a
  sample of the *unchanged-in-v2, byte-equal* paragraphs — a biased subset
  (shorter, later in the document, more captions/appendix) that any model
  with `v1self`-style edit-proneness signal scores low. Pilot: ident 0.43
  with v1self 0.64; the position/surface LR baselines alone give 0.44 on
  ident. Only ident ≫ 0.5 would indicate a code-path leak.
- `nosym` run ≪ default run → the signal was numeric/symbolic, not style —
  but first check that the nosym model actually trained (score std > 0, val
  AUC > 0.5): with digits masked the pilot's seed-0 run collapsed to a
  constant logit, which also yields ≈0.5 everywhere and proves nothing
