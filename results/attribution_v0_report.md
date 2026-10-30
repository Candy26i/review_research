# Attribution dataset v0 — overnight build report (2026-10-30)

Edit-attribution dataset ("old paper, new ink"): classify an (old_span,
new_span) paragraph revision as **human** (real pre-ChatGPT arXiv revision) or
**ai** (LLM edit of the same papers' v1 paragraphs). Topic and era are
controlled by construction; labels are exact.

## Pipeline (all overnight, Mac)

| stage | ok | lost | note |
|---|---|---|---|
| download sources | 7,673 | 60 | of 7,733 manifest files |
| extract archives | 7,294 | 439 | 376 were PDF-only submissions (no LaTeX) |
| latex -> text | 7,292 | 2 | |
| scrub + audit | 7,292 | — | 2,871 papers with >=1 version |
| version diffs | 4,405 pairs | — | pre2022_version_diffs.jsonl |
| AI edits (deepseek-chat) | 10,344 | 944 rejected | QC rejects: unchanged / length-ratio; ~3.3M tokens, ~$2 |

## Dataset (data/processed/attribution_v0, zip in repo)

| split | rows | human | ai | papers |
|---|---|---|---|---|
| train | 64,481 | 56,136 | 8,345 | 2,274 |
| test | 15,651 | 13,666 | 1,985 | 550 |

Zero train/test paper overlap; zero cross-split text dupes (QC-verified).
Class balance ~87/13 — the trainer class-weights and reports AUC/F1/balanced
accuracy, never raw accuracy.

## Adversarial QC (4-probe workflow) and fixes applied

Verdict on the first build: **SHIP-WITH-NOTES** with two verified critical
artifacts. Both were fixed by a same-night rebuild:

1. **Magnitude hole (critical, fixed in builder).** The aligner excluded human
   pairs with similarity in (0.60, 0.92) while 66% of AI edits land there; a
   magnitude band rule alone scored 90.3% test accuracy (+9.2pp over
   majority). Fix: `align(..., full_range=True)` labels the full band.
   After rebuild: band rule = 87.4% vs 87.3% majority (**+0.1pp — dead**);
   human rows in the band: 42%.
2. **Generator surface tells (critical, fixed symmetrically).** deepseek
   output collapses newlines/nbsp, strips bullets/"Caption:", ASCII-folds
   unicode — newline retention alone separated classes at 0.70 AUC. Fix:
   whitespace collapsed + leading markers stripped in BOTH classes at build
   time. After rebuild: 0 rows with newline/nbsp or leading markers.
   Unicode folding left as a trainer ablation (`--fold-unicode`) because it
   also removes a genuine em-dash style tell.
3. Metadata label leak (ratio null iff ai) — ratio now computed for both
   classes; 42 dupe rows and 8 contested-label rows dropped (deepseek's edit
   occasionally coincides exactly with the real human revision — n=3 pairs).
4. Human-pair mismatch tail — `--min-ratio` raised 0.15 -> 0.30 (QC manual
   read found wrong pairs only below 0.3).

Remaining known ceiling: surface-only logistic (similarity+length features)
reached **0.916 test AUC on the pre-fix build**; the trainer computes this
baseline on the fixed data automatically — the DeBERTa headline is its
**margin over the surface baseline**, not its absolute AUC.

## Next: PC baseline (paste below)

```
git pull --rebase
cd data/processed && python -c "import zipfile; zipfile.ZipFile('attribution_v0.zip').extractall()" && cd ../..
python src/train/train_attribution.py --data data/processed/attribution_v0 --out results/attribution_v0/deberta
python src/train/train_attribution.py --data data/processed/attribution_v0 --out results/attribution_v0/deberta_nfkc --fold-unicode
```

Trainer reports: test AUC with paper-bootstrap 95% CI, F1(ai), balanced acc,
AUC within magnitude bands [0.3,0.6)/[0.6,0.92)/[0.92,1], per-style AUC,
and the surface-LR baseline to beat. ~70k pairs, seq 512: expect ~25-35 min
per run on the 4090.

Interpretation guide:
- DeBERTa >> surface baseline AND band AUCs hold up -> real stylistic signal;
  scale to Qwen SFT on H100s.
- DeBERTa ≈ surface baseline -> attribution signal is mostly magnitude+surface;
  rethink before training big.
- grammar/polish styles near 0.5 while rewrite/expand high -> detector only
  catches heavy edits; report per-style.
