"""Post-hoc checks for the splice-pilot escalation write-up (CPU, sklearn only).

Reproduces every number quoted in results/splice_pilot/escalations_notes.md
that is not already in summary*.md:
  - LR baselines (position / position+surface) on EVERY test condition
  - per run: best-val epoch used, logit sd, Spearman(score, words),
    within-model paired true-v1self AUC gap with paper bootstrap CI
  - fusion check: paper-grouped 5-fold CV on test/true of
    position(+surface) features with vs without the model score
  - paired bootstrap of AUC_true between two runs (e.g. ctx0 vs ctx0_half)
  - micro: slot baseline on micro + word-diff of the top-20 micro positives

Usage (repo root):
  python src/analysis/splice_pilot_checks.py --out results/splice_pilot/checks.md
"""
from __future__ import annotations

import argparse
import difflib
import glob
import os
import re
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from analyze_splice_pilot import features, read_jsonl  # noqa: E402

CONDS = ['true', 'v1self', 'micro', 'ident', 'xdoc']


def key(r):
    return (r['paper'], r['cond'], r['para_idx'])


def paper_auc_boot(labels, s_a, s_b, papers, n_boot=1000, seed=0):
    """Paired paper-level bootstrap of AUC(s_a) - AUC(s_b)."""
    rng = np.random.default_rng(seed)
    labels, s_a, s_b, papers = map(np.asarray, (labels, s_a, s_b, papers))
    uniq = np.unique(papers)
    idx_by = {p: np.where(papers == p)[0] for p in uniq}
    point = roc_auc_score(labels, s_a) - roc_auc_score(labels, s_b)
    ds = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by[p] for p in pick])
        if labels[idx].min() == labels[idx].max():
            continue
        ds.append(roc_auc_score(labels[idx], s_a[idx]) - roc_auc_score(labels[idx], s_b[idx]))
    lo, hi = np.percentile(ds, [2.5, 97.5])
    return point, lo, hi


def best_epoch_from_log(path):
    txt = open(path, encoding='utf-8').read()
    vals = [float(m) for m in re.findall(r'=== epoch \d+: val AUC ([0-9.]+)', txt)]
    return (int(np.argmax(vals)), len(vals), vals) if vals else (None, 0, [])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data/processed/splice_pilot')
    ap.add_argument('--results', default='results/splice_pilot')
    ap.add_argument('--out', default='results/splice_pilot/checks.md')
    ap.add_argument('--n-boot', type=int, default=1000)
    args = ap.parse_args()
    L = []
    train = [r for r in read_jsonl(Path(args.data) / 'train.jsonl') if r['cond'] == 'true']
    test = read_jsonl(Path(args.data) / 'test.jsonl')
    by_cond = {c: [r for r in test if r['cond'] == c] for c in CONDS}

    # ---- A. LR baselines on every condition
    L += ['# Splice pilot - post-hoc checks', '',
          '## A. Slot baselines (LR trained on train/true) per test condition', '',
          '| cond | n | pos | position-only | position+surface |', '|---|---|---|---|---|']
    lr_scores = {}
    for surface in (False, True):
        xtr = np.array([features(r, surface) for r in train])
        ytr = [r['label'] for r in train]
        sc = StandardScaler().fit(xtr)
        clf = LogisticRegression(max_iter=2000, class_weight='balanced').fit(sc.transform(xtr), ytr)
        for c in CONDS:
            xte = np.array([features(r, surface) for r in by_cond[c]])
            lr_scores[(c, surface)] = clf.decision_function(sc.transform(xte))
    for c in CONDS:
        y = [r['label'] for r in by_cond[c]]
        L.append(f"| {c} | {len(y)} | {sum(y)} | {roc_auc_score(y, lr_scores[(c, False)]):.3f} | "
                 f"{roc_auc_score(y, lr_scores[(c, True)]):.3f} |")
    L.append('')

    # ---- B. per-run diagnostics
    runs = sorted(glob.glob(os.path.join(args.results, 'deberta_*')))
    preds = {}
    L += ['## B. Per-run diagnostics', '',
          '| run | best epoch (of) | val by epoch | logit sd | Spearman(score, words) on true | '
          'AUC_true | AUC_v1self | paired true-v1self gap [95% CI] |',
          '|---|---|---|---|---|---|---|---|']
    for d in runs:
        name = os.path.basename(d)[8:]
        rows = read_jsonl(os.path.join(d, 'predictions.jsonl'))
        preds[name] = {key(r): r['score'] for r in rows}
        log = os.path.join(args.results, f'train_{name}.log')
        be, ne, vals = best_epoch_from_log(log) if os.path.exists(log) else (None, 0, [])
        sc_all = np.array([r['score'] for r in rows])
        tr = [r for r in rows if r['cond'] == 'true']
        v1 = [r for r in rows if r['cond'] == 'v1self']
        rho = spearmanr([r['score'] for r in tr], [r['words'] for r in tr]).correlation
        rng = np.random.default_rng(0)
        papers = sorted({r['paper'] for r in tr})
        tr_by = {p: [r for r in tr if r['paper'] == p] for p in papers}
        v1_by = {p: [r for r in v1 if r['paper'] == p] for p in papers}

        def auc_of(sel, src):
            rr = [r for p in sel for r in src[p]]
            y = [r['label'] for r in rr]
            s = [r['score'] for r in rr]
            return roc_auc_score(y, s) if 0 < sum(y) < len(y) else np.nan

        a_t, a_v = auc_of(papers, tr_by), auc_of(papers, v1_by)
        gaps = []
        for _ in range(args.n_boot):
            pick = list(rng.choice(papers, size=len(papers), replace=True))
            g = auc_of(pick, tr_by) - auc_of(pick, v1_by)
            if not np.isnan(g):
                gaps.append(g)
        lo, hi = np.percentile(gaps, [2.5, 97.5])
        L.append(f"| {name} | {be} ({ne}) | {' / '.join(f'{v:.3f}' for v in vals)} | {sc_all.std():.4f} | "
                 f"{rho:+.2f} | {a_t:.3f} | {a_v:.3f} | {a_t - a_v:+.3f} [{lo:+.3f}, {hi:+.3f}] |")
    L.append('')

    # ---- C. fusion: does the model score add anything on top of the slot features?
    L += ['## C. Fusion check on test/true (paper-grouped 5-fold CV, LR, balanced)', '',
          'Out-of-fold AUC; "+score" adds the run logit as one extra feature.', '',
          '| run | pos | pos+score | pos+surf | pos+surf+score |', '|---|---|---|---|---|']
    tt = by_cond['true']
    y = np.array([r['label'] for r in tt])
    g = np.array([r['paper'] for r in tt])
    Xp = np.array([features(r, False) for r in tt])
    Xs = np.array([features(r, True) for r in tt])

    def cv_auc(X):
        oof = np.zeros(len(y))
        for tr_i, te_i in GroupKFold(n_splits=5).split(X, y, g):
            sc = StandardScaler().fit(X[tr_i])
            clf = LogisticRegression(max_iter=2000, class_weight='balanced').fit(sc.transform(X[tr_i]), y[tr_i])
            oof[te_i] = clf.decision_function(sc.transform(X[te_i]))
        return roc_auc_score(y, oof)

    base_p, base_s = cv_auc(Xp), cv_auc(Xs)
    for name, pm in preds.items():
        s = np.array([[pm[key(r)]] for r in tt])
        L.append(f"| {name} | {base_p:.3f} | {cv_auc(np.hstack([Xp, s])):.3f} | {base_s:.3f} | "
                 f"{cv_auc(np.hstack([Xs, s])):.3f} |")
    L.append('')

    # ---- D. paired run-vs-run bootstraps on AUC_true
    L += ['## D. Paired paper-bootstrap of AUC_true between runs', '',
          '| A | B | AUC_A - AUC_B [95% CI] |', '|---|---|---|']
    for a, b in [('ctx0', 'ctx0_half'), ('ctx1', 'ctx1_half'), ('ctx1', 'ctx1_seed1'), ('ctx1', 'ctx1_seed2'),
                 ('ctx1', 'ctx1_len512'), ('ctx1', 'nosym_seed2_lr1e5'), ('ctx1_half', 'ctx1_quarter')]:
        if a in preds and b in preds:
            sa = [preds[a][key(r)] for r in tt]
            sb = [preds[b][key(r)] for r in tt]
            pt, lo, hi = paper_auc_boot(y, sa, sb, g, args.n_boot)
            L.append(f"| {a} | {b} | {pt:+.4f} [{lo:+.4f}, {hi:+.4f}] |")
    L.append('')

    # ---- E. micro: top-20 highest-scored micro positives (ctx1), diffed against v1 text
    L += ['## E. micro condition', '']
    ym = [r['label'] for r in by_cond['micro']]
    L.append(f"Slot baselines on micro: position-only {roc_auc_score(ym, lr_scores[('micro', False)]):.3f}, "
             f"position+surface {roc_auc_score(ym, lr_scores[('micro', True)]):.3f} "
             "(model micro AUCs are in summary*.md).")
    v1text = {(r['paper'], r['para_idx']): r['text'] for r in by_cond['v1self']}
    if 'ctx1' in preds:
        mp = [r for r in by_cond['micro'] if r['label'] == 1]
        mp.sort(key=lambda r: -preds['ctx1'][key(r)])
        L += ['', 'Top-20 ctx1-scored micro positives, word-level diff vs the v1 paragraph at the same slot:', '']
        for r in mp[:20]:
            v1 = v1text.get((r['paper'], r['para_idx']), '')
            a, b = v1.split(), r['text'].split()
            sm = difflib.SequenceMatcher(a=a, b=b)
            ops = [(t, ' '.join(a[i1:i2])[:60], ' '.join(b[j1:j2])[:60])
                   for t, i1, i2, j1, j2 in sm.get_opcodes() if t != 'equal']
            L.append(f"- score {preds['ctx1'][key(r)]:+.3f} | ratio {r.get('change_ratio')} | "
                     f"{r['section'][:25]} | {r['kind']} | {len(ops)} edit(s): "
                     + '; '.join(f"{t} [{x}] -> [{y}]" for t, x, y in ops[:3]))
    Path(args.out).write_text('\n'.join(L) + '\n', encoding='utf-8')
    print('\n'.join(L))


if __name__ == '__main__':
    main()
