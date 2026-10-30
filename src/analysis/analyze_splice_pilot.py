"""Analyze frozen-splice pilot runs: per-condition AUC with paper-level
bootstrap CIs, a PAIRED paper-level bootstrap of the decision quantity delta
itself, position-only and surface-feature LR baselines, slices, and the
pre-registered verdict.

Decision quantity: delta = AUC_true(model) - max(AUC_v1self(model), AUC_posLR),
recomputed inside every bootstrap resample of test papers (all three AUCs on
the same resample), so the verdict is read off the delta CI:
  CI low  >= 0.05 and point >= 0.10 -> SCALE (genuine single-doc edit signal)
  CI high <  0.05                   -> DEAD at pilot scale (run the
                                       pre-registered escalations before accepting)
  otherwise                         -> INCONCLUSIVE / WEAK (fusion component only)
The headline gate is the ctx1 run; a SCALE verdict additionally requires the
ctx0 run's delta >= 0.05 (ctx1-only signal = seam artifact — see
docs/SPLICE_PILOT.md). ident is a code-path sanity check (vacuous for text
channels by construction); micro is the regime-E rung; xdoc is the regime-B
foreign-text ceiling that separates editedness from topical novelty.

Run from repo root (sklearn only, no GPU):
  python3 src/analysis/analyze_splice_pilot.py \
      --pred ctx1=results/splice_pilot/deberta_ctx1/predictions.jsonl
"""
from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

SECTION_BUCKETS = [
    ('abstract', r'abstract'), ('intro', r'introduction'),
    ('related', r'related|background'), ('method', r'method|approach|model'),
    ('experiment', r'experiment|setup|evaluation|result'),
    ('discussion', r'discussion|analysis|ablation'),
    ('conclusion', r'conclusion|limitation|future'), ('appendix', r'appendix'),
]
KINDS = ('para', 'caption', 'bullet')
CONDS = ('true', 'v1self', 'micro', 'ident', 'xdoc')


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def section_bucket(name: str) -> str:
    low = name.lower()
    for bucket, pat in SECTION_BUCKETS:
        if re.search(pat, low):
            return bucket
    return 'other'


def features(r: dict, surface: bool) -> list[float]:
    rel = r['para_idx'] / max(1, r['n_paras'] - 1)
    feats = [rel, rel * rel, r['para_idx'], r['n_paras'],
             r['words'], np.log1p(r['words'])]
    bucket = section_bucket(r['section'])
    feats += [1.0 if bucket == b else 0.0 for b, _ in SECTION_BUCKETS] + \
             [1.0 if bucket == 'other' else 0.0]
    feats += [1.0 if r['kind'] == k else 0.0 for k in KINDS]
    if surface:
        t = r['text']
        n_ch = max(1, len(t))
        n_w = max(1, len(t.split()))
        feats += [
            sum(c.isdigit() for c in t) / n_ch,
            sum(c in '.,;:()[]' for c in t) / n_ch,
            sum(c.isupper() for c in t) / n_ch,
            len(set(t.lower().split())) / n_w,
            float(np.mean([len(w) for w in t.split()])) if t.split() else 0.0,
            t.count('<cit.>') / n_w,
            t.count('<ref>') / n_w,
        ]
    return feats


def fit_lr_scores(train_rows, test_rows, surface: bool) -> list[float]:
    xtr = np.array([features(r, surface) for r in train_rows])
    xte = np.array([features(r, surface) for r in test_rows])
    ytr = np.array([r['label'] for r in train_rows])
    scaler = StandardScaler().fit(xtr)
    clf = LogisticRegression(max_iter=2000, class_weight='balanced')
    clf.fit(scaler.transform(xtr), ytr)
    return clf.predict_proba(scaler.transform(xte))[:, 1].tolist()


def by_paper_pairs(rows, scores=None):
    out = defaultdict(list)
    for k, r in enumerate(rows):
        s = scores[k] if scores is not None else r['score']
        out[r['paper']].append((r['label'], s))
    return out


def pooled_auc(bp: dict, papers) -> float:
    labels, scores = [], []
    for p in papers:
        for lab, sc in bp.get(p, ()):
            labels.append(lab)
            scores.append(sc)
    if 0 < sum(labels) < len(labels):
        return roc_auc_score(labels, scores)
    return float('nan')


def bootstrap_auc(bp: dict, n_boot: int, seed: int = 0):
    papers = sorted(bp)
    point = pooled_auc(bp, papers)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        sample = rng.choice(papers, size=len(papers), replace=True)
        a = pooled_auc(bp, sample)
        if not math.isnan(a):
            vals.append(a)
    lo, hi = (np.percentile(vals, [2.5, 97.5]) if vals else (float('nan'),) * 2)
    return point, lo, hi


def delta_bootstrap(bp_true: dict, bp_v1: dict, bp_lr: dict, n_boot: int,
                    seed: int = 0):
    """Paired paper-level bootstrap of
    delta = AUC_true - max(AUC_v1self, AUC_posLR): one paper resample per
    iteration, all three AUCs recomputed on it."""
    papers = sorted(set(bp_true) & set(bp_v1) & set(bp_lr))
    pt_true = pooled_auc(bp_true, papers)
    pt_conf = max(pooled_auc(bp_v1, papers), pooled_auc(bp_lr, papers))
    point = pt_true - pt_conf
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        sample = rng.choice(papers, size=len(papers), replace=True)
        at = pooled_auc(bp_true, sample)
        ac = max(pooled_auc(bp_v1, sample), pooled_auc(bp_lr, sample))
        if not (math.isnan(at) or math.isnan(ac)):
            vals.append(at - ac)
    lo, hi = (np.percentile(vals, [2.5, 97.5]) if vals else (float('nan'),) * 2)
    return point, pt_true, pt_conf, lo, hi, len(papers)


def slice_auc(rows, pred) -> float:
    labs = [r['label'] for r in rows]
    if 0 < sum(labs) < len(labs):
        return roc_auc_score(labs, [pred(r) for r in rows])
    return float('nan')


def main() -> None:
    ap = argparse.ArgumentParser(description='Frozen-splice pilot analysis.')
    ap.add_argument('--data', default='data/processed/splice_pilot')
    ap.add_argument('--pred', nargs='+', required=True,
                    help='name=path/to/predictions.jsonl (repeatable)')
    ap.add_argument('--bootstrap', type=int, default=1000)
    ap.add_argument('--out', default='results/splice_pilot/summary.md')
    args = ap.parse_args()

    data_dir = Path(args.data)
    train_true = [r for r in read_jsonl(data_dir / 'train.jsonl') if r['cond'] == 'true']
    test_rows = read_jsonl(data_dir / 'test.jsonl')
    test_true = [r for r in test_rows if r['cond'] == 'true']

    lines = ['# Frozen-splice pilot — summary', '']
    lr_pos_scores = fit_lr_scores(train_true, test_true, surface=False)
    lr_surf_scores = fit_lr_scores(train_true, test_true, surface=True)
    yte = [r['label'] for r in test_true]
    pos_auc = roc_auc_score(yte, lr_pos_scores)
    surf_auc = roc_auc_score(yte, lr_surf_scores)
    bp_lr = by_paper_pairs(test_true, lr_pos_scores)
    lines += [f'- position-only LR baseline (test true): AUC **{pos_auc:.3f}**',
              f'- position+surface LR baseline (test true): AUC **{surf_auc:.3f}**', '']

    for spec in args.pred:
        name, path = spec.split('=', 1)
        preds = read_jsonl(path)
        lines += [f'## run: {name}', '',
                  '| condition | AUC | 95% CI | n | pos |', '|---|---|---|---|---|']
        bps = {}
        for cond in CONDS:
            rows = [r for r in preds if r['cond'] == cond]
            if not rows:
                lines.append(f'| {cond} | — | missing from predictions | 0 | 0 |')
                continue
            bps[cond] = by_paper_pairs(rows)
            point, lo, hi = bootstrap_auc(bps[cond], args.bootstrap)
            lines.append(f'| {cond} | {point:.3f} | [{lo:.3f}, {hi:.3f}] | '
                         f'{len(rows)} | {sum(r["label"] for r in rows)} |')
        lines.append('')

        true_preds = [r for r in preds if r['cond'] == 'true']
        if true_preds:
            lines.append('Slices of cond=true (AUC):')
            for lo_r, hi_r in ((0.15, 0.30), (0.30, 0.45), (0.45, 0.61)):
                sub = [r for r in true_preds
                       if r['label'] == 0
                       or ((r.get('change_ratio') or 0) >= lo_r
                           and (r.get('change_ratio') or 1) < hi_r)]
                if sum(r['label'] for r in sub) >= 10:
                    a = slice_auc(sub, lambda r: r['score'])
                    lines.append(f'- change_ratio [{lo_r},{hi_r}): {a:.3f} '
                                 f'(pos={sum(r["label"] for r in sub)})')
            for kind in KINDS:
                sub = [r for r in true_preds if r['kind'] == kind]
                a = slice_auc(sub, lambda r: r['score'])
                if not math.isnan(a):
                    lines.append(f'- kind={kind}: {a:.3f} '
                                 f'(pos={sum(r["label"] for r in sub)})')
            # novelty signature probe: intro/related >> method/experiment would
            # mean the model keys on topical newness, not editedness
            lines.append('- by section bucket (novelty signature if '
                         'intro/related >> method/experiment):')
            for bucket in ('abstract', 'intro', 'related', 'method',
                           'experiment', 'discussion', 'conclusion', 'other'):
                sub = [r for r in true_preds if section_bucket(r['section']) == bucket]
                a = slice_auc(sub, lambda r: r['score'])
                n_pos = sum(r['label'] for r in sub)
                if not math.isnan(a) and n_pos >= 5:
                    lines.append(f'  - {bucket}: {a:.3f} (pos={n_pos})')
            lines.append('')

        if 'true' not in bps or 'v1self' not in bps:
            lines += ['**verdict skipped: predictions are missing the true '
                      'and/or v1self condition — incomplete run file.**', '']
            continue
        point, at, conf, lo, hi, n_papers = delta_bootstrap(
            bps['true'], bps['v1self'], bp_lr, args.bootstrap)
        verdict = ('SCALE: genuine single-doc edit signal'
                   if lo >= 0.05 and point >= 0.10
                   else 'DEAD at pilot scale: run the pre-registered escalations '
                        '(CTX 100, train-size curve) before accepting; then pivot '
                        'Q2 to restricted channels' if hi < 0.05
                   else 'INCONCLUSIVE/WEAK: fusion component only, or escalate')
        ident_note = ''
        if 'ident' in bps:
            ident_pt = pooled_auc(bps['ident'], sorted(bps['ident']))
            ident_note = (f'  (ident sanity check: {ident_pt:.3f} — code-path only, '
                          'vacuous for text channels by construction)')
        lines += [f'**delta = AUC_true − max(AUC_v1self, pos-LR) = {at:.3f} − '
                  f'{conf:.3f} = {point:+.3f}, 95% CI [{lo:+.3f}, {hi:+.3f}] '
                  f'({n_papers} papers)** -> {verdict}{ident_note}',
                  'Reminder: SCALE additionally requires the ctx0 run at '
                  'delta >= 0.05 (ctx1-only signal = seam artifact).', '']

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text('\n'.join(lines))
    print('\n'.join(lines))
    print(f'-> {out}')


if __name__ == '__main__':
    main()
