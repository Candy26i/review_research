"""Train the edit-attribution classifier: (old_text, new_text) -> human vs ai.

Cross-encoder over "old [SEP] new". Featurizes ONLY the two texts — never
ratio/style/model (the metadata would leak the label). Positive class = ai.

Built-in controls (QC requirements for attribution_v0):
  - class weighting for the ~87/13 human/ai imbalance; reports AUC, F1(ai) and
    balanced accuracy (threshold picked on val), never raw accuracy alone
  - surface-only logistic baseline (similarity + length features, no words
    read) computed on the same splits — the headline is DeBERTa's MARGIN over
    this, not its absolute AUC
  - magnitude-matched slices: test AUC within ratio bands, so a magnitude cue
    cannot masquerade as style
  - per-style AUC (each ai style vs all human)
  - --fold-unicode ablation (NFKC both texts; run with and without)

Portable: cuda > mps > cpu. On the 4090 box, from the repo root:
  python3 src/train/train_attribution.py --data data/processed/attribution_v0 \
      --out results/attribution_v0/deberta
Ablation:
  python3 src/train/train_attribution.py --fold-unicode \
      --out results/attribution_v0/deberta_nfkc
Cross-generator held-out eval (train once with --save-model, then):
  python3 src/train/train_attribution.py \
      --eval-only results/attribution_v0/deberta/model \
      --test-file data/processed/attribution_xgen/test.jsonl \
      --out results/attribution_v0/deberta_xgen
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
import unicodedata
from pathlib import Path


def read_jsonl(path):
    with open(path, encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def seed64(key: str) -> int:
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], 'big')


def auc_score(labels, scores) -> float:
    """Mann-Whitney AUC with tie handling; nan if one class is absent."""
    n = len(scores)
    order = sorted(range(n), key=lambda i: scores[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    pos_ranks = [r for r, y in zip(ranks, labels) if y == 1]
    n1 = len(pos_ranks)
    n0 = n - n1
    if n1 == 0 or n0 == 0:
        return float('nan')
    u = sum(pos_ranks) - n1 * (n1 + 1) / 2
    return u / (n1 * n0)


def paper_bootstrap_auc(rows, scores, iters=1000, seed=0) -> tuple[float, float]:
    """95% CI for AUC under paper-level resampling."""
    import numpy as np
    by_paper: dict[str, list[int]] = {}
    for k, r in enumerate(rows):
        by_paper.setdefault(r['paper'], []).append(k)
    papers = sorted(by_paper)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(iters):
        pick = rng.choice(len(papers), size=len(papers), replace=True)
        idx = [k for pi in pick for k in by_paper[papers[pi]]]
        a = auc_score([rows[k]['y'] for k in idx], [scores[k] for k in idx])
        if a == a:
            vals.append(a)
    if not vals:
        return float('nan'), float('nan')
    vals.sort()
    return vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals))]


def surface_features(r) -> list[float]:
    """Cheap non-lexical cues a lazy model could exploit. The baseline."""
    old, new = r['old_text'], r['new_text']
    wo, wn = old.split(), new.split()
    so, sn = set(w.lower() for w in wo), set(w.lower() for w in wn)
    jac = len(so & sn) / max(1, len(so | sn))
    lr = len(wn) / max(1, len(wo))
    def nonascii(s):
        return sum(ord(c) > 127 for c in s) / max(1, len(s))
    def bit(s, ch):
        return float(ch in s)
    return [r['ratio'], jac, lr, abs(math.log(max(lr, 1e-6))),
            nonascii(new) - nonascii(old),
            bit(new, '—') - bit(old, '—'),      # em dash
            bit(new, '“') - bit(old, '“'),      # curly quote
            float(len(wn)), float(len(wo))]


def count_features(r) -> list[float]:
    """Count deltas the generator prompts make class-conditional ("keep every
    number and citation marker"): QC 2026-10-30 found human edits change the
    <cit.>/<ref> count in 21% of rows vs 0.4% of ai rows. Included so the
    cheap baseline is a fair ceiling, not a strawman."""
    old, new = r['old_text'], r['new_text']
    def n(s, tok):
        return s.count(tok)
    def digits(s):
        return sum(c.isdigit() for c in s)
    return [len(new) / max(1, len(old)),
            float(n(new, '<cit.>') - n(old, '<cit.>')),
            float(n(new, '<ref>') - n(old, '<ref>')),
            float(digits(new) - digits(old)),
            float(n(new, '<cit.>') != n(old, '<cit.>') or n(new, '<ref>') != n(old, '<ref>'))]


def surface_baseline(train_rows, test_rows) -> dict:
    """Two cheap baselines on the same 9 surface + 5 count features:
      surface_lr   linear (the original, kept for continuity)
      surface_hgb  HistGradientBoosting — the honest cheap-feature ceiling.
    QC 2026-10-30: on attribution_v0 the linear model gives 0.75 but a tree
    on the same cues gives 0.90; the headline margin must be quoted against
    the tree, not the line."""
    try:
        from sklearn.ensemble import HistGradientBoostingClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        return {'note': 'scikit-learn not installed; baseline skipped'}
    if len({r['y'] for r in train_rows}) < 2 or len({r['y'] for r in test_rows}) < 2:
        return {'note': 'single-class split (debug --limit?); baseline skipped'}
    import numpy as np
    Xtr = np.array([surface_features(r) + count_features(r) for r in train_rows])
    Xte = np.array([surface_features(r) + count_features(r) for r in test_rows])
    ytr = [r['y'] for r in train_rows]
    yte = [r['y'] for r in test_rows]
    sc = StandardScaler().fit(Xtr)
    lr = LogisticRegression(max_iter=2000, class_weight='balanced')
    lr.fit(sc.transform(Xtr), ytr)
    s_lr = lr.predict_proba(sc.transform(Xte))[:, 1].tolist()
    hgb = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05,
                                         class_weight='balanced', random_state=0)
    hgb.fit(Xtr, ytr)
    s_hgb = hgb.predict_proba(Xte)[:, 1].tolist()
    return {'surface_lr': {'auc': auc_score(yte, s_lr), 'n_features': Xtr.shape[1]},
            'surface_hgb': {'auc': auc_score(yte, s_hgb), 'n_features': Xtr.shape[1]},
            'scores_hgb': s_hgb}


def surface_refit_cv(test_rows) -> dict:
    """Paper-grouped 5-fold CV HGB fit ON the test set itself. On a
    cross-generator --test-file the transferred surface_hgb only measures
    baseline *transfer*; the honest cheap ceiling for that set is a refit
    (xgen QC 2026-10-31) — quote the margin against this too."""
    try:
        from sklearn.ensemble import HistGradientBoostingClassifier
        from sklearn.model_selection import GroupKFold
    except ImportError:
        return {}
    import numpy as np
    ys = np.array([r['y'] for r in test_rows])
    papers = [r['paper'] for r in test_rows]
    if len(set(ys.tolist())) < 2 or len(set(papers)) < 5:
        return {}
    X = np.array([surface_features(r) + count_features(r) for r in test_rows])
    oof = np.zeros(len(ys))
    for tr_i, te_i in GroupKFold(n_splits=5).split(X, ys, papers):
        if len(set(ys[tr_i].tolist())) < 2:
            return {}
        hgb = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05,
                                             class_weight='balanced', random_state=0)
        hgb.fit(X[tr_i], ys[tr_i])
        oof[te_i] = hgb.predict_proba(X[te_i])[:, 1]
    return {'surface_hgb_refit_cv': {'auc': auc_score(ys.tolist(), oof.tolist()),
                                     'n_features': X.shape[1]}}


RATIO_BANDS = ((0.30, 0.60), (0.60, 0.92), (0.92, 1.01))


def fold(s: str) -> str:
    return unicodedata.normalize('NFKC', s)


def main() -> None:
    ap = argparse.ArgumentParser(description='Edit-attribution cross-encoder.')
    ap.add_argument('--data', default='data/processed/attribution_v0')
    ap.add_argument('--out', default='results/attribution_v0/run')
    ap.add_argument('--model', default='microsoft/deberta-v3-base')
    ap.add_argument('--fold-unicode', action='store_true',
                    help='NFKC-fold both texts (ablation: kills the AI '
                         'ASCII-normalization artifact AND the em-dash tell)')
    ap.add_argument('--epochs', type=int, default=2)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--max-len', type=int, default=512)
    ap.add_argument('--lr', type=float, default=2e-5)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--train-frac', type=float, default=1.0,
                    help='fraction of train papers (deterministic hash subsample)')
    ap.add_argument('--test-file', default='',
                    help='override <data>/test.jsonl (e.g. a cross-generator '
                         'held-out set); train/val/baseline still use --data')
    ap.add_argument('--eval-only', default='',
                    help='path to a dir saved by --save-model: skip training, '
                         'load those weights and just evaluate')
    ap.add_argument('--save-model', action='store_true',
                    help='save the best checkpoint to <out>/model for later '
                         '--eval-only runs (cross-generator tests)')
    ap.add_argument('--limit', type=int, default=0, help='debug: cap rows')
    args = ap.parse_args()

    import torch
    from torch.utils.data import DataLoader
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                              get_linear_schedule_with_warmup)

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        device, amp_dtype = 'cuda', (torch.bfloat16 if torch.cuda.is_bf16_supported()
                                     else torch.float16)
    elif getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
        device, amp_dtype = 'mps', None
    else:
        device, amp_dtype = 'cpu', None
    print(f'device: {device} (amp: {amp_dtype})')
    if device == 'mps' and 'deberta' in args.model.lower():
        print('!! WARNING: DeBERTa backward is broken on MPS (NaN from step 1).'
              '\n!! On a Mac use --model roberta-base; DeBERTa works on CUDA/CPU.')

    data_dir = Path(args.data)
    def prep(rows):
        out = []
        for r in rows:
            r['y'] = int(r['label'] == 'ai')
            if args.fold_unicode:
                r['old_text'], r['new_text'] = fold(r['old_text']), fold(r['new_text'])
            out.append(r)
        return out
    train_all = prep(read_jsonl(data_dir / 'train.jsonl'))
    test_path = Path(args.test_file) if args.test_file else data_dir / 'test.jsonl'
    test_rows = prep(read_jsonl(test_path))
    val_papers = {p for p in {r['paper'] for r in train_all}
                  if seed64(f'val|{p}') % 10 == 0}
    train_rows = [r for r in train_all if r['paper'] not in val_papers]
    val_rows = [r for r in train_all if r['paper'] in val_papers]
    if args.train_frac < 1.0:
        papers_all = sorted({r['paper'] for r in train_rows})
        keep_frac = {p for p in papers_all
                     if (seed64(f'frac|{p}') % 10000) / 10000 < args.train_frac}
        train_rows = [r for r in train_rows if r['paper'] in keep_frac]
        print(f'train-frac {args.train_frac}: {len(keep_frac)}/{len(papers_all)} '
              'train papers kept')
    if args.limit:
        import random
        rng = random.Random(args.seed)   # rows are ordered human-first; shuffle before capping
        for rows in (train_rows, val_rows, test_rows):
            rng.shuffle(rows)
        train_rows = train_rows[:args.limit]
        val_rows = val_rows[:max(50, args.limit // 4)]
        test_rows = test_rows[:max(100, args.limit // 2)]
    n_pos = sum(r['y'] for r in train_rows)
    print(f'train {len(train_rows)} rows ({n_pos} ai) | val {len(val_rows)} | '
          f'test {len(test_rows)} rows over {len({r["paper"] for r in test_rows})} papers')

    model_src = args.eval_only or args.model
    tok = AutoTokenizer.from_pretrained(model_src)
    model = AutoModelForSequenceClassification.from_pretrained(model_src, num_labels=1)
    model.float().to(device)   # transformers>=5 loads checkpoint dtype (fp16); AdamW needs fp32

    def collate(rows):
        enc = tok([r['old_text'] for r in rows], [r['new_text'] for r in rows],
                  truncation=True, max_length=args.max_len,
                  padding=True, return_tensors='pt')
        labels = torch.tensor([float(r['y']) for r in rows])
        return enc, labels

    n_epochs = 0 if args.eval_only else args.epochs
    if n_epochs:
        loader = DataLoader(train_rows, batch_size=args.batch, shuffle=True,
                            collate_fn=collate)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
        total_steps = len(loader) * args.epochs
        sched = get_linear_schedule_with_warmup(opt, int(0.06 * total_steps), total_steps)
        pos_weight = torch.tensor([(len(train_rows) - n_pos) / max(1, n_pos)]).to(device)
        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        scaler = torch.amp.GradScaler('cuda') if amp_dtype == torch.float16 else None

    @torch.no_grad()
    def score_rows(rows, bs=None):
        model.eval()
        out = []
        bs = bs or args.batch * 2
        for k in range(0, len(rows), bs):
            enc, _ = collate(rows[k:k + bs])
            enc = {k2: v.to(device) for k2, v in enc.items()}
            logits = model(**enc).logits.squeeze(-1)
            out.extend(logits.float().cpu().tolist())
        model.train()
        return out

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_val, best_state = -1.0, None
    bad_steps = 0
    t0 = time.time()
    for epoch in range(n_epochs):
        for step, (enc, labels) in enumerate(loader):
            enc = {k: v.to(device) for k, v in enc.items()}
            labels = labels.to(device)
            opt.zero_grad()
            if amp_dtype is not None:
                with torch.autocast('cuda', dtype=amp_dtype):
                    loss = loss_fn(model(**enc).logits.squeeze(-1), labels)
            else:
                loss = loss_fn(model(**enc).logits.squeeze(-1), labels)
            if not torch.isfinite(loss):
                bad_steps += 1
                if bad_steps >= 3:
                    raise RuntimeError(
                        f'non-finite loss x{bad_steps} at epoch {epoch} step {step} '
                        f'on {device} — if this is MPS with a DeBERTa model, that '
                        'backend is broken; use --model roberta-base or CUDA/CPU.')
                continue
            bad_steps = 0
            if scaler:
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                opt.step()
            sched.step()
            if step % 200 == 0:
                el = time.time() - t0
                print(f'epoch {epoch} step {step}/{len(loader)} '
                      f'loss {loss.item():.4f} [{el:.0f}s]', flush=True)
        val_auc = auc_score([r['y'] for r in val_rows], score_rows(val_rows))
        print(f'=== epoch {epoch}: val AUC {val_auc:.4f}')
        if val_auc > best_val:
            best_val = val_auc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    elif not args.eval_only:
        print('WARNING: val AUC was nan in every epoch; scoring with last weights')
    if args.save_model and not args.eval_only:
        model.save_pretrained(out_dir / 'model')
        tok.save_pretrained(out_dir / 'model')
        print(f"checkpoint saved -> {out_dir / 'model'}")

    # threshold: maximize F1(ai) on val
    val_scores = score_rows(val_rows)
    if args.eval_only:
        best_val = auc_score([r['y'] for r in val_rows], val_scores)
    cands = sorted(set(round(s, 2) for s in val_scores))
    best_f1, thresh = -1.0, 0.0
    for t in cands:
        tp = sum(s >= t and r['y'] for s, r in zip(val_scores, val_rows))
        fp = sum(s >= t and not r['y'] for s, r in zip(val_scores, val_rows))
        fn = sum(s < t and r['y'] for s, r in zip(val_scores, val_rows))
        f1 = 2 * tp / max(1, 2 * tp + fp + fn)
        if f1 > best_f1:
            best_f1, thresh = f1, t

    scores = score_rows(test_rows)
    ys = [r['y'] for r in test_rows]
    a = auc_score(ys, scores)
    lo, hi = paper_bootstrap_auc(test_rows, scores, seed=args.seed)
    preds = [int(s >= thresh) for s in scores]
    tp = sum(p and y for p, y in zip(preds, ys))
    fp = sum(p and not y for p, y in zip(preds, ys))
    fn = sum((not p) and y for p, y in zip(preds, ys))
    tn = sum((not p) and (not y) for p, y in zip(preds, ys))
    f1_ai = 2 * tp / max(1, 2 * tp + fp + fn)
    bal_acc = 0.5 * (tp / max(1, tp + fn) + tn / max(1, tn + fp))

    metrics = {'config': vars(args), 'device': device, 'val_auc': best_val,
               'test': {'auc': a, 'auc_ci95': [lo, hi], 'f1_ai': f1_ai,
                        'balanced_acc': bal_acc, 'threshold': thresh,
                        'n': len(test_rows), 'n_ai': sum(ys)},
               'bands': {}, 'styles': {}, 'baselines': {}}
    print(f'test: AUC {a:.4f} [{lo:.4f},{hi:.4f}] | F1(ai) {f1_ai:.4f} | '
          f'balanced acc {bal_acc:.4f}')

    # magnitude-matched slices
    for blo, bhi in RATIO_BANDS:
        idx = [k for k, r in enumerate(test_rows) if blo <= r['ratio'] < bhi]
        ba = auc_score([ys[k] for k in idx], [scores[k] for k in idx])
        metrics['bands'][f'[{blo},{bhi})'] = {'auc': ba, 'n': len(idx),
                                              'n_ai': sum(ys[k] for k in idx)}
        print(f'band [{blo},{bhi}): AUC {ba:.4f} (n={len(idx)})')

    # per-style: each ai style vs ALL human test rows
    styles = sorted({r['style'] for r in test_rows if r['style']})
    human_idx = [k for k, r in enumerate(test_rows) if r['y'] == 0]
    for st in styles:
        st_idx = [k for k, r in enumerate(test_rows) if r.get('style') == st]
        idx = human_idx + st_idx
        sa = auc_score([ys[k] for k in idx], [scores[k] for k in idx])
        metrics['styles'][st] = {'auc': sa, 'n_ai': len(st_idx)}
        print(f'style {st}: AUC {sa:.4f} (n_ai={len(st_idx)})')

    # per-generator: when the test set mixes generator models (cross-gen sets)
    gens = sorted({r.get('model') for r in test_rows if r.get('model')})
    if len(gens) > 1:
        metrics['generators'] = {}
        for g in gens:
            g_idx = [k for k, r in enumerate(test_rows) if r.get('model') == g]
            idx = human_idx + g_idx
            ga = auc_score([ys[k] for k in idx], [scores[k] for k in idx])
            metrics['generators'][g] = {'auc': ga, 'n_ai': len(g_idx)}
            print(f'generator {g}: AUC {ga:.4f} (n_ai={len(g_idx)})')

    # surface baseline — the number to beat
    sb = surface_baseline(train_rows, test_rows)
    s_hgb = sb.pop('scores_hgb', None)
    metrics['baselines'].update(sb)
    metrics['baselines']['majority_acc'] = max(sum(ys), len(ys) - sum(ys)) / len(ys)
    rf = surface_refit_cv(test_rows)
    metrics['baselines'].update(rf)
    if 'surface_hgb_refit_cv' in rf:
        print(f"surface HGB refit on test (paper-grouped 5-fold CV): "
              f"{rf['surface_hgb_refit_cv']['auc']:.4f} — the within-set cheap ceiling")
    if 'surface_hgb' in sb:
        a = metrics['test']['auc']
        metrics['margin'] = {'vs_surface_lr': a - sb['surface_lr']['auc'],
                             'vs_surface_hgb': a - sb['surface_hgb']['auc']}
        if 'surface_hgb_refit_cv' in rf:
            metrics['margin']['vs_surface_hgb_refit_cv'] = (
                a - rf['surface_hgb_refit_cv']['auc'])
        print(f"surface baselines: LR {sb['surface_lr']['auc']:.4f} | "
              f"HGB {sb['surface_hgb']['auc']:.4f} (the number to beat) | "
              f"margin vs HGB {metrics['margin']['vs_surface_hgb']:+.4f}")
        # does the cheap ceiling add anything on top of the model? paper-grouped
        # 5-fold CV logistic stack of (model logit, HGB logit) on test, out-of-fold
        try:
            import numpy as np
            from sklearn.linear_model import LogisticRegression
            from sklearn.model_selection import GroupKFold
            papers = [r['paper'] for r in test_rows]
            if len(set(papers)) >= 5:
                X = np.array([[m, math.log(max(h, 1e-6) / max(1 - h, 1e-6))]
                              for m, h in zip(scores, s_hgb)])
                oof = np.zeros(len(ys))
                for tr_i, te_i in GroupKFold(n_splits=5).split(X, ys, papers):
                    clf = LogisticRegression(class_weight='balanced').fit(X[tr_i], np.array(ys)[tr_i])
                    oof[te_i] = clf.decision_function(X[te_i])
                metrics['baselines']['stack_model_plus_hgb_cv_auc'] = auc_score(ys, oof.tolist())
                print(f"grouped-CV stack(model + HGB): AUC "
                      f"{metrics['baselines']['stack_model_plus_hgb_cv_auc']:.4f} "
                      f"(vs model alone {a:.4f}: if equal, the model subsumes the cheap cues)")
        except ImportError:
            pass
    else:
        print(f'surface baseline: {sb}')

    with open(out_dir / 'predictions.jsonl', 'w', encoding='utf-8') as f:
        for r, s in zip(test_rows, scores):
            f.write(json.dumps({
                'paper': r['paper'], 'label': r['label'], 'score': s,
                'ratio': r['ratio'], 'style': r.get('style'),
                'model': r.get('model'),
                'section': r.get('section'), 'kind': r.get('kind'),
            }, ensure_ascii=False) + '\n')

    def _san(x):
        if isinstance(x, dict):
            return {k: _san(v) for k, v in x.items()}
        if isinstance(x, list):
            return [_san(v) for v in x]
        if isinstance(x, float) and x != x:
            return None
        return x

    with open(out_dir / 'metrics.json', 'w', encoding='utf-8') as f:
        json.dump(_san(metrics), f, indent=2, default=str)
    print(f"=== done in {(time.time() - t0) / 60:.1f} min -> {out_dir}")


if __name__ == '__main__':
    main()
