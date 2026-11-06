"""Which-generator attribution: (old_text, new_text) -> {human, deepseek, ...}.

Multi-class cross-encoder over attribution_v1. Classes = labels with TRAIN
support (human + the full-coverage generators). Probe models that exist only
in the test split (claude-sonnet, gpt-5.5, deepseek-reasoner) are NOT
classes: their rows are scored separately and reported as predicted-class
histograms — if sonnet rows land on haiku, the detector carries a FAMILY
(vendor) fingerprint rather than a checkpoint fingerprint.

Reported: per-class one-vs-rest AUC, argmax confusion matrix, macro-F1,
binary collapse (p_ai = 1 - p_human, comparable to the v0 headline), and
the probe histograms. Same paper-level val split as the binary trainer.

Run from repo root (4090; smoke with --limit 600 first):
  python3 src/train/train_attribution_mc.py \
      --data data/processed/attribution_v1 --out results/attribution_v1/mc
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, 'src/train')
from train_attribution import auc_score, read_jsonl, seed64  # noqa: E402


def cls_of(r) -> str:
    return r.get('model') or 'human'


def main() -> None:
    ap = argparse.ArgumentParser(description='Which-generator cross-encoder.')
    ap.add_argument('--data', default='data/processed/attribution_v1')
    ap.add_argument('--out', default='results/attribution_v1/mc')
    ap.add_argument('--model', default='microsoft/deberta-v3-base')
    ap.add_argument('--epochs', type=int, default=2)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--max-len', type=int, default=512)
    ap.add_argument('--lr', type=float, default=2e-5)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--min-train-rows', type=int, default=500,
                    help='a label needs this many train rows to be a class; '
                         'rarer test-only models become probes')
    ap.add_argument('--exclude-models', default='',
                    help='comma-separated model ids to drop entirely (e.g. a '
                         'partial, era-biased family like an unfinished gemini)')
    ap.add_argument('--limit', type=int, default=0, help='debug: cap rows')
    ap.add_argument('--save-model', action='store_true')
    args = ap.parse_args()

    import torch
    from torch.utils.data import DataLoader
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                              get_linear_schedule_with_warmup)

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        device, amp_dtype = 'cuda', (torch.bfloat16 if torch.cuda.is_bf16_supported()
                                     else torch.float16)
    else:
        device, amp_dtype = 'cpu', None
    print(f'device: {device} (amp: {amp_dtype})')

    data_dir = Path(args.data)
    train_all = read_jsonl(data_dir / 'train.jsonl')
    test_rows = read_jsonl(data_dir / 'test.jsonl')
    excl = {m for m in args.exclude_models.split(',') if m}
    if excl:
        train_all = [r for r in train_all if cls_of(r) not in excl]
        test_rows = [r for r in test_rows if cls_of(r) not in excl]
        print(f'excluded models: {sorted(excl)}')

    train_counts = Counter(cls_of(r) for r in train_all)
    classes = sorted(c for c, n in train_counts.items() if n >= args.min_train_rows)
    c2i = {c: i for i, c in enumerate(classes)}
    print(f'classes ({len(classes)}): '
          f'{ {c: train_counts[c] for c in classes} }')
    dropped = {c: n for c, n in train_counts.items() if c not in c2i}
    if dropped:
        print(f'train labels below --min-train-rows, excluded: {dropped}')
    train_all = [r for r in train_all if cls_of(r) in c2i]
    for r in train_all:
        r['y'] = c2i[cls_of(r)]
    known_test = [r for r in test_rows if cls_of(r) in c2i]
    probe_test = [r for r in test_rows if cls_of(r) not in c2i]
    for r in known_test:
        r['y'] = c2i[cls_of(r)]

    val_papers = {p for p in {r['paper'] for r in train_all}
                  if seed64(f'val|{p}') % 10 == 0}
    train_rows = [r for r in train_all if r['paper'] not in val_papers]
    val_rows = [r for r in train_all if r['paper'] in val_papers]
    if args.limit:
        import random
        rng = random.Random(args.seed)
        for rows in (train_rows, val_rows, known_test, probe_test):
            rng.shuffle(rows)
        train_rows = train_rows[:args.limit]
        val_rows = val_rows[:max(100, args.limit // 4)]
        known_test = known_test[:max(200, args.limit // 2)]
        probe_test = probe_test[:max(100, args.limit // 4)]
    print(f'train {len(train_rows)} | val {len(val_rows)} | '
          f'test known {len(known_test)} / probe {len(probe_test)}')

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=len(classes))
    model.float().to(device)

    def collate(rows):
        enc = tok([r['old_text'] for r in rows], [r['new_text'] for r in rows],
                  truncation=True, max_length=args.max_len,
                  padding=True, return_tensors='pt')
        # probe rows carry no 'y' (test-only models are not classes); the
        # label is unused at inference, so default harmlessly
        labels = torch.tensor([r.get('y', 0) for r in rows], dtype=torch.long)
        return enc, labels

    loader = DataLoader(train_rows, batch_size=args.batch, shuffle=True,
                        collate_fn=collate)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(loader) * args.epochs
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * total_steps), total_steps)
    counts = Counter(r['y'] for r in train_rows)
    w = torch.tensor([len(train_rows) / (len(classes) * max(1, counts[i]))
                      for i in range(len(classes))])
    w = w.clamp(max=20.0).to(device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=w)
    scaler = torch.amp.GradScaler('cuda') if amp_dtype == torch.float16 else None

    @torch.no_grad()
    def probs_of(rows, bs=None):
        model.eval()
        out = []
        bs = bs or args.batch * 2
        for k in range(0, len(rows), bs):
            enc, _ = collate(rows[k:k + bs])
            enc = {k2: v.to(device) for k2, v in enc.items()}
            logits = model(**enc).logits
            out.extend(torch.softmax(logits.float(), dim=-1).cpu().tolist())
        model.train()
        return out

    def macro_ovr_auc(rows, probs):
        aucs = []
        for c, i in c2i.items():
            ys = [int(r['y'] == i) for r in rows]
            if 0 < sum(ys) < len(ys):
                aucs.append(auc_score(ys, [p[i] for p in probs]))
        return sum(aucs) / len(aucs) if aucs else float('nan')

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_val, best_state = -1.0, None
    t0 = time.time()
    for epoch in range(args.epochs):
        for step, (enc, labels) in enumerate(loader):
            enc = {k: v.to(device) for k, v in enc.items()}
            labels = labels.to(device)
            opt.zero_grad()
            if amp_dtype is not None:
                with torch.autocast('cuda', dtype=amp_dtype):
                    loss = loss_fn(model(**enc).logits, labels)
            else:
                loss = loss_fn(model(**enc).logits, labels)
            if scaler:
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                opt.step()
            sched.step()
            if step % 200 == 0:
                print(f'epoch {epoch} step {step}/{len(loader)} '
                      f'loss {loss.item():.4f} [{time.time() - t0:.0f}s]', flush=True)
        val_auc = macro_ovr_auc(val_rows, probs_of(val_rows))
        print(f'=== epoch {epoch}: val macro-ovr-AUC {val_auc:.4f}')
        if val_auc > best_val:
            best_val = val_auc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    else:
        print('WARNING: val macro-AUC was nan every epoch; scoring last weights')
    if args.save_model:
        model.config.id2label = {i: c for i, c in enumerate(classes)}
        model.config.label2id = dict(c2i)
        model.save_pretrained(out_dir / 'model')
        tok.save_pretrained(out_dir / 'model')
        print(f"checkpoint saved -> {out_dir / 'model'}")

    probs = probs_of(known_test)
    preds = [max(range(len(classes)), key=lambda i: p[i]) for p in probs]
    ys = [r['y'] for r in known_test]
    probe_probs = {m: probs_of([r for r in probe_test if cls_of(r) == m])
                   for m in sorted({cls_of(r) for r in probe_test})}

    per_class = {}
    for c, i in c2i.items():
        ys_i = [int(y == i) for y in ys]
        a = (auc_score(ys_i, [p[i] for p in probs])
             if 0 < sum(ys_i) < len(ys_i) else float('nan'))
        tp = sum(p == i and y == i for p, y in zip(preds, ys))
        fp = sum(p == i and y != i for p, y in zip(preds, ys))
        fn = sum(p != i and y == i for p, y in zip(preds, ys))
        f1 = 2 * tp / max(1, 2 * tp + fp + fn)
        per_class[c] = {'ovr_auc': a, 'f1': f1, 'support': sum(ys_i)}
        print(f'{c:28s} ovr-AUC {a:.4f} | F1 {f1:.4f} | n {sum(ys_i)}')
    macro_f1 = sum(v['f1'] for v in per_class.values()) / len(per_class)

    conf = [[0] * len(classes) for _ in classes]
    for p, y in zip(preds, ys):
        conf[y][p] += 1
    print('confusion rows=true cols=pred, order:', classes)
    for c, row in zip(classes, conf):
        tot = max(1, sum(row))
        print(f'{c:28s} {row}  recall% ' +
              ' '.join(f'{100 * x / tot:5.1f}' for x in row))

    # which-generator CONDITIONED ON AI: the human-dominated ovr numbers mostly
    # re-express the binary result; this is the discriminability that matters
    ai_classes = [c for c in classes if c != 'human']
    ai_rows = [(k, r) for k, r in enumerate(known_test) if cls_of(r) != 'human']
    ai_only: dict = {}
    if len(ai_classes) > 1 and ai_rows:
        def ai_argmax(p):
            return max(ai_classes, key=lambda c: p[c2i[c]])
        hits = sum(ai_argmax(probs[k]) == cls_of(r) for k, r in ai_rows)
        ai_only = {'acc': hits / len(ai_rows), 'n': len(ai_rows),
                   'chance': 1 / len(ai_classes), 'recall': {}}
        for c in ai_classes:
            rows_c = [(k, r) for k, r in ai_rows if cls_of(r) == c]
            if rows_c:
                ai_only['recall'][c] = (
                    sum(ai_argmax(probs[k]) == c for k, _ in rows_c) / len(rows_c))
        print(f"which-generator | AI rows only: acc {ai_only['acc']:.4f} "
              f"(n={ai_only['n']}, chance {ai_only['chance']:.2f})")

    # cheap-feature baseline for the SAME multi-class task (project QC rule:
    # quote margins over a tree on surface cues, never absolute numbers alone)
    baseline: dict = {}
    try:
        import numpy as np
        from sklearn.ensemble import HistGradientBoostingClassifier
        from train_attribution import count_features, surface_features
        Xtr = np.array([surface_features(r) + count_features(r) for r in train_rows])
        Xte = np.array([surface_features(r) + count_features(r) for r in known_test])
        hgb = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05,
                                             class_weight='balanced', random_state=0)
        hgb.fit(Xtr, [r['y'] for r in train_rows])
        Pb = hgb.predict_proba(Xte)
        col = {int(c): j for j, c in enumerate(hgb.classes_)}
        b_aucs = []
        for c, i in c2i.items():
            ys_i = [int(y == i) for y in ys]
            if 0 < sum(ys_i) < len(ys_i) and i in col:
                b_aucs.append(auc_score(ys_i, Pb[:, col[i]].tolist()))
        cands = [c for c in ai_classes if c2i[c] in col]
        b_acc = None
        if cands and ai_rows:
            b_hits = sum(max(cands, key=lambda c: Pb[k, col[c2i[c]]]) == cls_of(r)
                         for k, r in ai_rows)
            b_acc = b_hits / len(ai_rows)
        baseline['surface_hgb_mc'] = {
            'macro_ovr_auc': (sum(b_aucs) / len(b_aucs)) if b_aucs else None,
            'ai_only_acc': b_acc, 'n_features': int(Xtr.shape[1])}
        print(f"surface HGB (multi-class, {Xtr.shape[1]} feats): macro ovr-AUC "
              f"{baseline['surface_hgb_mc']['macro_ovr_auc']:.4f} | AI-only acc "
              f"{b_acc if b_acc is None else round(b_acc, 4)} — numbers to beat")
    except ImportError:
        pass

    # binary collapse — WITH the unseen probe rows included (they are all ai),
    # plus per-probe-model AUC vs human: the cross-generator question again
    i_h = c2i.get('human')
    bin_auc = bin_auc_probes = None
    probe_auc_vs_human: dict = {}
    if i_h is not None:
        y_bin = [int(r['y'] != i_h) for r in known_test]
        s_bin = [1 - p[i_h] for p in probs]
        bin_auc = auc_score(y_bin, s_bin)
        s_h = [1 - p[i_h] for p, r in zip(probs, known_test) if r['y'] == i_h]
        flat = [p for m in probe_probs for p in probe_probs[m]]
        if flat:
            bin_auc_probes = auc_score(y_bin + [1] * len(flat),
                                       s_bin + [1 - p[i_h] for p in flat])
            for m, pm in probe_probs.items():
                probe_auc_vs_human[m] = auc_score(
                    [0] * len(s_h) + [1] * len(pm),
                    s_h + [1 - p[i_h] for p in pm])
        print(f'binary collapse: known {bin_auc:.4f}' +
              (f' | +unseen probes {bin_auc_probes:.4f}' if bin_auc_probes else ''))
        for m, a in probe_auc_vs_human.items():
            print(f'  unseen {m} vs human: AUC {a:.4f}')

    # probes -> predicted-class histogram (family-fingerprint evidence)
    probe_hist = {}
    for m, pm in probe_probs.items():
        hist = Counter(classes[max(range(len(classes)), key=lambda i: p[i])]
                       for p in pm)
        total = sum(hist.values())
        probe_hist[m] = {c: n / total for c, n in hist.most_common()}
        print(f'probe {m} (n={total}): ' +
              ', '.join(f'{c} {n / total:.0%}' for c, n in hist.most_common(3)))

    metrics = {'config': vars(args), 'device': device, 'classes': classes,
               'val_macro_ovr_auc': best_val,
               'test': {'macro_f1': macro_f1, 'per_class': per_class,
                        'ai_only': ai_only,
                        'binary_collapse_auc': bin_auc,
                        'binary_collapse_auc_with_probes': bin_auc_probes,
                        'probe_auc_vs_human': probe_auc_vs_human,
                        'n_known': len(known_test), 'n_probe': len(probe_test)},
               'baselines': baseline,
               'confusion': {'order': classes, 'rows_true': conf},
               'probes': probe_hist,
               'notes': ['gemini-3.8-flash is a 23% manifest-prefix (era-'
                         'biased) partial family: quote its row as preliminary '
                         'until the class is completed or --exclude-models it']}

    with open(out_dir / 'predictions.jsonl', 'w', encoding='utf-8') as f:
        probe_flat_rows = [(r, p) for m in probe_probs
                           for r, p in zip([x for x in probe_test if cls_of(x) == m],
                                           probe_probs[m])]
        for r, p in list(zip(known_test, probs)) + probe_flat_rows:
            f.write(json.dumps({
                'paper': r['paper'], 'true': cls_of(r),
                'pred': classes[max(range(len(classes)), key=lambda i: p[i])],
                'probs': {c: round(p[c2i[c]], 5) for c in classes},
                'ratio': r.get('ratio'), 'style': r.get('style')},
                ensure_ascii=False) + '\n')

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
    print(f'=== done in {(time.time() - t0) / 60:.1f} min -> {out_dir}')


if __name__ == '__main__':
    main()
