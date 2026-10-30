"""Train the frozen-splice paragraph tagger (Q2 pilot).

Portable: auto-selects cuda > mps > cpu; needs torch + transformers +
sentencepiece (and a network connection on first run to fetch the model).
Trains on cond='true' train-split docs, selects the best epoch by val AUC
(10% of train papers), then scores EVERY test-split condition
(true / v1self / micro / ident) and writes metrics.json + predictions.jsonl.

On the 4090 box, from the repo root:
  python3 src/train/train_splice_tagger.py --data data/processed/splice_pilot \
      --out results/splice_pilot/deberta_ctx1
Ablations:
  --context 0        paragraph alone (no seam visibility)
  --strip-symbols    digits masked (style channel only)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
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


NUM_RE = re.compile(r'\d+(?:\.\d+)?')


def build_input(row: dict, sep: str, context: int, strip_symbols: bool) -> str:
    text = row['text']
    if context:
        text = f"{row.get('prev', '')} {sep} {text} {sep} {row.get('next', '')}"
    if strip_symbols:
        text = NUM_RE.sub('<num>', text)
    return text


def main() -> None:
    ap = argparse.ArgumentParser(description='Frozen-splice paragraph tagger.')
    ap.add_argument('--data', default='data/processed/splice_pilot')
    ap.add_argument('--out', default='results/splice_pilot/run')
    ap.add_argument('--model', default='microsoft/deberta-v3-base')
    ap.add_argument('--context', type=int, default=1, choices=(0, 1))
    ap.add_argument('--strip-symbols', action='store_true')
    ap.add_argument('--epochs', type=int, default=2)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--max-len', type=int, default=384)
    ap.add_argument('--lr', type=float, default=2e-5)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--train-frac', type=float, default=1.0,
                    help='train on this fraction of train papers (deterministic '
                         'paper-hash subsample; the learning-curve escalation)')
    ap.add_argument('--limit', type=int, default=0,
                    help='debug: cap train/eval rows')
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
    train_all = [r for r in read_jsonl(data_dir / 'train.jsonl') if r['cond'] == 'true']
    test_all = read_jsonl(data_dir / 'test.jsonl')
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
        train_rows, val_rows = train_rows[:args.limit], val_rows[:max(50, args.limit // 4)]
        # cap test by PAPER so every condition keeps its rows and positives
        keep: list = []
        for r in test_all:
            if r['paper'] not in keep:
                if len(keep) >= max(2, args.limit // 150):
                    break
                keep.append(r['paper'])
        test_all = [r for r in test_all if r['paper'] in keep]
    n_pos = sum(r['label'] for r in train_rows)
    print(f'train {len(train_rows)} rows ({n_pos} pos) | val {len(val_rows)} | '
          f'test {len(test_all)} rows over {len({r["paper"] for r in test_all})} papers')

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=1)
    model.float().to(device)   # transformers>=5 loads checkpoint dtype (fp16) by default; AdamW needs fp32 master weights
    sep = tok.sep_token or '[SEP]'

    def collate(rows):
        texts = [build_input(r, sep, args.context, args.strip_symbols) for r in rows]
        enc = tok(texts, truncation=True, max_length=args.max_len,
                  padding=True, return_tensors='pt')
        labels = torch.tensor([float(r['label']) for r in rows])
        return enc, labels

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
    for epoch in range(args.epochs):
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
            if step % 100 == 0:
                el = time.time() - t0
                print(f'epoch {epoch} step {step}/{len(loader)} '
                      f'loss {loss.item():.4f} [{el:.0f}s]', flush=True)
        val_auc = auc_score([r['label'] for r in val_rows], score_rows(val_rows))
        print(f'=== epoch {epoch}: val AUC {val_auc:.4f}')
        if val_auc > best_val:
            best_val = val_auc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    else:
        print('WARNING: val AUC was nan in every epoch (single-class val set?); '
              'scoring with last-epoch weights')

    conds = sorted({r['cond'] for r in test_all})
    metrics = {'config': vars(args), 'device': device, 'val_auc': best_val,
               'test': {}}
    with open(out_dir / 'predictions.jsonl', 'w', encoding='utf-8') as f:
        for cond in conds:
            rows = [r for r in test_all if r['cond'] == cond]
            scores = score_rows(rows)
            labels = [r['label'] for r in rows]
            a = auc_score(labels, scores)
            metrics['test'][cond] = {'auc': a, 'n': len(rows),
                                     'n_pos': sum(labels)}
            print(f'test[{cond}]: AUC {a:.4f} (n={len(rows)}, pos={sum(labels)})')
            for r, s in zip(rows, scores):
                f.write(json.dumps({
                    'paper': r['paper'], 'cond': cond, 'para_idx': r['para_idx'],
                    'label': r['label'], 'score': s, 'section': r['section'],
                    'kind': r['kind'], 'words': r['words'],
                    'change_ratio': r.get('change_ratio'), 'corpus': r['corpus'],
                }, ensure_ascii=False) + '\n')
    def _san(x):
        if isinstance(x, dict):
            return {k: _san(v) for k, v in x.items()}
        if isinstance(x, float) and x != x:   # NaN -> null for strict parsers
            return None
        return x

    with open(out_dir / 'metrics.json', 'w', encoding='utf-8') as f:
        json.dump(_san(metrics), f, indent=2, default=str)
    print(f"=== done in {(time.time() - t0) / 60:.1f} min -> {out_dir}")


if __name__ == '__main__':
    main()
