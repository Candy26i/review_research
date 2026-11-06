"""Reconcile locally rebuilt pre-2022 v1 texts against the attribution_v0 ai rows.

The pre2022_texts/ tree was built on another machine; when it is rebuilt
elsewhere (different OS, no SIGALRM converter timeout) the paragraph sampler in
make_ai_edits.py only reproduces the same (paper, block, style) combos if the
texts parse identically. For every paper this replays the sampler and checks
that each deepseek ai row already in attribution_v0 (paper, cleaned old_text,
style) is one of the sampled (block, style) picks.

Paper status:
  match         every v0 ai row of the paper is reproduced (block text + style)
  mismatch      at least one v0 ai row is not reproduced        -> skipped
  missing_text  v1 text absent locally                            -> skipped
  unverifiable  paper has no ai row in v0 (nothing to compare)    -> kept

Writes a manifest of kept papers for make_ai_edits.py --manifest.

Run from repo root:
  python3 src/process/reconcile_pre2022_texts.py
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, 'src/collect')
sys.path.insert(0, 'src/process')
from util import read_jsonl  # noqa: E402
from build_splice_pilot import parse_blocks, seed64  # noqa: E402
from build_attribution_dataset import clean, TEST_MOD  # noqa: E402
from make_ai_edits import EDIT_STYLES  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', default='data/processed/pre2022_manifest.jsonl')
    ap.add_argument('--texts', default='data/processed/pre2022_texts/')
    ap.add_argument('--v0', default='data/processed/attribution_v0')
    ap.add_argument('--out-manifest', default='data/processed/pre2022_reconciled_manifest.jsonl')
    ap.add_argument('--report', default='results/attribution_v0/pre2022_reconcile.json')
    ap.add_argument('--seed', type=int, default=20261030)
    ap.add_argument('--paras-per-paper', type=int, default=4)
    args = ap.parse_args()

    v0 = defaultdict(list)
    for split in ('train', 'test'):
        with open(Path(args.v0) / f'{split}.jsonl', encoding='utf-8') as f:
            for line in f:
                r = json.loads(line)
                if r['label'] == 'ai':
                    v0[r['paper']].append((r['old_text'], r['style']))

    styles = sorted(EDIT_STYLES)
    status, kept, detail = Counter(), [], []
    rows_total = rows_ok = rows_text_only = 0
    by_split = {'test': Counter(), 'train': Counter()}
    for p in read_jsonl(args.manifest):
        aid = p['arxiv_id']
        split = 'test' if seed64(f'split|{aid}') % TEST_MOD == 0 else 'train'
        v1 = min(v['v'] for v in p['versions'])
        path = Path(args.texts) / aid.replace('/', '_') / f'v{v1}.md'
        want = v0.get(aid, [])
        rows_total += len(want)
        if not path.exists():
            st = 'missing_text'
        else:
            blocks = parse_blocks(path.read_text(encoding='utf-8'))
            cls_idx = [i for i, b in enumerate(blocks) if b['cls']]
            picks = set()
            if len(cls_idx) >= args.paras_per_paper:
                rng = np.random.default_rng(seed64(f'{args.seed}|{aid}|ai_edits'))
                chosen = rng.choice(len(cls_idx), size=args.paras_per_paper, replace=False)
                for j in sorted(chosen):
                    style = styles[int(rng.integers(len(styles)))]
                    picks.add((clean(blocks[cls_idx[j]]['text']), style))
            all_texts = {clean(b['text']) for b in blocks}
            ok = sum(w in picks for w in want)
            rows_ok += ok
            rows_text_only += sum((w not in picks) and (w[0] in all_texts) for w in want)
            st = 'unverifiable' if not want else 'match' if ok == len(want) else 'mismatch'
            if st == 'mismatch':
                detail.append({'paper': aid, 'split': split, 'v0_rows': len(want), 'reproduced': ok})
        status[st] += 1
        by_split[split][st] += 1
        if st in ('match', 'unverifiable'):
            kept.append(p)

    with open(args.out_manifest, 'w', encoding='utf-8') as f:
        for p in kept:
            f.write(json.dumps(p, ensure_ascii=False) + '\n')
    verifiable = status['match'] + status['mismatch']
    rep = {'papers': dict(status), 'by_split': {k: dict(v) for k, v in by_split.items()},
           'paper_match_rate': round(status['match'] / max(1, verifiable), 4),
           'v0_ai_rows': rows_total, 'rows_reproduced': rows_ok,
           'row_match_rate': round(rows_ok / max(1, rows_total), 4),
           'rows_text_present_but_not_sampled': rows_text_only,
           'kept_papers': len(kept), 'mismatched': detail}
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(rep, indent=1), encoding='utf-8')
    rep_print = dict(rep); rep_print['mismatched'] = rep_print['mismatched'][:10]
    print(json.dumps(rep_print, indent=1))


if __name__ == '__main__':
    main()
