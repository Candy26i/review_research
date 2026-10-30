"""Assemble the edit-attribution dataset: (old_span, new_span) -> label.

Two classes, both drawn from the SAME pre-ChatGPT papers so topic and era
cannot leak into the label:
  human  real v1->v2 paragraph revisions (aligned rewrite pairs, plus micro
         pairs kept with their ratio so analysis can slice by edit size)
  ai     styled LLM edits of v1 paragraphs from make_ai_edits.py

Paper-level 80/20 split. Rows: {old_text, new_text, label, paper, section,
kind, ratio|style, model, split}.

Run from repo root (after the pre-2022 pipeline + make_ai_edits):
  python3 src/process/build_attribution_dataset.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, 'src/collect')
sys.path.insert(0, 'src/process')
from util import read_jsonl  # noqa: E402
from build_splice_pilot import align, parse_blocks, seed64  # noqa: E402

TEST_MOD = 5  # paper_hash % 5 == 0 -> test


def main() -> None:
    ap = argparse.ArgumentParser(description='Edit-attribution dataset builder.')
    ap.add_argument('--manifest', default='data/processed/pre2022_manifest.jsonl')
    ap.add_argument('--texts', default='data/processed/pre2022_texts/')
    ap.add_argument('--ai-log', default='data/processed/ai_edits_log.jsonl')
    ap.add_argument('--out-dir', default='data/processed/attribution_v0')
    ap.add_argument('--min-ratio', type=float, default=0.15)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    # ---- human class: aligned real revisions
    n_papers = n_missing = 0
    for p in read_jsonl(args.manifest):
        safe_id = p['arxiv_id'].replace('/', '_')
        vs = sorted(v['v'] for v in p['versions'])
        f1 = Path(args.texts) / safe_id / f'v{vs[0]}.md'
        f2 = Path(args.texts) / safe_id / f'v{vs[-1]}.md'
        if not f1.exists() or not f2.exists():
            n_missing += 1
            continue
        b1 = parse_blocks(f1.read_text(encoding='utf-8'))
        b2 = parse_blocks(f2.read_text(encoding='utf-8'))
        _, rewrites, micro = align(b1, b2)
        pairs = [(i, j, r) for i, j, r in rewrites + micro if r >= args.min_ratio]
        if not pairs:
            continue
        n_papers += 1
        split = 'test' if seed64(f"split|{p['arxiv_id']}") % TEST_MOD == 0 else 'train'
        for i, j, r in pairs:
            rows.append({'label': 'human', 'paper': p['arxiv_id'], 'split': split,
                         'section': b1[i]['section'], 'kind': b1[i]['kind'],
                         'ratio': round(r, 4), 'style': None, 'model': None,
                         'old_text': b1[i]['text'], 'new_text': b2[j]['text']})

    # ---- ai class: generator output (same papers, same split hash)
    n_ai = n_ai_bad = 0
    if Path(args.ai_log).exists():
        for r in read_jsonl(args.ai_log):
            if r.get('error') or not r.get('new_text'):
                n_ai_bad += 1
                continue
            split = 'test' if seed64(f"split|{r['arxiv_id']}") % TEST_MOD == 0 else 'train'
            rows.append({'label': 'ai', 'paper': r['arxiv_id'], 'split': split,
                         'section': r.get('section'), 'kind': r.get('kind'),
                         'ratio': None, 'style': r['style'], 'model': r['model'],
                         'old_text': r['old_text'], 'new_text': r['new_text']})
            n_ai += 1

    for split in ('train', 'test'):
        part = [r for r in rows if r['split'] == split]
        with open(out_dir / f'{split}.jsonl', 'w', encoding='utf-8') as f:
            for r in part:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')
        n_h = sum(r['label'] == 'human' for r in part)
        n_a = len(part) - n_h
        print(f'{split}: {len(part)} rows (human {n_h} / ai {n_a}) over '
              f'{len({r["paper"] for r in part})} papers')
    print(f'human-source papers: {n_papers} (texts missing: {n_missing}) | '
          f'ai rows kept: {n_ai} (rejected: {n_ai_bad}) -> {out_dir}')


if __name__ == '__main__':
    main()
