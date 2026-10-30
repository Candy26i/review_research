"""Assemble the edit-attribution dataset: (old_span, new_span) -> label.

Two classes, both drawn from the SAME pre-ChatGPT papers so topic and era
cannot leak into the label:
  human  real v1->v2 paragraph revisions (full-range aligned pairs, incl. the
         0.60-0.92 similarity band — excluding it leaves a magnitude hole the
         AI class does not share; QC found a band rule alone scores 90% acc)
  ai     styled LLM edits of v1 paragraphs from make_ai_edits.py

Artifact controls applied symmetrically to BOTH classes (QC findings):
  - whitespace collapsed to single spaces (AI generators strip newlines/nbsp;
    raw newline retention alone separated the classes at 0.70 AUC)
  - leading structural markers stripped (Caption:, * bullets, <cit.>)
  - word-level similarity ratio stored for BOTH classes (null-iff-ai was a
    perfect label leak)
  - exact duplicate rows dropped; (old,new) pairs appearing under both labels
    dropped entirely (AI edit coincided with the real human revision)

Paper-level 80/20 split. Rows: {old_text, new_text, label, paper, section,
kind, ratio, style, model, split}.

Run from repo root (after the pre-2022 pipeline + make_ai_edits):
  python3 src/process/build_attribution_dataset.py
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, 'src/collect')
sys.path.insert(0, 'src/process')
from util import read_jsonl  # noqa: E402
from build_splice_pilot import align, parse_blocks, seed64, word_ratio  # noqa: E402

TEST_MOD = 5  # paper_hash % 5 == 0 -> test
WS_RE = re.compile(r'\s+')
LEAD_MARKER_RE = re.compile(r'^(?:caption:\s*|\*\s+|<cit\.>\s*)+', re.I)


def clean(text: str) -> str:
    """Symmetric surface normalization (both classes, both sides)."""
    s = WS_RE.sub(' ', text).strip()
    return LEAD_MARKER_RE.sub('', s).strip()


def main() -> None:
    ap = argparse.ArgumentParser(description='Edit-attribution dataset builder.')
    ap.add_argument('--manifest', default='data/processed/pre2022_manifest.jsonl')
    ap.add_argument('--texts', default='data/processed/pre2022_texts/')
    ap.add_argument('--ai-log', default='data/processed/ai_edits_log.jsonl')
    ap.add_argument('--out-dir', default='data/processed/attribution_v0')
    ap.add_argument('--min-ratio', type=float, default=0.30,
                    help='human pairs below this similarity are likely '
                         'mismatches (QC-verified) and are dropped')
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    # ---- human class: aligned real revisions (full range incl. 0.60-0.92)
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
        _, rewrites, micro = align(b1, b2, full_range=True)
        pairs = [(i, j, r) for i, j, r in rewrites + micro if r >= args.min_ratio]
        if not pairs:
            continue
        n_papers += 1
        split = 'test' if seed64(f"split|{p['arxiv_id']}") % TEST_MOD == 0 else 'train'
        for i, j, r in pairs:
            rows.append({'label': 'human', 'paper': p['arxiv_id'], 'split': split,
                         'section': b1[i]['section'], 'kind': b1[i]['kind'],
                         'ratio': round(r, 4), 'style': None, 'model': None,
                         'old_text': clean(b1[i]['text']),
                         'new_text': clean(b2[j]['text'])})

    # ---- ai class: generator output (same papers, same split hash)
    n_ai = n_ai_bad = 0
    if Path(args.ai_log).exists():
        for r in read_jsonl(args.ai_log):
            if r.get('error') or not r.get('new_text'):
                n_ai_bad += 1
                continue
            split = 'test' if seed64(f"split|{r['arxiv_id']}") % TEST_MOD == 0 else 'train'
            old, new = clean(r['old_text']), clean(r['new_text'])
            r_ai = word_ratio(old, new)
            if r_ai < args.min_ratio:
                # same floor as the human class — below it sit mismatches and
                # generator refusals (QC found one that slipped the length gate)
                n_ai_bad += 1
                continue
            rows.append({'label': 'ai', 'paper': r['arxiv_id'], 'split': split,
                         'section': r.get('section'), 'kind': r.get('kind'),
                         'ratio': round(r_ai, 4),
                         'style': r['style'], 'model': r['model'],
                         'old_text': old, 'new_text': new})
            n_ai += 1

    # ---- symmetric filters: no-ops, exact dupes, contradictory-label pairs
    n_noop = sum(r['old_text'] == r['new_text'] for r in rows)
    rows = [r for r in rows if r['old_text'] != r['new_text']]
    seen: set[tuple] = set()
    deduped: list[dict] = []
    for r in rows:
        key = (r['label'], r['old_text'], r['new_text'])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(r)
    n_dupes = len(rows) - len(deduped)
    by_pair: dict[tuple, set] = {}
    for r in deduped:
        by_pair.setdefault((r['old_text'], r['new_text']), set()).add(r['label'])
    contested = {k for k, labels in by_pair.items() if len(labels) > 1}
    n_contested = sum((r['old_text'], r['new_text']) in contested for r in deduped)
    rows = [r for r in deduped if (r['old_text'], r['new_text']) not in contested]

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
          f'ai rows kept: {n_ai} (rejected: {n_ai_bad})')
    print(f'dropped: {n_noop} no-ops after cleaning, {n_dupes} dupes, '
          f'{n_contested} contested-label rows -> {out_dir}')


if __name__ == '__main__':
    main()
