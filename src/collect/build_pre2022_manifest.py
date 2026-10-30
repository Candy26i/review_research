"""Select guaranteed-human revision pairs: papers whose EVERY version predates
ChatGPT (cutoff 2022-11-01), from the Kaggle snapshot conversion.

These are the clean "human edit" class for attribution training and the
pre-contamination anchor for delta-quality labels. Filters: cs.* primary
category, >=2 versions, all versions before the cutoff, >=14 days between
first and last version (a meaningful revision window), v1 no older than
2015 (modern LaTeX/writing style, closer to the AI-edit contrast set).
Stratified by v1 year with recent years weighted up.

Run from repo root:
  python3 src/collect/build_pre2022_manifest.py --target 3000
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict

sys.path.insert(0, 'src/collect')
from util import read_jsonl  # noqa: E402

CUTOFF = '2022-11-01'
MIN_V1_YEAR = 2015
MIN_SPAN_DAYS = 14
MAX_VERSIONS = 6
# sampling weight per v1-year bucket (closer to the AI-edit era = more useful)
YEAR_WEIGHTS = {2015: 0.5, 2016: 0.5, 2017: 0.8, 2018: 0.8,
                2019: 1.2, 2020: 1.2, 2021: 1.5, 2022: 1.5}


def seed64(key: str) -> int:
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], 'big')


def main() -> None:
    ap = argparse.ArgumentParser(description='Pre-ChatGPT human revision-pair manifest.')
    ap.add_argument('--bulk', default='data/processed/arxiv_bulk_kaggle.jsonl')
    ap.add_argument('--out', default='data/processed/pre2022_manifest.jsonl')
    ap.add_argument('--target', type=int, default=3000)
    ap.add_argument('--seed', type=int, default=20261030)
    args = ap.parse_args()

    by_year: dict[int, list[dict]] = defaultdict(list)
    n_read = n_eligible = 0
    for row in read_jsonl(args.bulk):
        n_read += 1
        cats = row.get('categories') or []
        if not cats or not cats[0].startswith('cs.'):
            continue
        vs = row.get('versions') or []
        if not (2 <= len(vs) <= MAX_VERSIONS):
            continue
        dates = [v.get('date') or '' for v in vs]
        if not all(dates) or max(dates) >= CUTOFF:
            continue  # every version must predate ChatGPT
        try:
            year = int(dates[0][:4])
        except ValueError:
            continue
        if year < MIN_V1_YEAR:
            continue
        # revision window: at least MIN_SPAN_DAYS between first and last version
        from datetime import datetime
        try:
            span = (datetime.fromisoformat(dates[-1][:19])
                    - datetime.fromisoformat(dates[0][:19])).days
        except ValueError:
            continue
        if span < MIN_SPAN_DAYS:
            continue
        n_eligible += 1
        by_year[year].append(row)

    weights = {y: YEAR_WEIGHTS.get(y, 1.0) * len(by_year[y]) for y in by_year}
    total_w = sum(weights.values())
    print(f'read {n_read} | eligible {n_eligible} | per-year pool: '
          f'{ {y: len(v) for y, v in sorted(by_year.items())} }')

    picked: list[dict] = []
    for year in sorted(by_year):
        quota = round(args.target * weights[year] / total_w)
        pool = sorted(by_year[year], key=lambda r: seed64(f"{args.seed}|{r['arxiv_id']}"))
        picked.extend(pool[:quota])
    picked.sort(key=lambda r: r['arxiv_id'])

    n_files = 0
    with open(args.out, 'w') as f:
        for r in picked:
            n_files += len(r['versions'])
            f.write(json.dumps({
                'arxiv_id': r['arxiv_id'], 'title': r['title'],
                'n_versions': len(r['versions']),
                'chain_source': 'kaggle_pre2022',
                'categories': r.get('categories'),
                'versions': [{'v': v['v'], 'date': v['date'],
                              'size_kb': v.get('size_kb'),
                              'source_type': v.get('source_type'),
                              'stage': 'pre_chatgpt', 'stage_fine': 'pre_chatgpt'}
                             for v in r['versions']],
            }, ensure_ascii=False) + '\n')
    yr_hist = defaultdict(int)
    for r in picked:
        yr_hist[int(r['versions'][0]['date'][:4])] += 1
    print(f'picked {len(picked)} papers / {n_files} version files -> {args.out}')
    print(f'by v1 year: {dict(sorted(yr_hist.items()))}')


if __name__ == '__main__':
    main()
