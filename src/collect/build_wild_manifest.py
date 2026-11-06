"""Select WILD post-ChatGPT revision papers: latest revision on/after the
cutoff (default 2023-01-01) — the scan corpus for the AI-edit prevalence
study ("reverse telescope": the trained attributor scores real revisions).

Counterpart of build_pre2022_manifest.py. There, EVERY version predates
ChatGPT, so the human label is construction-valid; here the revisions have
UNKNOWN authorship — that is the point. Filters mirror the pre-2022 set so
the scan distribution matches training: cs.* primary, 2-6 versions,
>=14-day span, v1 no older than 2015. Stratified EQUALLY by last-revision
year (the prevalence time axis), deterministic hash order per year.

All versions are listed; which (old, new) pair the scan aligns (v1 vs
last, or consecutive) is the scan script's decision, not the manifest's.

Run from repo root:
  python3 src/collect/build_wild_manifest.py --target 3000
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from datetime import datetime

sys.path.insert(0, 'src/collect')
from util import read_jsonl  # noqa: E402

MIN_V1_YEAR = 2015
MIN_SPAN_DAYS = 14
MAX_VERSIONS = 6


def seed64(key: str) -> int:
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], 'big')


def main() -> None:
    ap = argparse.ArgumentParser(description='Wild post-ChatGPT revision manifest.')
    ap.add_argument('--bulk', default='data/processed/arxiv_bulk_kaggle.jsonl')
    ap.add_argument('--out', default='data/processed/wild_manifest.jsonl')
    ap.add_argument('--revised-after', default='2023-01-01',
                    help='keep papers whose LAST version is on/after this date')
    ap.add_argument('--target', type=int, default=3000)
    ap.add_argument('--seed', type=int, default=20261105)
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
        if not all(dates) or max(dates) < args.revised_after:
            continue  # latest revision must land in the LLM era
        try:
            v1_year = int(dates[0][:4])
            rev_year = int(max(dates)[:4])
        except ValueError:
            continue
        if v1_year < MIN_V1_YEAR:
            continue
        try:
            span = (datetime.fromisoformat(max(dates)[:19])
                    - datetime.fromisoformat(dates[0][:19])).days
        except ValueError:
            continue
        if span < MIN_SPAN_DAYS:
            continue
        n_eligible += 1
        by_year[rev_year].append(row)

    years = sorted(by_year)
    print(f'read {n_read} | eligible {n_eligible} | per-revision-year pool: '
          f'{ {y: len(v) for y, v in sorted(by_year.items())} }')
    if not years:
        sys.exit('no eligible papers — check --bulk and --revised-after')

    # equal quota per revision year (the time axis), topped up newest-first
    pools = {y: sorted(by_year[y], key=lambda r: seed64(f"{args.seed}|{r['arxiv_id']}"))
             for y in years}
    quota = {y: min(args.target // len(years), len(pools[y])) for y in years}
    rem = args.target - sum(quota.values())
    for y in sorted(years, reverse=True):
        take = min(rem, len(pools[y]) - quota[y])
        quota[y] += take
        rem -= take
    picked = [r for y in years for r in pools[y][:quota[y]]]
    picked.sort(key=lambda r: r['arxiv_id'])

    n_files = 0
    with open(args.out, 'w') as f:
        for r in picked:
            n_files += len(r['versions'])
            f.write(json.dumps({
                'arxiv_id': r['arxiv_id'], 'title': r['title'],
                'n_versions': len(r['versions']),
                'chain_source': 'kaggle_wild',
                'categories': r.get('categories'),
                'revision_year': int(max(v['date'] for v in r['versions'])[:4]),
                'versions': [{'v': v['v'], 'date': v['date'],
                              'size_kb': v.get('size_kb'),
                              'source_type': v.get('source_type'),
                              'stage': 'wild_post_chatgpt',
                              'stage_fine': 'wild_post_chatgpt'}
                             for v in r['versions']],
            }, ensure_ascii=False) + '\n')
    yr_hist = defaultdict(int)
    for r in picked:
        yr_hist[int(max(v['date'] for v in r['versions'])[:4])] += 1
    print(f'picked {len(picked)} papers / {n_files} version files -> {args.out}')
    print(f'by revision year: {dict(sorted(yr_hist.items()))}')


if __name__ == '__main__':
    main()
