"""Convert the Kaggle arXiv metadata snapshot into our bulk-harvest schema.

The snapshot (arxiv-metadata-oai-snapshot.json, one JSON object per line) covers ALL of
arXiv with full version lists — it replaces OAI harvesting for any historical window.

Run from repo root:
  python3 src/collect/convert_kaggle_snapshot.py
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from email.utils import parsedate_to_datetime

sys.path.insert(0, 'src/collect')
from util import norm_title  # noqa: E402

CAT_PREFIXES = ('cs.', 'stat.', 'eess.')


def main() -> None:
    ap = argparse.ArgumentParser(description='Kaggle snapshot -> bulk schema.')
    ap.add_argument('--snapshot', default='data/raw/kaggle/arxiv-metadata-oai-snapshot.json')
    ap.add_argument('--out', default='data/processed/arxiv_bulk_kaggle.jsonl')
    ap.add_argument('--all-categories', action='store_true',
                    help='keep every record (default: only cs./stat./eess. primaries)')
    args = ap.parse_args()

    n_in = n_kept = 0
    with open(args.snapshot) as fin, open(args.out, 'w') as fout:
        for line in fin:
            n_in += 1
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            cats = (r.get('categories') or '').split()
            if not args.all_categories:
                if not any(c.startswith(CAT_PREFIXES) for c in cats):
                    continue
            title = ' '.join((r.get('title') or '').split())
            versions = []
            for v in (r.get('versions') or []):
                m = re.match(r'v(\d+)', v.get('version', ''))
                if not m:
                    continue
                try:
                    iso = parsedate_to_datetime(v.get('created', '')).isoformat()
                except Exception:
                    iso = None
                versions.append({'v': int(m.group(1)), 'date': iso,
                                 'size_kb': None, 'source_type': None})
            versions.sort(key=lambda x: x['v'])
            if not title or not versions:
                continue
            fout.write(json.dumps({
                'arxiv_id': r.get('id', ''),
                'title': title,
                'title_norm': norm_title(title),
                'authors_raw': (r.get('authors') or '').strip(),
                'versions': versions,
                'n_versions': len(versions),
                'categories': cats,
                'datestamp': r.get('update_date', ''),
            }, ensure_ascii=False) + '\n')
            n_kept += 1
            if n_kept % 200000 == 0:
                print(f'  {n_kept} kept / {n_in} read', flush=True)
    print(f'=== done: kept {n_kept} of {n_in} records -> {args.out}')


if __name__ == '__main__':
    main()
