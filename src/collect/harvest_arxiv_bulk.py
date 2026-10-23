"""Bulk-harvest arXiv metadata (with full version lists) via OAI-PMH ListRecords.

The sanctioned bulk path: ~1000 records per page with resumptionToken flow control,
arXivRaw format (title, authors, and ALL <version> entries with dates). One harvest
replaces thousands of per-title api/query searches plus per-id GetRecord calls.

Writes one JSONL row per record: {"arxiv_id", "title", "title_norm", "authors_raw",
"versions": [{"v", "date", "size_kb", "source_type"}], "n_versions", "categories",
"datestamp"}. Resumable: --out is append-only; a harvest interrupted mid-stream can be
resumed with --resume-token (printed on every page).

Run from repo root:
  python3 src/collect/harvest_arxiv_bulk.py --set cs --from-date 2024-06-01
"""
from __future__ import annotations

import argparse
import re
import sys
import time
import xml.etree.ElementTree as ET
from email.utils import parsedate_to_datetime
from pathlib import Path

sys.path.insert(0, 'src/collect')
from util import append_jsonl, http_get, norm_title, RateLimiter  # noqa: E402

OAI = 'https://oaipmh.arxiv.org/oai'
NS = {'oai': 'http://www.openarchives.org/OAI/2.0/',
      'raw': 'http://arxiv.org/OAI/arXivRaw/'}


def parse_size_kb(s: str | None) -> int | None:
    if not s:
        return None
    m = re.match(r'(\d+)', s)
    return int(m.group(1)) if m else None


def parse_record(rec) -> dict | None:
    header = rec.find('oai:header', NS)
    if header is None or header.get('status') == 'deleted':
        return None
    meta = rec.find('oai:metadata/raw:arXivRaw', NS)
    if meta is None:
        return None
    aid = (meta.findtext('raw:id', '', NS) or '').strip()
    title = ' '.join((meta.findtext('raw:title', '', NS) or '').split())
    versions = []
    for v in meta.findall('raw:version', NS):
        label = v.get('version', '')
        m = re.match(r'v(\d+)', label)
        if not m:
            continue
        date_txt = v.findtext('raw:date', '', NS)
        try:
            iso = parsedate_to_datetime(date_txt).isoformat()
        except Exception:
            iso = None
        versions.append({'v': int(m.group(1)), 'date': iso,
                         'size_kb': parse_size_kb(v.findtext('raw:size', '', NS)),
                         'source_type': v.findtext('raw:source_type', None, NS)})
    versions.sort(key=lambda x: x['v'])
    if not aid or not title or not versions:
        return None
    return {
        'arxiv_id': aid,
        'title': title,
        'title_norm': norm_title(title),
        'authors_raw': (meta.findtext('raw:authors', '', NS) or '').strip(),
        'versions': versions,
        'n_versions': len(versions),
        'categories': (meta.findtext('raw:categories', '', NS) or '').split(),
        'datestamp': header.findtext('oai:datestamp', '', NS),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description='Bulk OAI-PMH arXivRaw harvest.')
    ap.add_argument('--set', dest='set_spec', default='cs')
    ap.add_argument('--from-date', default='2024-06-01')
    ap.add_argument('--until-date', default=None)
    ap.add_argument('--out', default='data/processed/arxiv_bulk.jsonl')
    ap.add_argument('--resume-token', default=None)
    ap.add_argument('--max-pages', type=int, default=0, help='0 = until exhausted')
    args = ap.parse_args()

    limiter = RateLimiter(3.1)
    token = args.resume_token
    n_pages = n_rows = 0
    t0 = time.monotonic()
    while True:
        if token:
            params = {'verb': 'ListRecords', 'resumptionToken': token}
        else:
            params = {'verb': 'ListRecords', 'metadataPrefix': 'arXivRaw',
                      'set': args.set_spec, 'from': args.from_date}
            if args.until_date:
                params['until'] = args.until_date
        resp = http_get(OAI, params=params, limiter=limiter, timeout=180)
        root = ET.fromstring(resp.content)
        err = root.find('oai:error', NS)
        if err is not None:
            print(f'OAI error: {err.get("code")}: {(err.text or "")[:200]}')
            break
        lr = root.find('oai:ListRecords', NS)
        if lr is None:
            print('no ListRecords element; stopping')
            break
        page_rows = 0
        for rec in lr.findall('oai:record', NS):
            row = parse_record(rec)
            if row:
                append_jsonl(args.out, row)
                page_rows += 1
        n_pages += 1
        n_rows += page_rows
        tok_el = lr.find('oai:resumptionToken', NS)
        token = (tok_el.text or '').strip() if tok_el is not None else ''
        total = tok_el.get('completeListSize') if tok_el is not None else '?'
        el = time.monotonic() - t0
        print(f'page {n_pages}: +{page_rows} rows (total {n_rows}/{total}) '
              f'[{el:.0f}s] token={token[:40] or "DONE"}', flush=True)
        if not token:
            break
        if args.max_pages and n_pages >= args.max_pages:
            print(f'stopping at --max-pages; resume with --resume-token "{token}"')
            break
    print(f'\n=== harvest done: {n_rows} rows, {n_pages} pages -> {args.out}')


if __name__ == '__main__':
    main()
