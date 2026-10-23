"""Join ACL paper titles against the bulk arXiv harvest — local, instant matching.

Replaces per-title arXiv API searches AND per-id OAI chain fetches: the bulk harvest
rows already carry full version lists. Appends to the same acl_matches/acl_chains files
the API path uses (checkpoint-compatible: existing rows are kept, their papers skipped).
Papers with no bulk hit are left unwritten so a later API-matcher pass can pick them up.

Run from repo root:
  python3 src/collect/join_acl_bulk.py
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict

sys.path.insert(0, 'src/collect')
from util import append_jsonl, done_keys, read_jsonl  # noqa: E402


def surnames(full_names: list[str]) -> set[str]:
    return {n.split()[-1].lower() for n in full_names if n.split()}


def main() -> None:
    ap = argparse.ArgumentParser(description='Local title join: ACL papers x bulk harvest.')
    ap.add_argument('--papers', default='data/processed/acl_papers.jsonl')
    ap.add_argument('--bulk', default='data/processed/arxiv_bulk.jsonl')
    ap.add_argument('--matches-out', default='data/processed/acl_matches.jsonl')
    ap.add_argument('--chains-out', default='data/processed/acl_chains.jsonl')
    args = ap.parse_args()

    papers = read_jsonl(args.papers)
    done_match = done_keys(args.matches_out, 'openreview_id')
    done_chain = done_keys(args.chains_out, 'arxiv_id')

    by_title: dict[str, list[dict]] = defaultdict(list)
    n_bulk = 0
    for row in read_jsonl(args.bulk):
        by_title[row['title_norm']].append(row)
        n_bulk += 1
    print(f'bulk rows: {n_bulk} | distinct titles: {len(by_title)} | '
          f'acl papers: {len(papers)} | already matched: {len(done_match)}')

    n_hit = n_ambig = n_miss = n_chain = 0
    for p in papers:
        if p['openreview_id'] in done_match:
            continue
        cands = by_title.get(p['title_norm'], [])
        if not cands:
            n_miss += 1
            continue
        pick = None
        if len(cands) == 1:
            pick = cands[0]
        else:
            acl_sn = surnames(p.get('authors') or [])
            scored = []
            for c in cands:
                raw = (c.get('authors_raw') or '').lower()
                overlap = sum(1 for s in acl_sn if s and s in raw)
                scored.append((overlap, c))
            scored.sort(key=lambda t: -t[0])
            if scored and (len(scored) == 1 or scored[0][0] > scored[1][0]):
                pick = scored[0][1]
            else:
                n_ambig += 1
                continue
        n_hit += 1
        versions = pick['versions']
        acl_sn = surnames(p.get('authors') or [])
        raw_auth = (pick.get('authors_raw') or '').lower()
        overlap = (sum(1 for s in acl_sn if s and s in raw_auth) / len(acl_sn)
                   if acl_sn else None)
        append_jsonl(args.matches_out, {
            'openreview_id': p['openreview_id'], 'iclr_year': 0, 'title': p['title'],
            'decision': p.get('decision', 'accepted'), 'mean_score': None, 'scores': [],
            'n_reviews': 0, 'arxiv_id': pick['arxiv_id'], 'arxiv_title': pick['title'],
            'latest_version': versions[-1]['v'], 'published': versions[0]['date'],
            'updated': versions[-1]['date'], 'match_method': 'bulk',
            'match_score': 100.0, 'author_overlap': overlap, 'date_plausible': None,
        })
        if pick['arxiv_id'] not in done_chain:
            append_jsonl(args.chains_out, {
                'arxiv_id': pick['arxiv_id'], 'openreview_id': p['openreview_id'],
                'iclr_year': 0, 'title': p['title'],
                'decision': p.get('decision', 'accepted'), 'mean_score': None,
                'scores': [], 'n_reviews': 0, 'n_versions': len(versions),
                'chain_source': 'bulk',
                'versions': [{**v, 'stage': 'unknown', 'stage_fine': 'unknown'}
                             for v in versions],
            })
            n_chain += 1

    print(f'\n=== join summary ===')
    print(f'new matches: {n_hit} | ambiguous (skipped): {n_ambig} | '
          f'no bulk hit (left for API matcher): {n_miss} | new chains: {n_chain}')
    multi = sum(1 for r in read_jsonl(args.chains_out) if r.get('n_versions', 0) >= 2)
    print(f'chains with >=2 versions so far: {multi}')


if __name__ == '__main__':
    main()
