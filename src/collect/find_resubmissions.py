"""E5: rejected-then-resubmitted ICLR papers with human scores in BOTH rounds.

Finds (round1, round2) pairs of ICLR submissions of the same paper where round 1
was rejected, both rounds carry >= 1 human review score, and round 2 was decided
(accepted or rejected). Each pair is then matched to the bulk arXiv harvest and,
per round, to the latest arXiv version visible by that year's decision cutoff —
i.e. the version that round's reviewers could have seen. Human score deltas are
the ground truth the LLM score deltas get anchored against.

Score-scale note: realized score scales differ by ICLR year (2017-19 & 2021: 1-10;
2020: {1,3,6,8}; 2022-2025: {1,3,5,6,8,10}). Raw deltas are flagged `same_scale`
only within a compatible group; `human_delta_z` (per-year z of the submission-mean
distribution) works across scales.

Coverage note: the bulk harvest only holds arXiv cs records with OAI datestamp >=
2024-06-01, so old pairs match only when the paper was touched on arXiv after that
date — expect good coverage only for recent year-pairs (reported per year-pair).

Pure-local: no arXiv API fallback in this script. OVERWRITES its three outputs.

Run from repo root:
  python3 src/collect/find_resubmissions.py
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from rapidfuzz import fuzz

sys.path.insert(0, 'src/collect')
from iclr_timeline import TIMELINES  # noqa: E402
from util import norm_title  # noqa: E402

# Years whose realized score scales are mutually compatible for RAW deltas.
RAW_SCALE_GROUPS = [{2022, 2023, 2024, 2025}, {2017, 2018, 2019, 2021}]
SURNAME_OVERLAP_MIN = 0.30   # fraction of the smaller surname set
ABSTRACT_SIM_MIN = 70.0      # rapidfuzz token_set_ratio


# ---------------------------------------------------------------- normalization

def normalize_decision(raw: str | None) -> str:
    """Case-insensitive normalization of the 15-ish raw OpenReview variants."""
    d = (raw or '').strip().lower()
    if d.startswith('accept'):
        return 'accepted'
    if d == 'reject':
        return 'rejected'
    if 'withdrawn' in d:
        return 'withdrawn'
    if 'desk' in d:
        return 'desk_rejected'
    return 'other'  # e.g. 'Invite to Workshop Track', empty


def surname_set(authors_str: str | None) -> set[str]:
    """Last tokens of comma-split full names, lowercased."""
    out = set()
    for name in (authors_str or '').split(','):
        toks = name.split()
        if toks:
            out.add(toks[-1].lower())
    return out


def surname_overlap(a: set[str], b: set[str]) -> float | None:
    """|intersection| over the smaller set (robust to author-list growth)."""
    if not a or not b:
        return None
    return len(a & b) / min(len(a), len(b))


def check_identity(r1: dict, r2: dict) -> dict:
    """Same-paper check for two same-title rows.

    Both author strings non-empty -> surname overlap >= 30%.
    Either empty -> abstract token_set_ratio >= 70 when both abstracts exist,
    else keep with weak_identity=True.
    """
    s1, s2 = surname_set(r1['authors']), surname_set(r2['authors'])
    if s1 and s2:
        ov = surname_overlap(s1, s2)
        return {'ok': ov >= SURNAME_OVERLAP_MIN, 'method': 'authors',
                'author_overlap': round(ov, 4), 'abstract_sim': None,
                'weak_identity': False}
    a1 = (r1.get('abstract') or '').strip()
    a2 = (r2.get('abstract') or '').strip()
    if a1 and a2:
        sim = float(fuzz.token_set_ratio(a1, a2))
        return {'ok': sim >= ABSTRACT_SIM_MIN, 'method': 'abstract',
                'author_overlap': None, 'abstract_sim': round(sim, 2),
                'weak_identity': False}
    return {'ok': True, 'method': 'none', 'author_overlap': None,
            'abstract_sim': None, 'weak_identity': True}


# ------------------------------------------------------------ version selection

def parse_dt(s: str) -> datetime:
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def visible_by(version_dt: datetime, cutoff: datetime) -> bool:
    """A version is visible to a round's reviewers iff posted on or before the
    decision-cutoff DAY (a version posted ON the decision day still counts)."""
    return version_dt.astimezone(timezone.utc).date() <= cutoff.date()


def select_versions(versions: list[dict], year1: int, year2: int
                    ) -> tuple[dict | None, dict | None, str | None]:
    """Latest version visible by each round's decision cutoff.

    Returns (v_round1, v_round2, fail_reason); reason is None on success.
    """
    tl1, tl2 = TIMELINES.get(year1), TIMELINES.get(year2)
    if tl1 is None:
        return None, None, f'no_timeline_year1_{year1}'
    if tl2 is None:
        return None, None, f'no_timeline_year2_{year2}'
    now = datetime.now(timezone.utc)
    parsed: list[tuple[datetime, dict]] = []
    for v in versions:
        try:
            dt = parse_dt(v['date'])
        except (KeyError, TypeError, ValueError):
            return None, None, 'bad_version_date'
        if dt.year < 1991 or dt > now:
            return None, None, 'insane_version_date'
        parsed.append((dt, v))
    parsed.sort(key=lambda t: (t[0], t[1].get('v', 0)))
    v1 = next((v for dt, v in reversed(parsed)
               if visible_by(dt, tl1.decision_cutoff)), None)
    v2 = next((v for dt, v in reversed(parsed)
               if visible_by(dt, tl2.decision_cutoff)), None)
    if v1 is None:
        return None, None, 'no_version_before_round1_decision'
    if v2 is None:
        return None, None, 'no_version_before_round2_decision'
    if v1.get('v') == v2.get('v'):
        return None, None, 'unchanged_between_rounds'
    return v1, v2, None


# ---------------------------------------------------------------- arXiv matching

def pick_bulk_match(cands: list[dict], sn: set[str]
                    ) -> tuple[dict | None, str]:
    """Mirror join_acl_bulk: single candidate wins; several -> strict-best
    surname-substring overlap against authors_raw, else ambiguous."""
    if len(cands) == 1:
        return cands[0], 'bulk_single'
    scored = []
    for c in cands:
        raw = (c.get('authors_raw') or '').lower()
        overlap = sum(1 for s in sn if s and s in raw)
        scored.append((overlap, c))
    scored.sort(key=lambda t: -t[0])
    if scored and (len(scored) == 1 or scored[0][0] > scored[1][0]):
        return scored[0][1], 'bulk_surname'
    return None, 'ambiguous'


# ------------------------------------------------------------------ pair finding

def load_parquet_rows(path: str) -> list[dict]:
    df = pd.read_parquet(path)
    rows = []
    for rec in df.itertuples(index=False):
        scores = rec.scores
        scores = [] if scores is None else list(np.asarray(scores).tolist())
        rows.append({
            'year': int(rec.year), 'id': rec.id, 'title': rec.title or '',
            'title_norm': norm_title(rec.title or ''),
            'abstract': rec.abstract or '', 'authors': rec.authors or '',
            'decision_raw': rec.decision or '',
            'decision': normalize_decision(rec.decision),
            'scores': scores,
        })
    return rows


def year_score_stats(rows: list[dict]) -> dict[int, tuple[float, float]]:
    """Per-year (mean, sd) of all-submission mean scores (rows with >= 1 score)."""
    by_year: dict[int, list[float]] = defaultdict(list)
    for r in rows:
        if r['scores']:
            by_year[r['year']].append(float(np.mean(r['scores'])))
    return {y: (float(np.mean(v)), float(np.std(v))) for y, v in by_year.items()}


def build_candidate_pairs(rows: list[dict], min_gap: int, max_gap: int
                          ) -> list[tuple[dict, dict]]:
    """(row1, row2): row1 rejected, row2 = next appearance of the same title,
    gap in [min_gap, max_gap]. Titles in 3+ years yield consecutive pairs."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        if r['title_norm']:
            groups[r['title_norm']].append(r)
    pairs = []
    for g in groups.values():
        if len(g) < 2:
            continue
        g.sort(key=lambda r: (r['year'], r['id']))
        for i, r1 in enumerate(g):
            if r1['decision'] != 'rejected':
                continue
            r2 = next((r for r in g[i + 1:] if r['year'] > r1['year']), None)
            if r2 is None:
                continue
            if min_gap <= r2['year'] - r1['year'] <= max_gap:
                pairs.append((r1, r2))
    return pairs


def same_scale_pair(year1: int, year2: int) -> bool:
    return any(year1 in grp and year2 in grp for grp in RAW_SCALE_GROUPS)


# ------------------------------------------------------------------------- main

def write_jsonl(path: str, rows: list[dict]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('w') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')


def main() -> None:
    ap = argparse.ArgumentParser(
        description='E5: rejected->resubmitted ICLR pairs with human scores in both rounds.')
    ap.add_argument('--parquet', default='data/raw/iclr/iclr25v2.parquet')
    ap.add_argument('--bulk', default='data/processed/arxiv_bulk.jsonl')
    ap.add_argument('--pairs-out', default='data/processed/e5_pairs.jsonl')
    ap.add_argument('--manifest-out', default='data/processed/e5_manifest.jsonl')
    ap.add_argument('--papers-out', default='data/processed/e5_papers.jsonl')
    ap.add_argument('--min-gap-years', type=int, default=1)
    ap.add_argument('--max-gap-years', type=int, default=3)
    args = ap.parse_args()

    rows = load_parquet_rows(args.parquet)
    stats = year_score_stats(rows)
    candidates = build_candidate_pairs(rows, args.min_gap_years, args.max_gap_years)
    print(f'parquet rows: {len(rows)} | candidate pairs '
          f'(round1 rejected, gap {args.min_gap_years}-{args.max_gap_years}y): '
          f'{len(candidates)}')

    # --- steps 1-2 per pair: identity, decisions, scores, deltas -------------
    pair_rows: list[dict] = []
    needed_titles: set[str] = set()
    for r1, r2 in candidates:
        ident = check_identity(r1, r2)
        rec: dict = {
            'pair_key': f"{r1['id']}+{r2['id']}",
            'title': r2['title'], 'title_norm': r2['title_norm'],
            'year1': r1['year'], 'year2': r2['year'],
            'gap_years': r2['year'] - r1['year'],
            'or_id1': r1['id'], 'or_id2': r2['id'],
            'decision1': r1['decision'], 'decision2': r2['decision'],
            'decision2_raw': r2['decision_raw'],
            'scores1': r1['scores'], 'scores2': r2['scores'],
            'n_reviews1': len(r1['scores']), 'n_reviews2': len(r2['scores']),
            'identity_method': ident['method'],
            'author_overlap': ident['author_overlap'],
            'abstract_sim': ident['abstract_sim'],
            'weak_identity': ident['weak_identity'],
            'mean1': None, 'mean2': None, 'human_delta_raw': None,
            'same_scale': same_scale_pair(r1['year'], r2['year']),
            'human_delta_z': None,
            'arxiv_id': None, 'arxiv_title': None, 'arxiv_match_method': None,
            'n_bulk_candidates': 0,
            'v_round1': None, 'v_round2': None, 'version_reason': None,
            'status': 'excluded', 'exclude_reason': None,
        }
        if not ident['ok']:
            rec['exclude_reason'] = 'identity_mismatch'
        elif r2['decision'] not in ('accepted', 'rejected'):
            rec['exclude_reason'] = f"decision2_{r2['decision']}"
        elif not r1['scores'] or not r2['scores']:
            rec['exclude_reason'] = 'missing_scores'
        if rec['exclude_reason'] is None:
            m1 = float(np.mean(r1['scores']))
            m2 = float(np.mean(r2['scores']))
            rec['mean1'], rec['mean2'] = m1, m2
            rec['human_delta_raw'] = m2 - m1
            s1, s2 = stats.get(r1['year']), stats.get(r2['year'])
            if s1 and s2 and s1[1] > 0 and s2[1] > 0:
                rec['human_delta_z'] = ((m2 - s2[0]) / s2[1]) - ((m1 - s1[0]) / s1[1])
            rec['status'] = 'pending_match'
            needed_titles.add(rec['title_norm'])
        pair_rows.append(rec)

    n_identity_ok = sum(1 for p in pair_rows
                        if p['exclude_reason'] != 'identity_mismatch')
    n_score_complete = sum(1 for p in pair_rows if p['status'] == 'pending_match')
    print(f'identity-passed: {n_identity_ok} | score-complete (decided round 2, '
          f'>=1 score both rounds): {n_score_complete}')

    # --- step 3: local bulk join (only titles we still need) -----------------
    by_title: dict[str, list[dict]] = defaultdict(list)
    n_bulk = 0
    with open(args.bulk) as f:
        for line in f:
            if not line.strip():
                continue
            n_bulk += 1
            row = json.loads(line)
            if row.get('title_norm') in needed_titles:
                by_title[row['title_norm']].append(row)
    print(f'bulk rows scanned: {n_bulk} | titles with >=1 bulk hit: '
          f'{len(by_title)}/{len(needed_titles)}')

    rows_by_id = {r['id']: r for r in rows}
    coverage: dict[tuple[int, int], list[int]] = defaultdict(lambda: [0, 0])
    manifest_rows: list[dict] = []
    papers_rows: list[dict] = []
    for rec in pair_rows:
        if rec['status'] != 'pending_match':
            continue
        yp = (rec['year1'], rec['year2'])
        coverage[yp][1] += 1
        cands = by_title.get(rec['title_norm'], [])
        rec['n_bulk_candidates'] = len(cands)
        if not cands:
            rec['exclude_reason'] = 'no_bulk_match'
            continue
        sn = (surname_set(rows_by_id[rec['or_id2']]['authors'])
              | surname_set(rows_by_id[rec['or_id1']]['authors']))
        pick, how = pick_bulk_match(cands, sn)
        if pick is None:
            rec['exclude_reason'] = 'bulk_match_ambiguous'
            continue
        coverage[yp][0] += 1
        rec['arxiv_id'] = pick['arxiv_id']
        rec['arxiv_title'] = pick['title']
        rec['arxiv_match_method'] = how

        # --- step 4: version selection -----------------------------------
        v1, v2, reason = select_versions(pick.get('versions', []),
                                         rec['year1'], rec['year2'])
        rec['version_reason'] = reason
        if reason is not None:
            rec['exclude_reason'] = f'version_{reason}'
            continue
        rec['v_round1'], rec['v_round2'] = v1, v2
        rec['status'] = 'manifest'
        rec['exclude_reason'] = None

        # --- step 5: manifest + papers rows -------------------------------
        manifest_rows.append({
            'arxiv_id': pick['arxiv_id'], 'openreview_id': rec['pair_key'],
            'iclr_year': 0, 'title': rec['title'], 'decision': 'unknown',
            'mean_score': None, 'scores': [], 'n_reviews': 0, 'n_versions': 2,
            'chain_source': 'bulk',
            'versions': [
                {**v1, 'stage': 'round1', 'stage_fine': 'round1'},
                {**v2, 'stage': 'round2', 'stage_fine': 'round2'},
            ],
            'strata': {'decision': 'unknown', 'version_bucket': '2',
                       'spans_stages': False},
            'categories': pick.get('categories', []), 'fresh': False,
            'e5': True, 'year1': rec['year1'], 'year2': rec['year2'],
            'or_id1': rec['or_id1'], 'or_id2': rec['or_id2'],
            'mean1': rec['mean1'], 'mean2': rec['mean2'],
            'human_delta_raw': rec['human_delta_raw'],
            'same_scale': rec['same_scale'],
            'human_delta_z': rec['human_delta_z'],
            'decision2': rec['decision2'],
        })
        auth2 = [a.strip() for a in rows_by_id[rec['or_id2']]['authors'].split(',')
                 if a.strip()]
        auth1 = [a.strip() for a in rows_by_id[rec['or_id1']]['authors'].split(',')
                 if a.strip()]
        authors = auth2 + [a for a in auth1 if a not in auth2]
        papers_rows.append({
            'openreview_id': rec['pair_key'], 'year': 0, 'title': rec['title'],
            'title_norm': rec['title_norm'], 'authors': authors,
            'decision_raw': '', 'decision': 'unknown', 'scores': [],
            'n_reviews': 0, 'mean_score': None,
            'abstract': rows_by_id[rec['or_id2']]['abstract'],
        })

    # mark still-pending (shouldn't happen) and write outputs ------------------
    for rec in pair_rows:
        if rec['status'] == 'pending_match':
            rec['status'] = 'excluded'
        if rec['status'] == 'excluded' and rec['exclude_reason'] is None:
            rec['exclude_reason'] = 'unknown'
    write_jsonl(args.pairs_out, pair_rows)
    write_jsonl(args.manifest_out, manifest_rows)
    write_jsonl(args.papers_out, papers_rows)

    # --- summary --------------------------------------------------------------
    print('\n=== E5 summary ===')
    print(f'candidate pairs: {len(pair_rows)} | identity-passed: {n_identity_ok} | '
          f'score-complete: {n_score_complete}')
    reasons = Counter(p['exclude_reason'] for p in pair_rows
                      if p['status'] == 'excluded')
    print(f'exclusions: {dict(reasons.most_common())}')
    print('\narXiv bulk match coverage by year-pair (matched/score-complete):')
    for yp in sorted(coverage):
        m, t = coverage[yp]
        print(f'  {yp[0]}->{yp[1]}: {m}/{t} ({m / t:.0%})')
    print('  NOTE: bulk harvest = arXiv cs with OAI datestamp >= 2024-06-01; old '
          'pairs match only if the paper was touched on arXiv after that date, so '
          'good coverage is expected only for recent year-pairs.')
    n_matched = sum(v[0] for v in coverage.values())
    print(f'\narXiv-matched: {n_matched} | version-selected (final manifest): '
          f'{len(manifest_rows)} | papers rows: {len(papers_rows)}')
    dup_arxiv = len(manifest_rows) - len({m["arxiv_id"] for m in manifest_rows})
    if dup_arxiv:
        print(f'  (manifest rows sharing an arxiv_id with another row: {dup_arxiv} '
              '— chained resubmissions; downloads are keyed arxiv_id+v and dedupe)')

    def delta_stats(vals: list[float], label: str) -> None:
        if not vals:
            print(f'{label}: n=0')
            return
        arr = np.array(vals)
        print(f'{label}: n={len(arr)} mean={arr.mean():+.3f} sd={arr.std():.3f} '
              f'positive={np.mean(arr > 0):.0%}')

    complete = [p for p in pair_rows if p['mean1'] is not None]
    final = [p for p in pair_rows if p['status'] == 'manifest']
    print('\nhuman_delta distributions:')
    delta_stats([p['human_delta_raw'] for p in complete if p['same_scale']],
                'raw (same-scale, all score-complete)')
    delta_stats([p['human_delta_z'] for p in complete
                 if p['human_delta_z'] is not None],
                'z    (all score-complete)')
    delta_stats([p['human_delta_raw'] for p in final if p['same_scale']],
                'raw (same-scale, final manifest)')
    delta_stats([p['human_delta_z'] for p in final
                 if p['human_delta_z'] is not None],
                'z    (final manifest)')
    print(f'\ndecision2 split (score-complete): '
          f'{dict(Counter(p["decision2"] for p in complete))}')
    print(f'decision2 split (final manifest): '
          f'{dict(Counter(p["decision2"] for p in final))}')


if __name__ == '__main__':
    main()
