#!/usr/bin/env python3
"""Independent re-implementation of the planned E1-H analysis (version-slope comparison).

Usage:
    python3 recompute.py --logs <main_log> <sub_log> [--b1-all] [--per-paper]

Prints ONE JSON object. Reads only the two logs, the manifest and the existence of the
text files (positions). Writes nothing. Fixed random seeds, so a re-run on the same logs
prints the same numbers.

Specification implemented (choices made where the spec is silent are marked CHOICE):

  position   0-based index of a version among the paper's manifest versions that have a
             text file data/processed/texts/<id with / -> _>/v<k>.md, sorted by v.
  usable     parsed_ok is True (strictly the JSON value true) and rating is a number;
             for arm harness_sub additionally process_ok is True and, unless --b1-all,
             not reminded and not review_before_report.
             CHOICE: usability is evaluated with the variant's rule BEFORE
             de-duplication (so under the clean rule an earlier clean row is not
             displaced by a later parsed-but-unclean row of the same key).
  dedup      one row per key; the LAST usable row in file order wins; failed rows never
             replace anything. CHOICE: file order = main log, then sub log.
             CHOICE: rows of every cfg_sig are pooled (the spec does not mention
             cfg_sig); the number of distinct cfg_sig values per arm is reported.
  cells      (arxiv_id, v) -> arm -> ratings, only versions with a planned position.
  comparison (a, b): cells with >= 1 rating in both arms; papers with >= 2 such versions.
             Per paper and arm: version mean (unweighted mean over replicates), OLS slope
             of version means on the PLANNED positions (not re-indexed), Sxx, and
             level = mean of version means.  d = slope_b - slope_a.
  pooled     sum(Sxx*d)/sum(Sxx); sign-flip p (two-sided, Monte Carlo,
             p = (1 + #{|T*| >= |T|}) / (B + 1); exact enumeration if <= 16 papers);
             paper bootstrap 95% percentile CI (weights resampled with the papers).
  unweighted mean(d), one-sample t-test, t-based 95% CI.
  level      mean(level_b - level_a), paper bootstrap 95% percentile CI, sign-flip p.
  per arm    usable runs; mean SD across replicates per paper-version with >= 2
             replicates (CHOICE: sample SD, ddof=1; ddof=0 also reported).
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import sys
from collections import Counter, defaultdict

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '1')  # stay single-threaded: other jobs share this machine

import numpy as np
from scipy import stats

ARMS = ('direct', 'harness', 'harness_sub')
COMPARISONS = (('direct', 'harness'), ('direct', 'harness_sub'), ('harness', 'harness_sub'))
ALLOWED_RATINGS = (1, 3, 5, 6, 8, 10)
DEFAULT_REPO = '/Users/madili/Desktop/review papaer research'
TOL = 1e-12


# ------------------------------------------------------------------ inputs

def read_log(path: str) -> tuple[list[dict], int]:
    """Rows in file order; (rows, number of unparseable non-empty lines)."""
    rows, bad = [], 0
    with open(path, 'rb') as fh:
        data = fh.read()
    for line in data.split(b'\n'):  # bytes split: never break on U+2028 etc.
        if not line.strip():
            continue
        try:
            obj = json.loads(line.decode('utf-8', 'replace'))
        except ValueError:
            bad += 1
            continue
        if isinstance(obj, dict):
            rows.append(obj)
        else:
            bad += 1
    return rows, bad


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for block in iter(lambda: fh.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def planned_positions(manifest_path: str, texts_dir: str) -> tuple[dict, dict]:
    """{(arxiv_id, v): position}, and per-paper info. Uses manifest + text files only."""
    pos: dict = {}
    info: dict = {}
    with open(manifest_path, 'r', encoding='utf-8') as fh:
        for line in fh:
            if not line.strip():
                continue
            paper = json.loads(line)
            aid = paper['arxiv_id']
            safe = aid.replace('/', '_')
            vs = sorted({int(x['v']) for x in paper['versions']})
            have = [v for v in vs if os.path.isfile(os.path.join(texts_dir, safe, f'v{v}.md'))]
            for i, v in enumerate(have):
                pos[(aid, v)] = i
            info[aid] = {'manifest_versions': vs, 'text_versions': have}
    return pos, info


# ------------------------------------------------------------------ row rules

def is_usable(row: dict, b1_all: bool) -> bool:
    if row.get('parsed_ok') is not True:
        return False
    r = row.get('rating')
    if isinstance(r, bool) or not isinstance(r, (int, float)):
        return False
    if row.get('arm') == 'harness_sub':
        if row.get('process_ok') is not True:
            return False
        if not b1_all and (row.get('reminded') or row.get('review_before_report')):
            return False
    return True


def dedup_usable(rows: list[dict], b1_all: bool) -> tuple[dict, dict]:
    """key -> last usable row in file order; and key -> number of usable rows."""
    kept: dict = {}
    n_usable: Counter = Counter()
    for row in rows:
        if is_usable(row, b1_all):
            kept[row.get('key')] = row
            n_usable[row.get('key')] += 1
    return kept, n_usable


# ------------------------------------------------------------------ statistics

def ols_slope(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    xc = x - x.mean()
    sxx = float(np.sum(xc * xc))
    return float(np.sum(xc * (y - y.mean())) / sxx), sxx


def signflip_p(values: np.ndarray, weights: np.ndarray, draws: int, seed: int) -> dict:
    """Two-sided sign-flip p for T = sum(w * v) / sum(w)."""
    n = len(values)
    wv = weights * values / weights.sum()
    obs = abs(float(wv.sum()))
    if n <= 16:  # exact: all 2^n sign patterns
        signs = np.array(list(itertools.product((-1.0, 1.0), repeat=n)))
        t = np.abs(signs @ wv)
        return {'p': float(np.mean(t >= obs - TOL)), 'method': 'exact', 'draws': int(len(t))}
    rng = np.random.default_rng(seed)
    count = 0
    left = draws
    while left > 0:
        m = min(left, 20000)
        signs = rng.integers(0, 2, size=(m, n)).astype(np.float64) * 2.0 - 1.0
        count += int(np.sum(np.abs(signs @ wv) >= obs - TOL))
        left -= m
    return {'p': (count + 1) / (draws + 1), 'method': 'monte_carlo', 'draws': draws,
            'n_as_extreme': count}


def boot_ci(values: np.ndarray, weights: np.ndarray, resamples: int, seed: int) -> list[float]:
    """Paper-bootstrap 95% percentile CI of sum(w * v) / sum(w)."""
    n = len(values)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(resamples, n))
    est = (weights[idx] * values[idx]).sum(axis=1) / weights[idx].sum(axis=1)
    lo, hi = np.percentile(est, [2.5, 97.5])
    return [float(lo), float(hi)]


def t_summary(d: np.ndarray) -> dict:
    n = len(d)
    out = {'mean': float(d.mean()) if n else None, 'n': n, 'sd': None, 't': None,
           'df': n - 1, 'p': None, 'ci95': None}
    if n >= 2:
        sd = float(d.std(ddof=1))
        out['sd'] = sd
        if sd > 0:
            res = stats.ttest_1samp(d, 0.0)
            half = stats.t.ppf(0.975, n - 1) * sd / math.sqrt(n)
            out.update(t=float(res.statistic), p=float(res.pvalue),
                       ci95=[float(d.mean() - half), float(d.mean() + half)])
    return out


def compare(cells: dict, pos: dict, a: str, b: str, draws: int, resamples: int,
            seed: int, per_paper: bool) -> dict:
    by_paper: dict = defaultdict(list)
    n_shared_cells = 0
    for (aid, v), arms in cells.items():
        if arms.get(a) and arms.get(b):
            by_paper[aid].append(v)
            n_shared_cells += 1
    papers = sorted(p for p, vs in by_paper.items() if len(vs) >= 2)
    out: dict = {
        'arm_a': a, 'arm_b': b,
        'n_cells_shared_before_paper_filter': n_shared_cells,
        'n_papers_with_any_shared_cell': len(by_paper),
        'n_papers_dropped_fewer_than_2_versions': len(by_paper) - len(papers),
        'n_papers': len(papers), 'n_cells': sum(len(by_paper[p]) for p in papers),
    }
    if len(papers) < 2:
        out['error'] = 'fewer than 2 papers'
        return out

    rows = []
    balance: Counter = Counter()
    n_runs = {a: 0, b: 0}
    for aid in papers:
        vs = sorted(by_paper[aid])
        x = np.array([pos[(aid, v)] for v in vs], dtype=float)
        rec = {'arxiv_id': aid, 'versions': vs, 'positions': [int(t) for t in x]}
        for arm in (a, b):
            means = np.array([np.mean(cells[(aid, v)][arm]) for v in vs], dtype=float)
            slope, sxx = ols_slope(x, means)
            rec[f'slope_{arm}'] = slope
            rec[f'level_{arm}'] = float(means.mean())
            rec['sxx'] = sxx
            n_runs[arm] += sum(len(cells[(aid, v)][arm]) for v in vs)
        for v in vs:
            balance[(len(cells[(aid, v)][a]), len(cells[(aid, v)][b]))] += 1
        rec['d'] = rec[f'slope_{b}'] - rec[f'slope_{a}']
        rec['level_diff'] = rec[f'level_{b}'] - rec[f'level_{a}']
        rows.append(rec)

    sxx = np.array([r['sxx'] for r in rows])
    d = np.array([r['d'] for r in rows])
    lev = np.array([r['level_diff'] for r in rows])
    sa = np.array([r[f'slope_{a}'] for r in rows])
    sb = np.array([r[f'slope_{b}'] for r in rows])
    ones = np.ones(len(rows))

    out['n_runs_used'] = n_runs
    out['replicate_balance'] = {
        'cells_by_reps_a_b': {f'{i}/{j}': n for (i, j), n in sorted(balance.items())},
        'n_cells_equal_reps': sum(n for (i, j), n in balance.items() if i == j),
        'n_cells_3_and_3': balance.get((3, 3), 0),
        'mean_reps_per_cell': {a: n_runs[a] / out['n_cells'], b: n_runs[b] / out['n_cells']},
    }
    out['versions_per_paper'] = {str(k): n for k, n in
                                 sorted(Counter(len(r['versions']) for r in rows).items())}
    out['sum_sxx'] = float(sxx.sum())
    out['pooled'] = {
        f'slope_{a}': float(np.sum(sxx * sa) / sxx.sum()),
        f'slope_{b}': float(np.sum(sxx * sb) / sxx.sum()),
        'diff_b_minus_a': float(np.sum(sxx * d) / sxx.sum()),
        'signflip': signflip_p(d, sxx, draws, seed + 1),
        'boot_ci95': boot_ci(d, sxx, resamples, seed + 2),
        'boot_resamples': resamples,
    }
    out['unweighted'] = {
        f'mean_slope_{a}': float(sa.mean()), f'mean_slope_{b}': float(sb.mean()),
        'diff_b_minus_a': t_summary(d),
        'n_d_positive': int(np.sum(d > TOL)), 'n_d_negative': int(np.sum(d < -TOL)),
        'n_d_zero': int(np.sum(np.abs(d) <= TOL)),
    }
    out['level'] = {
        f'mean_level_{a}': float(np.mean([r[f'level_{a}'] for r in rows])),
        f'mean_level_{b}': float(np.mean([r[f'level_{b}'] for r in rows])),
        'diff_b_minus_a': float(lev.mean()),
        'boot_ci95': boot_ci(lev, ones, resamples, seed + 3),
        'signflip': signflip_p(lev, ones, draws, seed + 4),
        'boot_resamples': resamples,
        'n_diff_positive': int(np.sum(lev > TOL)), 'n_diff_negative': int(np.sum(lev < -TOL)),
        'n_diff_zero': int(np.sum(np.abs(lev) <= TOL)),
    }
    if per_paper:
        out['per_paper'] = rows
    return out


# ------------------------------------------------------------------ main

def main() -> None:
    ap = argparse.ArgumentParser(description='Independent recomputation of the E1-H analysis.')
    ap.add_argument('--logs', nargs=2, required=True, metavar=('MAIN_LOG', 'SUB_LOG'))
    ap.add_argument('--b1-all', action='store_true',
                    help='harness_sub: keep reminded / review_before_report runs')
    ap.add_argument('--manifest', default=DEFAULT_REPO + '/data/processed/pilot_manifest.jsonl')
    ap.add_argument('--texts', default=DEFAULT_REPO + '/data/processed/texts')
    ap.add_argument('--draws', type=int, default=100000, help='sign-flip draws (>= 20000)')
    ap.add_argument('--resamples', type=int, default=20000, help='bootstrap resamples (>= 10000)')
    ap.add_argument('--seed', type=int, default=20261112)
    ap.add_argument('--reps', type=int, default=3, help='planned replicates per paper-version')
    ap.add_argument('--per-paper', action='store_true', help='include per-paper slopes')
    cfg = ap.parse_args()
    if cfg.draws < 20000 or cfg.resamples < 10000:
        ap.error('--draws must be >= 20000 and --resamples >= 10000')

    pos, info = planned_positions(cfg.manifest, cfg.texts)
    all_rows: list[dict] = []
    inputs = []
    for path in cfg.logs:
        rows, bad = read_log(path)
        inputs.append({'path': path, 'sha256': sha256_file(path), 'rows': len(rows),
                       'unparseable_lines': bad,
                       'arms': dict(Counter(str(r.get('arm')) for r in rows))})
        all_rows.extend(rows)

    # ---- sanity checks on the raw rows (reported, never silently "fixed")
    key_mismatch = [r.get('key') for r in all_rows
                    if r.get('key') != f"{r.get('arxiv_id')}v{r.get('v')}|{r.get('arm')}|r{r.get('rep')}"]
    kept, n_usable = dedup_usable(all_rows, cfg.b1_all)
    multi_usable = {k: n for k, n in n_usable.items() if n > 1}
    no_position = sorted(k for k, r in kept.items() if (r['arxiv_id'], r['v']) not in pos)
    bad_rating = sorted(k for k, r in kept.items() if r['rating'] not in ALLOWED_RATINGS)
    bad_rep = sorted(k for k, r in kept.items() if r.get('rep') not in range(cfg.reps))
    bad_arm = sorted(str(k) for k, r in kept.items() if r.get('arm') not in ARMS)
    sha_by_cell: dict = defaultdict(set)
    for r in kept.values():
        sha_by_cell[(r['arxiv_id'], r['v'])].add(r.get('text_sha'))
    sha_conflicts = sorted(f'{a}v{v}' for (a, v), s in sha_by_cell.items() if len(s) > 1)

    # ---- cells
    cells: dict = defaultdict(lambda: defaultdict(list))
    for k in sorted(kept):
        r = kept[k]
        if (r['arxiv_id'], r['v']) in pos and r.get('arm') in ARMS:
            cells[(r['arxiv_id'], r['v'])][r['arm']].append(float(r['rating']))

    per_arm = {}
    for arm in ARMS:
        arm_rows = [r for r in all_rows if r.get('arm') == arm]
        arm_kept = [r for r in kept.values() if r.get('arm') == arm]
        arm_cells = {c: v[arm] for c, v in cells.items() if v.get(arm)}
        multi = [np.array(v) for v in arm_cells.values() if len(v) >= 2]
        ratings = [x for v in arm_cells.values() for x in v]
        entry = {
            'rows_in_log': len(arm_rows),
            'distinct_keys_in_log': len({r.get('key') for r in arm_rows}),
            'rows_parsed_ok': sum(r.get('parsed_ok') is True for r in arm_rows),
            'rows_usable_before_dedup': sum(is_usable(r, cfg.b1_all) for r in arm_rows),
            'usable_runs': len(arm_kept),
            'usable_runs_with_position': len(ratings),
            'planned_runs': len(pos) * cfg.reps,
            'cells_with_data': len(arm_cells),
            'papers_with_data': len({c[0] for c in arm_cells}),
            'cells_by_n_reps': {str(k): n for k, n in
                                sorted(Counter(len(v) for v in arm_cells.values()).items())},
            'cfg_sigs_among_usable': dict(Counter(str(r.get('cfg_sig')) for r in arm_kept)),
            'rating_hist': {str(int(k)): n for k, n in sorted(Counter(ratings).items())},
            'mean_rating_over_runs': float(np.mean(ratings)) if ratings else None,
            'mean_of_cell_means': float(np.mean([np.mean(v) for v in arm_cells.values()]))
            if arm_cells else None,
            'rep_sd': {
                'n_cells_ge2': len(multi),
                'mean_sd_ddof1': float(np.mean([v.std(ddof=1) for v in multi])) if multi else None,
                'mean_sd_ddof0': float(np.mean([v.std(ddof=0) for v in multi])) if multi else None,
                'pooled_sd': float(math.sqrt(sum(((v - v.mean()) ** 2).sum() for v in multi)
                                             / sum(len(v) - 1 for v in multi))) if multi else None,
                'share_cells_all_reps_equal': float(np.mean([np.ptp(v) == 0 for v in multi]))
                if multi else None,
            },
        }
        if arm == 'harness_sub':
            entry['process'] = {
                'process_ok_true': sum(r.get('process_ok') is True for r in arm_rows),
                'process_ok_false': sum(r.get('process_ok') is False for r in arm_rows),
                'process_ok_null': sum(r.get('process_ok') is None for r in arm_rows),
                'parsed_and_reminded': sum(r.get('parsed_ok') is True and bool(r.get('reminded'))
                                           for r in arm_rows),
                'parsed_and_review_before_report': sum(
                    r.get('parsed_ok') is True and bool(r.get('review_before_report'))
                    for r in arm_rows),
            }
        per_arm[arm] = entry

    result = {
        'variant': 'b1-all' if cfg.b1_all else 'clean',
        'settings': {'signflip_draws': cfg.draws, 'bootstrap_resamples': cfg.resamples,
                     'seed': cfg.seed, 'sd_ddof': 1,
                     'signflip_p_formula': '(1 + #{|T*| >= |T|}) / (draws + 1); exact if <= 16 papers'},
        'inputs': inputs,
        'design': {'papers_in_manifest': len(info), 'versions_in_manifest':
                   sum(len(i['manifest_versions']) for i in info.values()),
                   'versions_with_text': len(pos),
                   'versions_without_text': sorted(
                       f'{a}v{v}' for a, i in info.items()
                       for v in i['manifest_versions'] if v not in i['text_versions'])},
        'data_checks': {
            'rows_total': len(all_rows),
            'rows_key_field_mismatch': key_mismatch,
            'keys_with_several_usable_rows': multi_usable,
            'usable_keys_without_planned_position': no_position,
            'usable_keys_rating_off_scale': bad_rating,
            'usable_keys_rep_out_of_range': bad_rep,
            'usable_keys_unknown_arm': bad_arm,
            'cells_with_conflicting_text_sha': sha_conflicts,
            'usable_rows_coerced': sorted(k for k, r in kept.items() if r.get('coerced')),
            'usable_rows_truncated': sorted(k for k, r in kept.items() if r.get('truncated')),
        },
        'per_arm': per_arm,
        'comparisons': {},
    }
    for i, (a, b) in enumerate(COMPARISONS):
        result['comparisons'][f'{a}__vs__{b}'] = compare(
            cells, pos, a, b, cfg.draws, cfg.resamples, cfg.seed + 100 * (i + 1), cfg.per_paper)
    json.dump(result, sys.stdout, indent=1, sort_keys=False)
    sys.stdout.write('\n')


if __name__ == '__main__':
    main()
