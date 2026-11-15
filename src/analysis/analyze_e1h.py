"""E1-H analysis: does an agent harness change the version-score slope?

Input: the logs written by src/review/run_e1_harness.py (arms 'direct' = A and
'harness' = B0) and src/review/run_e1_harness_sub.py (arm 'harness_sub' = B1, the harness
forced to complete one subagent review before scoring).

The unit of inference is the paper. Replicate ratings are averaged per paper-version
and arm. A version's position is its index among the paper's versions that have a text
(0, 1, 2, ... — the coding of the original E1), fixed BEFORE any filtering, so a
missing cell never compresses the scale. Each comparison uses only the paper-versions
scored in both of its arms, so the two arms see identical inputs.

Analysis plan, fixed on 2026-11-12 before any full-run data existed:

  PRIMARY COMPARISON   harness - direct (B0 - A).
  SECONDARY            harness_sub - direct (B1 - A) and harness_sub - harness (B1 - B0).
                       B1 differs from B0 by a required process AND by roughly three
                       times the model calls, so a B1 effect cannot be attributed to
                       agent structure without a compute-matched direct pipeline.
  B1 USABLE RUNS       only runs whose process the program verified (subagent had the
                       paper, its report was delivered, the review came after it) AND
                       that were clean: no reminder, no review written before the
                       report. --b1-all repeats the analysis on every verified run as a
                       sensitivity check. Process failures are counted and reported,
                       never silently dropped.
  WHAT B1 IS           two turns (paper delivered with "reply READY", then the process
                       instruction) plus one forked subagent. B1 - B0 therefore bundles
                       the two-turn delivery with the subagent step; separating them
                       needs a control arm with the same two turns and no subagent.

  For each comparison:
  PRIMARY ESTIMATE     pooled slope difference, in rating points per revision step. Each
                       paper's slope difference is weighted by Sxx, the spread of its
                       version positions — the within-paper regression slope, the same
                       quantity as the "+0.11 per step" reported for E1; it equals the
                       version x arm coefficient of  rating ~ paper*arm + pos + pos:arm.
                       Test: sign-flip randomization over papers (under no effect each
                       paper's slope difference is equally likely to have either sign).
                       In simulations of this design its false-positive rate is 4.5% at
                       the 5% level; the paper bootstrap's is 6-7% because three long
                       papers hold 45% of the weight, so the bootstrap is used only for
                       the 95% CI, which therefore runs slightly narrow.
  SECONDARY ESTIMATE   unweighted mean of the per-paper slope differences with a paired
                       t-test. Every paper counts equally, so two-version papers
                       dominate it.

Changed after the full data were in (2026-11-12, found by an independent check): the
tolerance of the sign-flip test, which had undercounted exact ties (primary p 0.270 ->
0.275; exact enumeration gives 0.273), and the wording of the run-count lines. No
estimate or interval changed.

The two estimates answer slightly different questions (per revision step pooled over the
corpus vs. per paper) and can disagree; both are printed, plus the split by number of
versions. A larger slope means later versions score higher. It does not show the scores
are more accurate.

Run from the repo root:
    python3 src/analysis/analyze_e1h.py
"""
from __future__ import annotations

import argparse
import json
import math
import statistics as st
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

ARM_LABEL = {'direct': 'A direct call', 'harness': 'B0 harness',
             'harness_sub': 'B1 harness + forced subagent review'}
COMPARISONS = (('direct', 'harness', 'PRIMARY'), ('direct', 'harness_sub', 'secondary'),
               ('harness', 'harness_sub', 'secondary'))
DEFAULT_LOGS = ('data/processed/e1h_reviews_log.jsonl',
                'data/processed/e1h_sub_reviews_log.jsonl')


def read_log(path: str) -> list[dict]:
    """Tolerant reader: the runner may be appending, so a torn last line is skipped."""
    rows = []
    for line in Path(path).read_text(encoding='utf-8').split('\n'):
        if line.strip():
            try:
                rows.append(json.loads(line))
            except ValueError:
                continue
    return rows


B1_ALL = False  # set by --b1-all: sensitivity analysis on every verified B1 run


def usable(r: dict) -> bool:
    """A rating that may enter the analysis. For B1 the primary set is the clean runs:
    process verified, no reminder needed, and no review written before the subagent's
    report (a late fork would be seeded with the main agent's own review)."""
    if not r.get('parsed_ok'):
        return False
    if r.get('arm') != 'harness_sub':
        return True
    if r.get('process_ok') is not True:
        return False
    return B1_ALL or not (r.get('reminded') or r.get('review_before_report'))


def slope_and_sxx(pos: list[float], y: list[float]) -> tuple[float, float]:
    x = np.asarray(pos, dtype=float)
    yy = np.asarray(y, dtype=float)
    xc = x - x.mean()
    sxx = float((xc ** 2).sum())
    return float((xc * (yy - yy.mean())).sum() / sxx), sxx


def pooled(values, weights) -> float:
    v, w = np.asarray(values, dtype=float), np.asarray(weights, dtype=float)
    return float((w * v).sum() / w.sum())


def boot_ci(values, weights, n_boot: int = 10000, seed: int = 42) -> tuple[float, float]:
    """Paper bootstrap 95% percentile CI of the weighted mean."""
    v, w = np.asarray(values, dtype=float), np.asarray(weights, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), size=(n_boot, len(v)))
    est = (w[idx] * v[idx]).sum(axis=1) / w[idx].sum(axis=1)
    return float(np.percentile(est, 2.5)), float(np.percentile(est, 97.5))


def flip_p(values, weights, n_flip: int = 20000, seed: int = 42) -> float:
    """Two-sided sign-flip randomization p-value for the weighted mean. The statistic
    lives on a coarse lattice, so many sign patterns tie with the observed value exactly;
    the tolerance must exceed the rounding applied to the inputs or ties are undercounted."""
    wd = np.asarray(weights, dtype=float) * np.asarray(values, dtype=float)
    obs = abs(wd.sum())
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(n_flip, len(wd)))
    return float(((np.abs((signs * wd).sum(axis=1)) >= obs - 1e-6).sum() + 1) / (n_flip + 1))


def fmt_p(p: float) -> str:
    return 'p < 0.0001' if p < 0.0001 else f'p = {p:.4f}'


def fmt(mean: float, ci: tuple[float, float]) -> str:
    return f"{mean:+.3f} (95% CI [{ci[0]:+.3f}, {ci[1]:+.3f}])"


def compare(a: str, b: str, role: str, cell: dict, planned_pos: dict, md: list[str]) -> None:
    """Slope comparison b - a on the paper-versions usable in both arms."""
    both = {k: d for k, d in cell.items() if d.get(a) and d.get(b)}
    md += ['', f"## {role}: {ARM_LABEL[b]}  minus  {ARM_LABEL[a]}", '']
    balance = Counter((len(d[a]), len(d[b])) for d in both.values())
    n_planned = sum(len(d) for d in planned_pos.values())
    md.append(f"- paper-versions usable in both arms: {len(both)} of {n_planned} planned | "
              f"replicates ({a}, {b}) per cell: "
              + ', '.join(f"{k}: {v}" for k, v in sorted(balance.items(), reverse=True)))
    by_paper: dict = defaultdict(dict)
    for (aid, v), d in both.items():
        by_paper[aid][v] = d
    slopes: dict = {a: [], b: []}
    levels: dict = {a: [], b: []}
    sxx, nver = [], []
    for aid, vd in sorted(by_paper.items()):
        vs = sorted(vd)
        if len(vs) < 2:
            continue
        pos = [planned_pos[aid][v] for v in vs]
        nver.append(len(vs))
        for arm in (a, b):
            means = [st.mean(vd[v][arm]) for v in vs]
            s, w = slope_and_sxx(pos, means)
            slopes[arm].append(s)
            levels[arm].append(st.mean(means))
        sxx.append(w)
    n = len(sxx)
    md.append(f"- papers with >= 2 such versions: {n}")
    if n < 3:
        md.append('- not enough papers for inference yet')
        return

    # rounding removes floating-point residue so exact zeros stay exact
    diffs = [round(y - x, 9) for x, y in zip(slopes[a], slopes[b])]
    ones = [1.0] * n
    md.append('- pooled slope (rating points per revision step):')
    for arm in (a, b):
        md.append(f"    - {arm}: {fmt(pooled(slopes[arm], sxx), boot_ci(slopes[arm], sxx))}")
    md.append(f"    - **difference: {fmt(pooled(diffs, sxx), boot_ci(diffs, sxx))}, "
              f"sign-flip {fmt_p(flip_p(diffs, sxx))}**")
    w_arr = np.asarray(sxx)
    share = sorted(w_arr / w_arr.sum(), reverse=True)
    md.append(f"    - weight concentration: the 3 heaviest papers carry {sum(share[:3]):.0%}; "
              f"two-version papers carry "
              f"{sum(w for w, k in zip(sxx, nver) if k == 2) / w_arr.sum():.0%}")
    md.append('- unweighted mean of per-paper slopes:')
    for arm in (a, b):
        md.append(f"    - {arm}: {fmt(st.mean(slopes[arm]), boot_ci(slopes[arm], ones))}")
    sd = st.stdev(diffs)
    if sd == 0:
        md.append(f"    - difference: {st.mean(diffs):+.3f} — every per-paper difference is "
                  f"identical, no test possible")
    else:
        t = stats.ttest_1samp(diffs, 0.0)
        md.append(f"    - difference: {fmt(st.mean(diffs), boot_ci(diffs, ones))}, paired "
                  f"t({n - 1}) = {t.statistic:.2f}, p = {t.pvalue:.4f} "
                  f"({sum(d == 0 for d in diffs)} of {n} differences are exactly zero)")
        md.append(f"    - SD of the per-paper differences {sd:.3f}: with {n} papers this test "
                  f"detects a difference of about {2.8 * sd / math.sqrt(n):.2f} points per "
                  f"step at 80% power; anything smaller would likely look like no effect")
    d_arr = np.asarray(diffs)
    for label, keep in (('two-version papers', lambda k: k == 2),
                        ('papers with 3+ versions', lambda k: k >= 3)):
        sel = [i for i, k in enumerate(nver) if keep(k)]
        if len(sel) >= 2:
            md.append(f"- {label} (n = {len(sel)}): pooled difference "
                      f"{pooled(d_arr[sel], w_arr[sel]):+.3f}, unweighted {d_arr[sel].mean():+.3f}")
    lvl = [y - x for x, y in zip(levels[a], levels[b])]
    md.append(f"- mean rating: {a} {st.mean(levels[a]):.3f}, {b} {st.mean(levels[b]):.3f}; "
              f"level difference {fmt(st.mean(lvl), boot_ci(lvl, ones))}, "
              f"sign-flip {fmt_p(flip_p(lvl, ones))}")


def main() -> None:
    ap = argparse.ArgumentParser(description='E1-H: version slopes by arm.')
    ap.add_argument('--logs', nargs='+', default=list(DEFAULT_LOGS))
    ap.add_argument('--manifest', default='data/processed/pilot_manifest.jsonl')
    ap.add_argument('--texts', default='data/processed/texts/')
    ap.add_argument('--reps', type=int, default=3)
    ap.add_argument('--out', default='results/e1h_summary.md')
    ap.add_argument('--b1-all', action='store_true',
                    help='sensitivity: use every process-verified B1 run, including those '
                         'that needed a reminder or wrote a review before the report')
    args = ap.parse_args()
    global B1_ALL
    B1_ALL = args.b1_all
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    raw = []
    for log in args.logs:
        if Path(log).exists():
            raw += [r for r in read_log(log) if r.get('arm') in ARM_LABEL]
    latest: dict[str, dict] = {}
    for r in raw:
        # a usable row is never replaced by a later failure of the same key
        if usable(r) or not usable(latest.get(r['key'], {})):
            latest[r['key']] = r
    rows = list(latest.values())
    ok = [r for r in rows if usable(r)]
    arms = [a for a in ARM_LABEL if any(r['arm'] == a for r in rows)]

    # planned design: positions come from the manifest + texts, never from what survived
    planned_pos: dict[str, dict] = {}
    for line in Path(args.manifest).read_text(encoding='utf-8').split('\n'):
        if not line.strip():
            continue
        p = json.loads(line)
        safe_id = p['arxiv_id'].replace('/', '_')
        vs = sorted(v['v'] for v in p['versions']
                    if (Path(args.texts) / safe_id / f"v{v['v']}.md").exists())
        planned_pos[p['arxiv_id']] = {v: i for i, v in enumerate(vs)}
    n_planned = sum(len(d) for d in planned_pos.values()) * args.reps

    md = ['# E1-H: version-score slope with and without DeepSeek Harness', '']
    md.append(f"- logs: {', '.join(f'`{x}`' for x in args.logs if Path(x).exists())} — "
              f"{len(raw)} attempts logged, {len(rows)} distinct runs")
    for a in arms:
        att = [r for r in raw if r['arm'] == a]
        ra = [r for r in rows if r['arm'] == a]
        line = f"- {ARM_LABEL[a]}: {sum(usable(r) for r in ra)} of {n_planned} planned runs usable"
        if a == 'harness_sub':
            pf = sum(r.get('process_ok') is False for r in ra)
            tf = sum(r.get('process_ok') is None for r in ra)
            verified = [r for r in ra if r.get('process_ok') is True]
            line += (f" ({'all verified runs' if B1_ALL else 'clean runs only'}) | attempts "
                     f"{len(att)} | process "
                     f"verified {len(verified)} | of those needed a reminder "
                     f"{sum(bool(r.get('reminded')) for r in verified)}, wrote a review "
                     f"before the report "
                     f"{sum(bool(r.get('review_before_report')) for r in verified)} | "
                     f"process failures {pf} ({pf / max(len(ra), 1):.1%} of runs) | "
                     f"failed attempts (technical) "
                     f"{sum(r.get('process_ok') is None for r in att)}, runs still without "
                     f"a result {tf}")
        else:
            line += (f" | attempts {len(att)} | failed attempts "
                     f"{sum(not r.get('parsed_ok') for r in att)}")
        md.append(line)
    warn = []
    for a in arms:
        sigs = {r.get('cfg_sig') for r in ok if r['arm'] == a}
        if len(sigs) > 1:
            warn.append(f"{a}: rows come from {len(sigs)} different configurations")
    sha: dict = defaultdict(set)
    for r in ok:
        sha[(r['arxiv_id'], r['v'])].add(r.get('text_sha'))
    if any(len(s) > 1 for s in sha.values()):
        warn.append('some paper-versions have different input text hashes across rows')
    for w in warn:
        md.append(f"- **WARNING: {w}**")

    # cell[(paper, v)][arm] -> replicate ratings
    cell: dict = defaultdict(lambda: defaultdict(list))
    for r in ok:
        if r['v'] in planned_pos.get(r['arxiv_id'], {}):
            cell[(r['arxiv_id'], r['v'])][r['arm']].append(r['rating'])

    for a, b, role in COMPARISONS:
        if a in arms and b in arms:
            compare(a, b, role, cell, planned_pos, md)

    md += ['', '## Replicate noise', '']
    for a in arms:
        sds = [st.stdev(d[a]) for d in cell.values() if len(d.get(a, [])) >= 2]
        if sds:
            md.append(f"- {a}: mean SD across replicates of the same paper-version "
                      f"{st.mean(sds):.3f} (n = {len(sds)} cells)")

    md += ['', '## What each arm did', '']
    for a in arms:
        ra = [r for r in ok if r['arm'] == a]
        if not ra:
            continue
        known = [r for r in ra if not r.get('usage_missing')
                 and sum((r.get('usage') or {}).values())]
        line = f"- {ARM_LABEL[a]}: {len(ra)} usable runs"
        if known:
            costs = [r['cost_cny_est'] for r in known if r.get('cost_cny_est') is not None]
            line += (f" | mean tokens in "
                     f"{st.mean(r['usage'].get('input', 0) + r['usage'].get('cache_read', 0) for r in known):,.0f}"
                     f" out {st.mean(r['usage'].get('output', 0) for r in known):,.0f}")
            if costs:
                line += f" | mean cost ¥{st.mean(costs):.3f} per run (list price)"
        line += (f" | usage unknown in {len(ra) - len(known)} runs"
                 f" | median latency {st.median(r['latency_s'] for r in ra):.0f}s"
                 f" | runs with a retry inside the run "
                 f"{sum(bool(r.get('llm_retries') or r.get('failed_attempts')) for r in ra)}")
        if a == 'harness':
            line += (f" | multi-step runs {sum((r.get('n_steps') or 0) > 1 for r in ra)}"
                     f" | runs with tool/agent events {sum(bool(r.get('other_events')) for r in ra)}")
        if a == 'harness_sub':
            line += (f" | mean share of the paper passed to the subagent "
                     f"{st.mean(r.get('coverage') or 0 for r in ra):.1%}"
                     f" | runs that started extra subagents "
                     f"{sum((r.get('n_children') or 0) > 1 for r in ra)}"
                     f" | runs that wrote a review before the report "
                     f"{sum(bool(r.get('review_before_report')) for r in ra)}")
        md.append(line)
    md.append('')
    md.append('A larger slope means later versions are scored higher; on its own it does '
              'not show that the scores are more accurate.')

    Path(args.out).write_text('\n'.join(md) + '\n', encoding='utf-8')
    print('\n'.join(md))
    print(f'\nwritten: {args.out}')


if __name__ == '__main__':
    main()
