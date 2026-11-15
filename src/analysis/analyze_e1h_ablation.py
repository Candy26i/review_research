"""E1-H ablation analysis: which part of the harness request lowers the rating level?

Input: the log written by src/review/run_e1_harness_ablation.py. Arms: bare (plain API
call, two replicates), sys / tools / ctx (bare plus ONE part of what dsh adds), replay
(bare plus all three), dsh (the real harness); one replicate each.

Analysis plan, fixed on 2026-11-14 before any ablation data existed:

  UNIT            the paper. For a contrast X - Y, each paper-version usable in both arms
                  gives one difference of mean ratings; these are averaged within paper,
                  then across papers (the level definition of E1-H).
  USABLE RUN      a valid review was parsed and the model ended its turn normally. Runs
                  that stopped to call a tool (API arms), or in which the harness took
                  more than one step or emitted tool/agent events (dsh arm), are counted
                  per arm and left out; --keep-tool-stops includes those that still
                  contain a valid review.
  PRIMARY         sys - bare, tools - bare, ctx - bare: 95% paper-bootstrap interval and
                  sign-flip p over papers; Holm adjustment across these three. Each is
                  the effect of ADDING one part to a bare request. It is not the effect
                  of removing that part from the harness request.
  REFERENCE       replay - bare and dsh - bare: the full effect in this session, to be
                  set beside E1-H's harness - direct = -0.48 [-0.62, -0.36].
  CONTENT CHECK   replay - dsh.
  PART COMPARISON sys - tools, sys - ctx, tools - ctx, and replay - sys / tools / ctx:
                  same interval and test, unadjusted, used only by the reading rule.
  ADDITIVITY      (replay - bare) minus the sum of the three single-part effects, on
                  paper-versions usable in all five API arms. Its standard error is
                  about 0.2, so only all-or-nothing departures are detectable; an
                  interval that includes 0 is not evidence that the parts add up.
  READING RULE    A part X is said to be enough on its own only if X - bare has Holm
                  p < 0.05, X differs from both other parts (interval excludes 0) and
                  replay - X includes 0. If only the first holds: "X lowers the rating
                  when added alone; this run cannot say it matters more than the
                  others". Parts are never ranked or given shares by point estimate.
                  An interval that includes 0 (replay - dsh, replay - X) is reported as
                  "no difference detected; about 0.2 points not excluded", not as
                  "equal" or "explained". The words "the harness effect" are used only
                  if dsh - bare excludes 0 in this session; otherwise part results are
                  statements about plain API requests.
  RETRIES         the share of runs with a retry inside the run is reported per arm,
                  with the reasons; the main contrasts are repeated without those runs.
  CACHE CHECK     the runner warms the provider's prompt cache so that every arm reads
                  its input from the cache (in E1-H, cache-miss runs rated 0.14 higher
                  than cache-hit runs of the same paper-version and arm, 95% CI -0.02
                  to +0.30: not established, not excludable). The table reports the
                  share of runs that read from cache. If any arm is below 95%, the main
                  contrasts are repeated on cache-hit runs only, and the unrestricted
                  result is not interpreted where the two disagree.
  SESSIONS        if the log holds runs from more than one invocation, the main
                  contrasts are repeated on paper-versions whose runs share one.
  NOT ANALYSED    version slopes. This run cannot say anything about them.

Power, simulated from the E1-H log before the run: a pairwise contrast has a standard
error of about 0.10 rating points (about 0.09 against the two-replicate bare arm). A part
carrying three quarters of the half-point effect is detected about 85-94% of the time, a
part carrying half of it about 45-63%, a quarter about 10-20%. replay - dsh can exclude
only differences larger than about 0.2.

Run from the repo root:
    python3 src/analysis/analyze_e1h_ablation.py
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ARM_LABEL = {'bare': 'bare API call', 'sys': 'bare + system prompt',
             'tools': 'bare + tool definitions', 'ctx': 'bare + runtime-context block',
             'replay': 'bare + all three (replay)', 'dsh': 'real harness'}
SINGLE = ('sys', 'tools', 'ctx')
MAIN = ([('sys', 'bare'), ('tools', 'bare'), ('ctx', 'bare'), ('replay', 'bare'),
         ('dsh', 'bare'), ('replay', 'dsh')])
KEEP_TOOL_STOPS = False


def read_log(path: str) -> list[dict]:
    rows = []
    for line in Path(path).read_text(encoding='utf-8').split('\n'):
        if line.strip():
            try:
                rows.append(json.loads(line))
            except ValueError:
                continue
    return rows


def tool_stopped(r: dict) -> bool:
    """The model tried to use a tool: an explicit stop in an API arm, or more than one
    step / any tool or agent event inside the real harness."""
    if r.get('tool_stop'):
        return True
    return r.get('arm') == 'dsh' and ((r.get('n_steps') or 1) > 1 or bool(r.get('other_events')))


def usable(r: dict) -> bool:
    if not r.get('parsed_ok'):
        return False
    return KEEP_TOOL_STOPS or not tool_stopped(r)


def cells_of(rows: list[dict]) -> dict:
    cell: dict = defaultdict(lambda: defaultdict(list))
    for r in rows:
        cell[(r['arxiv_id'], r['v'])][r['arm']].append(r['rating'])
    return cell


def paper_values(cell: dict, combo) -> dict[str, float]:
    """Per paper: mean over its paper-versions of a linear combination of arm means.
    combo is a list of (arm, weight); a paper-version counts only if every arm has data."""
    by_paper: dict = defaultdict(list)
    for (aid, _v), d in cell.items():
        if all(d.get(a) for a, _ in combo):
            by_paper[aid].append(sum(w * st.mean(d[a]) for a, w in combo))
    return {aid: st.mean(v) for aid, v in by_paper.items()}


def boot_ci(values, n_boot: int = 10000, seed: int = 42) -> tuple[float, float]:
    v = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    est = v[rng.integers(0, len(v), size=(n_boot, len(v)))].mean(axis=1)
    return float(np.percentile(est, 2.5)), float(np.percentile(est, 97.5))


def flip_p(values, n_flip: int = 20000, seed: int = 42) -> float:
    v = np.asarray(values, dtype=float)
    obs = abs(v.sum())
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(n_flip, len(v)))
    return float(((np.abs((signs * v).sum(axis=1)) >= obs - 1e-6).sum() + 1) / (n_flip + 1))


def fmt_p(p: float) -> str:
    return 'p < 0.0001' if p < 0.0001 else f'p = {p:.4f}'


def contrast(cell: dict, combo, label: str) -> tuple[str, float | None]:
    vals = paper_values(cell, combo)
    n_pv = sum(all(d.get(a) for a, _ in combo) for d in cell.values())
    if len(vals) < 3:
        return f"- {label}: not enough papers ({len(vals)})", None
    v = list(vals.values())
    p = flip_p(v)
    lo, hi = boot_ci(v)
    se = st.stdev(v) / len(v) ** 0.5
    signs = Counter('below' if x < -1e-9 else 'above' if x > 1e-9 else 'equal' for x in v)
    return (f"- {label}: **{st.mean(v):+.3f}** (95% CI [{lo:+.3f}, {hi:+.3f}], SE {se:.3f}), "
            f"sign-flip {fmt_p(p)} | papers {len(v)} (below {signs['below']}, above "
            f"{signs['above']}, equal {signs['equal']}) | paper-versions {n_pv}"), p


def compact(cell: dict, arms: list[str]) -> list[str]:
    """The main contrasts in one line each, for the sensitivity checks."""
    out = []
    for a, b in MAIN:
        if a in arms and b in arms:
            v = list(paper_values(cell, [(a, 1.0), (b, -1.0)]).values())
            if len(v) >= 3:
                lo, hi = boot_ci(v)
                out.append(f"    - {a} - {b}: {st.mean(v):+.3f} [{lo:+.3f}, {hi:+.3f}] "
                           f"({len(v)} papers)")
            else:
                out.append(f"    - {a} - {b}: not enough papers ({len(v)})")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description='E1-H ablation: level effect of each part.')
    ap.add_argument('--log', default='data/processed/e1h_ablation_log.jsonl')
    ap.add_argument('--out', default='results/e1h_ablation_summary.md')
    ap.add_argument('--keep-tool-stops', action='store_true',
                    help='sensitivity: keep runs that stopped for a tool call but still '
                         'contain a valid review')
    args = ap.parse_args()
    global KEEP_TOOL_STOPS
    KEEP_TOOL_STOPS = args.keep_tool_stops

    raw = [r for r in read_log(args.log) if r.get('arm') in ARM_LABEL]
    latest: dict[str, dict] = {}
    n_multi = 0
    for r in raw:
        prev = latest.get(r['key'])
        n_multi += bool(prev is not None and usable(prev) and usable(r))
        if usable(r) or prev is None or not usable(prev):
            latest[r['key']] = r
    rows = list(latest.values())
    ok = [r for r in rows if usable(r)]
    arms = [a for a in ARM_LABEL if any(r['arm'] == a for r in rows)]

    md = ['# E1-H ablation: which part of the harness request lowers the scores?', '']
    md.append(f"- log: `{args.log}` — {len(raw)} attempts logged, {len(rows)} distinct runs"
              + (' | tool-stop runs with a valid review are INCLUDED' if KEEP_TOOL_STOPS
                 else ''))
    if n_multi:
        md.append(f"- **WARNING: {n_multi} run(s) have more than one usable row; the latest "
                  f"is used**")
    for field, what in (('cfg_sig', 'configurations'), ('served_model', 'served models')):
        vals = {r.get(field) for r in ok}
        if len(vals) > 1:
            md.append(f"- **WARNING: usable rows come from {len(vals)} {what}**")
    sha: dict = defaultdict(set)
    for r in ok:
        sha[(r['arxiv_id'], r['v'])].add(r.get('text_sha'))
    if any(len(s) > 1 for s in sha.values()):
        md.append('- **WARNING: some paper-versions have different input text hashes**')
    n_flag = sum(bool(r.get('coerced') or r.get('truncated')) for r in ok)
    if n_flag:
        md.append(f"- {n_flag} usable row(s) are coerced or truncated")
    invocations = sorted({r.get('invocation') for r in ok if r.get('invocation')})
    if len(invocations) > 1:
        md.append(f"- **the log holds runs from {len(invocations)} invocations** (usable "
                  "runs per arm: " + '; '.join(
                      f"{i}: " + ', '.join(
                          f"{a} {sum(r['arm'] == a and r.get('invocation') == i for r in ok)}"
                          for a in arms) for i in invocations) + ')')

    cell = cells_of(ok)

    md += ['', '## What each arm did', '',
           '| arm | usable runs | tool use | failed runs | runs with a retry | mean rating | '
           'rating 3 | accept | mean output tokens | median reasoning chars | reasoning '
           'opens with "We need" | input read from cache | mean cost |',
           '|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    reasons: dict = defaultdict(Counter)
    for a in arms:
        att = [r for r in raw if r['arm'] == a]
        ra = [r for r in ok if r['arm'] == a]
        for r in att:
            for e in r.get('attempt_errors') or []:
                reasons[a][e.split(' (')[0].split(' after ')[0]] += 1
        if not ra:
            md.append(f"| {ARM_LABEL[a]} | 0 | | | | | | | | | | | |")
            continue
        think = [r['thinking_chars'] for r in ra if r.get('thinking_chars')]
        opens = [r for r in ra if r.get('thinking_open')]
        costs = [r['cost_cny_est'] for r in ra if r.get('cost_cny_est') is not None]
        cached = [r for r in ra if r.get('cache_share') is not None]
        think_s = f'{st.median(think):,.0f}' if think else 'n/a'
        open_s = 'n/a'
        if opens:
            n_we = sum(r['thinking_open'].startswith('We need') for r in opens)
            open_s = f"{n_we / len(opens):.1%} (n={len(opens)})"
        cache_s = (f"{sum(r['cache_share'] > 0.5 for r in cached) / len(cached):.1%} of runs"
                   if cached else 'n/a')
        cost_s = f"¥{st.mean(costs):.3f}" if costs else 'n/a'
        md.append(
            f"| {ARM_LABEL[a]} | {len(ra)} "
            f"| {sum(tool_stopped(r) for r in rows if r['arm'] == a)} "
            f"| {sum(not r.get('parsed_ok') and not tool_stopped(r) for r in rows if r['arm'] == a)} "
            f"| {sum(bool(r.get('failed_attempts') or r.get('llm_retries')) for r in ra)} "
            f"| {st.mean(r['rating'] for r in ra):.3f} "
            f"| {sum(r['rating'] == 3 for r in ra) / len(ra):.1%} "
            f"| {sum(bool(r['accept']) for r in ra) / len(ra):.1%} "
            f"| {st.mean(r['usage'].get('output', 0) for r in ra):,.0f} "
            f"| {think_s} | {open_s} | {cache_s} | {cost_s} |")
    md.append('')
    md.append('Reasoning length is not available for the real harness (its event stream '
              'caps reasoning text); the opening is. "Failed runs" are runs still without '
              'a result.')
    if any(reasons.values()):
        md.append('')
        md.append('Reasons of failed attempts inside API-arm runs (the real harness logs its '
                  'own): ' + '; '.join(
                      f"{a}: " + ', '.join(f"{k} x{n}" for k, n in c.most_common())
                      for a, c in reasons.items() if c))

    if 'bare' in arms:
        md += ['', '## Level difference against the concurrent bare call', '',
               'Single parts (primary):', '']
        pvals = {}
        for a in SINGLE:
            if a in arms:
                line, p = contrast(cell, [(a, 1.0), ('bare', -1.0)],
                                   f"{ARM_LABEL[a]} minus bare")
                md.append(line)
                if p is not None:
                    pvals[a] = p
        if len(pvals) > 1:
            order = sorted(pvals, key=pvals.get)
            adj, running = {}, 0.0
            for i, a in enumerate(order):
                running = max(running, min(1.0, pvals[a] * (len(order) - i)))
                adj[a] = running
            md.append('- Holm-adjusted p across the single parts: '
                      + ', '.join(f"{a} " + ('< 0.001' if adj[a] < 0.001 else f"{adj[a]:.4f}")
                                  for a in SINGLE if a in adj))
        md += ['', 'Full effect in this session (reference):', '']
        for a in ('replay', 'dsh'):
            if a in arms:
                md.append(contrast(cell, [(a, 1.0), ('bare', -1.0)],
                                   f"{ARM_LABEL[a]} minus bare")[0])
        md.append('- E1-H, for comparison: harness minus direct -0.483 '
                  '(95% CI [-0.615, -0.359]); -0.57 and -0.33 in its two run sessions')
        md += ['', 'Parts against each other and against the replay (for the reading rule):',
               '']
        for a, b in (('sys', 'tools'), ('sys', 'ctx'), ('tools', 'ctx'),
                     ('replay', 'sys'), ('replay', 'tools'), ('replay', 'ctx')):
            if a in arms and b in arms:
                md.append(contrast(cell, [(a, 1.0), (b, -1.0)], f"{a} minus {b}")[0])

    if 'replay' in arms and 'dsh' in arms:
        md += ['', '## Content check: replay against the real harness', '']
        md.append(contrast(cell, [('replay', 1.0), ('dsh', -1.0)],
                           'replay minus real harness')[0])
    if all(a in arms for a in ('bare', 'replay') + SINGLE):
        md += ['', '## Additivity', '']
        md.append(contrast(cell, [('replay', 1.0), ('sys', -1.0), ('tools', -1.0),
                                  ('ctx', -1.0), ('bare', 2.0)],
                           'replay effect minus the sum of the three single-part effects')[0])
        md.append('  (negative: the parts together lower the rating more than their sum; '
                  'only all-or-nothing departures are detectable at this sample size)')

    md += ['', '## Sensitivity checks', '']
    no_retry = [r for r in ok if not (r.get('failed_attempts') or r.get('llm_retries'))]
    md.append(f"- without runs that had a retry inside the run ({len(ok) - len(no_retry)} "
              f"dropped):")
    md += compact(cells_of(no_retry), arms)
    low_cache = [a for a in arms
                 if (c := [r for r in ok if r['arm'] == a and r.get('cache_share') is not None])
                 and sum(r['cache_share'] > 0.5 for r in c) / len(c) < 0.95]
    if low_cache:
        hit = [r for r in ok if (r.get('cache_share') or 0) > 0.5]
        md.append(f"- **cache-hit runs only** (arms below 95% cache hits: "
                  f"{', '.join(low_cache)}; {len(ok) - len(hit)} dropped):")
        md += compact(cells_of(hit), arms)
    else:
        md.append('- cache state: every arm read its input from the cache in at least 95% of '
                  'its runs; no restriction needed')
    if len(invocations) > 1:
        inv_of: dict = defaultdict(set)
        for r in ok:
            inv_of[(r['arxiv_id'], r['v'])].add(r.get('invocation'))
        same = [r for r in ok if len(inv_of[(r['arxiv_id'], r['v'])]) == 1]
        md.append(f"- paper-versions whose runs all come from one invocation "
                  f"({len(ok) - len(same)} runs dropped):")
        md += compact(cells_of(same), arms)

    md += ['', 'Each line gives its own standard error. Version slopes are not analysed.']
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text('\n'.join(md) + '\n', encoding='utf-8')
    print('\n'.join(md))
    print(f'\nwritten: {args.out}')


if __name__ == '__main__':
    main()
