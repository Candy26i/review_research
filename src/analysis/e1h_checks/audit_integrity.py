#!/usr/bin/env python3
"""E1-H data-integrity and confound audit.  READ-ONLY: never writes inside the repo.

    python3 audit_integrity.py --logs <main_log> <sub_log> [--repo PATH] [--out DIR]
                               [--skip-sessions] [--skip-raw] [--gap-min 5]

<main_log> holds arms direct + harness, <sub_log> holds arm harness_sub (order does not
matter: rows are routed by their `arm` field).  The script re-implements all parsing
itself (no runner import), makes no network request and starts no subprocess.

Sections (numbering follows the audit brief):
  1 keys            5 sleep contamination + sensitivity of headline numbers
  2 configuration   6 time confounds
  3 inputs          7 missingness
  4 raw re-derive   8 harness behaviour
Writes <out>/audit_report.txt (same text as stdout), <out>/row_flags.csv and
<out>/key_numbers.json.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import random
import re
import statistics as st
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy import stats

ARMS = ('direct', 'harness', 'harness_sub')
FIELDS = ('rating', 'soundness', 'presentation', 'contribution', 'confidence', 'accept')
ALLOWED_RATINGS = (1, 3, 5, 6, 8, 10)
PAIRS = (('direct', 'harness'), ('direct', 'harness_sub'), ('harness', 'harness_sub'))
PRIMER = ("\n\nDo not review yet. This message only delivers the submission and the review "
          "criteria. Reply with the single word READY and nothing else; how to proceed is "
          "explained in the next message.")
PROCESS_LEN = 846  # len(PROCESS) in run_e1_harness_sub.py; only used for task_chars arithmetic

OUT_LINES: list[str] = []


def P(*a) -> None:
    s = ' '.join(str(x) for x in a)
    OUT_LINES.append(s)
    print(s)


def H(title: str) -> None:
    P('')
    P('=' * 100)
    P(title)
    P('=' * 100)


# ------------------------------------------------------------------ loading

def read_log(path: str) -> list[dict]:
    rows = []
    with open(path, encoding='utf-8') as fh:
        for i, line in enumerate(fh.read().split('\n')):
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except ValueError:
                P(f'  [torn/unparseable line skipped: {path} line {i}]')
                continue
            r['_file'] = os.path.basename(path)
            r['_line'] = i
            rows.append(r)
    return rows


def usable(r: dict | None, mode: str = 'clean') -> bool:
    if not r or not r.get('parsed_ok'):
        return False
    if r.get('arm') != 'harness_sub':
        return True
    if r.get('process_ok') is not True:
        return False
    return mode == 'all' or not (r.get('reminded') or r.get('review_before_report'))


def dedup(rows: list[dict], mode: str = 'clean') -> dict[str, dict]:
    latest: dict[str, dict] = {}
    for r in rows:
        if usable(r, mode) or not usable(latest.get(r['key']), mode):
            latest[r['key']] = r
    return latest


def tparse(ts: str) -> datetime:
    return datetime.fromisoformat(ts)


def is_peak(t: datetime) -> bool:
    t = t.astimezone(timezone.utc)
    return t.weekday() < 5 and (1 <= t.hour < 4 or 6 <= t.hour < 10)


def find_reviews(text: str) -> list[dict]:
    """Independent extractor: every top-level JSON object in `text` that has all six
    score fields (scans with raw_decode; no fence stripping needed)."""
    dec = json.JSONDecoder()
    found, i = [], 0
    while True:
        i = text.find('{', i)
        if i < 0:
            break
        try:
            obj, end = dec.raw_decode(text, i)
        except ValueError:
            i += 1
            continue
        if isinstance(obj, dict) and all(f in obj for f in FIELDS):
            found.append(obj)
        i = end
    return found


def on_scale(rv: dict) -> bool:
    try:
        return (type(rv['rating']) is int and rv['rating'] in ALLOWED_RATINGS
                and all(type(rv[f]) is int and 1 <= rv[f] <= 4
                        for f in ('soundness', 'presentation', 'contribution'))
                and type(rv['confidence']) is int and 1 <= rv['confidence'] <= 5
                and type(rv['accept']) is bool)
    except Exception:
        return False


def diff_fields(rv: dict, row: dict) -> dict:
    return {f: (rv.get(f), row.get(f)) for f in FIELDS
            if rv.get(f) != row.get(f) or type(rv.get(f)) is not type(row.get(f))}


def read_events(path: Path) -> list[dict]:
    ev = []
    for line in path.read_text(encoding='utf-8', errors='replace').split('\n'):
        if line.strip():
            try:
                ev.append(json.loads(line))
            except ValueError:
                ev.append({'type': '_unparseable'})
    return ev


# ------------------------------------------------------------------ dsh session logs

class Sessions:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.index: dict[str, list[str]] = {}
        if root.is_dir():
            for b in os.listdir(root):
                d = root / b
                if d.is_dir():
                    for s in os.listdir(d):
                        self.index.setdefault(s, []).append(b)
        self.cache: dict[str, list[dict] | None] = {}

    def bucket(self, sid: str) -> str | None:
        return (self.index.get(sid) or [None])[0]

    def load(self, sid: str | None) -> list[dict] | None:
        if not sid or sid not in self.index:
            return None
        if sid in self.cache:
            return self.cache[sid]
        p = self.root / self.index[sid][0] / sid / 'session.v4.jsonl.zstd'
        rows: list[dict] | None
        try:
            import zstandard
            data = zstandard.ZstdDecompressor().stream_reader(
                io.BytesIO(p.read_bytes())).read().decode('utf-8', 'replace')
            rows = []
            for line in data.split('\n'):
                if line.strip():
                    try:
                        rows.append(json.loads(line))
                    except ValueError:
                        continue
        except Exception:
            rows = None
        self.cache[sid] = rows
        return rows


def mtext(d: dict) -> str:
    c = d.get('content')
    if c is None:
        c = (d.get('message') or {}).get('content')
    if isinstance(c, str):
        return c
    return ''.join(b.get('text', '') for b in (c or [])
                   if isinstance(b, dict) and b.get('type') == 'text')


def sha12(s: str) -> str:
    return hashlib.sha256(s.encode('utf-8', 'replace')).hexdigest()[:12]


def split_child(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    cut = None
    for i, r in enumerate(rows):
        if r.get('type') == 'session/end-seed' and (r.get('data') or {}).get('inherited'):
            cut = i
    if cut is None:
        return [], rows
    return rows[:cut], rows[cut + 1:]


def session_summary(rows: list[dict]) -> dict:
    times = [r['time'] for r in rows if isinstance(r.get('time'), (int, float))]
    s: dict = {'t0': min(times) / 1000 if times else None,
               't1': max(times) / 1000 if times else None}
    gaps = [(times[i] - times[i - 1]) / 1000 for i in range(1, len(times))]
    s['maxgap'] = max(gaps) if gaps else 0.0
    s['retries'] = sum(r.get('type') == 'llm/retry' for r in rows)
    codes: Counter = Counter()
    for r in rows:
        if r.get('type') == 'llm/retry':
            codes[((r.get('data') or {}).get('failure') or {}).get('code')] += 1
    s['retry_codes'] = dict(codes)
    s['sys_sha'] = sorted({sha12(mtext(r.get('data') or {})) for r in rows
                           if r.get('type') == 'system/message'})
    hdrs = [(r.get('data') or {}).get('header') or {} for r in rows
            if r.get('type') == 'request/header']
    s['hdr_sha'] = sorted({sha12(json.dumps(h, sort_keys=True)) for h in hdrs})
    s['hdr_cfg'] = sorted({json.dumps(h.get('config'), sort_keys=True) for h in hdrs})
    s['tool_names'] = sorted({','.join(t.get('name', '?') for t in (h.get('tools') or []))
                              for h in hdrs})
    models: Counter = Counter()
    for r in rows:
        if r.get('type') == 'assistant/message':
            src = (((r.get('data') or {}).get('message') or {}).get('source') or {})
            models[str(((src.get('replayState') or {}).get('response') or {}).get('model'))] += 1
    s['models'] = dict(models)
    um = [mtext(r.get('data') or {}) for r in rows if r.get('type') == 'user/message']
    s['user_msgs'] = um
    rc = [x for x in um if x.startswith('Current runtime context')]
    s['rc_sha'] = sorted({sha12(x) for x in rc})
    cw = [re.search(r'Current working directory: "([^"]*)"', x) for x in rc]
    s['cwd'] = sorted({m.group(1) for m in cw if m})
    s['children'] = [(r.get('data') or {}).get('childId') for r in rows
                     if r.get('type') == 'subagent/catalog']
    s['tools'] = [(r.get('data') or {}).get('name') for r in rows if r.get('type') == 'tool/call']
    s['n_turns'] = sum(r.get('type') == 'turn/start' for r in rows)
    s['turn_ends'] = [((r.get('data') or {}).get('reason') or {}).get('kind')
                      for r in rows if r.get('type') == 'turn/end']
    return s


def norm_ws(s: str) -> str:
    return re.sub(r'\s+', ' ', s)


def coverage(paper: str, text: str, size: int = 200) -> float:
    p, q = norm_ws(paper), norm_ws(text)
    chunks = [p[i:i + size] for i in range(0, max(len(p) - size + 1, 1), size)]
    return sum(c in q for c in chunks) / len(chunks)


def sub_process_facts(log: list[dict], sess: Sessions, paper: str) -> dict:
    """Independent re-derivation of the B1 process facts from the session logs."""
    f: dict = {'n_children': 0, 'tool_calls': {}, 'n_turns': 0, 'cov': 0.0, 'child_id': None,
               'child_tool': None, 'report_via': None, 'reviews': [], 'i_report': None,
               'result': None, 'child_done': False, 'child_review_in_report': None,
               'child_models': {}, 'child_desc': {}, 'child_retries': 0, 'child_retry_codes': {},
               'child_tools': [], 'child_t0': None, 'child_t1': None, 'child_maxgap': 0.0,
               'child_cwd': [], 'child_hdr_cfg': [], 'missing_child': 0}
    f['n_turns'] = sum(r.get('type') == 'turn/start' for r in log)
    f['tool_calls'] = dict(Counter((r.get('data') or {}).get('name') for r in log
                                   if r.get('type') == 'tool/call'))
    children = [(r.get('data') or {}).get('childId') for r in log
                if r.get('type') == 'subagent/catalog']
    f['n_children'] = len(children)
    i_call = None
    for i, r in enumerate(log):
        d = r.get('data') or {}
        if r.get('type') != 'tool/call' or d.get('name') not in ('subagent_fork', 'subagent'):
            continue
        cid = next(((x.get('data') or {}).get('childId') for x in log[i + 1:i + 5]
                    if x.get('type') == 'subagent/catalog'), None)
        child = sess.load(cid) if cid else None
        if child is None:
            if cid:
                f['missing_child'] += 1
            continue
        if d.get('name') == 'subagent':
            try:
                seen = json.loads(d.get('arguments') or '{}').get('prompt') or ''
            except ValueError:
                seen = ''
        else:
            seeded, _ = split_child(child)
            seen = ' '.join(mtext(x.get('data') or {}) for x in seeded
                            if x.get('type') == 'user/message')
        cov = coverage(paper, seen)
        f['cov'] = max(f['cov'], cov)
        if cov >= 0.9 and i_call is None:
            i_call, f['child_id'], f['child_tool'] = i, cid, d.get('name')
            _, own = split_child(child)
            cs = session_summary(own)
            f['child_done'] = 'completed' in cs['turn_ends']
            f['child_models'] = cs['models']
            f['child_retries'] = cs['retries']
            f['child_retry_codes'] = cs['retry_codes']
            f['child_tools'] = cs['tools']
            f['child_t0'], f['child_t1'], f['child_maxgap'] = cs['t0'], cs['t1'], cs['maxgap']
            f['child_cwd'] = cs['cwd']
            f['child_hdr_cfg'] = cs['hdr_cfg']
            desc = next(((c.get('data') or {}) for c in child
                         if c.get('type') == 'subagent/descriptor'), {})
            f['child_desc'] = {'agentModel': desc.get('agentModel'),
                               'agentReasoningEffort': desc.get('agentReasoningEffort')}
    if f['child_id'] and i_call is not None:
        sent = f"Agent {f['child_id']} sent a message"
        fin = f"Background subagent {f['child_id']} finished"
        for i in range(i_call + 1, len(log)):
            r = log[i]
            if r.get('type') != 'user/message':
                continue
            t = mtext(r.get('data') or {})
            if sent in t or (fin in t and 'Its closing message:' in t):
                f['i_report'] = i
                f['report_via'] = 'send_message' if sent in t else 'closing_message'
                cr = find_reviews(t)
                f['child_review_in_report'] = cr[-1] if cr else None
                break
    for i, r in enumerate(log):
        if r.get('type') != 'assistant/message':
            continue
        t = mtext(r.get('data') or {})
        rv = find_reviews(t)
        if rv:
            f['reviews'].append((i, rv[-1], len(rv)))
    if f['i_report'] is not None:
        after = [x for x in f['reviews'] if x[0] > f['i_report']]
        f['result'] = after[0][1] if after else None
    f['review_before_report'] = any(f['i_report'] is None or i < f['i_report']
                                    for i, _, _ in f['reviews'])
    return f


# ------------------------------------------------------------------ slope / level machinery

def cells_from(rows: list[dict], planned: dict) -> dict:
    cell: dict = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if (r['arxiv_id'], r['v']) in planned:
            cell[(r['arxiv_id'], r['v'])][r['arm']].append(r['rating'])
    return cell


def fmean(x) -> float:
    x = list(x)
    return sum(x) / len(x)


def compare(cell: dict, planned: dict, a: str, b: str) -> dict:
    both = {k: d for k, d in cell.items() if d.get(a) and d.get(b)}
    out: dict = {'n_pv': len(both)}
    if not both:
        return out
    out['pv_mean_a'] = fmean([fmean(d[a]) for d in both.values()])
    out['pv_mean_b'] = fmean([fmean(d[b]) for d in both.values()])
    out['pv_diff'] = out['pv_mean_b'] - out['pv_mean_a']
    by_paper: dict = defaultdict(dict)
    for (aid, v), d in both.items():
        by_paper[aid][v] = d
    sa, sb, sxx, la, lb = [], [], [], [], []
    for aid, vd in sorted(by_paper.items()):
        vs = sorted(vd)
        if len(vs) < 2:
            continue
        x = np.array([planned[(aid, v)] for v in vs], dtype=float)
        xc = x - x.mean()
        w = float((xc ** 2).sum())
        ya = np.array([fmean(vd[v][a]) for v in vs])
        yb = np.array([fmean(vd[v][b]) for v in vs])
        sa.append(float((xc * (ya - ya.mean())).sum() / w))
        sb.append(float((xc * (yb - yb.mean())).sum() / w))
        sxx.append(w)
        la.append(float(ya.mean()))
        lb.append(float(yb.mean()))
    out['n_papers'] = len(sxx)
    if sxx:
        w = np.array(sxx)
        out['slope_a'] = float((w * np.array(sa)).sum() / w.sum())
        out['slope_b'] = float((w * np.array(sb)).sum() / w.sum())
        out['slope_diff'] = out['slope_b'] - out['slope_a']
        out['unw_slope_diff'] = float(np.mean(np.array(sb) - np.array(sa)))
        out['level_a'] = float(np.mean(la))
        out['level_b'] = float(np.mean(lb))
        out['level_diff'] = out['level_b'] - out['level_a']
    return out


def fmt_cmp(c: dict) -> str:
    if c.get('n_pv', 0) == 0:
        return 'no matched paper-versions'
    s = (f"matched pv {c['n_pv']:3d} | pv-mean {c['pv_mean_a']:.3f} vs {c['pv_mean_b']:.3f} "
         f"(diff {c['pv_diff']:+.3f})")
    if 'slope_diff' in c:
        s += (f" | papers {c['n_papers']:2d} | paper-level {c['level_a']:.3f} vs {c['level_b']:.3f} "
              f"(diff {c['level_diff']:+.3f}) | pooled slope {c['slope_a']:+.3f} vs "
              f"{c['slope_b']:+.3f} (diff {c['slope_diff']:+.3f}; unweighted {c['unw_slope_diff']:+.3f})")
    return s


def paired_test(d: list[float]) -> str:
    if len(d) < 3:
        return f'n={len(d)} (too few)'
    a = np.array(d, dtype=float)
    if a.std(ddof=1) == 0:
        return f'n={len(a)} mean {a.mean():+.3f} (all differences identical)'
    t = stats.ttest_1samp(a, 0.0)
    try:
        wp = stats.wilcoxon(a).pvalue if np.any(a != 0) else float('nan')
    except ValueError:
        wp = float('nan')
    return (f'n={len(a)} mean {a.mean():+.3f} (SE {a.std(ddof=1) / np.sqrt(len(a)):.3f}), '
            f't={t.statistic:+.2f} p={t.pvalue:.3f}, Wilcoxon p={wp:.3f}, '
            f'share>0 {np.mean(a > 0):.2f} / =0 {np.mean(a == 0):.2f} / <0 {np.mean(a < 0):.2f}')


def chi2(table: list[list[int]]) -> str:
    t = np.array(table)
    t = t[t.sum(axis=1) > 0][:, t.sum(axis=0) > 0]
    if t.shape[0] < 2 or t.shape[1] < 2:
        return 'n/a'
    c = stats.chi2_contingency(t)
    return f'chi2={c[0]:.2f} dof={c[2]} p={c[1]:.3f}'


# ------------------------------------------------------------------ main

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--logs', nargs='+', required=True)
    ap.add_argument('--repo', default='/Users/madili/Desktop/review papaer research')
    ap.add_argument('--out', default=str(Path(__file__).resolve().parent / 'out'))
    ap.add_argument('--skip-sessions', action='store_true')
    ap.add_argument('--skip-raw', action='store_true')
    ap.add_argument('--gap-min', type=float, default=5.0,
                    help='a silence longer than this many minutes in the union completion '
                         'timeline separates two bursts')
    ap.add_argument('--wake-window', type=float, default=180.0,
                    help='seconds after the first completion of a burst that count as wake-up')
    ap.add_argument('--reps', type=int, default=3)
    args = ap.parse_args()
    repo = Path(args.repo)
    out_dir = Path(args.out)
    if repo.resolve() in out_dir.resolve().parents or out_dir.resolve() == repo.resolve():
        sys.exit('refusing to write inside the repo')
    out_dir.mkdir(parents=True, exist_ok=True)
    KN: dict = {}

    raw_rows: list[dict] = []
    for p in args.logs:
        raw_rows += read_log(p)
    raw_rows = [r for r in raw_rows if r.get('arm') in ARMS]
    for i, r in enumerate(raw_rows):
        r['_i'] = i
        r['_t'] = tparse(r['ts'])
        r['_flags'] = set()
    by_arm = {a: [r for r in raw_rows if r['arm'] == a] for a in ARMS}

    # planned design
    man = [json.loads(l) for l in (repo / 'data/processed/pilot_manifest.jsonl')
           .read_text(encoding='utf-8').split('\n') if l.strip()]
    planned: dict = {}
    paper_len: dict = {}
    paper_sha: dict = {}
    n_versions: dict = {}
    no_text = []
    for p in man:
        safe = p['arxiv_id'].replace('/', '_')
        have = []
        for v in sorted(x['v'] for x in p['versions']):
            tp = repo / 'data/processed/texts' / safe / f'v{v}.md'
            if tp.exists():
                have.append(v)
                b = tp.read_bytes()
                paper_len[(p['arxiv_id'], v)] = len(b.decode('utf-8', 'replace'))
                paper_sha[(p['arxiv_id'], v)] = hashlib.sha256(b).hexdigest()
            else:
                no_text.append((p['arxiv_id'], v))
        for i, v in enumerate(have):
            planned[(p['arxiv_id'], v)] = i
        n_versions[p['arxiv_id']] = len(have)
    planned_keys = {a: {f'{aid}v{v}|{a}|r{rep}' for (aid, v) in planned
                        for rep in range(args.reps)} for a in ARMS}

    latest = dedup(raw_rows, 'clean')
    ok = [r for r in latest.values() if usable(r, 'clean')]
    latest_all = dedup(raw_rows, 'all')
    ok_all = [r for r in latest_all.values() if usable(r, 'all')]

    # =============================================================== 1 KEYS
    H('1. KEYS')
    P(f'manifest: {len(man)} papers, {sum(len(p["versions"]) for p in man)} versions, '
      f'{len(planned)} with a text file (planned paper-versions); no text: {no_text}')
    P(f'versions-with-text per paper: {dict(sorted(Counter(n_versions.values()).items()))}')
    P(f'attempt rows: {len(raw_rows)}; distinct keys: {len(latest)}')
    for a in ARMS:
        rr = by_arm[a]
        keys = Counter(r['key'] for r in rr)
        bad_key = [r['key'] for r in rr
                   if r['key'] != f"{r['arxiv_id']}v{r['v']}|{r['arm']}|r{r['rep']}"]
        outside = sorted({r['key'] for r in rr if r['key'] not in planned_keys[a]})
        dup = {k: c for k, c in keys.items() if c > 1}
        multi_ok = {k: sum(usable(r, 'all') for r in rr if r['key'] == k) for k in dup}
        multi_ok = {k: c for k, c in multi_ok.items() if c > 1}
        n_ok = sum(1 for r in ok if r['arm'] == a)
        n_ok_all = sum(1 for r in ok_all if r['arm'] == a)
        failed_keys = {r['key'] for r in rr if not r.get('parsed_ok')}
        later_done = {k for k in failed_keys if usable(latest_all.get(k), 'all')}
        P(f'-- {a}: attempts {len(rr)} | distinct keys {len(keys)} | parsed_ok rows '
          f'{sum(bool(r.get("parsed_ok")) for r in rr)} | usable keys (clean) {n_ok} of '
          f'{len(planned_keys[a])} planned' + (f' | usable keys (b1-all) {n_ok_all}' if a == 'harness_sub' else ''))
        P(f'   key string != fields: {bad_key or "none"} | keys outside planned design: {outside or "none"}')
        P(f'   keys with >1 attempt row: {len(dup)} {dict(sorted(dup.items()))}')
        P(f'   keys with >1 usable row: {multi_ok or "none"}')
        P(f'   keys that ever failed: {len(failed_keys)}; of these usable now: {len(later_done)}; '
          f'still without a usable row: {sorted(failed_keys - later_done)}')
        okr = [r for r in ok if r['arm'] == a]
        rep_c = Counter(r['rep'] for r in okr)
        P(f'   usable rows by rep index: {dict(sorted(rep_c.items()))}  '
          f'({chi2([[rep_c.get(i, 0) for i in range(args.reps)], [len(planned) - rep_c.get(i, 0) for i in range(args.reps)]])} for done-vs-missing by rep)')
        per_pv = Counter((r['arxiv_id'], r['v']) for r in okr)
        dist = Counter(per_pv.get(k, 0) for k in planned)
        P(f'   usable replicates per planned paper-version: {dict(sorted(dist.items()))}')
        errs = Counter(re.sub(r'\s+', ' ', (r.get('error') or ''))[:110] for r in rr if not r.get('parsed_ok'))
        for e, c in errs.most_common():
            P(f'   error x{c}: {e}')
        KN[f'{a}_attempts'] = len(rr)
        KN[f'{a}_usable_clean'] = n_ok
        if a == 'harness_sub':
            KN['harness_sub_usable_all'] = n_ok_all
            KN['harness_sub_process_ok_values'] = dict(Counter(str(r.get('process_ok')) for r in rr))
    P('rating histogram of usable rows (clean):')
    for a in ARMS:
        c = Counter(r['rating'] for r in ok if r['arm'] == a)
        n = max(1, sum(c.values()))
        P(f'   {a:11s}: ' + '  '.join(f'{k}: {c.get(k, 0)} ({c.get(k, 0) / n:.0%})' for k in ALLOWED_RATINGS)
          + f'  | mean {np.mean([r["rating"] for r in ok if r["arm"] == a]):.3f} | accept=true {np.mean([bool(r["accept"]) for r in ok if r["arm"] == a]):.0%}')
    sid_c = Counter(r['session_id'] for r in raw_rows if r.get('session_id'))
    P(f'session_id reused by more than one row: {sum(c > 1 for c in sid_c.values())}')
    rp_c = Counter(r['raw_path'] for r in raw_rows if r.get('raw_path'))
    P(f'raw_path shared by several rows (later attempt overwrote the file): '
      f'{ {k: c for k, c in rp_c.items() if c > 1} or "none"}')
    rng_bad = [r['key'] for r in ok_all if not on_scale(r)]
    P(f'usable rows with an off-scale or wrongly typed score field: {rng_bad or "none"}')
    inc = Counter((r['arm'], r['rating'], r['accept']) for r in ok_all)
    odd = {k: c for k, c in inc.items() if (k[1] >= 6) != k[2]}
    P(f'rating/accept incoherent rows (accept != rating>=6): {odd or "none"}')
    pk_bad = [r['key'] for r in raw_rows if r.get('peak') is not None and r['peak'] != is_peak(r['_t'])]
    P(f'rows whose logged peak flag disagrees with ts: {pk_bad or "none"}')
    for f in sorted({r['_file'] for r in raw_rows}):
        rr = [r for r in raw_rows if r['_file'] == f]
        ks = Counter(tuple(sorted(k for k in r if not k.startswith('_'))) for r in rr)
        base = max(ks, key=ks.get)
        P(f'{f}: {len(ks)} distinct row schemas (sets of field names)')
        for k, c in ks.items():
            if k != base:
                who = [r for r in rr if tuple(sorted(x for x in r if not x.startswith('_'))) == k]
                P(f'   {c} rows lack {sorted(set(base) - set(k))} / have extra {sorted(set(k) - set(base))}; '
                  f'of these usable (b1-all): {[(r["key"], r["ts"]) for r in who if usable(r, "all")]}')

    # =============================================================== 2 CONFIG
    H('2. CONFIGURATION (usable rows, b1-all set)')
    cfg_fields = ('cfg_sig', 'prompt_sha', 'patch_sha', 'config_sha', 'harness_version', 'model',
                  'served_model', 'effort')
    for a in ARMS:
        rr = [r for r in ok_all if r['arm'] == a]
        P(f'-- {a} ({len(rr)} rows)')
        for f in cfg_fields:
            c = Counter(str(r.get(f)) for r in rr)
            P(f'   {f:16s} {"CONSTANT" if len(c) == 1 else "*** VARIES ***"}  {dict(c)}')
    ps = {a: {r.get('prompt_sha') for r in ok_all if r['arm'] == a} for a in ARMS}
    P(f'prompt_sha shared by direct and harness: {ps["direct"] == ps["harness"]}; '
      f'by all three arms: {ps["direct"] == ps["harness"] == ps["harness_sub"]}')
    for f in ('patch_sha', 'config_sha', 'harness_version'):
        vals = {a: {r.get(f) for r in ok_all if r['arm'] == a} for a in ARMS}
        P(f'{f} identical across arms: {vals["direct"] == vals["harness"] == vals["harness_sub"]}')
    now_prompt = hashlib.sha256((repo / 'src/review/prompts/review_v1.md').read_bytes()).hexdigest()
    now_patch = hashlib.sha256((repo / 'src/review/harness_lockdown.yml').read_bytes()).hexdigest()
    P(f'prompt file on disk now matches logged prompt_sha: {ps["direct"] == {now_prompt}}; '
      f'patch file matches logged patch_sha: '
      f'{ {r.get("patch_sha") for r in ok_all} == {now_patch}}')
    P('note: served_model is not recorded by the harness_sub runner (always None in its log); '
      'see the session-log check below.')

    # =============================================================== 3 INPUTS
    H('3. INPUTS')
    sha_by_pv: dict = defaultdict(set)
    tc: dict = defaultdict(lambda: defaultdict(set))
    for r in raw_rows:
        if r.get('text_sha'):
            sha_by_pv[(r['arxiv_id'], r['v'])].add(r['text_sha'])
        if r.get('task_chars'):
            tc[(r['arxiv_id'], r['v'])][r['arm']].add(r['task_chars'])
    P(f'paper-versions seen in logs: {len(sha_by_pv)}; with >1 distinct text_sha across all '
      f'arms/attempts: {sum(len(s) > 1 for s in sha_by_pv.values())}; rows without text_sha: '
      f'{sum(1 for r in raw_rows if not r.get("text_sha"))}')
    P(f'logged text_sha != sha256 of the text file on disk now: '
      f'{[k for k, s in sha_by_pv.items() if s != {paper_sha.get(k)}] or "none"}')
    prompt_raw = (repo / 'src/review/prompts/review_v1.md').read_text(encoding='utf-8')
    parts = re.split(r"(?m)^#\s+(System|User)\s*$", prompt_raw)
    sec = {parts[i]: parts[i + 1].strip() for i in range(1, len(parts) - 1, 2)}
    overhead = len(sec['System']) + 2 + len(sec['User']) - len('{PAPER_TEXT}')
    dd: Counter = Counter()
    for k, arms in tc.items():
        for a, s in arms.items():
            if len(s) > 1:
                dd[f'{a}: several task_chars for one paper-version'] += 1
            exp = paper_len[k] + overhead + (len(PRIMER) + PROCESS_LEN if a == 'harness_sub' else 0)
            dd[f'{a}: task_chars == len(paper)+{exp - paper_len[k]}'] += int(s == {exp})
            dd[f'{a}: task_chars UNEXPECTED'] += int(s != {exp})
    for k, v in sorted(dd.items()):
        P(f'   {k}: {v}')
    P(f'paper length (chars): min {min(paper_len.values())}, median '
      f'{int(st.median(paper_len.values()))}, max {max(paper_len.values())}')
    for a in ARMS:
        rr = [r for r in ok_all if r['arm'] == a]
        P(f'   {a}: truncated {sum(bool(r.get("truncated")) for r in rr)}, coerced '
          f'{sum(bool(r.get("coerced")) for r in rr)}, usage_missing '
          f'{sum(bool(r.get("usage_missing")) for r in rr)} of {len(rr)} usable rows')
    # token-level plausibility of the input that reached the model
    tok: dict = defaultdict(dict)
    for r in ok_all:
        u = r.get('usage_main') if r['arm'] == 'harness_sub' else r.get('usage')
        if u and r['arm'] != 'harness_sub':
            tok[(r['arxiv_id'], r['v'])].setdefault(r['arm'], []).append(u['input'] + u['cache_read'])
    ratio = [st.mean(d['direct']) / (paper_len[k] + overhead) for k, d in tok.items() if d.get('direct')]
    if ratio:
        P(f'direct: input tokens / task chars per paper-version: min {min(ratio):.3f} median '
          f'{st.median(ratio):.3f} max {max(ratio):.3f}')
    within = [max(d['direct']) - min(d['direct']) for d in tok.values() if len(d.get('direct', [])) > 1]
    if within:
        P(f'direct: spread of input tokens across replicates of one paper-version: max {max(within)} tokens')
    ovh = [st.mean(d['harness']) - st.mean(d['direct']) for d in tok.values()
           if d.get('direct') and d.get('harness')]
    if ovh:
        P(f'harness minus direct input tokens for the same paper-version (harness overhead): '
          f'min {min(ovh):.0f} median {st.median(ovh):.0f} max {max(ovh):.0f} (n={len(ovh)})')
        KN['harness_input_token_overhead_median'] = float(st.median(ovh))
    # identical texts across versions
    by_sha: dict = defaultdict(list)
    for k, s in paper_sha.items():
        by_sha[s].append(k)
    ident = sorted(sorted(v) for v in by_sha.values() if len(v) > 1)
    P(f'groups of planned paper-versions with byte-identical text: {len(ident)}')
    for g in ident:
        P(f'   {g[0][0]}: versions {[v for _, v in g]} (positions {[planned[k] for k in g]}; paper has '
          f'{n_versions[g[0][0]]} versions with text)')
    n_steps_total = sum(n - 1 for n in n_versions.values())
    n_ident_adj = sum(1 for g in ident for i in range(len(g) - 1)
                      if planned[g[i + 1]] - planned[g[i]] == 1)
    P(f'adjacent version steps with zero text change: {n_ident_adj} of {n_steps_total}')
    KN['identical_text_adjacent_steps'] = f'{n_ident_adj} of {n_steps_total}'
    P('placebo: rating of the later minus the earlier version for byte-identical text pairs (should be ~0 in every arm; '
      'its spread shows pure replicate noise at the paper-version level):')
    cell0 = cells_from(ok, planned)
    for a in ARMS:
        d = []
        for g in ident:
            for i in range(len(g) - 1):
                if cell0.get(g[i], {}).get(a) and cell0.get(g[i + 1], {}).get(a):
                    d.append(st.mean(cell0[g[i + 1]][a]) - st.mean(cell0[g[i]][a]))
        if d:
            P(f'   {a:11s}: pairs {len(d)}, mean diff {np.mean(d):+.3f}, mean |diff| {np.mean(np.abs(d)):.3f}, diffs {[round(x, 2) for x in d]}')
    # near-identical adjacent versions (shared-line share)
    near = []
    for p in man:
        vs = sorted(v for (aid, v) in planned if aid == p['arxiv_id'])
        safe = p['arxiv_id'].replace('/', '_')
        for v0, v1 in zip(vs, vs[1:]):
            l0 = [x for x in (repo / 'data/processed/texts' / safe / f'v{v0}.md').read_text('utf-8', 'replace').split('\n') if x.strip()]
            l1 = [x for x in (repo / 'data/processed/texts' / safe / f'v{v1}.md').read_text('utf-8', 'replace').split('\n') if x.strip()]
            c0, c1 = Counter(l0), Counter(l1)
            shared = sum((c0 & c1).values())
            near.append(((p['arxiv_id'], v0, v1), shared / max(len(l0), len(l1), 1)))
    q = [x for _, x in near]
    P(f'adjacent version pairs by share of identical non-empty lines: ==1.0: {sum(x == 1 for x in q)}, '
      f'>=0.99: {sum(x >= 0.99 for x in q)}, >=0.95: {sum(x >= 0.95 for x in q)}, <0.5: {sum(x < 0.5 for x in q)} '
      f'of {len(q)}; median {st.median(q):.3f}')

    # =============================================================== sessions pass
    sess = Sessions(repo / '.tools/dsh-home/sessions')
    S: dict[int, dict] = {}
    F: dict[int, dict] = {}
    if not args.skip_sessions:
        for r in raw_rows:
            if r['arm'] == 'direct' or not r.get('session_id'):
                continue
            log = sess.load(r['session_id'])
            if log is None:
                continue
            s = session_summary(log)
            s['bucket'] = sess.bucket(r['session_id'])
            S[r['_i']] = s
            if r['arm'] == 'harness_sub':
                paper = (repo / 'data/processed/texts' / r['arxiv_id'].replace('/', '_')
                         / f"v{r['v']}.md").read_bytes().decode('utf-8', 'replace')
                F[r['_i']] = sub_process_facts(log, sess, paper)
                F[r['_i']]['_log'] = log
            r['_wall'] = r['_t'].timestamp() - s['t0'] if s['t0'] else None
            r['_excess'] = (r['_wall'] - (r.get('latency_s') or 0)) if r['_wall'] is not None else None

    if S:
        H('2b/3b. WIRE-LEVEL CHECK FROM THE dsh SESSION LOGS (usable rows, b1-all set)')
        for a in ('harness', 'harness_sub'):
            rr = [r for r in ok_all if r['arm'] == a]
            got = [r for r in rr if r['_i'] in S]
            P(f'-- {a}: session log found for {len(got)} of {len(rr)} usable rows')
            agg: dict = defaultdict(Counter)
            exact = 0
            for r in got:
                s = S[r['_i']]
                agg['session bucket (cwd of the dsh process)'][s['bucket']] += 1
                agg['cwd shown to the model'][','.join(s['cwd'])] += 1
                agg['system prompt sha'][','.join(s['sys_sha'])] += 1
                agg['runtime-context block sha'][','.join(s['rc_sha'])] += 1
                agg['request header sha (config+tool catalog)'][','.join(s['hdr_sha'])] += 1
                agg['request config'][','.join(s['hdr_cfg'])] += 1
                agg['tool catalog'][' | '.join(s['tool_names'])] += 1
                for m, c in s['models'].items():
                    agg['served model per assistant message'][m] += c
                paper = (repo / 'data/processed/texts' / r['arxiv_id'].replace('/', '_')
                         / f"v{r['v']}.md").read_bytes().decode('utf-8', 'replace')
                task = sec['System'] + '\n\n' + sec['User'].replace('{PAPER_TEXT}', paper)
                if a == 'harness_sub':
                    task += PRIMER
                exact += bool(s['user_msgs'] and s['user_msgs'][0] == task)
                if a == 'harness_sub' and r['_i'] in F:
                    f = F[r['_i']]
                    for m, c in f['child_models'].items():
                        agg['child: served model per assistant message'][m] += c
                    agg['child: descriptor'][json.dumps(f['child_desc'], sort_keys=True)] += 1
                    agg['child: request config'][','.join(f['child_hdr_cfg'])] += 1
                    agg['child: cwd shown'][','.join(f['child_cwd'])] += 1
            for k, c in agg.items():
                P(f'   {k}: {"CONSTANT" if len(c) == 1 else "*** VARIES ***"} {dict(c)}')
            P(f'   first user message == review prompt + paper text'
              f'{" + PRIMER" if a == "harness_sub" else ""} (exact string match): {exact} of {len(got)}')
            KN[f'{a}_user_msg_exact'] = f'{exact} of {len(got)}'
        hs = {a: {x for r in ok_all if r['arm'] == a and r['_i'] in S for x in S[r['_i']]['sys_sha']}
              for a in ('harness', 'harness_sub')}
        P(f'system prompt identical between harness and harness_sub: {hs["harness"] == hs["harness_sub"]}')
        hh = {a: {x for r in ok_all if r['arm'] == a and r['_i'] in S for x in S[r['_i']]['hdr_sha']}
              for a in ('harness', 'harness_sub')}
        P(f'request header (config + tool catalog) identical between harness and harness_sub: '
          f'{hh["harness"] == hh["harness_sub"]}')
        P('direct arm: the request that was sent is not saved anywhere (raw file = answer text, '
          'usage, thinking_chars), so its wire parameters cannot be re-checked from data; only '
          'the token counts above constrain it.')

    # =============================================================== 2c effort comparability
    H('2c. HOW MUCH WORK EACH ARM DID ON THE SAME PAPER-VERSION (descriptive; usable rows)')
    work: dict = defaultdict(lambda: defaultdict(list))
    for r in ok_all:
        u = r.get('usage') or {}
        rec = {'output tokens': u.get('output', 0), 'input+cache tokens': u.get('input', 0) + u.get('cache_read', 0),
               'latency_s': r.get('latency_s') or 0}
        if r['arm'] == 'direct' and r.get('thinking_chars') is not None:
            rec['reasoning chars'] = r['thinking_chars']
        if r['arm'] == 'harness' and r['_i'] in S:
            am = [x for x in sess.load(r['session_id']) if x.get('type') == 'assistant/message']
            content = (((am[-1].get('data') or {}).get('message') or {}).get('content') or []) if am else []
            rec['reasoning chars'] = sum(len(c.get('text', '')) for c in content if isinstance(c, dict) and c.get('type') == 'reasoning')
        work[(r['arxiv_id'], r['v'])][r['arm']].append(rec)
    for a, b in PAIRS:
        both = [d for d in work.values() if d.get(a) and d.get(b)]
        for fld in ('output tokens', 'reasoning chars', 'input+cache tokens', 'latency_s'):
            xa = [np.mean([x[fld] for x in d[a] if fld in x]) for d in both if any(fld in x for x in d[a]) and any(fld in x for x in d[b])]
            xb = [np.mean([x[fld] for x in d[b] if fld in x]) for d in both if any(fld in x for x in d[a]) and any(fld in x for x in d[b])]
            if len(xa) >= 3 and min(xa) > 0:
                P(f'   {b:11s} vs {a:8s} {fld:19s}: mean {np.mean(xa):9.0f} -> {np.mean(xb):9.0f}; per-paper-version ratio median '
                  f'{np.median(np.array(xb) / np.array(xa)):.2f} (n={len(xa)})')
                KN[f'ratio_{b}_over_{a}_{fld}'] = round(float(np.median(np.array(xb) / np.array(xa))), 3)
    P('   (the direct request itself is not saved, so equal thinking/effort settings on the wire cannot be confirmed from '
      'data; the numbers above show the harness arm reasons much longer on identical input)')

    # =============================================================== 4 RAW RE-DERIVATION
    if not args.skip_raw:
        H('4. RE-DERIVATION OF THE SCORES FROM THE RAW OUTPUTS (every parsed_ok row)')
        res: dict = defaultdict(Counter)
        mism = []
        text_hash: dict = defaultdict(list)
        mt_off = []
        for r in raw_rows:
            if not r.get('parsed_ok'):
                continue
            a = r['arm']
            p = repo / (r.get('raw_path') or '__none__')
            if not p.is_file():
                res[a]['RAW FILE MISSING'] += 1
                mism.append((r['key'], 'raw file missing', r.get('raw_path')))
                continue
            stem = f"{r['arxiv_id'].replace('/', '_')}v{r['v']}_r{r['rep']}"
            if not p.name.startswith(stem + '.') or p.parent.name != a:
                res[a]['raw_path does not match key'] += 1
                mism.append((r['key'], 'raw_path name mismatch', r['raw_path']))
            mt_off.append((p.stat().st_mtime - r['_t'].timestamp(), r['key']))
            final_rv = None
            if a == 'direct':
                d = json.loads(p.read_text(encoding='utf-8'))
                c = find_reviews(d.get('text') or '')
                res[a][f'JSON objects with all score fields in answer: {len(c)}'] += 1
                res[a]['answer is a bare JSON object' if (d.get('text') or '').strip()[:1] == '{'
                       else 'answer has leading non-JSON text or a code fence'] += 1
                final_rv = c[-1] if c else None
                if d.get('usage') != r.get('usage'):
                    res[a]['usage in raw != usage in log'] += 1
                    mism.append((r['key'], 'usage mismatch'))
                if d.get('thinking_chars') != r.get('thinking_chars'):
                    res[a]['thinking_chars raw != log'] += 1
                text_hash[sha12(d.get('text') or '')].append(r['key'])
            else:
                ev = read_events(p)
                finals = [e for e in ev if e.get('type') == 'final']
                sids = {e.get('sessionId') for e in ev if e.get('type') == 'session'}
                if sids != {r.get('session_id')}:
                    res[a]['session id in raw stream != log'] += 1
                    mism.append((r['key'], 'session id mismatch', sorted(map(str, sids))))
                if any(e.get('type') == '_unparseable' for e in ev):
                    res[a]['unparseable line in raw stream'] += 1
                if a == 'harness':
                    res[a][f'final events: {len(finals)}'] += 1
                    ft = (finals[-1].get('text') or '') if finals else ''
                    c = find_reviews(ft)
                    final_rv = c[-1] if c else None
                    res[a]['answer is a bare JSON object' if ft.strip()[:1] == '{'
                           else 'answer has leading non-JSON text or a code fence'] += 1
                    n_step = sum(1 for e in ev if e.get('type') == 'status' and e.get('phase') == 'step_end')
                    if n_step != r.get('n_steps'):
                        res[a]['n_steps raw != log'] += 1
                    oth = Counter(e.get('type') for e in ev) - Counter({k: 10 ** 9 for k in ('session', 'status', 'thinking', 'text', 'final')})
                    if dict(oth) != (r.get('other_events') or {}):
                        res[a]['other_events raw != log'] += 1
                    text_hash[sha12(ft)].append(r['key'])
                else:
                    # (i) the event stream: every review-looking text / final event
                    stream_rv = []
                    for e in ev:
                        if e.get('type') in ('text', 'final'):
                            c = find_reviews(e.get('text') or '')
                            if c:
                                stream_rv.append(c[-1])
                    last_final = find_reviews((finals[-1].get('text') or '') if finals else '')
                    res[a]['stream: last `final` event is ' + ('a review' if last_final else 'NOT a review (closing remark)')] += 1
                    if not stream_rv:
                        res[a]['stream: no parseable review at all (text events are cut at 8 KiB)'] += 1
                    bad_stream = [diff_fields(x, r) for x in stream_rv if diff_fields(x, r)]
                    if bad_stream:
                        res[a]['stream: a review in the stream differs from the logged scores'] += 1
                        mism.append((r['key'], 'stream review differs from log', bad_stream))
                    elif stream_rv:
                        res[a]['stream: all reviews in the stream equal the logged scores'] += 1
                    # (ii) the session log: first review after the report
                    f = F.get(r['_i'])
                    if f is None:
                        res[a]['session log unavailable'] += 1
                        final_rv = stream_rv[-1] if stream_rv else None
                    else:
                        final_rv = f['result']
                        distinct = {json.dumps({k: x[1].get(k) for k in FIELDS}, sort_keys=True) for x in f['reviews']}
                        res[a][f'session: review outputs by the main agent: {len(f["reviews"])}'] += 1
                        if len(distinct) > 1:
                            res[a]['session: review outputs within the run DIFFER in scores'] += 1
                            mism.append((r['key'], 'several reviews with different scores in one run',
                                         [(i, {k: x.get(k) for k in FIELDS}) for i, x, _ in f['reviews']]))
                        if f['result'] is not None:
                            text_hash[sha12(json.dumps(f['result'], sort_keys=True))].append(r['key'])
            if final_rv is None:
                res[a]['NO REVIEW FOUND'] += 1
                mism.append((r['key'], 'no review found by the audit parser'))
                continue
            d = diff_fields(final_rv, r)
            if d:
                res[a]['MISMATCH with logged scores'] += 1
                mism.append((r['key'], 'score mismatch (raw, logged)', d))
            else:
                res[a]['scores identical to log (6 fields, value and type)'] += 1
            if not on_scale(final_rv):
                res[a]['raw value off-scale (needed coercion)'] += 1
        for a in ARMS:
            P(f'-- {a}')
            for k, v in sorted(res[a].items()):
                P(f'   {v:4d}  {k}')
            KN[f'{a}_raw_match'] = f"{res[a]['scores identical to log (6 fields, value and type)']} of {sum(bool(r.get('parsed_ok')) for r in by_arm[a])}"
        P(f'mismatches / problems: {len(mism)}')
        for m in mism:
            P('   ', m)
        dup_text = {h: ks for h, ks in text_hash.items() if len(ks) > 1}
        P(f'identical answer text shared by different runs (would indicate a cached/replayed response): '
          f'{len(dup_text)} {list(dup_text.values())[:5] if dup_text else ""}')
        if mt_off:
            late = [(round(o), k) for o, k in mt_off if abs(o) > 120]
            P(f'raw file mtime minus logged ts: median {st.median(o for o, _ in mt_off):+.1f}s; '
              f'files modified >120 s away from their row ts (overwritten later?): {late or "none"}')

    # =============================================================== bursts / sessions in time
    all_sorted = sorted(raw_rows, key=lambda r: (r['_t'], r['_i']))
    burst = 0
    bursts: list[dict] = []
    for i, r in enumerate(all_sorted):
        if i and (r['_t'] - all_sorted[i - 1]['_t']).total_seconds() > args.gap_min * 60:
            burst += 1
        r['_burst'] = burst
        if burst == len(bursts):
            bursts.append({'start': r['_t'], 'rows': []})
        bursts[burst]['rows'].append(r)
        bursts[burst]['end'] = r['_t']
    for b in bursts:
        b['n_ok'] = sum(usable(r, 'all') for r in b['rows'])
        b['trickle'] = len(b['rows']) < 15 and (b['end'] - b['start']).total_seconds() < 180
    for r in raw_rows:
        b = bursts[r['_burst']]
        r['_since_burst_start'] = (r['_t'] - b['start']).total_seconds()

    # =============================================================== 5 SLEEP
    H('5. SLEEP CONTAMINATION')
    P(f'union completion timeline (both logs), silence > {args.gap_min:g} min = new burst. '
      f'NOTE latency_s is measured with time.monotonic(), which does not advance while a Mac '
      f'sleeps, so latency_s cannot reveal a sleep; wall-clock comes from ts and the dsh session logs.')
    P('burst  start(UTC)        end(UTC)          dur    direct ok | harness ok/fail | sub ok/fail(tech)/other | peak rows')
    for i, b in enumerate(bursts):
        c = Counter((r['arm'], bool(r.get('parsed_ok'))) for r in b['rows'])
        P(f'  B{i:<3d} {b["start"].strftime("%m-%d %H:%M:%S")}  {b["end"].strftime("%m-%d %H:%M:%S")}  '
          f'{(b["end"] - b["start"]).total_seconds() / 60:5.1f}m   {c[("direct", True)]:4d}/{c[("direct", False)]}    | '
          f'{c[("harness", True)]:4d}/{c[("harness", False)]}       | {c[("harness_sub", True)]:4d}/{c[("harness_sub", False)]}'
          f'                   | {sum(bool(r.get("peak")) for r in b["rows"])} of {len(b["rows"])}'
          f'{"   <- trickle burst (isolated, <15 rows, <3 min)" if b["trickle"] else ""}')
    for i in range(1, len(bursts)):
        g = (bursts[i]['start'] - bursts[i - 1]['end']).total_seconds() / 60
        P(f'  gap before B{i}: {g:.1f} min')
    # flags
    for r in raw_rows:
        if not usable(r, 'all'):
            continue
        if (r.get('failed_attempts') or 0) > 0:
            r['_flags'].add('failed_attempt')
        if (r.get('llm_retries') or 0) > 0:
            r['_flags'].add('llm_retry')
        if r.get('_excess') is not None and r['_excess'] > 60:
            r['_flags'].add('wall_excess')
        if bursts[r['_burst']]['trickle']:
            r['_flags'].add('trickle_burst')
        if r['_burst'] > 0 and r['_since_burst_start'] <= args.wake_window:
            r['_flags'].add('wake_window')
        if r['_burst'] == 0 and r['_since_burst_start'] <= args.wake_window and bursts[0]['trickle']:
            r['_flags'].add('wake_window')
        if r.get('peak'):
            r['_flags'].add('peak')
    lat_thr = {}
    for a in ARMS:
        ls = sorted(r['latency_s'] for r in ok_all if r['arm'] == a and r.get('latency_s') is not None)
        if ls:
            q1, q3 = np.percentile(ls, [25, 75])
            lat_thr[a] = q3 + 3 * (q3 - q1)
            P(f'-- {a}: latency_s min {ls[0]:.1f} median {st.median(ls):.1f} p95 {np.percentile(ls, 95):.1f} '
              f'max {ls[-1]:.1f}; outlier threshold Q3+3*IQR = {lat_thr[a]:.0f}s; '
              f'rows above: {sum(x > lat_thr[a] for x in ls)}; rows < 5 s: {sum(x < 5 for x in ls)}')
    for r in ok_all:
        if r.get('latency_s') is not None and r['latency_s'] > lat_thr.get(r['arm'], 1e9):
            r['_flags'].add('latency_outlier')
    P('usable rows (b1-all) by flag and arm:')
    for fl in ('wall_excess', 'trickle_burst', 'failed_attempt', 'llm_retry', 'wake_window', 'latency_outlier', 'peak'):
        P(f'   {fl:16s} ' + ' | '.join(f'{a} {sum(fl in r["_flags"] for r in ok_all if r["arm"] == a):3d}' for a in ARMS))
    P('rows that provably or very probably spanned a machine sleep (wall_excess: dsh session '
      'started > latency_s + 60 s before ts; trickle_burst: completed in an isolated burst):')
    span = [r for r in ok_all if r['_flags'] & {'wall_excess', 'trickle_burst'}]
    for r in sorted(span, key=lambda r: r['_t']):
        P(f'   {r["key"]:34s} ts {r["ts"][5:19]} latency_s {r["latency_s"]:6.1f} wall '
          f'{("%.0f" % r["_wall"]) if r.get("_wall") is not None else "n/a":>6s}s failed_attempts '
          f'{r.get("failed_attempts")} llm_retries {r.get("llm_retries")} rating {r["rating"]} '
          f'(other usable reps of this cell: {[x["rating"] for x in ok_all if x["arm"] == r["arm"] and x["arxiv_id"] == r["arxiv_id"] and x["v"] == r["v"] and x is not r]}) '
          f'flags {sorted(r["_flags"])}')
    KN['sleep_spanning_usable_rows'] = [r['key'] for r in span]
    P('latency outliers:')
    for r in sorted((r for r in ok_all if 'latency_outlier' in r['_flags']), key=lambda r: r['_t']):
        P(f'   {r["key"]:34s} ts {r["ts"][5:19]} latency_s {r["latency_s"]:6.1f} retries {r.get("llm_retries")} rating {r["rating"]}')
    if S:
        codes: Counter = Counter()
        for r in ok_all:
            if r['_i'] in S:
                for c, n in S[r['_i']]['retry_codes'].items():
                    codes[(r['arm'], 'main', c)] += n
            if r['_i'] in F:
                for c, n in F[r['_i']]['child_retry_codes'].items():
                    codes[(r['arm'], 'child', c)] += n
        P(f'dsh internal retry causes in usable rows (session logs): {dict(codes)}')
        bad = [(r['key'], r.get('llm_retries'), S[r['_i']]['retries'] + (F[r['_i']]['child_retries'] if r['_i'] in F else 0))
               for r in ok_all if r['_i'] in S
               and r.get('llm_retries') != S[r['_i']]['retries'] + (F[r['_i']]['child_retries'] if r['_i'] in F else 0)]
        P(f'logged llm_retries != retries counted in session logs: {bad or "none"}')
    P('retry rate by burst (usable rows with llm_retries>0 or failed_attempts>0 / usable rows):')
    for i, b in enumerate(bursts):
        if b['n_ok']:
            P(f'   B{i}: ' + ' | '.join(
                f'{a} {sum(bool(r["_flags"] & {"llm_retry", "failed_attempt"}) for r in b["rows"] if r["arm"] == a and usable(r, "all"))}'
                f'/{sum(1 for r in b["rows"] if r["arm"] == a and usable(r, "all"))}' for a in ARMS))

    # sensitivity
    P('')
    P('SENSITIVITY of the headline numbers (clean B1 set; "pv-mean" = mean over matched '
      'paper-versions of the replicate mean; "paper-level" and "pooled slope" follow analyze_e1h.py):')
    base_cells = cells_from(ok, planned)
    scen = [
        ('baseline (all usable rows)', lambda r: False),
        ('drop sleep-spanning rows (wall_excess or trickle_burst)', lambda r: bool(r['_flags'] & {'wall_excess', 'trickle_burst'})),
        ('drop rows with failed_attempts>0 or llm_retries>0', lambda r: bool(r['_flags'] & {'failed_attempt', 'llm_retry'})),
        (f'drop rows completed within {args.wake_window:.0f}s of a burst start (wake-up)', lambda r: 'wake_window' in r['_flags']),
        ('drop latency outliers', lambda r: 'latency_outlier' in r['_flags']),
        ('drop all of the above', lambda r: bool(r['_flags'] & {'wall_excess', 'trickle_burst', 'failed_attempt', 'llm_retry', 'wake_window', 'latency_outlier'})),
        ('drop peak-price rows', lambda r: 'peak' in r['_flags']),
        ('keep only peak-price rows', lambda r: 'peak' not in r['_flags']),
    ]
    sens = {}
    for name, drop in scen:
        kept = [r for r in ok if not drop(r)]
        cell = cells_from(kept, planned)
        P(f'* {name}: rows kept ' + ', '.join(f'{a} {sum(r["arm"] == a for r in kept)}' for a in ARMS))
        sens[name] = {}
        for a, b in PAIRS:
            c = compare(cell, planned, a, b)
            sens[name][f'{b}-{a}'] = c
            P(f'     {b:11s} - {a:8s}: {fmt_cmp(c)}')
    # byte-identical versions are the same input: pool their replicates and give both positions the pooled ratings
    pooled_cell: dict = defaultdict(lambda: defaultdict(list))
    for k, d in base_cells.items():
        for a, v in d.items():
            pooled_cell[k][a] = list(v)
    for g in ident:
        for a in ARMS:
            allv = [x for k in g for x in base_cells.get(k, {}).get(a, [])]
            for k in g:
                if allv:
                    pooled_cell[k][a] = list(allv)
    name = 'pool replicates across byte-identical versions (same input => same expected rating)'
    P(f'* {name}:')
    sens[name] = {}
    for a, b in PAIRS:
        c = compare(pooled_cell, planned, a, b)
        sens[name][f'{b}-{a}'] = c
        P(f'     {b:11s} - {a:8s}: {fmt_cmp(c)}')
    KN['sensitivity'] = sens
    P('reference: how much the same number of rows dropped AT RANDOM (per arm) moves the estimates (200 draws; '
      'SD of the change, and the share of random draws with a change at least as large in absolute value as the scenario):')
    rng = random.Random(7)
    base = sens[scen[0][0]]
    okl = {a: [r for r in ok if r['arm'] == a] for a in ARMS}
    for name, drop in scen[1:7]:
        nd = {a: sum(drop(r) for r in okl[a]) for a in ARMS}
        ch: dict = defaultdict(list)
        for _ in range(200):
            kept = []
            for a in ARMS:
                kept += rng.sample(okl[a], len(okl[a]) - nd[a])
            cell = cells_from(kept, planned)
            for a, b in PAIRS:
                c = compare(cell, planned, a, b)
                if 'slope_diff' in c:
                    ch[(f'{b}-{a}', 'slope_diff')].append(c['slope_diff'] - base[f'{b}-{a}']['slope_diff'])
                    ch[(f'{b}-{a}', 'pv_diff')].append(c['pv_diff'] - base[f'{b}-{a}']['pv_diff'])
        P(f'* {name} (drops {nd}):')
        for a, b in PAIRS:
            k = f'{b}-{a}'
            if 'slope_diff' not in sens[name][k]:
                continue
            for q in ('pv_diff', 'slope_diff'):
                obs = sens[name][k][q] - base[k][q]
                arr = np.array(ch[(k, q)])
                P(f'     {k:20s} {q:10s}: scenario change {obs:+.3f}; random-drop SD {arr.std():.3f}; '
                  f'share of random draws with |change| >= |scenario| {np.mean(np.abs(arr) >= abs(obs) - 1e-12):.2f}')
    tri = {k: d for k, d in base_cells.items() if all(d.get(a) for a in ARMS)}
    if tri:
        P(f'paper-versions usable in all three arms: {len(tri)}; pv-mean rating '
          + ', '.join(f'{a} {np.mean([st.mean(d[a]) for d in tri.values()]):.3f}' for a in ARMS))
    cell_all = cells_from(ok_all, planned)
    P('b1-all variant (every process-verified B1 run):')
    for a, b in PAIRS[1:]:
        P(f'     {b:11s} - {a:8s}: {fmt_cmp(compare(cell_all, planned, a, b))}')

    # =============================================================== 6 TIME CONFOUNDS
    H('6. TIME CONFOUNDS')
    big = [i for i, b in enumerate(bursts) if b['n_ok'] >= 30]
    P(f'bursts with >= 30 usable rows: {["B%d" % i for i in big]}')
    P('usable rows (clean) by burst and arm, with share of the arm:')
    for i, b in enumerate(bursts):
        if not b['n_ok']:
            continue
        P(f'   B{i} [{b["start"].strftime("%m-%d %H:%M")}-{b["end"].strftime("%H:%M")} UTC, '
          f'{"peak" if is_peak(b["start"]) else "off-peak"}]: ' + ' | '.join(
            f'{a} {sum(1 for r in ok if r["arm"] == a and r["_burst"] == i):3d} '
            f'({sum(1 for r in ok if r["arm"] == a and r["_burst"] == i) / max(1, sum(1 for r in ok if r["arm"] == a)):.1%})'
            for a in ARMS))
    P(f'arm x burst independence (usable rows): '
      f'{chi2([[sum(1 for r in ok if r["arm"] == a and r["_burst"] == i) for i in range(len(bursts))] for a in ARMS])}')
    P('arm x peak (usable rows): ' + ' | '.join(
        f'{a} peak {sum(bool(r.get("peak")) for r in ok if r["arm"] == a)}/{sum(1 for r in ok if r["arm"] == a)}' for a in ARMS)
      + '  ' + chi2([[sum(bool(r.get('peak')) for r in ok if r['arm'] == a), sum(not r.get('peak') for r in ok if r['arm'] == a)] for a in ARMS]))
    P('interleaving inside each large burst (completion time, minutes after burst start):')
    for i in big:
        b = bursts[i]
        tt = {a: [(r['_t'] - b['start']).total_seconds() / 60 for r in b['rows'] if r['arm'] == a and usable(r, 'all')] for a in ARMS}
        P(f'   B{i}: ' + ' | '.join(f'{a} n={len(tt[a])} median {st.median(tt[a]):.1f} IQR [{np.percentile(tt[a], 25):.1f}, {np.percentile(tt[a], 75):.1f}]'
                                     for a in ARMS if tt[a]))
        for a, c in PAIRS:
            if len(tt[a]) > 5 and len(tt[c]) > 5:
                ks = stats.ks_2samp(tt[a], tt[c])
                P(f'       KS {a} vs {c}: D={ks.statistic:.3f} p={ks.pvalue:.3f}')
        seq = [r['arm'] for r in sorted(b['rows'], key=lambda r: (r['_t'], r['_i'])) if r['arm'] in ('direct', 'harness') and usable(r, 'all')]
        if len(seq) > 20:
            runs = 1 + sum(seq[j] != seq[j - 1] for j in range(1, len(seq)))
            n1, n2 = seq.count('direct'), seq.count('harness')
            mu = 2 * n1 * n2 / (n1 + n2) + 1
            sd = np.sqrt(2 * n1 * n2 * (2 * n1 * n2 - n1 - n2) / ((n1 + n2) ** 2 * (n1 + n2 - 1)))
            P(f'       direct/harness completion order: {runs} runs, expected {mu:.1f} under random mixing (z={(runs - mu) / sd:+.2f})')
    # paired time offset between arms for the same paper-version
    tm: dict = defaultdict(lambda: defaultdict(list))
    for r in ok:
        tm[(r['arxiv_id'], r['v'])][r['arm']].append(r['_t'].timestamp())
    for a, c in PAIRS:
        off = [(st.mean(d[c]) - st.mean(d[a])) / 60 for d in tm.values() if d.get(a) and d.get(c)]
        if off:
            P(f'   mean completion time of {c} minus {a} for the same paper-version: median {st.median(off):+.1f} min, '
              f'mean {st.mean(off):+.1f} min, IQR [{np.percentile(off, 25):+.1f}, {np.percentile(off, 75):+.1f}] (n={len(off)})')
    # peak vs off-peak and burst vs burst, same paper-version, within arm
    P('')
    P('within-arm, same paper-version: mean rating of peak rows minus off-peak rows (paper-versions having both):')
    pooled_d = []
    for a in ARMS:
        g: dict = defaultdict(lambda: {True: [], False: []})
        for r in (ok if a == 'harness_sub' else ok_all):
            if r['arm'] == a:
                g[(r['arxiv_id'], r['v'])][bool(r.get('peak'))].append(r)
        d = [st.mean(x['rating'] for x in v[True]) - st.mean(x['rating'] for x in v[False]) for v in g.values() if v[True] and v[False]]
        pooled_d += d
        P(f'   {a:11s}: {paired_test(d)}')
        KN[f'peak_minus_offpeak_{a}'] = paired_test(d)
        for fld, fn in (('output tokens', lambda x: (x.get('usage') or {}).get('output', 0)),
                        ('latency_s', lambda x: x.get('latency_s') or 0)):
            dd2 = [st.mean(fn(x) for x in v[True]) / max(1e-9, st.mean(fn(x) for x in v[False])) for v in g.values() if v[True] and v[False]]
            if dd2:
                P(f'                {fld}: median ratio peak/off-peak {st.median(dd2):.2f} (n={len(dd2)})')
        for sf in ('soundness', 'presentation', 'contribution', 'confidence'):
            dd3 = [st.mean(x[sf] for x in v[True]) - st.mean(x[sf] for x in v[False]) for v in g.values() if v[True] and v[False]]
            if len(dd3) >= 3:
                P(f'                {sf}: mean diff {np.mean(dd3):+.3f} (p={stats.ttest_1samp(dd3, 0).pvalue:.3f})')
    P(f'   all arms pooled: {paired_test(pooled_d)}')
    KN['peak_minus_offpeak_pooled'] = paired_test(pooled_d)
    if len(big) >= 2:
        P('same comparison between the large bursts (later minus earlier):')
        for x in range(len(big)):
            for y in range(x + 1, len(big)):
                for a in ARMS:
                    g2: dict = defaultdict(lambda: {0: [], 1: []})
                    for r in ok:
                        if r['arm'] == a and r['_burst'] in (big[x], big[y]):
                            g2[(r['arxiv_id'], r['v'])][int(r['_burst'] == big[y])].append(r['rating'])
                    d = [st.mean(v[1]) - st.mean(v[0]) for v in g2.values() if v[0] and v[1]]
                    P(f'   B{big[y]} - B{big[x]} {a:11s}: {paired_test(d)}')
    P('arm differences estimated separately inside each period (only rows of that period):')
    for name, keep in (('off-peak rows only', lambda r: not r.get('peak')), ('peak rows only', lambda r: bool(r.get('peak')))):
        cell = cells_from([r for r in ok if keep(r)], planned)
        for a, c in PAIRS:
            P(f'   {name:19s} {c:11s} - {a:8s}: {fmt_cmp(compare(cell, planned, a, c))}')
    # drift inside bursts
    P('drift inside a burst: Spearman correlation of completion time with (rating - mean rating of the same paper-version and arm):')
    mean_cell = {(k, a): st.mean(v) for k, d in cells_from(ok_all, planned).items() for a, v in d.items() if len(v) >= 2}
    for i in big:
        for a in ARMS:
            xs, ys = [], []
            for r in ok_all:
                if r['arm'] == a and r['_burst'] == i and ((r['arxiv_id'], r['v']), a) in mean_cell:
                    xs.append(r['_t'].timestamp())
                    ys.append(r['rating'] - mean_cell[((r['arxiv_id'], r['v']), a)])
            if len(xs) > 10:
                sp = stats.spearmanr(xs, ys)
                P(f'   B{i} {a:11s}: rho={sp.statistic:+.3f} p={sp.pvalue:.3f} (n={len(xs)})')
    P('prompt-cache state (provider-side): cache-miss rows (cached share of input < 50%) vs cache-hit rows of the same paper-version:')
    for a in ('direct', 'harness'):
        g3: dict = defaultdict(lambda: {True: [], False: []})
        n_miss = 0
        for r in ok:
            if r['arm'] != a:
                continue
            u = r.get('usage') or {}
            tot = (u.get('cache_read', 0) + u.get('input', 0)) or 1
            m = u.get('cache_read', 0) / tot < 0.5
            n_miss += m
            g3[(r['arxiv_id'], r['v'])][m].append(r['rating'])
        d = [st.mean(v[True]) - st.mean(v[False]) for v in g3.values() if v[True] and v[False]]
        P(f'   {a:11s}: cache-miss rows {n_miss}; rating miss minus hit: {paired_test(d)}')
    # rep x burst, position x burst
    P('rep index x burst (usable rows) — is rep confounded with session?')
    for a in ARMS:
        tab = [[sum(1 for r in ok if r['arm'] == a and r['rep'] == rep and r['_burst'] == i) for i in big] for rep in range(args.reps)]
        P(f'   {a:11s}: rows rep0..rep{args.reps - 1} x bursts {["B%d" % i for i in big]} = {tab}  {chi2(tab)}')
    P('version position x period (usable rows) — is position confounded with session/peak?')
    for a in ARMS:
        rr = [r for r in ok if r['arm'] == a]
        pos_peak = [planned[(r['arxiv_id'], r['v'])] for r in rr if r.get('peak')]
        pos_off = [planned[(r['arxiv_id'], r['v'])] for r in rr if not r.get('peak')]
        tab = [[sum(p == 0 for p in pos_off), sum(p > 0 for p in pos_off)], [sum(p == 0 for p in pos_peak), sum(p > 0 for p in pos_peak)]]
        rel_off = [planned[(r['arxiv_id'], r['v'])] / max(1, n_versions[r['arxiv_id']] - 1) for r in rr if not r.get('peak')]
        rel_peak = [planned[(r['arxiv_id'], r['v'])] / max(1, n_versions[r['arxiv_id']] - 1) for r in rr if r.get('peak')]
        if pos_peak and pos_off:
            mw = stats.mannwhitneyu(rel_off, rel_peak)
            P(f'   {a:11s}: mean position off-peak {np.mean(pos_off):.2f} vs peak {np.mean(pos_peak):.2f}; first-version share '
              f'off-peak {tab[0][0] / max(1, sum(tab[0])):.2f} vs peak {tab[1][0] / max(1, sum(tab[1])):.2f} ({chi2(tab)}); '
              f'relative position (0=first,1=last) {np.mean(rel_off):.3f} vs {np.mean(rel_peak):.3f} (MW p={mw.pvalue:.3f})')
        else:
            P(f'   {a:11s}: only one period present')
    # share of peak rows by position, per arm (direct evidence for slope bias)
    P('share of peak rows among usable rows by version position (0 / 1 / 2+):')
    for a in ARMS:
        rr = [r for r in ok if r['arm'] == a]
        sh = []
        for lo, hi in ((0, 0), (1, 1), (2, 99)):
            sel = [r for r in rr if lo <= planned[(r['arxiv_id'], r['v'])] <= hi]
            sh.append(f'{sum(bool(r.get("peak")) for r in sel)}/{len(sel)}')
        P(f'   {a:11s}: {sh}')

    # =============================================================== 7 MISSINGNESS
    H('7. MISSINGNESS (snapshot is mid-run; some is expected)')
    cnt = {a: Counter((r['arxiv_id'], r['v']) for r in ok if r['arm'] == a) for a in ARMS}
    attempted = {a: {r['key'] for r in by_arm[a]} for a in ARMS}
    for a in ARMS:
        miss_keys = planned_keys[a] - {r['key'] for r in ok if r['arm'] == a}
        never = miss_keys - attempted[a]
        zero = sorted(k for k in planned if cnt[a].get(k, 0) == 0)
        P(f'-- {a}: planned runs {len(planned_keys[a])}, usable {sum(cnt[a].values())}, missing {len(miss_keys)} '
          f'(never attempted {len(never)}, attempted without a usable result {len(miss_keys - never)})')
        P(f'   paper-versions with 0 usable replicates: {len(zero)}: {[f"{x}v{v}" for x, v in zero]}')
        n_done = np.array([cnt[a].get(k, 0) for k in planned])
        L = np.array([paper_len[k] for k in planned])
        pos = np.array([planned[k] for k in planned])
        rel = np.array([planned[k] / max(1, n_versions[k[0]] - 1) for k in planned])
        nv = np.array([n_versions[k[0]] for k in planned])
        sp_l = stats.spearmanr(L, n_done)
        sp_p = stats.spearmanr(pos, n_done)
        sp_r = stats.spearmanr(rel, n_done)
        sp_n = stats.spearmanr(nv, n_done)
        P(f'   usable replicates per paper-version vs paper length: Spearman rho={sp_l.statistic:+.3f} p={sp_l.pvalue:.3f}; '
          f'vs position: rho={sp_p.statistic:+.3f} p={sp_p.pvalue:.3f}; vs relative position: rho={sp_r.statistic:+.3f} '
          f'p={sp_r.pvalue:.3f}; vs number of versions of the paper: rho={sp_n.statistic:+.3f} p={sp_n.pvalue:.3f}')
        ter = np.percentile(L, [33.3, 66.7])
        P('   mean usable replicates by length tercile (short/mid/long): '
          + ' / '.join(f'{n_done[(L > lo) & (L <= hi)].mean():.2f}' for lo, hi in ((-1, ter[0]), (ter[0], ter[1]), (ter[1], 1e12)))
          + ' | by position 0 / 1 / 2+: ' + ' / '.join(f'{n_done[(pos >= lo) & (pos <= hi)].mean():.2f}' for lo, hi in ((0, 0), (1, 1), (2, 99)))
          + f' | first vs last version of each paper: {n_done[rel == 0].mean():.2f} / {n_done[rel == 1].mean():.2f}')
        KN[f'{a}_pv_zero_usable'] = len(zero)
    # technical failures vs length
    for a in ('harness', 'harness_sub'):
        fl = [paper_len[(r['arxiv_id'], r['v'])] for r in by_arm[a] if not r.get('parsed_ok')]
        okl = [paper_len[(r['arxiv_id'], r['v'])] for r in by_arm[a] if r.get('parsed_ok')]
        if fl and okl:
            mw = stats.mannwhitneyu(fl, okl)
            P(f'   {a}: paper length of failed attempts median {int(st.median(fl))} vs successful {int(st.median(okl))} (MW p={mw.pvalue:.3f}, n_failed={len(fl)})')
    P('matched cells:')
    for a, b in PAIRS:
        both = [k for k in planned if cnt[a].get(k) and cnt[b].get(k)]
        papers2 = Counter(k[0] for k in both)
        P(f'   {a} & {b}: {len(both)} of {len(planned)} paper-versions; papers with >=2 matched versions: '
          f'{sum(c >= 2 for c in papers2.values())} of {len(man)}; papers with all versions matched: '
          f'{sum(papers2.get(p["arxiv_id"], 0) == n_versions[p["arxiv_id"]] for p in man)}')
    P(f'   all three arms: {sum(1 for k in planned if all(cnt[a].get(k) for a in ARMS))} of {len(planned)}')
    # replicate imbalance direction
    for a, b in PAIRS:
        d = [cnt[b].get(k, 0) - cnt[a].get(k, 0) for k in planned]
        P(f'   replicates {b} minus {a} per paper-version: mean {np.mean(d):+.2f}; by position 0/1/2+: '
          + ' / '.join(f'{np.mean([x for x, k in zip(d, planned) if lo <= planned[k] <= hi]):+.2f}' for lo, hi in ((0, 0), (1, 1), (2, 99))))
    # reconstruct the first-invocation task order (seed 42 / 43) as a randomness check
    def build(arms: tuple[str, ...]) -> list[str]:
        t = []
        for p in man:
            for vr in p['versions']:
                if (p['arxiv_id'], vr['v']) not in planned:
                    continue
                for a in arms:
                    for rep in range(args.reps):
                        t.append(f"{p['arxiv_id']}v{vr['v']}|{a}|r{rep}")
        return t
    def reconstruct(label: str, arms: tuple[str, ...], seed: int, finished) -> None:
        """Replay the runner's queue. Each invocation shuffles (fixed seed) the keys that are
        not finished yet and works through them in order with 12 workers, so the rows of one
        invocation occupy a compact range of queue indices. For every burst of rows the replay
        tries (a) continuing the current queue and (b) a fresh queue built from what was
        finished before the burst, and keeps the one that fits. Shows whether the missing keys
        are simply the unprocessed tail of the queue."""
        rows = [r for r in raw_rows if r['arm'] in arms]
        rows.sort(key=lambda r: (r['_file'], r['_line']))
        done: set = set()
        inv: list[dict] = []
        cur = None
        unexplained = 0
        for b in sorted({r['_burst'] for r in rows}):
            br = [r for r in rows if r['_burst'] == b]

            def excess(c: dict | None) -> float:
                if c is None or any(r['key'] not in c['order'] or r['key'] in c['seen'] for r in br):
                    return float('inf')
                return max(c['order'][r['key']] for r in br) - (len(c['seen']) + len(br))
            t = [k for k in build(arms) if k not in done]
            random.Random(seed).shuffle(t)
            fresh = {'order': {k: i for i, k in enumerate(t)}, 'queue': t, 'seen': set(), 'rows': 0,
                     'first': br[0]['ts'], 'bursts': []}
            e_cont, e_fresh = excess(cur), excess(fresh)
            if min(e_cont, e_fresh) > 40:
                unexplained += len(br)
            else:
                if e_fresh < e_cont:
                    cur = fresh
                    inv.append(cur)
                for r in br:
                    cur['seen'].add(r['key'])
                cur['rows'] += len(br)
                cur['last'] = br[-1]['ts']
                cur['bursts'].append(f'B{b}')
            for r in br:
                if finished(r):
                    done.add(r['key'])
        P(f'-- queue replay for {label} (seed {seed}): {len(inv)} invocations explain {sum(c["rows"] for c in inv)} of {len(rows)} rows; rows not explained: {unexplained}')
        for n, c in enumerate(inv):
            mx = max(c['order'][k] for k in c['seen'])
            holes = [c['queue'][i] for i in range(mx + 1) if c['queue'][i] not in c['seen']]
            P(f'   invocation {n + 1} (bursts {c["bursts"]}): first row {c["first"][5:19]}, last row {c["last"][5:19]}, queue length {len(c["queue"])}, rows {c["rows"]}, '
              f'highest queue index reached {mx}, keys below that index without any row (in flight when it stopped, never logged): '
              f'{len(holes)} {dict(Counter(k.split("|")[1] for k in holes))} {holes if len(holes) <= 12 else ""}')
        if inv:
            c = inv[-1]
            mx = max(c['order'][k] for k in c['seen'])
            tail = set(c['queue'][mx + 1:])
            holes = {c['queue'][i] for i in range(mx + 1) if c['queue'][i] not in c['seen']}
            miss = set()
            for a in arms:
                miss |= planned_keys[a] - {r['key'] for r in raw_rows if r['arm'] == a and finished(r)}
            failed_rows = {r['key'] for r in rows if not finished(r)}
            P(f'   keys without a finished row now: {len(miss)} = never-started tail of the last queue {len(miss & tail)} '
              f'+ in flight at the snapshot {len(miss & holes)} + failed in the last invocation and not redone '
              f'{len((miss & failed_rows) - tail - holes)} + unexplained {len(miss - tail - holes - failed_rows)}')
    reconstruct('main log (direct + harness)', ('direct', 'harness'), 42, lambda r: bool(r.get('parsed_ok')))
    reconstruct('sub log (harness_sub)', ('harness_sub',), 43, lambda r: r.get('process_ok') is not None)
    # is the number of empty cells unusual for a random subset of that size?
    rng2 = random.Random(11)
    for a in ARMS:
        have = {r['key'] for r in ok if r['arm'] == a}
        obs0 = sum(1 for k in planned if cnt[a].get(k, 0) == 0)
        allk = sorted(planned_keys[a])
        sims = []
        for _ in range(2000):
            c = Counter(k.split('|')[0] for k in rng2.sample(allk, len(have)))
            sims.append(sum(1 for (aid, v) in planned if c.get(f'{aid}v{v}', 0) == 0))
        P(f'   {a}: empty paper-version cells {obs0}; a random subset of {len(have)} of {len(allk)} keys gives {np.mean(sims):.1f} on average '
          f'(P(sim >= observed) = {np.mean(np.array(sims) >= obs0):.4f})')
    spent = {f: round(sum(r.get('guard_cny') or 0 for r in raw_rows if r['_file'] == f), 2) for f in sorted({r['_file'] for r in raw_rows})}
    P(f'budget guard spent so far per log (caps: 120 main / 110 sub by default): {spent}')
    KN['guard_spent_cny'] = spent

    # =============================================================== 8 HARNESS BEHAVIOUR
    H('8. HARNESS BEHAVIOUR')
    hr = [r for r in ok_all if r['arm'] == 'harness']
    P(f'harness usable rows {len(hr)}: n_steps > 1: {sum((r.get("n_steps") or 0) > 1 for r in hr)}; '
      f'non-empty other_events: {sum(bool(r.get("other_events")) for r in hr)}; n_steps values {dict(Counter(r.get("n_steps") for r in hr))}')
    if S:
        P(f'   session logs: tool calls in harness runs: {dict(Counter(len(S[r["_i"]]["tools"]) for r in hr if r["_i"] in S))} (key = number of tool calls); '
          f'child sessions started: {sum(len(S[r["_i"]]["children"]) for r in hr if r["_i"] in S)}; '
          f'turns: {dict(Counter(S[r["_i"]]["n_turns"] for r in hr if r["_i"] in S))}')
    KN['harness_n_steps_gt1'] = sum((r.get('n_steps') or 0) > 1 for r in hr)
    KN['harness_other_events'] = sum(bool(r.get('other_events')) for r in hr)
    sr_all = by_arm['harness_sub']
    P(f'harness_sub attempt rows {len(sr_all)}:')
    for f in ('process_ok', 'reminded', 'review_before_report', 'n_children', 'coverage', 'child_model',
              'child_effort', 'child_tool', 'report_via', 'n_turns', 'n_review_outputs'):
        P(f'   {f:22s} all attempts {dict(Counter(str(r.get(f)) for r in sr_all))}')
        P(f'   {"":22s} process_ok=True {dict(Counter(str(r.get(f)) for r in sr_all if r.get("process_ok") is True))}')
    ver = [r for r in latest_all.values() if r['arm'] == 'harness_sub' and r.get('process_ok') is True]
    P(f'   verified runs {len(ver)}: clean {sum(usable(r, "clean") for r in ver)}, reminded '
      f'{sum(bool(r.get("reminded")) for r in ver)}, review_before_report {sum(bool(r.get("review_before_report")) for r in ver)}; '
      f'process failures (process_ok False): {sum(r.get("process_ok") is False for r in sr_all)}; '
      f'technical failures (None): {sum(r.get("process_ok") is None for r in sr_all)}')
    tc_c = Counter(json.dumps(r.get('tool_calls') or {}, sort_keys=True) for r in ver)
    P(f'   tool_calls of verified runs: {dict(tc_c)}')
    extra = [r for r in ver if (r.get('tool_calls') or {}) != {'subagent_fork': 1}]
    P(f'   verified runs whose main agent made a tool call other than the single subagent_fork '
      f'(the prompt forbids it; not part of the "clean" filter): {len(extra)} of {len(ver)}')
    KN['b1_runs_with_extra_tool_calls'] = f'{len(extra)} of {len(ver)}'
    if extra:
        cell_x = cells_from([r for r in ok if not (r['arm'] == 'harness_sub' and (r.get('tool_calls') or {}) != {'subagent_fork': 1})], planned)
        for a, b in PAIRS[1:]:
            P(f'     dropping them: {b:11s} - {a:8s}: {fmt_cmp(compare(cell_x, planned, a, b))}')
    if F:
        agree: Counter = Counter()
        cov_bad, child_rating = [], []
        for r in ver:
            f = F.get(r['_i'])
            if f is None:
                agree['session log missing'] += 1
                continue
            chk = {
                'n_children': f['n_children'] == r.get('n_children'),
                'n_turns': f['n_turns'] == r.get('n_turns'),
                'tool_calls': f['tool_calls'] == (r.get('tool_calls') or {}),
                'report_via': r.get('report_via') in (None, f['report_via']),
                'child_tool': f['child_tool'] == r.get('child_tool'),
                'n_review_outputs': len(f['reviews']) == r.get('n_review_outputs'),
                'review_before_report': f['review_before_report'] == bool(r.get('review_before_report')),
                'coverage': abs(round(f['cov'], 3) - (r.get('coverage') or 0)) < 1e-9,
                'child_done': f['child_done'],
                'report_found': f['i_report'] is not None,
                'result_found': f['result'] is not None,
            }
            for k, v in chk.items():
                agree[(k, v)] += 1
            if f['cov'] < 0.999:
                cov_bad.append((r['key'], round(f['cov'], 3)))
            if f['child_review_in_report'] is not None and f['result'] is not None:
                child_rating.append((r['key'], f['child_review_in_report'].get('rating'), f['result'].get('rating')))
        P('   audit re-derivation of the process facts from the session logs vs logged fields (True = agree / holds):')
        for k in ('n_children', 'n_turns', 'tool_calls', 'report_via', 'child_tool', 'n_review_outputs',
                  'review_before_report', 'coverage', 'child_done', 'report_found', 'result_found'):
            P(f'      {k:22s} True {agree[(k, True)]:4d}  False {agree[(k, False)]:4d}')
        if agree['session log missing']:
            P(f'      session log missing: {agree["session log missing"]}')
        P(f'   runs where the audit-computed coverage < 0.999: {cov_bad or "none"}')
        same = sum(1 for _, c, m in child_rating if c == m)
        P(f'   runs in which the subagent\'s delivered report itself contained a full scored review JSON: '
          f'{len(child_rating)} of {len(ver)}; the main agent\'s final rating equals the subagent\'s rating in {same} of them')
        KN['b1_child_report_contains_scored_review'] = f'{len(child_rating)} of {len(ver)} (main rating == child rating in {same})'
        ct = Counter(json.dumps(Counter(F[r['_i']]['child_tools']), sort_keys=True) for r in ver if r['_i'] in F)
        P(f'   tool calls made by the subagent: {dict(ct)}')
        seqs: Counter = Counter()
        for r in ver:
            if r['_i'] not in S:
                continue
            lab = []
            for x in S[r['_i']]['user_msgs']:
                lab.append('RUNTIME_CONTEXT' if x.startswith('Current runtime context') else
                           'TASK' if x.startswith('You are an expert peer reviewer') else
                           'PROCESS' if x.startswith('Now review the submission above') else
                           'REMINDER' if x.startswith('The required process is not complete') else
                           'CHILD_MESSAGE' if re.match(r'Agent \S+ sent a message', x) else
                           'CHILD_FINISHED' if x.startswith('Background subagent') else 'OTHER')
            seqs[' > '.join(lab)] += 1
        P('   sequence of user-role messages seen by the main agent:')
        for k, v in seqs.most_common():
            P(f'      {v:4d}  {k}')
        wd = Counter(S[r['_i']]['bucket'] for r in ver if r['_i'] in S)
        P(f'   dsh working-directory bucket of verified runs: {dict(wd)}')

    # =============================================================== outputs
    with open(out_dir / 'row_flags.csv', 'w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        w.writerow(['key', 'arm', 'arxiv_id', 'v', 'rep', 'position', 'ts', 'burst', 'peak', 'parsed_ok', 'process_ok',
                    'usable_clean', 'usable_b1all', 'is_deduped_row', 'rating', 'latency_s', 'wall_s',
                    'failed_attempts', 'llm_retries', 'flags', 'log_file', 'log_line'])
        for r in raw_rows:
            w.writerow([r['key'], r['arm'], r['arxiv_id'], r['v'], r['rep'], planned.get((r['arxiv_id'], r['v'])),
                        r['ts'], f"B{r['_burst']}", r.get('peak'), r.get('parsed_ok'), r.get('process_ok'),
                        usable(r, 'clean'), usable(r, 'all'), latest.get(r['key']) is r, r.get('rating'),
                        r.get('latency_s'), None if r.get('_wall') is None else round(r['_wall'], 1),
                        r.get('failed_attempts'), r.get('llm_retries'), ';'.join(sorted(r['_flags'])),
                        r['_file'], r['_line']])
    (out_dir / 'key_numbers.json').write_text(json.dumps(KN, indent=1, default=str), encoding='utf-8')
    (out_dir / 'audit_report.txt').write_text('\n'.join(OUT_LINES) + '\n', encoding='utf-8')
    print(f'\nwritten: {out_dir}/audit_report.txt, row_flags.csv, key_numbers.json')


if __name__ == '__main__':
    main()
