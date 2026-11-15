#!/usr/bin/env python3
"""E1-H audit: is the LEVEL difference between arms real, and where does it come from?

Read-only. Own parsing code (does not import the runner modules, reads no .env, makes no
network request, starts no process).

    python3 level_audit.py --logs <main_log> <sub_log> [--repo <repo root>] [--no-sessions]

Sections printed:
  0  rows, de-duplication, usable counts
  1  what the two requests share / what dsh adds (from the real session logs)
  2  parsing check on every usable row (raw output re-parsed, compared with the log)
  3  the level gap: estimates with CIs, rating distributions, sub-scores, text measures,
     by position / decision / length
  4  per-paper consistency, sign test
  5  human ICLR scores as external reference
  6  within-arm association between answer/thinking length and rating
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import io
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

ALLOWED = [1, 3, 5, 6, 8, 10]
ARMS = ('direct', 'harness', 'harness_sub')
DEFAULT_REPO = '/Users/madili/Desktop/review papaer research'
RNG_SEED = 20261112


# ------------------------------------------------------------------ helpers

def read_jsonl(path):
    rows = []
    for line in Path(path).read_text(encoding='utf-8').split('\n'):
        if line.strip():
            try:
                rows.append(json.loads(line))
            except ValueError:
                continue
    return rows


def usable(r, b1_all=False):
    if not r.get('parsed_ok'):
        return False
    if r.get('arm') != 'harness_sub':
        return True
    if r.get('process_ok') is not True:
        return False
    return b1_all or not (r.get('reminded') or r.get('review_before_report'))


def dedupe(rows, b1_all=False):
    """One row per key; a usable row is never replaced by a later failed one; among
    several usable rows the last in file order wins."""
    best = {}
    for r in rows:
        k = r['key']
        if k not in best:
            best[k] = r
        elif usable(r, b1_all) or not usable(best[k], b1_all):
            best[k] = r
    return best


def strip_fences(s):
    s = s.strip()
    m = re.match(r"^```[a-zA-Z]*\s*\n(.*?)\n?\s*```\s*$", s, re.DOTALL)
    return m.group(1).strip() if m else s


def extract_review(text):
    s = strip_fences(text)
    try:
        return json.loads(s)
    except ValueError:
        i, j = s.find('{'), s.rfind('}')
        if i == -1 or j <= i:
            raise
        return json.loads(s[i:j + 1])


def coerce_rating(value):
    x = float(value)
    if x in ALLOWED:
        return int(x), False
    return min(ALLOWED, key=lambda a: (abs(a - x), a)), True


def top_level_json_objects(text):
    """All top-level JSON objects decodable anywhere in the text (own scanner)."""
    dec = json.JSONDecoder()
    out, i = [], 0
    while True:
        i = text.find('{', i)
        if i == -1:
            break
        try:
            obj, end = dec.raw_decode(text, i)
        except ValueError:
            i += 1
            continue
        if isinstance(obj, dict):
            out.append(obj)
        i = end
    return out


def load_session(repo, session_id):
    paths = glob.glob(os.path.join(repo, '.tools', 'dsh-home', 'sessions', '*', session_id,
                                   'session.v4.jsonl.zstd'))
    if not paths:
        return None
    import zstandard
    try:
        data = zstandard.ZstdDecompressor().stream_reader(
            io.BytesIO(Path(paths[0]).read_bytes())).read().decode('utf-8', 'replace')
    except Exception:
        return None
    recs = []
    for line in data.split('\n'):
        if line.strip():
            try:
                recs.append(json.loads(line))
            except ValueError:
                continue
    return recs


def msg_blocks(data):
    c = data.get('content')
    if c is None:
        c = (data.get('message') or {}).get('content')
    if isinstance(c, str):
        return [{'type': 'text', 'text': c}]
    return [b for b in (c or []) if isinstance(b, dict)]


def boot_ci_cluster(values, clusters, stat=np.mean, n_boot=10000, seed=RNG_SEED):
    """Percentile CI resampling whole clusters (papers)."""
    values = np.asarray(values, dtype=float)
    cl = np.asarray(clusters)
    ids = sorted(set(cl.tolist()))
    groups = [values[cl == c] for c in ids]
    rng = np.random.default_rng(seed)
    est = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, len(groups), len(groups))
        est[b] = stat(np.concatenate([groups[i] for i in pick]))
    return float(np.percentile(est, 2.5)), float(np.percentile(est, 97.5))


def boot_ci(values, n_boot=10000, seed=RNG_SEED):
    v = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), size=(n_boot, len(v)))
    est = v[idx].mean(axis=1)
    return float(np.percentile(est, 2.5)), float(np.percentile(est, 97.5))


def flip_p(values, n_flip=20000, seed=RNG_SEED):
    v = np.asarray(values, dtype=float)
    obs = abs(v.sum())
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(n_flip, len(v)))
    return float(((np.abs((signs * v).sum(axis=1)) >= obs - 1e-12).sum() + 1) / (n_flip + 1))


def fmt_ci(m, ci):
    return f"{m:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}]"


def mean(xs):
    xs = list(xs)
    return float(np.mean(xs)) if xs else float('nan')


# ------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--logs', nargs=2, required=True, metavar=('MAIN_LOG', 'SUB_LOG'))
    ap.add_argument('--repo', default=DEFAULT_REPO)
    ap.add_argument('--no-sessions', action='store_true',
                    help='skip reading dsh session logs (sections that need them are skipped)')
    ap.add_argument('--b1-all', action='store_true')
    ap.add_argument('--dump', default=None, help='write the per-run table here (jsonl)')
    cfg = ap.parse_args()
    repo = cfg.repo
    P = print

    # ---------------- 0. rows
    raw_rows = read_jsonl(cfg.logs[0]) + read_jsonl(cfg.logs[1])
    best = dedupe(raw_rows, cfg.b1_all)
    rows = [r for r in best.values() if usable(r, cfg.b1_all)]
    P('# 0. rows')
    P(f'attempt rows: {len(raw_rows)}; distinct keys: {len(best)}; usable keys: {len(rows)}')
    for a in ARMS:
        n_att = sum(r['arm'] == a for r in raw_rows)
        P(f'  {a}: attempts {n_att}, usable keys {sum(r["arm"] == a for r in rows)}, '
          f'keys with >1 usable attempt '
          f'{sum(v > 1 for v in Counter(r["key"] for r in raw_rows if r["arm"] == a and usable(r, cfg.b1_all)).values())}')

    manifest = read_jsonl(os.path.join(repo, 'data/processed/pilot_manifest.jsonl'))
    man = {m['arxiv_id']: m for m in manifest}
    pos, text_len = {}, {}
    for m in manifest:
        safe = m['arxiv_id'].replace('/', '_')
        vs = sorted(v['v'] for v in m['versions']
                    if os.path.exists(os.path.join(repo, 'data/processed/texts', safe, f"v{v['v']}.md")))
        for i, v in enumerate(vs):
            pos[(m['arxiv_id'], v)] = i
    P(f'planned paper-versions (manifest + text file): {len(pos)} in {len(manifest)} papers')

    # consistency of the paper text between arms
    sha = defaultdict(set)
    tch = defaultdict(lambda: defaultdict(set))
    for r in rows:
        sha[(r['arxiv_id'], r['v'])].add(r['text_sha'])
        tch[(r['arxiv_id'], r['v'])][r['arm']].add(r['task_chars'])
    P(f'paper-versions whose usable rows carry more than one text_sha: '
      f'{sum(len(s) > 1 for s in sha.values())}')
    P(f'paper-versions where direct and harness task_chars differ: '
      f'{sum(1 for d in tch.values() if d.get("direct") and d.get("harness") and d["direct"] != d["harness"])}')
    P(f'truncated usable rows: {sum(bool(r.get("truncated")) for r in rows)}')
    P(f'served_model by arm: '
      f'{dict(Counter((r["arm"], r.get("served_model") or r.get("child_model")) for r in rows))}')
    P(f'cfg_sig by arm: {dict(Counter((r["arm"], r.get("cfg_sig")) for r in rows))}')
    P(f'harness usable rows with n_steps != 1: '
      f'{sum(1 for r in rows if r["arm"] == "harness" and r.get("n_steps") != 1)}; '
      f'with other_events: {sum(1 for r in rows if r["arm"] == "harness" and r.get("other_events"))}; '
      f'with llm_retries > 0: {sum(1 for r in rows if r["arm"] == "harness" and r.get("llm_retries"))}')

    # ---------------- raw outputs: re-parse every usable row
    P('\n# 2. parsing check (every usable row, raw output re-parsed with own code)')
    prompt_raw = Path(os.path.join(repo, 'src/review/prompts/review_v1.md')).read_text(encoding='utf-8')
    parts = re.split(r"(?m)^#\s+(System|User)\s*$", prompt_raw)
    sect = {parts[i]: parts[i + 1].strip() for i in range(1, len(parts) - 1, 2)}

    def task_text_for(r):
        safe = r['arxiv_id'].replace('/', '_')
        b = Path(os.path.join(repo, 'data/processed/texts', safe, f"v{r['v']}.md")).read_bytes()
        paper = b.decode('utf-8', errors='replace')
        if len(paper) > 380000:
            paper = paper[:380000] + '\n[truncated]'
        return hashlib.sha256(b).hexdigest(), sect['System'] + '\n\n' + sect['User'].replace('{PAPER_TEXT}', paper)

    chk = {a: Counter() for a in ARMS}
    problems = []
    for r in rows:
        a = r['arm']
        c = chk[a]
        c['rows'] += 1
        rp = os.path.join(repo, r.get('raw_path') or '')
        if not r.get('raw_path') or not os.path.exists(rp):
            c['raw_missing'] += 1
            problems.append((r['key'], 'raw file missing'))
            continue
        body = Path(rp).read_text(encoding='utf-8', errors='replace')
        answer = None
        if a == 'direct':
            d = json.loads(body)
            answer = d['text']
            r['_answer_chars'] = len(answer)
            if d.get('usage') != r.get('usage'):
                c['raw_usage_differs_from_log'] += 1
        else:
            evs = []
            for line in body.split('\n'):
                if line.strip():
                    try:
                        evs.append(json.loads(line))
                    except ValueError:
                        c['bad_event_lines'] += 1
            finals = [e for e in evs if e.get('type') == 'final']
            texts = [e for e in evs if e.get('type') == 'text']
            sids = [e.get('sessionId') for e in evs if e.get('type') == 'session']
            r['_n_final'] = len(finals)
            r['_n_text_events'] = len(texts)
            r['_stream_sids'] = sids
            if a == 'harness':
                c[f'final_events={len(finals)}'] += 1
                c[f'text_events={len(texts)}'] += 1
                if finals:
                    answer = finals[0].get('text') or ''
                    if len(finals) > 1 and any((f.get('text') or '') != answer for f in finals):
                        c['finals_differ'] += 1
                    if texts and ''.join(t.get('text', '') for t in texts) != answer:
                        c['text_events_differ_from_final'] += 1
                        problems.append((r['key'], 'concatenated text events != final text'))
                if sids and sids[0] != r.get('session_id'):
                    c['session_id_mismatch'] += 1
            else:  # harness_sub: the runner takes the rating from the session log (first
                # review after the subagent's report). In the stream the review is a `text`
                # event; the `final` event may carry a later closing remark instead.
                revs = []
                for t in texts:
                    try:
                        rv = extract_review(t.get('text') or '')
                        float(rv['rating'])
                        revs.append(t.get('text') or '')
                    except Exception:
                        pass
                c[f'text_events_that_parse_as_review={len(revs)}'] += 1
                n_fin_rev = 0
                for f in finals:
                    try:
                        float(extract_review(f.get('text') or '')['rating'])
                        n_fin_rev += 1
                    except Exception:
                        pass
                c[f'final_events_that_parse_as_review={n_fin_rev}'] += 1
                answer = revs[0] if revs else None
                if len(revs) > 1:
                    rts = [extract_review(t)['rating'] for t in revs]
                    c['several_reviews_in_stream'] += 1
                    if len(set(rts)) > 1:
                        c['several_reviews_in_stream_ratings_differ'] += 1
                        problems.append((r['key'], f'{len(revs)} review texts in stream, ratings {rts}'))
                if answer is None:
                    c['no_review_text_event_in_stream(8KiB cap)'] += 1
                    continue
                c[f'tool_call_events={sum(e.get("type") == "tool_call" for e in evs)}'] += 1
                if sids and r.get('session_id') not in sids:
                    c['session_id_mismatch'] += 1
            if answer is not None:
                r['_answer_chars'] = len(answer)
        if answer is None:
            c['no_answer_in_raw'] += 1
            problems.append((r['key'], 'no parsable answer in raw'))
            continue
        try:
            rev = extract_review(answer)
        except Exception as exc:
            c['reparse_failed'] += 1
            problems.append((r['key'], f'reparse failed {exc!r}'))
            continue
        r['_review'] = rev
        s = strip_fences(answer)
        c['answer_is_pure_json'] += int(s.startswith('{') and s.endswith('}'))
        c['answer_fenced'] += int(answer.strip().startswith('```'))
        objs = top_level_json_objects(answer)
        c[f'top_level_json_objects={len(objs)}'] += 1
        n_rating_keys = len(re.findall(r'"rating"\s*:', answer))
        c[f'"rating"_keys_in_answer={n_rating_keys}'] += 1
        try:
            rt, co = coerce_rating(rev['rating'])
        except Exception:
            c['rating_not_numeric'] += 1
            continue
        if rt != r['rating']:
            c['rating_MISMATCH'] += 1
            problems.append((r['key'], f'log rating {r["rating"]} vs raw {rev["rating"]}'))
        else:
            c['rating_matches_log'] += 1
        c['rating_coerced'] += int(co)
        if co:
            problems.append((r['key'], f'rating coerced from {rev["rating"]!r} to {rt}'))
        for f, hi in (('soundness', 4), ('presentation', 4), ('contribution', 4), ('confidence', 5)):
            try:
                x = float(rev[f])
                iv = min(hi, max(1, int(round(x))))
                if iv != r[f]:
                    c[f'{f}_MISMATCH'] += 1
                if iv != x:
                    c[f'{f}_coerced'] += 1
            except Exception:
                c[f'{f}_unparsable'] += 1
        acc = rev.get('accept')
        if isinstance(acc, str):
            c['accept_was_string'] += 1
            acc = acc.strip().lower() in ('true', 'yes', '1')
        if bool(acc) != r['accept']:
            c['accept_MISMATCH'] += 1
        c['log_coerced_flag'] += int(bool(r.get('coerced')))
        extra = set(rev) - {'summary', 'strengths', 'weaknesses', 'questions', 'soundness',
                            'presentation', 'contribution', 'rating', 'confidence', 'accept'}
        c['extra_fields'] += int(bool(extra))
        for f in ('strengths', 'weaknesses', 'questions'):
            v = rev.get(f)
            r[f'_n_{f}'] = len(v) if isinstance(v, list) else None
            r[f'_chars_{f}'] = sum(len(str(x)) for x in v) if isinstance(v, list) else None
        r['_chars_summary'] = len(str(rev.get('summary') or ''))
        r['_accept_consistent'] = (bool(acc) == (rt >= 6))
    for a in ARMS:
        P(f'  {a}: ' + ', '.join(f'{k}={v}' for k, v in sorted(chk[a].items())))
    P(f'  problems listed: {len(problems)}')
    for k, why in problems[:30]:
        P(f'    {k}: {why}')

    # ---------------- 1. sessions: what dsh added in the real runs
    sess_stats = {}
    if not cfg.no_sessions:
        P('\n# 1. what dsh sent in the real runs (from dsh session logs)')
        sys_sha, tool_sets, ctx_sha, hdr_cfg = Counter(), Counter(), Counter(), Counter()
        n_seen = n_missing = n_task_equal = n_task_diff = n_text_eq = n_text_ne = 0
        n_asst = Counter()
        for r in rows:
            if r['arm'] != 'harness':
                continue
            recs = load_session(repo, r.get('session_id') or '')
            if recs is None:
                n_missing += 1
                continue
            n_seen += 1
            user_msgs = [x for x in recs if x.get('type') == 'user/message']
            for x in recs:
                t, d = x.get('type'), x.get('data') or {}
                if t == 'system/message':
                    s = ''.join(b.get('text', '') for b in msg_blocks(d))
                    sys_sha[(hashlib.sha256(s.encode()).hexdigest()[:12], len(s))] += 1
                    sess_stats.setdefault('system_text', s)
                elif t == 'request/header':
                    h = d.get('header') or {}
                    tools = h.get('tools') or []
                    tool_sets[(len(tools), tuple(tl.get('name') for tl in tools),
                               len(json.dumps(tools)))] += 1
                    hdr_cfg[json.dumps(h.get('config'), sort_keys=True)] += 1
            # the task text as dsh recorded it
            task_msgs = [x for x in user_msgs
                         if ((x.get('data') or {}).get('source') or {}).get('kind') == 'user']
            ctx_msgs = [x for x in user_msgs
                        if ((x.get('data') or {}).get('source') or {}).get('kind') == 'runtime-context']
            for x in ctx_msgs:
                s = ''.join(b.get('text', '') for b in msg_blocks(x['data']))
                ctx_sha[(hashlib.sha256(s.encode()).hexdigest()[:12], len(s))] += 1
            if len(task_msgs) == 1:
                got = ''.join(b.get('text', '') for b in msg_blocks(task_msgs[0]['data']))
                sha_now, want = task_text_for(r)
                if sha_now != r['text_sha']:
                    sess_stats['text_file_changed'] = sess_stats.get('text_file_changed', 0) + 1
                if got == want:
                    n_task_equal += 1
                else:
                    n_task_diff += 1
                    if got.strip() == want.strip():
                        sess_stats['task_diff_ws_only'] = sess_stats.get('task_diff_ws_only', 0) + 1
                    else:
                        sess_stats.setdefault('task_diff_examples', []).append(
                            (r['key'], len(got), len(want)))
            asst = [x for x in recs if x.get('type') == 'assistant/message']
            n_asst[len(asst)] += 1
            reasoning = sum(len(b.get('text', '')) for x in asst for b in msg_blocks(x['data'])
                            if b.get('type') == 'reasoning')
            text = ''.join(b.get('text', '') for x in asst for b in msg_blocks(x['data'])
                           if b.get('type') == 'text')
            r['_thinking_chars'] = reasoning
            r['_session_text_chars'] = len(text)
            n_types = Counter(x.get('type') for x in recs)
            r['_tool_calls'] = n_types.get('tool/call', 0)
            # the final event text vs the session's assistant text
            rp = os.path.join(repo, r.get('raw_path') or '')
            fin = None
            for line in Path(rp).read_text(encoding='utf-8', errors='replace').split('\n'):
                if line.strip():
                    try:
                        e = json.loads(line)
                    except ValueError:
                        continue
                    if e.get('type') == 'final':
                        fin = e.get('text') or ''
                        break
            if fin is not None and fin == text:
                n_text_eq += 1
            else:
                n_text_ne += 1
        P(f'  harness usable rows: sessions read {n_seen}, session log missing {n_missing}')
        P(f'  system prompt variants (sha12, chars): {dict(sys_sha)}')
        P(f'  tool catalog variants (n tools, names, json chars): {dict(tool_sets)}')
        P(f'  runtime-context block variants (sha12, chars): {dict(ctx_sha)}')
        P(f'  request config variants: {dict(hdr_cfg)}')
        P(f'  task text recorded by dsh == rules + "\\n\\n" + user template with the paper '
          f'(what the direct arm sends): equal {n_task_equal}, different {n_task_diff} '
          f'(of which whitespace-only at the ends: {sess_stats.get("task_diff_ws_only", 0)}); '
          f'examples {sess_stats.get("task_diff_examples", [])[:5]}')
        P(f'  text files whose sha differs now from the logged text_sha: '
          f'{sess_stats.get("text_file_changed", 0)}')
        P(f'  assistant messages per harness session: {dict(n_asst)}; '
          f'tool calls in any harness session: {sum(r.get("_tool_calls", 0) for r in rows if r["arm"] == "harness")}')
        P(f'  final-event text == session assistant text: {n_text_eq}; different: {n_text_ne}')
        # harness_sub: thinking of the main agent and the child
        n_sub = 0
        subchk = Counter()
        sub_examples = []
        for r in rows:
            if r['arm'] != 'harness_sub':
                continue
            recs = load_session(repo, r.get('session_id') or '')
            if recs is None:
                continue
            n_sub += 1
            asst = [x for x in recs if x.get('type') == 'assistant/message']
            # every assistant text in the MAIN session that parses as a review (full text;
            # the stream's `text` events are capped at 8 KiB and may not parse)
            revs = []
            for i, x in enumerate(recs):
                if x.get('type') != 'assistant/message':
                    continue
                t = ''.join(b.get('text', '') for b in msg_blocks(x['data']) if b.get('type') == 'text')
                if not t.strip():
                    continue
                try:
                    rv = extract_review(t)
                    rt, co = coerce_rating(rv['rating'])
                    for f in ('soundness', 'presentation', 'contribution', 'confidence', 'accept'):
                        rv[f]
                except Exception:
                    continue
                revs.append((i, rt, co, rv, t))
            subchk[f'reviews_in_main_session={len(revs)}'] += 1
            if not revs:
                continue
            rts = [x[1] for x in revs]
            subchk['logged_rating_is_first_review'] += int(rts[0] == r['rating'])
            subchk['logged_rating_is_last_review'] += int(rts[-1] == r['rating'])
            subchk['any_review_coerced'] += int(any(x[2] for x in revs))
            if len(set(rts)) > 1:
                subchk['reviews_disagree_on_rating'] += 1
                sub_examples.append((r['key'], rts, r['rating']))
            if len(revs) > 1:
                same_scores = all(all(x[3].get(f) == revs[0][3].get(f) for f in
                                      ('soundness', 'presentation', 'contribution', 'confidence', 'accept', 'rating'))
                                  for x in revs)
                subchk['several_reviews_all_scores_identical'] += int(same_scores)
                subchk['several_reviews_text_identical'] += int(all(x[4] == revs[0][4] for x in revs))
            r['_last_review_rating'] = rts[-1]
            # use the first review of the session for the text measures if the stream had none
            if r.get('_n_weaknesses') is None:
                rv, t = revs[0][3], revs[0][4]
                r['_answer_chars'] = len(t)
                for f in ('strengths', 'weaknesses', 'questions'):
                    v = rv.get(f)
                    r[f'_n_{f}'] = len(v) if isinstance(v, list) else None
                    r[f'_chars_{f}'] = sum(len(str(z)) for z in v) if isinstance(v, list) else None
                r['_chars_summary'] = len(str(rv.get('summary') or ''))
                r['_accept_consistent'] = (bool(rv.get('accept')) == (revs[0][1] >= 6))
        P('  harness_sub, reviews found in the main session log: '
          + ', '.join(f'{k}={v}' for k, v in sorted(subchk.items())))
        for ex in sub_examples[:20]:
            P(f'    ratings differ between review outputs: {ex[0]} outputs {ex[1]} logged {ex[2]}')
        for r in rows:
            if r['arm'] != 'harness_sub':
                continue
            recs = load_session(repo, r.get('session_id') or '')
            if recs is None:
                continue
            asst = [x for x in recs if x.get('type') == 'assistant/message']
            r['_thinking_chars_main'] = sum(len(b.get('text', '')) for x in asst
                                            for b in msg_blocks(x['data']) if b.get('type') == 'reasoning')
            # the review-producing message = last assistant message that parses as a review
            for x in asst:
                t = ''.join(b.get('text', '') for b in msg_blocks(x['data']) if b.get('type') == 'text')
                try:
                    extract_review(t)['rating']
                except Exception:
                    continue
                r['_thinking_chars_final_msg'] = sum(len(b.get('text', '')) for b in msg_blocks(x['data'])
                                                     if b.get('type') == 'reasoning')
                r['_final_msg_out_tokens'] = (x['data'].get('usage') or {}).get('outputTokens')
                break
        P(f'  harness_sub usable rows with a readable main session: {n_sub}')

    for r in rows:
        if r['arm'] == 'direct':
            r['_thinking_chars'] = r.get('thinking_chars')

    # ---------------- 3. level gap
    cell = defaultdict(lambda: defaultdict(list))
    for r in rows:
        cell[(r['arxiv_id'], r['v'])][r['arm']].append(r)

    def cellmean(rs, f='rating'):
        return mean(float(x[f]) for x in rs)

    P('\n# 3. level gap (b minus a) on paper-versions usable in both arms')
    summary = {}
    for a, b in (('direct', 'harness'), ('direct', 'harness_sub'), ('harness', 'harness_sub')):
        both = {k: d for k, d in cell.items() if d.get(a) and d.get(b)}
        papers = sorted({k[0] for k in both})
        P(f'\n## {b} minus {a}: {len(both)} paper-versions in {len(papers)} papers; '
          f'runs {sum(len(d[a]) for d in both.values())} vs {sum(len(d[b]) for d in both.values())}')
        cd = np.array([cellmean(d[b]) - cellmean(d[a]) for d in both.values()])
        cl = np.array([k[0] for k in both])
        # (i) interim definition: papers with >= 2 common versions, paper mean of cell means
        per_paper = {}
        for p in papers:
            ks = [k for k in both if k[0] == p]
            per_paper[p] = (mean(cellmean(both[k][a]) for k in ks), mean(cellmean(both[k][b]) for k in ks), len(ks))
        d2 = [y - x for x, y, n in per_paper.values() if n >= 2]
        P(f'  (i) interim definition, papers with >=2 common versions (n={len(d2)}): '
          f'mean {a} {mean(x for x, y, n in per_paper.values() if n >= 2):.3f}, '
          f'{b} {mean(y for x, y, n in per_paper.values() if n >= 2):.3f}; diff {fmt_ci(mean(d2), boot_ci(d2))}; sign-flip p={flip_p(d2):.4f}')
        d1 = [y - x for x, y, n in per_paper.values()]
        t1 = stats.ttest_1samp(d1, 0.0)
        P(f'  (ii) all papers with >=1 common version (n={len(d1)}): diff {fmt_ci(mean(d1), boot_ci(d1))}; '
          f'paired t({len(d1) - 1})={t1.statistic:.2f}, p={t1.pvalue:.2e}; sign-flip p={flip_p(d1):.4f}; '
          f'SD of per-paper diffs {np.std(d1, ddof=1):.3f}')
        P(f'  (iii) cell-level mean of (cell mean {b} - cell mean {a}) (n={len(cd)}): '
          f'{fmt_ci(cd.mean(), boot_ci_cluster(cd, cl))} (paper-cluster bootstrap); median cell diff {np.median(cd):+.3f}; '
          f'cells with diff<0 {int((cd < 0).sum())}, =0 {int((cd == 0).sum())}, >0 {int((cd > 0).sum())}')
        full = [k for k, d in both.items() if len(d[a]) == 3 and len(d[b]) == 3]
        if full:
            fd = np.array([cellmean(both[k][b]) - cellmean(both[k][a]) for k in full])
            P(f'  (iv) only cells with 3 replicates in both arms (n={len(full)}): '
              f'{fmt_ci(fd.mean(), boot_ci_cluster(fd, np.array([k[0] for k in full])))}')
        # (v) rep-matched run pairs
        pr, prc = [], []
        for k, d in both.items():
            ra = {x['rep']: x for x in d[a]}
            rb = {x['rep']: x for x in d[b]}
            for rep in set(ra) & set(rb):
                pr.append(rb[rep]['rating'] - ra[rep]['rating'])
                prc.append(k[0])
        pr = np.array(pr, dtype=float)
        P(f'  (v) same-rep run pairs (n={len(pr)}): mean diff {fmt_ci(pr.mean(), boot_ci_cluster(pr, np.array(prc)))}; '
          f'pairs lower/equal/higher: {int((pr < 0).sum())}/{int((pr == 0).sum())}/{int((pr > 0).sum())}; '
          f'pair-difference histogram {dict(sorted(Counter(pr.astype(int).tolist()).items()))}')
        # (vi) run-level OLS with paper-version fixed effects, cluster-robust by paper
        try:
            import pandas as pd
            import statsmodels.formula.api as smf
            recs = [{'rating': x['rating'], 'arm_b': int(arm == b), 'cellid': f'{k[0]}v{k[1]}',
                     'paper': k[0]} for k, d in both.items() for arm in (a, b) for x in d[arm]]
            df = pd.DataFrame(recs)
            df['y'] = df['rating'] - df.groupby('cellid')['rating'].transform('mean')
            df['x'] = df['arm_b'] - df.groupby('cellid')['arm_b'].transform('mean')
            fit = smf.ols('y ~ x - 1', df).fit(cov_type='cluster', cov_kwds={'groups': df['paper']})
            ci = fit.conf_int().loc['x']
            P(f'  (vi) run-level OLS with paper-version fixed effects, SE clustered by paper: '
              f'{fit.params["x"]:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}], p={fit.pvalues["x"]:.2e}')
        except Exception as exc:  # pragma: no cover
            P(f'  (vi) skipped: {exc!r}')
        summary[(a, b)] = dict(both=both, per_paper=per_paper)

        # distributions on the common cells
        P(f'  rating distribution, all usable runs on the common cells (count; cell-weighted share):')
        for arm in (a, b):
            cnt = Counter(x['rating'] for d in both.values() for x in d[arm])
            wsh = Counter()
            for d in both.values():
                for x in d[arm]:
                    wsh[x['rating']] += 1.0 / len(d[arm])
            tot = sum(cnt.values())
            P(f'    {arm:12s} n={tot}: ' + ', '.join(
                f'{rt}: {cnt.get(rt, 0)} ({cnt.get(rt, 0) / tot:.1%}; cw {wsh.get(rt, 0) / len(both):.1%})' for rt in ALLOWED)
              + f' | run mean {mean(x["rating"] for d in both.values() for x in d[arm]):.3f}'
              + f' | cell-weighted mean {mean(cellmean(d[arm]) for d in both.values()):.3f}')

        def cw(arm, fn):
            return mean(mean(fn(x) for x in d[arm]) for d in both.values())

        def cw_diff_ci(fn):
            dd = np.array([mean(fn(x) for x in d[b]) - mean(fn(x) for x in d[a]) for d in both.values()])
            return fmt_ci(dd.mean(), boot_ci_cluster(dd, cl))
        P('  cell-weighted shares / means (a, b, diff with paper-cluster bootstrap CI):')
        for name, fn in (('accept=true', lambda x: float(bool(x['accept']))),
                         ('rating>=6', lambda x: float(x['rating'] >= 6)),
                         ('rating>=8', lambda x: float(x['rating'] >= 8)),
                         ('rating<=3', lambda x: float(x['rating'] <= 3)),
                         ('rating==5', lambda x: float(x['rating'] == 5)),
                         ('rating==6', lambda x: float(x['rating'] == 6)),
                         ('soundness', lambda x: float(x['soundness'])),
                         ('presentation', lambda x: float(x['presentation'])),
                         ('contribution', lambda x: float(x['contribution'])),
                         ('confidence', lambda x: float(x['confidence']))):
            P(f'    {name:13s} {a} {cw(a, fn):.3f} | {b} {cw(b, fn):.3f} | diff {cw_diff_ci(fn)}')
        # measures that may be missing on some rows: restrict to cells where both arms have it
        for name, key in (('n_strengths', '_n_strengths'), ('n_weaknesses', '_n_weaknesses'),
                          ('n_questions', '_n_questions'), ('chars_strengths', '_chars_strengths'),
                          ('chars_weaknesses', '_chars_weaknesses'), ('chars_questions', '_chars_questions'),
                          ('chars_summary', '_chars_summary'), ('answer_chars', '_answer_chars'),
                          ('accept consistent with rating>=6', '_accept_consistent'),
                          ('output tokens', ('usage', 'output')), ('input tokens (uncached)', ('usage', 'input')),
                          ('cache-read tokens', ('usage', 'cache_read')),
                          ('thinking chars', '_thinking_chars'),
                          ('latency s', 'latency_s')):
            def get(x, key=key):
                if isinstance(key, tuple):
                    return (x.get(key[0]) or {}).get(key[1])
                return x.get(key)
            va, vb, dd, cc = [], [], [], []
            for k, d in both.items():
                xa = [float(get(x)) for x in d[a] if get(x) is not None]
                xb = [float(get(x)) for x in d[b] if get(x) is not None]
                if xa and xb:
                    va.append(np.mean(xa)); vb.append(np.mean(xb)); dd.append(np.mean(xb) - np.mean(xa)); cc.append(k[0])
            if len(dd) >= 5:
                P(f'    {name:13s} {a} {np.mean(va):.2f} | {b} {np.mean(vb):.2f} | ratio {np.mean(vb) / np.mean(va):.2f} | '
                  f'diff {fmt_ci(np.mean(dd), boot_ci_cluster(np.array(dd), np.array(cc)))} (cells {len(dd)})')
        if b == 'harness_sub':
            for name, key in (('main-agent output tokens', ('usage_main', 'output')),
                              ('final-message output tokens', '_final_msg_out_tokens'),
                              ('final-message thinking chars', '_thinking_chars_final_msg')):
                vals = [float((x.get(key[0]) or {}).get(key[1]) if isinstance(key, tuple) else x.get(key))
                        for d in both.values() for x in d[b]
                        if ((x.get(key[0]) or {}).get(key[1]) if isinstance(key, tuple) else x.get(key)) is not None]
                if vals:
                    P(f'    {b} {name}: mean {np.mean(vals):.0f}, median {np.median(vals):.0f} (runs {len(vals)})')

        # where in the scale does the mass move? split-replicate baseline (no regression to the mean)
        if a == 'direct':
            P('  by baseline level (classified by ONE direct replicate, outcome = the OTHER direct replicates '
              'vs the b arm; cells with >=2 direct replicates):')
            byb = defaultdict(lambda: [[], [], []])
            for k, d in both.items():
                ra = sorted(d[a], key=lambda x: x['rep'])
                if len(ra) < 2:
                    continue
                base = ra[0]['rating']
                rest = mean(x['rating'] for x in ra[1:])
                byb[base][0].append(rest)
                byb[base][1].append(cellmean(d[b]))
                byb[base][2].append(k[0])
            for base in sorted(byb):
                xa, xb, cc = byb[base]
                dd = np.array(xb) - np.array(xa)
                ci = boot_ci_cluster(dd, np.array(cc)) if len(set(cc)) >= 3 else (float('nan'), float('nan'))
                P(f'    direct rep-0 rating {base}: cells {len(dd)}, other direct reps {np.mean(xa):.2f}, '
                  f'{b} {np.mean(xb):.2f}, diff {fmt_ci(dd.mean(), ci)}')
            # cross-tab of cell modal ratings (descriptive, subject to regression to the mean)
            xt = Counter()
            for d in both.values():
                ma = sorted(Counter(x['rating'] for x in d[a]).items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
                mb = sorted(Counter(x['rating'] for x in d[b]).items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
                xt[(ma, mb)] += 1
            P(f'  cross-tab of cell modal rating ({a} rows -> {b} columns {ALLOWED}):')
            for ra_ in ALLOWED:
                if any(xt.get((ra_, rb_), 0) for rb_ in ALLOWED):
                    P(f'    {a}={ra_}: ' + ' '.join(f'{xt.get((ra_, rb_), 0):3d}' for rb_ in ALLOWED))

        # net flow between rating categories: within a cell the two arms are independent
        # draws, so the cell's joint distribution is the product of its two marginals;
        # averaged over cells (each cell weight 1). net(i->j) = P(a=i,b=j) - P(a=j,b=i).
        joint = defaultdict(float)
        for d in both.values():
            pa = Counter(x['rating'] for x in d[a])
            pb = Counter(x['rating'] for x in d[b])
            for i, ni in pa.items():
                for j, nj in pb.items():
                    joint[(i, j)] += (ni / len(d[a])) * (nj / len(d[b])) / len(both)
        P(f'  cell-weighted joint distribution P({a}=row, {b}=col) in %, columns {ALLOWED}:')
        for i in ALLOWED:
            if any(joint.get((i, j), 0) for j in ALLOWED):
                P(f'    {a}={i:2d}: ' + ' '.join(f'{100 * joint.get((i, j), 0):5.1f}' for j in ALLOWED))
        P(f'  P(same rating) {sum(v for (i, j), v in joint.items() if i == j):.3f}, '
          f'P({b} lower) {sum(v for (i, j), v in joint.items() if j < i):.3f}, '
          f'P({b} higher) {sum(v for (i, j), v in joint.items() if j > i):.3f}')
        flows = []
        tot_gap = sum((j - i) * v for (i, j), v in joint.items())
        for ii, i in enumerate(ALLOWED):
            for j in ALLOWED[:ii]:
                net = joint.get((i, j), 0) - joint.get((j, i), 0)
                if abs(net) > 1e-12:
                    flows.append((net * (j - i), i, j, net))
        P(f'  net downward flows (share of cell-pairs; contribution to the mean gap {tot_gap:+.3f}):')
        for contrib, i, j, net in sorted(flows):
            P(f'    {i} -> {j}: net {100 * net:+.1f}% of pairs, contributes {contrib:+.3f} points '
              f'({contrib / tot_gap:.0%} of the gap)')
        idx = {rt: n for n, rt in enumerate(ALLOWED)}
        od = np.array([mean(idx[x['rating']] for x in d[b]) - mean(idx[x['rating']] for x in d[a]) for d in both.values()])
        P(f'  gap in scale STEPS (category index 1<3<5<6<8<10): {fmt_ci(od.mean(), boot_ci_cluster(od, cl))}')

        # time halves and identical answers
        all_ts = sorted(x['ts'] for d in both.values() for arm in (a, b) for x in d[arm])
        mid = all_ts[len(all_ts) // 2]
        for label, pred in ((f'runs before {mid}', lambda x: x['ts'] < mid), (f'runs from {mid}', lambda x: x['ts'] >= mid)):
            ks = [k for k, d in both.items() if any(pred(x) for x in d[a]) and any(pred(x) for x in d[b])]
            if len(ks) >= 5:
                dd = np.array([mean(x['rating'] for x in both[k][b] if pred(x)) - mean(x['rating'] for x in both[k][a] if pred(x)) for k in ks])
                P(f'  {label}: cells {len(ks)}, diff {fmt_ci(dd.mean(), boot_ci_cluster(dd, np.array([k[0] for k in ks])))}')
        # output-token ratio: is the gap larger where the b arm wrote/thought more?
        tr, gd, cc_ = [], [], []
        for k, d in both.items():
            oa = mean((x.get('usage') or {}).get('output', 0) for x in d[a])
            ob = mean((x.get('usage') or {}).get('output', 0) for x in d[b])
            if oa > 0 and ob > 0:
                tr.append(math.log(ob / oa)); gd.append(cellmean(d[b]) - cellmean(d[a])); cc_.append(k[0])
        tr, gd, cc_ = np.array(tr), np.array(gd), np.array(cc_)
        rho, pv = stats.spearmanr(tr, gd)
        P(f'  cell-level Spearman between log(output tokens {b}/{a}) and the rating gap: {rho:+.3f} (p={pv:.3f}, cells {len(tr)}, not cluster-adjusted)')
        q = np.percentile(tr, [33.333, 66.667])
        for label, m in (('lowest third of token ratio', tr <= q[0]), ('middle third', (tr > q[0]) & (tr <= q[1])), ('highest third', tr > q[1])):
            P(f'    {label}: cells {int(m.sum())}, mean token ratio {math.exp(tr[m].mean()):.2f}, gap {fmt_ci(gd[m].mean(), boot_ci_cluster(gd[m], cc_[m]))}')

        # by position, decision, length
        P('  cell-level gap by subgroup (paper-cluster bootstrap CI):')

        def sub(label, pred):
            ks = [k for k in both if pred(k)]
            if len(ks) < 3:
                P(f'    {label}: n={len(ks)} (too few)')
                return
            dd = np.array([cellmean(both[k][b]) - cellmean(both[k][a]) for k in ks])
            cc = np.array([k[0] for k in ks])
            ci = boot_ci_cluster(dd, cc) if len(set(cc.tolist())) >= 3 else (float('nan'), float('nan'))
            P(f'    {label}: cells {len(ks)}, papers {len(set(cc.tolist()))}, '
              f'{a} {mean(cellmean(both[k][a]) for k in ks):.3f}, {b} {mean(cellmean(both[k][b]) for k in ks):.3f}, '
              f'diff {fmt_ci(dd.mean(), ci)}')
        sub('position 0', lambda k: pos[k] == 0)
        sub('position 1', lambda k: pos[k] == 1)
        sub('position 2+', lambda k: pos[k] >= 2)
        sub('decision accepted', lambda k: man[k[0]]['decision'] == 'accepted')
        sub('decision rejected', lambda k: man[k[0]]['decision'] == 'rejected')
        hs = sorted(man[p]['mean_score'] for p in papers)
        med = float(np.median(hs))
        sub(f'human mean_score <= {med:.2f}', lambda k: man[k[0]]['mean_score'] <= med)
        sub(f'human mean_score > {med:.2f}', lambda k: man[k[0]]['mean_score'] > med)
        tc = {k: next(iter(tch[k][a])) if tch[k].get(a) else next(iter(tch[k][b])) for k in both}
        q1, q2 = np.percentile(list(tc.values()), [33.333, 66.667])
        sub(f'task_chars <= {q1:.0f} (short third)', lambda k: tc[k] <= q1)
        sub(f'task_chars in ({q1:.0f}, {q2:.0f}]', lambda k: q1 < tc[k] <= q2)
        sub(f'task_chars > {q2:.0f} (long third)', lambda k: tc[k] > q2)
        # paper-level correlation of the gap with length / human score / position slope
        pl = [(mean(tc[k] for k in both if k[0] == p), per_paper[p][1] - per_paper[p][0],
               man[p]['mean_score'], per_paper[p][0]) for p in papers]
        pl = np.array(pl)
        rho, pv = stats.spearmanr(pl[:, 0], pl[:, 1])
        P(f'    Spearman over papers: gap vs mean task_chars rho={rho:+.3f} (p={pv:.3f}); '
          f'gap vs human mean_score rho={stats.spearmanr(pl[:, 2], pl[:, 1])[0]:+.3f} '
          f'(p={stats.spearmanr(pl[:, 2], pl[:, 1])[1]:.3f})')
        # time: peak flag and hour
        hours = defaultdict(lambda: defaultdict(list))
        for k, d in both.items():
            for arm in (a, b):
                for x in d[arm]:
                    hours[x['ts'][:13]][arm].append(x['rating'])
        P(f'    run timestamps: {a} {min(x["ts"] for d in both.values() for x in d[a])} .. '
          f'{max(x["ts"] for d in both.values() for x in d[a])}; {b} '
          f'{min(x["ts"] for d in both.values() for x in d[b])} .. {max(x["ts"] for d in both.values() for x in d[b])}')
        P(f'    peak-price share of runs: {a} {mean(bool(x["peak"]) for d in both.values() for x in d[a]):.2f}, '
          f'{b} {mean(bool(x["peak"]) for d in both.values() for x in d[b]):.2f}')
        for pk in (False, True):
            ks = [k for k, d in both.items() if any(bool(x['peak']) == pk for x in d[a]) and any(bool(x['peak']) == pk for x in d[b])]
            if len(ks) >= 5:
                dd = np.array([mean(x['rating'] for x in both[k][b] if bool(x['peak']) == pk)
                               - mean(x['rating'] for x in both[k][a] if bool(x['peak']) == pk) for k in ks])
                P(f'    only runs with peak={pk} (cells {len(ks)}): diff '
                  f'{fmt_ci(dd.mean(), boot_ci_cluster(dd, np.array([k[0] for k in ks])))}')

        # ---------------- 4. per-paper consistency
        # rounding removes floating-point residue so exact ties stay ties
        lower = sum(round(y - x, 9) < 0 for x, y, n in per_paper.values())
        higher = sum(round(y - x, 9) > 0 for x, y, n in per_paper.values())
        ties = sum(round(y - x, 9) == 0 for x, y, n in per_paper.values())
        bt = stats.binomtest(lower, lower + higher, 0.5) if lower + higher else None
        P(f'  # 4. per-paper consistency: {b} paper mean below {a} in {lower} papers, above in {higher}, '
          f'equal in {ties} (of {len(per_paper)}); exact sign test p={bt.pvalue:.2e}' if bt else '')
        w = stats.wilcoxon(d1) if len(d1) > 5 else None
        if w is not None:
            P(f'     Wilcoxon signed-rank p={w.pvalue:.2e}; per-paper diff quartiles '
              f'{np.percentile(d1, [0, 25, 50, 75, 100]).round(3).tolist()}')
        worst = sorted(((y - x, p) for p, (x, y, n) in per_paper.items()))
        P('     most negative papers: ' + ', '.join(f'{p} {d:+.2f}' for d, p in worst[:5])
          + ' | most positive: ' + ', '.join(f'{p} {d:+.2f}' for d, p in worst[-5:]))
        # leave-one-paper-out range
        loo = [mean(d1[:i] + d1[i + 1:]) for i in range(len(d1))]
        P(f'     leave-one-paper-out range of the mean per-paper gap: [{min(loo):+.3f}, {max(loo):+.3f}]')

    # ---------------- extra robustness checks
    P('\n# 3b. other checks')
    P('  all usable rows per arm (NOT restricted to common paper-versions):')
    for arm in ARMS:
        rs = [x for d in cell.values() for x in d.get(arm, [])]
        cnt = Counter(x['rating'] for x in rs)
        P(f'    {arm:12s} n={len(rs)}: ' + ', '.join(f'{rt}: {cnt.get(rt, 0)}' for rt in ALLOWED)
          + f' | mean {mean(x["rating"] for x in rs):.3f} | accept {mean(bool(x["accept"]) for x in rs):.3f}')
    for a, b in (('direct', 'harness'), ('direct', 'harness_sub')):
        both = {k: d for k, d in cell.items() if d.get(a) and d.get(b)}
        P(f'  floor: cells where EVERY replicate is 3 (rating 1 is never used), common cells of {a}/{b}: '
          + ', '.join(f'{arm} {sum(all(x["rating"] == 3 for x in d[arm]) for d in both.values())} of {len(both)}'
                      for arm in (a, b)))
    one_call = lambda x: x.get('tool_calls') == {'subagent_fork': 1}  # noqa: E731
    subs = [x for d in cell.values() for x in d.get('harness_sub', [])]
    P(f'  harness_sub usable runs whose main agent made tool calls beyond one subagent_fork: '
      f'{sum(not one_call(x) for x in subs)} of {len(subs)}: '
      f'{dict(Counter(json.dumps(x.get("tool_calls"), sort_keys=True) for x in subs if not one_call(x)))}')
    both = {k: d for k, d in cell.items() if d.get('direct') and [x for x in d.get('harness_sub', []) if one_call(x)]}
    if both:
        dd = np.array([mean(x['rating'] for x in d['harness_sub'] if one_call(x)) - cellmean(d['direct']) for d in both.values()])
        P(f'  harness_sub - direct using only harness_sub runs with exactly one tool call: '
          f'{fmt_ci(dd.mean(), boot_ci_cluster(dd, np.array([k[0] for k in both])))} (cells {len(dd)})')
    for arm in ARMS:
        dup = tot = 0
        for d in cell.values():
            rs = d.get(arm, [])
            if len(rs) >= 2:
                tot += 1
                key = [(x['rating'], x['soundness'], x['presentation'], x['contribution'], x['confidence'],
                        x.get('_answer_chars'), x.get('_chars_weaknesses')) for x in rs]
                dup += int(len(set(key)) < len(key))
        P(f'  {arm}: cells with >=2 replicates {tot}; cells where two replicates have identical scores AND identical '
          f'answer length (possible repeated output): {dup}')
    # direct arm: does the prompt-cache state go with the rating? (within paper-version)
    for arm in ('direct', 'harness'):
        dd, cc = [], []
        for k, d in cell.items():
            rs = d.get(arm, [])
            hit = [x['rating'] for x in rs if (x.get('usage') or {}).get('cache_read', 0) > 8000]
            miss = [x['rating'] for x in rs if (x.get('usage') or {}).get('cache_read', 0) <= 8000]
            if hit and miss:
                dd.append(np.mean(hit) - np.mean(miss)); cc.append(k[0])
        n_hit = sum((x.get('usage') or {}).get('cache_read', 0) > 8000 for d in cell.values() for x in d.get(arm, []))
        n_all = sum(len(d.get(arm, [])) for d in cell.values())
        if len(dd) >= 5:
            P(f'  {arm}: runs with the paper served from the prompt cache (cache_read > 8000 tokens) {n_hit} of {n_all}; '
              f'within paper-version, rating(cache hit) - rating(cache miss): '
              f'{fmt_ci(np.mean(dd), boot_ci_cluster(np.array(dd), np.array(cc)))} (cells {len(dd)})')
        else:
            P(f'  {arm}: cache hit runs {n_hit} of {n_all}; too few cells with both states ({len(dd)})')
    # failed attempts / retries
    for arm in ('direct', 'harness'):
        rs = [x for d in cell.values() for x in d.get(arm, [])]
        P(f'  {arm}: usable runs with failed_attempts > 0: {sum(bool(x.get("failed_attempts")) for x in rs)}, '
          f'with llm_retries > 0: {sum(bool(x.get("llm_retries")) for x in rs)}')
    hr = [x for d in cell.values() for x in d.get('harness', [])]
    if any(x.get('llm_retries') for x in hr):
        dd = []
        for k, d in cell.items():
            rs = d.get('harness', [])
            a_ = [x['rating'] for x in rs if x.get('llm_retries')]
            b_ = [x['rating'] for x in rs if not x.get('llm_retries')]
            if a_ and b_:
                dd.append(np.mean(a_) - np.mean(b_))
        P(f'  harness: rating(runs with an internal retry) - rating(without), within paper-version: '
          f'mean {np.mean(dd) if dd else float("nan"):+.3f} over {len(dd)} cells')
    # the gap when the harness runs with an internal dsh retry are dropped
    both = {k: d for k, d in cell.items() if d.get('direct') and [x for x in d.get('harness', []) if not x.get('llm_retries')]}
    dd = np.array([mean(x['rating'] for x in d['harness'] if not x.get('llm_retries')) - cellmean(d['direct']) for d in both.values()])
    P(f'  harness - direct without the harness runs that had an internal retry: '
      f'{fmt_ci(dd.mean(), boot_ci_cluster(dd, np.array([k[0] for k in both])))} (cells {len(dd)})')
    # direct vs harness_sub using only direct runs made after the first harness_sub run
    sub_ts = [x['ts'] for d in cell.values() for x in d.get('harness_sub', [])]
    if sub_ts:
        t0 = min(sub_ts)
        both = {k: d for k, d in cell.items() if d.get('harness_sub') and [x for x in d.get('direct', []) if x['ts'] >= t0]}
        dd = np.array([cellmean(d['harness_sub']) - mean(x['rating'] for x in d['direct'] if x['ts'] >= t0) for d in both.values()])
        P(f'  harness_sub - direct using only direct runs made after the first harness_sub run ({t0}): '
          f'{fmt_ci(dd.mean(), boot_ci_cluster(dd, np.array([k[0] for k in both])))} (cells {len(dd)})')

    # ---------------- 5. human reference
    P('\n# 5. human ICLR scores as external reference (descriptive)')
    tri = {k for k, d in cell.items() if all(d.get(arm) for arm in ARMS)}
    P(f'paper-versions usable in all three arms: {len(tri)} in {len({k[0] for k in tri})} papers')
    for scope in ('all versions', 'first version (position 0)'):
        ks_all = [k for k in tri if scope == 'all versions' or pos[k] == 0]
        ps = sorted({k[0] for k in ks_all})
        hum = np.array([man[p]['mean_score'] for p in ps])
        out = []
        for arm in ARMS:
            llm = np.array([mean(cellmean(cell[k][arm]) for k in ks_all if k[0] == p) for p in ps])
            out.append(f'{arm} LLM-human {fmt_ci((llm - hum).mean(), boot_ci(llm - hum))}, Spearman {stats.spearmanr(llm, hum)[0]:+.3f}')
        P(f'  identical paper-versions in all arms, {scope} ({len(ps)} papers): ' + ' | '.join(out))
    for scope in ('all versions', 'first version (position 0)'):
        P(f'## {scope}')
        pm = {}
        for arm in ARMS:
            d = {}
            for p in man:
                ks = [k for k in cell if k[0] == p and cell[k].get(arm)
                      and (scope == 'all versions' or pos[k] == 0)]
                if ks:
                    d[p] = mean(cellmean(cell[k][arm]) for k in ks)
            pm[arm] = d
        for arm in ARMS:
            ps = sorted(pm[arm])
            llm = np.array([pm[arm][p] for p in ps])
            hum = np.array([man[p]['mean_score'] for p in ps])
            diff = llm - hum
            rho, pv = stats.spearmanr(llm, hum)
            pr_, ppv = stats.pearsonr(llm, hum)
            P(f'  {arm:12s} papers {len(ps)}: LLM mean {llm.mean():.3f}, human mean {hum.mean():.3f}, '
              f'mean (LLM - human) {fmt_ci(diff.mean(), boot_ci(diff))}, mean |LLM - human| {np.abs(diff).mean():.3f}, '
              f'RMSE {math.sqrt((diff ** 2).mean()):.3f}; Spearman {rho:+.3f} (p={pv:.3f}), Pearson {pr_:+.3f}; '
              f'SD of LLM paper means {llm.std(ddof=1):.3f} (human {hum.std(ddof=1):.3f})')
        common = sorted(set.intersection(*(set(pm[arm]) for arm in ARMS)))
        P(f'  on the {len(common)} papers present in all three arms:')
        hum = np.array([man[p]['mean_score'] for p in common])
        L = {arm: np.array([pm[arm][p] for p in common]) for arm in ARMS}
        for arm in ARMS:
            diff = L[arm] - hum
            P(f'    {arm:12s} mean (LLM - human) {fmt_ci(diff.mean(), boot_ci(diff))}, mean |.| {np.abs(diff).mean():.3f}, '
              f'Spearman {stats.spearmanr(L[arm], hum)[0]:+.3f}; '
              f'AUC accepted-vs-rejected {auc([L[arm][i] for i, p in enumerate(common) if man[p]["decision"] == "accepted"], [L[arm][i] for i, p in enumerate(common) if man[p]["decision"] == "rejected"]):.3f}')
        rng = np.random.default_rng(RNG_SEED)
        for x, y in (('direct', 'harness'), ('direct', 'harness_sub'), ('harness', 'harness_sub')):
            est = []
            n = len(common)
            for _ in range(5000):
                idx = rng.integers(0, n, n)
                if len(set(hum[idx].tolist())) < 3:
                    continue
                est.append(stats.spearmanr(L[y][idx], hum[idx])[0] - stats.spearmanr(L[x][idx], hum[idx])[0])
            est = np.array(est)
            est = est[~np.isnan(est)]
            P(f'    Spearman({y}) - Spearman({x}) = '
              f'{stats.spearmanr(L[y], hum)[0] - stats.spearmanr(L[x], hum)[0]:+.3f} '
              f'[{np.percentile(est, 2.5):+.3f}, {np.percentile(est, 97.5):+.3f}] (paper bootstrap)')
        P(f'    human mean_score by decision on these papers: accepted '
          f'{mean(man[p]["mean_score"] for p in common if man[p]["decision"] == "accepted"):.2f} '
          f'(n={sum(man[p]["decision"] == "accepted" for p in common)}), rejected '
          f'{mean(man[p]["mean_score"] for p in common if man[p]["decision"] == "rejected"):.2f} '
          f'(n={sum(man[p]["decision"] == "rejected" for p in common)})')
        for arm in ARMS:
            P(f'    {arm:12s} LLM paper mean by decision: accepted '
              f'{mean(pm[arm][p] for p in common if man[p]["decision"] == "accepted"):.2f}, rejected '
              f'{mean(pm[arm][p] for p in common if man[p]["decision"] == "rejected"):.2f}')

    # ---------------- 6. within-arm: length vs rating
    P('\n# 6. within-arm association of answer / thinking length with rating')
    try:
        import pandas as pd
        import statsmodels.formula.api as smf
        for arm in ARMS:
            recs = []
            for k, d in cell.items():
                for x in d.get(arm, []):
                    recs.append({'rating': float(x['rating']), 'out': float((x.get('usage') or {}).get('output') or np.nan),
                                 'think': float(x['_thinking_chars']) if x.get('_thinking_chars') is not None else np.nan,
                                 'ans': float(x['_answer_chars']) if x.get('_answer_chars') is not None else np.nan,
                                 'wk': float(x['_chars_weaknesses']) if x.get('_chars_weaknesses') is not None else np.nan,
                                 'cellid': f'{k[0]}v{k[1]}', 'paper': k[0], 'task_chars': x['task_chars']})
            df = pd.DataFrame(recs)
            P(f'## {arm}: runs {len(df)}; output tokens mean {df["out"].mean():.0f}, median {df["out"].median():.0f}, '
              f'p10 {df["out"].quantile(.1):.0f}, p90 {df["out"].quantile(.9):.0f}')
            for col, label in (('out', 'output tokens'), ('think', 'thinking chars'), ('ans', 'answer chars'),
                               ('wk', 'weaknesses chars')):
                sub_ = df.dropna(subset=[col])
                if len(sub_) < 20:
                    P(f'  {label}: not available')
                    continue
                rho, pv = stats.spearmanr(sub_[col], sub_['rating'])
                # between papers
                g = sub_.groupby('paper').agg(x=(col, 'mean'), y=('rating', 'mean'))
                rb, pb = stats.spearmanr(g['x'], g['y'])
                # within paper-version: demean log length and rating in cells with >= 2 runs
                s2 = sub_[sub_.groupby('cellid')['rating'].transform('size') >= 2].copy()
                s2['lx'] = np.log(s2[col].clip(lower=1))
                s2['lxd'] = s2['lx'] - s2.groupby('cellid')['lx'].transform('mean')
                s2['yd'] = s2['rating'] - s2.groupby('cellid')['rating'].transform('mean')
                fit = smf.ols('yd ~ lxd - 1', s2).fit(cov_type='cluster', cov_kwds={'groups': s2['paper']})
                ci = fit.conf_int().loc['lxd']
                rw, pw = stats.spearmanr(s2['lxd'], s2['yd'])
                P(f'  {label}: run-level Spearman with rating {rho:+.3f} (p={pv:.3g}); between papers (paper means, n={len(g)}) '
                  f'{rb:+.3f} (p={pb:.3g}); within paper-version (n={len(s2)} runs in {s2["cellid"].nunique()} cells) Spearman {rw:+.3f} '
                  f'(p={pw:.3g}), OLS rating per log-unit of length {fit.params["lxd"]:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}] '
                  f'(i.e. per doubling {fit.params["lxd"] * math.log(2):+.3f})')
            # quartiles of output tokens within arm
            df['q'] = pd.qcut(df['out'], 4, labels=False, duplicates='drop')
            P('  mean rating by within-arm quartile of output tokens: '
              + ', '.join(f'Q{int(q) + 1} {g_["rating"].mean():.2f} (out {g_["out"].mean():.0f}, n={len(g_)})'
                          for q, g_ in df.groupby('q')))
        # does adjusting for output length change the arm gap? (post-treatment variable: descriptive only)
        for a, b in (('direct', 'harness'),):
            both = summary[(a, b)]['both']
            for col, key in (('log output tokens', lambda x: (x.get('usage') or {}).get('output')),
                             ('log thinking chars', lambda x: x.get('_thinking_chars'))):
                recs = [{'rating': float(x['rating']), 'arm_b': int(arm == b), 'cellid': f'{k[0]}v{k[1]}', 'paper': k[0],
                         'lx': math.log(max(float(key(x)), 1.0))}
                        for k, d in both.items() for arm in (a, b) for x in d[arm] if key(x) is not None]
                df = pd.DataFrame(recs)
                keep = df.groupby('cellid')['arm_b'].transform('nunique') == 2
                df = df[keep].copy()
                for c_ in ('rating', 'arm_b', 'lx'):
                    df[c_ + '_d'] = df[c_] - df.groupby('cellid')[c_].transform('mean')
                f0 = smf.ols('rating_d ~ arm_b_d - 1', df).fit(cov_type='cluster', cov_kwds={'groups': df['paper']})
                f1 = smf.ols('rating_d ~ arm_b_d + lx_d - 1', df).fit(cov_type='cluster', cov_kwds={'groups': df['paper']})
                c1 = f1.conf_int()
                P(f'  {b} - {a}, cell fixed effects (runs {len(df)}): gap {f0.params["arm_b_d"]:+.3f}; adding {col}: gap '
                  f'{f1.params["arm_b_d"]:+.3f} [{c1.loc["arm_b_d", 0]:+.3f}, {c1.loc["arm_b_d", 1]:+.3f}], '
                  f'length coefficient {f1.params["lx_d"]:+.3f} [{c1.loc["lx_d", 0]:+.3f}, {c1.loc["lx_d", 1]:+.3f}] '
                  f'(descriptive; length is itself an outcome of the arm)')
    except Exception as exc:  # pragma: no cover
        P(f'  section 6 failed: {exc!r}')

    if cfg.dump:
        with open(cfg.dump, 'w', encoding='utf-8') as fh:
            for r in rows:
                o = {k: v for k, v in r.items() if k != '_review' and k != '_stream_sids'}
                fh.write(json.dumps(o) + '\n')
        P(f'\nper-run table written to {cfg.dump}')


def auc(pos_vals, neg_vals):
    """P(accepted paper scored above rejected paper), ties count 1/2."""
    if not pos_vals or not neg_vals:
        return float('nan')
    w = 0.0
    for x in pos_vals:
        for y in neg_vals:
            w += 1.0 if x > y else 0.5 if x == y else 0.0
    return w / (len(pos_vals) * len(neg_vals))


if __name__ == '__main__':
    main()
