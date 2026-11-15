"""E1-H arm B1: the harness must complete one subagent review before it scores.

Same paper text, review criteria, model, lockdown patch and environment as the 'harness'
arm of run_e1_harness.py (B0). The difference is a required process, run as two turns of
one dsh session:

  turn 1  the review rules and the paper, plus "do not review yet, reply READY"
  turn 2  call `subagent_fork` once (the forked child is seeded with turn 1, so it reads
          the identical paper text without the main agent re-typing it), wait for its
          report, then output the final review JSON

Why a fork and two turns: a spawned `subagent` does not see the conversation, and making
the main agent copy the paper into the child's prompt failed in a smoke test on
realistic papers (58k chars: 59% copied; 130k chars: 68% copied or invalid tool JSON,
about CNY 1 and 11 minutes per run). A fork inherits only COMPLETED turns, hence turn 1.

DeepSeek's Anthropic-format API cannot force a tool call in thinking mode (tool_choice
any is treated as auto; a named tool returns 400), so the process is stated in the
prompt and VERIFIED BY THIS PROGRAM from dsh's own session logs:

  1. the main agent started a subagent that provably had >= 90% of the paper text
     (fork: in its seeded history; spawn: in its prompt)
  2. that child ran to completion and its report was delivered to the main agent
  3. a valid review JSON was produced AFTER the report arrived (the first one counts)

A run that misses a step gets ONE reminder on the same session. If the process is still
incomplete it is logged as a process failure (process_ok = false): its rating is not
used, it is not re-run, and the failure rate is reported. Technical failures (API errors,
timeouts) leave process_ok = null and are retried on a later invocation.

Token usage and cost are the sum over the main agent and every child session (a forked
child's log starts with a copy of the parent's history, which is not counted twice).

Run from the repo root:
    python3 src/review/run_e1_harness_sub.py --sample-papers 2 --reps 1 --log ... --out-dir ...   # smoke
    python3 src/review/run_e1_harness_sub.py --wait-offpeak
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import io
import json
import os
import random
import re
import signal
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, 'src/collect')
sys.path.insert(0, 'src/review')
from util import read_jsonl, append_jsonl  # noqa: E402
from run_llm_review import (  # noqa: E402
    ALLOWED_RATINGS, load_api_key, load_prompt, select_papers, validate_review,
)
import run_e1_harness as R  # noqa: E402

ARM = 'harness_sub'
DELEGATION_TOOLS = ('subagent_fork', 'subagent')
PRIMER = (
    "\n\nDo not review yet. This message only delivers the submission and the review "
    "criteria. Reply with the single word READY and nothing else; how to proceed is "
    "explained in the next message."
)
PROCESS = (
    "Now review the submission above, following this required process:\n"
    "1. Call the `subagent_fork` tool exactly once to obtain one independent review. The "
    "forked subagent already sees the submission and the review criteria, so its prompt "
    "only needs the task: ask it to check the main contributions and whether the evidence "
    "supports the conclusions, and to report well-supported strengths and limitations "
    "with the sections they refer to.\n"
    "2. End your turn and wait. The subagent's report will be delivered to you.\n"
    "3. When the report arrives, check its points against the paper and output the final "
    "review JSON object described in the first message, and nothing else.\n"
    "Do not write your own review before the report arrives. Do not call any other tool "
    "and do not start any other agent. All agents may use only the paper text provided "
    "in the first message."
)
REMINDER = (
    "The required process is not complete. A final review only counts if (1) you called "
    "`subagent_fork` so that a subagent reviewed the submission, (2) its report was "
    "delivered to you, and (3) you then output the final review JSON object. Do whatever "
    "is still missing now: if no subagent has reported yet, call `subagent_fork` once and "
    "wait for its report; once you have the report, output the final review JSON object "
    "and nothing else. Do not call any other tool."
)


def norm_ws(s: str) -> str:
    return re.sub(r'\s+', ' ', s)


def coverage(paper: str, text: str, size: int = 200) -> float:
    """Share of the paper's consecutive 200-char chunks found verbatim (whitespace-
    normalized) in `text`."""
    p, q = norm_ws(paper), norm_ws(text)
    chunks = [p[i:i + size] for i in range(0, max(len(p) - size + 1, 1), size)]
    return sum(c in q for c in chunks) / len(chunks)


def load_session(session_dir: str) -> list[dict] | None:
    paths = glob.glob(str(R.TOOLS / 'dsh-home' / 'sessions' / '*' / session_dir
                          / 'session.v4.jsonl.zstd'))
    if not paths:
        return None
    try:
        import zstandard
        data = zstandard.ZstdDecompressor().stream_reader(
            io.BytesIO(Path(paths[0]).read_bytes())).read().decode('utf-8', 'replace')
    except Exception:
        return None
    rows = []
    for line in data.split('\n'):
        if line.strip():
            try:
                rows.append(json.loads(line))
            except ValueError:
                continue
    return rows


def split_seed(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """(seeded copy of the parent's history, the child's own records).

    A resumed parent writes its own `session/end-seed {}` marker, and a fork started
    after that copies it, so the cut is the LAST marker flagged inherited — the one dsh
    writes when it finishes seeding the child."""
    cut = None
    for i, r in enumerate(rows):
        if r.get('type') == 'session/end-seed' and (r.get('data') or {}).get('inherited'):
            cut = i
    if cut is None:  # older shape: take the last marker before the child's descriptor
        for i, r in enumerate(rows):
            if r.get('type') == 'subagent/descriptor':
                break
            if r.get('type') == 'session/end-seed':
                cut = i
    if cut is None:
        return [], rows
    return rows[:cut], rows[cut + 1:]


def message_text(data: dict) -> str:
    """Text of a user/assistant message record (both shapes dsh writes)."""
    content = data.get('content')
    if content is None:
        content = (data.get('message') or {}).get('content')
    if isinstance(content, str):
        return content
    return ''.join(b.get('text', '') for b in (content or [])
                   if isinstance(b, dict) and b.get('type') == 'text')


def add_usage(total: dict, rows: list[dict]) -> int:
    """Add the committed usage of every assistant message; return the retry count."""
    retries = 0
    for r in rows:
        d = r.get('data') or {}
        if r.get('type') == 'llm/retry':
            retries += 1
        elif r.get('type') == 'assistant/message' and isinstance(d, dict):
            u = d.get('usage') or {}
            total['input'] += u.get('inputTokens', 0) or 0
            total['cache_read'] += u.get('cacheReadTokens', 0) or 0
            total['output'] += u.get('outputTokens', 0) or 0
            total['messages'] += 1
    return retries


def process_facts(session_id: str | None, paper: str) -> dict:
    """Everything the compliance check and the accounting need, from the session logs."""
    f: dict = {'log_found': False, 'process_ok': False, 'coverage': 0.0, 'child_id': None,
               'child_tool': None, 'child_done': False, 'child_model': None,
               'child_effort': None, 'report_delivered': False, 'report_via': None,
               'review': None,
               'coerced': False, 'n_review_outputs': 0, 'review_before_report': False,
               'n_children': 0, 'n_turns': 0, 'llm_retries': 0, 'tool_calls': {},
               'usage': {'input': 0, 'cache_read': 0, 'output': 0, 'messages': 0},
               'usage_main': {'input': 0, 'cache_read': 0, 'output': 0, 'messages': 0},
               'missing_child_logs': 0}
    log = load_session(session_id) if session_id else None
    if log is None:
        return f
    f['log_found'] = True
    f['n_turns'] = sum(r.get('type') == 'turn/start' for r in log)
    f['tool_calls'] = dict(Counter((r.get('data') or {}).get('name') for r in log
                                   if r.get('type') == 'tool/call'))
    children = [(r.get('data') or {}).get('childId') for r in log
                if r.get('type') == 'subagent/catalog']
    f['n_children'] = len(children)
    child_logs = {cid: load_session(cid) for cid in children if cid}

    i_call = None
    for i, r in enumerate(log):
        d = r.get('data') or {}
        if r.get('type') != 'tool/call' or d.get('name') not in DELEGATION_TOOLS:
            continue
        cid = next(((x.get('data') or {}).get('childId') for x in log[i + 1:i + 5]
                    if x.get('type') == 'subagent/catalog'), None)
        if d.get('name') == 'subagent':  # spawned: the paper must be in its prompt
            try:
                seen = json.loads(d.get('arguments') or '{}').get('prompt') or ''
            except ValueError:
                seen = ''
        else:  # forked: the paper must be in the history it was seeded with
            seeded, _ = split_seed(child_logs.get(cid) or [])
            seen = ' '.join(message_text(x.get('data') or {}) for x in seeded
                            if x.get('type') == 'user/message')
        cov = coverage(paper, seen)
        f['coverage'] = max(f['coverage'], cov)
        if cov >= 0.9 and cid and i_call is None:
            i_call, f['child_id'], f['child_tool'] = i, cid, d.get('name')
    i_report = None
    if f['child_id']:
        # a child normally reports with send_message; one that just ends its turn has
        # its answer delivered inside dsh's finish notice instead
        sent = f"Agent {f['child_id']} sent a message"
        finished = f"Background subagent {f['child_id']} finished"
        for i in range(i_call + 1, len(log)):
            r = log[i]
            if r.get('type') != 'user/message':
                continue
            text = message_text(r.get('data') or {})
            if sent in text or (finished in text and 'Its closing message:' in text):
                i_report = i
                f['report_delivered'] = True
                f['report_via'] = 'send_message' if sent in text else 'closing_message'
                break
    for i, r in enumerate(log):
        if r.get('type') != 'assistant/message':
            continue
        text = message_text(r.get('data') or {})
        if not text.strip():
            continue
        try:
            review, coerced = validate_review(R.extract_review(text))
        except Exception:  # not a review
            continue
        f['n_review_outputs'] += 1
        if i_report is None or i < i_report:
            f['review_before_report'] = True
        elif f['review'] is None:  # the FIRST review after the report is the result
            f['review'], f['coerced'] = review, coerced

    f['llm_retries'] += add_usage(f['usage_main'], log)
    for k in ('input', 'cache_read', 'output', 'messages'):
        f['usage'][k] += f['usage_main'][k]
    for cid in children:
        child = child_logs.get(cid)
        if child is None:
            f['missing_child_logs'] += 1
            continue
        _, own = split_seed(child)  # the seeded copy of the parent is not counted twice
        f['llm_retries'] += add_usage(f['usage'], own)
        if cid == f['child_id']:
            f['child_done'] = any(c.get('type') == 'turn/end'
                                  and ((c.get('data') or {}).get('reason') or {}).get('kind')
                                  == 'completed' for c in own)
            desc = next(((c.get('data') or {}) for c in child
                         if c.get('type') == 'subagent/descriptor'), {})
            f['child_model'] = desc.get('agentModel')
            f['child_effort'] = desc.get('agentReasoningEffort')
    f['process_ok'] = bool(i_call is not None and f['child_done'] and f['report_delivered']
                           and f['review'] is not None)
    return f


def run_dsh(text: str, api_key: str, cfg: argparse.Namespace, timeout: int,
            session_id: str | None = None) -> tuple[int | None, str, str]:
    cmd = [R.dsh_bin(), '--profile', 'headless', '--patch', str(R.REPO_ROOT / cfg.patch),
           '--json', '--permission-mode', 'read-only', '--approval-policy', 'never']
    if session_id:
        cmd += ['--session-id', session_id]
    cmd += ['-']
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True, encoding='utf-8',
                            errors='replace', env=R.harness_env(api_key), cwd=cfg.workspace,
                            start_new_session=True)
    try:
        out, err = proc.communicate(text, timeout=timeout)
        return proc.returncode, out, err
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        out, err = proc.communicate()
        return None, out or '', err or ''


def stream_facts(stdout: str) -> tuple[str | None, bool]:
    """(session id, whether the run printed a completed `final` event)."""
    sid, completed = None, False
    for line in stdout.split('\n'):
        if not line.strip():
            continue
        try:
            ev = json.loads(line)
        except ValueError:
            continue
        if ev.get('type') == 'session':
            sid = ev.get('sessionId')
        elif ev.get('type') == 'final':
            completed = (ev.get('reason') or {}).get('kind') == 'completed'
    return sid, completed


def run_task(task: dict, cfg: argparse.Namespace, api_key: str, system: str,
             user_template: str, meta: dict, budget: R.Budget) -> dict | None:
    """Never raises. Appends its own row; returns it, or None when not started."""
    if R.STOP.is_set() or budget.exhausted():
        return None
    row = {
        'key': task['key'], 'arxiv_id': task['arxiv_id'], 'v': task['v'], 'rep': task['rep'],
        'arm': ARM, 'model': cfg.model, 'effort': cfg.effort,
        'rating': None, 'soundness': None, 'presentation': None, 'contribution': None,
        'confidence': None, 'accept': None, 'coerced': False, 'parsed_ok': False,
        'process_ok': None, 'reminded': False, 'coverage': None, 'child_tool': None,
        'n_children': None, 'n_turns': None, 'n_review_outputs': None,
        'review_before_report': None, 'tool_calls': {}, 'child_model': None,
        'child_effort': None, 'truncated': False, 'text_sha': None, 'task_chars': 0,
        'usage': {}, 'usage_main': {}, 'usage_missing': True, 'llm_retries': 0,
        'failed_attempts': 0, 'session_id': None, 'cost_cny_est': None, 'guard_cny': 0.0,
        'peak': None, 'latency_s': None, 'ts': None, 'raw_path': None, 'error': None,
        **meta,
    }
    t0 = time.monotonic()
    task_chars = 0
    streams: list[str] = []
    facts = None
    paper = ''
    sid = None
    try:
        raw_bytes = Path(task['text_path']).read_bytes()
        paper = raw_bytes.decode('utf-8', errors='replace')
        row['text_sha'] = hashlib.sha256(raw_bytes).hexdigest()
        if len(paper) > cfg.max_chars:
            paper = paper[:cfg.max_chars] + '\n[truncated]'
            row['truncated'] = True
        turn1 = system + '\n\n' + user_template.replace('{PAPER_TEXT}', paper) + PRIMER
        task_chars = row['task_chars'] = len(turn1) + len(PROCESS)

        # turn 1: deliver the paper (one retry for a failed process, never for a timeout)
        for attempt in range(2):
            code, out, err = run_dsh(turn1, api_key, cfg, cfg.primer_timeout)
            streams.append(out)
            sid, completed = stream_facts(out)
            if completed:
                break
            row['failed_attempts'] += 1
            sid = None
            if code is None or attempt or R.STOP.wait(10.0):
                raise RuntimeError(f'turn 1 failed (exit {code}): {err.strip()[-300:]}')
        row['session_id'] = sid

        # turn 2: the required process
        code, out, err = run_dsh(PROCESS, api_key, cfg, cfg.harness_timeout, session_id=sid)
        streams.append(out)
        facts = process_facts(sid, paper)
        if not facts['process_ok'] and (code is None or code != 0):
            row['failed_attempts'] += 1
            raise RuntimeError(f"turn 2 {'timed out' if code is None else f'exit {code}'}: "
                               f"{err.strip()[-300:]}")
        if not facts['process_ok'] and not R.STOP.is_set():
            row['reminded'] = True  # one reminder on the same session, then the verdict
            code, out, err = run_dsh(REMINDER, api_key, cfg, cfg.harness_timeout,
                                     session_id=sid)
            streams.append(out)
            facts = process_facts(sid, paper)
            if not facts['process_ok'] and (code is None or code != 0):
                row['failed_attempts'] += 1
                raise RuntimeError(f'reminder run failed (exit {code}): '
                                   f'{err.strip()[-300:]}')
        row['process_ok'] = facts['process_ok']
        if facts['process_ok']:
            row.update(facts['review'])
            row['coerced'] = facts['coerced']
            row['parsed_ok'] = True
        else:
            row['error'] = 'process incomplete after one reminder'
    except Exception as exc:  # technical failure: process_ok stays None, retried later
        row['error'] = f'{type(exc).__name__}: {str(exc)[:400]}'
    if facts is None and sid:  # failed after the session existed: still account for it
        facts = process_facts(sid, paper)
    if facts is not None:
        row.update(coverage=round(facts['coverage'], 3), child_tool=facts['child_tool'],
                   report_via=facts['report_via'],
                   n_children=facts['n_children'], n_turns=facts['n_turns'],
                   n_review_outputs=facts['n_review_outputs'],
                   review_before_report=facts['review_before_report'],
                   tool_calls=facts['tool_calls'], child_model=facts['child_model'],
                   child_effort=facts['child_effort'], llm_retries=facts['llm_retries'],
                   usage={k: facts['usage'][k] for k in ('input', 'cache_read', 'output')},
                   usage_main={k: facts['usage_main'][k]
                               for k in ('input', 'cache_read', 'output')},
                   usage_missing=not (facts['log_found'] and facts['usage']['messages']
                                      and not facts['missing_child_logs']))
    now = time.time()
    row['latency_s'] = round(time.monotonic() - t0, 1)
    row['ts'] = datetime.fromtimestamp(now, tz=timezone.utc).isoformat(timespec='seconds')
    row['peak'] = R.is_peak(now)
    one_call = R.flat_estimate_cny(task_chars, now)
    if row['usage']:
        row['cost_cny_est'] = round(R.cost_cny(row['usage'], now), 5)
    known = row['cost_cny_est'] if (row['cost_cny_est'] is not None
                                    and not row['usage_missing']) else None
    # dsh's internal retries here are mostly empty responses while the main agent waits
    # for its child: a cached re-read and almost no output, so they are charged as such
    retry = R.cost_cny({'input': 2000, 'cache_read': task_chars // 3, 'output': 1000}, now)
    guard = (known if known is not None else (3 * one_call if task_chars else 0.0)) \
        + one_call * row['failed_attempts'] + retry * row['llm_retries']
    row['guard_cny'] = round(guard, 5)
    budget.add(row['guard_cny'])

    if streams:
        try:
            raw_dir = Path(cfg.out_dir) / ARM
            raw_dir.mkdir(parents=True, exist_ok=True)
            raw_path = raw_dir / f"{task['safe_id']}v{task['v']}_r{task['rep']}.events.jsonl"
            with open(raw_path, 'w', encoding='utf-8', errors='backslashreplace') as fh:
                fh.write('\n'.join(streams))
            row['raw_path'] = str(raw_path)
        except OSError as exc:
            row['raw_path'] = f'(not written: {type(exc).__name__})'
    try:
        with R.LOG_LOCK:
            append_jsonl(cfg.log, row)
    except Exception as exc:
        print(f'\nLOG WRITE FAILED for {row["key"]}: {exc}', flush=True)
        R.STOP.set()
    budget.note(bool(row['parsed_ok']), cfg.abort_after)
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description='E1-H arm B1: forced subagent review.')
    ap.add_argument('--manifest', default='data/processed/pilot_manifest.jsonl')
    ap.add_argument('--texts', default='data/processed/texts/')
    ap.add_argument('--prompt', default='src/review/prompts/review_v1.md')
    ap.add_argument('--patch', default='src/review/harness_lockdown.yml')
    ap.add_argument('--log', default='data/processed/e1h_sub_reviews_log.jsonl')
    ap.add_argument('--out-dir', default='data/processed/reviews_e1h/')
    ap.add_argument('--model', default='deepseek-flash',
                    help='label only: the harness profile decides the model (deepseek-flash)')
    ap.add_argument('--effort', default='high', help='label only (harness adapter default)')
    ap.add_argument('--api-key-env', default='DEEPSEEK_API_KEY')
    ap.add_argument('--workspace', default='/private/tmp/ws-e1h-sub')
    ap.add_argument('--reps', type=int, default=3)
    ap.add_argument('--sample-papers', type=int, default=0)
    ap.add_argument('--workers', type=int, default=12)
    ap.add_argument('--primer-timeout', type=int, default=600,
                    help='total seconds for turn 1 (delivering the paper)')
    ap.add_argument('--harness-timeout', type=int, default=1200,
                    help='total seconds for turn 2 (main agent plus its child)')
    ap.add_argument('--max-chars', type=int, default=380000)
    ap.add_argument('--budget-cny', type=float, default=110.0,
                    help='total guard spend allowed for this log, across invocations')
    ap.add_argument('--abort-after', type=int, default=12,
                    help='stop after this many consecutive runs without a usable result')
    ap.add_argument('--wait-offpeak', action='store_true')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--seed', type=int, default=43)
    cfg = ap.parse_args()

    api_key = load_api_key(cfg.api_key_env, mock=False)
    system, user_template = load_prompt(cfg.prompt)
    ws = Path(cfg.workspace)
    ws.mkdir(parents=True, exist_ok=True)
    if any(ws.iterdir()):
        sys.exit(f'harness workspace {ws} is not empty — refusing to run')
    for d in ('fakehome', 'dsh-home'):
        (R.TOOLS / d).mkdir(parents=True, exist_ok=True)
    ver = subprocess.run([R.dsh_bin(), '--version'], capture_output=True, text=True,
                         env=R.harness_env(''), cwd=cfg.workspace, timeout=120)
    if ver.returncode != 0:
        sys.exit(f'dsh --version failed: {ver.stderr.strip()[:300]}')
    meta = {'prompt_sha': hashlib.sha256(Path(cfg.prompt).read_bytes()).hexdigest(),
            'harness_version': ver.stdout.strip(),
            'patch_sha': hashlib.sha256((R.REPO_ROOT / cfg.patch).read_bytes()).hexdigest(),
            'config_sha': R.assert_lockdown(cfg)}
    meta['cfg_sig'] = hashlib.sha256(json.dumps(
        [meta['prompt_sha'], meta['patch_sha'], meta['harness_version'], meta['config_sha'],
         cfg.model, cfg.effort, cfg.max_chars, PRIMER, PROCESS, REMINDER]
    ).encode()).hexdigest()[:16]

    prior = list(read_jsonl(cfg.log)) if Path(cfg.log).exists() else []
    # a run is finished once the process verdict exists (usable result or process failure)
    done = {r['key'] for r in prior
            if r.get('process_ok') is not None and r.get('cfg_sig') == meta['cfg_sig']}
    spent_before = sum(r.get('guard_cny') or 0.0 for r in prior)

    papers = select_papers(read_jsonl(cfg.manifest), cfg.sample_papers)
    tasks, n_skipped, n_missing = [], 0, 0
    for paper in papers:
        safe_id = paper['arxiv_id'].replace('/', '_')
        for ver_row in paper['versions']:
            v = ver_row['v']
            text_path = Path(cfg.texts) / safe_id / f'v{v}.md'
            if not text_path.exists():
                n_missing += 1
                continue
            for rep in range(cfg.reps):
                key = f"{paper['arxiv_id']}v{v}|{ARM}|r{rep}"
                if key in done:
                    n_skipped += 1
                    continue
                tasks.append({'key': key, 'arxiv_id': paper['arxiv_id'], 'safe_id': safe_id,
                              'v': v, 'rep': rep, 'text_path': str(text_path)})
    random.Random(cfg.seed).shuffle(tasks)
    if cfg.limit > 0:
        tasks = tasks[:cfg.limit]
    print(f"papers: {len(papers)} | tasks: {len(tasks)} new, {n_skipped} already finished "
          f"| missing texts: {n_missing} | dsh={meta['harness_version']} "
          f"| cfg_sig={meta['cfg_sig']} | guard spend so far ¥{spent_before:.2f} of cap "
          f"¥{cfg.budget_cny}", flush=True)

    if cfg.wait_offpeak and R.is_peak(time.time()):
        print('peak pricing now — waiting for off-peak ...', flush=True)
        while R.is_peak(time.time()):
            time.sleep(30)
        print('off-peak; starting.', flush=True)

    budget = R.Budget(cfg.budget_cny, spent_before)
    balance_before = R.account_balance_cny(api_key)
    print(f'account balance before: ¥{balance_before}', flush=True)
    s: Counter = Counter()
    hist: Counter = Counter()
    n_not_started = 0
    pbar = tqdm(total=len(tasks), desc='e1h-sub', unit='run', mininterval=30)
    ex = ThreadPoolExecutor(max_workers=cfg.workers)
    try:
        futures = [ex.submit(run_task, t, cfg, api_key, system, user_template, meta, budget)
                   for t in tasks]
        for fut in as_completed(futures):
            row = fut.result()
            pbar.update(1)
            if row is None:
                n_not_started += 1
                continue
            s['runs'] += 1
            s['usable'] += bool(row['parsed_ok'])
            s['process_failed'] += row['process_ok'] is False
            s['technical_failed'] += row['process_ok'] is None
            s['reminded'] += bool(row['reminded'])
            s['extra_children'] += max((row['n_children'] or 0) - 1, 0)
            s['in'] += row['usage'].get('input', 0)
            s['hit'] += row['usage'].get('cache_read', 0)
            s['out'] += row['usage'].get('output', 0)
            if row['parsed_ok']:
                hist[row['rating']] += 1
    except BaseException:
        R.STOP.set()
        print('\ninterrupted — cancelling queued tasks, waiting for in-flight runs ...',
              flush=True)
        raise
    finally:
        ex.shutdown(wait=True, cancel_futures=True)
        pbar.close()

    print('\n=== summary ===')
    print(f"{ARM}: runs {s['runs']} | usable {s['usable']} | process failures "
          f"{s['process_failed']} | technical failures {s['technical_failed']} | needed a "
          f"reminder {s['reminded']} | extra subagents started {s['extra_children']}")
    print(f"   tokens in {s['in']:,} cache-hit {s['hit']:,} out {s['out']:,}")
    print('   ratings: ' + ', '.join(f'{r}: {hist.get(r, 0)}' for r in ALLOWED_RATINGS))
    balance_after = R.account_balance_cny(api_key)
    real = (f'¥{balance_before - balance_after:.2f}'
            if balance_before is not None and balance_after is not None else 'unknown')
    print(f"guard spend for this log (conservative): ¥{budget.spent:.2f} of cap "
          f"¥{cfg.budget_cny} | real spend this invocation by account balance: {real} "
          f"(balance now ¥{balance_after}; other runs may be spending at the same time)")
    if n_not_started:
        why = 'aborted' if R.STOP.is_set() else 'budget cap reached'
        print(f'{n_not_started} tasks were NOT started ({why}); re-run to resume.')
    print(f'log: {cfg.log} | raw: {cfg.out_dir}')


if __name__ == '__main__':
    main()
