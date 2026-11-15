"""E1-H: does an agent harness change how review scores track paper versions?

Two arms score the SAME task text (review_v1 rules + one scrubbed paper version):
  direct   one call to DeepSeek's Anthropic-format Messages API, no system prompt, no tools
  harness  the same text piped to DeepSeek Harness (`dsh --profile headless`), which adds
           its own system prompt, tool catalog, runtime-context block, request metadata
           (session log, plugin list, harness headers) and agent loop

Everything else is held fixed. The direct arm sends exactly the sampling parameters dsh
0.2.2-rc.1 puts on the wire by default (captured with a record-and-forward proxy on
2026-11-12): model deepseek-flash, thinking enabled, output_config.effort high,
max_tokens 256000, streaming, no temperature. Each paper version is an independent
session; neither arm is told which version it is reading.

The harness runs locked down (src/review/harness_lockdown.yml), with telemetry disabled,
a read-only sandbox, approvals auto-rejected, an empty neutral working directory, and an
environment that contains only the DeepSeek key. The run refuses to start if the composed
config does not show the lockdown.

Rows are appended by the worker threads themselves (append-only JSONL keyed on
"{arxiv_id}v{v}|{arm}|r{rep}"), so nothing is lost if the main loop dies. Only parsed_ok
rows written under the SAME configuration count as done; a re-run retries the rest. Tasks
from both arms are shuffled together so the arms share the same wall-clock mix. Spending
is bounded by a conservative guard that persists across invocations of the same log.

Run from the repo root:
    python3 src/review/run_e1_harness.py --sample-papers 2 --reps 1 --log ... --out-dir ...   # smoke
    python3 src/review/run_e1_harness.py --wait-offpeak                                      # full pilot set
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
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import requests
from tqdm import tqdm

sys.path.insert(0, 'src/collect')
sys.path.insert(0, 'src/review')
from util import read_jsonl, append_jsonl  # noqa: E402
from run_llm_review import (  # noqa: E402
    ALLOWED_RATINGS, load_api_key, load_prompt, select_papers, strip_fences,
    validate_review,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
TOOLS = REPO_ROOT / '.tools'
ARMS = ('direct', 'harness')
RETRY_STATUSES = {429, 500, 502, 503, 504, 529}
MAX_ATTEMPTS = 6  # dsh retries a step up to 5 times; give the direct arm the same room
# patch entries that must be off in the composed config, or the run does not start
REQUIRED_DISABLED = ('tool-web', 'web', 'web-search-deepseek', 'web-deepseek-api',
                     'web-fetch-http', 'tool-bash', 'tool-jobs', 'tool-fs',
                     'tool-fs-search', 'tool-workflow', 'tool-working-directory',
                     'goal-round-driver')
KNOWN_EVENTS = {'session', 'status', 'thinking', 'text', 'final'}
# deepseek-flash list price, USD per 1M tokens: (cache hit, cache miss, output).
# Peak = 01:00-04:00 and 06:00-10:00 UTC, Mon-Fri; off-peak is half price.
PRICE_PEAK = (0.006, 0.30, 1.20)
PRICE_OFFPEAK = (0.003, 0.15, 0.60)
USD_TO_CNY = 7.2  # for the budget guard only; the account balance is the ground truth

STOP = threading.Event()   # set on abort, budget exhaustion, or a failure streak
LOG_LOCK = threading.Lock()


def is_peak(ts: float) -> bool:
    t = datetime.fromtimestamp(ts, tz=timezone.utc)
    return t.weekday() < 5 and (1 <= t.hour < 4 or 6 <= t.hour < 10)


def cost_cny(usage: dict, ts: float) -> float:
    hit, miss, out = PRICE_PEAK if is_peak(ts) else PRICE_OFFPEAK
    usd = (usage.get('cache_read', 0) * hit + usage.get('input', 0) * miss
           + usage.get('output', 0) * out) / 1e6
    return usd * USD_TO_CNY


def flat_estimate_cny(task_chars: int, ts: float) -> float:
    """Generous charge for an attempt whose usage was never reported."""
    return cost_cny({'input': task_chars // 3 + 4000, 'cache_read': 0, 'output': 20000}, ts)


def extract_review(text: str) -> dict:
    """Parse the JSON review out of a free-text answer (no response_format in either arm)."""
    s = strip_fences(text)
    try:
        return json.loads(s)
    except ValueError:
        i, j = s.find('{'), s.rfind('}')
        if i == -1 or j <= i:
            raise
        return json.loads(s[i:j + 1])


# ---------------------------------------------------------------- direct arm

def direct_call(task_text: str, api_key: str, cfg: argparse.Namespace, acct: dict) -> dict:
    url = cfg.base_url.rstrip('/') + '/v1/messages'
    payload = {
        "model": cfg.model, "stream": True, "max_tokens": cfg.max_tokens,
        "thinking": {"type": "enabled"}, "output_config": {"effort": cfg.effort},
        "messages": [{"role": "user", "content": [{"type": "text", "text": task_text}]}],
    }
    headers = {"x-api-key": api_key, "anthropic-version": "2023-06-01",
               "content-type": "application/json", "accept": "text/event-stream"}
    last_err = 'unknown error'
    for attempt in range(MAX_ATTEMPTS):
        if attempt:
            acct['failed_attempts'] += 1
            if STOP.wait(min(10.0 * attempt, 60)):
                break
        try:
            with requests.post(url, json=payload, headers=headers, stream=True,
                               timeout=(30, cfg.timeout)) as resp:
                if resp.status_code in RETRY_STATUSES:
                    last_err = f'HTTP {resp.status_code}'
                    continue
                if resp.status_code != 200:
                    raise RuntimeError(f'HTTP {resp.status_code}: {resp.text[:300]}')
                text: list[str] = []
                thinking_chars = bad_lines = 0
                usage: dict = {}
                stop = served = stream_error = None
                # bytes lines: str.splitlines would also break on U+2028/2029/0085
                for raw in resp.iter_lines():
                    if not raw.startswith(b'data:'):
                        continue
                    try:
                        ev = json.loads(raw[5:].decode('utf-8', 'replace').strip())
                    except ValueError:
                        bad_lines += 1
                        continue
                    kind = ev.get('type')
                    if kind == 'message_start':
                        usage.update(ev['message'].get('usage') or {})
                        served = ev['message'].get('model')
                    elif kind == 'content_block_delta':
                        d = ev.get('delta') or {}
                        if d.get('type') == 'text_delta':
                            text.append(d.get('text', ''))
                        elif d.get('type') == 'thinking_delta':
                            thinking_chars += len(d.get('thinking', ''))
                    elif kind == 'message_delta':
                        usage.update(ev.get('usage') or {})
                        stop = (ev.get('delta') or {}).get('stop_reason')
                    elif kind == 'error':
                        stream_error = json.dumps(ev)[:300]
                        break
                answer = ''.join(text)
                if stream_error is not None:
                    last_err = f'stream error: {stream_error}'
                    continue
                if stop is None:
                    last_err = 'stream ended without message_delta'
                    continue
                if stop != 'end_turn':
                    raise RuntimeError(f'stop_reason {stop}')
                if not answer.strip():
                    last_err = 'empty answer'
                    continue
                return {
                    'text': answer, 'thinking_chars': thinking_chars, 'served_model': served,
                    'n_steps': 1, 'other_events': {}, 'session_id': None, 'llm_retries': 0,
                    'bad_stream_lines': bad_lines, 'usage_missing': False,
                    'usage': {'input': usage.get('input_tokens', 0) or 0,
                              'cache_read': usage.get('cache_read_input_tokens', 0) or 0,
                              'output': usage.get('output_tokens', 0) or 0},
                    'raw': None,
                }
        except requests.RequestException as exc:
            last_err = f'connection error: {type(exc).__name__}'
    else:
        acct['failed_attempts'] += 1  # the last attempt failed too
    raise RuntimeError(f'request failed after {attempt + 1} attempts ({last_err})')


# ---------------------------------------------------------------- harness arm

def harness_env(api_key: str) -> dict:
    """The ONLY variables the harness process sees — no other provider keys, real HOME
    replaced, telemetry transport never constructed."""
    return {
        'PATH': f"{TOOLS / 'node' / 'bin'}:/usr/bin:/bin",
        'HOME': str(TOOLS / 'fakehome'),
        'DSH_HOME': str(TOOLS / 'dsh-home'),
        'LANG': 'en_US.UTF-8',
        'DSH_TELEMETRY_MODE': 'DISABLED',
        'DSH_PERMISSION_MODE': 'read-only',
        'DEEPSEEK_API_KEY': api_key,
    }


def dsh_bin() -> str:
    return str(TOOLS / 'dsh' / 'node_modules' / '.bin' / 'dsh')


def harness_cmd(cfg: argparse.Namespace) -> list[str]:
    return [dsh_bin(), '--profile', 'headless', '--patch', str(REPO_ROOT / cfg.patch),
            '--json', '--permission-mode', 'read-only', '--approval-policy', 'never', '-']


def session_facts(session_id: str | None) -> dict:
    """Committed usage, served model and retry count from dsh's own session log — the
    --json stream omits usage whenever an internal retry happened."""
    facts: dict = {'usage': None, 'served_model': None, 'llm_retries': 0}
    if not session_id:
        return facts
    paths = glob.glob(str(TOOLS / 'dsh-home' / 'sessions' / '*' / session_id
                          / 'session.v4.jsonl.zstd'))
    if not paths:
        return facts
    try:
        import zstandard
        data = zstandard.ZstdDecompressor().stream_reader(
            io.BytesIO(Path(paths[0]).read_bytes())).read().decode('utf-8', 'replace')
    except Exception:  # missing module, half-written log
        return facts
    usage = {'input': 0, 'cache_read': 0, 'output': 0}
    seen = False
    for line in data.split('\n'):
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        kind, d = rec.get('type'), rec.get('data') or {}
        if kind == 'llm/retry':
            facts['llm_retries'] += 1
        elif kind == 'assistant/message' and isinstance(d, dict):
            u = d.get('usage') or {}
            if u:
                seen = True
                usage['input'] += u.get('inputTokens', 0) or 0
                usage['cache_read'] += u.get('cacheReadTokens', 0) or 0
                usage['output'] += u.get('outputTokens', 0) or 0
            src = ((d.get('message') or {}).get('source') or {})
            model = ((src.get('replayState') or {}).get('response') or {}).get('model')
            facts['served_model'] = model or facts['served_model']
    if seen:
        facts['usage'] = usage
    return facts


def run_dsh_once(task_text: str, api_key: str, cfg: argparse.Namespace) -> tuple[int | None, str, str]:
    """One dsh process in its own process group. Returns (returncode or None on timeout,
    stdout, stderr); on timeout the whole group is killed and partial output is kept."""
    proc = subprocess.Popen(harness_cmd(cfg), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True, encoding='utf-8',
                            errors='replace', env=harness_env(api_key), cwd=cfg.workspace,
                            start_new_session=True)
    try:
        out, err = proc.communicate(task_text, timeout=cfg.harness_timeout)
        return proc.returncode, out, err
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        out, err = proc.communicate()
        return None, out or '', err or ''


def harness_call(task_text: str, api_key: str, cfg: argparse.Namespace, acct: dict) -> dict:
    last_err = 'unknown error'
    for attempt in range(2):  # one retry, and only for a failed process — never for a timeout
        if attempt:
            if STOP.wait(10.0):
                break
        code, stdout, stderr = run_dsh_once(task_text, api_key, cfg)
        events = []
        for line in stdout.split('\n'):  # not splitlines(): JSON may carry raw U+2028
            if line.strip():
                try:
                    events.append(json.loads(line))
                except ValueError:
                    continue
        final = next((e for e in events if e.get('type') == 'final'), None)
        sid = next((e.get('sessionId') for e in events if e.get('type') == 'session'), None)
        completed = final is not None and (final.get('reason') or {}).get('kind') == 'completed'
        if not completed:
            acct['failed_attempts'] += 1
            if code is None:
                raise RuntimeError(f'timeout after {cfg.harness_timeout}s (not retried)')
            last_err = f'exit {code}: {stderr.strip()[-300:]}'
            continue
        usage = {'input': 0, 'cache_read': 0, 'output': 0}
        n_steps, stream_usage_missing = 0, False
        for e in events:
            if e.get('type') == 'status' and e.get('phase') == 'step_end':
                n_steps += 1
                u = e.get('usage') or {}
                stream_usage_missing |= not u
                usage['input'] += u.get('inputTokens', 0) or 0
                usage['cache_read'] += u.get('cacheReadTokens', 0) or 0
                usage['output'] += u.get('outputTokens', 0) or 0
        facts = session_facts(sid)
        usage_missing = stream_usage_missing
        if facts['usage'] is not None:  # the session log is authoritative
            usage, usage_missing = facts['usage'], False
        return {
            'text': final.get('text') or '',
            # the stream caps reasoning text at 8 KiB, so its length is not a measure
            'thinking_chars': None,
            'served_model': facts['served_model'], 'n_steps': n_steps,
            # anything beyond plain thinking/text is the harness doing agent work
            'other_events': dict(Counter(e.get('type') for e in events
                                         if e.get('type') not in KNOWN_EVENTS)),
            'session_id': sid, 'llm_retries': facts['llm_retries'], 'bad_stream_lines': 0,
            'usage': usage, 'usage_missing': usage_missing, 'raw': stdout,
        }
    raise RuntimeError(last_err)


def account_balance_cny(api_key: str) -> float | None:
    """Real remaining balance — the ground truth the estimated spend is checked against."""
    try:
        r = requests.get('https://api.deepseek.com/user/balance', timeout=30,
                         headers={'Authorization': f'Bearer {api_key}'})
        for info in r.json().get('balance_infos', []):
            if info.get('currency') == 'CNY':
                return float(info['total_balance'])
    except (requests.RequestException, ValueError, KeyError):
        pass
    return None


def assert_lockdown(cfg: argparse.Namespace) -> str:
    """Compose the profile (no model call) and refuse to run unless every lockdown entry
    is disabled — a dsh upgrade that renames an id must not silently re-enable a tool."""
    p = subprocess.run([dsh_bin(), '--profile', 'headless', '--patch',
                        str(REPO_ROOT / cfg.patch), '--dump-config'],
                       capture_output=True, text=True, env=harness_env(''),
                       cwd=cfg.workspace, timeout=180)
    if p.returncode != 0:
        sys.exit(f'dsh --dump-config failed: {p.stderr.strip()[:300]}')
    entries = {e.split('\n', 1)[0].strip(): e
               for e in re.split(r'(?m)^- id: ', p.stdout)[1:]}
    bad = [i for i in REQUIRED_DISABLED
           if i not in entries or not re.search(r'(?m)^  disabled: true\s*$', entries[i])]
    if bad:
        sys.exit(f'lockdown not in effect for: {bad} — refusing to run')
    return hashlib.sha256(p.stdout.encode('utf-8')).hexdigest()


# ---------------------------------------------------------------- per-task worker

class Budget:
    def __init__(self, cap_cny: float, already_spent: float) -> None:
        self.cap, self.spent, self.lock = cap_cny, already_spent, threading.Lock()
        self.fail_streak = 0

    def add(self, amount: float) -> None:
        with self.lock:
            self.spent += amount

    def exhausted(self) -> bool:
        with self.lock:
            return self.spent >= self.cap

    def note(self, ok: bool, abort_after: int) -> None:
        with self.lock:
            self.fail_streak = 0 if ok else self.fail_streak + 1
            streak = self.fail_streak
        if streak >= abort_after and not STOP.is_set():
            STOP.set()
            print(f'\nABORT: {streak} consecutive failed runs — stopping.', flush=True)


def run_task(task: dict, cfg: argparse.Namespace, api_key: str, system: str,
             user_template: str, meta: dict, budget: Budget) -> dict | None:
    """Never raises. Appends its own row to the log; returns it for the tally, or None
    when the task was not started (abort or budget)."""
    if STOP.is_set() or budget.exhausted():
        return None
    row = {
        'key': task['key'], 'arxiv_id': task['arxiv_id'], 'v': task['v'], 'rep': task['rep'],
        'arm': task['arm'], 'model': cfg.model, 'effort': cfg.effort,
        'rating': None, 'soundness': None, 'presentation': None, 'contribution': None,
        'confidence': None, 'accept': None, 'coerced': False, 'parsed_ok': False,
        'truncated': False, 'text_sha': None, 'task_chars': 0, 'usage': {},
        'usage_missing': True, 'n_steps': None, 'other_events': {}, 'llm_retries': 0,
        'failed_attempts': 0, 'thinking_chars': None, 'served_model': None,
        'session_id': None, 'bad_stream_lines': 0, 'cost_cny_est': None, 'guard_cny': 0.0,
        'peak': None, 'latency_s': None, 'ts': None, 'raw_path': None, 'error': None,
        **meta,
    }
    t0 = time.monotonic()
    acct = {'failed_attempts': 0}
    out = None
    task_chars = 0
    try:
        raw_bytes = Path(task['text_path']).read_bytes()
        paper = raw_bytes.decode('utf-8', errors='replace')
        row['text_sha'] = hashlib.sha256(raw_bytes).hexdigest()
        if len(paper) > cfg.max_chars:
            paper = paper[:cfg.max_chars] + '\n[truncated]'
            row['truncated'] = True
        # ONE task string for both arms: the review rules, then the paper.
        task_text = system + '\n\n' + user_template.replace('{PAPER_TEXT}', paper)
        task_chars = row['task_chars'] = len(task_text)
        out = (direct_call if task['arm'] == 'direct' else harness_call)(
            task_text, api_key, cfg, acct)
        row.update(usage=out['usage'], usage_missing=out['usage_missing'],
                   n_steps=out['n_steps'], other_events=out['other_events'],
                   thinking_chars=out['thinking_chars'], served_model=out['served_model'],
                   session_id=out['session_id'], llm_retries=out['llm_retries'],
                   bad_stream_lines=out['bad_stream_lines'])
        review, coerced = validate_review(extract_review(out['text']))
        row.update(review)
        row['coerced'] = coerced
        row['parsed_ok'] = True
    except Exception as exc:  # API failure, harness failure, unparseable answer, bad file
        row['error'] = f'{type(exc).__name__}: {str(exc)[:400]}'
    now = time.time()
    row['failed_attempts'] = acct['failed_attempts']
    row['latency_s'] = round(time.monotonic() - t0, 1)
    row['ts'] = datetime.fromtimestamp(now, tz=timezone.utc).isoformat(timespec='seconds')
    row['peak'] = is_peak(now)
    flat = flat_estimate_cny(task_chars, now)
    if not row['usage_missing']:
        row['cost_cny_est'] = round(cost_cny(row['usage'], now), 5)
    # The guard must not under-count: unreported usage, every failed attempt and every
    # dsh-internal retry is charged a generous flat estimate on top of what was reported.
    guard = (row['cost_cny_est'] if row['cost_cny_est'] is not None else flat) \
        + flat * (row['failed_attempts'] + row['llm_retries'])
    if out is None and not row['failed_attempts']:
        guard = flat if task_chars else 0.0
    row['guard_cny'] = round(guard, 5)
    budget.add(row['guard_cny'])

    if out is not None:
        try:
            raw_dir = Path(cfg.out_dir) / task['arm']
            raw_dir.mkdir(parents=True, exist_ok=True)
            stem = f"{task['safe_id']}v{task['v']}_r{task['rep']}"
            if out['raw'] is not None:  # harness: the full event stream
                raw_path = raw_dir / f'{stem}.events.jsonl'
                body = out['raw']
            else:
                raw_path = raw_dir / f'{stem}.json'
                body = json.dumps({'text': out['text'], 'usage': out['usage'],
                                   'thinking_chars': out['thinking_chars']}, indent=1)
            with open(raw_path, 'w', encoding='utf-8', errors='backslashreplace') as f:
                f.write(body)
            row['raw_path'] = str(raw_path)
        except OSError as exc:
            row['raw_path'] = f'(not written: {type(exc).__name__})'
    try:
        with LOG_LOCK:
            append_jsonl(cfg.log, row)
    except Exception as exc:  # a full disk must not take the pool down silently
        print(f'\nLOG WRITE FAILED for {row["key"]}: {exc}', flush=True)
        STOP.set()
    budget.note(row['parsed_ok'], cfg.abort_after)
    return row


# ---------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser(description='E1 with and without DeepSeek Harness.')
    ap.add_argument('--manifest', default='data/processed/pilot_manifest.jsonl')
    ap.add_argument('--texts', default='data/processed/texts/')
    ap.add_argument('--prompt', default='src/review/prompts/review_v1.md')
    ap.add_argument('--patch', default='src/review/harness_lockdown.yml')
    ap.add_argument('--log', default='data/processed/e1h_reviews_log.jsonl')
    ap.add_argument('--out-dir', default='data/processed/reviews_e1h/')
    ap.add_argument('--arms', default='direct,harness')
    ap.add_argument('--model', default='deepseek-flash')
    ap.add_argument('--effort', default='high',
                    help='must equal what the harness sends (its adapter default is high)')
    ap.add_argument('--max-tokens', type=int, default=256000)
    ap.add_argument('--base-url', default='https://api.deepseek.com/anthropic')
    ap.add_argument('--api-key-env', default='DEEPSEEK_API_KEY')
    ap.add_argument('--workspace', default='/private/tmp/ws-e1h',
                    help='empty working directory for the harness; a neutral path, because '
                         'the harness shows the model its cwd')
    ap.add_argument('--reps', type=int, default=3)
    ap.add_argument('--sample-papers', type=int, default=0)
    ap.add_argument('--workers', type=int, default=12)
    ap.add_argument('--timeout', type=int, default=900,
                    help='direct arm: seconds without a byte before the read gives up')
    ap.add_argument('--harness-timeout', type=int, default=600,
                    help='harness arm: total seconds for one dsh run (smoke max was 130)')
    ap.add_argument('--max-chars', type=int, default=380000)
    ap.add_argument('--budget-cny', type=float, default=120.0,
                    help='total guard spend allowed for this log, across invocations')
    ap.add_argument('--abort-after', type=int, default=12,
                    help='stop after this many consecutive failed runs')
    ap.add_argument('--wait-offpeak', action='store_true',
                    help='sleep until DeepSeek off-peak pricing before starting')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--seed', type=int, default=42)
    cfg = ap.parse_args()

    arms = [a.strip() for a in cfg.arms.split(',') if a.strip()]
    if not arms or any(a not in ARMS for a in arms):
        ap.error(f'--arms must be a comma list drawn from {ARMS}')
    api_key = load_api_key(cfg.api_key_env, mock=False)
    system, user_template = load_prompt(cfg.prompt)

    meta = {'prompt_sha': hashlib.sha256(Path(cfg.prompt).read_bytes()).hexdigest(),
            'harness_version': None, 'patch_sha': None, 'config_sha': None}
    if 'harness' in arms:
        ws = Path(cfg.workspace)
        ws.mkdir(parents=True, exist_ok=True)
        if any(ws.iterdir()):
            sys.exit(f'harness workspace {ws} is not empty — refusing to run')
        for d in ('fakehome', 'dsh-home'):
            (TOOLS / d).mkdir(parents=True, exist_ok=True)
        ver = subprocess.run([dsh_bin(), '--version'], capture_output=True, text=True,
                             env=harness_env(''), cwd=cfg.workspace, timeout=120)
        if ver.returncode != 0:
            sys.exit(f'dsh --version failed: {ver.stderr.strip()[:300]}')
        meta['harness_version'] = ver.stdout.strip()
        meta['patch_sha'] = hashlib.sha256((REPO_ROOT / cfg.patch).read_bytes()).hexdigest()
        meta['config_sha'] = assert_lockdown(cfg)
    # rows only count as done when written under this exact configuration
    meta['cfg_sig'] = hashlib.sha256(json.dumps(
        [meta['prompt_sha'], meta['patch_sha'], meta['harness_version'], meta['config_sha'],
         cfg.model, cfg.effort, cfg.max_tokens, cfg.max_chars]).encode()).hexdigest()[:16]

    prior = list(read_jsonl(cfg.log)) if Path(cfg.log).exists() else []
    done = {r['key'] for r in prior if r.get('parsed_ok') and r.get('cfg_sig') == meta['cfg_sig']}
    spent_before = sum(r.get('guard_cny') or r.get('cost_cny_est') or 0.0 for r in prior)

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
            for arm in arms:
                for rep in range(cfg.reps):
                    key = f"{paper['arxiv_id']}v{v}|{arm}|r{rep}"
                    if key in done:
                        n_skipped += 1
                        continue
                    tasks.append({'key': key, 'arxiv_id': paper['arxiv_id'], 'safe_id': safe_id,
                                  'v': v, 'rep': rep, 'arm': arm, 'text_path': str(text_path)})
    random.Random(cfg.seed).shuffle(tasks)  # interleave arms, versions and papers in time
    if cfg.limit > 0:
        tasks = tasks[:cfg.limit]
    print(f"papers: {len(papers)} | tasks: {len(tasks)} new ({dict(Counter(t['arm'] for t in tasks))}), "
          f"{n_skipped} already ok | missing texts: {n_missing} | model={cfg.model} "
          f"effort={cfg.effort} | dsh={meta['harness_version']} | cfg_sig={meta['cfg_sig']} "
          f"| guard spend so far ¥{spent_before:.2f} of cap ¥{cfg.budget_cny}", flush=True)

    if cfg.wait_offpeak and is_peak(time.time()):
        print('peak pricing now — waiting for off-peak ...', flush=True)
        while is_peak(time.time()):
            time.sleep(30)
        print('off-peak; starting.', flush=True)

    budget = Budget(cfg.budget_cny, spent_before)
    balance_before = account_balance_cny(api_key)
    print(f'account balance before: ¥{balance_before}', flush=True)
    stats: dict = {a: Counter() for a in arms}
    hist: dict = {a: Counter() for a in arms}
    n_not_started = 0
    pbar = tqdm(total=len(tasks), desc='e1h', unit='run', mininterval=30)
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
            s = stats[row['arm']]
            s['runs'] += 1
            s['ok'] += bool(row['parsed_ok'])
            s['in'] += row['usage'].get('input', 0)
            s['hit'] += row['usage'].get('cache_read', 0)
            s['out'] += row['usage'].get('output', 0)
            s['no_usage'] += bool(row['usage_missing'])
            s['multi_step'] += bool((row['n_steps'] or 0) > 1)
            s['agent_events'] += bool(row['other_events'])
            s['retries'] += row['llm_retries'] + row['failed_attempts']
            if row['parsed_ok']:
                hist[row['arm']][row['rating']] += 1
    except BaseException:
        STOP.set()  # stop launching; in-flight runs finish and log their own rows
        print('\ninterrupted — cancelling queued tasks, waiting for in-flight runs ...',
              flush=True)
        raise
    finally:
        ex.shutdown(wait=True, cancel_futures=True)
        pbar.close()

    print('\n=== summary ===')
    for a in arms:
        s = stats[a]
        print(f"{a}: runs {s['runs']} | parsed_ok {s['ok']} | tokens in {s['in']:,} "
              f"cache-hit {s['hit']:,} out {s['out']:,} | usage unknown {s['no_usage']} "
              f"| retries {s['retries']} | multi-step {s['multi_step']} "
              f"| runs with tool/agent events {s['agent_events']}")
        print('   ratings: ' + ', '.join(f'{r}: {hist[a].get(r, 0)}' for r in ALLOWED_RATINGS))
    balance_after = account_balance_cny(api_key)
    real = (f'¥{balance_before - balance_after:.2f}'
            if balance_before is not None and balance_after is not None else 'unknown')
    print(f"guard spend for this log (conservative): ¥{budget.spent:.2f} of cap "
          f"¥{cfg.budget_cny} | real spend this invocation by account balance: {real} "
          f"(balance now ¥{balance_after}; may lag by a few minutes)")
    if n_not_started:
        why = 'aborted' if STOP.is_set() else 'budget cap reached'
        print(f'{n_not_started} tasks were NOT started ({why}); re-run to resume.')
    print(f'log: {cfg.log} | raw: {cfg.out_dir}')


if __name__ == '__main__':
    main()
