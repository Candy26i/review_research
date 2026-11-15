"""E1-H ablation: which part of what DeepSeek Harness adds lowers the review scores?

E1-H found that the same review request scores about half a point lower when it goes
through `dsh` than as a bare API call, with no tool use. `dsh` adds three things to the
request content (captured on the wire with the final lock-down patch, saved in
src/review/dsh_request_template.json):

  system   a 906-character system prompt ("You are a coding agent ...")
  tools    10 tool definitions
  ctx      a 599-character runtime-context text block placed after the paper

Arms, all on the same task text (review rules + one paper version), all with the sampling
parameters of E1-H (deepseek-flash, thinking enabled, effort high, max_tokens 256000,
streaming, no temperature):

  bare    plain API call, identical to the `direct` arm of E1-H (concurrent baseline)
  sys     bare + the dsh system prompt
  tools   bare + the dsh tool definitions
  ctx     bare + the runtime-context block after the paper
  replay  bare + all three, i.e. the content of the request dsh sends, sent by this script
  dsh     the real harness (same code path as the `harness` arm of E1-H)

`replay` does not copy dsh's request metadata (harness headers, plugin list, session-log
delivery); this script identifies itself as an ordinary API client. If `replay` matches
`dsh`, the request content explains the harness effect; if not, the remainder is the
metadata or something else dsh does.

A concurrent `bare` arm is required because E1-H showed that ratings of the same
paper-version drift between run sessions. The queue is blocked by paper-version: the
blocks are shuffled and the arms are shuffled inside each block, so all arms of a
paper-version run within a minute or two of each other and an interruption leaves whole
blocks. `bare` enters most contrasts and is the cheapest arm, so it gets two replicates;
every other arm gets one.

Retries inside a run follow E1-H's direct arm (transport errors and empty answers are
retried); every failed attempt is logged with its reason in `attempt_errors`.

DeepSeek caches prompt prefixes. Left alone, `sys` and `tools` would always miss the cache
(their prefix is new) while the other arms hit caches left by E1-H, so cache state would
be confounded with arm. Before the queue starts, every (paper-version, request prefix) is
therefore sent once with max_tokens 8, which builds the cache; the real runs then all read
their input from the cache. Each row records how much of its input came from the cache.

No tool is ever executed here. If the model stops to call a tool in an API arm, the run is
logged with tool_stop = true and is not retried; the analysis counts such runs and leaves
them out of the primary set.

Run from the repo root:
    python3 src/review/run_e1_harness_ablation.py --sample-papers 2 --limit 12 --log ... --out-dir ...   # smoke
    python3 src/review/run_e1_harness_ablation.py                                                        # 142 paper-versions, 6 arms (2 bare replicates, 1 of each other arm)
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import random
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
import run_e1_harness as R  # noqa: E402
from util import read_jsonl, append_jsonl  # noqa: E402

API_ARMS = ('bare', 'sys', 'tools', 'ctx', 'replay')
ARMS = API_ARMS + ('dsh',)
# arms that share a prompt prefix up to the end of the paper share a cache entry
PREFIX_CLASS = {'bare': 'bare', 'ctx': 'bare', 'sys': 'sys', 'tools': 'tools',
                'replay': 'replay', 'dsh': 'replay'}
WARM_MAX_TOKENS = 8
WARM_SETTLE_S = 20   # a new cache entry was readable 6 s after the call in a direct test
ARM_STREAK: Counter = Counter()   # consecutive failed runs per arm
ARM_STREAK_LOCK = threading.Lock()


def build_payload(arm: str, task_text: str, cfg: argparse.Namespace, tmpl: dict) -> dict:
    """`bare` must stay identical to the payload of R.direct_call."""
    content = [{"type": "text", "text": task_text}]
    if arm in ('ctx', 'replay'):
        content.append({"type": "text", "text": tmpl['context_block']})
    payload = {
        "model": cfg.model, "stream": True, "max_tokens": cfg.max_tokens,
        "thinking": {"type": "enabled"}, "output_config": {"effort": cfg.effort},
        "messages": [{"role": "user", "content": content}],
    }
    if arm in ('sys', 'replay'):
        payload["system"] = tmpl['system']
    if arm in ('tools', 'replay'):
        payload["tools"] = tmpl['tools']
    return payload


def api_call(arm: str, task_text: str, api_key: str, cfg: argparse.Namespace,
             acct: dict, tmpl: dict) -> dict:
    """R.direct_call with a per-arm payload; also keeps the reasoning text and reports a
    stop for a tool call instead of raising."""
    url = cfg.base_url.rstrip('/') + '/v1/messages'
    payload = build_payload(arm, task_text, cfg, tmpl)
    headers = {"x-api-key": api_key, "anthropic-version": "2023-06-01",
               "content-type": "application/json", "accept": "text/event-stream"}
    last_err, t_att = 'unknown error', time.monotonic()
    for attempt in range(R.MAX_ATTEMPTS):
        if attempt:
            acct['failed_attempts'] += 1
            acct.setdefault('attempt_errors', []).append(
                f'{last_err} after {time.monotonic() - t_att:.0f}s')
            if R.STOP.wait(min(10.0 * attempt, 60)):
                break
        t_att = time.monotonic()
        try:
            with requests.post(url, json=payload, headers=headers, stream=True,
                               timeout=(30, cfg.timeout)) as resp:
                if resp.status_code in R.RETRY_STATUSES:
                    last_err = f'HTTP {resp.status_code}'
                    continue
                if resp.status_code != 200:
                    raise RuntimeError(f'HTTP {resp.status_code}: {resp.text[:300]}')
                text: list[str] = []
                thinking: list[str] = []
                tool_calls: list[str] = []
                tool_json: list[str] = []
                bad_lines = 0
                usage: dict = {}
                stop = served = stream_error = None
                for raw in resp.iter_lines():
                    if not raw.startswith(b'data:'):
                        continue
                    try:
                        ev = json.loads(raw[5:].decode('utf-8', 'replace').strip())
                    except ValueError:
                        bad_lines += 1
                        continue
                    if not isinstance(ev, dict):
                        bad_lines += 1
                        continue
                    kind = ev.get('type')
                    if kind == 'message_start':
                        usage.update(ev['message'].get('usage') or {})
                        served = ev['message'].get('model')
                    elif kind == 'content_block_start':
                        block = ev.get('content_block') or {}
                        if block.get('type') == 'tool_use':
                            tool_calls.append(block.get('name') or '?')
                    elif kind == 'content_block_delta':
                        d = ev.get('delta') or {}
                        if d.get('type') == 'text_delta':
                            text.append(d.get('text', ''))
                        elif d.get('type') == 'thinking_delta':
                            thinking.append(d.get('thinking', ''))
                        elif d.get('type') == 'input_json_delta':
                            tool_json.append(d.get('partial_json', ''))
                    elif kind == 'message_delta':
                        usage.update(ev.get('usage') or {})
                        stop = (ev.get('delta') or {}).get('stop_reason') or stop
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
                if stop not in ('end_turn', 'tool_use'):
                    raise RuntimeError(f'stop_reason {stop}')
                if stop == 'end_turn' and not answer.strip():
                    last_err = (f"empty answer ({sum(map(len, thinking))} reasoning chars, "
                                f"{usage.get('output_tokens')} output tokens)")
                    continue
                think = ''.join(thinking)
                return {
                    'text': answer, 'thinking': think, 'thinking_chars': len(think),
                    'stop_reason': stop, 'tool_calls': tool_calls,
                    'tool_input': ''.join(tool_json), 'served_model': served,
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
        acct['failed_attempts'] += 1
        acct.setdefault('attempt_errors', []).append(
            f'{last_err} after {time.monotonic() - t_att:.0f}s')
    raise RuntimeError(f'request failed after {attempt + 1} attempts ({last_err})')


def warm_call(prefix_class: str, task_text: str, api_key: str, cfg: argparse.Namespace,
              tmpl: dict) -> dict:
    """Build the provider-side prompt cache for one request prefix: the same request,
    cut off after a few output tokens. Returns the usage the provider reported."""
    payload = build_payload(prefix_class, task_text, cfg, tmpl)
    payload['max_tokens'] = WARM_MAX_TOKENS
    headers = {"x-api-key": api_key, "anthropic-version": "2023-06-01",
               "content-type": "application/json", "accept": "text/event-stream"}
    last_err = 'unknown error'
    for attempt in range(3):
        if attempt and R.STOP.wait(5.0 * attempt):
            break
        try:
            with requests.post(cfg.base_url.rstrip('/') + '/v1/messages', json=payload,
                               headers=headers, stream=True, timeout=(30, 300)) as resp:
                if resp.status_code in R.RETRY_STATUSES:
                    last_err = f'HTTP {resp.status_code}'
                    continue
                if resp.status_code != 200:
                    raise RuntimeError(f'HTTP {resp.status_code}: {resp.text[:300]}')
                usage: dict = {}
                for raw in resp.iter_lines():
                    if not raw.startswith(b'data:'):
                        continue
                    try:
                        ev = json.loads(raw[5:].decode('utf-8', 'replace').strip())
                    except ValueError:
                        continue
                    if ev.get('type') == 'message_start':
                        usage.update(ev['message'].get('usage') or {})
                    elif ev.get('type') == 'message_delta':
                        usage.update(ev.get('usage') or {})
                if 'input_tokens' not in usage:
                    last_err = 'no usage reported'
                    continue
                return {'input': usage.get('input_tokens', 0) or 0,
                        'cache_read': usage.get('cache_read_input_tokens', 0) or 0,
                        'output': usage.get('output_tokens', 0) or 0}
        except requests.RequestException as exc:
            last_err = f'connection error: {type(exc).__name__}'
    raise RuntimeError(f'warm-up failed ({last_err})')


def warm_task(w: dict, cfg: argparse.Namespace, api_key: str, system: str,
              user_template: str, meta: dict, budget: R.Budget, tmpl: dict) -> dict | None:
    """One warm-up call, logged like a run so its cost counts toward the budget guard."""
    if R.STOP.is_set() or budget.exhausted():
        return None
    row = {'key': w['key'], 'arxiv_id': w['arxiv_id'], 'v': w['v'], 'arm': w['arm'],
           'parsed_ok': False, 'tool_stop': False, 'usage': {}, 'cost_cny_est': None,
           'guard_cny': 0.0, 'ts': None, 'error': None, **meta}
    task_chars = 0
    try:
        paper = Path(w['text_path']).read_bytes().decode('utf-8', errors='replace')
        if len(paper) > cfg.max_chars:
            paper = paper[:cfg.max_chars] + '\n[truncated]'
        task_text = system + '\n\n' + user_template.replace('{PAPER_TEXT}', paper)
        task_chars = len(task_text)
        row['usage'] = warm_call(w['prefix_class'], task_text, api_key, cfg, tmpl)
        row['cost_cny_est'] = round(R.cost_cny(row['usage'], time.time()), 5)
    except Exception as exc:
        row['error'] = f'{type(exc).__name__}: {str(exc)[:300]}'
    now = time.time()
    row['ts'] = datetime.fromtimestamp(now, tz=timezone.utc).isoformat(timespec='seconds')
    # a failed warm-up may still have been billed for its input
    row['guard_cny'] = round(row['cost_cny_est'] if row['cost_cny_est'] is not None else
                             R.cost_cny({'input': task_chars // 3 + 4000}, now), 5)
    budget.add(row['guard_cny'])
    try:
        with R.LOG_LOCK:
            append_jsonl(cfg.log, row)
    except Exception as exc:
        print(f'\nLOG WRITE FAILED for {row["key"]}: {exc}', flush=True)
        R.STOP.set()
    return row


def dsh_call(task_text: str, api_key: str, cfg: argparse.Namespace, acct: dict) -> dict:
    out = R.harness_call(task_text, api_key, cfg, acct)
    head = ''
    for line in out['raw'].split('\n'):
        if '"thinking"' in line:
            try:
                ev = json.loads(line)
            except ValueError:
                continue
            if isinstance(ev, dict) and ev.get('type') == 'thinking':
                head = ev.get('text') if isinstance(ev.get('text'), str) else ''
                break
    # the --json stream caps reasoning text at 8 KiB: enough for the opening, not the length
    out.update(thinking=head, stop_reason='end_turn', tool_calls=[], tool_input='')
    return out


def run_task(task: dict, cfg: argparse.Namespace, api_key: str, system: str,
             user_template: str, meta: dict, budget: R.Budget, tmpl: dict) -> dict | None:
    """Never raises. Appends its own row; returns it, or None when not started."""
    if R.STOP.is_set() or budget.exhausted():
        return None
    if not cfg.allow_peak and R.is_peak(time.time()):
        return None  # prices double and the price period would differ from earlier runs
    row = {
        'key': task['key'], 'arxiv_id': task['arxiv_id'], 'v': task['v'], 'rep': task['rep'],
        'arm': task['arm'], 'model': cfg.model, 'effort': cfg.effort,
        'rating': None, 'soundness': None, 'presentation': None, 'contribution': None,
        'confidence': None, 'accept': None, 'coerced': False, 'parsed_ok': False,
        'truncated': False, 'text_sha': None, 'task_chars': 0, 'usage': {},
        'usage_missing': True, 'n_steps': None, 'other_events': {}, 'llm_retries': 0,
        'failed_attempts': 0, 'attempt_errors': [], 'thinking_chars': None,
        'thinking_open': None,
        'stop_reason': None, 'tool_stop': False, 'tool_calls': [], 'served_model': None,
        'session_id': None, 'bad_stream_lines': 0, 'cost_cny_est': None, 'guard_cny': 0.0,
        'peak': None, 'cache_share': None, 'latency_s': None, 'wall_s': None, 'ts': None,
        'raw_path': None, 'error': None, **meta,
    }
    t0, w0 = time.monotonic(), time.time()
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
        task_text = system + '\n\n' + user_template.replace('{PAPER_TEXT}', paper)
        task_chars = row['task_chars'] = len(task_text)
        if task['arm'] == 'dsh':
            out = dsh_call(task_text, api_key, cfg, acct)
        else:
            out = api_call(task['arm'], task_text, api_key, cfg, acct, tmpl)
        row.update(usage=out['usage'], usage_missing=out['usage_missing'],
                   n_steps=out['n_steps'], other_events=out['other_events'],
                   thinking_chars=out['thinking_chars'], served_model=out['served_model'],
                   session_id=out['session_id'], llm_retries=out['llm_retries'],
                   bad_stream_lines=out['bad_stream_lines'], stop_reason=out['stop_reason'],
                   tool_calls=out['tool_calls'],
                   tool_stop=out['stop_reason'] == 'tool_use',
                   thinking_open=' '.join(out['thinking'].split())[:80])
        review, coerced = R.validate_review(R.extract_review(out['text']))
        row.update(review)
        row['coerced'] = coerced
        row['parsed_ok'] = True
    except Exception as exc:
        row['error'] = f'{type(exc).__name__}: {str(exc)[:400]}'
    now = time.time()
    row['failed_attempts'] = acct['failed_attempts']
    row['attempt_errors'] = acct.get('attempt_errors', [])
    n_in = (row['usage'].get('input', 0) or 0) + (row['usage'].get('cache_read', 0) or 0)
    row['cache_share'] = round((row['usage'].get('cache_read', 0) or 0) / n_in, 4) if n_in else None
    row['latency_s'] = round(time.monotonic() - t0, 1)
    row['wall_s'] = round(now - w0, 1)  # differs from latency_s when the machine slept
    row['ts'] = datetime.fromtimestamp(now, tz=timezone.utc).isoformat(timespec='seconds')
    row['peak'] = R.is_peak(now)
    flat = R.flat_estimate_cny(task_chars, now)
    if not row['usage_missing']:
        row['cost_cny_est'] = round(R.cost_cny(row['usage'], now), 5)
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
            if out['raw'] is not None:
                raw_path = raw_dir / f'{stem}.events.jsonl'
                body = out['raw']
            else:
                raw_path = raw_dir / f'{stem}.json'
                body = json.dumps({'text': out['text'], 'thinking': out['thinking'],
                                   'usage': out['usage'], 'stop_reason': out['stop_reason'],
                                   'tool_calls': out['tool_calls'],
                                   'tool_input': out.get('tool_input', '')}, indent=1)
            with open(raw_path, 'w', encoding='utf-8', errors='backslashreplace') as f:
                f.write(body)
            row['raw_path'] = str(raw_path)
        except OSError as exc:
            row['raw_path'] = f'(not written: {type(exc).__name__})'
    try:
        with R.LOG_LOCK:
            append_jsonl(cfg.log, row)
    except Exception as exc:
        print(f'\nLOG WRITE FAILED for {row["key"]}: {exc}', flush=True)
        R.STOP.set()
    # a stop for a tool call is an outcome, not a technical failure
    ok = bool(row['parsed_ok'] or row['tool_stop'])
    budget.note(ok, cfg.abort_after)
    # the global streak is reset by other arms' successes, so one broken arm needs its own
    with ARM_STREAK_LOCK:
        ARM_STREAK[row['arm']] = 0 if ok else ARM_STREAK[row['arm']] + 1
        broken = ARM_STREAK[row['arm']] >= cfg.abort_after
    if broken and not R.STOP.is_set():
        R.STOP.set()
        print(f"\nABORT: {cfg.abort_after} consecutive failed runs in arm {row['arm']} — "
              f"stopping.", flush=True)
    return row


def finished(r: dict, cfg_sig: str) -> bool:
    return r.get('cfg_sig') == cfg_sig and bool(r.get('parsed_ok') or r.get('tool_stop'))


def main() -> None:
    ap = argparse.ArgumentParser(description='E1-H ablation of the harness request content.')
    ap.add_argument('--manifest', default='data/processed/pilot_manifest.jsonl')
    ap.add_argument('--texts', default='data/processed/texts/')
    ap.add_argument('--prompt', default='src/review/prompts/review_v1.md')
    ap.add_argument('--patch', default='src/review/harness_lockdown.yml')
    ap.add_argument('--template', default='src/review/dsh_request_template.json')
    ap.add_argument('--log', default='data/processed/e1h_ablation_log.jsonl')
    ap.add_argument('--out-dir', default='data/processed/reviews_e1h_ablation/')
    ap.add_argument('--arms', default=','.join(ARMS))
    ap.add_argument('--model', default='deepseek-flash')
    ap.add_argument('--effort', default='high')
    ap.add_argument('--max-tokens', type=int, default=256000)
    ap.add_argument('--base-url', default='https://api.deepseek.com/anthropic')
    ap.add_argument('--api-key-env', default='DEEPSEEK_API_KEY')
    ap.add_argument('--workspace', default='/private/tmp/ws-e1h')
    ap.add_argument('--reps', type=int, default=1)
    ap.add_argument('--bare-reps', type=int, default=2,
                    help='replicates of the bare arm, which enters most contrasts')
    ap.add_argument('--sample-papers', type=int, default=0)
    ap.add_argument('--workers', type=int, default=20)
    ap.add_argument('--timeout', type=int, default=900)
    ap.add_argument('--harness-timeout', type=int, default=600)
    ap.add_argument('--max-chars', type=int, default=380000)
    ap.add_argument('--budget-cny', type=float, default=65.0,
                    help='total guard spend allowed for this log, across invocations')
    ap.add_argument('--abort-after', type=int, default=12)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--seed', type=int, default=44)
    ap.add_argument('--allow-peak', action='store_true',
                    help='run during DeepSeek peak pricing (double price, different period)')
    ap.add_argument('--no-warm', action='store_true',
                    help='skip the cache warm-up (cache state then differs between arms)')
    cfg = ap.parse_args()

    arms = [a.strip() for a in cfg.arms.split(',') if a.strip()]
    if not arms or any(a not in ARMS for a in arms):
        ap.error(f'--arms must be a comma list drawn from {ARMS}')
    api_key = R.load_api_key(cfg.api_key_env, mock=False)
    system, user_template = R.load_prompt(cfg.prompt)
    tmpl = json.loads(Path(cfg.template).read_text(encoding='utf-8'))
    for k in ('model', 'max_tokens'):
        if tmpl['params'][k] != getattr(cfg, k):
            sys.exit(f'--{k.replace("_", "-")} differs from the captured dsh request')
    if tmpl['params']['output_config'] != {'effort': cfg.effort}:
        sys.exit('--effort differs from the captured dsh request')
    # the context block names the working directory; replay and the real harness must agree
    if f'"{cfg.workspace}"' not in tmpl['context_block']:
        sys.exit('--workspace differs from the working directory in the captured context block')

    meta = {'prompt_sha': hashlib.sha256(Path(cfg.prompt).read_bytes()).hexdigest(),
            'template_sha': hashlib.sha256(Path(cfg.template).read_bytes()).hexdigest(),
            'harness_version': None, 'patch_sha': None, 'config_sha': None}
    # The harness is probed whatever --arms says (local commands, no model call), so the
    # configuration signature, and with it what counts as finished, never depends on --arms.
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
    meta['harness_version'] = ver.stdout.strip()
    meta['patch_sha'] = hashlib.sha256((R.REPO_ROOT / cfg.patch).read_bytes()).hexdigest()
    meta['config_sha'] = R.assert_lockdown(cfg)
    if tmpl.get('dsh_version') != meta['harness_version']:
        sys.exit('the request template was captured from a different dsh version')
    meta['cfg_sig'] = hashlib.sha256(json.dumps(
        [meta['prompt_sha'], meta['template_sha'], meta['patch_sha'], meta['harness_version'],
         meta['config_sha'], cfg.model, cfg.effort, cfg.max_tokens, cfg.max_chars]
    ).encode()).hexdigest()[:16]

    meta['invocation'] = datetime.now(timezone.utc).isoformat(timespec='seconds')  # not in cfg_sig

    Path(cfg.log).parent.mkdir(parents=True, exist_ok=True)
    lock_file = open(cfg.log + '.lock', 'w')  # held until the process exits
    try:
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        sys.exit(f'another invocation is running on {cfg.log} — refusing to start')
    prior, n_bad = [], 0
    if Path(cfg.log).exists():
        raw_log = Path(cfg.log).read_bytes()
        if raw_log and not raw_log.endswith(b'\n'):  # a cut last line must not swallow the next row
            with open(cfg.log, 'ab') as f:
                f.write(b'\n')
        for line in raw_log.decode('utf-8', 'replace').split('\n'):
            if line.strip():
                try:
                    prior.append(json.loads(line))
                except ValueError:
                    n_bad += 1
    if n_bad:
        print(f'WARNING: {n_bad} unreadable line(s) in {cfg.log} were ignored', flush=True)
    done = {r['key'] for r in prior if finished(r, meta['cfg_sig'])}
    spent_before = sum(r.get('guard_cny') or 0.0 for r in prior)

    papers = R.select_papers(read_jsonl(cfg.manifest), cfg.sample_papers)
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
                for rep in range(cfg.bare_reps if arm == 'bare' else cfg.reps):
                    key = f"{paper['arxiv_id']}v{v}|{arm}|r{rep}"
                    if key in done:
                        n_skipped += 1
                        continue
                    tasks.append({'key': key, 'arxiv_id': paper['arxiv_id'], 'safe_id': safe_id,
                                  'v': v, 'rep': rep, 'arm': arm, 'text_path': str(text_path)})
    rng = random.Random(cfg.seed)
    blocks: dict = {}
    for t in tasks:
        blocks.setdefault((t['arxiv_id'], t['v']), []).append(t)
    block_order = list(blocks)
    rng.shuffle(block_order)
    tasks = []
    for k in block_order:
        rng.shuffle(blocks[k])
        tasks += blocks[k]
    if cfg.limit > 0:
        tasks = tasks[:cfg.limit]
    print(f"papers: {len(papers)} | tasks: {len(tasks)} new "
          f"({dict(Counter(t['arm'] for t in tasks))}), {n_skipped} already finished | "
          f"missing texts: {n_missing} | model={cfg.model} effort={cfg.effort} | "
          f"dsh={meta['harness_version']} | cfg_sig={meta['cfg_sig']} | guard spend so far "
          f"¥{spent_before:.2f} of cap ¥{cfg.budget_cny} | peak pricing now: "
          f"{R.is_peak(time.time())}", flush=True)
    if R.is_peak(time.time()) and not cfg.allow_peak:
        sys.exit('DeepSeek peak pricing is in effect — not starting (see --allow-peak)')

    budget = R.Budget(cfg.budget_cny, spent_before)
    balance_before = R.account_balance_cny(api_key)
    print(f'account balance before: ¥{balance_before}', flush=True)

    if not cfg.no_warm and tasks:
        warm: dict = {}
        for t in tasks:
            pc = PREFIX_CLASS[t['arm']]
            warm.setdefault((t['text_path'], pc), {
                'key': f"{t['arxiv_id']}v{t['v']}|warm:{pc}", 'arxiv_id': t['arxiv_id'],
                'v': t['v'], 'arm': f'warm:{pc}', 'prefix_class': pc,
                'text_path': t['text_path']})
        wstats: Counter = Counter()
        with ThreadPoolExecutor(max_workers=cfg.workers) as wex:
            for wrow in wex.map(lambda w: warm_task(w, cfg, api_key, system, user_template,
                                                    meta, budget, tmpl), warm.values()):
                if wrow is None:
                    wstats['not started'] += 1
                    continue
                u = wrow['usage']
                wstats['failed' if wrow['error'] else 'ok'] += 1
                wstats['already cached'] += bool(
                    u and u.get('cache_read', 0) > 0.5 * (u.get('input', 0) + u.get('cache_read', 0)))
                wstats['cost'] += wrow['cost_cny_est'] or 0.0
        print(f"cache warm-up: {len(warm)} prefixes | ok {wstats['ok']} | failed "
              f"{wstats['failed']} | not started {wstats['not started']} | already cached "
              f"{wstats['already cached']} | est ¥{wstats['cost']:.2f}", flush=True)
        if wstats['failed'] > 0.1 * len(warm):
            sys.exit('more than 10% of the warm-up calls failed — not starting the runs')
        time.sleep(WARM_SETTLE_S)
    stats: dict = {a: Counter() for a in arms}
    hist: dict = {a: Counter() for a in arms}
    n_not_started = 0
    pbar = tqdm(total=len(tasks), desc='e1h-abl', unit='run', mininterval=30)
    ex = ThreadPoolExecutor(max_workers=cfg.workers)
    try:
        futures = [ex.submit(run_task, t, cfg, api_key, system, user_template, meta, budget,
                             tmpl) for t in tasks]
        for fut in as_completed(futures):
            row = fut.result()
            pbar.update(1)
            if row is None:
                n_not_started += 1
                continue
            s = stats[row['arm']]
            s['runs'] += 1
            s['ok'] += bool(row['parsed_ok'])
            s['tool_stop'] += bool(row['tool_stop'])
            s['out'] += row['usage'].get('output', 0)
            s['retries'] += row['llm_retries'] + row['failed_attempts']
            s['cost'] += row['cost_cny_est'] or 0.0
            if row['parsed_ok'] and not row['tool_stop']:
                hist[row['arm']][row['rating']] += 1
    except BaseException:
        R.STOP.set()
        print('\ninterrupted — cancelling queued tasks, waiting for in-flight runs ...',
              flush=True)
        raise
    finally:
        ex.shutdown(wait=True, cancel_futures=True)
        pbar.close()

    print('\n=== summary ===')
    for a in arms:
        s = stats[a]
        n = sum(hist[a].values())
        mean = sum(k * c for k, c in hist[a].items()) / n if n else float('nan')
        print(f"{a}: runs {s['runs']} | parsed_ok {s['ok']} | stopped for a tool call "
              f"{s['tool_stop']} | mean rating {mean:.3f} (n={n}) | out tokens "
              f"{s['out']:,} | retries {s['retries']} | est ¥{s['cost']:.2f}")
        print('   ratings: ' + ', '.join(f'{r}: {hist[a].get(r, 0)}'
                                         for r in R.ALLOWED_RATINGS))
    balance_after = R.account_balance_cny(api_key)
    real = (f'¥{balance_before - balance_after:.2f}'
            if balance_before is not None and balance_after is not None else 'unknown')
    print(f"guard spend for this log (conservative): ¥{budget.spent:.2f} of cap "
          f"¥{cfg.budget_cny} | real spend this invocation by account balance: {real} "
          f"(balance now ¥{balance_after}; may lag by a few minutes)")
    if n_not_started:
        why = 'aborted' if R.STOP.is_set() else 'budget cap reached'
        print(f'{n_not_started} tasks were NOT started ({why}); re-run to resume.')
    print(f'log: {cfg.log} | raw: {cfg.out_dir}')


if __name__ == '__main__':
    main()
