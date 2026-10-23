"""E2: pairwise strength judgments over adjacent version pairs of the SAME paper.

For each manifest paper and each adjacent version pair (v_i, v_{i+1}) whose scrubbed texts
both exist, present the two versions as two anonymized submissions (A/B) and ask the model
which is the STRONGER submission. By default each pair runs in both presentation orders
(--orders both: once early-as-A, once early-as-B) so position bias is measurable and
removable; --orders random instead draws one seeded order per pair. Checkpointed
(append-only JSONL keyed on "key"), concurrent, resumable. --mock runs the exact same
parse/validate/log/save path with deterministic fake verdicts and no network access.

The logged "picked_version" maps the positional verdict back onto versions
('early'|'later'|'tie'): with order AB the early version is shown as Submission A, with BA
as Submission B.

Reuses run_llm_review.py's building blocks (key loading, prompt splitting, HTTP retry loop,
fence stripping, range coercion) by import; run_llm_review.py is unchanged.

Run from the repo root:
    python3 src/review/run_pairwise_review.py --mock --limit 10
    python3 src/review/run_pairwise_review.py            # real run, DeepSeek, 93 pairs x 2 orders
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / 'src' / 'collect'))
sys.path.insert(0, str(REPO_ROOT / 'src' / 'review'))
from util import append_jsonl, done_keys, read_jsonl  # noqa: E402
from run_llm_review import (  # noqa: E402  — reused building blocks, not duplicated
    PARSE_RETRY_LINE, PRICE_IN_PER_M, PRICE_OUT_PER_M,
    coerce_to_range, http_chat, load_api_key, load_prompt, strip_fences,
)

ORDERS = ('AB', 'BA')  # AB = early version shown as Submission A, later as Submission B
BETTER_VALUES = ('A', 'B', 'tie')
MARGIN_VALUES = ('slight', 'clear', 'decisive')
TRUNCATION_MARKER = '\n[truncated]'


# ---------------------------------------------------------------- verdict mapping

def picked_version(better: str, order: str) -> str:
    """Map the positional verdict (A|B|tie) back onto versions: 'early'|'later'|'tie'.
    With order 'AB' the early version is Submission A; with 'BA' it is Submission B."""
    if better == 'tie':
        return 'tie'
    return 'early' if (better == 'A') == (order == 'AB') else 'later'


def validate_verdict(obj) -> tuple[dict, bool]:
    """Normalize the model's pairwise JSON. Raises on a missing/invalid 'better' or a
    missing/non-numeric 'confidence'; an unknown 'margin' is coerced to 'slight'."""
    if not isinstance(obj, dict):
        raise ValueError('response is not a JSON object')
    coerced = False
    better_raw = str(obj['better']).strip()
    better = {'a': 'A', 'b': 'B', 'tie': 'tie'}.get(better_raw.lower())
    if better is None:
        raise ValueError(f"'better' not in {BETTER_VALUES}: {better_raw[:40]!r}")
    coerced |= better != better_raw
    margin = str(obj.get('margin', '')).strip().lower()
    if margin not in MARGIN_VALUES:
        margin = 'slight'
        coerced = True
    confidence, c = coerce_to_range(obj['confidence'], 1, 5)
    coerced |= c
    reason = obj.get('reason')
    reason = str(reason).strip() if reason is not None else ''
    return {'better': better, 'margin': margin, 'confidence': confidence,
            'reason': reason}, coerced


# ---------------------------------------------------------------- prompt / texts

def build_pair_messages(system: str, user_template: str, text_a: str, text_b: str) -> list[dict]:
    """Substitute both slots in ONE pass, so slot-like strings inside a paper text are
    never themselves substituted."""
    user_text = re.sub(r'\{PAPER_A\}|\{PAPER_B\}',
                       lambda m: text_a if m.group() == '{PAPER_A}' else text_b,
                       user_template)
    return [{'role': 'system', 'content': system},
            {'role': 'user', 'content': user_text}]


def load_text(path: str | Path, max_chars: int) -> tuple[str, bool, str]:
    """Read a scrubbed version text; truncate to max_chars with a marker when over."""
    raw = Path(path).read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    text = raw.decode('utf-8', errors='replace')
    truncated = len(text) > max_chars
    if truncated:
        text = text[:max_chars] + TRUNCATION_MARKER
    return text, truncated, sha


# ---------------------------------------------------------------- mock

def mock_pairwise_response(task: dict, model: str, n_prompt_chars: int) -> dict:
    """Deterministic fake chat body keyed on the task hash. Covers better=A/B/tie (so the
    picked_version mapping is exercised in both orders), all margins plus one invalid
    margin ('somewhat' -> coerced to 'slight'), varying confidence, and ~1/10 fenced
    output (exercises fence stripping)."""
    h = int(hashlib.sha256(('pairwise|' + task['key']).encode()).hexdigest(), 16)
    better = BETTER_VALUES[h % 3]
    margin = (MARGIN_VALUES + ('somewhat',))[(h >> 8) % 4]
    verdict = {
        'better': better,
        'margin': margin,
        'confidence': 1 + (h >> 16) % 5,
        'reason': f"Mock verdict ({'tie' if better == 'tie' else 'Submission ' + better}): "
                  "one version has clearer ablations in Section 4 and tighter claims.",
    }
    content = json.dumps(verdict)
    if int(hashlib.sha256(('fence|' + task['key']).encode()).hexdigest(), 16) % 10 == 0:
        content = f"```json\n{content}\n```"
    n_in = n_prompt_chars // 4
    return {'id': f"mock-{hashlib.sha256(task['key'].encode()).hexdigest()[:12]}",
            'model': model, 'mock': True,
            'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': content},
                         'finish_reason': 'stop'}],
            'usage': {'prompt_tokens': n_in, 'completion_tokens': 120,
                      'total_tokens': n_in + 120}}


# ---------------------------------------------------------------- per-task worker

def run_task(task: dict, cfg: argparse.Namespace, api_key: str | None,
             system: str, user_template: str, prompt_sha: str) -> dict:
    text_early, trunc_early, sha_early = load_text(task['path_from'], cfg.max_chars_each)
    text_later, trunc_later, sha_later = load_text(task['path_to'], cfg.max_chars_each)
    if task['order'] == 'AB':
        text_a, text_b = text_early, text_later
        trunc_a, trunc_b = trunc_early, trunc_later
        sha_a, sha_b = sha_early, sha_later
    else:
        text_a, text_b = text_later, text_early
        trunc_a, trunc_b = trunc_later, trunc_early
        sha_a, sha_b = sha_later, sha_early

    row = {
        'key': task['key'], 'arxiv_id': task['arxiv_id'],
        'v_from': task['v_from'], 'v_to': task['v_to'],
        'order': task['order'], 'rep': task['rep'],
        'model': cfg.model, 'temperature': cfg.temperature,
        'better': None, 'picked_version': None, 'margin': None, 'confidence': None,
        'reason': None, 'coerced': False, 'parsed_ok': False, 'parse_retries': 0,
        'truncated_a': trunc_a, 'truncated_b': trunc_b,
        'prompt_sha': prompt_sha, 'text_sha_a': sha_a, 'text_sha_b': sha_b,
        'usage': {}, 'latency_s': None, 'raw_path': None, 'error': None,
    }

    messages = build_pair_messages(system, user_template, text_a, text_b)
    body = None
    t0 = time.monotonic()
    try:
        for parse_attempt in range(2):  # initial call + at most ONE parse re-request
            if parse_attempt == 1:
                row['parse_retries'] = 1
                messages = build_pair_messages(system + '\n' + PARSE_RETRY_LINE,
                                               user_template, text_a, text_b)
            if cfg.mock:
                body = mock_pairwise_response(task, cfg.model, len(text_a) + len(text_b))
            else:
                body = http_chat(cfg, api_key, messages)
            content = body['choices'][0]['message']['content']
            row['usage'] = body.get('usage', {}) or {}
            try:
                verdict, coerced = validate_verdict(json.loads(strip_fences(content)))
            except (ValueError, KeyError, TypeError) as exc:
                row['error'] = f"parse/validate failed: {type(exc).__name__}: {exc}"
                continue
            row.update(verdict)
            row['picked_version'] = picked_version(verdict['better'], task['order'])
            row['coerced'] = coerced
            row['parsed_ok'] = True
            row['error'] = None
            break
    except Exception as exc:  # HTTP 400, exhausted retries, malformed body, ...
        row['error'] = str(exc)[:500]
    row['latency_s'] = round(time.monotonic() - t0, 3)

    if body is not None:
        raw_dir = Path(cfg.out_dir) / f"{cfg.model}-pairwise"
        raw_dir.mkdir(parents=True, exist_ok=True)
        raw_path = raw_dir / (f"{task['safe_id']}_v{task['v_from']}v{task['v_to']}"
                              f"_{task['order']}_t{cfg.temperature}_r{task['rep']}.json")
        raw_path.write_text(json.dumps(body, ensure_ascii=False, indent=2), encoding='utf-8')
        row['raw_path'] = str(raw_path)
    return row


# ---------------------------------------------------------------- task building

def build_tasks(cfg: argparse.Namespace, papers: list[dict],
                done: set) -> tuple[list[dict], int, int]:
    """One task per paper x adjacent version pair (both texts present) x order x rep."""
    tasks, n_skipped, n_missing_pairs = [], 0, 0
    texts_dir = Path(cfg.texts)
    for paper in papers:
        safe_id = paper['arxiv_id'].replace('/', '_')
        versions = sorted(paper['versions'], key=lambda ver: ver['v'])
        for ver_lo, ver_hi in zip(versions, versions[1:]):
            v_from, v_to = ver_lo['v'], ver_hi['v']
            path_from = texts_dir / safe_id / f"v{v_from}.md"
            path_to = texts_dir / safe_id / f"v{v_to}.md"
            if not (path_from.exists() and path_to.exists()):
                n_missing_pairs += 1
                continue
            if cfg.orders == 'both':
                orders = list(ORDERS)
            else:  # 'random': one seeded order per pair, stable across runs/resumes
                pair_h = int(hashlib.sha256(
                    f"{cfg.seed}|{paper['arxiv_id']}|v{v_from}v{v_to}".encode()
                ).hexdigest(), 16)
                orders = [ORDERS[pair_h % 2]]
            for order in orders:
                for rep in range(cfg.reps):
                    key = (f"{paper['arxiv_id']}|v{v_from}v{v_to}|{order}|{cfg.model}"
                           f"|t{cfg.temperature}|r{rep}")
                    if key in done:
                        n_skipped += 1
                        continue
                    tasks.append({'key': key, 'arxiv_id': paper['arxiv_id'],
                                  'safe_id': safe_id, 'v_from': v_from, 'v_to': v_to,
                                  'order': order, 'rep': rep,
                                  'path_from': str(path_from), 'path_to': str(path_to)})
    return tasks, n_skipped, n_missing_pairs


# ---------------------------------------------------------------- summaries

def pct(part: int, whole: int) -> str:
    return f"{part}/{whole} ({part / whole * 100:.1f}%)" if whole else f"{part}/0 (n/a)"


def summarize_consistency(log_path: str, model: str, temperature: float) -> None:
    """Position-robustness over the WHOLE log (not just this run): among (pair, rep)
    groups with both orders parsed, the share where the two orders picked the same
    VERSION (tie==tie counts as agreement; the both-tie count is also shown)."""
    rows = [r for r in read_jsonl(log_path)
            if r.get('parsed_ok') and r.get('model') == model
            and r.get('temperature') == temperature]
    groups: dict[tuple, dict] = {}
    for r in rows:
        g = (r['arxiv_id'], r['v_from'], r['v_to'], r['rep'])
        groups.setdefault(g, {})[r['order']] = r['picked_version']
    both = [g for g in groups.values() if 'AB' in g and 'BA' in g]
    agree = sum(1 for g in both if g['AB'] == g['BA'])
    both_tie = sum(1 for g in both if g['AB'] == g['BA'] == 'tie')
    print(f"position-robustness (whole log, this model/temp): orders agree on the same "
          f"version in {pct(agree, len(both))} of (pair, rep) groups with both orders done"
          f" [both-tie: {both_tie}]")


def main() -> None:
    ap = argparse.ArgumentParser(
        description='E2: pairwise strength judgments over adjacent version pairs of the same paper.')
    ap.add_argument('--manifest', default='data/processed/pilot_manifest.jsonl')
    ap.add_argument('--texts', default='data/processed/texts/')
    ap.add_argument('--out-dir', default='data/processed/reviews/')
    ap.add_argument('--log', default='data/processed/pairwise_log.jsonl',
                    help='append-only output log (also the checkpoint, keyed on "key")')
    ap.add_argument('--prompt', default='src/review/prompts/pairwise_v1.md')
    ap.add_argument('--model', default='deepseek-chat')
    ap.add_argument('--base-url', default='https://api.deepseek.com')
    ap.add_argument('--api-key-env', default='DEEPSEEK_API_KEY')
    ap.add_argument('--reps', type=int, default=1)
    ap.add_argument('--temperature', type=float, default=1.0)
    ap.add_argument('--max-tokens', type=int, default=2000,
                    help='completion cap (pass 16000 for deepseek-reasoner runs)')
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--limit', type=int, default=0, help='max API calls this run (0 = no cap)')
    ap.add_argument('--max-chars-each', type=int, default=190000,
                    help='per-paper char cap (two papers must share the 128K-token context)')
    ap.add_argument('--orders', choices=('both', 'random'), default='both',
                    help="both = each pair once as early-as-A and once swapped; "
                         "random = one seeded order per pair")
    ap.add_argument('--mock', action='store_true', help='no network; deterministic fake verdicts')
    ap.add_argument('--seed', type=int, default=42)
    cfg = ap.parse_args()

    api_key = load_api_key(cfg.api_key_env, cfg.mock)
    system, user_template = load_prompt(cfg.prompt)
    if 'json' not in system.lower():
        sys.exit("Prompt system message must mention 'json' (required by response_format json_object).")
    if '{PAPER_A}' not in user_template or '{PAPER_B}' not in user_template:
        sys.exit('Prompt user message must contain both {PAPER_A} and {PAPER_B} slots.')
    prompt_sha = hashlib.sha256(Path(cfg.prompt).read_bytes()).hexdigest()

    papers = read_jsonl(cfg.manifest)
    done = done_keys(cfg.log, 'key')
    tasks, n_skipped, n_missing_pairs = build_tasks(cfg, papers, done)

    random.Random(cfg.seed).shuffle(tasks)  # blinding against order effects
    n_capped = 0
    if cfg.limit > 0 and len(tasks) > cfg.limit:
        n_capped = len(tasks) - cfg.limit
        tasks = tasks[:cfg.limit]

    print(f"papers: {len(papers)} | orders={cfg.orders} | tasks: {len(tasks)} new, "
          f"{n_skipped} checkpointed, {n_capped} over --limit | pairs missing a text: "
          f"{n_missing_pairs} | model={cfg.model} t={cfg.temperature} reps={cfg.reps} "
          f"mock={cfg.mock}")

    lock = threading.Lock()
    picked: Counter = Counter()
    better_counts: Counter = Counter()
    n_failed = n_parsed = n_coerced = n_trunc = 0
    tok_in = tok_out = 0

    pbar = tqdm(total=len(tasks), desc='pairwise', unit='call')
    with ThreadPoolExecutor(max_workers=cfg.workers) as ex:
        futures = [ex.submit(run_task, t, cfg, api_key, system, user_template, prompt_sha)
                   for t in tasks]
        for fut in as_completed(futures):
            row = fut.result()
            with lock:
                append_jsonl(cfg.log, row)
                if row['error'] is not None or not row['parsed_ok']:
                    n_failed += 1
                if row['parsed_ok']:
                    n_parsed += 1
                    picked[row['picked_version']] += 1
                    better_counts[row['better']] += 1
                n_coerced += bool(row['coerced'])
                n_trunc += bool(row['truncated_a'] or row['truncated_b'])
                usage = row.get('usage') or {}
                tok_in += usage.get('prompt_tokens', 0) or 0
                tok_out += usage.get('completion_tokens', 0) or 0
                pbar.update(1)
    pbar.close()

    made = len(tasks)
    print("\n=== summary ===")
    print(f"calls made: {made} | skipped (checkpointed): {n_skipped} | failed: {n_failed} "
          f"| parsed_ok: {pct(n_parsed, made)} | coerced: {n_coerced} "
          f"| rows with truncation: {n_trunc}")
    print("picked_version shares (parsed rows this run): "
          + " | ".join(f"{which}: {pct(picked.get(which, 0), n_parsed)}"
                       for which in ('later', 'early', 'tie')))
    print(f"position share P(better==A): {pct(better_counts.get('A', 0), n_parsed)}")
    summarize_consistency(cfg.log, cfg.model, cfg.temperature)
    cost = tok_in / 1e6 * PRICE_IN_PER_M + tok_out / 1e6 * PRICE_OUT_PER_M
    print(f"tokens: prompt={tok_in:,} completion={tok_out:,} | approx cost = ${cost:.4f} "
          f"(deepseek-chat list price; approximate)")
    print(f"log: {cfg.log} | raw: {Path(cfg.out_dir) / (cfg.model + '-pairwise')}")


if __name__ == '__main__':
    main()
