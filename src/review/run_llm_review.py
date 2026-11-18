"""E0: score paper versions with an OpenAI-compatible chat API (DeepSeek first).

For each selected paper x version x repetition, send the scrubbed anonymized text to the
model with a fixed review prompt and log the structured scores. Checkpointed (append-only
JSONL keyed on "key"), concurrent, cost-tracked, resumable. --mock runs the exact same
parse/validate/log/save path without any network access.

Multimodal arm: with --images-dir set, each task also attaches up to --max-images figure
PNGs from {images-dir}/{safe_id}/v{v}/fig_*.png (sorted, 12 MB base64 budget) as
data-URI image parts after the text part — OpenAI-compatible vision format, which the
Gemini endpoint (https://generativelanguage.googleapis.com/v1beta/openai) also accepts.
Tasks with images use the key format "{arxiv_id}v{v}|{model}|t{temperature}|img|r{rep}";
text-only keys keep the exact historical format so old checkpoint rows stay valid.

Run from the repo root:
    python3 src/review/run_llm_review.py --mock --sample-papers 4 --reps 2
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import random
import re
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from tqdm import tqdm

sys.path.insert(0, 'src/collect')
from util import read_jsonl, append_jsonl, done_keys  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]

ALLOWED_RATINGS = [1, 3, 5, 6, 8, 10]
SCORE_FIELDS = ("rating", "soundness", "presentation", "contribution", "confidence")
RETRY_STATUSES = {429, 500, 502, 503, 504}
MAX_ATTEMPTS = 4
# DeepSeek list price (USD per million tokens), for the approximate cost line only.
PRICE_IN_PER_M = 0.28
PRICE_OUT_PER_M = 0.42
# Anthropic list prices (USD per million tokens), matched by substring of --model.
ANTHROPIC_PRICES = {
    'opus-5-5': (4.00, 20.00), 'sonnet-5-5': (2.00, 10.00), 'haiku-5-5': (0.10, 0.50),
}
ANTHROPIC_EFFORTS = {'low', 'medium', 'high', 'xhigh', 'max'}

PARSE_RETRY_LINE = "Previous response was not valid JSON. Respond with the JSON object only."

# Multimodal arm: cap on the total base64 image payload per request (the whole request
# body must stay well under ~15 MB; scrubbed text adds at most ~0.4 MB on top of this).
IMAGE_B64_BUDGET = 12 * 1024 * 1024
FIGURES_NOTE = "The paper's figures are attached as images, in document order."
MOCK_TOKENS_PER_IMAGE = 258


# ---------------------------------------------------------------- prompt handling

def load_prompt(path: str | Path) -> tuple[str, str]:
    """Split the prompt file on '# System' / '# User' headings into the two messages."""
    raw = Path(path).read_text(encoding='utf-8')
    parts = re.split(r"(?m)^#\s+(System|User)\s*$", raw)
    sections: dict[str, str] = {}
    for i in range(1, len(parts) - 1, 2):
        sections[parts[i]] = parts[i + 1].strip()
    if 'System' not in sections or 'User' not in sections:
        sys.exit(f"Prompt file {path} must contain '# System' and '# User' headings.")
    return sections['System'], sections['User']


def build_messages(system: str, user_template: str, paper_text: str,
                   image_uris: list[str] | None = None) -> list[dict]:
    """Text-only: user content is a plain string (exact historical shape). With images:
    user content is an array of parts — one text part, then one image_url part per image."""
    user_text = user_template.replace("{PAPER_TEXT}", paper_text)
    if not image_uris:
        user_content: str | list[dict] = user_text
    else:
        user_content = [{"type": "text", "text": user_text + "\n\n" + FIGURES_NOTE}]
        user_content += [{"type": "image_url", "image_url": {"url": uri}} for uri in image_uris]
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user_content},
    ]


# ---------------------------------------------------------------- key loading

def load_api_key(env_name: str, mock: bool) -> str | None:
    key = os.environ.get(env_name)
    if not key:
        env_file = REPO_ROOT / '.env'
        if env_file.exists():
            for line in env_file.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                if line.startswith('export '):
                    line = line[len('export '):].lstrip()
                name, sep, value = line.partition('=')
                if sep and name.strip() == env_name:
                    key = value.strip().strip('"').strip("'")
                    break
    if not key and not mock:
        sys.exit(f"No API key: export {env_name}=<key> or put a '{env_name}=...' line in {REPO_ROOT / '.env'}")
    return key


# ---------------------------------------------------------------- paper selection

def stratum_of(row: dict) -> tuple:
    s = row.get('strata', {})
    return (s.get('decision'), s.get('version_bucket'))


def select_papers(rows: list[dict], n: int) -> list[dict]:
    """N>0: round-robin across strata in manifest order (deterministic; larger N is a
    superset of smaller N, and every stratum is represented as soon as N >= #strata)."""
    if n <= 0 or n >= len(rows):
        return rows
    groups: dict[tuple, list[dict]] = {}
    order: list[tuple] = []
    for r in rows:
        s = stratum_of(r)
        if s not in groups:
            groups[s] = []
            order.append(s)
        groups[s].append(r)
    selected: list[dict] = []
    rnd = 0
    while len(selected) < n:
        took_any = False
        for s in order:
            if rnd < len(groups[s]):
                selected.append(groups[s][rnd])
                took_any = True
                if len(selected) >= n:
                    break
        if not took_any:
            break
        rnd += 1
    return selected


# ---------------------------------------------------------------- figure images

def discover_images(images_dir: Path, safe_id: str, v, max_images: int) -> tuple[list[str], int]:
    """Pick up to max_images of {images_dir}/{safe_id}/v{v}/fig_*.png (sorted) whose
    cumulative base64 size fits IMAGE_B64_BUDGET. Trailing images beyond the budget are
    dropped (the first over-budget image and everything after it). Uses file sizes only,
    so task building stays cheap and the selection is deterministic across resumes.
    Returns (paths, n_dropped_for_budget); a missing/empty dir gives ([], 0)."""
    fig_dir = images_dir / safe_id / f"v{v}"
    if not fig_dir.is_dir():
        return [], 0
    candidates = sorted(fig_dir.glob('fig_*.png'))[:max_images]
    picked: list[str] = []
    cum = 0
    for i, p in enumerate(candidates):
        b64_len = 4 * ((p.stat().st_size + 2) // 3)  # exact base64 length for this file
        if cum + b64_len > IMAGE_B64_BUDGET:
            return picked, len(candidates) - i
        cum += b64_len
        picked.append(str(p))
    return picked, 0


def encode_image_uris(paths: list[str]) -> list[str]:
    return ['data:image/png;base64,' + base64.b64encode(Path(p).read_bytes()).decode('ascii')
            for p in paths]


# ---------------------------------------------------------------- API calls

# OpenAI-hosted gpt-5+ models renamed max_tokens and reject a temperature field. The
# first HTTP 400 that says so adapts the payload; the adaptation is remembered per
# (base_url, model) so later calls in this process start adapted. DeepSeek and Gemini
# never return these errors, so their requests are unchanged.
_PARAM_ADAPT: dict[tuple[str, str], set[str]] = {}
_PARAM_LOCK = threading.Lock()
# Set by the main loop (cost cap, run of failures, Ctrl-C): tasks that have not started
# yet return None instead of calling the API; calls already in flight are still logged.
STOP = threading.Event()


def http_chat(cfg: argparse.Namespace, api_key: str, messages: list[dict]) -> dict:
    """POST /chat/completions with retries on 429/5xx/connection; no retry on HTTP 400
    except the two OpenAI parameter adaptations (same handling as make_ai_edits)."""
    url = cfg.base_url.rstrip('/') + '/chat/completions'
    payload = {
        "model": cfg.model,
        "messages": messages,
        "temperature": cfg.temperature,
        "max_tokens": cfg.max_tokens,
        "response_format": {"type": "json_object"},
    }
    # thinking control (Gemini OpenAI-compat honors reasoning_effort; 'none' disables)
    if getattr(cfg, 'reasoning_effort', ''):
        payload["reasoning_effort"] = cfg.reasoning_effort
    if getattr(cfg, 'service_tier', ''):
        payload["service_tier"] = cfg.service_tier
    if getattr(cfg, 'prompt_cache_mode', ''):
        # OpenAI gpt-5.6+ bills implicit cache WRITES at 1.25x input; 'explicit' with no
        # breakpoints writes nothing, which is what a one-review-per-text run wants.
        payload["prompt_cache_options"] = {"mode": cfg.prompt_cache_mode}
    adapt_key = (cfg.base_url, cfg.model)
    with _PARAM_LOCK:
        adapted = set(_PARAM_ADAPT.get(adapt_key, ()))
    if 'max_completion_tokens' in adapted:
        payload['max_completion_tokens'] = payload.pop('max_tokens')
    if 'no_temperature' in adapted:
        payload.pop('temperature', None)
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    is_openai = 'api.openai.com' in cfg.base_url
    flex = getattr(cfg, 'service_tier', '') == 'flex'
    # flex: slower, and a documented 429 "resource unavailable" / 408 that needs patience
    max_attempts = 8 if flex else MAX_ATTEMPTS
    timeout = 900 if flex else 600
    retry_statuses = RETRY_STATUSES | {408} if flex else RETRY_STATUSES
    last_err = "unknown error"
    attempt = 0
    while attempt < max_attempts:
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=timeout)
        except requests.RequestException as exc:
            last_err = f"connection error: {type(exc).__name__}"
            attempt += 1
            time.sleep(5.0 * attempt)
            continue
        if resp.status_code == 429 and 'insufficient_quota' in resp.text:
            raise RuntimeError(f"HTTP 429 insufficient_quota: {resp.text[:300]}")
        if resp.status_code in retry_statuses:
            last_err = f"HTTP {resp.status_code}"
            attempt += 1
            default_wait = min(120.0, 5.0 * 2 ** (attempt - 1)) if flex else 5.0 * attempt
            try:
                wait = float(resp.headers.get('Retry-After', default_wait))
            except ValueError:
                wait = default_wait
            time.sleep(min(wait, 120))
            continue
        if resp.status_code == 400 and is_openai:
            # parameter adaptation: does not count as an attempt
            tag = None
            if 'max_completion_tokens' in resp.text and 'max_tokens' in payload:
                payload['max_completion_tokens'] = payload.pop('max_tokens')
                tag = 'max_completion_tokens'
            elif "'temperature'" in resp.text and 'temperature' in payload:
                payload.pop('temperature')
                tag = 'no_temperature'
            if tag:
                with _PARAM_LOCK:
                    _PARAM_ADAPT.setdefault(adapt_key, set()).add(tag)
                continue
        if resp.status_code != 200:
            # 400 and other non-retryable statuses: fail immediately with the body text.
            raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:500]}")
        body = resp.json()
        if is_openai:
            # keep a record of what was sent (everything except the messages)
            body['_request'] = {k: v for k, v in payload.items() if k != 'messages'}
        return body
    raise RuntimeError(f"request failed after {max_attempts} attempts ({last_err})")


_anthropic_client = None
_anthropic_lock = threading.Lock()


def get_anthropic_client(api_key: str):
    global _anthropic_client
    with _anthropic_lock:
        if _anthropic_client is None:
            import anthropic
            # Accept-Encoding gzip: the bundled httpx2 is incompatible with the
            # installed zstandard (Decompressor.decompress lacks output_buffer_limit),
            # so zstd responses fail as "Connection error".
            _anthropic_client = anthropic.Anthropic(
                api_key=api_key, max_retries=3, timeout=600.0,
                default_headers={'Accept-Encoding': 'gzip'})
    return _anthropic_client


def anthropic_chat(cfg: argparse.Namespace, api_key: str, messages: list[dict]) -> dict:
    """Call the Anthropic Messages API (text-only arm) and convert the response to
    the OpenAI chat-completion shape the rest of the pipeline expects.

    Temperature is never sent (current Claude models reject non-default values; the
    key's t-segment stays a label). Thinking is adaptive and cannot be disabled on
    Opus 5.5; --reasoning-effort maps to output_config.effort (low..max). A safety
    refusal becomes an error row — no server-side fallback, so the arm never
    silently switches models."""
    import anthropic
    client = get_anthropic_client(api_key)
    system = messages[0]['content']
    user = messages[1]['content']
    if not isinstance(user, str):
        raise RuntimeError('anthropic arm is text-only (no --images-dir support yet)')
    kwargs: dict = {"model": cfg.model, "max_tokens": cfg.max_tokens, "system": system,
                    "messages": [{"role": "user", "content": user}]}
    if getattr(cfg, 'reasoning_effort', ''):
        kwargs['output_config'] = {'effort': cfg.reasoning_effort}
    try:
        resp = client.messages.create(**kwargs)
    except anthropic.APIError as exc:
        raise RuntimeError(f"anthropic API error: {str(exc)[:300]}") from exc
    if resp.stop_reason == 'refusal':
        cat = getattr(getattr(resp, 'stop_details', None), 'category', None)
        raise RuntimeError(f"refusal (category={cat})")
    text = ''.join(b.text for b in resp.content if getattr(b, 'type', '') == 'text')
    u = resp.usage
    return {
        "id": resp.id, "model": resp.model, "provider": "anthropic",
        "stop_reason": resp.stop_reason,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                     "finish_reason": resp.stop_reason}],
        "usage": {"prompt_tokens": u.input_tokens, "completion_tokens": u.output_tokens,
                  "total_tokens": u.input_tokens + u.output_tokens},
    }


def mock_response(key: str, paper_text: str, model: str, n_images: int = 0) -> dict:
    """Deterministic fake chat-completion body; same shape as the real API response.
    Fake usage charges MOCK_TOKENS_PER_IMAGE prompt tokens per attached image."""
    h = int(hashlib.sha256(key.encode()).hexdigest(), 16)
    rating = ALLOWED_RATINGS[h % len(ALLOWED_RATINGS)]
    review = {
        "summary": "Mock review: the paper proposes a method and evaluates it on benchmarks. "
                   "Claims are stated in the abstract and tested in Section 4.",
        "strengths": ["Clear motivation in Section 1.", "Ablations in Section 4 support the main claim."],
        "weaknesses": ["Baselines in Table 2 are limited.", "Theory in Section 3 lacks assumptions."],
        "questions": ["How sensitive are results to the hyperparameters of Section 4.1?"],
        "soundness": 2 + (h >> 8) % 2,
        "presentation": 3,
        "contribution": 2 + (h >> 16) % 2,
        "rating": rating,
        "confidence": 4,
        "accept": rating >= 6,
    }
    content = json.dumps(review, indent=1)
    # ~one task in ten comes back fenced, to exercise fence-stripping (salted so the
    # fence draw is independent of the rating draw)
    if int(hashlib.sha256(('fence|' + key).encode()).hexdigest(), 16) % 10 == 0:
        content = f"```json\n{content}\n```"
    n_in = len(paper_text) // 4 + MOCK_TOKENS_PER_IMAGE * n_images
    return {
        "id": f"mock-{hashlib.sha256(key.encode()).hexdigest()[:12]}",
        "model": model,
        "mock": True,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content},
                     "finish_reason": "stop"}],
        "usage": {"prompt_tokens": n_in, "completion_tokens": 250, "total_tokens": n_in + 250},
    }


# ---------------------------------------------------------------- parsing / validation

def strip_fences(s: str) -> str:
    s = s.strip()
    m = re.match(r"^```[a-zA-Z]*\s*\n(.*?)\n?\s*```\s*$", s, re.DOTALL)
    return m.group(1).strip() if m else s


def coerce_to_set(value, allowed: list[int]) -> tuple[int, bool]:
    x = float(value)
    if x in allowed:
        return int(x), False
    return min(allowed, key=lambda a: (abs(a - x), a)), True


def coerce_to_range(value, lo: int, hi: int) -> tuple[int, bool]:
    x = float(value)
    iv = min(hi, max(lo, int(round(x))))
    return iv, iv != x


def validate_review(review: dict) -> tuple[dict, bool]:
    """Coerce scores onto their scales. Raises on missing/non-numeric fields."""
    if not isinstance(review, dict):
        raise ValueError("review is not a JSON object")
    coerced = False
    out: dict = {}
    out['rating'], c = coerce_to_set(review['rating'], ALLOWED_RATINGS)
    coerced |= c
    for f, hi in (('soundness', 4), ('presentation', 4), ('contribution', 4), ('confidence', 5)):
        out[f], c = coerce_to_range(review[f], 1, hi)
        coerced |= c
    acc = review['accept']
    if isinstance(acc, str):
        out['accept'] = acc.strip().lower() in ('true', 'yes', '1')
        coerced = True
    else:
        out['accept'] = bool(acc)
    return out, coerced


# ---------------------------------------------------------------- per-task worker

def run_task(task: dict, cfg: argparse.Namespace, api_key: str | None,
             system: str, user_template: str, prompt_sha: str) -> dict | None:
    if STOP.is_set():
        return None
    text_path = Path(task['text_path'])
    raw_bytes = text_path.read_bytes()
    text_sha = hashlib.sha256(raw_bytes).hexdigest()
    text = raw_bytes.decode('utf-8', errors='replace')
    truncated = False
    if len(text) > cfg.max_chars:
        text = text[:cfg.max_chars] + '\n[truncated]'
        truncated = True

    image_paths = task.get('images') or []
    image_uris = encode_image_uris(image_paths)
    n_images = len(image_uris)
    # Display label for the log's "model" field: --model-label wins; otherwise the raw
    # model name, with '+figs' appended when images are attached. Text-only runs with no
    # --model-label therefore keep "model" exactly as historical rows have it.
    re_val = getattr(cfg, 'reasoning_effort', '')
    label = cfg.model_label or (cfg.model + ('+figs' if n_images else '')
                                + (f'-re{re_val}' if re_val else ''))

    row = {
        "key": task['key'], "arxiv_id": task['arxiv_id'], "v": task['v'], "rep": task['rep'],
        "model": label, "api_model": cfg.model, "temperature": cfg.temperature,
        "modality": 'text+figures' if n_images else 'text',
        "n_images": n_images, "n_images_dropped": task.get('n_images_dropped', 0),
        "rating": None, "soundness": None, "presentation": None, "contribution": None,
        "confidence": None, "accept": None, "coerced": False, "parsed_ok": False,
        "parse_retries": 0, "truncated": truncated, "prompt_sha": prompt_sha,
        "text_sha": text_sha, "usage": {}, "latency_s": None, "raw_path": None, "error": None,
    }

    messages = build_messages(system, user_template, text, image_uris)
    body = None
    t0 = time.monotonic()
    try:
        for parse_attempt in range(2):  # initial call + at most ONE parse re-request
            if parse_attempt == 1:
                row['parse_retries'] = 1
                messages = build_messages(system + '\n' + PARSE_RETRY_LINE, user_template,
                                          text, image_uris)
            if cfg.mock:
                body = mock_response(task['key'], text, cfg.model, n_images)
            elif cfg.provider == 'anthropic':
                body = anthropic_chat(cfg, api_key, messages)
            else:
                body = http_chat(cfg, api_key, messages)
            content = body['choices'][0]['message']['content']
            row['usage'] = body.get('usage', {}) or {}
            if isinstance(body.get('_request'), dict):  # OpenAI host only
                row['request'] = body['_request']
                row['service_tier'] = body.get('service_tier')
                row['finish_reason'] = body['choices'][0].get('finish_reason')
            try:
                review, coerced = validate_review(json.loads(strip_fences(content)))
            except (ValueError, KeyError, TypeError) as exc:
                row['error'] = f"parse/validate failed: {type(exc).__name__}: {exc}"
                continue
            row.update(review)
            row['coerced'] = coerced
            row['parsed_ok'] = True
            row['error'] = None
            break
    except Exception as exc:  # HTTP 400, exhausted retries, malformed body, ...
        row['error'] = str(exc)[:500]
    row['latency_s'] = round(time.monotonic() - t0, 3)

    if body is not None:
        # Raw responses go under the display label so the text and figure arms of the
        # same model never overwrite each other (text-only, no label: unchanged path).
        raw_dir = Path(cfg.out_dir) / label
        raw_dir.mkdir(parents=True, exist_ok=True)
        raw_path = raw_dir / f"{task['safe_id']}v{task['v']}_t{cfg.temperature}_r{task['rep']}.json"
        raw_path.write_text(json.dumps(body, ensure_ascii=False, indent=2), encoding='utf-8')
        row['raw_path'] = str(raw_path)
    return row


# ---------------------------------------------------------------- main

def build_tasks(cfg: argparse.Namespace, papers: list[dict], done: set) -> tuple[list[dict], int, int]:
    tasks, n_skipped, n_missing = [], 0, 0
    texts_dir = Path(cfg.texts)
    for paper in papers:
        safe_id = paper['arxiv_id'].replace('/', '_')
        for ver in paper['versions']:
            v = ver['v']
            text_path = texts_dir / safe_id / f"v{v}.md"
            if not text_path.exists():
                n_missing += 1
                continue
            images: list[str] = []
            n_dropped = 0
            if cfg.images_dir:
                images, n_dropped = discover_images(Path(cfg.images_dir), safe_id, v,
                                                    cfg.max_images)
            # Keys with images gain an '|img' segment; text-only keys (including tasks
            # under --images-dir whose figure dir is missing/empty) keep the exact
            # historical format, so existing checkpoint rows are still skipped on resume.
            img_seg = '|img' if images else ''
            # reasoning-effort runs are a separate arm: their keys gain a '|re:' segment
            # so they never collide with the default-thinking runs of the same model
            re_seg = (f"|re:{cfg.reasoning_effort}"
                      if getattr(cfg, 'reasoning_effort', '') else '')
            for rep in range(cfg.reps):
                key = (f"{paper['arxiv_id']}v{v}|{cfg.model}|t{cfg.temperature}"
                       f"{img_seg}{re_seg}|r{rep}")
                if key in done:
                    n_skipped += 1
                    continue
                tasks.append({"key": key, "arxiv_id": paper['arxiv_id'], "safe_id": safe_id,
                              "v": v, "rep": rep, "text_path": str(text_path),
                              "images": images, "n_images_dropped": n_dropped})
    return tasks, n_skipped, n_missing


def main() -> None:
    ap = argparse.ArgumentParser(description="Score paper versions with an OpenAI-compatible chat API.")
    ap.add_argument('--manifest', default='data/processed/pilot_manifest.jsonl')
    ap.add_argument('--texts', default='data/processed/texts/')
    ap.add_argument('--out-dir', default='data/processed/reviews/')
    ap.add_argument('--log', default='data/processed/reviews_log.jsonl')
    ap.add_argument('--prompt', default='src/review/prompts/review_v1.md')
    ap.add_argument('--model', default='deepseek-chat')
    ap.add_argument('--provider', default='openai', choices=('openai', 'anthropic'),
                    help="'openai' = any OpenAI-compatible /chat/completions endpoint "
                         "(the default); 'anthropic' = the Anthropic Messages API")
    ap.add_argument('--base-url', default='https://api.deepseek.com')
    ap.add_argument('--api-key-env', default='DEEPSEEK_API_KEY')
    ap.add_argument('--images-dir', default='',
                    help="attach figure PNGs from {images-dir}/{safe_id}/v{v}/fig_*.png "
                         "('' = text-only, the default)")
    ap.add_argument('--max-images', type=int, default=20,
                    help='max figure images attached per task (default 20)')
    ap.add_argument('--model-label', default='',
                    help="display label for the log's 'model' field ('' = auto: model name, "
                         "with '+figs' appended when images are attached)")
    ap.add_argument('--reps', type=int, default=3)
    ap.add_argument('--temperature', type=float, default=1.0)
    ap.add_argument('--max-tokens', type=int, default=2000)
    ap.add_argument('--sample-papers', type=int, default=0,
                    help='0 = all papers; N>0 = round-robin across strata in manifest order')
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--limit', type=int, default=0, help='max API calls this run (0 = no cap)')
    ap.add_argument('--max-chars', type=int, default=380000)
    ap.add_argument('--reasoning-effort', default='',
                    help="set the OpenAI-compat reasoning_effort ('none' disables thinking "
                         "on Gemini); empty = omit the field (provider default)")
    ap.add_argument('--retry-failed', action='store_true',
                    help='on resume, retry keys whose rows all failed (up to two failed rows '
                         'per key); default keeps the old rule: any logged key is done')
    ap.add_argument('--stop-after-failures', type=int, default=0,
                    help='stop starting new calls after N consecutive failed rows (0 = off)')
    ap.add_argument('--max-cost-usd', type=float, default=0.0,
                    help='stop starting new calls once this invocation has spent this much, '
                         'priced with --price-in/--price-out (0 = off)')
    ap.add_argument('--price-in', type=float, default=0.0,
                    help='USD per 1M prompt tokens, for --max-cost-usd and the cost line')
    ap.add_argument('--price-out', type=float, default=0.0,
                    help='USD per 1M completion tokens (reasoning included)')
    ap.add_argument('--service-tier', default='', choices=('', 'default', 'flex'),
                    help="OpenAI service_tier (e.g. 'flex' = batch-rate pricing, slower); "
                         "empty = omit the field")
    ap.add_argument('--prompt-cache-mode', default='', choices=('', 'implicit', 'explicit'),
                    help="OpenAI prompt_cache_options.mode; 'explicit' (no breakpoints) avoids "
                         "the 1.25x cache-write rate on gpt-5.6+; empty = omit the field")
    ap.add_argument('--mock', action='store_true', help='no network; deterministic fake responses')
    ap.add_argument('--seed', type=int, default=42)
    cfg = ap.parse_args()
    if cfg.max_images < 0:
        # a negative value would be a silent Python negative slice in discover_images
        # (dropping trailing figures while still attaching the rest under an img key)
        ap.error('--max-images must be >= 0')
    if cfg.provider == 'anthropic':
        if cfg.api_key_env == 'DEEPSEEK_API_KEY':
            cfg.api_key_env = 'ANTHROPIC_API_KEY'
        if cfg.images_dir:
            ap.error('--provider anthropic is text-only for now (drop --images-dir)')
        if cfg.reasoning_effort and cfg.reasoning_effort not in ANTHROPIC_EFFORTS:
            ap.error(f"--reasoning-effort for anthropic must be one of "
                     f"{sorted(ANTHROPIC_EFFORTS)} (thinking cannot be disabled)")

    api_key = load_api_key(cfg.api_key_env, cfg.mock)
    system, user_template = load_prompt(cfg.prompt)
    if 'json' not in system.lower():
        sys.exit("Prompt system message must mention 'json' (required by response_format json_object).")
    prompt_sha = hashlib.sha256(Path(cfg.prompt).read_bytes()).hexdigest()

    rows = read_jsonl(cfg.manifest)
    papers = select_papers(rows, cfg.sample_papers)
    done = done_keys(cfg.log, 'key')
    if cfg.retry_failed:
        ok_keys: set = set()
        n_bad: Counter = Counter()
        for r in read_jsonl(cfg.log):
            if r.get('parsed_ok') and r.get('error') is None:
                ok_keys.add(r.get('key'))
            else:
                n_bad[r.get('key')] += 1
        done = ok_keys | {k for k, n in n_bad.items() if n >= 2}
    tasks, n_skipped, n_missing = build_tasks(cfg, papers, done)

    random.Random(cfg.seed).shuffle(tasks)  # blinding against order effects
    n_capped = 0
    if cfg.limit > 0 and len(tasks) > cfg.limit:
        n_capped = len(tasks) - cfg.limit
        tasks = tasks[:cfg.limit]

    print(f"papers selected: {len(papers)}/{len(rows)} | tasks: {len(tasks)} new, "
          f"{n_skipped} checkpointed, {n_capped} over --limit | missing text files: {n_missing} "
          f"| model={cfg.model} t={cfg.temperature} reps={cfg.reps} mock={cfg.mock}")
    if cfg.images_dir:
        n_img_tasks = sum(1 for t in tasks if t['images'])
        print(f"images: dir={cfg.images_dir} max={cfg.max_images} | tasks with figures: "
              f"{n_img_tasks}/{len(tasks)} (tasks without a figure dir run text-only)")

    lock = threading.Lock()
    hist: Counter = Counter()
    n_failed = n_parsed = n_coerced = n_trunc = 0
    tok_in = tok_out = 0
    n_images_sent = n_images_dropped = 0
    labels_used: set[str] = set()

    n_stopped = n_consec_fail = 0
    stop_reason = ''
    pbar = tqdm(total=len(tasks), desc='reviews', unit='call')
    with ThreadPoolExecutor(max_workers=cfg.workers) as ex:
        futures = [ex.submit(run_task, t, cfg, api_key, system, user_template, prompt_sha)
                   for t in tasks]
        seen: set = set()
        while True:
            try:
                for fut in as_completed(futures):
                    if fut in seen:
                        continue
                    seen.add(fut)
                    row = fut.result()
                    if row is None:  # never started: STOP was set
                        n_stopped += 1
                        pbar.update(1)
                        continue
                    with lock:
                        append_jsonl(cfg.log, row)
                        failed_now = row['error'] is not None or not row['parsed_ok']
                        if failed_now:
                            n_failed += 1
                        n_consec_fail = n_consec_fail + 1 if failed_now else 0
                        if row['parsed_ok']:
                            n_parsed += 1
                            hist[row['rating']] += 1
                        n_coerced += bool(row['coerced'])
                        n_trunc += bool(row['truncated'])
                        usage = row.get('usage') or {}
                        tok_in += usage.get('prompt_tokens', 0) or 0
                        tok_out += usage.get('completion_tokens', 0) or 0
                        n_images_sent += row.get('n_images', 0)
                        n_images_dropped += row.get('n_images_dropped', 0)
                        labels_used.add(row['model'])
                        pbar.update(1)
                        if not STOP.is_set():
                            spent = tok_in / 1e6 * cfg.price_in + tok_out / 1e6 * cfg.price_out
                            if cfg.stop_after_failures and n_consec_fail >= cfg.stop_after_failures:
                                stop_reason = f"{n_consec_fail} consecutive failed calls"
                            elif cfg.max_cost_usd and spent >= cfg.max_cost_usd:
                                stop_reason = f"cost cap reached (${spent:.2f})"
                            if stop_reason:
                                STOP.set()
                                print(f"\nSTOP: {stop_reason}; no new calls will start",
                                      file=sys.stderr)
                break
            except KeyboardInterrupt:
                STOP.set()
                stop_reason = stop_reason or 'interrupted'
                print("\ninterrupted: no new calls will start; logging the calls in flight",
                      file=sys.stderr)
    pbar.close()

    made = len(tasks) - n_stopped
    price_in, price_out = PRICE_IN_PER_M, PRICE_OUT_PER_M
    if cfg.provider == 'anthropic':
        for frag, (pi, po) in ANTHROPIC_PRICES.items():
            if frag in cfg.model:
                price_in, price_out = pi, po
                break
    if cfg.price_in or cfg.price_out:
        price_in, price_out = cfg.price_in, cfg.price_out
    cost = tok_in / 1e6 * price_in + tok_out / 1e6 * price_out
    print("\n=== summary ===")
    print(f"calls made: {made} | skipped (checkpointed): {n_skipped} | failed: {n_failed}")
    if stop_reason:
        print(f"STOPPED EARLY: {stop_reason} | tasks not started: {n_stopped}")
    print(f"parsed_ok: {n_parsed}/{made} ({(n_parsed / made * 100) if made else 0:.1f}%) "
          f"| coerced: {n_coerced} | truncated: {n_trunc}")
    print("rating histogram: " + ", ".join(f"{r}: {hist.get(r, 0)}" for r in ALLOWED_RATINGS))
    print(f"tokens: prompt={tok_in:,} completion={tok_out:,}")
    if cfg.images_dir:
        print(f"images attached: {n_images_sent} | dropped for the "
              f"{IMAGE_B64_BUDGET // (1024 * 1024)} MB base64 budget: {n_images_dropped}")
    print(f"approx cost = {tok_in:,}/1M * ${price_in} + {tok_out:,}/1M * ${price_out} "
          f"= ${cost:.4f} (approximate; list price, ignores cache discounts)")
    raw_dirs = ", ".join(str(Path(cfg.out_dir) / lab) for lab in sorted(labels_used or {cfg.model}))
    print(f"log: {cfg.log} | raw: {raw_dirs}")


if __name__ == '__main__':
    main()
