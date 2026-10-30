"""Generate AI-edited paragraph pairs — the 'AI edit' class for attribution.

"Old paper, new ink": samples classifiable paragraphs from scrubbed v1 texts of
PRE-ChatGPT papers and has an LLM apply one of several edit styles, so both
attribution classes (real pre-2022 human edits vs these) come from the same
papers — topic and era are controlled by construction, and labels are exact.

Checkpointed append-only JSONL keyed "{arxiv_id}|b{block_idx}|{style}|{model}";
rows carry (old_text, new_text, style, model, usage, error). Failed/rejected
rows are logged with an error (prune before retrying, as with review logs).

Run from repo root:
  python3 src/process/make_ai_edits.py --mock --max-papers 2
  python3 src/process/make_ai_edits.py --model deepseek-chat --max-papers 500
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import requests
from tqdm import tqdm

sys.path.insert(0, 'src/collect')
sys.path.insert(0, 'src/process')
sys.path.insert(0, 'src/review')
from util import append_jsonl, done_keys, read_jsonl  # noqa: E402
from build_splice_pilot import parse_blocks, norm, seed64  # noqa: E402
from run_llm_review import load_api_key  # noqa: E402

SYSTEM = ("You are editing one paragraph from an academic paper. Apply the "
          "requested edit and return ONLY the revised paragraph — no preamble, "
          "no quotes, no commentary.")

EDIT_STYLES = {
    'polish':  ("Improve the clarity, flow and wording of this paragraph. Keep "
                "every technical claim, number and citation marker exactly as "
                "it is.", 0.70, 1.40),
    'grammar': ("Fix grammar, awkward phrasing and word choice with minimal "
                "changes. Keep the content identical.", 0.75, 1.30),
    'rewrite': ("Rewrite this paragraph in your own words, preserving all "
                "technical claims, numbers and citation markers.", 0.60, 1.60),
    'expand':  ("Expand this paragraph for clarity: unpack dense sentences and "
                "add connective reasoning, without inventing new facts. Stay "
                "under +40% length.", 1.02, 1.70),
    'compress': ("Tighten this paragraph: remove redundancy and filler while "
                 "keeping every technical claim, number and citation marker.",
                 0.45, 0.98),
}

PREAMBLE_RE = re.compile(
    r"^(here('s| is)\b[^\n]*:\s*|sure[,!][^\n]*\n+|certainly[,!][^\n]*\n+|"
    r"revised paragraph:?\s*|\*\*revised[^\n]*\*\*\s*)", re.I)
RETRY_STATUSES = {429, 500, 502, 503, 504}


REFUSAL_RE = re.compile(
    r"^\s*(i('m| am)\s+(not\s+able|unable)\s+to|i\s+can(not|'t)\s+(apply|edit|rewrite|help|assist|do)|"
    r"(sorry|apologi[sz]e)[,.!]|as an ai\b|the text you('ve| have)\s+(shared|provided)\s+(is|appears)|"
    r"this (text|paragraph|content)\s+(is not|isn't|does not appear|doesn't appear)\s+(a|to be a)\s+)",
    re.I)


def looks_like_refusal(text: str) -> bool:
    """Text-level refusal detector. The API-level check (stop_reason ==
    'refusal') missed a deepseek row that began "I'm not able to apply this
    edit as requested. The text you've shared is a PDDL domain definition..."
    (QC 2026-10-30). Only the opening ~200 chars are inspected so a paragraph
    that merely *mentions* an apology is not rejected."""
    return bool(REFUSAL_RE.search(text[:200]))


def clean_output(text: str) -> str:
    s = text.strip()
    s = PREAMBLE_RE.sub('', s).strip()
    if s.startswith('"') and s.endswith('"'):
        s = s[1:-1].strip()
    if s.startswith('```'):
        s = re.sub(r'^```[a-zA-Z]*\s*\n?|\n?```\s*$', '', s).strip()
    return s


def call_openai(cfg, api_key: str, style_prompt: str, paragraph: str) -> tuple[str, dict]:
    url = cfg.base_url.rstrip('/') + '/chat/completions'
    payload = {"model": cfg.model, "temperature": cfg.temperature,
               "max_tokens": cfg.max_tokens,
               "messages": [{"role": "system", "content": SYSTEM},
                            {"role": "user", "content": f"{style_prompt}\n\n{paragraph}"}]}
    headers = {"Authorization": f"Bearer {api_key}"}
    last = 'unknown'
    for attempt in range(4):
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=240)
        except requests.RequestException as exc:
            last = type(exc).__name__
            time.sleep(4 * (attempt + 1))
            continue
        if resp.status_code in RETRY_STATUSES:
            last = f'HTTP {resp.status_code}'
            time.sleep(min(float(resp.headers.get('Retry-After', 4 * (attempt + 1))), 90))
            continue
        if resp.status_code != 200:
            raise RuntimeError(f'HTTP {resp.status_code}: {resp.text[:300]}')
        body = resp.json()
        return body['choices'][0]['message']['content'], body.get('usage', {}) or {}
    raise RuntimeError(f'request failed after retries ({last})')


_anthropic_client = None
_anthropic_lock = threading.Lock()


def call_anthropic(cfg, api_key: str, style_prompt: str, paragraph: str) -> tuple[str, dict]:
    global _anthropic_client
    import anthropic
    with _anthropic_lock:
        if _anthropic_client is None:
            _anthropic_client = anthropic.Anthropic(
                api_key=api_key, max_retries=3, timeout=240.0,
                default_headers={'Accept-Encoding': 'gzip'})
    try:
        resp = _anthropic_client.messages.create(
            model=cfg.model, max_tokens=cfg.max_tokens, system=SYSTEM,
            messages=[{"role": "user", "content": f"{style_prompt}\n\n{paragraph}"}])
    except anthropic.APIError as exc:
        raise RuntimeError(f'anthropic API error: {str(exc)[:300]}') from exc
    if resp.stop_reason == 'refusal':
        raise RuntimeError('refusal')
    text = ''.join(b.text for b in resp.content if getattr(b, 'type', '') == 'text')
    u = resp.usage
    return text, {'prompt_tokens': u.input_tokens, 'completion_tokens': u.output_tokens}


def mock_edit(paragraph: str, style: str) -> str:
    words = paragraph.split()
    if style == 'compress':
        return ' '.join(words[: max(10, int(len(words) * 0.8))])
    if style == 'expand':
        return paragraph + ' Moreover, this holds for the general case as well.'
    return 'Indeed, ' + paragraph[0].lower() + paragraph[1:]


def run_one(task: dict, cfg, api_key: str | None) -> dict:
    style_prompt, lo, hi = EDIT_STYLES[task['style']]
    row = {**task, 'new_text': None, 'usage': {}, 'error': None}
    try:
        if cfg.mock:
            raw = mock_edit(task['old_text'], task['style'])
            usage = {}
        elif cfg.provider == 'anthropic':
            raw, usage = call_anthropic(cfg, api_key, style_prompt, task['old_text'])
        else:
            raw, usage = call_openai(cfg, api_key, style_prompt, task['old_text'])
        new = clean_output(raw)
        row['usage'] = usage
        ratio = len(new.split()) / max(1, len(task['old_text'].split()))
        if looks_like_refusal(new):
            row['error'] = 'refusal text: ' + new[:80]
        elif norm(new) == norm(task['old_text']):
            row['error'] = 'unchanged'
        elif not (lo <= ratio <= hi):
            row['error'] = f'length ratio {ratio:.2f} outside [{lo},{hi}]'
        elif len(new.split()) < 15:
            row['error'] = 'too short'
        else:
            row['new_text'] = new
            row['new_words'] = len(new.split())
    except Exception as exc:
        row['error'] = str(exc)[:300]
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description='AI-edit generator (attribution positives).')
    ap.add_argument('--manifest', default='data/processed/pre2022_manifest.jsonl')
    ap.add_argument('--texts', default='data/processed/pre2022_texts/')
    ap.add_argument('--out-log', default='data/processed/ai_edits_log.jsonl')
    ap.add_argument('--provider', default='openai', choices=('openai', 'anthropic'))
    ap.add_argument('--model', default='deepseek-chat')
    ap.add_argument('--base-url', default='https://api.deepseek.com')
    ap.add_argument('--api-key-env', default='DEEPSEEK_API_KEY')
    ap.add_argument('--paras-per-paper', type=int, default=4)
    ap.add_argument('--max-papers', type=int, default=0, help='0 = all')
    ap.add_argument('--limit', type=int, default=0, help='max API calls (0 = no cap)')
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--temperature', type=float, default=1.0)
    ap.add_argument('--max-tokens', type=int, default=2000)
    ap.add_argument('--seed', type=int, default=20261030)
    ap.add_argument('--mock', action='store_true')
    cfg = ap.parse_args()
    if cfg.provider == 'anthropic' and cfg.api_key_env == 'DEEPSEEK_API_KEY':
        cfg.api_key_env = 'ANTHROPIC_API_KEY'
    api_key = load_api_key(cfg.api_key_env, cfg.mock)

    papers = read_jsonl(cfg.manifest)
    if cfg.max_papers:
        papers = papers[:cfg.max_papers]
    done = done_keys(cfg.out_log, 'key')
    styles = sorted(EDIT_STYLES)

    tasks, n_skipped, n_missing = [], 0, 0
    for p in papers:
        safe_id = p['arxiv_id'].replace('/', '_')
        v1 = min(v['v'] for v in p['versions'])
        path = Path(cfg.texts) / safe_id / f'v{v1}.md'
        if not path.exists():
            n_missing += 1
            continue
        blocks = parse_blocks(path.read_text(encoding='utf-8'))
        cls_idx = [i for i, b in enumerate(blocks) if b['cls']]
        if len(cls_idx) < cfg.paras_per_paper:
            continue
        rng = np.random.default_rng(seed64(f"{cfg.seed}|{p['arxiv_id']}|ai_edits"))
        chosen = rng.choice(len(cls_idx), size=cfg.paras_per_paper, replace=False)
        for j in sorted(chosen):
            i = cls_idx[j]
            style = styles[int(rng.integers(len(styles)))]
            key = f"{p['arxiv_id']}|b{i}|{style}|{cfg.model}"
            if key in done:
                n_skipped += 1
                continue
            tasks.append({'key': key, 'arxiv_id': p['arxiv_id'], 'block_idx': i,
                          'v': v1, 'style': style, 'model': cfg.model,
                          'section': blocks[i]['section'], 'kind': blocks[i]['kind'],
                          'old_text': blocks[i]['text'],
                          'old_words': len(blocks[i]['text'].split())})
    if cfg.limit and len(tasks) > cfg.limit:
        tasks = tasks[:cfg.limit]
    print(f'papers: {len(papers)} (texts missing: {n_missing}) | tasks: {len(tasks)} new, '
          f'{n_skipped} checkpointed | model={cfg.model} mock={cfg.mock}')

    lock = threading.Lock()
    n_ok = n_fail = tok_in = tok_out = 0
    pbar = tqdm(total=len(tasks), desc='ai edits', unit='call')
    with ThreadPoolExecutor(max_workers=cfg.workers) as ex:
        futures = [ex.submit(run_one, t, cfg, api_key) for t in tasks]
        for fut in as_completed(futures):
            row = fut.result()
            with lock:
                append_jsonl(cfg.out_log, row)
                n_ok += row['error'] is None
                n_fail += row['error'] is not None
                tok_in += (row['usage'] or {}).get('prompt_tokens', 0) or 0
                tok_out += (row['usage'] or {}).get('completion_tokens', 0) or 0
                pbar.update(1)
    pbar.close()
    print(f'\nok: {n_ok} | failed/rejected: {n_fail} | tokens in/out: {tok_in:,}/{tok_out:,}')
    print(f'log: {cfg.out_log}')


if __name__ == '__main__':
    main()
