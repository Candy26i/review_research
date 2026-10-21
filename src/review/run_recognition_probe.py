"""Contamination/recognition probe: can the model identify anonymized pilot papers FROM MEMORY?

For each selected paper version, send the scrubbed anonymized text with the recognition
prompt and log what the model claims to remember (title/authors/venue/year/decision plus a
0..1 confidence), graded against ground truth from the pilot manifest and the ICLR corpus:

  title_correct       rapidfuzz token_sort_ratio(norm_title(guess), norm_title(truth)) >= 85
  authors_any_correct any guessed full name whose last token matches a true author surname
  venue_correct       venue_guess contains 'iclr' (case-insensitive)
  decision_correct    decision_guess == true decision

Checkpointed (append-only JSONL keyed on "{arxiv_id}v{v}|{model}"), concurrent, resumable.
--which anchor (default) probes one version per paper: the latest pre_submission /
during_review version, else v1 — the version a contaminated model is most likely to have
seen without the decision leaking in. --which all probes every version. --mock runs the
exact same parse/grade/log path with deterministic fake guesses (one exact-title recognizer,
one fuzzy-but-wrong guesser, one honest null, by key hash) and no network.

Reuses run_llm_review.py's building blocks (key loading, prompt splitting, message
building, HTTP retry loop, fence stripping) by import; run_llm_review.py is unchanged.

Run from the repo root:
    python3 src/review/run_recognition_probe.py --mock --out /tmp/recognition_mock.jsonl
    python3 src/review/run_recognition_probe.py            # real run, DeepSeek, 50 calls
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import threading
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from rapidfuzz import fuzz
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / 'src' / 'collect'))
sys.path.insert(0, str(REPO_ROOT / 'src' / 'review'))
from util import append_jsonl, done_keys, norm_title, read_jsonl  # noqa: E402
from run_llm_review import (  # noqa: E402  — reused building blocks, not duplicated
    PARSE_RETRY_LINE, PRICE_IN_PER_M, PRICE_OUT_PER_M,
    build_messages, http_chat, load_api_key, load_prompt, strip_fences,
)

TITLE_MATCH_THRESHOLD = 85
ANCHOR_STAGES = ('pre_submission', 'during_review')
GUESS_FIELDS = ('title_guess', 'authors_guess', 'venue_guess', 'year_guess', 'decision_guess')
GRADE_FIELDS = ('title_correct', 'authors_any_correct', 'venue_correct', 'decision_correct')


# ---------------------------------------------------------------- selection / ground truth

def pick_versions(paper: dict, which: str) -> list[dict]:
    """'anchor': the latest pre_submission/during_review version, else v1. 'all': every one."""
    versions = sorted(paper['versions'], key=lambda ver: ver['v'])
    if which == 'all':
        return versions
    anchors = [ver for ver in versions if ver.get('stage') in ANCHOR_STAGES]
    return [anchors[-1]] if anchors else [versions[0]]


def load_authors(path: str | Path, wanted: set) -> dict[str, list[str]]:
    """Stream the big ICLR corpus, keeping author lists only for the manifest's papers."""
    out: dict[str, list[str]] = {}
    p = Path(path)
    if not p.exists():
        return out
    with p.open() as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            oid = row.get('openreview_id')
            if oid in wanted and oid not in out:
                out[oid] = [str(a) for a in (row.get('authors') or [])]
                if len(out) == len(wanted):
                    break
    return out


# ---------------------------------------------------------------- parsing / grading

def surname_token(name: str) -> str:
    """Last whitespace token of a full name, casefolded, diacritics stripped (NFKD),
    letters/hyphen/apostrophe only. 'Müller' -> 'muller' so an ASCII guess matches an
    accented true surname, and 'Lió' -> 'lio' never collides with 'Li'."""
    toks = str(name).strip().split()
    if not toks:
        return ''
    decomposed = unicodedata.normalize('NFKD', toks[-1].casefold())
    ascii_form = ''.join(c for c in decomposed if not unicodedata.combining(c))
    return re.sub(r"[^a-z\-']", '', ascii_form)


def validate_probe(obj) -> dict:
    """Normalize the model's recognition JSON. Raises on missing/invalid 'recognized'."""
    if not isinstance(obj, dict):
        raise ValueError('response is not a JSON object')
    rec = obj['recognized']
    if isinstance(rec, str):
        rec = rec.strip().lower() in ('true', 'yes', '1')
    out: dict = {'recognized': bool(rec)}
    tg = obj.get('title_guess')
    out['title_guess'] = str(tg).strip() if isinstance(tg, str) and tg.strip() else None
    ag = obj.get('authors_guess')
    if isinstance(ag, str):
        ag = [ag]
    if isinstance(ag, list):
        names = [str(a).strip() for a in ag if str(a).strip()][:5]
        out['authors_guess'] = names or None
    else:
        out['authors_guess'] = None
    vg = obj.get('venue_guess')
    out['venue_guess'] = str(vg).strip() if isinstance(vg, str) and vg.strip() else None
    try:
        out['year_guess'] = int(obj.get('year_guess')) if obj.get('year_guess') is not None else None
    except (TypeError, ValueError):
        out['year_guess'] = None
    dg = obj.get('decision_guess')
    out['decision_guess'] = dg.strip().lower() if isinstance(dg, str) and dg.strip() else None
    try:
        cf = obj.get('confidence')
        out['confidence'] = min(1.0, max(0.0, float(cf))) if cf is not None else None
    except (TypeError, ValueError):
        out['confidence'] = None
    return out


def grade(guess: dict, task: dict) -> dict:
    """Graded booleans against the manifest/ICLR ground truth (null guess = incorrect)."""
    tg = guess.get('title_guess')
    title_correct = bool(tg) and fuzz.token_sort_ratio(
        norm_title(tg), norm_title(task['true_title'])) >= TITLE_MATCH_THRESHOLD
    true_surnames = {surname_token(a) for a in task['true_authors']} - {''}
    authors_any_correct = any(surname_token(g) in true_surnames
                              for g in (guess.get('authors_guess') or []))
    vg = guess.get('venue_guess')
    venue_correct = bool(vg) and 'iclr' in vg.lower()
    decision_correct = guess.get('decision_guess') == task['true_decision']
    return {'title_correct': title_correct, 'authors_any_correct': authors_any_correct,
            'venue_correct': venue_correct, 'decision_correct': decision_correct}


# ---------------------------------------------------------------- mock

def mock_probe_response(task: dict, model: str, n_prompt_chars: int) -> dict:
    """Deterministic fake chat body exercising the grader. By key hash, one of three modes:
    (0) exact true metadata, (1) fuzzy-but-wrong plausible guess, (2) honest null."""
    h = int(hashlib.sha256(('probe|' + task['key']).encode()).hexdigest(), 16)
    mode = h % 3
    if mode == 0:
        first_author = task['true_authors'][0] if task['true_authors'] else 'Alex Example'
        guess = {'recognized': True, 'title_guess': task['true_title'],
                 'authors_guess': [first_author], 'venue_guess': f"ICLR {task['true_year']}",
                 'year_guess': task['true_year'], 'decision_guess': task['true_decision'],
                 'confidence': 0.9}
    elif mode == 1:
        guess = {'recognized': True,
                 'title_guess': 'Amortized Spectral Kernel Routing for Federated Tabular Diffusion',
                 'authors_guess': ['Jordan Mockman', 'Casey Fakename'],
                 'venue_guess': 'NeurIPS', 'year_guess': 2021,
                 'decision_guess': 'rejected' if task['true_decision'] == 'accepted' else 'accepted',
                 'confidence': 0.6}
    else:
        guess = {'recognized': False, 'title_guess': None, 'authors_guess': None,
                 'venue_guess': None, 'year_guess': None, 'decision_guess': None,
                 'confidence': 0.05}
    n_in = n_prompt_chars // 4
    return {'id': f"mock-{hashlib.sha256(task['key'].encode()).hexdigest()[:12]}",
            'model': model, 'mock': True,
            'choices': [{'index': 0, 'message': {'role': 'assistant',
                                                 'content': json.dumps(guess)},
                         'finish_reason': 'stop'}],
            'usage': {'prompt_tokens': n_in, 'completion_tokens': 80, 'total_tokens': n_in + 80}}


# ---------------------------------------------------------------- per-task worker

def run_task(task: dict, cfg: argparse.Namespace, api_key: str | None,
             system: str, user_template: str, prompt_sha: str) -> dict:
    raw_bytes = Path(task['text_path']).read_bytes()
    text_sha = hashlib.sha256(raw_bytes).hexdigest()
    text = raw_bytes.decode('utf-8', errors='replace')
    truncated = len(text) > cfg.max_chars
    if truncated:
        text = text[:cfg.max_chars] + '\n[truncated]'

    row = {
        'key': task['key'], 'arxiv_id': task['arxiv_id'], 'v': task['v'],
        'stage': task['stage'], 'which': cfg.which, 'model': cfg.model,
        'temperature': cfg.temperature,
        'recognized': None,
        **{f: None for f in GUESS_FIELDS}, 'confidence': None,
        **{f: None for f in GRADE_FIELDS},
        'true_title': task['true_title'], 'true_decision': task['true_decision'],
        'true_year': task['true_year'], 'n_true_authors': len(task['true_authors']),
        'openreview_id': task['openreview_id'],
        'parsed_ok': False, 'parse_retries': 0, 'truncated': truncated,
        'prompt_sha': prompt_sha, 'text_sha': text_sha,
        'usage': {}, 'latency_s': None, 'error': None,
    }

    messages = build_messages(system, user_template, text)
    t0 = time.monotonic()
    try:
        for parse_attempt in range(2):  # initial call + at most ONE parse re-request
            if parse_attempt == 1:
                row['parse_retries'] = 1
                messages = build_messages(system + '\n' + PARSE_RETRY_LINE, user_template, text)
            if cfg.mock:
                body = mock_probe_response(task, cfg.model, len(text))
            else:
                body = http_chat(cfg, api_key, messages)
            content = body['choices'][0]['message']['content']
            row['usage'] = body.get('usage', {}) or {}
            try:
                guess = validate_probe(json.loads(strip_fences(content)))
            except (ValueError, KeyError, TypeError) as exc:
                row['error'] = f"parse/validate failed: {type(exc).__name__}: {exc}"
                continue
            row.update(guess)
            row.update(grade(guess, task))
            row['parsed_ok'] = True
            row['error'] = None
            break
    except Exception as exc:  # HTTP 400, exhausted retries, malformed body, ...
        row['error'] = str(exc)[:500]
    row['latency_s'] = round(time.monotonic() - t0, 3)
    return row


# ---------------------------------------------------------------- main

def build_tasks(cfg: argparse.Namespace, papers: list[dict],
                authors_by_oid: dict[str, list[str]], done: set) -> tuple[list[dict], int, int]:
    tasks, n_skipped, n_missing = [], 0, 0
    texts_dir = Path(cfg.texts)
    for paper in papers:
        safe_id = paper['arxiv_id'].replace('/', '_')
        for ver in pick_versions(paper, cfg.which):
            v = ver['v']
            text_path = texts_dir / safe_id / f"v{v}.md"
            if not text_path.exists():
                n_missing += 1
                continue
            key = f"{paper['arxiv_id']}v{v}|{cfg.model}"
            if key in done:
                n_skipped += 1
                continue
            tasks.append({'key': key, 'arxiv_id': paper['arxiv_id'], 'safe_id': safe_id,
                          'v': v, 'stage': ver.get('stage'), 'text_path': str(text_path),
                          'openreview_id': paper.get('openreview_id'),
                          'true_title': paper['title'], 'true_decision': paper['decision'],
                          'true_year': paper.get('iclr_year'),
                          'true_authors': authors_by_oid.get(paper.get('openreview_id'), [])})
    return tasks, n_skipped, n_missing


def pct(part: int, whole: int) -> str:
    return f"{part}/{whole} ({part / whole * 100:.0f}%)" if whole else f"{part}/0 (n/a)"


def summarize(rows: list[dict]) -> None:
    parsed = [r for r in rows if r['parsed_ok']]
    claims = [r for r in parsed if r['recognized']]
    print("\n=== recognition summary (this run) ===")
    print(f"recognition claim rate: {pct(len(claims), len(parsed))} of parsed rows")
    if claims:
        for f in GRADE_FIELDS:
            print(f"  {f:<20} among claims: {pct(sum(bool(r[f]) for r in claims), len(claims))}")
        for dec in ('accepted', 'rejected'):
            sub = [r for r in claims if r['true_decision'] == dec]
            print(f"  decision_correct | true={dec:<8}: {pct(sum(bool(r['decision_correct']) for r in sub), len(sub))}")
    for label, sub in (('claims', claims), ('non-claims', [r for r in parsed if not r['recognized']])):
        confs = [r['confidence'] for r in sub if r['confidence'] is not None]
        mean = f"{sum(confs) / len(confs):.3f}" if confs else 'n/a'
        print(f"  mean confidence ({label}): {mean}  [n={len(confs)}]")


def main() -> None:
    ap = argparse.ArgumentParser(description='Recognition/contamination probe over pilot paper versions.')
    ap.add_argument('--manifest', default='data/processed/pilot_manifest.jsonl')
    ap.add_argument('--texts', default='data/processed/texts/')
    ap.add_argument('--papers-log', default='data/processed/iclr_papers.jsonl',
                    help='ICLR corpus JSONL with authors[], keyed by openreview_id')
    ap.add_argument('--out', default='data/processed/recognition_log.jsonl',
                    help='append-only output log (also the checkpoint)')
    ap.add_argument('--prompt', default='src/review/prompts/recognition_v1.md')
    ap.add_argument('--model', default='deepseek-chat')
    ap.add_argument('--base-url', default='https://api.deepseek.com')
    ap.add_argument('--api-key-env', default='DEEPSEEK_API_KEY')
    ap.add_argument('--max-tokens', type=int, default=1000)
    ap.add_argument('--which', choices=('anchor', 'all'), default='anchor',
                    help="anchor = one version per paper (latest pre_submission/during_review, "
                         "else v1); all = every version")
    ap.add_argument('--limit', type=int, default=0, help='max API calls this run (0 = no cap)')
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--max-chars', type=int, default=380000)
    ap.add_argument('--mock', action='store_true', help='no network; deterministic fake guesses')
    cfg = ap.parse_args()
    cfg.temperature = 0.0  # recognition is a factual probe: always greedy

    api_key = load_api_key(cfg.api_key_env, cfg.mock)
    system, user_template = load_prompt(cfg.prompt)
    if 'json' not in system.lower():
        sys.exit("Prompt system message must mention 'json' (required by response_format json_object).")
    prompt_sha = hashlib.sha256(Path(cfg.prompt).read_bytes()).hexdigest()

    papers = read_jsonl(cfg.manifest)
    authors_by_oid = load_authors(cfg.papers_log, {p.get('openreview_id') for p in papers})
    n_no_authors = sum(1 for p in papers if not authors_by_oid.get(p.get('openreview_id')))
    done = done_keys(cfg.out, 'key')
    tasks, n_skipped, n_missing = build_tasks(cfg, papers, authors_by_oid, done)
    n_capped = 0
    if cfg.limit > 0 and len(tasks) > cfg.limit:
        n_capped = len(tasks) - cfg.limit
        tasks = tasks[:cfg.limit]

    print(f"papers: {len(papers)} | which={cfg.which} | tasks: {len(tasks)} new, "
          f"{n_skipped} checkpointed, {n_capped} over --limit | missing texts: {n_missing} "
          f"| papers w/o author list: {n_no_authors} | model={cfg.model} t={cfg.temperature} "
          f"mock={cfg.mock}")

    lock = threading.Lock()
    rows: list[dict] = []
    n_failed = tok_in = tok_out = 0
    pbar = tqdm(total=len(tasks), desc='probe', unit='call')
    with ThreadPoolExecutor(max_workers=cfg.workers) as ex:
        futures = [ex.submit(run_task, t, cfg, api_key, system, user_template, prompt_sha)
                   for t in tasks]
        for fut in as_completed(futures):
            row = fut.result()
            with lock:
                append_jsonl(cfg.out, row)
                rows.append(row)
                if row['error'] is not None or not row['parsed_ok']:
                    n_failed += 1
                usage = row.get('usage') or {}
                tok_in += usage.get('prompt_tokens', 0) or 0
                tok_out += usage.get('completion_tokens', 0) or 0
                pbar.update(1)
    pbar.close()

    n_parsed = sum(1 for r in rows if r['parsed_ok'])
    print(f"\ncalls made: {len(tasks)} | skipped (checkpointed): {n_skipped} | failed: {n_failed} "
          f"| parsed_ok: {pct(n_parsed, len(tasks))}")
    summarize(rows)
    cost = tok_in / 1e6 * PRICE_IN_PER_M + tok_out / 1e6 * PRICE_OUT_PER_M
    print(f"tokens: prompt={tok_in:,} completion={tok_out:,} | approx cost = ${cost:.4f} "
          f"(DeepSeek list price; approximate)")
    print(f"log: {cfg.out}")


if __name__ == '__main__':
    main()
