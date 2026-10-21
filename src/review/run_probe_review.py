"""E4: score the known-direction edit probes with the SAME rubric/machinery as the main study.

Consumes the probes manifest written by the E4 probe builder (data/processed/probes_log.jsonl,
one row per paper x variant). Expected row fields:
  arxiv_id    pilot paper id
  variant     probe variant name (e.g. anchor / typos / drop_results / ... — not hardcoded
              here; --variants filters on whatever names the log contains)
  anchor_v    the anchor version the probe was built from ("v" accepted as a fallback)
  text_path   path to the probe markdown (fallback: {probes-dir}/{safe_id}/{variant}.md)
  skipped     truthy (or "skip" truthy, or status=="skipped") = probe was not built; the
              row is excluded from scoring and only counted.

Every non-skipped row is scored --reps times with src/review/prompts/review_v1.md — the
exact rubric of the main study — through run_llm_review.py's request/parse/validate/save
machinery (imported, not duplicated; run_llm_review.py is unchanged). Checkpointed
append-only JSONL on

    key = "{arxiv_id}|{variant}|{model}|t{temperature}|r{rep}"

so re-running makes 0 new calls. Log rows match the main reviews log (rating/subscores/
accept/coerced/parsed_ok/usage/latency/raw_path/..., with "v" = the anchor version) PLUS
"variant" and "anchor_v"; the "model" field stays the raw model string (analysis groups
by variant itself, and analyze_probes.py pairs on the (arxiv_id, v) unit). Raw response
bodies land under {out-dir}/{model}-probes/ so probe raws never mix with main-study raws.

Run from the repo root:
    python3 src/review/run_probe_review.py --mock      # no network, same parse/log path
    python3 src/review/run_probe_review.py             # real run (orchestrator only)
"""
from __future__ import annotations

import argparse
import hashlib
import random
import sys
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / 'src' / 'collect'))
sys.path.insert(0, str(REPO_ROOT / 'src' / 'review'))
from util import append_jsonl, done_keys, read_jsonl  # noqa: E402
from run_llm_review import (  # noqa: E402  — reused machinery, not duplicated
    PRICE_IN_PER_M, PRICE_OUT_PER_M,
    load_api_key, load_prompt, run_task,
)

PATH_FIELDS = ('text_path', 'path', 'probe_path', 'out_path')


# ---------------------------------------------------------------- probes-log handling

def is_skipped(row: dict) -> bool:
    return bool(row.get('skipped') or row.get('skip')) \
        or str(row.get('status', '')).strip().lower() == 'skipped'


def probe_text_path(row: dict, probes_dir: Path, safe_id: str) -> Path:
    """The probe text file: the row's own path field if present (relative paths resolve
    against the cwd, i.e. the repo root), else {probes-dir}/{safe_id}/{variant}.md."""
    for f in PATH_FIELDS:
        p = row.get(f)
        if p:
            return Path(p)
    return probes_dir / safe_id / f"{row['variant']}.md"


def build_tasks(cfg: argparse.Namespace, probe_rows: list[dict],
                done: set) -> tuple[list[dict], dict]:
    """One task per non-skipped, variant-filtered probes_log row x rep, minus checkpointed
    keys. Task dicts are shaped for run_llm_review.run_task: 'safe_id'/'v' only feed the
    raw-response filename ({safe_id}_{variant}v{anchor_v}_t{T}_r{rep}.json)."""
    wanted = set(cfg.variants) if cfg.variants else None
    probes_dir = Path(cfg.probes_dir)
    counts = {'rows': len(probe_rows), 'skipped_in_log': 0, 'filtered_out': 0,
              'malformed': 0, 'duplicate': 0, 'missing_text': 0, 'checkpointed': 0}
    tasks: list[dict] = []
    seen_keys: set[str] = set()
    for row in probe_rows:
        if is_skipped(row):
            counts['skipped_in_log'] += 1
            continue
        arxiv_id, variant = row.get('arxiv_id'), row.get('variant')
        if not arxiv_id or not variant:
            counts['malformed'] += 1
            continue
        if wanted is not None and variant not in wanted:
            counts['filtered_out'] += 1
            continue
        safe_id = str(arxiv_id).replace('/', '_')
        text_path = probe_text_path(row, probes_dir, safe_id)
        if not text_path.exists():
            counts['missing_text'] += 1
            continue
        anchor_v = row.get('anchor_v', row.get('v'))
        for rep in range(cfg.reps):
            key = f"{arxiv_id}|{variant}|{cfg.model}|t{cfg.temperature}|r{rep}"
            if key in seen_keys:  # duplicate probes_log row
                counts['duplicate'] += 1
                continue
            seen_keys.add(key)
            if key in done:
                counts['checkpointed'] += 1
                continue
            tasks.append({'key': key, 'arxiv_id': arxiv_id, 'variant': variant,
                          'anchor_v': anchor_v, 'safe_id': f"{safe_id}_{variant}",
                          'v': anchor_v, 'rep': rep, 'text_path': str(text_path),
                          'images': []})
    return tasks, counts


# ---------------------------------------------------------------- per-task worker

def run_probe_task(task: dict, cfg: argparse.Namespace, api_key: str | None,
                   system: str, user_template: str, prompt_sha: str) -> dict:
    """run_llm_review.run_task does the request/parse/validate/save work (raws under
    {out-dir}/{model}-probes/ via cfg.model_label); this wrapper restores the raw model
    string in 'model' and adds the probe fields."""
    row = run_task(task, cfg, api_key, system, user_template, prompt_sha)
    row['model'] = cfg.model  # raw model string, not the '-probes' raw-dir label
    row['variant'] = task['variant']
    row['anchor_v'] = task['anchor_v']
    # 'v' stays (= anchor_v): main-log parity, and analyze_probes.py pairs rows on
    # the (arxiv_id, v) unit via row['v'] — it does not read 'anchor_v'.
    return row


# ---------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser(description='Score E4 probe variants with the main-study rubric.')
    ap.add_argument('--probes-log', default='data/processed/probes_log.jsonl',
                    help='probes manifest from the E4 probe builder')
    ap.add_argument('--probes-dir', default='data/processed/probes/',
                    help='probe texts root (fallback when a row has no text_path)')
    ap.add_argument('--out-dir', default='data/processed/reviews/',
                    help='raw response bodies go under {out-dir}/{model}-probes/')
    ap.add_argument('--log', default='data/processed/probe_reviews_log.jsonl',
                    help='append-only output log (also the checkpoint, keyed on "key")')
    ap.add_argument('--prompt', default='src/review/prompts/review_v1.md')
    ap.add_argument('--model', default='deepseek-reasoner')
    ap.add_argument('--base-url', default='https://api.deepseek.com')
    ap.add_argument('--api-key-env', default='DEEPSEEK_API_KEY')
    ap.add_argument('--reps', type=int, default=3)
    ap.add_argument('--temperature', type=float, default=1.0)
    ap.add_argument('--max-tokens', type=int, default=16000)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--limit', type=int, default=0, help='max API calls this run (0 = no cap)')
    ap.add_argument('--max-chars', type=int, default=380000)
    ap.add_argument('--mock', action='store_true', help='no network; deterministic fake responses')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--variants', nargs='+', default=None,
                    help='score only these variant names (default: all in the probes log)')
    cfg = ap.parse_args()
    cfg.model_label = f"{cfg.model}-probes"  # run_task raw-dir label; log 'model' stays raw

    api_key = load_api_key(cfg.api_key_env, cfg.mock)
    system, user_template = load_prompt(cfg.prompt)
    if 'json' not in system.lower():
        sys.exit("Prompt system message must mention 'json' (required by response_format json_object).")
    prompt_sha = hashlib.sha256(Path(cfg.prompt).read_bytes()).hexdigest()

    probe_rows = read_jsonl(cfg.probes_log)
    if not probe_rows:
        sys.exit(f"No probes in {cfg.probes_log} — run the E4 probe builder first.")
    done = done_keys(cfg.log, 'key')
    tasks, counts = build_tasks(cfg, probe_rows, done)

    random.Random(cfg.seed).shuffle(tasks)  # blinding against order effects
    n_capped = 0
    if cfg.limit > 0 and len(tasks) > cfg.limit:
        n_capped = len(tasks) - cfg.limit
        tasks = tasks[:cfg.limit]

    all_variants = sorted({t['variant'] for t in tasks})
    print(f"probes_log rows: {counts['rows']} (skipped-in-log: {counts['skipped_in_log']}, "
          f"variant-filtered: {counts['filtered_out']}, malformed: {counts['malformed']}, "
          f"duplicate: {counts['duplicate']}, missing texts: {counts['missing_text']})")
    print(f"tasks: {len(tasks)} new, {counts['checkpointed']} checkpointed, {n_capped} over "
          f"--limit | variants this run: {', '.join(all_variants) or '(none)'} | "
          f"model={cfg.model} t={cfg.temperature} reps={cfg.reps} mock={cfg.mock}")
    if cfg.variants:
        missing = set(cfg.variants) - {r.get('variant') for r in probe_rows}
        if missing:
            print(f"WARNING: --variants not present in the probes log: {', '.join(sorted(missing))}")

    lock = threading.Lock()
    n_failed = n_parsed = n_coerced = n_trunc = 0
    tok_in = tok_out = 0
    ratings_by_variant: dict[str, list[int]] = defaultdict(list)

    pbar = tqdm(total=len(tasks), desc='probe reviews', unit='call')
    with ThreadPoolExecutor(max_workers=cfg.workers) as ex:
        futures = [ex.submit(run_probe_task, t, cfg, api_key, system, user_template, prompt_sha)
                   for t in tasks]
        for fut in as_completed(futures):
            row = fut.result()
            with lock:
                append_jsonl(cfg.log, row)
                if row['error'] is not None or not row['parsed_ok']:
                    n_failed += 1
                if row['parsed_ok']:
                    n_parsed += 1
                    ratings_by_variant[row['variant']].append(row['rating'])
                n_coerced += bool(row['coerced'])
                n_trunc += bool(row['truncated'])
                usage = row.get('usage') or {}
                tok_in += usage.get('prompt_tokens', 0) or 0
                tok_out += usage.get('completion_tokens', 0) or 0
                pbar.update(1)
    pbar.close()

    made = len(tasks)
    cost = tok_in / 1e6 * PRICE_IN_PER_M + tok_out / 1e6 * PRICE_OUT_PER_M
    print("\n=== summary ===")
    print(f"calls made: {made} | skipped (checkpointed): {counts['checkpointed']} "
          f"| skipped (in probes log): {counts['skipped_in_log']} | failed: {n_failed}")
    print(f"parsed_ok: {n_parsed}/{made} ({(n_parsed / made * 100) if made else 0:.1f}%) "
          f"| coerced: {n_coerced} | truncated: {n_trunc}")
    print("per-variant mean rating (this run):")
    for variant in sorted(ratings_by_variant):
        rs = ratings_by_variant[variant]
        print(f"  {variant:<24} {sum(rs) / len(rs):5.2f}  [n={len(rs)}]")
    if not ratings_by_variant:
        print("  (no parsed rows)")
    print(f"tokens: prompt={tok_in:,} completion={tok_out:,}")
    print(f"approx cost = {tok_in:,}/1M * ${PRICE_IN_PER_M} + {tok_out:,}/1M * ${PRICE_OUT_PER_M} "
          f"= ${cost:.4f} (approximate; deepseek-chat list price, ignores cache discounts "
          f"and per-model price differences)")
    print(f"log: {cfg.log} | raw: {Path(cfg.out_dir) / cfg.model_label}")


if __name__ == '__main__':
    main()
