"""Download per-version arXiv source archives (e-print) for the pilot manifest.

For each manifest paper and each of its versions, fetches
https://arxiv.org/e-print/{arxiv_id}v{v} (301s to /src/; requests follows),
sniffs magic bytes to pick an extension, and saves the raw bytes. Archives are
untrusted and are NEVER extracted or decompressed here — later stages do that.
Resumable via the download log JSONL, checkpointed on key "{arxiv_id}v{v}".

Run from repo root: python3 src/collect/download_sources.py
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from tqdm import tqdm

from util import ARXIV_API_DELAY_S, RateLimiter, append_jsonl, done_keys, http_get, read_jsonl


def sniff(content: bytes) -> tuple[str, str]:
    """Magic-byte sniff -> (kind, extension)."""
    if content[:2] == b"\x1f\x8b":
        return "gzip", ".gz"
    if content[:4] == b"%PDF":
        return "pdf", ".pdf"
    return "other", ".bin"


def main() -> None:
    ap = argparse.ArgumentParser(description="Download arXiv e-print sources per version.")
    ap.add_argument("--manifest", default="data/processed/pilot_manifest.jsonl")
    ap.add_argument("--dest", default="data/raw/sources/")
    ap.add_argument("--log", default="data/processed/downloads.jsonl")
    ap.add_argument("--max-papers", type=int, default=0, help="cap manifest papers (0 = all)")
    ap.add_argument("--limit", type=int, default=0, help="max files this run (0 = no cap)")
    args = ap.parse_args()

    papers = read_jsonl(args.manifest)
    if args.max_papers:
        papers = papers[:args.max_papers]
    done = done_keys(args.log, "key")
    limiter = RateLimiter(ARXIV_API_DELAY_S)
    dest = Path(args.dest)

    n_dl = n_skipped = n_err = 0
    total_bytes = 0
    kinds: Counter = Counter()

    tasks = [(p, v) for p in papers for v in p.get("versions", [])]
    for paper, ver in tqdm(tasks, desc="downloads"):
        aid = paper["arxiv_id"]
        vnum = ver["v"]
        key = f"{aid}v{vnum}"
        if key in done:
            n_skipped += 1
            continue
        if args.limit and n_dl >= args.limit:
            break
        try:
            resp = http_get(f"https://arxiv.org/e-print/{aid}v{vnum}", limiter=limiter)
        except RuntimeError as exc:
            tqdm.write(f"download failed for {key}: {exc}")
            n_err += 1
            continue
        kind, ext = sniff(resp.content)
        safe_id = aid.replace("/", "_")
        out_dir = dest / safe_id
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"arXiv-{safe_id}v{vnum}{ext}"
        path.write_bytes(resp.content)
        append_jsonl(args.log, {
            "key": key,
            "arxiv_id": aid,
            "v": vnum,
            "path": str(path),
            "bytes": len(resp.content),
            "kind": kind,
            "content_type": resp.headers.get("Content-Type"),
            "stage": ver.get("stage"),
        })
        done.add(key)
        n_dl += 1
        total_bytes += len(resp.content)
        kinds[kind] += 1

    print("\n=== download_sources summary ===")
    print(f"papers: {len(papers)} | version tasks: {len(tasks)} | "
          f"downloaded this run: {n_dl} | skipped via checkpoint: {n_skipped} | errors: {n_err}")
    print(f"total downloaded: {total_bytes / 1e6:.1f} MB")
    print("kinds: " + (", ".join(f"{k}={v}" for k, v in sorted(kinds.items())) or "-"))


if __name__ == "__main__":
    main()
