"""Safely unpack downloaded arXiv source archives; fetch PDF fallback for versions without source.

Per downloads.jsonl row, extract the archive into {dest}/{safe_id}/v{v}/:
  - tar.gz trees via tarfile with filter='data' (member sizes summed first, 500 MB cap),
  - single-file gzipped TeX decompressed to main.tex (same cap).
Then scan the extracted tree (file count, *.tex paths, .bbl presence).
Manifest versions absent from the downloads log get the arXiv PDF instead.

Archive contents are untrusted: extraction uses tarfile's 'data' filter only, nothing
extracted is ever executed or imported.
"""
from __future__ import annotations

import argparse
import gzip
import shutil
import sys
import tarfile
from collections import Counter
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, "src/collect")
from util import (  # noqa: E402
    ARXIV_API_DELAY_S,
    RateLimiter,
    append_jsonl,
    done_keys,
    http_get,
    read_jsonl,
)

MAX_EXTRACT_BYTES = 500 * 1024 * 1024  # 500 MB per archive
GZIP_CHUNK = 1 << 20
PDF_URL = "https://arxiv.org/pdf/{arxiv_id}v{v}"
RAW_SOURCES_DIR = Path("data/raw/sources")


def safe_id_of(arxiv_id: str) -> str:
    return arxiv_id.replace("/", "_")


def scan_tree(root: Path) -> tuple[int, list[str], bool, int]:
    """Count regular files under root; return (n_files, tex_files(rel), has_bbl, bytes)."""
    n_files = 0
    tex_files: list[str] = []
    has_bbl = False
    total = 0
    for p in sorted(root.rglob("*")):
        if not p.is_file() or p.is_symlink():
            continue
        n_files += 1
        total += p.stat().st_size
        suffix = p.suffix.lower()
        if suffix == ".tex":
            tex_files.append(p.relative_to(root).as_posix())
        elif suffix == ".bbl":
            has_bbl = True
    return n_files, tex_files, has_bbl, total


def extract_tar(src: Path, out_dir: Path) -> None:
    """Extract a tar(.gz) with the stdlib 'data' filter, enforcing the size cap first."""
    with tarfile.open(src, "r:gz") as tf:
        members = tf.getmembers()
        pending = sum(m.size for m in members if m.isreg())
        if pending > MAX_EXTRACT_BYTES:
            raise ValueError(
                f"over_cap: archive would extract {pending} bytes (> {MAX_EXTRACT_BYTES})"
            )
        out_dir.mkdir(parents=True, exist_ok=True)
        tf.extractall(out_dir, filter="data")


def extract_single_gzip(src: Path, out_dir: Path) -> None:
    """Decompress a single-file gzip to {out_dir}/main.tex, enforcing the size cap."""
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "main.tex"
    written = 0
    try:
        with gzip.open(src, "rb") as fin, out_path.open("wb") as fout:
            while True:
                chunk = fin.read(GZIP_CHUNK)
                if not chunk:
                    break
                written += len(chunk)
                if written > MAX_EXTRACT_BYTES:
                    raise ValueError(
                        f"over_cap: gzip would extract more than {MAX_EXTRACT_BYTES} bytes"
                    )
                fout.write(chunk)
    except Exception:
        out_path.unlink(missing_ok=True)
        raise


def process_download(row: dict, dest: Path, log_path: Path) -> dict:
    arxiv_id = row["arxiv_id"]
    v = row["v"]
    safe_id = safe_id_of(arxiv_id)
    out_dir = dest / safe_id / f"v{v}"
    record = {
        "key": row["key"],
        "arxiv_id": arxiv_id,
        "v": v,
        "dir": str(out_dir),
        "kind": None,
        "n_files": 0,
        "tex_files": [],
        "has_bbl": False,
        "bytes_extracted": 0,
        "ok": False,
        "error": None,
    }
    try:
        src = Path(row["path"])
        if not src.is_file():
            raise FileNotFoundError(f"source archive missing: {src}")
        is_tar = False
        try:
            is_tar = tarfile.is_tarfile(src)
        except Exception:
            is_tar = False
        if is_tar:
            record["kind"] = "tar"
            extract_tar(src, out_dir)
        else:
            record["kind"] = "single_tex"
            extract_single_gzip(src, out_dir)
        n_files, tex_files, has_bbl, total = scan_tree(out_dir)
        record.update(
            n_files=n_files,
            tex_files=tex_files,
            has_bbl=has_bbl,
            bytes_extracted=total,
            ok=True,
        )
    except Exception as exc:  # bad archive: log, don't crash
        record["error"] = f"{type(exc).__name__}: {exc}"
        # Don't leave a partial tree behind: a failed row must not look extracted
        # to anything that walks {dest} instead of joining on ok=true.
        shutil.rmtree(out_dir, ignore_errors=True)
        try:  # drop the {safe_id} parent too if nothing else lives in it
            out_dir.parent.rmdir()
        except OSError:
            pass
    append_jsonl(log_path, record)
    return record


def missing_manifest_versions(manifest: list[dict], downloads: list[dict]) -> list[tuple[str, int]]:
    """(arxiv_id, v) pairs listed in the manifest but absent from the downloads log."""
    have = {(r["arxiv_id"], r["v"]) for r in downloads}
    missing: list[tuple[str, int]] = []
    for paper in manifest:
        for ver in paper.get("versions", []):
            pair = (paper["arxiv_id"], ver["v"])
            if pair not in have and pair not in missing:
                missing.append(pair)
    return missing


def process_pdf_fallback(
    arxiv_id: str, v: int, dest: Path, log_path: Path, limiter: RateLimiter
) -> dict:
    safe_id = safe_id_of(arxiv_id)
    out_dir = dest / safe_id / f"v{v}"
    record = {
        "key": f"{arxiv_id}v{v}",
        "arxiv_id": arxiv_id,
        "v": v,
        "dir": str(out_dir),
        "kind": "pdf",
        "n_files": 0,
        "tex_files": [],
        "has_bbl": False,
        "bytes_extracted": 0,
        "ok": False,
        "error": None,
    }
    try:
        resp = http_get(PDF_URL.format(arxiv_id=arxiv_id, v=v), limiter=limiter)
        if not resp.content.startswith(b"%PDF"):
            # arXiv can answer 200 with an HTML "unavailable" page; don't persist it as a PDF.
            raise ValueError(f"not_a_pdf: response does not start with %PDF ({resp.content[:40]!r})")
        raw_path = RAW_SOURCES_DIR / safe_id / f"arXiv-{safe_id}v{v}.pdf"
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.write_bytes(resp.content)
        out_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(raw_path, out_dir / "paper.pdf")
        record.update(n_files=1, bytes_extracted=len(resp.content), ok=True)
    except Exception as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
    append_jsonl(log_path, record)
    return record


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--downloads", default="data/processed/downloads.jsonl")
    ap.add_argument("--manifest", default="data/processed/pilot_manifest.jsonl")
    ap.add_argument("--dest", default="data/raw/extracted/")
    ap.add_argument("--log", default="data/processed/extracted.jsonl")
    ap.add_argument("--limit", type=int, default=0, help="max new items to process (0 = all)")
    args = ap.parse_args()

    dest = Path(args.dest)
    log_path = Path(args.log)
    downloads = read_jsonl(args.downloads)
    manifest = read_jsonl(args.manifest)
    done = done_keys(log_path, "key")

    results: list[dict] = []
    processed = 0

    todo = [r for r in downloads if r["key"] not in done]
    for row in tqdm(todo, desc="extract", unit="archive"):
        if args.limit and processed >= args.limit:
            break
        results.append(process_download(row, dest, log_path))
        processed += 1

    limiter = RateLimiter(ARXIV_API_DELAY_S)
    for arxiv_id, v in missing_manifest_versions(manifest, downloads):
        if args.limit and processed >= args.limit:
            break
        if f"{arxiv_id}v{v}" in done:
            continue
        print(f"PDF fallback: {arxiv_id}v{v}")
        results.append(process_pdf_fallback(arxiv_id, v, dest, log_path, limiter))
        processed += 1

    kinds = Counter(r["kind"] for r in results)
    failures = [r for r in results if not r["ok"]]
    total_mb = sum(r["bytes_extracted"] for r in results) / (1024 * 1024)
    print(f"\nProcessed {processed} new item(s) ({len(done)} already done, skipped).")
    print(f"Kinds: {dict(kinds)}")
    print(f"Failures: {len(failures)}")
    for r in failures:
        print(f"  {r['key']}: {r['error']}")
    print(f"Total extracted: {total_mb:.1f} MB")


if __name__ == "__main__":
    main()
