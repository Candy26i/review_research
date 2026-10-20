"""Build full arXiv version chains for matched ICLR papers via OAI-PMH arXivRaw.

Reads match rows from match_arxiv.py (key fields: arxiv_id, latest_version,
openreview_id, iclr_year, title, decision, mean_score, scores, n_reviews,
published). For each matched paper either synthesizes a single-version chain
without a network call (latest_version below --min-latest-version) or fetches
the arXivRaw GetRecord from OAI-PMH, caching the raw XML before parsing.
Each version timestamp gets ICLR review-stage labels from iclr_timeline.

Resumable twice over: the output JSONL is checkpointed on arxiv_id, and
cached XML files are parsed instead of re-fetched.

Run from repo root: python3 src/collect/fetch_versions.py
"""
from __future__ import annotations

import argparse
import re
import xml.etree.ElementTree as ET
from collections import Counter
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path

from tqdm import tqdm

from iclr_timeline import label_stage
from util import ARXIV_API_DELAY_S, RateLimiter, append_jsonl, done_keys, http_get, read_jsonl

OAI_URL = "https://oaipmh.arxiv.org/oai"
OAI_NS = "{http://www.openarchives.org/OAI/2.0/}"
RAW_NS = "{http://arxiv.org/OAI/arXivRaw/}"


def parse_dt(s: str | None) -> datetime | None:
    """Parse an ISO-8601 or RFC-2822 timestamp into an aware UTC datetime."""
    if not s:
        return None
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        try:
            dt = parsedate_to_datetime(s)
        except (TypeError, ValueError):
            return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def parse_arxivraw(xml_text: str | bytes, iclr_year: int | None) -> tuple[list[dict], str]:
    """Parse a GetRecord arXivRaw response -> (versions sorted by v, chain_source)."""
    root = ET.fromstring(xml_text)
    err = root.find(f"{OAI_NS}error")
    if err is not None:
        return [], f"oai_error:{err.get('code', 'unknown')}"
    versions = []
    for el in root.iter(f"{RAW_NS}version"):
        dt = parse_dt(el.findtext(f"{RAW_NS}date"))
        m = re.search(r"(\d+)", el.findtext(f"{RAW_NS}size") or "")
        stage, fine = label_stage(iclr_year, dt) if dt and iclr_year else ("unknown", "unknown")
        versions.append({
            "v": int(el.get("version", "v0").lstrip("v")),
            "date": dt.isoformat() if dt else None,
            "size_kb": int(m.group(1)) if m else None,
            "source_type": el.findtext(f"{RAW_NS}source_type"),
            "stage": stage,
            "stage_fine": fine,
        })
    versions.sort(key=lambda d: d["v"])
    return versions, "oai"


def synthesize_chain(row: dict) -> list[dict]:
    """Single-version chain built from the match row's published date (no network)."""
    dt = parse_dt(row.get("published"))
    iclr_year = row.get("iclr_year")
    stage, fine = label_stage(iclr_year, dt) if dt and iclr_year else ("unknown", "unknown")
    return [{
        "v": 1,
        "date": dt.isoformat() if dt else row.get("published"),
        "size_kb": None,
        "source_type": None,
        "stage": stage,
        "stage_fine": fine,
    }]


def main() -> None:
    ap = argparse.ArgumentParser(description="Fetch arXiv version chains via OAI-PMH arXivRaw.")
    ap.add_argument("--matches", default="data/processed/arxiv_matches.jsonl")
    ap.add_argument("--out", default="data/processed/version_chains.jsonl")
    ap.add_argument("--xml-cache", default="data/raw/oai_xml/")
    ap.add_argument("--min-latest-version", type=int, default=2,
                    help="OAI-query only papers with latest_version >= this; synthesize the rest")
    ap.add_argument("--oai-all", action="store_true",
                    help="also OAI-query papers whose latest_version is 1")
    ap.add_argument("--limit", type=int, default=0, help="max new rows this run (0 = no cap)")
    args = ap.parse_args()

    rows = read_jsonl(args.matches)
    done = done_keys(args.out, "arxiv_id")
    limiter = RateLimiter(ARXIV_API_DELAY_S)
    cache_dir = Path(args.xml_cache)
    cache_dir.mkdir(parents=True, exist_ok=True)

    n_new = n_skipped = n_null = n_err = 0
    nv_hist: Counter = Counter()
    src_counts: Counter = Counter()
    stage_counts: Counter = Counter()
    fine_counts: Counter = Counter()

    for row in tqdm(rows, desc="version chains"):
        aid = row.get("arxiv_id")
        if not aid:
            n_null += 1
            continue
        if aid in done:
            n_skipped += 1
            continue
        if args.limit and n_new >= args.limit:
            break

        latest = row.get("latest_version")
        if not args.oai_all and latest is not None and latest < args.min_latest_version:
            versions, chain_source = synthesize_chain(row), "synthesized"
        else:
            cache_file = cache_dir / (aid.replace("/", "_") + ".xml")
            if cache_file.exists():
                xml_text = cache_file.read_bytes()
            else:
                try:
                    resp = http_get(OAI_URL, params={
                        "verb": "GetRecord",
                        "identifier": f"oai:arXiv.org:{aid}",
                        "metadataPrefix": "arXivRaw",
                    }, limiter=limiter)
                except RuntimeError as exc:
                    tqdm.write(f"fetch failed for {aid}: {exc}")
                    n_err += 1
                    continue
                # Cache raw bytes: the OAI server sends text/xml without a charset,
                # so resp.text would mis-decode UTF-8 metadata as ISO-8859-1.
                xml_text = resp.content
                cache_file.write_bytes(xml_text)
            try:
                versions, chain_source = parse_arxivraw(xml_text, row.get("iclr_year"))
            except ET.ParseError as exc:
                tqdm.write(f"XML parse failed for {aid}: {exc} (cache dropped for refetch)")
                cache_file.unlink(missing_ok=True)
                n_err += 1
                continue

        append_jsonl(args.out, {
            "arxiv_id": aid,
            "openreview_id": row.get("openreview_id"),
            "iclr_year": row.get("iclr_year"),
            "title": row.get("title"),
            "decision": row.get("decision"),
            "mean_score": row.get("mean_score"),
            "scores": row.get("scores"),
            "n_reviews": row.get("n_reviews"),
            "n_versions": len(versions),
            "chain_source": chain_source,
            "versions": versions,
        })
        done.add(aid)
        n_new += 1
        nv_hist[len(versions)] += 1
        src_counts[chain_source] += 1
        for v in versions:
            stage_counts[v["stage"]] += 1
            fine_counts[v["stage_fine"]] += 1

    print("\n=== fetch_versions summary ===")
    print(f"input rows: {len(rows)} | null arxiv_id: {n_null} | "
          f"already done (skipped): {n_skipped} | written this run: {n_new} | "
          f"fetch/parse errors: {n_err}")
    print("chain_source: " + (", ".join(f"{k}={v}" for k, v in sorted(src_counts.items())) or "-"))
    print("n_versions histogram: " +
          (", ".join(f"{k}:{v}" for k, v in sorted(nv_hist.items())) or "-"))
    print("stage coverage (coarse): " +
          (", ".join(f"{k}={v}" for k, v in sorted(stage_counts.items())) or "-"))
    print("stage coverage (fine): " +
          (", ".join(f"{k}={v}" for k, v in sorted(fine_counts.items())) or "-"))


if __name__ == "__main__":
    main()
