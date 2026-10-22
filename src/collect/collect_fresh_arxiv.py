"""Collect a SCORE-FREE, post-training-cutoff corpus of multi-version arXiv papers.

Discovers recent submissions per category through the arXiv API search endpoint
(submittedDate-windowed, paginated, 1 request / 3 s), keeps papers whose latest
version from the Atom <id> suffix is >= --min-versions, interleaves the
per-category candidate lists round-robin (discovery order, deterministic), and
for each selected paper fetches the full arXivRaw version chain over OAI-PMH
(raw XML cached in --xml-cache; chain parser reused from fetch_versions).

Emits two append-only JSONL files, both checkpointed on arxiv_id, whose rows
match the pilot schemas so every downstream stage (download_sources,
extract_sources, latex_to_text, scrub_and_audit, review runner) works unchanged:
  --manifest-out : pilot_manifest-shaped rows (scores empty, decision/stages
                   "unknown", plus categories and fresh: true)
  --papers-out   : iclr_papers-shaped rows keyed openreview_id = arxiv_id -
                   the same key as the manifest rows, which keeps the
                   scrubber's author-audit join working.

These papers have no human review scores by construction; the corpus supports
version-tracking analyses only.

Run from repo root: python3 src/collect/collect_fresh_arxiv.py
"""
from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import Iterator

from tqdm import tqdm

from fetch_versions import OAI_URL, parse_arxivraw
from match_arxiv import ATOM, ID_RE
from util import (ARXIV_API_DELAY_S, RateLimiter, append_jsonl, done_keys,
                  http_get, norm_title)

API_URL = "http://export.arxiv.org/api/query"
OPENSEARCH = {"opensearch": "http://a9.com/-/spec/opensearch/1.1/"}
PAGE_SIZE = 100
MAX_PAGES_PER_CATEGORY = 5


class QueryBudget:
    """Cap on live arXiv HTTP requests this run (0 = unlimited)."""

    def __init__(self, cap: int):
        self.cap, self.used = cap, 0

    def spend(self) -> bool:
        if self.cap and self.used >= self.cap:
            return False
        self.used += 1
        return True


def pad_date(d: str) -> str:
    """YYYYMMDD[HHMM] -> 12-digit submittedDate bound (YYYYMMDDHHMM)."""
    d = str(d).strip()
    if not d.isdigit() or not 8 <= len(d) <= 12:
        raise ValueError(f"date must be 8-12 digits YYYYMMDD[HHMM], got {d!r}")
    return d.ljust(12, "0")


def parse_search_feed(xml_text: str | bytes) -> tuple[list[dict], int | None]:
    """Parse one arXiv API Atom page -> (entries, totalResults).

    Mirrors match_arxiv.parse_feed (same ATOM namespace map and ID_RE, both
    imported from it) and additionally collects each entry's category terms.
    """
    root = ET.fromstring(xml_text)
    total_txt = root.findtext("opensearch:totalResults", None, OPENSEARCH)
    total = int(total_txt) if total_txt and total_txt.strip().isdigit() else None
    entries: list[dict] = []
    for e in root.findall("atom:entry", ATOM):
        m = ID_RE.search((e.findtext("atom:id", "", ATOM) or "").strip())
        if not m:  # malformed/error entry
            continue
        entries.append({
            "arxiv_id": m.group(1),
            "latest_version": int(m.group(2)),
            "title": " ".join((e.findtext("atom:title", "", ATOM) or "").split()),
            "published": (e.findtext("atom:published", "", ATOM) or "").strip() or None,
            "updated": (e.findtext("atom:updated", "", ATOM) or "").strip() or None,
            "authors": [a.text.strip() for a in e.findall("atom:author/atom:name", ATOM)
                        if a.text and a.text.strip()],
            "categories": [c.get("term") for c in e.findall("atom:category", ATOM)
                           if c.get("term")],
        })
    return entries, total


def discover_category(cat: str, from_ts: str, to_ts: str, min_versions: int,
                      want: int, limiter: RateLimiter, budget: QueryBudget,
                      seen: set[str]) -> tuple[list[dict], int]:
    """Paginate one category's submittedDate-windowed search.

    Returns (candidates kept in discovery order, entries scanned). Ascending
    submittedDate sort makes discovery order deterministic across runs.
    Stops after MAX_PAGES_PER_CATEGORY pages, a short page (feed exhausted),
    `want` kept candidates, or an exhausted query budget.
    """
    query = f"cat:{cat} AND submittedDate:[{from_ts} TO {to_ts}]"
    kept: list[dict] = []
    n_scanned = 0
    for page in range(MAX_PAGES_PER_CATEGORY):
        if (want and len(kept) >= want) or not budget.spend():
            break
        resp = http_get(API_URL, params={
            "search_query": query,
            "start": page * PAGE_SIZE,
            "max_results": PAGE_SIZE,
            "sortBy": "submittedDate",
            "sortOrder": "ascending",
        }, limiter=limiter)
        try:
            entries, _total = parse_search_feed(resp.content)
        except ET.ParseError as exc:
            tqdm.write(f"[{cat}] page {page}: Atom parse error ({exc}); stopping category")
            break
        n_scanned += len(entries)
        for ent in entries:
            if ent["latest_version"] >= min_versions and ent["arxiv_id"] not in seen:
                seen.add(ent["arxiv_id"])
                kept.append(ent)
        if len(entries) < PAGE_SIZE:  # feed exhausted
            break
    return kept, n_scanned


def round_robin(per_cat: dict[str, list[dict]]) -> Iterator[tuple[str, dict]]:
    """Interleave categories one candidate at a time, in discovery order."""
    idx = {c: 0 for c in per_cat}
    while any(idx[c] < len(per_cat[c]) for c in per_cat):
        for c in per_cat:  # dict preserves --categories order
            if idx[c] < len(per_cat[c]):
                yield c, per_cat[c][idx[c]]
                idx[c] += 1


def fetch_chain(aid: str, cache_dir: Path, limiter: RateLimiter,
                budget: QueryBudget) -> tuple[list[dict] | None, str]:
    """OAI arXivRaw version chain, cache-first (same cache layout and raw-bytes
    caching rationale as fetch_versions). Returns (versions, chain_source) or
    (None, reason) on failure."""
    cache_file = cache_dir / (aid.replace("/", "_") + ".xml")
    if cache_file.exists():
        xml_text: bytes = cache_file.read_bytes()
    else:
        if not budget.spend():
            return None, "budget_exhausted"
        try:
            resp = http_get(OAI_URL, params={
                "verb": "GetRecord",
                "identifier": f"oai:arXiv.org:{aid}",
                "metadataPrefix": "arXivRaw",
            }, limiter=limiter)
        except RuntimeError as exc:
            return None, f"fetch_failed:{exc}"
        xml_text = resp.content
        cache_file.write_bytes(xml_text)
    try:
        # iclr_year=None -> every version gets stage/stage_fine "unknown".
        versions, chain_source = parse_arxivraw(xml_text, None)
    except ET.ParseError as exc:
        cache_file.unlink(missing_ok=True)  # drop bad cache for a later refetch
        return None, f"xml_parse_error:{exc}"
    if chain_source.startswith("oai_error"):
        return None, chain_source
    return versions, chain_source


def manifest_row(ent: dict, versions: list[dict], chain_source: str) -> dict:
    """fresh_manifest row: pilot_manifest schema with score fields emptied."""
    n_versions = len(versions)
    return {
        "arxiv_id": ent["arxiv_id"],
        "openreview_id": ent["arxiv_id"],  # key both files on arxiv_id
        "iclr_year": None,
        "title": ent["title"],
        "decision": "unknown",
        "mean_score": None,
        "scores": [],
        "n_reviews": 0,
        "n_versions": n_versions,
        "chain_source": chain_source,
        "versions": versions,
        "strata": {
            "decision": "unknown",
            "version_bucket": "3+" if n_versions >= 3 else str(n_versions),
            "spans_stages": False,
        },
        "categories": ent.get("categories", []),
        "fresh": True,
    }


def papers_row(ent: dict) -> dict:
    """fresh_papers row: iclr_papers schema; openreview_id = arxiv_id keeps the
    scrubber's author-audit join working."""
    return {
        "openreview_id": ent["arxiv_id"],
        "year": None,
        "title": ent["title"],
        "title_norm": norm_title(ent["title"]),
        "authors": ent.get("authors", []),
        "decision_raw": "",
        "decision": "unknown",
        "scores": [],
        "n_reviews": 0,
        "mean_score": None,
        "abstract": "",
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Collect a score-free multi-version arXiv corpus (post-cutoff window).")
    ap.add_argument("--categories", nargs="+",
                    default=["cs.LG", "cs.CL", "cs.CV", "stat.ML"])
    ap.add_argument("--from-date", default="20260701",
                    help="submittedDate lower bound, YYYYMMDD[HHMM]")
    ap.add_argument("--to-date", default="20260920",
                    help="submittedDate upper bound, YYYYMMDD[HHMM]")
    ap.add_argument("--min-versions", type=int, default=2,
                    help="keep papers whose latest arXiv version is >= this")
    ap.add_argument("--target", type=int, default=30,
                    help="papers to keep, round-robin across categories (0 = all candidates)")
    ap.add_argument("--manifest-out", default="data/processed/fresh_manifest.jsonl")
    ap.add_argument("--papers-out", default="data/processed/fresh_papers.jsonl")
    ap.add_argument("--xml-cache", default="data/raw/oai_xml/")
    ap.add_argument("--limit-queries", type=int, default=0,
                    help="max live arXiv HTTP requests this run, search+OAI (0 = no cap)")
    ap.add_argument("--seed", type=int, default=42,
                    help="interface parity with the pilot collectors; discovery-order "
                         "round-robin selection is already deterministic")
    args = ap.parse_args()

    from_ts, to_ts = pad_date(args.from_date), pad_date(args.to_date)
    limiter = RateLimiter(ARXIV_API_DELAY_S)
    budget = QueryBudget(args.limit_queries)
    cache_dir = Path(args.xml_cache)
    cache_dir.mkdir(parents=True, exist_ok=True)

    done_manifest = done_keys(args.manifest_out, "arxiv_id")
    done_papers = done_keys(args.papers_out, "openreview_id")

    # 1. DISCOVER - per-category windowed search; dedupe across categories.
    per_cat: dict[str, list[dict]] = {}
    scanned: Counter = Counter()
    seen: set[str] = set()
    for cat in args.categories:
        kept, n_scanned = discover_category(cat, from_ts, to_ts, args.min_versions,
                                            args.target, limiter, budget, seen)
        per_cat[cat] = kept
        scanned[cat] = n_scanned
        tqdm.write(f"[{cat}] entries scanned: {n_scanned} | "
                   f"candidates (>= {args.min_versions} versions): {len(kept)}")

    # 2-4. SELECT round-robin, fetch CHAINS, EMIT. Candidates already in the
    # manifest checkpoint count toward --target without being re-emitted, so a
    # resumed run converges on the same corpus instead of growing it.
    kept_per_cat: Counter = Counter()
    nv_hist: Counter = Counter()
    pub_dates: list[str] = []
    n_new = n_already = n_skipped = 0
    pbar = tqdm(total=args.target or None, desc="fresh corpus", unit="paper")
    for cat, ent in round_robin(per_cat):
        if args.target and n_new + n_already >= args.target:
            break
        aid = ent["arxiv_id"]
        if aid in done_manifest:
            n_already += 1
            kept_per_cat[cat] += 1
            pbar.update(1)
            continue
        versions, chain_source = fetch_chain(aid, cache_dir, limiter, budget)
        if versions is None:
            if chain_source == "budget_exhausted":
                tqdm.write("query budget exhausted; stopping (checkpoints make reruns resume)")
                break
            tqdm.write(f"skip {aid}: {chain_source}")
            n_skipped += 1
            continue
        if len(versions) < args.min_versions:
            tqdm.write(f"skip {aid}: OAI chain has {len(versions)} version(s) "
                       f"< --min-versions {args.min_versions}")
            n_skipped += 1
            continue

        # papers row first: if interrupted between the two appends, the next
        # run rewrites only the missing manifest row (both files keyed alike).
        if aid not in done_papers:
            append_jsonl(args.papers_out, papers_row(ent))
            done_papers.add(aid)
        append_jsonl(args.manifest_out, manifest_row(ent, versions, chain_source))
        done_manifest.add(aid)

        n_new += 1
        kept_per_cat[cat] += 1
        nv_hist[len(versions)] += 1
        if ent.get("published"):
            pub_dates.append(ent["published"])
        pbar.update(1)
    pbar.close()

    print("\n=== collect_fresh_arxiv summary ===")
    print(f"window: {from_ts} .. {to_ts} | min versions: {args.min_versions} | "
          f"target: {args.target}")
    for cat in args.categories:
        print(f"  {cat}: scanned={scanned[cat]} candidates={len(per_cat[cat])} "
              f"kept={kept_per_cat[cat]}")
    print(f"kept new: {n_new} | already checkpointed: {n_already} | "
          f"skipped (oai error / short chain): {n_skipped} | "
          f"live queries used: {budget.used}")
    print("n_versions histogram (new rows): " +
          (", ".join(f"{k}:{v}" for k, v in sorted(nv_hist.items())) or "-"))
    if pub_dates:
        print(f"published range actually covered (new rows): "
              f"{min(pub_dates)} .. {max(pub_dates)}")
    print(f"manifest: {args.manifest_out}")
    print(f"papers:   {args.papers_out}")


if __name__ == "__main__":
    main()
