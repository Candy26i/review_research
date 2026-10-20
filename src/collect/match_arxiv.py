"""Match ICLR papers to arXiv ids via the arXiv API title search; checkpointed.

Run from the repo root:
    python3 src/collect/match_arxiv.py [--in ...] [--out ...] [--decisions accepted rejected]
        [--years 2024 2025] [--sample N] [--seed 42] [--limit N] [--retry-none]
"""
from __future__ import annotations

import argparse
import hashlib
import re
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict

from rapidfuzz import fuzz
from tqdm import tqdm

from util import (ARXIV_API_DELAY_S, RateLimiter, append_jsonl, done_keys,
                  http_get, norm_title, read_jsonl)

API_URL = "http://export.arxiv.org/api/query"
ATOM = {"atom": "http://www.w3.org/2005/Atom"}
ID_RE = re.compile(r"arxiv\.org/abs/(.+)v(\d+)$")
FUZZY_ACCEPT = 95.0
MIN_AUTHOR_OVERLAP = 0.3


def norm_decision(raw: str) -> str:
    """Case-insensitive normalization of OpenReview decision strings."""
    d = raw.strip().lower()
    if d.startswith("desk reject") or d.startswith("desk_reject"):
        return "desk_rejected"
    if d.startswith("accept"):
        return "accepted"
    if d.startswith("reject"):
        return "rejected"
    if d.startswith("withdraw"):
        return "withdrawn"
    if "workshop" in d:
        return "workshop_invite"
    return d or "unknown"


def parse_feed(xml_text: str) -> list[dict]:
    """Parse an arXiv Atom feed into entry dicts (id, latest version, title, dates, authors)."""
    root = ET.fromstring(xml_text)
    entries: list[dict] = []
    for e in root.findall("atom:entry", ATOM):
        m = ID_RE.search((e.findtext("atom:id", "", ATOM) or "").strip())
        if not m:  # malformed/error entry
            continue
        entries.append({
            "arxiv_id": m.group(1),  # may be old-style like cs/0112017
            "latest_version": int(m.group(2)),
            "title": " ".join((e.findtext("atom:title", "", ATOM) or "").split()),
            "published": (e.findtext("atom:published", "", ATOM) or "").strip() or None,
            "updated": (e.findtext("atom:updated", "", ATOM) or "").strip() or None,
            "authors": [a.text.strip() for a in e.findall("atom:author/atom:name", ATOM)
                        if a.text and a.text.strip()],
        })
    return entries


def search_title(title: str, limiter: RateLimiter) -> list[dict]:
    """Title search; on zero hits retry once with a punctuation-stripped query."""
    def query(q: str) -> list[dict]:
        resp = http_get(API_URL, params={"search_query": f'ti:"{q}"', "start": 0,
                                         "max_results": 10}, limiter=limiter)
        try:
            return parse_feed(resp.text)
        except ET.ParseError:
            return []

    raw = " ".join(title.replace('"', " ").split())
    entries = query(raw)
    if not entries:
        stripped = norm_title(title)
        if stripped and stripped != raw.lower():
            entries = query(stripped)
    return entries


def surnames(names: list[str]) -> set[str]:
    return {n.split()[-1].lower() for n in names if n.strip()}


def author_list(row: dict) -> list[str]:
    a = row.get("authors")
    if isinstance(a, str):
        return [x.strip() for x in a.split(",") if x.strip()]
    if isinstance(a, (list, tuple)):
        return [str(x).strip() for x in a if str(x).strip()]
    return []


def pick_match(title_norm: str, iclr_authors: list[str],
               entries: list[dict]) -> tuple[dict | None, str, float | None, float | None]:
    """Return (entry|None, match_method, match_score, author_overlap)."""
    iclr_sur = surnames(iclr_authors)

    def overlap(entry: dict) -> float | None:
        ent_sur = surnames(entry["authors"])
        if not iclr_sur or not ent_sur:
            return None
        return len(iclr_sur & ent_sur) / len(iclr_authors)

    for e in entries:
        if norm_title(e["title"]) == title_norm:
            return e, "exact", 100.0, overlap(e)
    best, best_score = None, -1.0
    for e in entries:
        s = fuzz.token_sort_ratio(title_norm, norm_title(e["title"]))
        if s > best_score:
            best, best_score = e, s
    if best is not None and best_score >= FUZZY_ACCEPT:
        ov = overlap(best)
        if ov is None or ov >= MIN_AUTHOR_OVERLAP:
            return best, "fuzzy", float(best_score), ov
    return None, "none", None, None


def row_id(row: dict) -> str:
    return str(row.get("openreview_id") or row.get("id"))


def row_year(row: dict) -> int:
    return int(row.get("year") or row.get("iclr_year"))


def select_candidates(rows: list[dict], decisions: list[str], years: list[int] | None,
                      sample: int, seed: int) -> list[dict]:
    """Filter by decisions/years; optional deterministic stratified sample of size <= N."""
    wanted = {norm_decision(d) for d in decisions}
    rows = [r for r in rows if norm_decision(str(r.get("decision", ""))) in wanted]
    if years:
        yset = set(years)
        rows = [r for r in rows if row_year(r) in yset]
    if sample <= 0 or not rows:
        return rows
    strata: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        strata[(row_year(r), norm_decision(str(r.get("decision", ""))))].append(r)
    keys = sorted(strata)
    base, extra = divmod(sample, len(keys))
    picked: list[dict] = []
    for i, k in enumerate(keys):
        ranked = sorted(strata[k],
                        key=lambda r: hashlib.sha256(f"{seed}:{row_id(r)}".encode()).hexdigest())
        picked.extend(ranked[: base + (1 if i < extra else 0)])
    return picked


def published_year(published: str | None) -> int | None:
    m = re.match(r"(\d{4})", published or "")
    return int(m.group(1)) if m else None


def main() -> None:
    ap = argparse.ArgumentParser(description="Match ICLR papers to arXiv ids by title search.")
    ap.add_argument("--in", dest="inp", default="data/processed/iclr_papers.jsonl")
    ap.add_argument("--out", default="data/processed/arxiv_matches.jsonl")
    ap.add_argument("--decisions", nargs="+", default=["accepted", "rejected"])
    ap.add_argument("--years", nargs="+", type=int, default=None,
                    help="default: all years present in the input")
    ap.add_argument("--sample", type=int, default=0,
                    help="stratified sample size across (year x decision); 0 = all")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0,
                    help="consider only the first N candidates (done rows count); 0 = no cap")
    ap.add_argument("--retry-none", action="store_true",
                    help="re-query rows previously written with match_method 'none'")
    args = ap.parse_args()

    rows = read_jsonl(args.inp)
    if not rows:
        ap.error(f"no input rows found in {args.inp}")
    candidates = select_candidates(rows, args.decisions, args.years, args.sample, args.seed)

    if args.retry_none:
        last: dict[str, str | None] = {}
        for prev in read_jsonl(args.out):
            if "openreview_id" in prev:
                last[prev["openreview_id"]] = prev.get("match_method")
        done = {k for k, m in last.items() if m != "none"}
    else:
        done = done_keys(args.out, "openreview_id")

    window = candidates[: args.limit] if args.limit > 0 else candidates
    todo = [r for r in window if row_id(r) not in done]
    n_skipped = len(window) - len(todo)

    limiter = RateLimiter(ARXIV_API_DELAY_S)
    methods: Counter[str] = Counter()
    match_scores: list[float] = []
    n_multi_version = 0
    n_implausible = 0
    seen_this_run: set[str] = set()

    for row in tqdm(todo, desc="matching", unit="paper"):
        oid = row_id(row)
        if oid in seen_this_run:
            continue
        seen_this_run.add(oid)
        title = str(row.get("title", ""))
        title_norm = norm_title(title)
        iclr_year = row_year(row)
        try:
            entries = search_title(title, limiter)
        except RuntimeError as exc:
            tqdm.write(f"query failed for {oid}: {exc}")
            entries = []
        entry, method, score, auth_overlap = pick_match(title_norm, author_list(row), entries)

        scores = list(row.get("scores") or [])
        n_reviews = int(row.get("n_reviews", len(scores)))
        mean_score = row.get("mean_score")
        if mean_score is None and scores:
            mean_score = sum(scores) / len(scores)

        date_plausible = None
        if entry is not None:
            py = published_year(entry["published"])  # <published> is the v1 date
            if py is not None:
                date_plausible = iclr_year - 3 <= py <= iclr_year + 1

        append_jsonl(args.out, {
            "openreview_id": oid,
            "iclr_year": iclr_year,
            "title": title,
            "decision": row.get("decision"),
            "mean_score": mean_score,
            "scores": scores,
            "n_reviews": n_reviews,
            "arxiv_id": entry["arxiv_id"] if entry else None,
            "arxiv_title": entry["title"] if entry else None,
            "latest_version": entry["latest_version"] if entry else None,
            "published": entry["published"] if entry else None,
            "updated": entry["updated"] if entry else None,
            "match_method": method,
            "match_score": score,
            "author_overlap": auth_overlap,
            "date_plausible": date_plausible,
        })

        methods[method] += 1
        if score is not None:
            match_scores.append(score)
        if entry is not None and entry["latest_version"] >= 2:
            n_multi_version += 1
        if date_plausible is False:
            n_implausible += 1

    print("\n=== match_arxiv summary (this run) ===")
    print(f"input rows: {len(rows)} | candidates: {len(candidates)} | "
          f"already checkpointed: {n_skipped} | queried: {len(seen_this_run)}")
    for m in ("exact", "fuzzy", "none"):
        print(f"  match_method={m}: {methods.get(m, 0)}")
    if match_scores:
        print(f"mean match_score (matched rows): {sum(match_scores) / len(match_scores):.2f}")
    print(f"matched with latest_version >= 2 (version-chain candidates): {n_multi_version}")
    print(f"matched with implausible dates: {n_implausible}")
    print(f"output: {args.out}")


if __name__ == "__main__":
    main()
