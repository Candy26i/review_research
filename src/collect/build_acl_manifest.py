"""Build the FULL (no-sampling) score-free ACL manifest from version chains.

Keeps every chain row with n_versions >= --min-versions, re-attaches
venue/acl_id from the collect_acl papers file by openreview_id, and OVERWRITES
--out with rows shaped exactly like data/processed/fresh_manifest.jsonl
(arxiv_id, openreview_id, iclr_year, title, decision, mean_score, scores,
n_reviews, n_versions, chain_source, versions, strata, categories, fresh)
plus the two ACL fields "venue" and "acl_id". ACL chains carry iclr_year 0 and
stage 'unknown' on every version, so strata.spans_stages is always false.

Run from repo root:
    python3 src/collect/build_acl_manifest.py [--chains C] [--papers P]
        [--out O] [--min-versions M]
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from util import read_jsonl


def main() -> None:
    ap = argparse.ArgumentParser(
        description="ACL version chains -> full score-free manifest (no sampling).")
    ap.add_argument("--chains", default="data/processed/acl_chains.jsonl")
    ap.add_argument("--papers", default="data/processed/acl_papers.jsonl")
    ap.add_argument("--out", default="data/processed/acl_manifest.jsonl")
    ap.add_argument("--min-versions", type=int, default=2)
    args = ap.parse_args()

    chains = read_jsonl(args.chains)
    papers = {r["openreview_id"]: r for r in read_jsonl(args.papers)
              if "openreview_id" in r}

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_written = n_unjoined = 0
    versions_hist: Counter[int] = Counter()
    venue_counts: Counter[str] = Counter()
    with out_path.open("w") as f:
        for c in chains:
            n_versions = int(c.get("n_versions") or 0)
            if n_versions < args.min_versions:
                continue
            oid = c.get("openreview_id")
            paper = papers.get(oid)
            if paper is None:
                n_unjoined += 1
            # openreview_id IS the anthology ID for ACL rows, so the fallbacks
            # below stay truthful when the papers join misses.
            venue = (paper or {}).get("venue") or (
                str(oid).rsplit(".", 1)[0] if oid else None)
            acl_id = (paper or {}).get("acl_id") or oid
            versions = c.get("versions") or []
            rec = {
                "arxiv_id": c.get("arxiv_id"),
                "openreview_id": oid,
                "iclr_year": c.get("iclr_year"),
                "title": c.get("title"),
                "decision": c.get("decision", "accepted"),
                "mean_score": c.get("mean_score"),
                "scores": c.get("scores") or [],
                "n_reviews": c.get("n_reviews", 0),
                "n_versions": n_versions,
                "chain_source": c.get("chain_source"),
                "versions": versions,
                "strata": {
                    "decision": c.get("decision", "accepted"),
                    "version_bucket": "3+" if n_versions >= 3 else "2",
                    # all ACL stages are 'unknown' (iclr_year 0) -> always False
                    "spans_stages": len({v.get("stage") for v in versions}) >= 2,
                },
                "categories": c.get("categories") or [],
                "fresh": False,
                "venue": venue,
                "acl_id": acl_id,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_written += 1
            versions_hist[n_versions] += 1
            venue_counts[venue or "unknown"] += 1

    print("=== build_acl_manifest summary ===")
    print(f"chains read: {len(chains)} | eligible (n_versions >= "
          f"{args.min_versions}): {n_written} | written: {n_written} -> {out_path}")
    if n_unjoined:
        print(f"WARNING: {n_unjoined} chain rows had no papers-file match "
              f"(venue derived from the anthology ID)")
    print("n_versions histogram:")
    for n, cnt in sorted(versions_hist.items()):
        print(f"  {n:>3}: {cnt}")
    print("per-venue counts:")
    for venue, cnt in sorted(venue_counts.items()):
        print(f"  {venue:<22} {cnt}")


if __name__ == "__main__":
    main()
