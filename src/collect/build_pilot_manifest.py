"""Pick a stratified pilot sample from arXiv version chains.

Strata are decision (accepted/rejected) x version bucket ('2' vs '3+'). Slots are
allocated one at a time round-robin over a fixed stratum order, so the selection
for a larger --target is a superset of the selection for a smaller one. Within a
stratum, chains spanning >= 2 distinct coarse stages come first, then the rest,
each group ordered by sha256(f"{seed}:{arxiv_id}").

Usage: python3 src/collect/build_pilot_manifest.py [--chains C] [--out O]
       [--target N] [--min-versions M] [--seed S]
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from util import read_jsonl

STRATA = [("accepted", "2"), ("accepted", "3+"), ("rejected", "2"), ("rejected", "3+")]


def rank_key(seed: int, arxiv_id: str) -> str:
    return hashlib.sha256(f"{seed}:{arxiv_id}".encode()).hexdigest()


def allocate(avail: dict[tuple[str, str], int], target: int) -> dict[tuple[str, str], int]:
    """Round-robin one slot at a time over STRATA; the slot sequence is independent
    of target, so a larger target takes a strict prefix extension (superset)."""
    take = {k: 0 for k in STRATA}
    total = 0
    while total < target:
        progressed = False
        for k in STRATA:
            if total >= target:
                break
            if take[k] < avail[k]:
                take[k] += 1
                total += 1
                progressed = True
        if not progressed:
            break
    return take


def main() -> None:
    ap = argparse.ArgumentParser(description="stratified pilot sample from version chains")
    ap.add_argument("--chains", default="data/processed/version_chains.jsonl")
    ap.add_argument("--out", default="data/processed/pilot_manifest.jsonl")
    ap.add_argument("--target", type=int, default=50)
    ap.add_argument("--min-versions", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    chains = read_jsonl(args.chains)
    strata: dict[tuple[str, str], list[dict]] = {k: [] for k in STRATA}
    n_eligible = 0
    for c in chains:
        if c["n_versions"] < args.min_versions or c["decision"] not in ("accepted", "rejected"):
            continue
        n_eligible += 1
        bucket = "3+" if c["n_versions"] >= 3 else "2"
        strata[(c["decision"], bucket)].append(c)
    for rows in strata.values():
        rows.sort(key=lambda c: (len({v["stage"] for v in c["versions"]}) < 2,
                                 rank_key(args.seed, c["arxiv_id"])))

    avail = {k: len(v) for k, v in strata.items()}
    take = allocate(avail, args.target)
    ideal = allocate({k: args.target for k in STRATA}, args.target)  # quota if every stratum were full

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_selected = 0
    versions_hist: Counter[int] = Counter()
    with out_path.open("w") as f:
        for k in STRATA:
            for c in strata[k][: take[k]]:
                rec = dict(c)
                rec["strata"] = {
                    "decision": k[0],
                    "version_bucket": k[1],
                    "spans_stages": len({v["stage"] for v in c["versions"]}) >= 2,
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_selected += 1
                versions_hist[c["n_versions"]] += 1

    print(f"{len(chains)} chains read, {n_eligible} eligible "
          f"(n_versions >= {args.min_versions}, decision accepted/rejected)")
    print(f"\n{'stratum':<14} {'available':>9} {'quota':>6} {'selected':>9}")
    for k in STRATA:
        name = f"{k[0]}/{k[1]}"
        print(f"{name:<14} {avail[k]:>9} {ideal[k]:>6} {take[k]:>9}")
        if avail[k] < ideal[k]:
            print(f"  shortfall: {name} has only {avail[k]} of {ideal[k]} (topped up from other strata)")
    print(f"\nSelected {n_selected} / target {args.target} -> {out_path}")
    if n_selected < args.target:
        print(f"WARNING: only {n_selected} eligible chains in total; target not met")
    print("\nn_versions histogram (selected):")
    for n, cnt in sorted(versions_hist.items()):
        print(f"  {n:>3}: {'#' * cnt} ({cnt})")


if __name__ == "__main__":
    main()
