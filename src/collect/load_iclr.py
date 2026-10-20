"""Load ICLR papers from the raw OpenReview parquet into a clean JSONL.

Usage: python3 src/collect/load_iclr.py [--parquet P] [--years Y ...] [--out O]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from util import norm_title


def normalize_decision(raw: str) -> str:
    d = raw.strip().lower()
    if d.startswith("accept"):
        return "accepted"
    if d == "reject":
        return "rejected"
    if "desk" in d:
        return "desk_rejected"
    if "withdrawn" in d:
        return "withdrawn"
    if "workshop" in d:
        return "workshop_invite"
    if d == "":
        return "pending"
    return "other"


def main() -> None:
    ap = argparse.ArgumentParser(description="parquet -> clean JSONL of ICLR papers")
    ap.add_argument("--parquet", default="data/raw/iclr/iclr25v2.parquet")
    ap.add_argument("--years", nargs="+", type=int, default=list(range(2019, 2026)))
    ap.add_argument("--out", default="data/processed/iclr_papers.jsonl")
    args = ap.parse_args()

    df = pd.read_parquet(args.parquet)
    df = df[df["year"].isin(args.years)]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    years_out: list[int] = []
    decisions_out: list[str] = []
    other_seen: set[str] = set()
    with out_path.open("w") as f:
        for row in tqdm(df.itertuples(index=False), total=len(df), desc="papers"):
            authors = row.authors if isinstance(row.authors, str) else ""
            scores = [int(s) for s in row.scores.tolist()]
            decision = normalize_decision(row.decision)
            if decision == "other":
                other_seen.add(row.decision)
            rec = {
                "openreview_id": row.id,
                "year": int(row.year),
                "title": row.title,
                "title_norm": norm_title(row.title),
                "authors": authors.split(", ") if authors else [],
                "decision_raw": row.decision,
                "decision": decision,
                "scores": scores,
                "n_reviews": len(scores),
                "mean_score": sum(scores) / len(scores) if scores else None,
                "abstract": row.abstract,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            years_out.append(int(row.year))
            decisions_out.append(decision)

    table = pd.crosstab(
        pd.Series(years_out, name="year"),
        pd.Series(decisions_out, name="decision"),
        margins=True,
        margins_name="total",
    )
    print(table.to_string())
    print(f"\nWrote {len(years_out)} papers (years {sorted(set(args.years))}) -> {out_path}")
    if other_seen:
        print(f"Decision values mapped to 'other': {sorted(other_seen)}")


if __name__ == "__main__":
    main()
