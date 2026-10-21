"""Churn metrics between adjacent scrubbed versions of each pilot paper.

For every manifest paper and every adjacent version pair (v_i, v_{i+1}) whose
scrubbed files both exist under --texts, tokenizes both texts and compares them
with difflib.SequenceMatcher (autojunk=False). churn = 1 - ratio, classed as
cosmetic (< 0.02), minor (< 0.10) or major (>= 0.10). Doubles as the
extraction-consistency check: a pair with ratio < 0.30 almost certainly means
one version's extraction went wrong (e.g. main-tex mis-pick), and is warned
about loudly.

The output JSONL is rewritten from scratch each run (no checkpointing - the
whole pilot diffs in seconds).

Run from repo root: python3 src/process/diff_versions.py
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, "src/collect")
from util import read_jsonl  # noqa: E402

RATIO_WARN = 0.30


def tokenize(text: str, by: str) -> list[str]:
    return text.splitlines() if by == "lines" else text.split()


def classify(churn: float) -> str:
    if churn < 0.02:
        return "cosmetic"
    if churn < 0.10:
        return "minor"
    return "major"


def main() -> None:
    ap = argparse.ArgumentParser(description="Diff adjacent scrubbed paper versions.")
    ap.add_argument("--texts", default="data/processed/texts/")
    ap.add_argument("--manifest", default="data/processed/pilot_manifest.jsonl")
    ap.add_argument("--out", default="data/processed/version_diffs.jsonl",
                    help="rewritten from scratch each run")
    ap.add_argument("--by", default="words", choices=["words", "lines"],
                    help="diff unit (the words_* fields count this unit)")
    args = ap.parse_args()

    manifest = read_jsonl(args.manifest)
    texts = Path(args.texts)

    pairs: list[tuple[str, dict, dict]] = []
    n_missing = 0
    for paper in manifest:
        versions = sorted(paper.get("versions", []), key=lambda v: v["v"])
        safe_id = paper["arxiv_id"].replace("/", "_")
        for v_from, v_to in zip(versions, versions[1:]):
            if all((texts / safe_id / f"v{v['v']}.md").exists() for v in (v_from, v_to)):
                pairs.append((paper["arxiv_id"], v_from, v_to))
            else:
                n_missing += 1

    rows: list[dict] = []
    for aid, v_from, v_to in tqdm(pairs, desc="diffs"):
        safe_id = aid.replace("/", "_")
        a = tokenize((texts / safe_id / f"v{v_from['v']}.md").read_text(encoding="utf-8"),
                     args.by)
        b = tokenize((texts / safe_id / f"v{v_to['v']}.md").read_text(encoding="utf-8"),
                     args.by)
        sm = SequenceMatcher(None, a, b, autojunk=False)
        added = removed = 0
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag in ("replace", "delete"):
                removed += i2 - i1
            if tag in ("replace", "insert"):
                added += j2 - j1
        ratio = sm.ratio()
        # Round BEFORE classifying so the stored churn and its class always agree:
        # e.g. ratio 0.9 gives 1.0 - 0.9 = 0.09999999999999998 in floats, which would
        # class as 'minor' although the true churn is exactly the 0.10 'major' boundary.
        churn = round(1.0 - ratio, 6)
        rows.append({
            "arxiv_id": aid,
            "v_from": v_from["v"],
            "v_to": v_to["v"],
            "stage_from": v_from.get("stage"),
            "stage_to": v_to.get("stage"),
            "ratio": round(ratio, 6),
            "churn": churn,
            "churn_class": classify(churn),
            "words_from": len(a),
            "words_to": len(b),
            "words_added": added,
            "words_removed": removed,
        })

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:  # OVERWRITE each run
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("\n=== diff_versions summary ===")
    print(f"papers: {len(manifest)} | adjacent pairs diffed: {len(rows)} | "
          f"pairs skipped (missing file): {n_missing} | unit: {args.by}")
    hist = Counter(r["churn_class"] for r in rows)
    print("churn_class: " + (", ".join(f"{k}={hist[k]}" for k in ("cosmetic", "minor", "major"))
                             if rows else "-"))
    by_stage: dict[tuple, list[float]] = defaultdict(list)
    for r in rows:
        by_stage[(r["stage_from"], r["stage_to"])].append(r["churn"])
    print("churn by stage transition:")
    for (sf, st), churns in sorted(by_stage.items(), key=lambda kv: str(kv[0])):
        print(f"  {sf} -> {st}: n={len(churns)} mean_churn={sum(churns) / len(churns):.4f}")
    if rows:
        print(f"mean ratio: {sum(r['ratio'] for r in rows) / len(rows):.4f}")

    suspect = [r for r in rows if r["ratio"] < RATIO_WARN]
    if suspect:
        print(f"\n!!! WARNING: {len(suspect)} pair(s) with ratio < {RATIO_WARN} - "
              f"likely EXTRACTION INCONSISTENCY (e.g. main-tex mis-pick in one version):")
        for r in suspect:
            print(f"  !!! {r['arxiv_id']} v{r['v_from']}->v{r['v_to']} "
                  f"ratio={r['ratio']:.3f} words {r['words_from']}->{r['words_to']}")


if __name__ == "__main__":
    main()
