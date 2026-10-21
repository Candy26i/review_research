"""Scrub converted per-version markdown texts and audit residual identity leaks.

For each manifest paper/version, reads {in_dir}/{safe_id}/v{v}.md, then:
  1. SCRUB: masks emails -> [EMAIL] and URLs -> [URL]; removes whole lines that
     match SPECIFIC venue/acceptance/camera-ready/reviewer-thanks leak phrases
     (lines merely containing 'reviewer' or 'rebuttal' are kept - several pilot
     papers are ABOUT peer review / LLM judges); normalizes whitespace.
  2. AUDIT (counts only, removes nothing): word-boundary, case-insensitive hits
     of each author surname (last whitespace token of each full name; surnames
     shorter than 4 chars are false-positive-prone and counted separately as
     "short_surname_hits"), plus flag terms. All audit counts run on the FINAL
     scrubbed text - what a downstream LLM reviewer would actually see - except
     'github.com', which is counted on the raw input BEFORE URL masking (the
     masking would otherwise always hide it).

Writes {out_dir}/{safe_id}/v{v}.md. Resumable: the log JSONL is append-only and
checkpointed on key "{arxiv_id}v{v}".

Run from repo root: python3 src/process/scrub_and_audit.py
"""
from __future__ import annotations

import argparse
import hashlib
import re
import sys
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, "src/collect")
from util import append_jsonl, done_keys, read_jsonl  # noqa: E402

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
URL_RE = re.compile(r"(?:https?://|www\.)[^\s<>()\"'\]]+", re.IGNORECASE)
# scheme-less repo/model-hub links ('github.com/kaistAI/...') name the org =
# the authors' lab; mask them like full URLs
BARE_DOMAIN_RE = re.compile(
    r"\b(?:github\.com|gitlab\.com|bitbucket\.org|huggingface\.co|"
    r"[\w-]+\.github\.io|[\w-]+\.hf\.space|anonymous\.4open\.science|osf\.io)"
    r"(?:/[\w\-./%#?=&+~:@]*)?", re.IGNORECASE)
# leftovers like '<https://>', '<[URL]>', '[URL]/path' after masking
URL_HUSK_RE = re.compile(r"<\s*(?:https?:)?/{0,2}\s*>|<\[URL\]>|\[URL\](?:/[\w\-./%#?=&+~:@]+)+")

# Whole-line removal: only these SPECIFIC leak phrases, nothing broader.
LEAK_LINE_RES = [
    re.compile(r"(published|accepted)\s+(as|at|in|to)[^.\n]{0,60}"
               r"(iclr|neurips|icml|conference|workshop)", re.IGNORECASE),
    re.compile(r"camera.?ready", re.IGNORECASE),
    re.compile(r"we\s+(would like to\s+)?thank\s+the\s+(anonymous\s+)?"
               r"(reviewer|area chair|ac\b)", re.IGNORECASE),
    re.compile(r"(response|revision|reply)\s+to\s+(the\s+)?reviewer", re.IGNORECASE),
]

# Flag terms are prefix-matched at a word boundary so 'reviewer' also counts
# 'reviewers' and 'acknowledg' counts 'acknowledgements'.
FLAG_TERMS = ["reviewer", "rebuttal", "iclr", "icml", "neurips", "arxiv", "acknowledg"]
FLAG_RES = {t: re.compile(rf"\b{t}", re.IGNORECASE) for t in FLAG_TERMS}
GITHUB_RE = re.compile(r"\bgithub\.com", re.IGNORECASE)

# --- header-region cleanup (first HEADER_LINES lines only, short lines only):
# symbol-marked affiliation footnotes and author-block furniture that survive
# the LaTeX-level scrub, plus template keyword lines naming a venue.
HEADER_LINES = 40
FURNITURE_RE = re.compile(
    r"^\s*[\*†‡§¶⋆\d\s,.&-]*(equal contribution|corresponding author|"
    r"work (?:was |partly )?done|these authors|now at |part of (?:this )?work)",
    re.IGNORECASE)
KEYWORDS_LINE_RE = re.compile(
    r"^[A-Za-z ,;-]{0,60}\b(icml|iclr|neurips|aaai|cvpr)\b[ ,.;-]*$", re.IGNORECASE)
AFFIL_WORDS_RE = re.compile(
    r"\b(universit\w+|institute?s?\b|laborator\w+|college|polytechnique?|"
    r"academy of|research (?:center|centre|lab)|school of|department of|"
    r"google|deepmind|microsoft|openai|anthropic|nvidia|ibm research|amazon|"
    r"meta ai|tencent|alibaba|baidu|huawei|bytedance|"
    r"kaist|mit\b|eth z\w+|epfl|stanford|princeton|berkeley|carnegie mellon|"
    r"cmu\b|tsinghua|peking|oxford|cambridge|mila\b|vector institute)",
    re.IGNORECASE)
# pure layout residue, dropped anywhere: lone brackets, bare dimensions,
# minitoc tokens, a stray 'empty' from pagestyle
LAYOUT_LINE_RE = re.compile(
    r"^\s*(\[|\]|\[?\s*-?\d*\.?\d+\s*(in|pt|cm|mm|em|ex)\s*\]?|"
    r"\.?(?:toc)?mt[a-z]{3,}|empty)\s*$", re.IGNORECASE)


def scrub(raw: str) -> tuple[str, dict]:
    """Mask emails/URLs, drop leak-phrase/header-furniture/layout lines,
    normalize whitespace."""
    text, n_emails = EMAIL_RE.subn("[EMAIL]", raw)
    text, n_urls = URL_RE.subn("[URL]", text)
    text, n_bare = BARE_DOMAIN_RE.subn("[URL]", text)
    n_urls += n_bare
    text = URL_HUSK_RE.sub("[URL]", text)
    text = re.sub(r"\[URL\](?:\s*\[URL\])+", "[URL]", text)

    kept: list[str] = []
    n_leak_lines = n_header_lines = n_layout_lines = 0
    for i, line in enumerate(text.split("\n")):
        if any(rx.search(line) for rx in LEAK_LINE_RES):
            n_leak_lines += 1
            continue
        if LAYOUT_LINE_RE.match(line):
            n_layout_lines += 1
            continue
        if i < HEADER_LINES and not line.startswith("#") and len(line) <= 90:
            if FURNITURE_RE.match(line) or KEYWORDS_LINE_RE.match(line):
                n_header_lines += 1
                continue
            if AFFIL_WORDS_RE.search(line) and len(line.split()) <= 8:
                n_header_lines += 1
                continue
        kept.append(line.rstrip())  # strip trailing spaces
    text = "\n".join(kept)

    # Collapse runs of 3+ blank lines to exactly 2 blank lines.
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    # Collapse repeated interior spaces; leading indentation is left alone so
    # markdown code blocks / tables keep their shape.
    text = re.sub(r"(?<=\S) {2,}", " ", text)
    text = text.strip("\n") + "\n"

    return text, {"emails": n_emails, "urls": n_urls, "leak_lines": n_leak_lines,
                  "header_lines": n_header_lines, "layout_lines": n_layout_lines}


def surnames_of(authors: list[str]) -> list[str]:
    """Last whitespace token of each full name, deduped case-insensitively."""
    seen: dict[str, str] = {}
    for name in authors:
        tokens = name.split()
        if tokens:
            seen.setdefault(tokens[-1].lower(), tokens[-1])
    return list(seen.values())


def audit(final_text: str, raw_text: str, surnames: list[str]) -> dict:
    """Count residual leak signals; nothing is removed here."""
    long_hits: dict[str, int] = {}
    short_hits: dict[str, int] = {}
    for sn in surnames:
        n = len(re.findall(rf"\b{re.escape(sn)}\b", final_text, re.IGNORECASE))
        if n:
            (short_hits if len(sn) < 4 else long_hits)[sn.lower()] = n
    flags: dict = {
        "author_surnames": long_hits,
        "short_surname_hits": short_hits,
    }
    for term, rx in FLAG_RES.items():
        flags[term] = len(rx.findall(final_text))
    flags["github.com"] = len(GITHUB_RE.findall(raw_text))  # before URL masking
    # institution mentions anywhere in the final text (audit only; prose like
    # 'labelers from KAIST' is not removed, just surfaced for later filtering)
    flags["institution_hits"] = len(AFFIL_WORDS_RE.findall(final_text))
    return flags


def main() -> None:
    ap = argparse.ArgumentParser(description="Scrub markdown texts and audit identity leaks.")
    ap.add_argument("--in-dir", dest="in_dir", default="data/processed/texts_raw/")
    ap.add_argument("--manifest", default="data/processed/pilot_manifest.jsonl")
    ap.add_argument("--papers", default="data/processed/iclr_papers.jsonl")
    ap.add_argument("--out-dir", default="data/processed/texts/")
    ap.add_argument("--log", default="data/processed/scrub_log.jsonl")
    ap.add_argument("--limit", type=int, default=0, help="max files this run (0 = no cap)")
    args = ap.parse_args()

    manifest = read_jsonl(args.manifest)
    authors_by_or = {p["openreview_id"]: p.get("authors", []) for p in read_jsonl(args.papers)}
    done = done_keys(args.log, "key")
    in_dir, out_dir = Path(args.in_dir), Path(args.out_dir)

    n_done = n_skipped = n_missing = n_no_authors = 0
    tot_removed = {"emails": 0, "urls": 0, "leak_lines": 0,
                   "header_lines": 0, "layout_lines": 0}
    docs_surname_hit = docs_short_hit = docs_leak_lines = 0

    tasks = [(p, v) for p in manifest for v in p.get("versions", [])]
    for paper, ver in tqdm(tasks, desc="scrub"):
        aid, vnum = paper["arxiv_id"], ver["v"]
        key = f"{aid}v{vnum}"
        if key in done:
            n_skipped += 1
            continue
        if args.limit and n_done >= args.limit:
            break
        safe_id = aid.replace("/", "_")
        src = in_dir / safe_id / f"v{vnum}.md"
        if not src.exists():
            tqdm.write(f"missing input (not checkpointed, rerun later): {src}")
            n_missing += 1
            continue

        authors = authors_by_or.get(paper["openreview_id"], [])
        if not authors:
            n_no_authors += 1
        raw = src.read_text(encoding="utf-8", errors="replace")
        text, removed = scrub(raw)
        flags = audit(text, raw, surnames_of(authors))

        dst = out_dir / safe_id / f"v{vnum}.md"
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(text, encoding="utf-8")

        append_jsonl(args.log, {
            "key": key,
            "arxiv_id": aid,
            "v": vnum,
            "n_chars": len(text),
            "n_words": len(text.split()),
            "est_tokens": len(text) // 4,
            "removed": removed,
            "leak_flags": flags,
            "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        })
        done.add(key)
        n_done += 1
        for k in tot_removed:
            tot_removed[k] += removed[k]
        docs_surname_hit += bool(flags["author_surnames"])
        docs_short_hit += bool(flags["short_surname_hits"])
        docs_leak_lines += bool(removed["leak_lines"])

    print("\n=== scrub_and_audit summary ===")
    print(f"version tasks: {len(tasks)} | scrubbed this run: {n_done} | "
          f"skipped via checkpoint: {n_skipped} | missing input: {n_missing} | "
          f"papers without author lookup: {n_no_authors}")
    print(f"removed: emails={tot_removed['emails']} urls={tot_removed['urls']} "
          f"leak_lines={tot_removed['leak_lines']} "
          f"header_lines={tot_removed['header_lines']} "
          f"layout_lines={tot_removed['layout_lines']}")
    print(f"docs with any author-surname hit (>=4 chars): {docs_surname_hit} | "
          f"docs with short-surname hits (<4 chars, noisy): {docs_short_hit} | "
          f"docs with leak-phrase lines removed: {docs_leak_lines}")


if __name__ == "__main__":
    main()
