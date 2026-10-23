"""Collect ACL Anthology volumes into match_arxiv-ready paper rows.

For each --volumes entry (e.g. 2025.acl-long) this fetches the full-volume
BibTeX export (verified live 2026-10-22: https://aclanthology.org/volumes/
2025.acl-short.bib returns the whole volume in one response) and falls back to
the volume HTML listing page if the .bib is missing or parses to zero papers.
Raw responses are cached under --cache-dir, so reruns make no network calls.
One row per paper is APPENDED to --out, checkpointed on openreview_id.

Output rows follow the iclr_papers schema expected by src/collect/match_arxiv.py
plus "venue"/"acl_id". Both "year" and "iclr_year" are the integer 0: 0 has no
entry in iclr_timeline.TIMELINES, so every downstream version stage is labeled
'unknown' (ACL papers must not pick up an ICLR timeline), and BOTH keys are
needed because match_arxiv.row_year evaluates
int(row.get("year") or row.get("iclr_year")) — a lone falsy year=0 would fall
through to int(None).

Skipped: volume frontmatter (@proceedings entries / anthology IDs ending in
".0") and entries without authors (keynotes etc.).

Run from repo root:
    python3 src/collect/collect_acl.py [--volumes 2025.acl-long 2025.acl-short
        findings-acl ...] [--out ...] [--cache-dir ...] [--limit N]
"""
from __future__ import annotations

import argparse
import html as html_lib
import re
import time
import unicodedata
from pathlib import Path

import requests

from util import append_jsonl, done_keys, norm_title, read_jsonl, session

ANTHOLOGY_BASE = "https://aclanthology.org"
ANTHOLOGY_DELAY_S = 3.0  # no published rate-limit terms; stay arXiv-polite anyway

DEFAULT_VOLUMES = ["2025.acl-long", "2025.acl-short", "2025.findings-acl"]


# ---------------------------------------------------------------- polite HTTP

class _Limiter:
    def __init__(self, min_interval_s: float):
        self.min_interval_s = min_interval_s
        self._last = 0.0

    def wait(self) -> None:
        sleep_for = self._last + self.min_interval_s - time.monotonic()
        if sleep_for > 0:
            time.sleep(sleep_for)
        self._last = time.monotonic()


def polite_get(url: str, limiter: _Limiter, retries: int = 3,
               backoff_s: float = 5.0) -> str | None:
    """GET with retries; returns None on 404 (so callers can fall back)."""
    last_exc: Exception | None = None
    for attempt in range(retries):
        limiter.wait()
        try:
            resp = session().get(url, timeout=60)
        except requests.RequestException as exc:
            last_exc = exc
            time.sleep(backoff_s * (attempt + 1))
            continue
        if resp.status_code == 404:
            return None
        if resp.status_code in (429, 500, 502, 503, 504):
            retry_after = float(resp.headers.get("Retry-After", backoff_s * (attempt + 1)))
            time.sleep(min(retry_after, 120))
            continue
        resp.raise_for_status()
        # Anthology serves .bib as bare "text/x-bibtex" (no charset), which
        # requests decodes as ISO-8859-1, mojibake-ing literal UTF-8 (e.g. the
        # title of 2025.acl-long.1517). Force UTF-8 unless a charset is declared.
        if "charset" not in resp.headers.get("Content-Type", "").lower():
            resp.encoding = "utf-8"
        return resp.text
    raise RuntimeError(f"GET {url} failed after {retries} attempts") from last_exc


def fetch_cached(url: str, cache_file: Path, limiter: _Limiter) -> str | None:
    """Return cached text if present, else fetch (caching 200s; 404 -> None)."""
    if cache_file.exists():
        return cache_file.read_text(encoding="utf-8")
    text = polite_get(url, limiter)
    if text is not None:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(text, encoding="utf-8")
    return text


# ------------------------------------------------------------- LaTeX -> text

_ACCENTS = {"'": "\u0301", "`": "\u0300", '"': "\u0308", "^": "\u0302",
            "~": "\u0303", "=": "\u0304", ".": "\u0307"}
_ACCENT_CMDS = {"v": "\u030c", "c": "\u0327", "u": "\u0306", "H": "\u030b",
                "k": "\u0328", "r": "\u030a", "b": "\u0331", "d": "\u0323"}
_WORD_MACROS = {"ss": "ß", "l": "ł", "L": "Ł", "o": "ø", "O": "Ø", "ae": "æ",
                "AE": "Æ", "oe": "œ", "OE": "Œ", "aa": "å", "AA": "Å",
                "i": "ı", "dj": "đ", "DJ": "Đ", "dh": "ð", "DH": "Ð",
                "th": "þ", "TH": "Þ", "ng": "ŋ", "NG": "Ŋ"}
_WORD_RE = re.compile(
    r"\\(" + "|".join(sorted(_WORD_MACROS, key=len, reverse=True)) + r")(?![A-Za-z])\s*")

# typographic -> ASCII quotes, so the bib and HTML parse paths emit identical text
_PUNCT_MAP = str.maketrans({"’": "'", "‘": "'", "ʼ": "'",
                            "“": '"', "”": '"', "„": '"'})


def normalize_punct(s: str) -> str:
    return s.translate(_PUNCT_MAP)


def delatex(s: str) -> str:
    """Decode the LaTeX escapes Anthology .bib exports use into plain Unicode."""
    def acc(mark: str):
        def sub(m: re.Match) -> str:
            return unicodedata.normalize("NFC", m.group(1) + mark)
        return sub

    prev = None
    while prev != s:  # accents may be nested in braces, e.g. {\'{e}}
        prev = s
        for ch, mark in _ACCENTS.items():
            s = re.sub(r"\\" + re.escape(ch) + r"\s*\{?([A-Za-z])\}?", acc(mark), s)
        for cmd, mark in _ACCENT_CMDS.items():
            s = re.sub(r"\\" + cmd + r"\s*\{([A-Za-z])\}", acc(mark), s)
            s = re.sub(r"\\" + cmd + r"\s+([A-Za-z])", acc(mark), s)
    s = _WORD_RE.sub(lambda m: _WORD_MACROS[m.group(1)], s)
    for esc, lit in ((r"\&", "&"), (r"\%", "%"), (r"\_", "_"),
                     (r"\$", "$"), (r"\#", "#")):
        s = s.replace(esc, lit)
    s = re.sub(r"\\[A-Za-z]+\s*", "", s)  # drop any remaining macro
    s = s.replace("``", '"').replace("''", '"').replace("`", "'")  # TeX quotes
    s = s.replace("{", "").replace("}", "").replace("~", " ")
    return re.sub(r"\s+", " ", normalize_punct(s)).strip()


# ------------------------------------------------------------- BibTeX parser

def _scan_value(s: str, i: int) -> tuple[str, int]:
    """Scan one BibTeX field value starting at s[i] -> (raw value, next index)."""
    n = len(s)
    while i < n and s[i] in " \t\r\n":
        i += 1
    if i < n and s[i] == '"':
        depth, j = 0, i + 1
        while j < n:
            c = s[j]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
            elif c == '"' and depth == 0:
                return s[i + 1:j], j + 1
            j += 1
        return s[i + 1:], n
    if i < n and s[i] == "{":
        depth, j = 0, i
        while j < n:
            if s[j] == "{":
                depth += 1
            elif s[j] == "}":
                depth -= 1
                if depth == 0:
                    return s[i + 1:j], j + 1
            j += 1
        return s[i + 1:], n
    m = re.match(r"[^,\n]*", s[i:])
    return m.group(0).strip(), i + m.end()


_ENTRY_HEAD = re.compile(r"@([A-Za-z]+)\s*\{\s*([^,\s]+)\s*,")
_FIELD_HEAD = re.compile(r"([A-Za-z][A-Za-z0-9_-]*)\s*=")


def parse_bibtex(text: str) -> list[dict]:
    """Tolerant parser for Anthology volume .bib exports.

    Returns [{'type': 'inproceedings', 'key': ..., 'fields': {name: value}}],
    field values whitespace-collapsed but still LaTeX-escaped.
    """
    entries: list[dict] = []
    pos = 0
    while True:
        at = text.find("@", pos)
        if at < 0:
            break
        m = _ENTRY_HEAD.match(text, at)
        if not m:
            pos = at + 1
            continue
        open_i = text.index("{", at)
        depth, j = 0, open_i
        while j < len(text):
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
                if depth == 0:
                    break
            j += 1
        body = text[m.end():j]
        fields: dict[str, str] = {}
        k, n = 0, len(body)
        while k < n:
            while k < n and body[k] in " \t\r\n,":
                k += 1
            fm = _FIELD_HEAD.match(body, k)
            if not fm:
                break
            val, k = _scan_value(body, fm.end())
            fields[fm.group(1).lower()] = " ".join(val.split())
        entries.append({"type": m.group(1).lower(), "key": m.group(2), "fields": fields})
        pos = j + 1
    return entries


def split_authors(author_field: str) -> list[str]:
    """BibTeX author field -> list of 'First Last' full names (de-LaTeXed)."""
    names = []
    for raw in re.split(r"\s+and\s+", author_field.strip()):
        name = delatex(raw)
        if not name:
            continue
        parts = [p.strip() for p in name.split(",") if p.strip()]
        if len(parts) == 2:          # Last, First
            name = f"{parts[1]} {parts[0]}"
        elif len(parts) > 2:         # Last, Jr., First -> First Last, Jr.
            name = f"{parts[-1]} {parts[0]}, {', '.join(parts[1:-1])}"
        else:
            name = " ".join(parts) if parts else ""
        if name:
            names.append(name)
    return names


_ID_FROM_URL = re.compile(r"aclanthology\.org/([^/\s]+)/?\s*$")


def bib_entry_to_paper(entry: dict) -> dict | None:
    """One @inproceedings entry -> {'acl_id', 'title', 'authors', 'abstract'}."""
    f = entry["fields"]
    acl_id = None
    m = _ID_FROM_URL.search(f.get("url", ""))
    if m:
        acl_id = m.group(1).rstrip("/")
    elif f.get("doi", "").startswith("10.18653/v1/"):
        acl_id = f["doi"].split("10.18653/v1/", 1)[1]
    if not acl_id:
        return None
    return {
        "acl_id": acl_id,
        "title": delatex(f.get("title", "")),
        "authors": split_authors(f.get("author", "")),
        "abstract": delatex(f.get("abstract", "")),
    }


# -------------------------------------------------------- HTML listing parser

_PAPER_BLOCK = re.compile(
    r"<strong><a[^>]*href=[\"']?/([0-9A-Za-z._\-]+)/[\"']?[^>]*>(.*?)</a></strong>"
    r"(.*?)(?=<strong><a|$)", re.S)
_PEOPLE_LINK = re.compile(r"<a[^>]*href=[\"']?/people/[^>]*>([^<]+)</a>")
_TAG = re.compile(r"<[^>]+>")


def parse_volume_html(html: str, volume: str) -> list[dict]:
    """Volume listing page -> paper dicts (same shape as bib_entry_to_paper)."""
    papers = []
    for acl_id, title_html, tail in _PAPER_BLOCK.findall(html):
        if not acl_id.startswith(volume + "."):
            continue
        title = normalize_punct(html_lib.unescape(_TAG.sub("", title_html)))
        title = re.sub(r"\s+", " ", title).strip()
        authors = [normalize_punct(html_lib.unescape(a)).strip()
                   for a in _PEOPLE_LINK.findall(tail)]
        papers.append({"acl_id": acl_id, "title": title,
                       "authors": [a for a in authors if a], "abstract": ""})
    return papers


# --------------------------------------------------------------------- main

def collect_volume(volume: str, cache_dir: Path, limiter: _Limiter) -> tuple[list[dict], str]:
    """Fetch+parse one volume -> (papers, source 'bib'|'html')."""
    bib_text = fetch_cached(f"{ANTHOLOGY_BASE}/volumes/{volume}.bib",
                            cache_dir / f"{volume}.bib", limiter)
    if bib_text:
        papers = [p for e in parse_bibtex(bib_text)
                  if e["type"] == "inproceedings" and (p := bib_entry_to_paper(e))]
        if papers:
            return papers, "bib"
    html = fetch_cached(f"{ANTHOLOGY_BASE}/volumes/{volume}/",
                        cache_dir / f"{volume}.html", limiter)
    if html is None:
        raise RuntimeError(f"volume {volume}: neither .bib export nor HTML page found")
    return parse_volume_html(html, volume), "html"


def main() -> None:
    ap = argparse.ArgumentParser(
        description="ACL Anthology volumes -> match_arxiv-ready paper rows.")
    ap.add_argument("--volumes", nargs="+", default=DEFAULT_VOLUMES)
    ap.add_argument("--out", default="data/processed/acl_papers.jsonl")
    ap.add_argument("--cache-dir", default="data/raw/anthology/")
    ap.add_argument("--limit", type=int, default=0,
                    help="max new rows written this run (0 = no cap)")
    args = ap.parse_args()

    cache_dir = Path(args.cache_dir)
    limiter = _Limiter(ANTHOLOGY_DELAY_S)
    done = done_keys(args.out, "openreview_id")

    total_new = 0
    per_volume: list[tuple[str, str, int, int, int, int]] = []
    capped = False
    for volume in args.volumes:
        papers, source = collect_volume(volume, cache_dir, limiter)
        n_new = n_done = n_skipped = 0
        for p in papers:
            if p["acl_id"].endswith(".0") or not p["authors"] or not p["title"]:
                n_skipped += 1  # frontmatter / keynote-style entries
                continue
            if p["acl_id"] in done:
                n_done += 1
                continue
            if args.limit and total_new >= args.limit:
                capped = True
                break
            append_jsonl(args.out, {
                "openreview_id": p["acl_id"],
                # year 0 on purpose: no ICLR timeline -> stages all 'unknown';
                # iclr_year 0 as well so match_arxiv.row_year never sees None.
                "year": 0,
                "iclr_year": 0,
                "title": p["title"],
                "title_norm": norm_title(p["title"]),
                "authors": p["authors"],
                "decision_raw": "published",
                "decision": "accepted",
                "scores": [],
                "n_reviews": 0,
                "mean_score": None,
                "abstract": p.get("abstract", ""),
                "venue": volume,
                "acl_id": p["acl_id"],
            })
            done.add(p["acl_id"])
            n_new += 1
            total_new += 1
        per_volume.append((volume, source, len(papers), n_skipped, n_done, n_new))
        if capped:
            break

    print("\n=== collect_acl summary ===")
    print(f"{'volume':<22} {'source':<6} {'parsed':>7} {'skipped':>8} "
          f"{'already':>8} {'new':>6}")
    for volume, source, n_parsed, n_skipped, n_done, n_new in per_volume:
        print(f"{volume:<22} {source:<6} {n_parsed:>7} {n_skipped:>8} "
              f"{n_done:>8} {n_new:>6}")
    if capped:
        print(f"(stopped at --limit {args.limit})")
    print(f"new rows this run: {total_new} | total rows in {args.out}: "
          f"{len(read_jsonl(args.out))}")


if __name__ == "__main__":
    main()
