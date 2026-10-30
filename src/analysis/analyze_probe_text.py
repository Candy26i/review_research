"""E4 say-do gap: do reviews VERBALLY flag injected corruption, and does the
score follow? Scans each probe arm's raw review JSONs for corruption-specific
complaints (garbled/implausible/doesn't-add-up/abstract-vs-table phrases) and
reports the verbal catch rate per variant against the base false-positive rate.

Run from repo root:
  python3 src/analysis/analyze_probe_text.py
"""
from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path

PAT = re.compile(
    r'garbled|nonsensical|implausible|corrupted|scrambled|mangled'
    r'|do(es)?\s?n.?t add up|not monotonic|non-monoton'
    r'|impossible (value|number|entr)'
    r'|(numbers?|values?|entries) .{0,30}(garbled|corrupt|nonsens|implaus|scrambl|random|arbitrary)'
    r'|(abstract|text) (says|reports|states|claims) .{0,40}(table|but)'
    r'|internally inconsistent (number|result|value)', re.I)

ARMS = ('claude-opus-5-5-probes', 'deepseek-chat-probes',
        'deepseek-reasoner-probes', 'gemini-3.8-flash-probes')
VARIANTS = ('base', 'corrupt_numbers', 'shuffle_paragraphs', 'typos',
            'pad_filler', 'drop_appendix')


def review_text(body: dict) -> str | None:
    try:
        content = body['choices'][0]['message']['content']
        rev = json.loads(content.strip().strip('`').replace('json\n', '', 1))
    except Exception:
        return None
    if not isinstance(rev, dict):
        return None
    parts = [rev.get('summary', '')]
    for f in ('weaknesses', 'questions', 'strengths'):
        v = rev.get(f)
        if isinstance(v, list):
            parts += [str(x) for x in v]
    return ' '.join(parts)


def scan(raw_root: Path, arm: str, variant: str) -> tuple[int, int]:
    hits = total = 0
    for f in sorted(glob.glob(str(raw_root / arm / f'*_{variant}v*.json'))):
        with open(f, encoding='utf-8') as fh:
            body = json.load(fh)
        text = review_text(body)
        if text is None:
            continue
        total += 1
        hits += bool(PAT.search(text))
    return hits, total


def main() -> None:
    ap = argparse.ArgumentParser(description='E4 verbal-detection (say-do gap) scan.')
    ap.add_argument('--raw-root', default='data/processed/reviews')
    ap.add_argument('--out', default='results/e4_say_do_gap.md')
    args = ap.parse_args()

    root = Path(args.raw_root)
    lines = ['# E4 say-do gap: verbal corruption detection vs score consequence',
             '',
             'Share of reviews whose text explicitly complains about corruption-class',
             'problems (regex over summary/strengths/weaknesses/questions), per arm and',
             'variant. `base` is the false-positive rate of the same regex.',
             '',
             '| arm | ' + ' | '.join(VARIANTS) + ' |',
             '|---|' + '---|' * len(VARIANTS)]
    for arm in ARMS:
        cells = []
        for variant in VARIANTS:
            h, t = scan(root, arm, variant)
            cells.append(f'{h}/{t} ({h / t * 100:.0f}%)' if t else '—')
        lines.append(f'| {arm.removesuffix("-probes")} | ' + ' | '.join(cells) + ' |')
        print(lines[-1])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(f'-> {out}')


if __name__ == '__main__':
    main()
