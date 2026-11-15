"""Extract the overall ICLR rating a child stated in its free-text report.

Returns a dict: rating (point value or None), method, snippet, lo/hi when the child gave a
range, n_distinct (distinct values among same-class matches).  Patterns were developed on
the snapshot and every non-JSON extraction was read by hand (see audit notes)."""
import re
import b1lib as L

_SEP = r'\s*(?:\((?:[^)]{0,60})\))?\s*\**\s*(?:[:=]|is|of|would be|->|→|—|-|–|≈|~|likely|around|roughly|about)?\s*\**\s*[~≈]?\s*'
PATS = [
    ('json_key', re.compile(r'"rating"\s*:\s*"?(10|[1-9])\b')),
    ('rating_kv', re.compile(
        r'(?i)(?<![a-z])rating\b(?!s?\s*(?:scale|on\b))(?:\s+(?:likely|around|roughly|about|≈))?'
        + _SEP + r'(10|[1-9])(?![\d.])')),
    ('recommend_kv', re.compile(
        r'(?i)\b(?:recommendation|overall(?:\s+(?:score|assessment|recommendation|judg(?:e)?ment|verdict))?|verdict)\b'
        r'\s*\**\s*(?:[:=]|is|—|-|–)?\s*\**\s*(?:[a-z ,\-]{0,30}?)(10|[1-9])(?![\d.])'
        r'\s*(?:/\s*10|\(|—|–|-)\s*\**\s*(?:strong|reject|marginal|borderline|weak|accept|10)')),
    ('label_n', re.compile(
        r'(?i)\b(?:reject|accept|borderline|threshold)\w*\s*\((?:ICLR\s+|rating\s+|score\s+)?(10|[1-9])\)')),
    ('n_label', re.compile(
        r'(?i)(?<![\d./])(10|[1-9])\s*[\(—–-]\s*\**\s*(?:strong\s+reject|reject|marginally\s+below|'
        r'marginally\s+above|borderline|weak\s+(?:accept|reject)|accept|strong\s+accept)')),
]
_ORDER = {n: i for i, (n, _) in enumerate(PATS)}
_RANGE = re.compile(r'^\s*(?:[-–]|to\b)\s*(10|[1-9])(?![\d.]|\s*/\s*10)')
_LEAN = re.compile(r'(?i)lean(?:ing)?\s*:?\s*(?:toward\s+)?(10|[1-9])\b')


def from_text(rep: str) -> dict:
    cands = []
    for name, rx in PATS:
        for m in rx.finditer(rep):
            cands.append((m.start(), m.end(1), name, int(m.group(1))))
    out = {'rating': None, 'method': None, 'snippet': None, 'lo': None, 'hi': None,
           'n_distinct': 0, 'all': sorted({c[3] for c in cands})}
    if not cands:
        return out
    best = min(_ORDER[c[2]] for c in cands)
    same = [c for c in cands if _ORDER[c[2]] == best]
    start, end, name, val = same[0]          # the FIRST statement of the best class
    out.update(method=name, rating=val, n_distinct=len({c[3] for c in same}),
               snippet=rep[max(0, start - 70):end + 90].replace('\n', ' '))
    tail = rep[end:end + 60]
    m = _RANGE.match(tail)
    if m and int(m.group(1)) != val and name != 'json_key':
        lo, hi = sorted((val, int(m.group(1))))
        out.update(lo=lo, hi=hi, rating=None, method=name + '_range')
        lean = _LEAN.search(tail)
        if lean:
            out.update(rating=int(lean.group(1)), method=name + '_range_lean')
    return out


def child_rating(report: str) -> dict:
    j = L.parse_json_review(report)
    if j is not None:
        try:
            return {'rating': float(j['rating']), 'method': 'json_object', 'snippet': '',
                    'lo': None, 'hi': None, 'n_distinct': 1, 'all': [float(j['rating'])]}
        except (TypeError, ValueError):
            pass
    return from_text(report)
