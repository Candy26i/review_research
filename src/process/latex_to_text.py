"""Convert extracted per-version LaTeX sources into scrubbed, readable markdown.

For each ok row of the extraction log (data/processed/extracted.jsonl):
pick the main .tex (contains \\begin{document}), recursively flatten
\\input/\\include/\\subfile, strip comments, remove author/affiliation/
acknowledgment constructs at the LaTeX level (balanced-brace aware, so the
LLM reviewers can't see names, emails, or camera-ready markers), map the
structure (\\title/\\section/captions/bibliography) to markdown, then convert
the rest with pylatexenc. Falls back to a regex detex when pylatexenc fails
or loses too much text; kind=pdf rows go through pypdf page extraction.

Writes {out-dir}/{safe_id}/v{v}.md. Resumable via the log JSONL,
checkpointed on key "{arxiv_id}v{v}". One bad document never kills the run.

Run from repo root: python3 src/process/latex_to_text.py
"""
from __future__ import annotations

import argparse
import re
import signal
import sys
from collections import Counter
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, 'src/collect')
from util import append_jsonl, done_keys, read_jsonl  # noqa: E402

from pylatexenc.latex2text import LatexNodes2Text  # noqa: E402

MAX_FILE_BYTES = 8 * 1024 * 1024      # never inline a single file bigger than this
MAX_FLAT_CHARS = 30 * 1024 * 1024     # stop inlining once the flattened doc hits this
MAX_DEPTH = 24                        # \input nesting cap (cycle-safe anyway)
CONVERT_TIMEOUT_S = 240               # per-document pylatexenc budget
FALLBACK_RATIO = 0.4                  # pylatexenc output below this share of detex -> fallback
NEVER_INLINE = {'.sty', '.cls', '.bst', '.def', '.clo', '.bbx', '.cbx', '.dtx'}

BEGIN_DOC_RE = re.compile(r'\\begin\s*\{\s*document\s*\}')
ABSTRACT_ENV_RE = re.compile(
    r'\\begin\s*\{\s*abstract\s*\}.*?\\end\s*\{\s*abstract\s*\}', re.S)
INPUT_RE = re.compile(r'\\(?:input|include|subfile)\s*\{([^{}]+)\}')


# ---------------------------------------------------------------- low-level helpers

def read_tex_file(path: Path) -> str:
    data = path.read_bytes()[:MAX_FILE_BYTES]
    try:
        return data.decode('utf-8')
    except UnicodeDecodeError:
        return data.decode('latin-1', errors='replace')


def strip_line_comments(text: str) -> str:
    """Drop % comments to end of line, keeping escaped \\%."""
    out = []
    for line in text.split('\n'):
        i, cut = 0, None
        while i < len(line):
            c = line[i]
            if c == '\\':
                i += 2
                continue
            if c == '%':
                cut = i
                break
            i += 1
        out.append(line if cut is None else line[:cut])
    return '\n'.join(out)


def read_group(text: str, i: int, open_ch: str = '{', close_ch: str = '}'):
    """text[i] == open_ch. Return (content, index_after_close) or None if unbalanced."""
    depth, j, n = 0, i, len(text)
    while j < n:
        c = text[j]
        if c == '\\':
            j += 2
            continue
        if c == open_ch:
            depth += 1
        elif c == close_ch:
            depth -= 1
            if depth == 0:
                return text[i + 1:j], j + 1
        j += 1
    return None


def _skip_ws(text: str, i: int, max_newlines: int = 1) -> int:
    nl = 0
    while i < len(text) and text[i] in ' \t\n':
        if text[i] == '\n':
            nl += 1
            if nl > max_newlines:
                return i
        i += 1
    return i


def strip_commands(text: str, name_pat: str, max_braces: int, keep: str = '') -> str:
    """Remove \\name (optionally starred) plus its [..] options and up to
    max_braces balanced {..} argument groups. Brace-aware, multi-line safe."""
    rx = re.compile(r'\\(?:' + name_pat + r')\*?(?![a-zA-Z])')
    out, pos = [], 0
    while True:
        m = rx.search(text, pos)
        if not m:
            out.append(text[pos:])
            return ''.join(out)
        out.append(text[pos:m.start()])
        i, n_br, n_opt = m.end(), 0, 0
        while i < len(text):
            j = _skip_ws(text, i)
            if j < len(text) and text[j] == '[' and n_br == 0 and n_opt < 2:
                g = read_group(text, j, '[', ']')
                if g is None:
                    break
                i = g[1]
                n_opt += 1
                continue
            if j < len(text) and text[j] == '{' and n_br < max_braces:
                g = read_group(text, j)
                if g is None:
                    break
                i = g[1]
                n_br += 1
                continue
            break
        out.append(keep)
        pos = i


def replace_cmd_arg(text: str, name_pat: str, fmt) -> str:
    """Replace \\name[opt]{arg} with fmt(arg) (balanced). Commands without a
    braced arg are just removed."""
    rx = re.compile(r'\\(?:' + name_pat + r')\*?(?![a-zA-Z])')
    out, pos = [], 0
    while True:
        m = rx.search(text, pos)
        if not m:
            out.append(text[pos:])
            return ''.join(out)
        out.append(text[pos:m.start()])
        i = m.end()
        j = _skip_ws(text, i)
        if j < len(text) and text[j] == '[':
            g = read_group(text, j, '[', ']')
            if g is not None:
                i = g[1]
                j = _skip_ws(text, i)
        if j < len(text) and text[j] == '{':
            g = read_group(text, j)
            if g is not None:
                out.append(fmt(g[0]))
                pos = g[1]
                continue
        pos = i


def remove_envs(text: str, name_pat: str, repl: str = '') -> str:
    return re.sub(
        r'\\begin\s*\{\s*(?:' + name_pat + r')\s*\}.*?\\end\s*\{\s*(?:' + name_pat + r')\s*\}',
        repl, text, flags=re.S)


# ---------------------------------------------------------------- flatten

class Flattener:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.visited: set[Path] = set()
        self.missing: list[str] = []
        self.n_inlined = 0
        self.total_chars = 0

    def _resolve(self, name: str, from_dir: Path) -> Path | None:
        name = name.strip().replace('\\', '/')
        # kpathsea semantics: \input{foo.bar} tries foo.bar.tex before foo.bar,
        # so a dotted basename ('4.evaluation-self') still resolves to its .tex
        cands = [name] if name.lower().endswith('.tex') else [name + '.tex', name]
        for base in (from_dir, self.root):
            for nm in cands:
                try:
                    p = (base / nm).resolve()
                except OSError:
                    continue
                # containment: never follow paths escaping the extraction dir
                if p.is_file() and p.is_relative_to(self.root):
                    return p
        return None

    def flatten(self, path: Path, depth: int = 0) -> str:
        if depth > MAX_DEPTH:
            return ''
        text = strip_line_comments(read_tex_file(path))
        self.total_chars += len(text)

        def _inline(m: re.Match) -> str:
            name = m.group(1).strip()
            if Path(name).suffix.lower() in NEVER_INLINE:
                return ''
            p = self._resolve(name, path.parent)
            if p is None:
                self.missing.append(name)
                return f'[missing input: {name}]'
            if p in self.visited:
                return ''
            if self.total_chars > MAX_FLAT_CHARS:
                self.missing.append(f'{name} [size-capped]')
                return f'[missing input: {name}]'
            self.visited.add(p)
            self.n_inlined += 1
            return '\n' + self.flatten(p, depth + 1) + '\n'

        return INPUT_RE.sub(_inline, text)


# ---------------------------------------------------------------- macro expansion

# zero-argument \newcommand/\def text macros (method names, system names) would
# otherwise render as empty strings ('We introduce ()', headings '## ()')
NEWCMD_RE = re.compile(
    r'\\(?:re)?(?:new|provide)command\*?\s*'
    r'(?:\{\s*\\([a-zA-Z@]+)\s*\}|\\([a-zA-Z@]+))\s*(\[[^\]]*\])?\s*(?=\{)')
DEF_RE = re.compile(r'\\[egx]?def\s*\\([a-zA-Z@]+)\s*(?=\{)')
EXPAND_BLOCKLIST = {
    'title', 'author', 'section', 'subsection', 'subsubsection', 'paragraph',
    'caption', 'abstract', 'thanks', 'email', 'footnote', 'cite', 'citep',
    'citet', 'ref', 'eqref', 'url', 'href', 'input', 'include', 'bibliography',
    'appendix', 'maketitle', 'label', 'item', 'and', 'keywords', 'date',
    'begin', 'end', 'documentclass', 'usepackage',
    'year', 'month', 'day', 'today', 'empty',
}


def collect_simple_macros(text: str) -> dict[str, str]:
    defs: dict[str, str] = {}
    for rx in (NEWCMD_RE, DEF_RE):
        for m in rx.finditer(text):
            if rx is NEWCMD_RE:
                name = m.group(1) or m.group(2)
                if m.group(3):            # parameterized ([1] etc.): skip
                    continue
            else:
                name = m.group(1)
            if not name or len(name) < 2 or name.lower() in EXPAND_BLOCKLIST:
                continue
            # status toggles (\iclrfinalcopy, \acceptedtrue, ...) are handled by
            # detect_remove_finalcopy and must never be expanded as text
            if re.search(r'final|accept|camera|anonym|preprint|review|submi',
                         name, re.IGNORECASE):
                continue
            g = read_group(text, m.end())
            if g is None:
                continue
            body = g[0]
            if '#' in body or len(body) > 120:
                continue
            if re.search(r'\\(?:(?:re)?new|provide)command|\\[egx]?def(?![a-zA-Z])|'
                         r'\\(?:input|include)(?![a-zA-Z])', body):
                continue
            body = re.sub(r'\\xspace(?![a-zA-Z])', '', body).strip()
            defs.setdefault(name, body)
    return defs


def collect_tree_macros(root: Path) -> dict[str, str]:
    """Simple zero-arg macros defined in the tree's .sty files (method names
    sometimes live there; .sty files are never inlined, so collect only)."""
    defs: dict[str, str] = {}
    for p in sorted(root.rglob('*.sty')):
        try:
            if p.stat().st_size > 300_000:
                continue
            txt = strip_line_comments(read_tex_file(p))
        except OSError:
            continue
        for k, v in collect_simple_macros(txt).items():
            defs.setdefault(k, v)
    return defs


def expand_simple_macros(text: str, extra_defs: dict[str, str] | None = None) -> tuple[str, int]:
    """Expand zero-argument text macros so method/system names survive conversion."""
    defs = dict(extra_defs or {})
    defs.update(collect_simple_macros(text))  # in-document definitions win
    total = 0
    for _ in range(3):                     # bounded, handles nested simple macros
        changed = False
        for name, body in defs.items():
            rx = re.compile(r'\\' + re.escape(name) + r'(?![a-zA-Z@])')
            text, n = rx.subn(lambda _m, _b=body: _b, text)
            if n:
                total += n
                changed = True
        if not changed:
            break
    return text, total


# ---------------------------------------------------------------- strip / scrub

def strip_comment_blocks(text: str) -> str:
    text = remove_envs(text, r'comment')
    text = re.sub(r'\\iffalse.*?\\fi', '', text, flags=re.S)           # best-effort
    text = re.sub(r'\\makeatletter.*?\\makeatother', ' ', text, flags=re.S)
    return text


# (name regex, max braced args) — all removed together with their arguments.
SCRUB_CMDS = [
    (r'authors?', 1), (r'icmlauthor', 2), (r'icmlaffiliation', 2),
    (r'icmlcorrespondingauthor', 2), (r'icmladdress', 1), (r'icmlemail', 1),
    (r'icmlsetsymbol', 2), (r'icmltitlerunning', 1), (r'correspondingauthor', 2),
    (r'thanks', 1), (r'email', 1), (r'ead', 1), (r'affiliation', 1), (r'affil', 1),
    (r'institute', 1), (r'institution', 1), (r'address', 1), (r'authornote', 1),
    (r'authorrunning', 1), (r'titlerunning', 1), (r'orcid', 1), (r'homepage', 1),
    (r'equalcontrib(?:ution)?', 0),
    # elsarticle author-block commands (\cortext[cor1]{Corresponding author} etc.)
    (r'cortext', 1), (r'fntext', 1), (r'tnotetext', 1),
    (r'corref', 1), (r'fnref', 1), (r'tnoteref', 1),
    # running heads / pdf metadata can carry names and venue strings
    (r'date', 1), (r'hypersetup', 1), (r'markboth', 2), (r'markright', 1),
    (r'fancyhead', 1), (r'fancyfoot', 1), (r'fancyhf', 1),
    (r'[lcr]head', 1), (r'[lcr]foot', 1),
    # \thispagestyle{empty} etc. otherwise leak their argument as stray text
    (r'(?:this)?pagestyle', 1),
    # template keyword lines leak the TARGET venue ('Machine Learning, ICML')
    # on post-decision resubmissions — a direct outcome tell
    (r'icmlkeywords', 1), (r'keywords', 1),
    # minitoc / toc furniture renders as '.tocmtchapter'-style junk
    (r'dominitoc|minitoc|faketableofcontents|tableofcontents|mtcset[a-zA-Z]*', 2),
]

ACK_ENV_PAT = r'ack|acks|acknowledgments?|acknowledgements?|icmlauthorlist'
ACK_TITLE_RE = re.compile(r'acknowledg|author contribution|funding', re.I)
# ICLR's own style sets acknowledgments as \subsubsection*, and papers also use
# \paragraph{Acknowledgements}: match every heading level, incl. [short] titles.
HEAD_RE = re.compile(
    r'\\(section|subsection|subsubsection|paragraph)\*?\s*(?:\[[^\]]*\]\s*)?\{')

_BOUND_COMMON = (r'\\appendix(?![a-zA-Z])|\\end\s*\{\s*document\s*\}|'
                 r'\\bibliography(?![a-zA-Z])|\\printbibliography(?![a-zA-Z])|'
                 r'\\begin\s*\{\s*thebibliography')
# the removed body runs to the next heading of the same or higher level
BOUNDS = {
    'section': re.compile(r'\\section\*?(?![a-zA-Z])|' + _BOUND_COMMON),
    'subsection': re.compile(r'\\(?:sub)?section\*?(?![a-zA-Z])|' + _BOUND_COMMON),
    'subsubsection': re.compile(r'\\(?:sub){0,2}section\*?(?![a-zA-Z])|' + _BOUND_COMMON),
    'paragraph': re.compile(
        r'\\(?:sub){0,2}section\*?(?![a-zA-Z])|\\(?:sub)?paragraph\*?(?![a-zA-Z])|'
        + _BOUND_COMMON),
}


def remove_ack_sections(text: str) -> str:
    """Drop headings (\\section...\\paragraph) with acknowledgment-like titles
    plus their bodies, up to the next same-or-higher heading / appendix /
    bibliography."""
    for _ in range(40):  # safety bound
        hit = None
        for m in HEAD_RE.finditer(text):
            g = read_group(text, m.end() - 1)
            if g is None:
                continue
            if ACK_TITLE_RE.search(g[0]):
                hit = (m, g)
                break
        if hit is None:
            return text
        m, g = hit
        bound = BOUNDS[m.group(1)]
        nxt = bound.search(text, g[1])
        end = nxt.start() if nxt else len(text)
        text = text[:m.start()] + '\n' + text[end:]
    return text


FOOTNOTE_CMD_RE = re.compile(r'\\[a-zA-Z]*footnote(?:text|mark)?\*?(?![a-zA-Z])')
AUTHORISH_RE = re.compile(
    r'correspond|equal(?:ly)?[\s~-]*(?:contrib|advis)|work\s+(?:was\s+)?done|'
    r'intern(?:ship)?\s+at|these\s+authors|on\s+leave|now\s+(?:at|with)|'
    r'author\s+list|senior\s+(?:lead|author)|'
    r'@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}|\.edu\b|\.ac\.[a-z]{2}\b', re.I)


def scrub_author_footnotes(text: str) -> str:
    """Drop footnote-family commands (incl. custom \\nnfootnote etc.) whose
    content is authorship/correspondence metadata; keep content footnotes.
    Custom two-arg forms like \\customfootnotetext{1}{Equal contribution.} put
    the metadata in a SECOND group behind a marker-like first one — when the
    first group is a short marker, the second group is inspected too."""
    out, pos = [], 0
    while True:
        m = FOOTNOTE_CMD_RE.search(text, pos)
        if not m:
            out.append(text[pos:])
            return ''.join(out)
        out.append(text[pos:m.start()])
        j = _skip_ws(text, m.end())
        if j < len(text) and text[j] == '[':
            g = read_group(text, j, '[', ']')
            if g is not None:
                j = _skip_ws(text, g[1])
        if j < len(text) and text[j] == '{':
            g = read_group(text, j)
            if g is not None:
                if AUTHORISH_RE.search(g[0]):
                    pos = g[1]  # drop command + argument
                    continue
                if len(g[0].strip()) <= 3:  # marker arg ({1}, {*}, {†}, ...)
                    j2 = _skip_ws(text, g[1])
                    if j2 < len(text) and text[j2] == '{':
                        g2 = read_group(text, j2)
                        if g2 is not None and AUTHORISH_RE.search(g2[0]):
                            pos = g2[1]  # drop command + both groups
                            continue
        out.append(text[m.start():m.end()])
        pos = m.end()


CENTERISH_ENV_RE = re.compile(
    r'\\begin\s*\{\s*(center|minipage|tabular)\s*\}.*?\\end\s*\{\s*\1\s*\}', re.S)
SECTION_CMD_RE = re.compile(r'\\section\*?(?![a-zA-Z])')


def remove_author_title_blocks(text: str) -> str:
    """Hand-typeset title pages (no \\author macro): drop center/minipage/tabular
    environments that sit between \\begin{document} and the abstract/first
    section and whose content looks like an author block (email, .edu,
    correspondence, ...). Anchored at \\begin{document} because preamble macro
    soup (\\let\\se=\\section) would otherwise pull the cut before the block."""
    doc = BEGIN_DOC_RE.search(text)
    base = doc.end() if doc else 0
    stops = [m.start() for m in (ABSTRACT_ENV_RE.search(text, base),
                                 SECTION_CMD_RE.search(text, base)) if m]
    cut = min(stops) if stops else len(text)

    def _repl(m: re.Match) -> str:
        if base <= m.start() < cut and AUTHORISH_RE.search(m.group(0)):
            return '\n'
        return m.group(0)

    return CENTERISH_ENV_RE.sub(_repl, text)


FINAL_RES = [
    re.compile(r'\\iclrfinalcopy(?![a-zA-Z])'),
    re.compile(r'\\(?:acl|naacl|emnlp|aaai|cvpr|neurips|nips)?finalcopy(?![a-zA-Z])'),
    re.compile(r'\\final(?:true|version)?(?![a-zA-Z])'),
    re.compile(r'\\(?:accepted|cameraready|camera)true(?![a-zA-Z])'),
]
PKG_OPTS_RE = re.compile(r'(\\(?:usepackage|documentclass)\s*\[)([^\]]*)(\])(\s*\{[^}]*\})?')
ACCEPTED_OPT_RE = re.compile(r'\baccepted\b', re.I)
# 'final' is camera-ready only on a conference/journal style ([final]{acl}),
# never on e.g. [final]{microtype} or [final]{graphicx}
FINAL_OPT_RE = re.compile(r'\bfinal\b', re.I)
CONF_PKG_RE = re.compile(
    r'iclr|icml|neurips|nips|acl|emnlp|naacl|aistats|colm|corl|rlc|uai|tmlr', re.I)


def detect_remove_finalcopy(text: str) -> tuple[str, bool]:
    flag = False
    for rx in FINAL_RES:
        text, n = rx.subn('', text)
        flag = flag or n > 0

    def _is_final(opt: str, pkg: str) -> bool:
        return bool(ACCEPTED_OPT_RE.search(opt)
                    or (FINAL_OPT_RE.search(opt) and CONF_PKG_RE.search(pkg)))

    def _opts(m: re.Match) -> str:
        nonlocal flag
        opts, pkg = m.group(2), m.group(4) or ''
        if _is_final(opts, pkg):
            flag = True
            kept = ','.join(o for o in opts.split(',') if not _is_final(o, pkg))
            return m.group(1) + kept + m.group(3) + (m.group(4) or '')
        return m.group(0)

    text = PKG_OPTS_RE.sub(_opts, text)
    return text, flag


def scrub_anonymity(text: str) -> str:
    text = remove_envs(text, ACK_ENV_PAT)
    for pat, n_br in SCRUB_CMDS:
        text = strip_commands(text, pat, n_br)
    text = scrub_author_footnotes(text)
    text = remove_author_title_blocks(text)
    text = remove_ack_sections(text)
    # \def\cmd{...} definitions otherwise leak their bodies as stray text
    text = strip_commands(text, r'def\s*\\[a-zA-Z@]+\s*(?:#\d)*', 1)
    return text


# ---------------------------------------------------------------- structure map

# sentinels that survive both pylatexenc and the detex fallback
_HB, _HE = 'ZZQHDR', 'ZZQENDQZZ'
TITLE_RE = re.compile(r'\\(?:title|icmltitle)\*?\s*(?:\[[^\]]*\]\s*)?\{')


def extract_title(text: str) -> str | None:
    m = TITLE_RE.search(text)
    if not m:
        return None
    g = read_group(text, m.end() - 1)
    return g[0].strip() if g and g[0].strip() else None


def _hdr(level: int, content: str) -> str:
    return f'\n\n{_HB}{level}QZZ {content} {_HE}\n\n'


def rewrite_href(text: str) -> str:
    """\\href[opt]{url}{label} -> 'label <url>'. pylatexenc 2.11's default
    \\href spec indexes a second argument its walker never parses and raises
    IndexError, losing the whole document to the detex fallback; rewriting the
    command away avoids that. mailto: links are dropped (identity scrub)."""
    rx = re.compile(r'\\href(?![a-zA-Z])')
    out, pos = [], 0
    while True:
        m = rx.search(text, pos)
        if not m:
            out.append(text[pos:])
            return ''.join(out)
        out.append(text[pos:m.start()])
        i = m.end()
        j = _skip_ws(text, i)
        if j < len(text) and text[j] == '[':
            g = read_group(text, j, '[', ']')
            if g is not None:
                j = _skip_ws(text, g[1])
        url = label = None
        if j < len(text) and text[j] == '{':
            g = read_group(text, j)
            if g is not None:
                url = g[0].strip()
                i = g[1]
                j2 = _skip_ws(text, i)
                if j2 < len(text) and text[j2] == '{':
                    g2 = read_group(text, j2)
                    if g2 is not None:
                        label = g2[0].strip()
                        i = g2[1]
        if url is None or url.lower().startswith('mailto:') or '@' in url:
            pos = i  # malformed or an email link: drop it entirely
            continue
        out.append(f'{label} <{url}>' if label else f'<{url}>')
        pos = i


IMG_PATH_GROUP_RE = re.compile(
    r'\{\s*[^{}\s,=\[\]]+\.(?:pdf|png|jpe?g|eps|gif|bmp|tiff?)\s*\}', re.I)

# env optional args ([H], [1], [colback=...]) otherwise render as bracket junk
ENV_OPTS_RE = re.compile(
    r'(\\begin\s*\{\s*(?:tcolorbox|mdframed|shaded|algorithm[2e*]*|algorithmic|'
    r'lstlisting|minted|wrapfigure|wraptable|figure\*?|table\*?|tabular[xy*]?|'
    r'longtable)\s*\})\s*\[[^\]]*\]')
VSKIP_RE = re.compile(r'\\[hv]skip\s*-?\d*\.?\d+\s*(?:in|pt|cm|mm|em|ex)\b')
CMIDRULE_RE = re.compile(
    r'\\(?:cmidrule|cline)\*?(?:\([^)]*\))?(?:\[[^\]]*\])?\s*\{[^{}]*\}')
TABULAR_BEGIN_RE = re.compile(r'\\begin\s*\{\s*(tabular[xy*]?|array|longtable)\s*\}')


def clean_tables(text: str) -> str:
    """Drop tabular layout noise that otherwise lands in the text: rules,
    column specs, and \\multicolumn/\\multirow wrappers (their body is kept)."""
    text = CMIDRULE_RE.sub(' ', text)
    for cmd in ('multicolumn', 'multirow'):
        rx = re.compile(r'\\' + cmd + r'(?![a-zA-Z])')
        out, pos = [], 0
        while True:
            m = rx.search(text, pos)
            if not m:
                out.append(text[pos:])
                break
            out.append(text[pos:m.start()])
            i, groups = m.end(), []
            while len(groups) < 3 and i < len(text):
                j = _skip_ws(text, i)
                if j < len(text) and text[j] == '[':
                    g = read_group(text, j, '[', ']')
                    if g is None:
                        break
                    i = g[1]
                    continue
                if j < len(text) and text[j] == '{':
                    g = read_group(text, j)
                    if g is None:
                        break
                    groups.append(g[0])
                    i = g[1]
                    continue
                break
            out.append(' ' + groups[-1] + ' ' if groups else ' ')
            pos = i
        text = ''.join(out)
    # drop the column-spec group(s): \begin{tabular}{spec}, \begin{tabularx}{w}{spec}
    out, pos = [], 0
    while True:
        m = TABULAR_BEGIN_RE.search(text, pos)
        if not m:
            out.append(text[pos:])
            break
        out.append(text[pos:m.end()])
        i = m.end()
        n_groups = 2 if m.group(1).lower().startswith(('tabularx', 'tabulary')) else 1
        for _ in range(n_groups):
            j = _skip_ws(text, i)
            if j < len(text) and text[j] == '[':
                g = read_group(text, j, '[', ']')
                if g is not None:
                    i = g[1]
                    j = _skip_ws(text, i)
            if j < len(text) and text[j] == '{':
                g = read_group(text, j)
                if g is not None:
                    i = g[1]
        pos = i
    return ''.join(out)


VERB_RE = re.compile(r'\\verb\*?([^a-zA-Z0-9\s])(.*?)\1')


def map_structure(text: str) -> str:
    # pylatexenc 2.11 silently DROPS \texttt/\textsf/\verb content (inline code,
    # tool and dataset names) — unwrap them to backticked text first
    text = VERB_RE.sub(lambda m: '`' + m.group(2) + '`', text)
    text = replace_cmd_arg(text, r'text(?:tt|sf)|lstinline|codeword',
                           lambda a: '`' + a + '`')
    text = strip_commands(text, r'maketitle', 0)  # else renders placeholder + today's date
    text = strip_commands(text, r'includegraphics', 1)
    text = strip_commands(text, r'captionsetup', 1)  # else leaks 'labelfont=color=...' junk
    text = ENV_OPTS_RE.sub(r'\1', text)
    text = VSKIP_RE.sub(' ', text)
    text = clean_tables(text)
    # unknown custom figure macros (preamble-defined) render their braced image
    # paths as text; paths can carry venue strings like ICLR_2025_Template/x.jpg
    text = IMG_PATH_GROUP_RE.sub('{}', text)
    text = rewrite_href(text)
    text = remove_envs(text, r'tikzpicture')  # coordinate soup, never prose
    text = replace_cmd_arg(text, r'title|icmltitle', lambda a: _hdr(1, a))
    text = re.sub(r'\\begin\s*\{\s*abstract\s*\}', _hdr(2, 'Abstract'), text)
    text = re.sub(r'\\end\s*\{\s*abstract\s*\}', '\n\n', text)
    text = replace_cmd_arg(text, r'section', lambda a: _hdr(2, a))
    text = replace_cmd_arg(text, r'subsection', lambda a: _hdr(3, a))
    text = replace_cmd_arg(text, r'subsubsection', lambda a: _hdr(4, a))
    text = replace_cmd_arg(text, r'paragraph',
                           lambda a: f'\n\nZZQPARQZZ {a} {_HE} ')
    text = strip_commands(text, r'captionof', 1)  # drops {figure}; body text stays
    text = replace_cmd_arg(text, r'caption',
                           lambda a: f'\nZZQCAPQZZ {a} {_HE}\n')
    text = re.sub(r'\\appendix(?![a-zA-Z])', _hdr(2, 'Appendix'), text)
    refs = _hdr(2, 'References') + '(omitted)\n\n'
    text = remove_envs(text, r'thebibliography', refs)
    text = replace_cmd_arg(text, r'bibliography', lambda a: refs)
    text = strip_commands(text, r'printbibliography', 0, keep=refs)
    text = strip_commands(text, r'bibliographystyle', 1)
    return text


def finalize_markers(text: str) -> str:
    def hdr(m: re.Match) -> str:
        body = ' '.join(m.group(2).split())
        return '\n\n' + '#' * int(m.group(1)) + ' ' + body + '\n\n'

    text = re.sub(_HB + r'(\d)QZZ\s*(.*?)\s*' + _HE, hdr, text, flags=re.S)
    text = re.sub(r'ZZQPARQZZ\s*(.*?)\s*' + _HE,
                  lambda m: '\n\n**' + ' '.join(m.group(1).split()).rstrip('.') + '.** ',
                  text, flags=re.S)
    text = re.sub(r'ZZQCAPQZZ\s*(.*?)\s*' + _HE,
                  lambda m: '\nCaption: ' + ' '.join(m.group(1).split()) + '\n',
                  text, flags=re.S)
    text = re.sub(r'ZZQ[A-Z0-9]*QZZ', '', text)  # orphaned sentinels
    text = re.sub(r'[ \t]+\n', '\n', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip() + '\n'


# ---------------------------------------------------------------- converters

DETEX_DROP = [
    (r'usepackage|documentclass|bibliographystyle|bibliography|label|vspace|hspace|'
     r'setlength|addtolength|definecolor|pagestyle|thispagestyle|graphicspath|'
     r'usetikzlibrary|lstset|captionsetup|setcounter|addtocounter|numberwithin|'
     r'theoremstyle|crefname|Crefname|includegraphics|input|include|subfile|url', 1),
    (r'newcommand|renewcommand|providecommand|DeclareMathOperator|newtheorem|'
     r'newenvironment|renewenvironment', 2),
]


def detex_fallback(text: str) -> str:
    for pat, n_br in DETEX_DROP:
        text = strip_commands(text, pat, n_br)
    text = strip_commands(text, r'[cC]ite[a-zA-Z]*', 1, keep='<cit.>')
    text = strip_commands(text, r'(?:auto|name|page|eq|[cC])?ref', 1, keep='<ref>')
    text = re.sub(r'\\(?:begin|end)\s*\{[^}]*\}', '\n', text)
    text = re.sub(r'\\\\\*?(\[[^\]]*\])?', '\n', text)
    text = re.sub(r'\\([%&_#${}\s])', r'\1', text)
    text = re.sub(r'\\[a-zA-Z@]+\*?', '', text)
    text = text.translate(str.maketrans({'{': '', '}': '', '$': '', '~': ' ', '&': ' '}))
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text


class _Timeout(Exception):
    pass


def pylatexenc_convert(text: str) -> str:
    l2t = LatexNodes2Text(math_mode='text', keep_comments=False)
    if hasattr(signal, 'SIGALRM'):
        def _handler(signum, frame):
            raise _Timeout(f'pylatexenc exceeded {CONVERT_TIMEOUT_S}s')
        old = signal.signal(signal.SIGALRM, _handler)
        signal.setitimer(signal.ITIMER_REAL, CONVERT_TIMEOUT_S)
        try:
            return l2t.latex_to_text(text)
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, old)
    return l2t.latex_to_text(text)


def pdf_to_text(path: Path) -> str:
    from pypdf import PdfReader
    reader = PdfReader(str(path))
    pages = []
    for page in reader.pages:
        try:
            pages.append(page.extract_text() or '')
        except Exception:
            pages.append('')
    return ('\n\n'.join(pages)).strip() + '\n'


# ---------------------------------------------------------------- per-row pipeline

def pick_main_tex(root: Path) -> Path | None:
    cands = sorted(p for p in root.rglob('*')
                   if p.is_file() and p.suffix.lower() == '.tex')
    if not cands:
        return None
    with_doc = []
    for p in cands:
        try:
            if BEGIN_DOC_RE.search(strip_line_comments(read_tex_file(p))):
                with_doc.append(p)
        except OSError:
            continue
    pool = with_doc or cands
    if len(pool) == 1:
        return pool[0]
    # Several \begin{document} candidates (stale drafts happen): prefer the one
    # that actually stitches the paper together (most resolvable \input/\include),
    # then a main-ish name, then size.
    fl = Flattener(root)

    def n_resolvable_inputs(p: Path) -> int:
        try:
            txt = strip_line_comments(read_tex_file(p))
        except OSError:
            return 0
        n = 0
        for m in INPUT_RE.finditer(txt):
            name = m.group(1).strip()
            if Path(name).suffix.lower() in NEVER_INLINE:
                continue
            if fl._resolve(name, p.parent) is not None:
                n += 1
        return n

    def name_pref(p: Path) -> int:
        nm = p.name.lower()
        return 2 if 'main' in nm else (
            1 if any(k in nm for k in ('paper', 'conference')) else 0)

    return max(pool, key=lambda p: (n_resolvable_inputs(p), name_pref(p),
                                    p.stat().st_size))


def row_dir(row: dict) -> Path | None:
    for k in ('dir', 'extract_dir', 'out_dir', 'dest'):
        if row.get(k):
            return Path(row[k])
    return None


def row_pdf_path(row: dict) -> Path | None:
    for k in ('path', 'pdf_path', 'pdf', 'file'):
        if row.get(k) and str(row[k]).lower().endswith('.pdf'):
            return Path(row[k])
    d = row_dir(row)
    if d and d.is_dir():
        pdfs = sorted(d.rglob('*.pdf'))
        if pdfs:
            return pdfs[0]
    return None


def process_latex(root: Path) -> dict:
    main = pick_main_tex(root)
    if main is None:
        raise ValueError(f'no .tex files under {root}')
    fl = Flattener(root)
    fl.visited.add(main.resolve())
    text = fl.flatten(main)

    text = strip_comment_blocks(text)
    # detect camera-ready toggles BEFORE macro expansion: the .sty files define
    # \iclrfinalcopy etc. as zero-arg macros, and expanding them would hide the flag
    text, finalcopy = detect_remove_finalcopy(text)
    text, n_macros = expand_simple_macros(text, collect_tree_macros(root))
    text = scrub_anonymity(text)
    tex_title = extract_title(text)
    # drop the preamble: it converts to junk (newtheorem/tikz/lstset leftovers)
    m = BEGIN_DOC_RE.search(text)
    if m:
        pre, body = text[:m.start()], text[m.start():]
        # a few classes (e.g. googledeepmind) place the abstract environment
        # before \begin{document}; rescue it or the preamble drop deletes it
        if not ABSTRACT_ENV_RE.search(body):
            am = ABSTRACT_ENV_RE.search(pre)
            if am:
                bd = BEGIN_DOC_RE.match(body)
                at = bd.end() if bd else 0
                body = body[:at] + '\n' + am.group(0) + '\n' + body[at:]
        text = body
    text = map_structure(text)
    if f'{_HB}1QZZ' not in text and tex_title:
        text = _hdr(1, tex_title) + text

    detexed = detex_fallback(text)
    converter, converted = 'pylatexenc', None
    try:
        converted = pylatexenc_convert(text)
    except Exception:
        converted = None
    if converted is None or len(converted) < FALLBACK_RATIO * len(detexed):
        converter, converted = 'detex_fallback', detexed

    md = finalize_markers(converted)
    ct = clean_title(tex_title)
    if ct and not md.lstrip().startswith('# '):
        # title got mangled or sat in an unmapped construct: guarantee the H1
        md = f'# {ct}\n\n' + md
    return {
        'md': md,
        'main_tex': str(main.relative_to(root)),
        'converter': converter,
        'n_inputs_flattened': fl.n_inlined,
        'missing_inputs': fl.missing,
        'finalcopy_flag': finalcopy,
        'tex_title': ct,
        'n_macros_expanded': n_macros,
    }


def clean_title(raw: str | None) -> str | None:
    if not raw:
        return None
    # titles sometimes embed logos/spacing (\includegraphics, \vspace) that
    # would render as '< g r a p h i c s >' / '0pt1.2em' junk
    raw = strip_commands(
        raw, r'includegraphics|[hv]space|resizebox|scalebox|raisebox|includesvg', 2)
    try:
        t = LatexNodes2Text(math_mode='text').latex_to_text(raw)
    except Exception:
        t = detex_fallback(raw)
    t = re.sub(r'<\s*g\s*r\s*a\s*p\s*h\s*i\s*c\s*s\s*>', ' ', t)
    for _ in range(3):  # concatenated dims like '0pt1.2em' need repeated passes
        t, n = re.subn(r'-?\d+(?:\.\d+)?\s*(?:pt|in|cm|mm|em|ex)\b', ' ', t)
        if not n:
            break
    t = ' '.join(t.split()).strip(' *_–—:;,-')
    t = re.sub(r'^[^A-Za-z"“(]+', '', t)  # leading layout junk ('0pt1. ')
    return t or None


# ---------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser(
        description='Flatten, scrub, and convert extracted LaTeX sources to markdown.')
    ap.add_argument('--extracted', default='data/processed/extracted.jsonl')
    ap.add_argument('--out-dir', default='data/processed/texts_raw/')
    ap.add_argument('--log', default='data/processed/latex_log.jsonl')
    ap.add_argument('--limit', type=int, default=0, help='max docs this run (0 = no cap)')
    args = ap.parse_args()

    rows = read_jsonl(args.extracted)
    done = done_keys(args.log, 'key')
    out_root = Path(args.out_dir)

    n_proc = n_ckpt = n_notok = n_fail = 0
    converters: Counter = Counter()
    n_final = 0
    char_counts: list[int] = []

    for row in tqdm(rows, desc='latex_to_text'):
        if not row.get('ok'):
            n_notok += 1
            continue
        aid = row['arxiv_id']
        v = row['v']
        key = row.get('key') or f'{aid}v{v}'
        if key in done:
            n_ckpt += 1
            continue
        if args.limit and n_proc >= args.limit:
            break
        n_proc += 1

        log = {'key': key, 'arxiv_id': aid, 'v': v, 'main_tex': None,
               'converter': None, 'n_inputs_flattened': 0, 'missing_inputs': [],
               'finalcopy_flag': False, 'tex_title': None, 'n_macros_expanded': 0,
               'n_chars': 0, 'ok': False, 'error': None}
        try:
            if (row.get('kind') or '').lower() == 'pdf':
                pdf = row_pdf_path(row)
                if pdf is None or not pdf.is_file():
                    raise ValueError('kind=pdf but no .pdf file found for row')
                md = pdf_to_text(pdf)
                log['converter'] = 'pdf'
            else:
                d = row_dir(row)
                if d is None or not d.is_dir():
                    raise ValueError(f'extraction dir missing: {d}')
                res = process_latex(d)
                md = res.pop('md')
                log.update(res)

            safe_id = aid.replace('/', '_')
            out_path = out_root / safe_id / f'v{v}.md'
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(md, encoding='utf-8')
            log['n_chars'] = len(md)
            log['ok'] = True
        except Exception as exc:  # never let one document kill the run
            log['error'] = f'{type(exc).__name__}: {exc}'[:500]
            n_fail += 1
            tqdm.write(f'FAIL {key}: {log["error"]}')

        append_jsonl(args.log, log)
        done.add(key)
        if log['ok']:
            converters[log['converter']] += 1
            char_counts.append(log['n_chars'])
            if log['finalcopy_flag']:
                n_final += 1

    print('\n=== latex_to_text summary ===')
    print(f'rows: {len(rows)} | processed this run: {n_proc} | '
          f'skipped via checkpoint: {n_ckpt} | skipped not-ok: {n_notok} | failures: {n_fail}')
    print('converters: ' + (', '.join(f'{k}={v}' for k, v in sorted(converters.items())) or '-'))
    print(f'finalcopy_flag true: {n_final} of {sum(converters.values())} converted docs')
    if char_counts:
        cs = sorted(char_counts)
        if len(cs) >= 4:
            import statistics
            q1, med, q3 = statistics.quantiles(cs, n=4)
            print(f'n_chars quartiles: min={cs[0]} q1={q1:.0f} median={med:.0f} '
                  f'q3={q3:.0f} max={cs[-1]}')
        else:
            print(f'n_chars: {cs}')


if __name__ == '__main__':
    main()
