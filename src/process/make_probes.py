"""E4: length-matched, known-direction edit probes over the pilot papers' anchor texts.

Do LLM review scores track paper quality or document length? In the real version data
the two move together. E4 breaks the confound: for each selected paper's ANCHOR version
(latest pre_submission/during_review version, else v1 — same rule as analyze_scores.py)
it emits six deterministic variants whose quality direction is KNOWN and whose length
is CONTROLLED:

  variant             direction                 length vs base
  base                none (verbatim anchor)    0%
  shuffle_paragraphs  coherence DOWN            0%   (paragraphs permuted within sections)
  corrupt_numbers     soundness DOWN            0%   (digit-for-digit edits, 2nd half)
  typos               presentation DOWN         ~0%  (letter-level typos, ~1.5% of words)
  pad_filler          content UNCHANGED         +25..35% (generic academic filler)
  drop_appendix       core content UNCHANGED    <= -10% (appendix cut; else skipped)

The variants are later scored with the SAME rubric as the main study
(src/review/prompts/review_v1.md via run_llm_review.py's machinery). Paper selection
REUSES run_llm_review.select_papers, so the --papers 25 subset is the deterministic
stratified round-robin prefix of the main study's selection.

Hard guarantees:
  * byte-deterministic given --seed: every random draw uses numpy PCG64 seeded with
    seed + sha256(stable per-paper / per-section label); no global random state;
  * length tolerances enforced at generation time (violations raise):
    shuffle/corrupt/typos within +-1% of base words, pad_filler +25..35%,
    drop_appendix <= -10% (else the variant is skipped with reason 'appendix too small');
  * append-only JSONL log keyed on "{arxiv_id}|{variant}" (resumable via done_keys);
  * no network access.

Run from the repo root:
    python3 src/process/make_probes.py                      # 25 papers -> data/processed/probes/
    python3 src/process/make_probes.py --papers 3 --out-dir <dir> --log <dir>/probes_log.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from numpy.random import Generator, PCG64

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / 'src' / 'collect'))
sys.path.insert(0, str(REPO_ROOT / 'src' / 'review'))
from util import read_jsonl, append_jsonl, done_keys  # noqa: E402
from run_llm_review import select_papers  # noqa: E402  — same subset rule as the main study

VARIANTS = ('base', 'shuffle_paragraphs', 'corrupt_numbers', 'typos', 'pad_filler',
            'drop_appendix')
ANCHOR_STAGES = ('pre_submission', 'during_review')

# ---------------------------------------------------------------- text structure

HEADING_RE = re.compile(r'(?m)^## .*$')
PAR_SPLIT = re.compile(r'\n\s*\n')
PAR_SPLIT_KEEP = re.compile(r'(\n\s*\n)')
APPENDIX_RE = re.compile(r'(?m)^## Appendix\b.*$')
REFS_RE = re.compile(r'(?m)^## References\s*$')

# corrupt_numbers: candidate number tokens and the spans that must never be touched
NUM_RE = re.compile(r'(?<![\w.])\d+(?:\.\d+)*')
YEAR_RE = re.compile(r'(?<!\d)(?:19|20)\d\d(?!\d)')                      # 4-digit years
VERSION_RE = re.compile(r'(?<![A-Za-z])[vV]\d+\b')                       # v1, V2, ...
CITE_RE = re.compile(r'\[\d+(?:\s*,\s*\d+)*\]')                          # [3], [1, 2]
CAPFIG_RE = re.compile(r'(?m)^Caption:\s*\(?\d+\)?')                     # Caption: 3 ...
FIGREF_RE = re.compile(
    r'(?:Fig(?:ure)?s?|Tab(?:le)?s?|Sec(?:tion)?s?|Eq(?:uation)?s?|'
    r'Alg(?:orithm)?s?|App(?:endix)?|Chapter)\.?\s*~?\s*\(?\d+(?:\.\d+)*\)?', re.I)
NUM_SKIP_RES = (YEAR_RE, VERSION_RE, CITE_RE, CAPFIG_RE, FIGREF_RE)

# typos: candidate words and protected spans
WORD_RE = re.compile(r'(?<![A-Za-z])[A-Za-z]{4,}(?![A-Za-z])')
TYPO_SKIP_RES = (re.compile(r'(?m)^#.*$'),        # markdown headers (incl. title block)
                 re.compile(r'`[^`\n]*`'),        # backticked spans
                 re.compile(r'\[(?:URL|EMAIL)\]'))

TYPO_RATE = 0.015
NUMBER_RATE = 0.30
PAD_TARGET = 0.30          # aim +30% words
PAD_HARD_CAP = 0.345       # never exceed this (band is +25..35%)
PAD_MIN_SECTION_WORDS = 60  # sections whose share would be tiny are dropped, renormalized

# ~40 hand-written generic, content-free ML-paper filler sentences: no numbers, no
# method names, no placeholders, nothing identifying.
FILLER_BANK = [
    "The broader implications of this line of work extend to a variety of application domains, which we leave for future exploration.",
    "We believe that the perspective adopted here is complementary to, rather than in competition with, existing approaches in the literature.",
    "A comprehensive treatment of all related considerations is beyond the scope of the present paper and constitutes an interesting avenue for future work.",
    "It is worth emphasizing that the general framing introduced above is intended to be broadly applicable across a range of settings.",
    "For completeness, we note that similar observations have been made informally in adjacent areas of the field.",
    "The discussion in this section is intended to provide additional context rather than to introduce new technical material.",
    "We expect that practitioners will find these general considerations useful when adapting the ideas presented here to their own settings.",
    "As is standard in the literature, we defer a number of routine details to the interested reader.",
    "The overall picture that emerges is consistent with the intuition outlined in the introductory discussion.",
    "We remark in passing that the assumptions adopted throughout are standard and are made primarily for clarity of exposition.",
    "From a practical standpoint, the considerations discussed above are largely orthogonal to the specific implementation choices involved.",
    "It should be noted that the qualitative conclusions drawn here are expected to carry over to related settings with minor modifications.",
    "In the interest of readability, we keep the discussion at a relatively high level and avoid unnecessary formalism.",
    "These observations collectively suggest that further investigation along these lines may prove fruitful.",
    "We stress that the goal of this discussion is to situate the present work within the broader research landscape.",
    "The interplay between the various design considerations discussed above remains an interesting subject for further study.",
    "While a complete characterization is outside the scope of this work, the general trends are suggestive.",
    "It is natural to ask whether the ideas developed here admit extensions to more general settings, a question we leave open.",
    "The reader familiar with the surrounding literature will recognize that such trade-offs are common in practice.",
    "We anticipate that future work will shed additional light on the questions raised in this section.",
    "At a high level, the approach taken here reflects a balance between generality and practical applicability.",
    "This perspective aligns with a broader trend in the community toward methods that are simple, robust, and widely applicable.",
    "The considerations outlined above are intended as general guidance rather than prescriptive recommendations.",
    "Taken together, these points underscore the importance of careful experimental design in this area.",
    "A detailed exploration of these connections would require machinery beyond what is developed in this paper.",
    "We note that the presentation here favors clarity over maximal generality, in keeping with common practice.",
    "Such questions, while interesting in their own right, are tangential to the main thrust of the present work.",
    "The framework described above is deliberately modular, allowing individual components to be studied or replaced independently.",
    "It is our hope that the discussion in this section will serve as a useful reference point for subsequent studies.",
    "As with most empirical investigations, the conclusions drawn here should be interpreted in the context of the chosen experimental protocol.",
    "The general methodology adopted in this work follows established conventions in the field.",
    "We leave a systematic study of these additional considerations to future work.",
    "In summary, the observations above paint a coherent picture that is consistent with prior expectations.",
    "There are several natural directions in which the present analysis could be extended.",
    "The qualitative behavior described here appears to be robust across a variety of reasonable design choices.",
    "We emphasize that these remarks are intended to complement, not replace, the detailed treatment given elsewhere in the paper.",
    "More broadly, this line of inquiry touches on foundational questions that have long been of interest to the community.",
    "An exhaustive comparison across all conceivable configurations is impractical, and we therefore focus on representative cases.",
    "The discussion above abstracts away from implementation details in order to highlight the underlying principles.",
    "We conclude this section by noting that many of the themes discussed here recur throughout the remainder of the paper.",
]


# ---------------------------------------------------------------- small helpers

def stable_hash(label: str) -> int:
    """64-bit stable hash of a label (never Python's randomized hash())."""
    return int(hashlib.sha256(label.encode('utf-8')).hexdigest()[:16], 16)


def rng_for(seed: int, *parts) -> Generator:
    """Deterministic per-(paper[,section,variant]) RNG, as required: PCG64(seed + hash)."""
    return Generator(PCG64(seed + stable_hash('|'.join(str(p) for p in parts))))


def n_words(text: str) -> int:
    return len(text.split())


def anchor_version(paper: dict) -> int:
    """Latest manifest version with stage pre_submission/during_review, else v1
    (identical to analyze_scores.py's anchor rule)."""
    cands = [vv['v'] for vv in paper.get('versions', []) if vv.get('stage') in ANCHOR_STAGES]
    return max(cands) if cands else 1


def split_blocks(text: str) -> list[tuple[str | None, str]]:
    """[(heading, block_text)] where heading is the '## ...' line (None for the title
    block before the first '## '). Concatenating the block_texts reproduces text."""
    starts = [m.start() for m in HEADING_RE.finditer(text)]
    if not starts:
        return [(None, text)]
    blocks: list[tuple[str | None, str]] = []
    if starts[0] > 0:
        blocks.append((None, text[:starts[0]]))
    for i, s in enumerate(starts):
        e = starts[i + 1] if i + 1 < len(starts) else len(text)
        block = text[s:e]
        blocks.append((block.split('\n', 1)[0].strip(), block))
    return blocks


def protected_spans(text: str, regexes) -> list[tuple[int, int]]:
    spans = [m.span() for rx in regexes for m in rx.finditer(text)]
    return sorted(spans)


def overlaps(span: tuple[int, int], spans: list[tuple[int, int]]) -> bool:
    s, e = span
    return any(s < pe and ps < e for ps, pe in spans)


def shuffle_excluded(heading: str | None) -> bool:
    """Blocks exempt from paragraph shuffling / filler insertion: the title block,
    References, and any section whose heading contains 'Abstract'."""
    return (heading is None or heading.startswith('## References')
            or 'abstract' in heading.lower())


# ---------------------------------------------------------------- variant builders
# Each returns (new_text | None, edits_made: int, reason: str | None).

def make_shuffle_paragraphs(text: str, aid: str, seed: int) -> tuple[str, int, None]:
    out: list[str] = []
    moved = 0
    for idx, (heading, block) in enumerate(split_blocks(text)):
        if shuffle_excluded(heading):
            out.append(block)
            continue
        head_line, _, rest = block.partition('\n')
        pars = [p.strip() for p in PAR_SPLIT.split(rest) if p.strip()]
        if len(pars) < 2:
            out.append(block)
            continue
        rng = rng_for(seed, aid, 'shuffle_paragraphs', idx, heading)
        perm = rng.permutation(len(pars))
        moved += int(np.sum(perm != np.arange(len(pars))))
        out.append(head_line + '\n\n' + '\n\n'.join(pars[i] for i in perm) + '\n\n')
    return ''.join(out), moved, None


def corrupt_one_token(token: str, rng: Generator) -> str:
    """Change one random digit to a different digit; length/sign/decimal layout kept."""
    digit_pos = [i for i, c in enumerate(token) if c.isdigit()]
    p = digit_pos[int(rng.integers(len(digit_pos)))]
    alts = [d for d in '0123456789' if d != token[p]]
    return token[:p] + alts[int(rng.integers(len(alts)))] + token[p + 1:]


def make_corrupt_numbers(text: str, aid: str, seed: int) -> tuple[str, int, None]:
    parts = PAR_SPLIT_KEEP.split(text)  # even idx: paragraph text, odd idx: separators
    par_part_idx = [i for i in range(0, len(parts), 2) if parts[i].strip()]
    second_half = par_part_idx[len(par_part_idx) // 2:]
    cands: list[tuple[int, int, int]] = []  # (part index, start, end)
    for i in second_half:
        seg = parts[i]
        skip = protected_spans(seg, NUM_SKIP_RES)
        for m in NUM_RE.finditer(seg):
            if not overlaps(m.span(), skip):
                cands.append((i, m.start(), m.end()))
    rng = rng_for(seed, aid, 'corrupt_numbers')
    k = int(round(NUMBER_RATE * len(cands)))
    if k:
        chosen = sorted(int(c) for c in rng.choice(len(cands), size=k, replace=False))
        for ci in chosen:  # digit-for-digit replacement: offsets stay valid
            i, s, e = cands[ci]
            parts[i] = parts[i][:s] + corrupt_one_token(parts[i][s:e], rng) + parts[i][e:]
    return ''.join(parts), k, None


def typo_word(word: str, rng: Generator) -> str:
    """Seeded choice of: swap two adjacent letters / drop a letter / double a letter."""
    op = int(rng.integers(3))
    if op == 0:
        swappable = [i for i in range(len(word) - 1) if word[i] != word[i + 1]]
        if swappable:
            i = swappable[int(rng.integers(len(swappable)))]
            return word[:i] + word[i + 1] + word[i] + word[i + 2:]
        op = 2  # all-same-letter word: fall through to doubling
    if op == 1:
        i = int(rng.integers(len(word)))
        return word[:i] + word[i + 1:]
    i = int(rng.integers(len(word)))
    return word[:i + 1] + word[i] + word[i + 1:]


def make_typos(text: str, aid: str, seed: int) -> tuple[str, int, None]:
    skip = protected_spans(text, TYPO_SKIP_RES)
    cands = [m for m in WORD_RE.finditer(text) if not overlaps(m.span(), skip)]
    rng = rng_for(seed, aid, 'typos')
    k = int(round(TYPO_RATE * len(cands)))
    edits: list[tuple[int, int, str]] = []
    if k:
        chosen = sorted(int(c) for c in rng.choice(len(cands), size=k, replace=False))
        for ci in chosen:
            m = cands[ci]
            edits.append((m.start(), m.end(), typo_word(m.group(), rng)))
    for s, e, w in sorted(edits, reverse=True):
        text = text[:s] + w + text[e:]
    return text, len(edits), None


def group_sentences(sents: list[str], rng: Generator) -> list[str]:
    """Join sentences into paragraphs of 4-6 (a final paragraph may hold up to 7 so no
    fragment of fewer than 4 is left over; short tails only for tiny allocations)."""
    paras: list[str] = []
    i = 0
    while i < len(sents):
        rem = len(sents) - i
        if rem <= 7:
            size = rem
        else:
            size = int(rng.integers(4, 7))
            if rem - size in (1, 2, 3):   # rem 8 or 9: leave at least 4 for the tail
                size = rem - 4
        paras.append(' '.join(sents[i:i + size]))
        i += size
    return paras


def make_pad_filler(text: str, aid: str, seed: int) -> tuple[str, int, None]:
    blocks = split_blocks(text)
    base = n_words(text)
    target = PAD_TARGET * base
    hard_cap = PAD_HARD_CAP * base
    eligible: list[tuple[int, int]] = []  # (block index, content words)
    for bi, (heading, block) in enumerate(blocks):
        if shuffle_excluded(heading):
            continue
        w = n_words(block.partition('\n')[2])
        if w > 0:
            eligible.append((bi, w))
    rng = rng_for(seed, aid, 'pad_filler')
    if not eligible:  # degenerate doc: append everything to the last block
        eligible = [(len(blocks) - 1, 1)]
    wsum = sum(w for _, w in eligible)
    big = [(bi, w) for bi, w in eligible if target * w / wsum >= PAD_MIN_SECTION_WORDS]
    if big:
        eligible, wsum = big, sum(w for _, w in big)

    deck: list[int] = []
    last_drawn = -1
    added_total = 0
    n_paras = 0
    filler_by_block: dict[int, list[str]] = {}
    for bi, w in eligible:  # document order; proportional-to-section-length targets
        t_i = target * w / wsum
        sents: list[str] = []
        got = 0
        while got < t_i:
            if not deck:
                deck = [int(x) for x in rng.permutation(len(FILLER_BANK))]
                if deck[-1] == last_drawn:  # no immediate repeats across reshuffles
                    deck[0], deck[-1] = deck[-1], deck[0]
            last_drawn = deck.pop()
            s = FILLER_BANK[last_drawn]
            sw = len(s.split())
            if added_total + got + sw > hard_cap:
                deck.append(last_drawn)
                break
            sents.append(s)
            got += sw
        added_total += got
        if sents:
            paras = group_sentences(sents, rng)
            n_paras += len(paras)
            filler_by_block[bi] = paras
        if added_total >= hard_cap:
            break

    out: list[str] = []
    for bi, (heading, block) in enumerate(blocks):
        if bi in filler_by_block:  # insert at the section's closing boundary
            out.append(block.rstrip('\n') + '\n\n'
                       + '\n\n'.join(filler_by_block[bi]) + '\n\n')
        else:
            out.append(block)
    return ''.join(out), n_paras, None


def make_drop_appendix(text: str) -> tuple[str | None, int, str | None]:
    """Cut from the first '## Appendix' heading to just before the next '## References'
    after it (keeping References), or to end-of-document when References precedes the
    appendix — the common layout in this corpus ('## References\\n(omitted)\\n## Appendix')."""
    m = APPENDIX_RE.search(text)
    if not m:
        return None, 0, 'no appendix heading'
    m2 = REFS_RE.search(text, m.end())
    end = m2.start() if m2 else len(text)
    removed = n_words(text[m.start():end])
    return text[:m.start()] + text[end:], removed, None


# ---------------------------------------------------------------- driving loop

def build_variant(variant: str, base_text: str, aid: str,
                  seed: int) -> tuple[str | None, int, str | None]:
    if variant == 'base':
        return base_text, 0, None
    if variant == 'shuffle_paragraphs':
        return make_shuffle_paragraphs(base_text, aid, seed)
    if variant == 'corrupt_numbers':
        return make_corrupt_numbers(base_text, aid, seed)
    if variant == 'typos':
        return make_typos(base_text, aid, seed)
    if variant == 'pad_filler':
        return make_pad_filler(base_text, aid, seed)
    if variant == 'drop_appendix':
        return make_drop_appendix(base_text)
    raise ValueError(variant)


def check_tolerance(variant: str, pct: float) -> str | None:
    """None when within tolerance; a skip reason for drop_appendix; raises otherwise
    (an out-of-band length on variants 1-5 means a generator bug, not a data issue)."""
    if variant in ('base', 'shuffle_paragraphs', 'corrupt_numbers', 'typos'):
        if abs(pct) > 1.0:
            raise RuntimeError(f"{variant}: length changed {pct:+.2f}% (must be within +-1%)")
    elif variant == 'pad_filler':
        if not (25.0 <= pct <= 35.0):
            raise RuntimeError(f"pad_filler: length changed {pct:+.2f}% (must be +25..35%)")
    elif variant == 'drop_appendix':
        if pct > -10.0:
            return 'appendix too small'
    return None


def main() -> None:
    ap = argparse.ArgumentParser(
        description='E4: deterministic known-direction, length-controlled probe variants.')
    ap.add_argument('--manifest', default='data/processed/pilot_manifest.jsonl')
    ap.add_argument('--texts', default='data/processed/texts/')
    ap.add_argument('--papers', type=int, default=25,
                    help='stratified round-robin subset via run_llm_review.select_papers')
    ap.add_argument('--out-dir', default='data/processed/probes/')
    ap.add_argument('--log', default='data/processed/probes_log.jsonl',
                    help='append-only log/checkpoint, keyed on "{arxiv_id}|{variant}"')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--limit', type=int, default=0,
                    help='max variant tasks processed this run (0 = no cap)')
    cfg = ap.parse_args()

    rows = read_jsonl(cfg.manifest)
    papers = select_papers(rows, cfg.papers)
    done = done_keys(cfg.log, 'key')
    texts_dir = Path(cfg.texts)
    out_root = Path(cfg.out_dir)

    n_ckpt = n_missing = n_done = 0
    emitted: Counter = Counter()
    skipped: Counter = Counter()
    reasons: Counter = Counter()
    pcts: dict[str, list[float]] = defaultdict(list)
    capped = False

    for paper in papers:
        if capped:
            break
        aid = paper['arxiv_id']
        safe_id = aid.replace('/', '_')
        av = anchor_version(paper)
        text_path = texts_dir / safe_id / f'v{av}.md'
        if not text_path.exists():
            print(f"WARNING: missing anchor text {text_path} — paper skipped (not logged)")
            n_missing += 1
            continue
        base_text = text_path.read_text(encoding='utf-8')
        base_words = n_words(base_text)

        for variant in VARIANTS:
            key = f'{aid}|{variant}'
            if key in done:
                n_ckpt += 1
                continue
            if cfg.limit > 0 and n_done >= cfg.limit:
                capped = True
                break

            new_text, edits, reason = build_variant(variant, base_text, aid, cfg.seed)
            words = pct = None
            if new_text is not None:
                words = n_words(new_text)
                pct = (words - base_words) / base_words * 100.0
                reason = check_tolerance(variant, pct)  # raises on generator bugs

            path = None
            if new_text is not None and reason is None:
                vpath = out_root / safe_id / f'{variant}.md'
                vpath.parent.mkdir(parents=True, exist_ok=True)
                vpath.write_text(new_text, encoding='utf-8')
                path = str(vpath)
                emitted[variant] += 1
                pcts[variant].append(pct)
            else:
                skipped[variant] += 1
                reasons[reason] += 1

            append_jsonl(cfg.log, {
                'key': key, 'arxiv_id': aid, 'variant': variant, 'path': path,
                'anchor_v': int(av), 'n_words': None if words is None else int(words),
                'n_words_base': int(base_words),
                'pct_length_change': None if pct is None else round(float(pct), 3),
                'edits_made': int(edits), 'skipped': reason is not None,
                'reason': reason,
            })
            n_done += 1

    print(f"\n=== make_probes summary (seed={cfg.seed}) ===")
    print(f"papers selected: {len(papers)}/{len(rows)} | variant tasks this run: {n_done} "
          f"| checkpointed: {n_ckpt} | papers missing anchor text: {n_missing}"
          + (" | stopped at --limit" if capped else ""))
    for variant in VARIANTS:
        vals = pcts[variant]
        mean_pct = f"{sum(vals) / len(vals):+.2f}%" if vals else "n/a"
        print(f"  {variant:<20} emitted: {emitted[variant]:>3} | skipped: "
              f"{skipped[variant]:>3} | mean pct_length_change: {mean_pct}")
    if reasons:
        print("skip reasons: " + ", ".join(f"{r}: {c}" for r, c in sorted(reasons.items())))
    print(f"out dir: {out_root} | log: {cfg.log}")


if __name__ == '__main__':
    main()
