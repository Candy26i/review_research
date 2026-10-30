"""Build the frozen-splice pilot dataset (Q2 feasibility probe).

Question: given a single document, can a classifier find real author revisions
spliced into an otherwise-unchanged earlier version? For each paper with a
substantive v_from->v_to revision we align paragraphs (global descending-ratio
assignment inside block-level SequenceMatcher opcodes, with a moved-paragraph
guard and a content-overlap gate for low-ratio pairs) and build per-condition
documents:

  true    v_from doc with k<=3 paragraphs replaced by their substantially
          rewritten v_to counterparts (word ratio 0.15..0.60)  <- the task
  micro   v_from doc with k<=3 near-identical v_to paragraphs (ratio >=0.92,
          <1.0) spliced in          <- regime-E rung doubling as artifact probe
  ident   v_from doc with k<=3 v_to paragraphs that are byte-equal after
          whitespace collapse       <- code-path sanity check (vacuous by
                                       construction for text channels)
  v1self  untouched v_from doc, 'positives' = the v_from counterparts of the
          true splice positions     <- edit-proneness + position confound
  xdoc    v_from doc with the SAME slots filled by kind/length-matched
          rewritten-v_to paragraphs from OTHER test papers
                                    <- regime-B anchor: foreign-text ceiling,
                                       separates 'editedness' from topical novelty

Train split holds only 'true' docs; the other conditions exist only for test
papers (paper-level 80/20 split, one version pair per paper, arxiv_id-deduped
across corpora). All emitted text (text/prev/next) is whitespace-collapsed so
line-wrapping extraction artifacts cannot differ between spliced and native
blocks. Context snippets come from the SPLICED document so boundary seams stay
visible to models trained with context; adjacent splice slots are dropped so a
negative's window never contains spliced text.

Run from repo root:
  python3 src/process/build_splice_pilot.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np

sys.path.insert(0, 'src/collect')
from util import read_jsonl  # noqa: E402

CORPORA = [
    ('acl', 'data/processed/acl_version_diffs.jsonl', 'data/processed/acl_texts'),
    ('iclr', 'data/processed/version_diffs.jsonl', 'data/processed/texts'),
    ('fresh', 'data/processed/fresh_version_diffs.jsonl', 'data/processed/fresh_texts'),
]

SKIP_SECTION_RE = re.compile(r'references|bibliography|acknowledg', re.I)
SECTION_BUCKETS = [
    ('abstract', r'abstract'), ('intro', r'introduction'),
    ('related', r'related|background'), ('method', r'method|approach|model'),
    ('experiment', r'experiment|setup|evaluation|result'),
    ('discussion', r'discussion|analysis|ablation'),
    ('conclusion', r'conclusion|limitation|future'), ('appendix', r'appendix'),
]
MIN_WORDS, MAX_WORDS = 25, 700
MIN_CLS_BLOCKS = 10          # a doc needs this many classifiable paragraphs
MIN_PAIR_WORDS = 1500        # skip degenerate extractions
REWRITE_LO, REWRITE_HI = 0.15, 0.60
CONTENT_GATE_BELOW = 0.25    # low-ratio pairs must also share content words
CONTENT_JACCARD_MIN = 0.10
MICRO_LO = 0.92
MAX_SPLICES = 3
TEST_FRAC_MOD = 5            # paper_hash % 5 == 0 -> test (20%)
CTX_WORDS = 50
TEXT_CAP_WORDS = 450
XDOC_WORD_BAND = (0.6, 1.5)  # donor length relative to the true v2 paragraph


def seed64(key: str) -> int:
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], 'big')


def norm(s: str) -> str:
    return ' '.join(s.lower().split())


def collapse(s: str) -> str:
    return ' '.join(s.split())


def content_words(s: str) -> set[str]:
    return {w for w in s.lower().split() if len(w) >= 5}


def section_bucket(name: str) -> str:
    low = name.lower()
    for bucket, pat in SECTION_BUCKETS:
        if re.search(pat, low):
            return bucket
    return 'other'


def word_ratio(a: str, b: str) -> float:
    sm = SequenceMatcher(None, a.split(), b.split(), autojunk=False)
    if sm.real_quick_ratio() < REWRITE_LO:
        return 0.0
    return sm.ratio()


def cap_words(s: str, n: int) -> str:
    return ' '.join(s.split()[:n])


def tail_words(s: str, n: int) -> str:
    return ' '.join(s.split()[-n:])


def parse_blocks(text: str) -> list[dict]:
    section = 'Preamble'
    out = []
    for raw in re.split(r'\n\s*\n', text):
        s = raw.strip()
        if not s:
            continue
        if s.startswith('## '):
            section = s[3:].strip()
            out.append({'text': s, 'section': section, 'kind': 'header', 'cls': False})
            continue
        if s.startswith('# '):
            out.append({'text': s, 'section': section, 'kind': 'title', 'cls': False})
            continue
        kind = ('caption' if s.startswith('Caption:')
                else 'bullet' if s.lstrip().startswith('* ') else 'para')
        n_words = len(s.split())
        cls = (not SKIP_SECTION_RE.search(section)
               and MIN_WORDS <= n_words <= MAX_WORDS)
        out.append({'text': s, 'section': section, 'kind': kind, 'cls': cls})
    return out


def align(b1: list[dict], b2: list[dict], full_range: bool = False):
    """Returns (unchanged, rewrites, micro): lists of (i, j, ratio) with both
    sides classifiable.

    full_range=True also labels the ambiguous middle (REWRITE_HI..MICRO_LO) as
    rewrites — required for attribution, where excluding that band leaves a
    magnitude hole the AI class does not share (QC: band rule alone = 90% acc).

    unchanged = normalized-equal 'equal' opcode pairs. Inside each 'replace'
    opcode, candidate (i, j) pairs are assigned globally by DESCENDING ratio so
    an early v1 block cannot steal a later block's true partner. Guards:
    - moved-paragraph: a v2 text whose normalization equals ANY classifiable
      v1 block is never a rewrite positive (it is unrevised text that moved);
    - content gate: pairs below CONTENT_GATE_BELOW must share content words
      (Jaccard >= CONTENT_JACCARD_MIN over words of length >= 5), so function-
      word overlap of unrelated paragraphs cannot reach REWRITE_LO;
    - the ambiguous middle (REWRITE_HI..MICRO_LO) is consumed by assignment
      but labeled neither rewrite nor micro."""
    n1 = [norm(b['text']) for b in b1]
    n2 = [norm(b['text']) for b in b2]
    v1_norm_set = {n1[i] for i, b in enumerate(b1) if b['cls']}
    sm = SequenceMatcher(None, n1, n2, autojunk=False)
    unchanged, rewrites, micro = [], [], []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == 'equal':
            for i, j in zip(range(i1, i2), range(j1, j2)):
                if b1[i]['cls'] and b2[j]['cls']:
                    unchanged.append((i, j, 1.0))
        elif tag == 'replace':
            cands = []
            for i in range(i1, i2):
                if not b1[i]['cls']:
                    continue
                for j in range(j1, j2):
                    if not b2[j]['cls']:
                        continue
                    r = word_ratio(b1[i]['text'], b2[j]['text'])
                    if r >= REWRITE_LO:
                        cands.append((r, i, j))
            cands.sort(key=lambda t: (-t[0], t[1], t[2]))
            used_i, used_j = set(), set()
            for r, i, j in cands:
                if i in used_i or j in used_j:
                    continue
                used_i.add(i)
                used_j.add(j)
                if r >= MICRO_LO:
                    if n1[i] != n2[j]:
                        micro.append((i, j, r))
                    continue
                if r > REWRITE_HI and not full_range:
                    continue
                if n2[j] in v1_norm_set:
                    continue
                if r < CONTENT_GATE_BELOW:
                    c1, c2 = content_words(b1[i]['text']), content_words(b2[j]['text'])
                    union = c1 | c2
                    if not union or len(c1 & c2) / len(union) < CONTENT_JACCARD_MIN:
                        continue
                rewrites.append((i, j, r))
    return unchanged, rewrites, micro


def emit_doc(blocks: list[dict], positives: dict[int, float | None], meta: dict,
             cond: str, rows: list) -> None:
    cls_idx = [k for k, b in enumerate(blocks) if b['cls']]
    for rank, k in enumerate(cls_idx):
        b = blocks[k]
        ratio = positives.get(k) if k in positives else None
        rows.append({
            **meta, 'cond': cond, 'para_idx': rank, 'n_paras': len(cls_idx),
            'section': b['section'], 'kind': b['kind'],
            'words': len(b['text'].split()),
            'label': int(k in positives),
            'change_ratio': round(ratio, 4) if ratio is not None else None,
            'prev': tail_words(blocks[k - 1]['text'], CTX_WORDS) if k > 0 else '',
            'next': cap_words(blocks[k + 1]['text'], CTX_WORDS) if k + 1 < len(blocks) else '',
            'text': cap_words(b['text'], TEXT_CAP_WORDS),
        })


def splice(b1: list[dict], repl: list[tuple[int, dict, float | None]]):
    """Replace v1 block i with the given block's TEXT, keeping the v1 slot's
    section/kind metadata so analysis slices bin positives and negatives alike."""
    doc = [dict(b) for b in b1]
    positives: dict[int, float | None] = {}
    for i, block, r in repl:
        doc[i] = {'text': block['text'], 'section': b1[i]['section'],
                  'kind': b1[i]['kind'], 'cls': True}
        positives[i] = r
    return doc, positives


def drop_adjacent(pairs: list) -> list:
    """Keep a subset of (i, j, r) with no two v1 indices within 1 of each other,
    so no negative's context window contains spliced text."""
    kept = []
    for p in sorted(pairs, key=lambda t: t[0]):
        if kept and p[0] - kept[-1][0] <= 1:
            continue
        kept.append(p)
    return kept


def pick(rng: np.random.Generator, pairs: list, k: int) -> list:
    pairs = drop_adjacent(pairs)
    if len(pairs) <= k:
        return pairs
    idx = rng.choice(len(pairs), size=k, replace=False)
    return [pairs[i] for i in sorted(idx)]


def main() -> None:
    global CTX_WORDS
    ap = argparse.ArgumentParser(description='Frozen-splice pilot dataset builder.')
    ap.add_argument('--out-dir', default='data/processed/splice_pilot')
    ap.add_argument('--min-churn', type=float, default=0.02,
                    help='skip pairs below this churn (cosmetic)')
    ap.add_argument('--max-churn', type=float, default=0.70,
                    help='skip pairs above (extraction inconsistency)')
    ap.add_argument('--limit', type=int, default=0, help='papers cap, 0 = all')
    ap.add_argument('--seed', type=int, default=20261029)
    ap.add_argument('--ctx-words', type=int, default=CTX_WORDS,
                    help='context window words per side (escalation: 100)')
    args = ap.parse_args()
    CTX_WORDS = args.ctx_words

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # one pair per paper: the highest-churn eligible adjacent pair
    chosen: dict[str, dict] = {}
    for corpus, diffs_path, texts_dir in CORPORA:
        if not Path(diffs_path).exists():
            print(f'!! missing {diffs_path}, skipping {corpus}')
            continue
        for row in read_jsonl(diffs_path):
            aid = row['arxiv_id']
            if aid in chosen and chosen[aid]['corpus'] != corpus:
                continue  # dedupe across corpora (first corpus wins)
            if not (args.min_churn <= row['churn'] <= args.max_churn):
                continue
            if min(row['words_from'], row['words_to']) < MIN_PAIR_WORDS:
                continue
            cur = chosen.get(aid)
            if cur is None or row['churn'] > cur['churn']:
                chosen[aid] = {**row, 'corpus': corpus, 'texts_dir': texts_dir}

    papers = sorted(chosen.values(), key=lambda r: r['arxiv_id'])
    if args.limit:
        papers = papers[:args.limit]
    print(f'eligible papers (one pair each): {len(papers)}')

    train_rows, test_rows = [], []
    stats = {'papers_used': 0, 'papers_skipped': 0, 'by_corpus': {}, 'by_cond': {},
             'ratio_hist': {}, 'skip_reasons': {}}
    donors = []     # rewritten v_to paragraphs from TEST papers, for xdoc
    xdoc_ctx = []   # per test paper: what xdoc needs after the donor pool exists

    def skip(reason: str):
        stats['papers_skipped'] += 1
        stats['skip_reasons'][reason] = stats['skip_reasons'].get(reason, 0) + 1

    for p in papers:
        base = Path(p['texts_dir']) / p['arxiv_id']
        f1, f2 = base / f"v{p['v_from']}.md", base / f"v{p['v_to']}.md"
        if not f1.exists() or not f2.exists():
            skip('missing_text')
            continue
        b1 = parse_blocks(f1.read_text())
        b2 = parse_blocks(f2.read_text())
        if sum(b['cls'] for b in b1) < MIN_CLS_BLOCKS:
            skip('too_few_blocks')
            continue
        unchanged, rewrites, micro = align(b1, b2)
        if not rewrites:
            skip('no_rewrite_pairs')
            continue

        rng = np.random.default_rng(seed64(f"{args.seed}|{p['arxiv_id']}"))
        is_test = seed64(f"split|{p['arxiv_id']}") % TEST_FRAC_MOD == 0
        meta = {'corpus': p['corpus'], 'paper': p['arxiv_id'],
                'v_from': p['v_from'], 'v_to': p['v_to'],
                'churn': p['churn'], 'split': 'test' if is_test else 'train'}
        rows = test_rows if is_test else train_rows

        true_pairs = pick(rng, rewrites, MAX_SPLICES)
        doc, pos = splice(b1, [(i, b2[j], r) for i, j, r in true_pairs])
        emit_doc(doc, pos, meta, 'true', rows)
        stats['by_cond']['true'] = stats['by_cond'].get('true', 0) + len(pos)
        for _, _, r in true_pairs:
            bucket = f'{int(r * 10) / 10:.1f}'
            stats['ratio_hist'][bucket] = stats['ratio_hist'].get(bucket, 0) + 1

        if is_test:
            # v1self: untouched doc, positives at the true-splice v_from slots
            emit_doc(b1, {i: r for i, _, r in true_pairs}, meta, 'v1self', rows)
            stats['by_cond']['v1self'] = stats['by_cond'].get('v1self', 0) + len(true_pairs)
            # ident requires byte equality after whitespace collapse (not just
            # normalized equality), so the control is vacuous by construction
            ident_pairs = [(i, j, r) for i, j, r in unchanged
                           if collapse(b1[i]['text']) == collapse(b2[j]['text'])]
            for cond, pairs in (('micro', micro), ('ident', ident_pairs)):
                sel = pick(rng, pairs, MAX_SPLICES)
                if not sel:
                    continue
                doc_c, pos_c = splice(b1, [(i, b2[j], r) for i, j, r in sel])
                emit_doc(doc_c, pos_c, meta, cond, rows)
                stats['by_cond'][cond] = stats['by_cond'].get(cond, 0) + len(pos_c)
            # xdoc donors and deferred construction (needs the full donor pool)
            for i, j, r in rewrites:
                donors.append({'paper': p['arxiv_id'], 'text': b2[j]['text'],
                               'kind': b1[i]['kind'],
                               'words': len(b2[j]['text'].split()),
                               'bucket': section_bucket(b1[i]['section'])})
            xdoc_ctx.append({'meta': meta, 'b1': b1,
                             'slots': [(i, len(b2[j]['text'].split()),
                                        b1[i]['kind'], section_bucket(b1[i]['section']))
                                       for i, j, r in true_pairs]})

        stats['papers_used'] += 1
        c = p['corpus']
        stats['by_corpus'][c] = stats['by_corpus'].get(c, 0) + 1

    # xdoc: same slots as true, donor paragraphs from OTHER test papers,
    # matched on kind + length band (same section bucket preferred)
    for ctx in xdoc_ctx:
        rng = np.random.default_rng(seed64(f"{args.seed}|xdoc|{ctx['meta']['paper']}"))
        repl = []
        for i, w, kind, bucket in ctx['slots']:
            cands = [d for d in donors
                     if d['paper'] != ctx['meta']['paper'] and d['kind'] == kind
                     and XDOC_WORD_BAND[0] * w <= d['words'] <= XDOC_WORD_BAND[1] * w]
            same_bucket = [d for d in cands if d['bucket'] == bucket]
            pool = same_bucket or cands
            if not pool:
                continue
            d = pool[int(rng.integers(len(pool)))]
            repl.append((i, {'text': d['text']}, None))
        if not repl:
            continue
        doc_x, pos_x = splice(ctx['b1'], repl)
        emit_doc(doc_x, pos_x, ctx['meta'], 'xdoc', test_rows)
        stats['by_cond']['xdoc'] = stats['by_cond'].get('xdoc', 0) + len(pos_x)

    for name, rows in (('train', train_rows), ('test', test_rows)):
        path = out_dir / f'{name}.jsonl'
        with open(path, 'w') as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')
        n_pos = sum(r['label'] for r in rows)
        print(f'{name}: {len(rows)} rows ({n_pos} positive) -> {path}')

    n_test_papers = len({r['paper'] for r in test_rows})
    n_train_papers = len({r['paper'] for r in train_rows})
    stats['train_papers'], stats['test_papers'] = n_train_papers, n_test_papers
    with open(out_dir / 'stats.json', 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"papers used {stats['papers_used']} (train {n_train_papers} / test "
          f"{n_test_papers}), skipped {stats['papers_skipped']} "
          f"{stats['skip_reasons']}")
    print(f"positives by condition: {stats['by_cond']}")
    print(f"rewrite-ratio histogram: {dict(sorted(stats['ratio_hist'].items()))}")


if __name__ == '__main__':
    main()
