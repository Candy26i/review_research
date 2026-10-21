"""Collect the figures each paper version ACTUALLY references, as web-ready PNGs.

For each manifest version with an ok LaTeX row in latex_log: re-read the main
.tex, flatten \\input/\\include with the shared Flattener, strip comments, and
scan \\includegraphics refs in document order. Resolve each ref against the
main-tex dir, the version root, and any \\graphicspath dirs (probing
.png/.jpg/.jpeg/.pdf/.eps, with a unique-basename fallback anywhere in the
tree), dedup by resolved path, cap at --max-figs, drop identity-leaking
filenames (logo/author/affiliation/orcid/signature/headshot/photo_), and
convert the rest to RGB PNGs whose longest side is at most --max-px (rasters
via PIL, never upscaled; PDF figures via pymupdf, first page; eps/svg skipped).

Outputs {out-dir}/{safe_id}/v{v}/fig_{i:02d}.png (i is 1-based attach order).
The log row's files[].src is relative to the version root and files[].png is
relative to --out-dir, so the multimodal runner can locate both. Versions
without a LaTeX tree (converter=pdf rows, missing extractions such as
2301.12935v2) get a clean n_attached=0 row with a "reason". Resumable via the
log JSONL, checkpointed on key "{arxiv_id}v{v}". One bad figure or version
never kills the run.

Run from repo root: python3 src/process/collect_figures.py
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

from PIL import Image
from tqdm import tqdm

sys.path.insert(0, 'src/collect')
from util import append_jsonl, done_keys, read_jsonl  # noqa: E402

sys.path.insert(0, 'src/process')
from latex_to_text import Flattener  # noqa: E402

try:
    import pymupdf  # noqa: E402
except ImportError:  # older installs expose only the fitz name
    import fitz as pymupdf  # noqa: E402

RASTER_EXTS = {'.png', '.jpg', '.jpeg'}
TRY_EXTS = ['.png', '.jpg', '.jpeg', '.pdf', '.eps']

# ANONYMITY: figures whose filename carries identity tells are never attached
# (e.g. assets/Google_DeepMind_Logo_rgb_3320x512px.png in this corpus).
FILTER_TOKENS = ('logo', 'affiliation', 'orcid', 'signature', 'headshot',
                 'photo_', 'author')

INCLUDEGRAPHICS_RE = re.compile(r'\\includegraphics(?:\[[^\]]*\])?\{([^{}]+)\}')
GRAPHICSPATH_RE = re.compile(r'\\graphicspath\s*\{((?:\s*\{[^{}]*\}\s*)+)\}')
COMMENT_ENV_RE = re.compile(
    r'\\begin\s*\{\s*comment\s*\}.*?\\end\s*\{\s*comment\s*\}', re.S)


def is_filtered(filename: str) -> bool:
    low = filename.lower()
    return any(tok in low for tok in FILTER_TOKENS)


def parse_graphicspath(text: str) -> list[str]:
    """\\graphicspath{{dir1}{dir2}} -> ['dir1', 'dir2'] (simple parse, in order)."""
    dirs: list[str] = []
    for m in GRAPHICSPATH_RE.finditer(text):
        for g in re.findall(r'\{([^{}]*)\}', m.group(1)):
            g = g.strip()
            if g and g not in dirs:
                dirs.append(g)
    return dirs


def build_name_index(root: Path) -> dict[str, list[Path]]:
    idx: dict[str, list[Path]] = {}
    for p in sorted(root.rglob('*')):
        if p.is_file():
            idx.setdefault(p.name.lower(), []).append(p)
    return idx


def resolve_ref(ref: str, bases: list[Path], root: Path,
                name_idx: dict[str, list[Path]]) -> Path | None:
    """Resolve an \\includegraphics argument to a real file inside root."""
    name = ref.strip().replace('\\', '/')
    if not name:
        return None
    cands = [name] + [name + e for e in TRY_EXTS]
    for base in bases:
        for nm in cands:
            try:
                p = (base / nm).resolve()
            except OSError:
                continue
            # containment: never follow paths escaping the version tree
            if p.is_file() and p.is_relative_to(root):
                return p
    # final fallback: unique basename match anywhere in the tree
    bn = Path(name).name.lower()
    hits = list(name_idx.get(bn, []))
    if not hits:
        hits = [p for e in TRY_EXTS for p in name_idx.get(bn + e, [])]
    if not hits and ' ' in bn:
        # arXiv renames uploaded files, replacing spaces with underscores;
        # refs typed before upload ({figures/flow chart.pdf}) keep the space
        bn2 = bn.replace(' ', '_')
        hits = list(name_idx.get(bn2, []))
        if not hits:
            hits = [p for e in TRY_EXTS for p in name_idx.get(bn2 + e, [])]
    # same containment rule as the direct probes: resolve symlinks and keep
    # only files that really live inside the version tree, so a symlink in
    # the tree can never smuggle in an outside file (and dedup compares
    # resolved paths on both resolution routes)
    safe: set[Path] = set()
    for p in hits:
        try:
            rp = p.resolve()
        except OSError:
            continue
        if rp.is_file() and rp.is_relative_to(root):
            safe.add(rp)
    return next(iter(safe)) if len(safe) == 1 else None


def convert_raster(src: Path, dst: Path, max_px: int) -> tuple[int, int]:
    """png/jpg/jpeg -> RGB PNG, longest side <= max_px (never upscaled)."""
    with Image.open(src) as im:
        im = im.convert('RGB')
        if max(im.size) > max_px:
            im.thumbnail((max_px, max_px), Image.LANCZOS)
        im.save(dst, format='PNG')
        return im.size


def convert_pdf(src: Path, dst: Path, max_px: int) -> tuple[int, int]:
    """First PDF page -> PNG, rendered so the longest side is ~max_px."""
    doc = pymupdf.open(str(src))
    try:
        if doc.page_count < 1:
            raise ValueError('pdf has no pages')
        page = doc[0]
        rect = page.rect
        longest = max(rect.width, rect.height)
        zoom = (max_px / longest) if longest > 0 else 1.0
        pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), alpha=False)
        dst.write_bytes(pix.tobytes('png'))
        return pix.width, pix.height
    finally:
        doc.close()


def base_log(key: str, aid: str, v: int) -> dict:
    return {'key': key, 'arxiv_id': aid, 'v': v, 'n_referenced': 0,
            'n_attached': 0, 'skipped_filtered': 0, 'skipped_format': 0,
            'skipped_unresolved': 0, 'skipped_error': 0, 'total_png_bytes': 0,
            'files': [], 'skipped_files': [], 'ok': False, 'error': None}


def collect_version(log: dict, root: Path, main_tex: Path, out_root: Path,
                    safe_id: str, v: int, max_figs: int, max_px: int) -> None:
    """Fill `log` in place: scan, resolve, filter, convert. Per-file errors
    are counted, never raised."""
    fl = Flattener(root)
    fl.visited.add(main_tex.resolve())
    text = fl.flatten(main_tex)          # strips % comments per file
    text = COMMENT_ENV_RE.sub('', text)  # commented-out figures don't count

    gdirs = parse_graphicspath(text)
    main_dir = main_tex.parent.resolve()
    bases = [main_dir, root]
    for g in gdirs:
        bases.extend([main_dir / g, root / g])

    name_idx = build_name_index(root)
    key = log['key']

    # 1. referenced figures, document order, dedup by resolved path, cap
    resolved_order: list[Path] = []
    seen: set[Path] = set()
    unresolved: list[str] = []
    for m in INCLUDEGRAPHICS_RE.finditer(text):
        log['n_referenced'] += 1
        ref = m.group(1).strip()
        p = resolve_ref(ref, bases, root, name_idx)
        if p is None:
            if ref not in unresolved:
                unresolved.append(ref)
            continue
        if p in seen:
            continue
        seen.add(p)
        resolved_order.append(p)
    log['skipped_unresolved'] = len(unresolved)
    for ref in unresolved:
        log['skipped_files'].append({'src': ref, 'reason': 'unresolved'})
    capped = resolved_order[:max_figs]

    # 2 + 3. anonymity filter, then convert
    vdir = out_root / safe_id / f'v{v}'
    for p in capped:
        rel_src = str(p.relative_to(root))
        if is_filtered(p.name):
            log['skipped_filtered'] += 1
            log['skipped_files'].append({'src': rel_src, 'reason': 'filtered'})
            tqdm.write(f'SKIP (anonymity filter) {key}: {rel_src}')
            continue
        ext = p.suffix.lower()
        if ext not in RASTER_EXTS and ext != '.pdf':
            log['skipped_format'] += 1
            log['skipped_files'].append(
                {'src': rel_src, 'reason': f'format:{ext or "none"}'})
            continue
        vdir.mkdir(parents=True, exist_ok=True)
        dst = vdir / f'fig_{log["n_attached"] + 1:02d}.png'
        try:
            if ext == '.pdf':
                w, h = convert_pdf(p, dst, max_px)
            else:
                w, h = convert_raster(p, dst, max_px)
        except Exception as exc:  # one broken figure never kills the version
            log['skipped_error'] += 1
            log['skipped_files'].append(
                {'src': rel_src,
                 'reason': f'error:{type(exc).__name__}: {exc}'[:300]})
            dst.unlink(missing_ok=True)
            continue
        log['total_png_bytes'] += dst.stat().st_size
        log['n_attached'] += 1
        log['files'].append({'src': rel_src,
                             'png': f'{safe_id}/v{v}/{dst.name}',
                             'w': w, 'h': h})


def main() -> None:
    ap = argparse.ArgumentParser(
        description='Collect referenced figures per paper version as PNGs '
                    'for the multimodal review runner.')
    ap.add_argument('--manifest', default='data/processed/pilot_manifest.jsonl')
    ap.add_argument('--extracted', default='data/raw/extracted/')
    ap.add_argument('--latex-log', default='data/processed/latex_log.jsonl')
    ap.add_argument('--out-dir', default='data/processed/figures/')
    ap.add_argument('--log', default='data/processed/figures_log.jsonl')
    ap.add_argument('--max-figs', type=int, default=20,
                    help='max figures attached per version')
    ap.add_argument('--max-px', type=int, default=1024,
                    help='longest PNG side in pixels')
    ap.add_argument('--limit', type=int, default=0,
                    help='max versions this run (0 = no cap)')
    ap.add_argument('--papers', default='',
                    help="comma-separated arxiv_ids; restrict the run to these "
                         "papers ('' = all manifest papers, the default)")
    args = ap.parse_args()

    manifest = read_jsonl(args.manifest)
    if args.papers:
        wanted = {p.strip() for p in args.papers.split(',') if p.strip()}
        manifest = [m for m in manifest if m['arxiv_id'] in wanted]
        missing = wanted - {m['arxiv_id'] for m in manifest}
        if missing:
            sys.exit(f'--papers ids not in manifest: {sorted(missing)}')
        print(f'--papers filter: {len(manifest)} of {len(wanted)} requested papers')
    latex_rows = {r['key']: r for r in read_jsonl(args.latex_log)}
    done = done_keys(args.log, 'key')
    out_root = Path(args.out_dir)
    extracted = Path(args.extracted)

    todo = [(m['arxiv_id'], ver['v']) for m in manifest for ver in m['versions']]
    n_proc = n_ckpt = n_reason = n_fail = 0
    results: list[dict] = []

    for aid, v in tqdm(todo, desc='collect_figures'):
        key = f'{aid}v{v}'
        if key in done:
            n_ckpt += 1
            continue
        if args.limit and n_proc >= args.limit:
            break
        n_proc += 1
        safe_id = aid.replace('/', '_')
        log = base_log(key, aid, v)

        lrow = latex_rows.get(key)
        root = (extracted / safe_id / f'v{v}').resolve()
        main_tex = None
        reason = None
        if lrow is None:
            reason = 'no latex_log row (no extracted tree)'
        elif not lrow.get('ok'):
            reason = f"latex_log row not ok: {lrow.get('error')}"
        elif (lrow.get('converter') or '') == 'pdf':
            reason = 'kind=pdf source (no LaTeX tree)'
        elif not root.is_dir():
            reason = f'extracted tree missing: {root}'
        else:
            main_tex = root / (lrow.get('main_tex') or '')
            if not lrow.get('main_tex') or not main_tex.is_file():
                reason = f"main tex missing: {lrow.get('main_tex')}"

        if reason is not None:
            log['reason'] = reason
            log['ok'] = True
            n_reason += 1
            tqdm.write(f'NO-TREE {key}: {reason}')
        else:
            try:
                collect_version(log, root, main_tex, out_root, safe_id, v,
                                args.max_figs, args.max_px)
                log['ok'] = True
            except Exception as exc:  # never let one version kill the run
                log['error'] = f'{type(exc).__name__}: {exc}'[:500]
                n_fail += 1
                tqdm.write(f'FAIL {key}: {log["error"]}')

        append_jsonl(args.log, log)
        done.add(key)
        results.append(log)

    ok_rows = [r for r in results if r['ok']]
    print('\n=== collect_figures summary ===')
    print(f'versions: {len(todo)} | processed this run: {n_proc} | '
          f'skipped via checkpoint: {n_ckpt} | no-tree reason rows: {n_reason} | '
          f'failures: {n_fail}')
    hist = Counter(r['n_attached'] for r in ok_rows)
    print('attach-count histogram: '
          + (', '.join(f'{k} figs x {hist[k]}' for k in sorted(hist)) or '-'))
    tot_bytes = sum(r['total_png_bytes'] for r in ok_rows)
    tot_files = sum(r['n_attached'] for r in ok_rows)
    print(f'total PNG: {tot_bytes / 1e6:.1f} MB across {tot_files} files')
    print('skips: filtered={} format={} unresolved={} error={}'.format(
        sum(r['skipped_filtered'] for r in ok_rows),
        sum(r['skipped_format'] for r in ok_rows),
        sum(r['skipped_unresolved'] for r in ok_rows),
        sum(r['skipped_error'] for r in ok_rows)))
    top = sorted(ok_rows, key=lambda r: r['total_png_bytes'], reverse=True)[:5]
    for r in top:
        print(f'  largest: {r["key"]}  {r["total_png_bytes"] / 1e6:.2f} MB '
              f'({r["n_attached"]} figs)')


if __name__ == '__main__':
    main()
