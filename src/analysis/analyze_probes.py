"""E4 verdict from the probe review log: content sensitivity vs length sensitivity.

E4 scores length-matched, known-direction synthetic edits of each probe paper's anchor
version with the SAME review rubric as the main study. This script reads the resulting
probe review log and separates the two candidate drivers of the score:

  - length-preserved degradations (shuffle_paragraphs, corrupt_numbers, typos) damage the
    content while keeping the document length ~unchanged -> a content-sensitive reviewer
    must go NEGATIVE here;
  - pure length edits (pad_filler adds content-free filler, drop_appendix removes an
    appendix whose substance the body already states) change the length while keeping the
    reviewable content ~unchanged -> a length-driven reviewer goes POSITIVE on pad and
    NEGATIVE on drop.

Inputs (both append-only JSONL, written by the probe construction / scoring stages):

  --log  (default data/processed/probe_reviews_log.jsonl)
      one row per (paper, variant, model, rep) scoring call, same shape as
      run_llm_review.py rows (parsed_ok, rating, soundness, presentation, contribution,
      accept, ...) plus a 'variant' field ('condition'/'probe' accepted as aliases).
      The unmodified anchor text is the variant 'base' ('original'/'anchor'/'identity'/
      'unmodified' accepted as aliases). Version under 'v', 'version' or 'anchor_v'
      (run_probe_review.py writes 'anchor_v' only).
  --probes-log  (default data/processed/probes_log.jsonl)
      one row per constructed (paper, variant) text: used only for the construction
      context table (n built / n skipped / length ratio vs base, via tolerant field
      lookup). Missing file -> the table is skipped with a note.

Per model (every model in --log, or just --model):

1. PER-VARIANT EFFECTS - per (paper, variant) mean rating across reps, paired delta vs
   the same paper's base mean. Per variant: n papers, mean delta, 95% CI (t-interval on
   the paired deltas), share of papers with delta<0 / ==0 / >0, mean |delta|. Subscore
   deltas (soundness / presentation / contribution) per variant, with attribution checks:
   typos should hit presentation hardest, corrupt_numbers should hit soundness hardest.
2. MIXED MODEL - rating ~ C(variant, Treatment(reference='base')) + (1|paper) on
   rep-level rows (statsmodels MixedLM); each variant coefficient with 95% CI and p.
3. THE 2x2 VERDICT - content sensitivity = mean paired delta over the length-preserved
   degradations (each paper contributes the mean of its available degradation deltas);
   length sensitivity = pad_filler / drop_appendix deltas plus their asymmetry
   (delta(pad) - delta(drop) on papers with both). A 3-sentence plain-language verdict is
   chosen MECHANICALLY from fixed thresholds:
     content-sensitive  iff mean degradation delta < -0.3 and its 95% CI excludes 0;
     length-sensitive   iff delta(pad_filler) > +0.15 or delta(drop_appendix) < -0.15,
                        with that variant's 95% CI excluding 0 (the verdict states which
                        trigger fired).
4. NOISE CONTEXT - every effect is compared against the E0/E1 true-placebo noise floor
   (mean |delta| between adjacent real versions with cosmetic text churn and unchanged
   figures), parsed from the 'TRUE PLACEBO' table line of --e0-summary
   (default results/e0_scores_summary.md). No parseable line -> stated as unavailable;
   no line for this exact model -> the mean floor across the e0 models is used, with a
   note. The floor is a between-version quantity, so ratios are context, not a test.

Degenerate-input safety: variants that were never built (e.g. drop_appendix skips papers
without an appendix) are reported as missing and the dependent statistics degrade to
'n/a' with a note; models with 1 rep per cell still work (CIs are across papers only);
papers whose base was never scored are EXCLUDED from the paired analysis and listed; an
unparseable FINAL log line (torn tail of a concurrent append) is skipped, a malformed
line anywhere else still raises. The --out markdown report is OVERWRITTEN each run and
the headline verdict lines are printed to stdout.

Run from repo root: python3 src/analysis/analyze_probes.py
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy import stats as sps

sys.path.insert(0, "src/collect")
from util import read_jsonl  # noqa: E402

BASE_ALIASES = {"base", "original", "anchor", "identity", "unmodified"}
DEGRADATIONS = ("shuffle_paragraphs", "corrupt_numbers", "typos")
LENGTH_VARIANTS = ("pad_filler", "drop_appendix")
CANON_ORDER = [*DEGRADATIONS, *LENGTH_VARIANTS]
SUBSCORES = ("soundness", "presentation", "contribution")
EXPECTED_SUBSCORE = {"typos": "presentation", "corrupt_numbers": "soundness"}

CONTENT_THRESHOLD = -0.3   # mean degradation delta below this (CI excl. 0) -> content-sensitive
PAD_THRESHOLD = +0.15      # pad_filler delta above this (CI excl. 0)       -> length-sensitive
DROP_THRESHOLD = -0.15     # drop_appendix delta below this (CI excl. 0)    -> length-sensitive
TIE_EPS = 1e-9


# ---------------------------------------------------------------- formatting
def fmt(x, nd: int = 2, signed: bool = True) -> str:
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "n/a"
    return f"{x:+.{nd}f}" if signed else f"{x:.{nd}f}"


def pct(x) -> str:
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "n/a"
    return f"{100 * x:.1f}%"


def ci_str(lo, hi) -> str:
    if lo is None or hi is None:
        return "n/a"
    return f"[{lo:+.2f}, {hi:+.2f}]"


def _is_num(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


# ------------------------------------------------------------------- loading
def _read_log_tolerant(path: str | Path) -> list[dict]:
    """read_jsonl, but a concurrent appender can tear the LAST line mid-write: an
    unparseable final line is skipped; a malformed line anywhere else still raises."""
    p = Path(path)
    if not p.exists():
        return []
    with p.open() as f:
        lines = [ln for ln in f if ln.strip()]
    rows: list[dict] = []
    for i, line in enumerate(lines):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if i == len(lines) - 1:
                break
            raise
    return rows


def variant_of(row: dict) -> str | None:
    v = row.get("variant", row.get("condition", row.get("probe")))
    if v is None:
        return None
    v = str(v).strip()
    return "base" if v.lower() in BASE_ALIASES else v


def load_probe_reviews(path: str, model_filter: str | None):
    """parsed_ok rating rows -> {model: {(arxiv_id, v): {variant: [rep rows]}}}.

    A rep row is {'rep', 'rating', 'soundness', 'presentation', 'contribution'} with
    subscores None when absent/non-numeric. Returns also (n_total, n_used, n_no_variant).
    """
    rows = _read_log_tolerant(path)
    by_model: dict[str, dict[tuple, dict[str, list[dict]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list)))
    n_used = n_no_variant = 0
    for row in rows:
        if not row.get("parsed_ok"):
            continue
        rating = row.get("rating")
        aid = row.get("arxiv_id")
        if aid is None or not _is_num(rating):
            continue
        variant = variant_of(row)
        if variant is None:
            n_no_variant += 1
            continue
        model = str(row.get("model") or "unknown")
        if model_filter and model != model_filter:
            continue
        v = row.get("v", row.get("version", row.get("anchor_v")))
        unit = (str(aid), int(v) if _is_num(v) else -1)
        rec = {"rep": row.get("rep", 0), "rating": float(rating)}
        for f in SUBSCORES:
            rec[f] = float(row[f]) if _is_num(row.get(f)) else None
        by_model[model][unit][variant].append(rec)
        n_used += 1
    return by_model, len(rows), n_used, n_no_variant


def _first_num(row: dict, keys: tuple[str, ...]):
    for k in keys:
        if _is_num(row.get(k)):
            return float(row[k])
    return None


def load_probes_info(path: str):
    """Construction context per variant from the probes log (tolerant field lookup).

    Returns {variant: {'n': built, 'n_skipped': int, 'skip_reasons': Counter,
    'ratios': [variant_len / base_len]}} or None when the file is missing/empty.
    """
    rows = _read_log_tolerant(path)
    if not rows:
        return None
    info: dict[str, dict] = defaultdict(
        lambda: {"n": 0, "n_skipped": 0, "skip_reasons": Counter(), "ratios": []})
    for row in rows:
        variant = variant_of(row)
        if variant is None:
            continue
        d = info[variant]
        reason = row.get("skip_reason") or row.get("skipped_reason") or row.get("reason")
        skipped = bool(row.get("skipped")) or bool(reason) \
            or str(row.get("status", "")).lower() in ("skipped", "skip")
        if skipped:
            d["n_skipped"] += 1
            d["skip_reasons"][str(reason or "unspecified")] += 1
            continue
        d["n"] += 1
        ratio = _first_num(row, ("len_ratio", "length_ratio", "words_ratio"))
        if ratio is None:
            cur = _first_num(row, ("n_words", "words", "n_words_variant", "variant_words",
                                   "n_chars", "chars", "n_chars_variant"))
            base = _first_num(row, ("n_words_base", "base_n_words", "base_words",
                                    "n_chars_base", "base_n_chars", "base_chars"))
            if cur is not None and base is not None and base > 0:
                ratio = cur / base
        if ratio is not None:
            d["ratios"].append(ratio)
    return dict(info)


def parse_noise_floors(path: str) -> dict[str, dict]:
    """{model: {'n', 'mean_delta', 'mean_abs'}} from the e0 summary's per-model
    '| TRUE PLACEBO ... |' table lines (columns: subset, n, mean delta, mean |delta|,
    share). Unparseable cells / missing file -> that model (or all) absent."""
    p = Path(path)
    if not p.exists():
        return {}
    floors: dict[str, dict] = {}
    model = None
    for line in p.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^##\s+Model:\s*(.+?)\s*$", line)
        if m:
            model = m.group(1)
            continue
        if model is None or not line.lstrip().startswith("|") or "TRUE PLACEBO" not in line:
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 4:
            continue
        try:
            floors[model] = {"n": int(cells[1]), "mean_delta": float(cells[2]),
                             "mean_abs": float(cells[3].replace("\\", ""))}
        except ValueError:
            continue
    return floors


# ---------------------------------------------------------------- statistics
def t_ci(vals: list[float]):
    """(mean, lo, hi, n): 95% t-interval. n==0 -> all None; n==1 -> CI None;
    sd==0 -> degenerate CI [mean, mean]."""
    n = len(vals)
    if n == 0:
        return None, None, None, 0
    m = float(np.mean(vals))
    if n == 1:
        return m, None, None, 1
    sd = float(np.std(vals, ddof=1))
    if sd == 0.0:
        return m, m, m, n
    half = float(sps.t.ppf(0.975, n - 1)) * sd / math.sqrt(n)
    return m, m - half, m + half, n


def ci_excludes_zero(lo, hi) -> bool:
    return lo is not None and hi is not None and (lo > 0 or hi < 0)


# ------------------------------------------------------------ per-model work
def unit_label(unit: tuple) -> str:
    aid, v = unit
    return f"{aid}v{v}" if v >= 0 else aid


def cell_means(reps: list[dict]) -> dict:
    out = {"rating": float(np.mean([r["rating"] for r in reps])), "n_reps": len(reps)}
    for f in SUBSCORES:
        xs = [r[f] for r in reps if r[f] is not None]
        out[f] = float(np.mean(xs)) if xs else None
    return out


def analyze_model(model: str, units: dict, floors: dict[str, dict]):
    """Returns (markdown_section, headline_lines)."""
    md: list[str] = [f"## Model: {model}", ""]

    excluded = sorted(unit_label(u) for u, vd in units.items() if "base" not in vd)
    paired = {u: vd for u, vd in units.items() if "base" in vd}
    cells = {(u, var): cell_means(reps)
             for u, vd in paired.items() for var, reps in vd.items()}

    variants_present = sorted({var for vd in paired.values() for var in vd} - {"base"})
    variant_order = [v for v in CANON_ORDER if v in variants_present] \
        + [v for v in variants_present if v not in CANON_ORDER]
    missing_canon = [v for v in CANON_ORDER if v not in variants_present]

    rep_counts = [c["n_reps"] for c in cells.values()]
    n_reps_lo = min(rep_counts) if rep_counts else 0
    n_reps_hi = max(rep_counts) if rep_counts else 0

    base_means = [cells[(u, "base")]["rating"] for u in paired]
    md.append(f"- papers with a scored base: {len(paired)} "
              f"(mean base rating {fmt(np.mean(base_means) if base_means else None, 2, signed=False)}) "
              f"| reps per (paper, variant) cell: {n_reps_lo}-{n_reps_hi}")
    if excluded:
        md.append(f"- EXCLUDED (variants scored but base missing): {len(excluded)} paper(s) "
                  f"- {', '.join(excluded[:8])}{', ...' if len(excluded) > 8 else ''}")
    if missing_canon:
        md.append(f"- variants with no scored rows for this model: {', '.join(missing_canon)} "
                  f"(e.g. drop_appendix is skipped for papers without an appendix)")
    if n_reps_hi == 1:
        md.append("- note: 1 rep per cell - paired deltas carry no within-cell averaging; "
                  "CIs are across papers only")
    md.append("")

    # paired deltas: variant -> {unit: delta}; subscore deltas alongside
    deltas: dict[str, dict[tuple, float]] = defaultdict(dict)
    sub_deltas: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for u, vd in paired.items():
        base = cells[(u, "base")]
        for var in vd:
            if var == "base":
                continue
            c = cells[(u, var)]
            deltas[var][u] = c["rating"] - base["rating"]
            for f in SUBSCORES:
                if c[f] is not None and base[f] is not None:
                    sub_deltas[var][f].append(c[f] - base[f])

    # ------------------------------------------------- 1. per-variant effects
    md.append("### 1. Per-variant paired effects (variant mean minus the same paper's "
              "base mean, across reps)")
    md.append("")
    md.append("| variant | n papers | mean delta | 95% CI | delta<0 | delta==0 | delta>0 "
              "| mean \\|delta\\| |")
    md.append("|---|---|---|---|---|---|---|---|")
    stats: dict[str, dict] = {}
    for var in variant_order:
        ds = list(deltas[var].values())
        m, lo, hi, n = t_ci(ds)
        if n == 0:
            md.append(f"| {var} | 0 | n/a | n/a | n/a | n/a | n/a | n/a |")
            continue
        sh_neg = sum(d < -TIE_EPS for d in ds) / n
        sh_zero = sum(abs(d) <= TIE_EPS for d in ds) / n
        sh_pos = sum(d > TIE_EPS for d in ds) / n
        mean_abs = float(np.mean(np.abs(ds)))
        stats[var] = {"mean": m, "lo": lo, "hi": hi, "n": n, "mean_abs": mean_abs}
        md.append(f"| {var} | {n} | {fmt(m)} | {ci_str(lo, hi)} | {pct(sh_neg)} | "
                  f"{pct(sh_zero)} | {pct(sh_pos)} | {fmt(mean_abs, 2, signed=False)} |")
    md.append("")

    # subscore attribution
    md.append("Subscore deltas (same pairing; n in parentheses = papers where both sides "
              "report that subscore):")
    md.append("")
    md.append("| variant | d soundness | d presentation | d contribution | largest mover "
              "| expected | right subscore? |")
    md.append("|---|---|---|---|---|---|---|")
    attribution: dict[str, str] = {}
    for var in variant_order:
        if not deltas[var]:
            continue
        row_cells, movers = [], {}
        for f in SUBSCORES:
            xs = sub_deltas[var][f]
            if xs:
                mu = float(np.mean(xs))
                movers[f] = mu
                row_cells.append(f"{fmt(mu)} (n={len(xs)})")
            else:
                row_cells.append("n/a")
        if movers:
            top_abs = max(abs(x) for x in movers.values())
            top = sorted(f for f, x in movers.items() if abs(x) >= top_abs - TIE_EPS)
            largest = "tie: " + "/".join(top) if (len(top) > 1 and top_abs > TIE_EPS) \
                else ("none (all 0)" if top_abs <= TIE_EPS else top[0])
        else:
            largest = "n/a"
        expected = EXPECTED_SUBSCORE.get(var, "-")
        if expected == "-" or not movers:
            right = "-" if expected == "-" else "n/a"
        else:
            right = "YES" if largest == expected else "NO"
            attribution[var] = right
        md.append(f"| {var} | {row_cells[0]} | {row_cells[1]} | {row_cells[2]} | "
                  f"{largest} | {expected} | {right} |")
    for var, exp in EXPECTED_SUBSCORE.items():
        if var in deltas and deltas[var]:
            md.append(f"- {var} -> {exp} should move most: "
                      f"**{attribution.get(var, 'n/a')}**")
        else:
            md.append(f"- {var} -> {exp} should move most: n/a (variant not scored)")
    md.append("")

    # ------------------------------------------------------- 2. mixed model
    md.append("### 2. Mixed model: rating ~ C(variant, base reference) + (1|paper), "
              "rep-level rows")
    md.append("")
    mm_coefs: dict[str, dict] = {}
    try:
        import pandas as pd
        import statsmodels.formula.api as smf
        reg_rows = [{"rating": r["rating"], "variant": var, "paper": unit_label(u)}
                    for u, vd in paired.items() for var, reps in vd.items() for r in reps]
        df = pd.DataFrame(reg_rows)
        if not df.empty and df["paper"].nunique() >= 3 and df["variant"].nunique() >= 2:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fit = smf.mixedlm("rating ~ C(variant, Treatment(reference='base'))",
                                  df, groups=df["paper"]).fit(reml=True)
            md.append(f"- n rep-level rows: {len(df)} | papers: {df['paper'].nunique()} "
                      f"| variants: {df['variant'].nunique() - 1} + base")
            md.append("")
            md.append("| variant (vs base) | coef | 95% CI | p |")
            md.append("|---|---|---|---|")
            for name in fit.params.index:
                m = re.search(r"\[T\.(.+)\]$", name)
                if not m:
                    continue
                var = m.group(1)
                coef, se, p = float(fit.params[name]), float(fit.bse[name]), float(fit.pvalues[name])
                mm_coefs[var] = {"coef": coef, "lo": coef - 1.96 * se,
                                 "hi": coef + 1.96 * se, "p": p}
                md.append(f"| {var} | {fmt(coef)} | "
                          f"{ci_str(coef - 1.96 * se, coef + 1.96 * se)} | "
                          f"{fmt(p, 4, signed=False)} |")
        else:
            md.append("- n/a (needs >= 3 papers with a scored base and >= 1 non-base variant)")
    except Exception as exc:
        md.append(f"- n/a ({type(exc).__name__}: {exc})")
    md.append("")

    # ---------------------------------------------------------- 3. verdict
    md.append("### 3. The 2x2 verdict: content sensitivity vs length sensitivity")
    md.append("")

    # content axis: per paper, mean delta over its available length-preserved degradations
    deg_present = [v for v in DEGRADATIONS if deltas[v]]
    per_unit_deg: list[float] = []
    for u in paired:
        ds = [deltas[v][u] for v in deg_present if u in deltas[v]]
        if ds:
            per_unit_deg.append(float(np.mean(ds)))
    c_mean, c_lo, c_hi, c_n = t_ci(per_unit_deg)
    content_fired = (c_mean is not None and c_mean < CONTENT_THRESHOLD
                     and ci_excludes_zero(c_lo, c_hi))

    # length axis: pad_filler / drop_appendix individually + their asymmetry
    pad = stats.get("pad_filler")
    drop = stats.get("drop_appendix")
    pad_fired = (pad is not None and pad["mean"] > PAD_THRESHOLD
                 and ci_excludes_zero(pad["lo"], pad["hi"]))
    drop_fired = (drop is not None and drop["mean"] < DROP_THRESHOLD
                  and ci_excludes_zero(drop["lo"], drop["hi"]))
    length_fired = pad_fired or drop_fired
    triggers = [name for name, f in (("pad_filler", pad_fired),
                                     ("drop_appendix", drop_fired)) if f]

    both_units = [u for u in paired
                  if u in deltas["pad_filler"] and u in deltas["drop_appendix"]]
    asym = [deltas["pad_filler"][u] - deltas["drop_appendix"][u] for u in both_units]
    a_mean, a_lo, a_hi, a_n = t_ci(asym)

    md.append(f"- content sensitivity (mean paired delta over length-preserved "
              f"degradations {', '.join(deg_present) if deg_present else '(none scored)'}): "
              f"{fmt(c_mean)} (95% CI {ci_str(c_lo, c_hi)}, n={c_n} papers) | threshold "
              f"< {CONTENT_THRESHOLD:+.2f} with CI excluding 0 -> "
              f"{'FIRED' if content_fired else 'not fired'}")
    md.append(f"- length sensitivity: pad_filler "
              + (f"{fmt(pad['mean'])} (95% CI {ci_str(pad['lo'], pad['hi'])}, "
                 f"n={pad['n']}) vs > {PAD_THRESHOLD:+.2f} -> "
                 f"{'FIRED' if pad_fired else 'not fired'}" if pad else "n/a (not scored)")
              + " | drop_appendix "
              + (f"{fmt(drop['mean'])} (95% CI {ci_str(drop['lo'], drop['hi'])}, "
                 f"n={drop['n']}) vs < {DROP_THRESHOLD:+.2f} -> "
                 f"{'FIRED' if drop_fired else 'not fired'}" if drop else "n/a (not scored)"))
    md.append(f"- pad-minus-drop asymmetry (papers with both, n={a_n}): {fmt(a_mean)} "
              f"(95% CI {ci_str(a_lo, a_hi)})"
              + ("" if a_n else " - n/a: no paper has both length variants"))
    md.append("")

    # a model with no paired data on EITHER axis gets no quadrant: 'neither fired'
    # would wrongly read as a measured null
    assessable = c_mean is not None or pad is not None or drop is not None
    quadrant = {
        (True, False): "content-sensitive, length-insensitive: consistent with a genuine "
                       "quality signal at these effect sizes",
        (True, True): "content-sensitive AND length-sensitive: a real quality signal "
                      "contaminated by length bias",
        (False, True): "content-insensitive but length-sensitive: behaves like a "
                       "length-driven reviewer",
        (False, False): "neither fired: effects are below both thresholds (noise-level "
                        "at this sample size)",
    }[(content_fired, length_fired)]
    if not assessable:
        quadrant = ("not assessable: no variant has paired (base, variant) scores for "
                    "this model, so neither axis could be evaluated")

    mark = "**<= this model**"
    md.append("|  | length-sensitive: NO | length-sensitive: YES |")
    md.append("|---|---|---|")
    md.append(f"| content-sensitive: YES | genuine quality signal"
              f"{' ' + mark if content_fired and not length_fired else ''} "
              f"| quality signal + length bias"
              f"{' ' + mark if content_fired and length_fired else ''} |")
    md.append(f"| content-sensitive: NO | insensitive / noise-level"
              f"{' ' + mark if assessable and not content_fired and not length_fired else ''} "
              f"| length-driven reviewer"
              f"{' ' + mark if not content_fired and length_fired else ''} |")
    md.append("")

    if c_mean is None:
        s1 = ("Content sensitivity could not be assessed: none of the length-preserved "
              "degradations (shuffle_paragraphs, corrupt_numbers, typos) has scored "
              "paired rows for this model.")
    else:
        s1 = (f"This model {'IS' if content_fired else 'is NOT'} content-sensitive by the "
              f"mechanical rule: the mean rating delta across the length-preserved "
              f"degradations is {fmt(c_mean)} (95% CI {ci_str(c_lo, c_hi)}, n={c_n}), "
              f"which {'crosses' if content_fired else 'does not cross'} the "
              f"{CONTENT_THRESHOLD:+.2f} threshold with a CI excluding 0.")
    if pad is None and drop is None:
        s2 = ("Length sensitivity could not be assessed: neither pad_filler nor "
              "drop_appendix has scored paired rows for this model.")
    else:
        parts = []
        if pad:
            parts.append(f"pad_filler moved ratings by {fmt(pad['mean'])} "
                         f"(95% CI {ci_str(pad['lo'], pad['hi'])}) against the "
                         f"{PAD_THRESHOLD:+.2f} threshold")
        if drop:
            parts.append(f"drop_appendix by {fmt(drop['mean'])} "
                         f"(95% CI {ci_str(drop['lo'], drop['hi'])}) against the "
                         f"{DROP_THRESHOLD:+.2f} threshold")
        s2 = (f"It {'IS' if length_fired else 'is NOT'} length-sensitive: "
              + " and ".join(parts)
              + (f", so the trigger fired on {' and '.join(triggers)}" if length_fired
                 else ", so neither trigger fired") + ".")
    s3 = f"Verdict: {quadrant}."
    md.append(f"{s1} {s2} {s3}")
    md.append("")

    # ------------------------------------------------------ 4. noise context
    md.append("### 4. Noise context (E0/E1 true-placebo floor)")
    md.append("")
    floor = floors.get(model)
    floor_abs = None
    if floor is not None:
        floor_abs = floor["mean_abs"]
        md.append(f"- true-placebo noise floor for this model (from the e0 summary): "
                  f"mean |delta| {fmt(floor_abs, 2, signed=False)} (n={floor['n']} "
                  f"adjacent cosmetic pairs, mean delta {fmt(floor['mean_delta'])})")
    elif floors:
        vals = [f["mean_abs"] for f in floors.values()]
        floor_abs = float(np.mean(vals))
        md.append(f"- no true-placebo line for this exact model; using the mean floor "
                  f"across the {len(floors)} e0 models: mean |delta| "
                  f"{fmt(floor_abs, 2, signed=False)} "
                  f"(range {min(vals):.2f}-{max(vals):.2f})")
    else:
        md.append("- true-placebo noise floor UNAVAILABLE (no parseable 'TRUE PLACEBO' "
                  "line in the e0 summary) - effects below are reported without a noise "
                  "reference.")
    if floor_abs is not None and floor_abs > 0:
        md.append("- the floor is a between-adjacent-version quantity from real revisions, "
                  "so ratios are context, not a significance test.")
        md.append("")
        md.append("| effect | mean delta | \\|effect\\| / noise floor | exceeds floor? |")
        md.append("|---|---|---|---|")
        rows = [("content (mean degradation delta)", c_mean)] \
            + [(var, stats[var]["mean"]) for var in variant_order if var in stats]
        for label, m in rows:
            if m is None:
                md.append(f"| {label} | n/a | n/a | n/a |")
            else:
                ratio = abs(m) / floor_abs
                md.append(f"| {label} | {fmt(m)} | {ratio:.1f}x | "
                          f"{'YES' if abs(m) > floor_abs else 'no'} |")
    md.append("")

    floor_note = (f"floor |d| {fmt(floor_abs, 2, signed=False)}" if floor_abs is not None
                  else "floor n/a")
    headlines = [
        f"[{model}] content: mean degradation delta {fmt(c_mean)} "
        f"(CI {ci_str(c_lo, c_hi)}, n={c_n}) -> "
        + ("n/a (no paired degradation data)" if c_mean is None
           else ("CONTENT-SENSITIVE" if content_fired else "not content-sensitive")),
        f"[{model}] length: pad {fmt(pad['mean']) if pad else 'n/a'} / drop "
        f"{fmt(drop['mean']) if drop else 'n/a'} | asymmetry {fmt(a_mean)} -> "
        + ("n/a (no paired length-variant data)" if pad is None and drop is None
           else (f"LENGTH-SENSITIVE (via {', '.join(triggers)})" if length_fired
                 else "not length-sensitive")),
        f"[{model}] verdict: {quadrant} | {floor_note}",
    ]
    return "\n".join(md), headlines


# ------------------------------------------------------------------- driver
def main() -> None:
    ap = argparse.ArgumentParser(description="E4 probe analysis: content vs length sensitivity.")
    ap.add_argument("--log", default="data/processed/probe_reviews_log.jsonl",
                    help="probe review log (run_llm_review-shaped rows + 'variant')")
    ap.add_argument("--probes-log", default="data/processed/probes_log.jsonl",
                    help="probe construction log (context table only)")
    ap.add_argument("--e0-summary", default="results/e0_scores_summary.md",
                    help="e0 report to parse the TRUE PLACEBO noise floor from")
    ap.add_argument("--model", default=None,
                    help="only this model (default: every model in the log, separately)")
    ap.add_argument("--out", default="results/e4_probes_summary.md",
                    help="markdown report, OVERWRITTEN each run")
    args = ap.parse_args()

    by_model, n_total, n_used, n_no_variant = load_probe_reviews(args.log, args.model)
    probes_info = load_probes_info(args.probes_log)
    floors = parse_noise_floors(args.e0_summary)

    header = ["# E4: synthetic edit probes - content sensitivity vs length sensitivity", "",
              f"- review log: `{args.log}` - {n_total} rows, {n_used} parsed_ok rating "
              f"rows used" + (f", {n_no_variant} rows without a variant field skipped"
                              if n_no_variant else "")
              + (f" (model filter: {args.model})" if args.model else ""),
              f"- thresholds (fixed upfront): content-sensitive iff mean degradation "
              f"delta < {CONTENT_THRESHOLD:+.2f} with 95% CI excluding 0; "
              f"length-sensitive iff pad_filler delta > {PAD_THRESHOLD:+.2f} or "
              f"drop_appendix delta < {DROP_THRESHOLD:+.2f} with 95% CI excluding 0",
              ""]

    header.append("## Probe construction context (from the probes log)")
    header.append("")
    if probes_info is None:
        header.append(f"_Probes log `{args.probes_log}` missing or empty - construction "
                      f"context skipped._")
    else:
        header.append("| variant | n built | n skipped | mean length ratio vs base |")
        header.append("|---|---|---|---|")
        order = [v for v in ["base", *CANON_ORDER] if v in probes_info] \
            + sorted(v for v in probes_info if v not in ("base", *CANON_ORDER))
        for var in order:
            d = probes_info[var]
            ratio = (f"{np.mean(d['ratios']):.2f} (n={len(d['ratios'])})"
                     if d["ratios"] else "n/a")
            skip = str(d["n_skipped"])
            if d["n_skipped"]:
                top = ", ".join(f"{k}: {c}" for k, c in d["skip_reasons"].most_common(2))
                skip += f" ({top})"
            header.append(f"| {var} | {d['n']} | {skip} | {ratio} |")
    header.append("")

    sections, headlines = [], []
    for model in sorted(by_model):
        sec, hl = analyze_model(model, by_model[model], floors)
        sections.append(sec)
        headlines.extend(hl)
    if not by_model:
        sections.append("_No parsed_ok rating rows with a variant field found in the "
                        "probe review log - nothing to analyze._")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(header + sections) + "\n", encoding="utf-8")

    print("\n=== analyze_probes (E4) summary ===")
    print(f"log rows: {n_total} | parsed_ok used: {n_used} | models: "
          f"{', '.join(sorted(by_model)) or 'none'}")
    for line in headlines:
        print(line)
    print(f"report -> {out}")


if __name__ == "__main__":
    main()
