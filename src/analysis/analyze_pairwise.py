"""E2 verdict: does pairwise presentation beat absolute scoring at ordering versions?

Reads the pairwise judgment log written by the E2 runner (one JSON row per
(model, paper pair, presentation order, rep) call), keeps only parsed_ok rows, and
reports per model:

1. LATER-PICK RATE - share of judgments picking the chronologically later version,
   overall and split by churn_class (diffs joined on arxiv_id+v_from+v_to), each with a
   95% Wilson CI plus tie/early shares; the TRUE-PLACEBO subset (churn_class ==
   'cosmetic' AND figs_any_change == False: text and figures both unchanged) is reported
   separately - there is no real difference to find there, so later-pick should be
   ~= early-pick.
2. POSITION BIAS - P(picking presentation label A) with a Wilson CI over non-tie
   judgments whose label is known; plus order consistency: among pairs judged in BOTH
   presentation orders, the share whose per-order modal choice names the same version in
   both orders vs flips with position; one-order-only pairs are counted and excluded.
3. ABSOLUTE-MODE COMPARISON - from the --reviews-log scores of the SAME model string:
   adjacent-pair direction from per-version mean ratings (later > earlier -> 'later',
   equal -> 'tie'), tabled against the pairwise majority direction on the SAME pairs
   (restricted to pairs scored in both logs; n reported), with the 3x3 cross-tab and the
   share of pairs where the two modes agree.
4. MARGIN CALIBRATION - later-pick/tie/early shares by the runner's margin bucket
   (slight / clear / decisive), with a Wilson CI on later-pick.

Expected pairwise-log row (canonical):
    {"arxiv_id": "...", "v_from": 1, "v_to": 2,    # chronological pair, v_from < v_to
     "order": "earlier_first" | "later_first",      # which version was shown as label A
     "choice": "A" | "B" | "tie",                   # the model's pick, in label space
     "margin": "slight" | "clear" | "decisive" | null,
     "model": "...", "parsed_ok": true, ...}
Tolerated variants: versions as v_a/v_b (or a_v/b_v) in PRESENTATION order (the order
field is then unnecessary); a boolean a_is_later / later_first / swapped instead of
'order'; the pick under choice/pick/winner/better/preferred (values a/b/tie or
earlier/later) or as picked_v / chosen_v (a version number). A parsed_ok row whose pair
or pick cannot be resolved is counted as unresolved and reported, never guessed.

Degenerate-input guards: a missing diffs FILE skips the churn split with a note, while
judged pairs missing from a present diffs file fall into an 'unknown (pair not in
diffs)' bucket (count reported); pairs judged in only one order are excluded from order
consistency and counted; a model absent from the reviews log (or a missing reviews-log
file) skips section 3 with a note; ties and label-less rows are excluded from position
bias and counted. Both logs are appended to by concurrently-running callers, so an
unparseable FINAL line is skipped (torn mid-write tail); a malformed line anywhere else
still raises.

The --out markdown report is OVERWRITTEN each run; headline numbers are printed.

Run from repo root: python3 src/analysis/analyze_pairwise.py
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, "src/collect")
from util import read_jsonl  # noqa: E402

Z95 = 1.959963984540054
TIE_EPS = 1e-9
CHURN_ORDER = ["cosmetic", "minor", "major"]
MARGIN_ORDER = ["slight", "clear", "decisive"]
TIE_WORDS = {"tie", "equal", "same", "neither", "draw", "no_preference"}


# ---------------------------------------------------------------- formatting / stats
def pct(x) -> str:
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "n/a"
    return f"{100 * x:.1f}%"


def wilson_ci(k: int, n: int) -> tuple[float | None, float | None]:
    """95% Wilson score interval for a binomial proportion k/n."""
    if n == 0:
        return None, None
    p = k / n
    z2 = Z95 * Z95
    denom = 1 + z2 / n
    center = (p + z2 / (2 * n)) / denom
    half = Z95 * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def _is_int(x) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def _is_num(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


# ------------------------------------------------------------------------- loading
def _read_log_tolerant(path: str | Path) -> list[dict]:
    """read_jsonl for a log another process may append to while we run.

    A read can catch the final row mid-write, so an unparseable LAST line is skipped
    (the torn tail of a concurrent append). A malformed line anywhere else still
    raises - that would be real corruption, not a race, and must not be silent.
    """
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
                break  # torn trailing line from a concurrent append
            raise
    return rows


def _order_a_is_later(row: dict) -> bool | None:
    """Which version was shown as label A? True = the later one. None = unknown."""
    if isinstance(row.get("a_is_later"), bool):
        return row["a_is_later"]
    for f in ("later_first", "swapped", "reversed"):
        if isinstance(row.get(f), bool):
            return row[f]
    o = row.get("order")
    if isinstance(o, str):
        o = o.strip().lower()
        if o in ("earlier_first", "from_first", "chrono", "chronological", "ab", "forward"):
            return False
        if o in ("later_first", "to_first", "ba", "backward"):
            return True
    return None


def normalize_judgment(row: dict) -> dict | None:
    """One parsed_ok log row -> a canonical judgment, or None when unresolvable.

    Returns {aid, v_from, v_to, a_is_later, pick in {'later','earlier','tie'},
    label in {'A','B',None}, margin}. v_from < v_to always (chronological order).
    """
    aid = row.get("arxiv_id")
    if aid is None:
        return None
    aid = str(aid)

    v_a = row.get("v_a", row.get("a_v"))
    v_b = row.get("v_b", row.get("b_v"))
    if _is_int(v_a) and _is_int(v_b) and v_a != v_b:
        v_from, v_to = min(v_a, v_b), max(v_a, v_b)
        a_is_later = v_a > v_b
    elif _is_int(row.get("v_from")) and _is_int(row.get("v_to")):
        v_from, v_to = row["v_from"], row["v_to"]
        if v_from == v_to:
            return None
        if v_from > v_to:
            v_from, v_to = v_to, v_from
        a_is_later = _order_a_is_later(row)
    else:
        return None

    pick: str | None = None
    label: str | None = None
    raw = next((row[f] for f in ("choice", "pick", "winner", "better", "preferred")
                if isinstance(row.get(f), str) and row[f].strip()), None)
    if raw is not None:
        c = raw.strip().lower()
        if c in TIE_WORDS:
            pick = "tie"
        elif c in ("a", "b"):
            if a_is_later is None:
                return None  # a label-space pick with no way to map label -> version
            label = c.upper()
            pick = "later" if ((label == "A") == a_is_later) else "earlier"
        elif c in ("later", "newer", "second_version", "v_to", "to"):
            pick = "later"
        elif c in ("earlier", "older", "first_version", "v_from", "from"):
            pick = "earlier"
    if pick is None:
        pv = row.get("picked_v", row.get("chosen_v"))
        if _is_int(pv):
            pick = "later" if pv == v_to else ("earlier" if pv == v_from else None)
    if pick is None:
        return None
    if label is None and pick != "tie" and a_is_later is not None:
        label = "A" if ((pick == "later") == a_is_later) else "B"

    m = row.get("margin")
    margin = m.strip().lower() if isinstance(m, str) and m.strip() else None
    return {"aid": aid, "v_from": v_from, "v_to": v_to, "a_is_later": a_is_later,
            "pick": pick, "label": label, "margin": margin}


def load_pairwise(path: str, model_filter: str | None):
    """-> ({model: [judgments]}, n_rows, n_parsed_ok_kept, {model: n_unresolved})."""
    rows = _read_log_tolerant(path)
    by_model: dict[str, list[dict]] = defaultdict(list)
    unresolved: Counter = Counter()
    n_parsed = 0
    for row in rows:
        if not row.get("parsed_ok"):
            continue
        model = str(row.get("model") or "unknown")
        if model_filter and model != model_filter:
            continue
        n_parsed += 1
        j = normalize_judgment(row)
        if j is None:
            unresolved[model] += 1
        else:
            by_model[model].append(j)
    return by_model, len(rows), n_parsed, unresolved


def load_diffs(path: str):
    """(arxiv_id, v_from, v_to) -> full diff row, or None when the file is missing."""
    if not Path(path).exists():
        return None
    return {(str(r["arxiv_id"]), int(r["v_from"]), int(r["v_to"])): r
            for r in read_jsonl(path)}


def load_absolute_means(path: str):
    """{model: {(arxiv_id, v): mean rating across reps}} from parsed_ok reviews-log
    rows (rating top-level or under 'review'; version under 'v' or 'version').
    None when the file is missing."""
    if not Path(path).exists():
        return None
    acc: dict[str, dict[tuple, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in _read_log_tolerant(path):
        if not row.get("parsed_ok"):
            continue
        review = row.get("review") if isinstance(row.get("review"), dict) else {}
        rating = row.get("rating", review.get("rating"))
        aid = row.get("arxiv_id")
        v = row.get("v", row.get("version"))
        if aid is None or v is None or not _is_num(rating):
            continue
        acc[str(row.get("model") or "unknown")][(str(aid), int(v))].append(float(rating))
    return {m: {k: sum(vs) / len(vs) for k, vs in d.items()} for m, d in acc.items()}


# ------------------------------------------------------------------------ analysis
def rate_line(label: str, sub: list[dict]) -> str:
    """One markdown row: n, later-pick share with Wilson CI, tie share, early share."""
    n = len(sub)
    if n == 0:
        return f"| {label} | 0 | n/a | n/a | n/a | n/a |"
    c = Counter(j["pick"] for j in sub)
    lo, hi = wilson_ci(c["later"], n)
    return (f"| {label} | {n} | {pct(c['later'] / n)} | [{pct(lo)}, {pct(hi)}] | "
            f"{pct(c['tie'] / n)} | {pct(c['earlier'] / n)} |")


def modal_pick(picks: list[str]) -> str | None:
    """The unique most common pick, or None when tied for most common."""
    top = Counter(picks).most_common(2)
    if len(top) == 2 and top[0][1] == top[1][1]:
        return None
    return top[0][0]


def pair_direction(picks: list[str]) -> str:
    """Majority direction of a pair's judgments: later / earlier / tie (incl. split)."""
    c = Counter(picks)
    if c["later"] > c["earlier"]:
        return "later"
    if c["earlier"] > c["later"]:
        return "earlier"
    return "tie"


def analyze_model(model: str, js: list[dict], diffs, means: dict | None,
                  manifest_ids: set, n_unresolved: int):
    """Returns (markdown_section, headline_lines)."""
    md: list[str] = [f"## Model: {model}", ""]
    pairs_set = {(j["aid"], j["v_from"], j["v_to"]) for j in js}
    papers = {j["aid"] for j in js}
    n_outside = sum(1 for a in papers if a not in manifest_ids)
    md.append(f"- judgments used: {len(js)} | distinct pairs: {len(pairs_set)} | "
              f"papers: {len(papers)}"
              + (f" ({n_outside} not in manifest)" if n_outside else "")
              + (f" | unresolved parsed_ok rows skipped: {n_unresolved}"
                 if n_unresolved else ""))
    md.append("")

    # ------------------------------------------------------- 1. later-pick rate
    md.append("### 1. Later-pick rate")
    md.append("")
    md.append("| subset | n | later-pick | 95% CI | tie | early |")
    md.append("|---|---|---|---|---|---|")
    md.append(rate_line("ALL", js))

    c_all = Counter(j["pick"] for j in js)
    n_all = len(js)
    all_later = c_all["later"] / n_all if n_all else None
    all_lo, all_hi = wilson_ci(c_all["later"], n_all)
    churn_rates: dict[str, float] = {}
    placebo_later = placebo_early = placebo_n = None

    if diffs is None:
        md.append("")
        md.append("_Note: diffs file not found - churn split and placebo subset skipped._")
    else:
        by_cls: dict[str, list[dict]] = defaultdict(list)
        placebo: list[dict] = []
        cos_figs: list[dict] = []
        n_nodiff = 0
        for j in js:
            row = diffs.get((j["aid"], j["v_from"], j["v_to"]))
            if row is None:
                n_nodiff += 1
                by_cls["unknown (pair not in diffs)"].append(j)
                continue
            cls = str(row.get("churn_class") or "unknown")
            by_cls[cls].append(j)
            if cls == "cosmetic":
                fac = row.get("figs_any_change")  # absent -> neither subset
                if fac is False:
                    placebo.append(j)
                elif fac is True:
                    cos_figs.append(j)
        for cls in CHURN_ORDER:
            if by_cls.get(cls):
                md.append(rate_line(cls, by_cls[cls]))
                churn_rates[cls] = (Counter(j["pick"] for j in by_cls[cls])["later"]
                                    / len(by_cls[cls]))
        for cls in sorted(set(by_cls) - set(CHURN_ORDER)):
            md.append(rate_line(cls, by_cls[cls]))
        md.append(rate_line("TRUE PLACEBO (cosmetic & figs unchanged)", placebo))
        md.append(rate_line("cosmetic & figure changes", cos_figs))
        md.append("")
        if placebo:
            cp = Counter(j["pick"] for j in placebo)
            placebo_n = len(placebo)
            placebo_later = cp["later"] / placebo_n
            placebo_early = cp["earlier"] / placebo_n
            md.append(f"- TRUE-PLACEBO check (no real difference to find, so later-pick "
                      f"should ~= early-pick): later {pct(placebo_later)} vs early "
                      f"{pct(placebo_early)} (diff "
                      f"{100 * (placebo_later - placebo_early):+.1f} pts, n={placebo_n}, "
                      f"ties {pct(cp['tie'] / placebo_n)})")
        else:
            md.append("- TRUE-PLACEBO check: n/a (no judged cosmetic pair with "
                      "figs_any_change == False)")
        if n_nodiff:
            md.append(f"- judgments whose pair is missing from the diffs file: {n_nodiff} "
                      f"(bucketed as 'unknown (pair not in diffs)')")
    md.append("")

    # --------------------------------------------------------- 2. position bias
    md.append("### 2. Position bias")
    md.append("")
    nonties = [j for j in js if j["pick"] != "tie"]
    labeled = [j for j in nonties if j["label"] in ("A", "B")]
    n_ties = len(js) - len(nonties)
    n_nolabel = len(nonties) - len(labeled)
    p_a = None
    if labeled:
        n_a = sum(j["label"] == "A" for j in labeled)
        p_a = n_a / len(labeled)
        lo, hi = wilson_ci(n_a, len(labeled))
        md.append(f"- P(pick label A): {pct(p_a)} [{pct(lo)}, {pct(hi)}] over "
                  f"n={len(labeled)} non-tie labeled judgments ({n_ties} tie(s) and "
                  f"{n_nolabel} label-unknown excluded); no-bias reference is 50%")
    else:
        md.append(f"- P(pick label A): n/a (no non-tie judgment with a known label; "
                  f"{n_ties} tie(s), {n_nolabel} label-unknown)")

    by_pair_order: dict[tuple, dict[bool, list[str]]] = \
        defaultdict(lambda: {False: [], True: []})
    n_noorder = 0
    for j in js:
        if j["a_is_later"] is None:
            n_noorder += 1
            continue
        by_pair_order[(j["aid"], j["v_from"], j["v_to"])][j["a_is_later"]].append(j["pick"])
    agree = flip = tiey = one_order = 0
    for d in by_pair_order.values():
        if d[False] and d[True]:
            m0, m1 = modal_pick(d[False]), modal_pick(d[True])
            if m0 is None or m1 is None or "tie" in (m0, m1):
                tiey += 1
            elif m0 == m1:
                agree += 1
            else:
                flip += 1
        else:
            one_order += 1
    both = agree + flip + tiey
    if both:
        md.append(f"- pairs judged in both orders: {both} - same version both orders: "
                  f"{agree} ({pct(agree / both)}) | flips with position: {flip} "
                  f"({pct(flip / both)}) | tie-involved/ambiguous: {tiey} "
                  f"({pct(tiey / both)})")
    else:
        md.append("- order consistency: n/a (no pair judged in both presentation orders)")
    md.append(f"- one-order-only pairs (excluded from consistency): {one_order}"
              + (f" | judgments without order info: {n_noorder}" if n_noorder else ""))
    md.append("")

    # --------------------------------------- 3. pairwise vs absolute comparison
    md.append("### 3. Pairwise vs absolute scoring (same pairs)")
    md.append("")
    pw_later = ab_later = agree_share = None
    n_common = 0
    if means is None:
        md.append(f"_Note: model '{model}' absent from the reviews log (or reviews log "
                  "missing) - absolute-mode comparison skipped._")
    else:
        pair_picks: dict[tuple, list[str]] = defaultdict(list)
        for j in js:
            pair_picks[(j["aid"], j["v_from"], j["v_to"])].append(j["pick"])
        common: list[tuple[str, str]] = []  # (pairwise direction, absolute direction)
        excluded = 0
        for aid, vf, vt in sorted(pair_picks):
            mf, mt = means.get((aid, vf)), means.get((aid, vt))
            if mf is None or mt is None:
                excluded += 1
                continue
            delta = mt - mf
            ab = "later" if delta > TIE_EPS else ("earlier" if delta < -TIE_EPS else "tie")
            common.append((pair_direction(pair_picks[(aid, vf, vt)]), ab))
        if not common:
            md.append(f"_Note: no judged pair has absolute scores for both versions in "
                      f"the reviews log ({excluded} pairs lack them) - comparison "
                      "skipped._")
        else:
            n_common = len(common)
            cpw = Counter(p for p, _ in common)
            cab = Counter(a for _, a in common)
            pw_later, ab_later = cpw["later"] / n_common, cab["later"] / n_common
            md.append("| mode | n pairs | later | tie | early |")
            md.append("|---|---|---|---|---|")
            md.append(f"| pairwise (majority of judgments) | {n_common} | "
                      f"{pct(pw_later)} | {pct(cpw['tie'] / n_common)} | "
                      f"{pct(cpw['earlier'] / n_common)} |")
            md.append(f"| absolute (mean-rating direction) | {n_common} | "
                      f"{pct(ab_later)} | {pct(cab['tie'] / n_common)} | "
                      f"{pct(cab['earlier'] / n_common)} |")
            md.append("")
            md.append("Cross-tab (rows = pairwise, columns = absolute):")
            md.append("")
            md.append("| pairwise \\ absolute | later | tie | earlier |")
            md.append("|---|---|---|---|")
            cross = Counter(common)
            for r in ("later", "tie", "earlier"):
                md.append(f"| {r} | {cross[(r, 'later')]} | {cross[(r, 'tie')]} | "
                          f"{cross[(r, 'earlier')]} |")
            n_agree = sum(1 for p, a in common if p == a)
            agree_share = n_agree / n_common
            md.append("")
            md.append(f"- agreement on the same pairs: {n_agree}/{n_common} "
                      f"({pct(agree_share)})")
            md.append(f"- judged pairs without absolute scores for both versions: "
                      f"{excluded} (excluded)")
    md.append("")

    # ----------------------------------------------------- 4. margin calibration
    md.append("### 4. Margin calibration")
    md.append("")
    by_margin: dict[str, list[dict]] = defaultdict(list)
    for j in js:
        by_margin[j["margin"] or "(none)"].append(j)
    md.append("| margin | n | later-pick | 95% CI | tie | early |")
    md.append("|---|---|---|---|---|---|")
    margin_rates: dict[str, float] = {}
    for mname in MARGIN_ORDER + sorted(set(by_margin) - set(MARGIN_ORDER)):
        if by_margin.get(mname):
            md.append(rate_line(mname, by_margin[mname]))
            margin_rates[mname] = (Counter(j["pick"] for j in by_margin[mname])["later"]
                                   / len(by_margin[mname]))
    md.append("")

    # ---------------------------------------------------------------- headlines
    hl1 = (f"[{model}] later-pick {pct(all_later)} [{pct(all_lo)}, {pct(all_hi)}] "
           f"(n={n_all})")
    for cls in CHURN_ORDER:
        if cls in churn_rates:
            hl1 += f" | {cls} {pct(churn_rates[cls])}"
    if placebo_later is not None:
        hl1 += f" | placebo later {pct(placebo_later)} vs early {pct(placebo_early)}"
    hl2 = (f"[{model}] P(label A) {pct(p_a)} (n={len(labeled)})"
           + (f" | both-order pairs {both}: agree {pct(agree / both)}, "
              f"flip {pct(flip / both)}" if both else " | no both-order pairs"))
    hl3 = (f"[{model}] pairwise vs absolute (n={n_common} pairs): later "
           f"{pct(pw_later)} vs {pct(ab_later)} | agreement {pct(agree_share)}"
           if n_common else f"[{model}] pairwise vs absolute: n/a")
    hl4 = ("[{}] margins: {}".format(
        model, " / ".join(f"{m} {pct(margin_rates[m])}"
                          for m in MARGIN_ORDER if m in margin_rates))
        if any(m in margin_rates for m in MARGIN_ORDER)
        else f"[{model}] margins: n/a")
    return "\n".join(md), [hl1, hl2, hl3, hl4]


# -------------------------------------------------------------------------- driver
def main() -> None:
    ap = argparse.ArgumentParser(
        description="E2 pairwise-discrimination analysis.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--log", default="data/processed/pairwise_log.jsonl",
                    help="pairwise judgment log from the E2 runner")
    ap.add_argument("--diffs", default="data/processed/version_diffs.jsonl",
                    help="adjacent-pair diffs with churn_class / figs_any_change")
    ap.add_argument("--manifest", default="data/processed/pilot_manifest.jsonl",
                    help="paper manifest (used to flag papers outside the corpus)")
    ap.add_argument("--reviews-log", default="data/processed/reviews_log.jsonl",
                    help="absolute-mode scores for the section-3 comparison")
    ap.add_argument("--model", default=None,
                    help="only this model (default: every model in the log, separately)")
    ap.add_argument("--out", default="results/e2_pairwise_summary.md",
                    help="markdown report, OVERWRITTEN each run")
    args = ap.parse_args()

    by_model, n_rows, n_parsed, unresolved = load_pairwise(args.log, args.model)
    diffs = load_diffs(args.diffs)
    manifest = read_jsonl(args.manifest)
    manifest_ids = {str(m.get("arxiv_id")) for m in manifest}
    abs_means = load_absolute_means(args.reviews_log)

    header = ["# E2: Pairwise version discrimination", "",
              f"- pairwise log: `{args.log}` - {n_rows} rows, {n_parsed} parsed_ok"
              + (f" (model filter: {args.model})" if args.model else "")
              + (f", {sum(unresolved.values())} unresolved "
                 f"({', '.join(f'{m}: {c}' for m, c in sorted(unresolved.items()))})"
                 if unresolved else ""),
              f"- diffs: `{args.diffs}`"
              + ("" if diffs is not None else " (MISSING - churn split skipped)")
              + f" | manifest: `{args.manifest}` ({len(manifest)} papers)",
              f"- reviews log (absolute mode): `{args.reviews_log}`"
              + (f" - models present: {', '.join(sorted(abs_means)) or 'none'}"
                 if abs_means is not None else " (MISSING - section 3 skipped)"),
              ""]

    sections, headlines = [], []
    for model in sorted(by_model):
        means = (abs_means or {}).get(model)
        sec, hl = analyze_model(model, by_model[model], diffs, means, manifest_ids,
                                unresolved.get(model, 0))
        sections.append(sec)
        headlines.extend(hl)
    if not by_model:
        sections.append("_No parsed_ok pairwise judgments found in the log - nothing "
                        "to analyze._")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(header + sections) + "\n")

    print("\n=== analyze_pairwise summary ===")
    print(f"log rows: {n_rows} | parsed_ok kept: {n_parsed} | models: "
          f"{', '.join(sorted(by_model)) or 'none'}")
    for line in headlines:
        print(line)
    print(f"report -> {out}")


if __name__ == "__main__":
    main()
