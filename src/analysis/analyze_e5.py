"""E5 verdict: do LLM score deltas track REAL HUMAN score deltas?

E5 anchors the version effect against ground truth. Each E5 pair is one paper scored
by the LLM at two revision rounds (manifest version stages 'round1' and 'round2' -
pre/post rebuttal, or rejected-then-resubmitted across ICLR years) for which the TRUE
human reviewer score delta is known. Per model (parsed_ok log rows only):

1. LLM DELTA - llm_delta = mean rating over round2 version(s) minus mean rating over
   round1 version(s), per paper (means first across reps within a version, then across
   the versions that share a stage, so an uneven rep count cannot tilt a stage mean).
2. CORRELATION - Pearson + Spearman of llm_delta vs human_delta on (a) same_scale
   pairs with human_delta_raw (raw deltas are only comparable within one realized
   ICLR score scale) and (b) ALL pairs with human_delta_z (per-scale z-scored).
   The scatter is summarized as a direction table: human up/down (pairs with a zero
   human delta are excluded and counted) x LLM up/flat/down, with the sign-agreement
   rate ((up,up)+(down,down)) / table-n and its Wilson 95% CI.
3. MAGNITUDE - OLS llm_delta ~ human_delta (slope +/- 95% CI, p) on both subsets,
   plus llm_delta vs round-2 decision: mean llm_delta for eventually re-accepted vs
   re-rejected pairs and the ROC AUC of llm_delta for re-acceptance (guarded: needs
   both classes; flagged when the minority class has < 5 pairs).
4. BASELINE - mean llm_delta overall with a 95% t-CI and a one-sample t-test vs 0
   (resubmissions are revised to address reviews, so the average should be positive),
   compared against the TRUE-PLACEBO noise floor parsed from --noise-summary
   (results/acl_scores_summary.md '| TRUE PLACEBO ... |' per-model table lines; if
   the model is absent but the file has exactly one floor, that one is used with a
   note, else n/a).
5. CAVEATS - auto-written honesty block: pair counts, match-coverage bias (pairs
   exist only for papers with an arXiv record in the post-2024-06 bulk harvest, i.e.
   recently revised papers), weak_identity count, same_scale count, scale warning.

Expected --pairs row schema (one row per paper pair, written by the E5 pair builder):
    arxiv_id (join key; 'paper_id'/'id' accepted; null on excluded rows),
    human_delta_raw (float|null; only meaningful when same_scale), human_delta_z
    (float|null), same_scale (bool), weak_identity (bool), decision2 (round-2
    decision; 'decision_round2'/'decision_r2' accepted; normalized
    case-insensitively: startswith 'accept' -> accepted, startswith 'reject' ->
    rejected, contains 'withdrawn' -> withdrawn, contains 'desk' -> desk_rejected),
    optional pair_type (e.g. rebuttal / resubmission). Rows with status ==
    'excluded' were dropped by the builder before scoring (exclude_reason says why,
    e.g. no_bulk_match); they are counted for the coverage/caveat sections but never
    joined. Rows without a status field are treated as includable.
The --manifest is in the fresh/ACL schema; version stages must be round1/round2
(read from versions[].stage, falling back to versions[].stage_fine).

Degenerate-input safety: correlations/regressions need >= 3 points with variation,
AUC needs both classes, the t-test needs n >= 2 and spread; a paper missing a scored
round, a scored paper without a pair row, and a pair row never scored are each
counted and reported, never crashed on. Missing input files yield an honest stub
report. A torn final log line (concurrent appender) is skipped; earlier corruption
still raises.

Outputs: --out markdown report and optional --json-out machine-readable per-model
stats, both OVERWRITTEN each run.

Run from repo root: python3 src/analysis/analyze_e5.py
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy import stats as sps

sys.path.insert(0, "src/collect")
from util import read_jsonl  # noqa: E402

TIE_EPS = 1e-9
ROUNDS = ("round1", "round2")


# ---------------------------------------------------------------- formatting
def fmt(x, nd: int = 2, signed: bool = False) -> str:
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "n/a"
    return f"{x:+.{nd}f}" if signed else f"{x:.{nd}f}"


def pct(x) -> str:
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "n/a"
    return f"{100 * x:.1f}%"


def finite_or_none(x):
    """json.dumps would emit invalid bare NaN - map non-finite floats to null."""
    if x is None:
        return None
    x = float(x)
    return x if math.isfinite(x) else None


def _is_num(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _fnum(x):
    return float(x) if _is_num(x) else None


# ------------------------------------------------------------------- loading
def _read_log_tolerant(path: str | Path) -> list[dict]:
    """read_jsonl for a log another process may be appending to: an unparseable
    LAST line is a torn concurrent-append tail and is skipped; a malformed line
    anywhere else is real corruption and still raises."""
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


def load_reviews(path: str, model_filter: str | None):
    """parsed_ok rating rows -> {model: [{arxiv_id, v, rep, rating}]}.
    Version accepted under 'v' or 'version'; rep under 'rep'/'rep_idx'/'replicate'."""
    rows = _read_log_tolerant(path)
    recs: dict[str, list[dict]] = defaultdict(list)
    n_used = 0
    for row in rows:
        if not row.get("parsed_ok"):
            continue
        aid = row.get("arxiv_id")
        v = row.get("v", row.get("version"))
        rating = row.get("rating")
        if rating is None and isinstance(row.get("review"), dict):
            rating = row["review"].get("rating")
        if aid is None or v is None or not _is_num(rating):
            continue
        model = str(row.get("model") or "unknown")
        if model_filter and model != model_filter:
            continue
        recs[model].append({
            "arxiv_id": str(aid), "v": int(v),
            "rep": row.get("rep", row.get("rep_idx", row.get("replicate", 0))),
            "rating": float(rating),
        })
        n_used += 1
    return recs, len(rows), n_used


def norm_decision(raw) -> str:
    """ICLR decision variants -> accepted/rejected/withdrawn/desk_rejected/unknown."""
    if raw is None:
        return "unknown"
    s = str(raw).strip().lower()
    if s.startswith("accept"):
        return "accepted"
    if s.startswith("reject"):
        return "rejected"
    if "desk" in s:
        return "desk_rejected"
    if "withdrawn" in s:
        return "withdrawn"
    return "unknown"


def load_pairs(path: str):
    """--pairs rows -> ({arxiv_id: pair dict}, meta dict).

    Rows with status == 'excluded' never enter the join; their exclude_reason
    counts feed the caveats. Last row wins on a duplicate arxiv_id (counted)."""
    rows = read_jsonl(path)
    pairs: dict[str, dict] = {}
    meta = {"n_rows": len(rows), "n_excluded": 0, "n_bad": 0,
            "exclude_reasons": Counter()}
    for r in rows:
        if str(r.get("status") or "") == "excluded":
            meta["n_excluded"] += 1
            meta["exclude_reasons"][str(r.get("exclude_reason") or "unspecified")] += 1
            continue
        aid = r.get("arxiv_id", r.get("paper_id", r.get("id")))
        if aid is None:
            meta["n_bad"] += 1
            continue
        aid = str(aid)
        if aid in pairs:
            meta["n_bad"] += 1
        dec2 = r.get("decision2", r.get("decision_round2", r.get("decision_r2")))
        pairs[aid] = {
            "arxiv_id": aid,
            "human_delta_raw": _fnum(r.get("human_delta_raw")),
            "human_delta_z": _fnum(r.get("human_delta_z")),
            "same_scale": bool(r.get("same_scale")),
            "weak_identity": bool(r.get("weak_identity")),
            "decision2": norm_decision(dec2),
            "pair_type": r.get("pair_type", r.get("kind")),
        }
    return pairs, meta


def load_round_versions(path: str):
    """manifest -> {arxiv_id: {'round1': [v,...], 'round2': [v,...]}} (stage first,
    stage_fine fallback; versions with any other stage are ignored for E5)."""
    rounds: dict[str, dict[str, list[int]]] = {}
    for m in read_jsonl(path):
        aid = str(m.get("arxiv_id"))
        d: dict[str, list[int]] = {r: [] for r in ROUNDS}
        for vv in m.get("versions") or []:
            stage = vv.get("stage")
            if stage not in ROUNDS:
                stage = vv.get("stage_fine")
            if stage in ROUNDS and _is_num(vv.get("v")):
                d[stage].append(int(vv["v"]))
        rounds[aid] = d
    return rounds


def parse_noise_floors(path: str) -> dict[str, dict]:
    """{model: {'n','mean_delta','mean_abs'}} from an E0-style summary's per-model
    '| TRUE PLACEBO ... |' table lines. Missing file / unparseable cells -> absent."""
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
def corr_pair(xs, ys):
    """(pearson, p, spearman, p) or Nones when degenerate (<3 points or constant)."""
    if len(xs) < 3 or len(set(xs)) < 2 or len(set(ys)) < 2:
        return None, None, None, None
    pr = sps.pearsonr(xs, ys)
    sr = sps.spearmanr(xs, ys)
    return float(pr.statistic), float(pr.pvalue), float(sr.statistic), float(sr.pvalue)


def regress(xs, ys):
    """OLS y ~ x -> (slope, lo, hi, p, n) or Nones when degenerate."""
    n = len(xs)
    if n < 3 or len(set(xs)) < 2:
        return None, None, None, None, n
    res = sps.linregress(xs, ys)
    if not math.isfinite(res.stderr) or res.stderr == 0:
        return float(res.slope), None, None, _fnum(res.pvalue), n
    half = float(sps.t.ppf(0.975, n - 2)) * float(res.stderr)
    return (float(res.slope), float(res.slope) - half, float(res.slope) + half,
            float(res.pvalue), n)


def wilson_ci(k: int, n: int, z: float = 1.96):
    if n == 0:
        return None, None
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return center - half, center + half


def safe_auc(labels: list[int], scores: list[float]):
    """ROC AUC via the Mann-Whitney U relation; None unless both classes present."""
    pos = [s for lab, s in zip(labels, scores) if lab == 1]
    neg = [s for lab, s in zip(labels, scores) if lab == 0]
    if not pos or not neg:
        return None
    u = sps.mannwhitneyu(pos, neg, alternative="two-sided").statistic
    return float(u) / (len(pos) * len(neg))


def t_ci(vals: list[float]):
    """(mean, lo, hi, p_vs_0, n): 95% t-CI and one-sample t-test against 0.
    n==0 -> all None; n==1 or sd==0 -> CI/p None."""
    n = len(vals)
    if n == 0:
        return None, None, None, None, 0
    m = float(np.mean(vals))
    sd = float(np.std(vals, ddof=1)) if n >= 2 else 0.0
    if n < 2 or sd == 0.0:
        return m, None, None, None, n
    half = float(sps.t.ppf(0.975, n - 1)) * sd / math.sqrt(n)
    p = float(sps.ttest_1samp(vals, 0.0).pvalue)
    return m, m - half, m + half, p, n


def dir_of(x: float, eps: float) -> str:
    return "flat" if abs(x) < eps else ("up" if x > 0 else "down")


# ------------------------------------------------------------ per-model work
def llm_deltas(recs: list[dict], rounds: dict[str, dict[str, list[int]]]):
    """Per-paper llm_delta for papers with BOTH rounds scored.

    Returns ({arxiv_id: {'llm_delta','llm_r1','llm_r2','n_vers_r1','n_vers_r2'}},
    n_missing_round, n_no_manifest). A stage mean is the unweighted mean of the
    per-version rep means of that stage's scored versions."""
    by_pv: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in recs:
        by_pv[r["arxiv_id"]][r["v"]].append(r["rating"])
    out: dict[str, dict] = {}
    n_missing_round = n_no_manifest = 0
    for aid, vd in by_pv.items():
        rd = rounds.get(aid)
        if rd is None:
            n_no_manifest += 1
            continue
        stage_means = {}
        stage_nvers = {}
        for stage in ROUNDS:
            vmeans = [float(np.mean(vd[v])) for v in rd[stage] if v in vd]
            stage_nvers[stage] = len(vmeans)
            stage_means[stage] = float(np.mean(vmeans)) if vmeans else None
        if stage_means["round1"] is None or stage_means["round2"] is None:
            n_missing_round += 1
            continue
        out[aid] = {
            "llm_r1": stage_means["round1"], "llm_r2": stage_means["round2"],
            "llm_delta": stage_means["round2"] - stage_means["round1"],
            "n_vers_r1": stage_nvers["round1"], "n_vers_r2": stage_nvers["round2"],
        }
    return out, n_missing_round, n_no_manifest


def analyze_model(model: str, recs: list[dict], pairs: dict[str, dict],
                  rounds: dict, floors: dict[str, dict], pair_meta: dict):
    """Returns (markdown_section, headline_lines, json_stats)."""
    deltas, n_missing_round, n_no_manifest = llm_deltas(recs, rounds)
    joined = []  # pair dict + llm fields, papers with both a pair row and a delta
    for aid, d in deltas.items():
        if aid in pairs:
            joined.append({**pairs[aid], **d})
    joined.sort(key=lambda r: r["arxiv_id"])
    n_no_pair_row = len(deltas) - len(joined)
    n_pair_unscored = len(pairs) - len(joined)

    md: list[str] = [f"## Model: {model}", ""]
    hl: list[str] = []
    js: dict = {"model": model}

    # ------------------------------------------------------------- coverage
    md.append("### Coverage")
    md.append("")
    md.append(f"- parsed_ok reviews: {len(recs)} | papers with both rounds scored: "
              f"{len(deltas)} | joined to a pair row: {len(joined)}")
    md.append(f"- skipped: {n_missing_round} paper(s) missing a scored round, "
              f"{n_no_manifest} scored paper(s) absent from the manifest, "
              f"{n_no_pair_row} scored paper(s) without a pair row, "
              f"{n_pair_unscored} pair row(s) never scored by this model")
    multi = [r for r in joined if r["n_vers_r1"] > 1 or r["n_vers_r2"] > 1]
    if multi:
        md.append(f"- {len(multi)} pair(s) have > 1 scored version in a round "
                  f"(per-version means averaged within the round)")
    types = Counter(str(r["pair_type"]) for r in joined if r.get("pair_type"))
    if types:
        md.append("- pair types: " + ", ".join(f"{k}: {v}" for k, v in
                                               sorted(types.items())))
    md.append("")
    js.update({"n_reviews": len(recs), "n_scored_pairs": len(joined),
               "n_both_rounds": len(deltas), "n_missing_round": n_missing_round,
               "n_no_manifest": n_no_manifest, "n_no_pair_row": n_no_pair_row,
               "n_pair_unscored": n_pair_unscored})

    if not joined:
        md.append("_No scored pair joins a pair row - nothing to correlate._")
        md.append("")
        hl.append(f"[{model}] E5: no scored pairs - verdict n/a")
        return "\n".join(md), hl, js

    # -------------------------------------------- 2. correlation with humans
    md.append("### 1. Correlation: LLM delta vs human delta")
    md.append("")
    md.append("| subset | n | Pearson r | p | Spearman rho | p |")
    md.append("|---|---|---|---|---|---|")
    subsets = {
        "same_scale, human_delta_raw": [(r["llm_delta"], r["human_delta_raw"])
                                        for r in joined if r["same_scale"]
                                        and r["human_delta_raw"] is not None],
        "all pairs, human_delta_z": [(r["llm_delta"], r["human_delta_z"])
                                     for r in joined
                                     if r["human_delta_z"] is not None],
    }
    for label, pts in subsets.items():
        xs = [h for _l, h in pts]
        ys = [l for l, _h in pts]
        pr, pr_p, sr, sr_p = corr_pair(xs, ys)
        if pr is None and len(pts) < 3:
            md.append(f"| {label} | {len(pts)} | n/a (n<3) | | n/a (n<3) | |")
        elif pr is None:
            md.append(f"| {label} | {len(pts)} | n/a (constant) | | n/a (constant) | |")
        else:
            md.append(f"| {label} | {len(pts)} | {fmt(pr, 2, signed=True)} | "
                      f"{fmt(pr_p, 4)} | {fmt(sr, 2, signed=True)} | {fmt(sr_p, 4)} |")
        key = "raw" if "raw" in label else "z"
        js.update({f"pearson_{key}": finite_or_none(pr),
                   f"pearson_{key}_p": finite_or_none(pr_p),
                   f"spearman_{key}": finite_or_none(sr),
                   f"spearman_{key}_p": finite_or_none(sr_p),
                   f"n_{key}": len(pts)})
    md.append("")

    # direction table: human up/down x llm up/flat/down (z preferred, raw fallback)
    table = {h: Counter() for h in ("up", "down")}
    n_human_flat = n_human_missing = 0
    for r in joined:
        hd = r["human_delta_z"] if r["human_delta_z"] is not None \
            else r["human_delta_raw"]
        if hd is None:
            n_human_missing += 1
            continue
        h = dir_of(hd, TIE_EPS)
        if h == "flat":
            n_human_flat += 1
            continue
        table[h][dir_of(r["llm_delta"], TIE_EPS)] += 1
    n_table = sum(sum(c.values()) for c in table.values())
    n_agree = table["up"]["up"] + table["down"]["down"]
    agree = n_agree / n_table if n_table else None
    w_lo, w_hi = wilson_ci(n_agree, n_table)
    md.append("Direction table (human delta from human_delta_z, raw fallback; "
              f"{n_human_flat} human-flat pair(s) excluded"
              + (f", {n_human_missing} without any human delta" if n_human_missing
                 else "") + "):")
    md.append("")
    md.append("| human \\ LLM | LLM up | LLM flat | LLM down |")
    md.append("|---|---|---|---|")
    for h in ("up", "down"):
        md.append(f"| human {h} | {table[h]['up']} | {table[h]['flat']} | "
                  f"{table[h]['down']} |")
    md.append("")
    if n_table:
        md.append(f"- sign agreement (up,up + down,down): {n_agree}/{n_table} = "
                  f"{pct(agree)} (Wilson 95% CI [{pct(w_lo)}, {pct(w_hi)}]); "
                  f"an LLM-flat pair counts as disagreement")
    else:
        md.append("- sign agreement: n/a (no pair with a nonzero human delta)")
    md.append("")
    js.update({"direction_table": {h: dict(table[h]) for h in table},
               "n_human_flat": n_human_flat, "agreement": finite_or_none(agree),
               "agreement_ci": [finite_or_none(w_lo), finite_or_none(w_hi)],
               "n_agreement_base": n_table})

    # ---------------------------------------------------------- 3. magnitude
    md.append("### 2. Magnitude")
    md.append("")
    md.append("OLS llm_delta ~ human_delta (slope = LLM rating points per human "
              "rating point):")
    md.append("")
    md.append("| subset | n | slope | 95% CI | p |")
    md.append("|---|---|---|---|---|")
    for label, pts in subsets.items():
        xs = [h for _l, h in pts]
        ys = [l for l, _h in pts]
        slope, lo, hi, p, n = regress(xs, ys)
        if slope is None:
            md.append(f"| {label} | {n} | n/a (n<3 or constant x) | | |")
        else:
            ci = f"[{fmt(lo, 2, signed=True)}, {fmt(hi, 2, signed=True)}]" \
                if lo is not None else "n/a"
            md.append(f"| {label} | {n} | {fmt(slope, 2, signed=True)} | {ci} | "
                      f"{fmt(p, 4)} |")
        key = "raw" if "raw" in label else "z"
        js.update({f"slope_{key}": finite_or_none(slope),
                   f"slope_{key}_ci": [finite_or_none(lo), finite_or_none(hi)],
                   f"slope_{key}_p": finite_or_none(p)})
    md.append("")

    dec_groups: dict[str, list[float]] = defaultdict(list)
    for r in joined:
        dec_groups[r["decision2"]].append(r["llm_delta"])
    acc, rej = dec_groups.get("accepted", []), dec_groups.get("rejected", [])
    md.append("Does the LLM delta separate eventual round-2 re-acceptance?")
    md.append("")
    md.append(f"- mean llm_delta: re-accepted {fmt(np.mean(acc) if acc else None, 2, signed=True)} "
              f"(n={len(acc)}) vs re-rejected "
              f"{fmt(np.mean(rej) if rej else None, 2, signed=True)} (n={len(rej)})")
    other = {k: len(v) for k, v in dec_groups.items()
             if k not in ("accepted", "rejected")}
    if other:
        md.append("- excluded decisions: " + ", ".join(f"{k}: {n}" for k, n in
                                                       sorted(other.items())))
    labels = [1] * len(acc) + [0] * len(rej)
    auc = safe_auc(labels, acc + rej)
    if auc is None:
        md.append(f"- ROC AUC(llm_delta -> re-accepted): n/a "
                  f"(one class only: {len(acc)} accepted / {len(rej)} rejected)")
    else:
        note = (" - CAUTION: minority class has < 5 pairs"
                if min(len(acc), len(rej)) < 5 else "")
        md.append(f"- ROC AUC(llm_delta -> re-accepted): {fmt(auc, 2)} "
                  f"(n={len(acc)}+{len(rej)}){note}")
    md.append("")
    js.update({"auc_decision2": finite_or_none(auc),
               "n_decision2": {"accepted": len(acc), "rejected": len(rej)}})

    # ----------------------------------------------------------- 4. baseline
    md.append("### 3. Baseline: overall LLM delta vs the placebo noise floor")
    md.append("")
    all_deltas = [r["llm_delta"] for r in joined]
    m, lo, hi, p, n = t_ci(all_deltas)
    ci = f"[{fmt(lo, 2, signed=True)}, {fmt(hi, 2, signed=True)}]" \
        if lo is not None else "n/a (n<2 or sd==0)"
    md.append(f"- mean llm_delta over all {n} scored pairs: {fmt(m, 2, signed=True)} "
              f"(95% CI {ci}, one-sample t vs 0: p={fmt(p, 4)}) - resubmissions are "
              f"revised to address reviews, so this should be positive")
    floor = floors.get(model)
    floor_note = ""
    if floor is None and len(floors) == 1:
        only = next(iter(floors))
        floor = floors[only]
        floor_note = f" (model '{model}' absent; using the only floor, '{only}')"
    if floor is None:
        md.append("- TRUE-PLACEBO noise floor: n/a (not found in --noise-summary)")
    else:
        ratio = abs(m) / floor["mean_abs"] if m is not None and floor["mean_abs"] else None
        md.append(f"- TRUE-PLACEBO noise floor{floor_note}: mean |delta| "
                  f"{fmt(floor['mean_abs'], 2)} (mean delta "
                  f"{fmt(floor['mean_delta'], 2, signed=True)}, n={floor['n']} "
                  f"cosmetic pairs) -> |mean llm_delta| / floor = {fmt(ratio, 2)}"
                  + (" (above the floor)" if ratio is not None and ratio > 1
                     else " (NOT above the floor)" if ratio is not None else ""))
        js["noise_floor"] = floor
        js["floor_ratio"] = finite_or_none(ratio)
    md.append("")
    js.update({"mean_llm_delta": finite_or_none(m),
               "llm_delta_ci": [finite_or_none(lo), finite_or_none(hi)],
               "llm_delta_t_p": finite_or_none(p)})

    # ------------------------------------------------------------ 5. caveats
    n_weak = sum(r["weak_identity"] for r in joined)
    n_same = sum(r["same_scale"] for r in joined)
    md.append("### 4. Caveats (read before quoting)")
    md.append("")
    reasons = pair_meta.get("exclude_reasons") or {}
    reason_str = ", ".join(f"{k}: {v}" for k, v in
                           sorted(reasons.items(), key=lambda kv: -kv[1])) or "none"
    md.append(f"1. n: {len(joined)} scored pairs ({pair_meta.get('n_rows', len(pairs))} "
              f"rows in the pairs file, {pair_meta.get('n_excluded', 0)} excluded by "
              f"the builder [{reason_str}]; {n_pair_unscored} includable pair(s) not "
              f"scored by this model, {n_missing_round} missing a scored round). "
              f"Small-n statistics above carry their own guards.")
    md.append(f"2. Match-coverage bias: a pair survives only when its paper's arXiv "
              f"record is in the bulk harvest (datestamp >= 2024-06-01"
              + (f"; no_bulk_match alone removed {reasons['no_bulk_match']} pair(s)"
                 if reasons.get("no_bulk_match") else "")
              + "), i.e. papers touched on arXiv recently - older or never-revised "
                "submissions cannot enter, so the sample over-represents actively "
                "revised work.")
    md.append(f"3. weak_identity: {n_weak}/{len(joined)} scored pairs were matched "
              f"across rounds on title similarity without a strong author "
              f"confirmation - some may not be the same underlying paper.")
    md.append(f"4. same_scale: {n_same}/{len(joined)} scored pairs have both rounds "
              f"on the same realized ICLR score scale; human_delta_raw is only "
              f"interpretable there (2022-2025 are mutually compatible; other years "
              f"are not), hence the z-scored column for the rest.")
    md.append(f"5. Human deltas are themselves noisy (reviewer turnover, different "
              f"committees across years for resubmissions), which attenuates every "
              f"correlation above toward 0.")
    md.append("")
    js.update({"n_weak_identity": n_weak, "n_same_scale": n_same})

    hl.append(f"[{model}] E5 n={len(joined)}: pearson raw {fmt(js.get('pearson_raw'), 2)} "
              f"(n={js.get('n_raw')}) / z {fmt(js.get('pearson_z'), 2)} "
              f"(n={js.get('n_z')}) | sign agreement {pct(agree)}")
    hl.append(f"[{model}] slope raw {fmt(js.get('slope_raw'), 2)} "
              f"(p={fmt(js.get('slope_raw_p'), 4)}) | AUC(re-accept) {fmt(auc, 2)} | "
              f"mean llm_delta {fmt(m, 2, signed=True)} (p={fmt(p, 4)})")
    return "\n".join(md), hl, js


# ------------------------------------------------------------------- driver
def main() -> None:
    ap = argparse.ArgumentParser(
        description="E5 analysis: LLM score deltas vs real human score deltas.")
    ap.add_argument("--log", default="data/processed/e5_reviews_log.jsonl")
    ap.add_argument("--pairs", default="data/processed/e5_pairs.jsonl")
    ap.add_argument("--manifest", default="data/processed/e5_manifest.jsonl")
    ap.add_argument("--noise-summary", default="results/acl_scores_summary.md",
                    help="E0-style summary whose TRUE-PLACEBO lines give the noise floor")
    ap.add_argument("--model", default=None,
                    help="only this model (default: every model in the log, separately)")
    ap.add_argument("--out", default="results/e5_summary.md",
                    help="markdown report, OVERWRITTEN each run")
    ap.add_argument("--json-out", default=None,
                    help="optional machine-readable per-model stats JSON, OVERWRITTEN")
    args = ap.parse_args()

    recs_by_model, n_total, n_used = load_reviews(args.log, args.model)
    pairs, pair_meta = load_pairs(args.pairs)
    rounds = load_round_versions(args.manifest)
    floors = parse_noise_floors(args.noise_summary)

    header = [
        "# E5: LLM score deltas vs REAL human score deltas", "",
        f"- log: `{args.log}` - {n_total} rows, {n_used} parsed_ok rating rows used"
        + (f" (model filter: {args.model})" if args.model else ""),
        f"- pairs: `{args.pairs}` - {pair_meta['n_rows']} rows, {len(pairs)} "
        f"includable, {pair_meta['n_excluded']} excluded by the builder"
        + (f" ({pair_meta['n_bad']} duplicate/keyless row(s))"
           if pair_meta["n_bad"] else ""),
        f"- manifest: `{args.manifest}` - {len(rounds)} papers | noise floor: "
        f"`{args.noise_summary}`"
        + ("" if floors else " (no TRUE-PLACEBO lines found)"),
        "",
    ]

    sections, headlines, stats = [], [], []
    for model in sorted(recs_by_model):
        sec, hl, js = analyze_model(model, recs_by_model[model], pairs, rounds,
                                    floors, pair_meta)
        sections.append(sec)
        headlines.extend(hl)
        stats.append(js)
    if not recs_by_model:
        sections.append("_No parsed_ok rating rows found in the log - nothing to "
                        "analyze. (Run the E5 scoring stage first.)_")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(header + sections) + "\n")

    if args.json_out:
        jp = Path(args.json_out)
        jp.parent.mkdir(parents=True, exist_ok=True)
        jp.write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n")

    print("\n=== analyze_e5 summary ===")
    print(f"log rows: {n_total} | parsed_ok used: {n_used} | pairs usable: "
          f"{len(pairs)} | models: {', '.join(sorted(recs_by_model)) or 'none'}")
    for line in headlines:
        print(line)
    print(f"report -> {out}")
    if args.json_out:
        print(f"json stats -> {args.json_out}")


if __name__ == "__main__":
    main()
