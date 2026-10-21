"""E0 statistics from the review log: does the LLM score track versions, and how noisy is it?

Reads the review log written by the scoring stage (one JSON row per (model, paper,
version, rep) call), keeps only parsed_ok rows, aggregates rating mean/SD per
(model, paper, version) across reps, and reports per model:

1. VERSION TRACKING - per-paper Kendall tau-b (scipy.stats.kendalltau) between version
   index and per-version mean rating (papers with >= 2 scored versions): mean tau, shares
   of tau>0 / tau==0(ties) / tau<0, a one-line histogram; plus adjacent-pair direction
   accuracy P(mean v_{i+1} > v_i) / P(=) / P(<), split by churn_class joined from the
   diffs file on (arxiv_id, v_from, v_to) - cosmetic pairs SHOULD be '='.
2. RELIABILITY - test-retest SD per (paper, version) across reps (mean/median); ICC(2,1);
   and signal-to-noise: between-version SD of per-version means within a paper vs the
   test-retest SD.
3. HUMAN ANCHOR - for each paper, the latest version whose stage is pre_submission or
   during_review (fall back to v1): Pearson+Spearman of LLM mean rating vs human
   mean_score (overall and within accepted / rejected), and accept/reject ROC AUC of the
   LLM rating, of the LLM accept-vote share, and of the human mean_score as reference.
4. DISTRIBUTIONS - LLM rating histogram vs human per-review score histogram (manifest
   'scores' flattened); mean LLM rating for accepted vs rejected papers.

ICC(2,1) (two-way random effects, absolute agreement, single rating; Shrout & Fleiss
1979) is computed with numpy from the standard two-way ANOVA decomposition. For the
n x k matrix X (rows = targets = paper-versions, columns = raters = rep index):

    SST = sum((X - grand_mean)^2)
    SSR = k * sum_rows((row_mean - grand_mean)^2)     MSR = SSR / (n - 1)
    SSC = n * sum_cols((col_mean - grand_mean)^2)     MSC = SSC / (k - 1)
    SSE = SST - SSR - SSC                             MSE = SSE / ((n - 1) * (k - 1))
    ICC(2,1) = (MSR - MSE) / (MSR + (k - 1) * MSE + k * (MSC - MSE) / n)

The matrix must be complete, so k is the modal rep count among targets with >= 2 reps;
targets with more reps are truncated to their first k (by rep index) and targets with
fewer are dropped (the report says how many).

Degenerate-input safety: with < 2 reps the reliability section prints 'n/a (reps<2)';
all-tied papers give tau = nan and are excluded from the mean tau but counted in the
'all tied' share; a missing diffs file skips the churn split with a note.

Report formatting: rating-scale numbers use one decimal; unitless statistics
(tau / r / AUC / ICC) use two decimals so they stay distinguishable.

Both outputs (--out markdown report, --per-paper-out JSONL) are OVERWRITTEN each run.

Run from repo root: python3 src/analysis/analyze_scores.py
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy import stats as sps
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

sys.path.insert(0, "src/collect")
from util import read_jsonl  # noqa: E402

PRE_DECISION_STAGES = {"pre_submission", "during_review"}
CHURN_ORDER = ["cosmetic", "minor", "major", "unknown"]
TIE_EPS = 1e-9


# ---------------------------------------------------------------- formatting
def fmt(x, nd: int = 1) -> str:
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "n/a"
    return f"{x:.{nd}f}"


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


# ------------------------------------------------------------------- loading
def _is_num(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def vote_to_accept(vote):
    """Map a recommendation to 1 (accept) / 0 (reject) / None (absent or unparseable)."""
    if vote is None:
        return None
    if isinstance(vote, bool):
        return 1 if vote else 0
    s = str(vote).strip().lower()
    if "accept" in s:
        return 1
    if "reject" in s:
        return 0
    return None


def load_reviews(path: str, model_filter: str | None):
    """Parsed_ok rows only -> {model: [ {arxiv_id, v, rep, rating, accept} ]}.

    Tolerant to small schema variations: version under 'v' or 'version', rep under
    'rep'/'rep_idx'/'replicate', rating/vote either top-level or inside a 'review' dict.
    The vote may be named 'vote', 'accept' (run_llm_review.py writes a top-level boolean
    'accept'), or 'recommendation'; the first non-None among them wins.
    """
    rows = read_jsonl(path)
    recs: dict[str, list[dict]] = defaultdict(list)
    n_used = 0
    for row in rows:
        if not row.get("parsed_ok"):
            continue
        review = row.get("review") if isinstance(row.get("review"), dict) else {}
        aid = row.get("arxiv_id")
        v = row.get("v", row.get("version"))
        rating = row.get("rating", review.get("rating"))
        if aid is None or v is None or not _is_num(rating):
            continue
        model = str(row.get("model") or "unknown")
        if model_filter and model != model_filter:
            continue
        rep = row.get("rep", row.get("rep_idx", row.get("replicate", 0)))
        vote = next((x for x in (row.get("vote"), row.get("accept"),
                                 row.get("recommendation"), review.get("vote"),
                                 review.get("accept"), review.get("recommendation"))
                     if x is not None), None)
        recs[model].append({
            "arxiv_id": str(aid), "v": int(v), "rep": rep,
            "rating": float(rating), "accept": vote_to_accept(vote),
        })
        n_used += 1
    return recs, len(rows), n_used


def load_diffs(path: str):
    """(arxiv_id, v_from, v_to) -> churn_class, or None when the file is missing."""
    if not Path(path).exists():
        return None
    return {(str(r["arxiv_id"]), int(r["v_from"]), int(r["v_to"])): r.get("churn_class", "unknown")
            for r in read_jsonl(path)}


def anchor_version(mrow: dict) -> int:
    """Latest manifest version with stage pre_submission/during_review, else v1."""
    cands = [vv["v"] for vv in mrow.get("versions", []) if vv.get("stage") in PRE_DECISION_STAGES]
    return max(cands) if cands else 1


# ---------------------------------------------------------------- statistics
def icc21(mat: np.ndarray) -> float:
    """ICC(2,1) via the two-way ANOVA formula documented in the module docstring."""
    n, k = mat.shape
    grand = mat.mean()
    row_means = mat.mean(axis=1)
    col_means = mat.mean(axis=0)
    sst = float(((mat - grand) ** 2).sum())
    ssr = float(k * ((row_means - grand) ** 2).sum())
    ssc = float(n * ((col_means - grand) ** 2).sum())
    sse = sst - ssr - ssc
    msr = ssr / (n - 1)
    msc = ssc / (k - 1)
    mse = sse / ((n - 1) * (k - 1))
    denom = msr + (k - 1) * mse + k * (msc - mse) / n
    return (msr - mse) / denom if abs(denom) > 1e-12 else float("nan")


def safe_auc(labels: list[int], scores: list[float]):
    if len(labels) < 2 or len(set(labels)) < 2:
        return None
    return float(roc_auc_score(labels, scores))


def corr_pair(xs: list[float], ys: list[float]):
    """(pearson, spearman) or (None, None) when degenerate (<3 points or constant)."""
    if len(xs) < 3 or len(set(xs)) < 2 or len(set(ys)) < 2:
        return None, None
    return float(sps.pearsonr(xs, ys).statistic), float(sps.spearmanr(xs, ys).statistic)


def tau_histogram_line(taus: list[float], n_tied: int) -> str:
    bins = [(-1.0, -0.5), (-0.5, 0.0), (0.0, 0.5), (0.5, 1.0)]
    counts = [0] * len(bins)
    for t in taus:
        for i, (lo, hi) in enumerate(bins):
            if (lo <= t < hi) or (i == len(bins) - 1 and t == hi):
                counts[i] += 1
                break
    parts = [f"[{lo:+.1f},{hi:+.1f}{']' if i == len(bins) - 1 else ')'}:{c}"
             for i, ((lo, hi), c) in enumerate(zip(bins, counts))]
    return "tau hist: " + " ".join(parts) + f" all-tied(nan):{n_tied}"


# ------------------------------------------------------------ per-model work
def analyze_model(model: str, recs: list[dict], manifest: list[dict], diffs):
    """Returns (markdown_section, headline_lines, per_paper_rows)."""
    man = {str(m["arxiv_id"]): m for m in manifest}

    # per (paper, version) aggregates across reps
    by_pv: dict[str, dict[int, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for r in recs:
        by_pv[r["arxiv_id"]][r["v"]].append(r)
    vstats: dict[tuple[str, int], dict] = {}
    for aid, vd in by_pv.items():
        for v, lst in vd.items():
            ratings = [r["rating"] for r in lst]
            accepts = [r["accept"] for r in lst if r["accept"] is not None]
            # numeric-aware rep sort: str-sorting would put rep 10 before rep 2, so the
            # ICC 'first k reps' truncation could keep different rep subsets per target
            reps_sorted = sorted(lst, key=lambda r: ((0, float(r["rep"]), "")
                                                     if _is_num(r["rep"])
                                                     else (1, 0.0, str(r["rep"]))))
            vstats[(aid, v)] = {
                "mean": float(np.mean(ratings)),
                "sd": float(np.std(ratings, ddof=1)) if len(ratings) >= 2 else None,
                "n": len(ratings),
                "accept_share": (sum(accepts) / len(accepts)) if accepts else None,
                "ratings_by_rep": [r["rating"] for r in reps_sorted],
            }

    md: list[str] = [f"## Model: {model}", ""]
    md.append(f"- parsed_ok reviews: {len(recs)} | papers scored: {len(by_pv)} | "
              f"paper-versions scored: {len(vstats)}")
    md.append("")

    # ---------------------------------------------------- 1. version tracking
    taus_finite: list[float] = []
    n_tied = 0
    paper_tau: dict[str, float | None] = {}
    pairs: list[tuple[str, int, int, str, str | None]] = []  # aid, v_from, v_to, dir, churn
    for aid in tqdm(sorted(by_pv), desc=f"{model}: papers", leave=False):
        vs = sorted(by_pv[aid])
        means = [vstats[(aid, v)]["mean"] for v in vs]
        if len(vs) >= 2:
            tau = float(sps.kendalltau(vs, means).statistic)
            paper_tau[aid] = tau
            if math.isfinite(tau):
                taus_finite.append(tau)
            else:
                n_tied += 1  # constant ratings across versions -> tau = nan
        else:
            paper_tau[aid] = None
        for a, b in zip(vs, vs[1:]):
            m1, m2 = vstats[(aid, a)]["mean"], vstats[(aid, b)]["mean"]
            d = "=" if abs(m2 - m1) < TIE_EPS else (">" if m2 > m1 else "<")
            cls = diffs.get((aid, a, b), "unknown") if diffs is not None else None
            pairs.append((aid, a, b, d, cls))

    n_multi = len(taus_finite) + n_tied
    mean_tau = float(np.mean(taus_finite)) if taus_finite else None
    sh_pos = sum(t > 0 for t in taus_finite) / n_multi if n_multi else None
    sh_zero = (sum(t == 0 for t in taus_finite) + n_tied) / n_multi if n_multi else None
    sh_neg = sum(t < 0 for t in taus_finite) / n_multi if n_multi else None

    md.append("### 1. Version tracking")
    md.append("")
    md.append(f"- papers with >= 2 scored versions: {n_multi}")
    md.append(f"- mean Kendall tau-b (all-tied excluded): {fmt(mean_tau, 2)} "
              f"(n={len(taus_finite)})")
    md.append(f"- tau>0: {pct(sh_pos)} | tau==0 (ties, incl. all-tied): {pct(sh_zero)} | "
              f"tau<0: {pct(sh_neg)} | all-tied papers: {n_tied}")
    md.append(f"- {tau_histogram_line(taus_finite, n_tied)}")
    md.append("")
    md.append("Adjacent-pair direction (mean rating of v_{i+1} vs v_i):")
    md.append("")
    md.append("| churn_class | n | P(>) | P(=) | P(<) |")
    md.append("|---|---|---|---|---|")

    def dir_row(label: str, sub: list):
        n = len(sub)
        if n == 0:
            return f"| {label} | 0 | n/a | n/a | n/a |"
        c = Counter(p[3] for p in sub)
        return (f"| {label} | {n} | {pct(c['>'] / n)} | {pct(c['='] / n)} | "
                f"{pct(c['<'] / n)} |")

    md.append(dir_row("ALL", pairs))
    cosmetic_eq = None
    if diffs is None:
        md.append("")
        md.append("_Note: diffs file not found - churn split skipped._")
    else:
        for cls in CHURN_ORDER:
            sub = [p for p in pairs if p[4] == cls]
            if sub:
                md.append(dir_row(cls, sub))
        cos = [p for p in pairs if p[4] == "cosmetic"]
        if cos:
            cosmetic_eq = sum(p[3] == "=" for p in cos) / len(cos)
            md.append("")
            md.append(f"Cosmetic pairs (should be '='): {sum(p[3] == '=' for p in cos)}"
                      f"/{len(cos)} tied ({pct(cosmetic_eq)}).")
    md.append("")

    # -------------------------------------------------------- 2. reliability
    md.append("### 2. Reliability")
    md.append("")
    retest_targets = [key for key, s in vstats.items() if s["n"] >= 2]
    icc = None
    mean_sd = med_sd = None
    mean_between = None
    if not retest_targets:
        md.append("n/a (reps<2)")
    else:
        sds = [vstats[t]["sd"] for t in retest_targets]
        mean_sd, med_sd = float(np.mean(sds)), float(np.median(sds))
        md.append(f"- test-retest SD across reps (per paper-version, n={len(sds)}): "
                  f"mean {fmt(mean_sd)} | median {fmt(med_sd)}")
        # ICC(2,1): complete n x k matrix (see module docstring)
        k = Counter(vstats[t]["n"] for t in retest_targets).most_common(1)[0][0]
        mat_rows, dropped = [], 0
        for t in retest_targets:
            reps = vstats[t]["ratings_by_rep"]
            if len(reps) >= k:
                mat_rows.append(reps[:k])
            else:
                dropped += 1
        if len(mat_rows) >= 2 and k >= 2:
            icc = icc21(np.array(mat_rows, dtype=float))
            md.append(f"- ICC(2,1) (targets=paper-versions n={len(mat_rows)}, raters=rep "
                      f"index k={k}, {dropped} incomplete target(s) dropped): {fmt(icc, 2)}")
        else:
            md.append(f"- ICC(2,1): n/a (needs >= 2 complete targets; k={k})")
        between = []
        for aid, vd in by_pv.items():
            if len(vd) >= 2:
                between.append(float(np.std([vstats[(aid, v)]["mean"] for v in vd], ddof=1)))
        if between:
            mean_between = float(np.mean(between))
            ratio = mean_between / mean_sd if mean_sd and mean_sd > 0 else None
            md.append(f"- signal vs noise: between-version SD of per-version means "
                      f"(within paper, n={len(between)}) mean {fmt(mean_between)} vs "
                      f"test-retest SD mean {fmt(mean_sd)} -> ratio {fmt(ratio, 2)}")
        else:
            md.append("- signal vs noise: n/a (no paper with >= 2 scored versions)")
    md.append("")

    # ------------------------------------------------------- 3. human anchor
    md.append("### 3. Human anchor")
    md.append("")
    anchor = []  # per paper: dict(aid, llm, vote_share, human, decision)
    n_anchor_unscored = 0
    for aid in sorted(by_pv):
        mrow = man.get(aid)
        if mrow is None:
            continue
        av = anchor_version(mrow)
        if (aid, av) not in vstats:
            n_anchor_unscored += 1
            continue
        anchor.append({
            "aid": aid, "v": av,
            "llm": vstats[(aid, av)]["mean"],
            "vote_share": vstats[(aid, av)]["accept_share"],
            "human": mrow.get("mean_score"),
            "decision": mrow.get("decision"),
        })
    md.append(f"- anchor version = latest pre_submission/during_review (fallback v1); "
              f"papers with scored anchor: {len(anchor)} "
              f"({n_anchor_unscored} skipped: anchor version not scored)")

    corr_sets = {"overall": anchor,
                 "accepted": [a for a in anchor if a["decision"] == "accepted"],
                 "rejected": [a for a in anchor if a["decision"] == "rejected"]}
    md.append("")
    md.append("| subset | n | Pearson r | Spearman rho |")
    md.append("|---|---|---|---|")
    pearson_overall = None
    for name, sub in corr_sets.items():
        sub = [a for a in sub if a["human"] is not None]
        pr, sr = corr_pair([a["llm"] for a in sub], [a["human"] for a in sub])
        if name == "overall":
            pearson_overall = pr
        md.append(f"| {name} | {len(sub)} | {fmt(pr, 2)} | {fmt(sr, 2)} |")

    dec = [a for a in anchor if a["decision"] in ("accepted", "rejected")]
    labels = [1 if a["decision"] == "accepted" else 0 for a in dec]
    auc_llm = safe_auc(labels, [a["llm"] for a in dec])
    voted = [a for a in dec if a["vote_share"] is not None]
    auc_vote = safe_auc([1 if a["decision"] == "accepted" else 0 for a in voted],
                        [a["vote_share"] for a in voted])
    human = [a for a in dec if a["human"] is not None]
    auc_human = safe_auc([1 if a["decision"] == "accepted" else 0 for a in human],
                         [a["human"] for a in human])
    md.append("")
    md.append("| accept/reject discrimination | n | ROC AUC |")
    md.append("|---|---|---|")
    md.append(f"| LLM mean rating (anchor) | {len(dec)} | {fmt(auc_llm, 2)} |")
    md.append(f"| LLM accept-vote share (anchor) | {len(voted)} | {fmt(auc_vote, 2)} |")
    md.append(f"| human mean_score (reference) | {len(human)} | {fmt(auc_human, 2)} |")
    md.append("")

    # ------------------------------------------------------ 4. distributions
    md.append("### 4. Distributions")
    md.append("")
    llm_ratings = [r["rating"] for r in recs]
    human_scores = [s for m in manifest for s in (m.get("scores") or [])
                    if isinstance(s, (int, float))]
    llm_hist = Counter(min(10, max(1, round(x))) for x in llm_ratings)
    hum_hist = Counter(min(10, max(1, round(x))) for x in human_scores)
    md.append("| score | LLM n | LLM % | human n | human % |")
    md.append("|---|---|---|---|---|")
    for b in range(1, 11):
        ln, hn = llm_hist.get(b, 0), hum_hist.get(b, 0)
        md.append(f"| {b} | {ln} | {pct(ln / len(llm_ratings)) if llm_ratings else 'n/a'} "
                  f"| {hn} | {pct(hn / len(human_scores)) if human_scores else 'n/a'} |")
    md.append("")
    md.append(f"- LLM ratings: mean {fmt(np.mean(llm_ratings) if llm_ratings else None)} "
              f"SD {fmt(np.std(llm_ratings, ddof=1) if len(llm_ratings) >= 2 else None)} | "
              f"human per-review scores: mean "
              f"{fmt(np.mean(human_scores) if human_scores else None)} "
              f"SD {fmt(np.std(human_scores, ddof=1) if len(human_scores) >= 2 else None)} "
              f"(scales differ by ICLR year)")
    # mean LLM rating (all parsed ratings) grouped by paper decision
    by_dec: dict[str, list[float]] = defaultdict(list)
    for r in recs:
        d = man.get(r["arxiv_id"], {}).get("decision")
        if d in ("accepted", "rejected"):
            by_dec[d].append(r["rating"])
    md.append(f"- mean LLM rating (all scored versions/reps): accepted "
              f"{fmt(np.mean(by_dec['accepted']) if by_dec['accepted'] else None)} "
              f"(n={len(by_dec['accepted'])}) vs rejected "
              f"{fmt(np.mean(by_dec['rejected']) if by_dec['rejected'] else None)} "
              f"(n={len(by_dec['rejected'])})")
    md.append("")

    # ------------------------------------------------------ per-paper output
    per_paper = []
    for aid in sorted(by_pv):
        vd = by_pv[aid]
        sds = [vstats[(aid, v)]["sd"] for v in vd if vstats[(aid, v)]["sd"] is not None]
        per_paper.append({
            "model": model,
            "arxiv_id": aid,
            "decision": man.get(aid, {}).get("decision"),
            "n_versions_scored": len(vd),
            "ratings_by_version": {str(v): round(vstats[(aid, v)]["mean"], 3)
                                   for v in sorted(vd)},
            "tau": finite_or_none(paper_tau.get(aid)),
            "mean_sd": round(float(np.mean(sds)), 3) if sds else None,
            "human_mean": man.get(aid, {}).get("mean_score"),
        })

    headline = [
        f"[{model}] mean tau {fmt(mean_tau, 2)} (n={len(taus_finite)}, all-tied {n_tied}) | "
        f"P(up) {pct(sh_pos)}" + (f" | cosmetic '=' {pct(cosmetic_eq)}"
                                  if cosmetic_eq is not None else ""),
        f"[{model}] retest SD mean {fmt(mean_sd)} | ICC(2,1) {fmt(icc, 2)} | "
        f"between-version SD mean {fmt(mean_between)}",
        f"[{model}] human anchor: pearson {fmt(pearson_overall, 2)} | AUC rating "
        f"{fmt(auc_llm, 2)} / votes {fmt(auc_vote, 2)} / human {fmt(auc_human, 2)}",
    ]
    return "\n".join(md), headline, per_paper


# ------------------------------------------------------------------- driver
def main() -> None:
    ap = argparse.ArgumentParser(description="E0 analysis of LLM review scores.")
    ap.add_argument("--log", default="data/processed/reviews_log.jsonl")
    ap.add_argument("--manifest", default="data/processed/pilot_manifest.jsonl")
    ap.add_argument("--diffs", default="data/processed/version_diffs.jsonl")
    ap.add_argument("--model", default=None,
                    help="only this model (default: every model in the log, separately)")
    ap.add_argument("--out", default="results/e0_scores_summary.md",
                    help="markdown report, OVERWRITTEN each run")
    ap.add_argument("--per-paper-out", default="results/e0_per_paper.jsonl",
                    help="per (model, paper) rows, OVERWRITTEN each run")
    args = ap.parse_args()

    recs_by_model, n_total, n_used = load_reviews(args.log, args.model)
    manifest = read_jsonl(args.manifest)
    diffs = load_diffs(args.diffs)

    header = ["# E0: LLM review-score analysis", "",
              f"- log: `{args.log}` - {n_total} rows, {n_used} parsed_ok rating rows used"
              + (f" (model filter: {args.model})" if args.model else ""),
              f"- manifest: `{args.manifest}` ({len(manifest)} papers) | diffs: `{args.diffs}`"
              + ("" if diffs is not None else " (MISSING - churn split skipped)"),
              ""]

    sections, headlines, per_paper_rows = [], [], []
    for model in sorted(recs_by_model):
        sec, hl, pp = analyze_model(model, recs_by_model[model], manifest, diffs)
        sections.append(sec)
        headlines.extend(hl)
        per_paper_rows.extend(pp)
    if not recs_by_model:
        sections.append("_No parsed_ok rating rows found in the log - nothing to analyze._")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(header + sections) + "\n")

    ppo = Path(args.per_paper_out)
    ppo.parent.mkdir(parents=True, exist_ok=True)
    with ppo.open("w") as f:
        for row in per_paper_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("\n=== analyze_scores summary ===")
    print(f"log rows: {n_total} | parsed_ok used: {n_used} | models: "
          f"{', '.join(sorted(recs_by_model)) or 'none'}")
    for line in headlines:
        print(line)
    print(f"report -> {out}")
    print(f"per-paper rows -> {ppo} ({len(per_paper_rows)} rows)")


if __name__ == "__main__":
    main()
