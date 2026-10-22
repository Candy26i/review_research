"""Contamination split: is the version-tracking effect driven by papers the model remembers?

Joins the review log to the recognition log on arxiv_id and splits each model's papers
by a recognition flag (--split recognized | authors_any_correct; the recognition log has
one probe row per paper, taken at its anchor version). Papers absent from the recognition
log, or whose flag is not a boolean, fall into an 'unprobed' bucket that is reported but
excluded from every inferential test.

Per model label in the review log:

1. GROUP SIZES + DESCRIPTIVES - per group (flag=True / flag=False / unprobed): papers,
   papers with >= 2 scored versions, rep-level rows, mean rating level, mean finite
   per-paper Kendall tau-b (version order vs per-version mean rating; all-tied papers
   give tau = nan, are excluded from the mean, and are counted separately), and
   adjacent-pair direction shares P(>)/P(=)/P(<) of per-version mean ratings.
2. THE TEST - interaction mixed model on rep-level rows of papers with >= 2 scored
   versions (probed papers only): statsmodels MixedLM 'rating ~ pos * recog' with a
   random intercept per paper (groups=paper), where pos is the 0-based index of the
   version within the paper's sorted scored versions and recog is 1 for flag=True.
   The 'pos:recog' coefficient (Wald 95% CI, p) is the contamination signal: positive
   and significant = the version effect is LARGER on remembered papers. Because the
   interaction is low-powered at the pilot's split (~12 vs ~38 papers), the simple
   version effect ('rating ~ pos') is ALSO fit within each group separately.
3. HUMAN ANCHOR - accept/reject ROC AUC of the anchor-version mean LLM rating within
   each group (anchor = latest pre_submission/during_review version, falling back to v1,
   exactly as in analyze_scores.anchor_version); 'n/a' when a group has < 2 decided
   papers or only one decision class.
4. VERDICT - three mechanical sentences per model: 'effect survives on <flag=False
   label> papers' iff that group's version coefficient is positive with p < 0.05;
   'contamination-amplified' iff the interaction coefficient is positive with p < 0.05;
   otherwise 'underpowered/indeterminate', naming the blocker.

Degenerate-input safety: a model with no rows in one group -> that group's single-group
model and the interaction model are 'n/a' with the reason; all-tied papers are handled
as above; reps = 1 is fine (every test runs on rep-level rows). Non-converging or
singular mixed models are caught and reported as 'n/a (<exception>)'.

The --out markdown report is OVERWRITTEN each run. No network access, no LLM calls.

Run from repo root: python3 src/analysis/analyze_contamination_split.py
"""
from __future__ import annotations

import argparse
import math
import sys
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats as sps

_ROOT = Path(__file__).resolve().parents[2]
for _p in (_ROOT / "src" / "collect", _ROOT / "src" / "analysis"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
from util import read_jsonl  # noqa: E402
from analyze_scores import anchor_version, fmt, load_reviews, pct, safe_auc  # noqa: E402

TIE_EPS = 1e-9
SPLIT_LABELS = {
    "recognized": ("recognized", "unrecognized"),
    "authors_any_correct": ("authors-recalled", "authors-not-recalled"),
}
UNPROBED = "unprobed"


def fmt_p(p) -> str:
    if p is None or (isinstance(p, float) and not math.isfinite(p)):
        return "n/a"
    return "<0.0001" if p < 1e-4 else f"{p:.4f}"


def fmt_fit(fit: dict | None, reason: str | None) -> str:
    if fit is None:
        return f"n/a ({reason})"
    return (f"{fit['coef']:+.2f}/step (95% CI [{fit['lo']:+.2f}, {fit['hi']:+.2f}], "
            f"p={fmt_p(fit['p'])}; n={fit['n']} rows, {fit['papers']} papers)")


# ---------------------------------------------------------------------- loading
def load_recognition(path: str, split: str):
    """arxiv_id -> bool flag (first row per paper wins). Returns (flags, n_rows, n_dup)."""
    flags: dict[str, bool] = {}
    n_rows = n_dup = 0
    for row in read_jsonl(path):
        aid = row.get("arxiv_id")
        if aid is None:
            continue
        aid = str(aid)
        n_rows += 1
        if aid in flags:
            n_dup += 1
            continue
        val = row.get(split)
        if isinstance(val, bool):
            flags[aid] = val
    return flags, n_rows, n_dup


# ------------------------------------------------------------------ statistics
def fit_pos_model(df: pd.DataFrame, with_interaction: bool):
    """MixedLM 'rating ~ pos [* recog]' + (1|paper). Returns (fit_dict|None, reason|None).

    fit_dict reports the coefficient of interest: 'pos:recog' for the interaction
    model, 'pos' for a single-group model.
    """
    if df.empty:
        return None, "no rows"
    n_papers = df["paper"].nunique()
    if n_papers < 3:
        return None, f"only {n_papers} paper(s) (<3)"
    if df["pos"].nunique() < 2:
        return None, "only one version position"
    if with_interaction and df["recog"].nunique() < 2:
        return None, "one group empty (recog constant)"
    formula = "rating ~ pos * recog" if with_interaction else "rating ~ pos"
    name = "pos:recog" if with_interaction else "pos"
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = smf.mixedlm(formula, df, groups=df["paper"]).fit(reml=True)
        coef = float(fit.params[name])
        se = float(fit.bse[name])
        p = float(fit.pvalues[name])
        if not (math.isfinite(coef) and math.isfinite(se)):
            return None, "non-finite estimate (singular fit)"
        out = {"coef": coef, "lo": coef - 1.96 * se, "hi": coef + 1.96 * se, "p": p,
               "n": len(df), "papers": n_papers}
        if with_interaction:
            base = float(fit.params["pos"])
            base_se = float(fit.bse["pos"])
            out["base_pos"] = base
            out["base_pos_lo"] = base - 1.96 * base_se
            out["base_pos_hi"] = base + 1.96 * base_se
            out["base_pos_p"] = float(fit.pvalues["pos"])
        return out, None
    except Exception as exc:  # non-convergence, singular matrices, ...
        return None, f"{type(exc).__name__}: {exc}"


# -------------------------------------------------------------- per-model work
def analyze_model(model: str, recs: list[dict], flags: dict[str, bool],
                  man: dict[str, dict], split: str):
    """Returns (markdown_section, stats_dict) for one model label."""
    lab_true, lab_false = SPLIT_LABELS[split]

    by_pv: dict[str, dict[int, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for r in recs:
        by_pv[r["arxiv_id"]][r["v"]].append(r)
    vmean = {(aid, v): float(np.mean([x["rating"] for x in lst]))
             for aid, vd in by_pv.items() for v, lst in vd.items()}

    def group_of(aid: str) -> str:
        val = flags.get(aid)
        if val is True:
            return lab_true
        if val is False:
            return lab_false
        return UNPROBED

    group_aids: dict[str, list[str]] = {lab_true: [], lab_false: [], UNPROBED: []}
    for aid in sorted(by_pv):
        group_aids[group_of(aid)].append(aid)

    # ------------------------------------------- 1. sizes + descriptives per group
    def describe(aids: list[str]) -> dict:
        rep_ratings: list[float] = []
        taus: list[float] = []
        n_tied = 0
        n_multi = 0
        dirs: Counter = Counter()
        n_pairs = 0
        for aid in aids:
            vs = sorted(by_pv[aid])
            for v in vs:
                rep_ratings.extend(x["rating"] for x in by_pv[aid][v])
            if len(vs) >= 2:
                n_multi += 1
                means = [vmean[(aid, v)] for v in vs]
                tau = float(sps.kendalltau(vs, means).statistic)
                if math.isfinite(tau):
                    taus.append(tau)
                else:
                    n_tied += 1  # all versions tied -> tau = nan
                for a, b in zip(vs, vs[1:]):
                    d = vmean[(aid, b)] - vmean[(aid, a)]
                    dirs["=" if abs(d) < TIE_EPS else (">" if d > 0 else "<")] += 1
                    n_pairs += 1
        return {
            "papers": len(aids),
            "papers_multi": n_multi,
            "rows": len(rep_ratings),
            "mean_rating": float(np.mean(rep_ratings)) if rep_ratings else None,
            "mean_tau": float(np.mean(taus)) if taus else None,
            "n_tau": len(taus),
            "all_tied": n_tied,
            "n_pairs": n_pairs,
            "p_up": dirs[">"] / n_pairs if n_pairs else None,
            "p_eq": dirs["="] / n_pairs if n_pairs else None,
            "p_down": dirs["<"] / n_pairs if n_pairs else None,
        }

    groups = {name: describe(aids) for name, aids in group_aids.items()}

    md: list[str] = [f"## Model: {model}", ""]
    md.append(f"- parsed_ok reviews: {len(recs)} | papers scored: {len(by_pv)} | "
              f"split field: `{split}`")
    md.append("")
    md.append("### 1. Group sizes and descriptives")
    md.append("")
    md.append("| group | papers | papers >=2 versions | rep rows | mean rating | "
              "mean tau (n) | all-tied | pairs | P(>) | P(=) | P(<) |")
    md.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for name in (lab_true, lab_false, UNPROBED):
        g = groups[name]
        tag = " (excluded from tests)" if name == UNPROBED and g["papers"] else ""
        md.append(f"| {name}{tag} | {g['papers']} | {g['papers_multi']} | {g['rows']} | "
                  f"{fmt(g['mean_rating'], 1)} | {fmt(g['mean_tau'], 2)} ({g['n_tau']}) | "
                  f"{g['all_tied']} | {g['n_pairs']} | {pct(g['p_up'])} | "
                  f"{pct(g['p_eq'])} | {pct(g['p_down'])} |")
    md.append("")

    # --------------------------------------------------- 2. the interaction test
    reg_rows = []
    for name, recog in ((lab_true, 1), (lab_false, 0)):
        for aid in group_aids[name]:
            vs = sorted(by_pv[aid])
            if len(vs) < 2:
                continue
            for pos, v in enumerate(vs):
                for r in by_pv[aid][v]:
                    reg_rows.append({"rating": r["rating"], "pos": pos,
                                     "recog": recog, "paper": aid})
    df = pd.DataFrame(reg_rows)

    inter, inter_reason = fit_pos_model(df, with_interaction=True)
    gm: dict[str, dict | None] = {}
    gm_reason: dict[str, str | None] = {}
    for name, recog in ((lab_true, 1), (lab_false, 0)):
        sub = df[df["recog"] == recog] if not df.empty else df
        gm[name], gm_reason[name] = fit_pos_model(sub, with_interaction=False)

    md.append("### 2. Version-effect tests (rep-level MixedLM, probed papers with "
              ">= 2 scored versions)")
    md.append("")
    md.append(f"- interaction model rating ~ pos * recog + (1|paper), recog=1 for "
              f"{lab_true}: pos:recog = {fmt_fit(inter, inter_reason)}")
    if inter is not None:
        md.append(f"  - baseline slope in the {lab_false} group (the 'pos' main effect): "
                  f"{inter['base_pos']:+.2f}/step (95% CI [{inter['base_pos_lo']:+.2f}, "
                  f"{inter['base_pos_hi']:+.2f}], p={fmt_p(inter['base_pos_p'])})")
    md.append(f"- single-group model, {lab_true}: rating ~ pos + (1|paper) = "
              f"{fmt_fit(gm[lab_true], gm_reason[lab_true])}")
    md.append(f"- single-group model, {lab_false}: rating ~ pos + (1|paper) = "
              f"{fmt_fit(gm[lab_false], gm_reason[lab_false])}")
    md.append("")

    # ------------------------------------------------------------ 3. human anchor
    md.append("### 3. Human anchor: accept/reject AUC of anchor-version mean rating")
    md.append("")
    md.append("| group | n decided | accepted/rejected | ROC AUC |")
    md.append("|---|---|---|---|")
    auc_stats: dict[str, dict] = {}
    for name in (lab_true, lab_false, UNPROBED):
        labels: list[int] = []
        scores: list[float] = []
        for aid in group_aids[name]:
            mrow = man.get(aid)
            if mrow is None or mrow.get("decision") not in ("accepted", "rejected"):
                continue
            av = anchor_version(mrow)
            if (aid, av) not in vmean:
                continue
            labels.append(1 if mrow["decision"] == "accepted" else 0)
            scores.append(vmean[(aid, av)])
        auc = safe_auc(labels, scores)
        n_acc = sum(labels)
        n_rej = len(labels) - n_acc
        if auc is None:
            cell = ("n/a (<2 papers)" if len(labels) < 2 else "n/a (one class)")
        else:
            cell = fmt(auc, 2) + (" (tiny n, unstable)" if len(labels) < 6 else "")
        auc_stats[name] = {"n": len(labels), "n_acc": n_acc, "n_rej": n_rej, "auc": auc}
        md.append(f"| {name} | {len(labels)} | {n_acc}/{n_rej} | {cell} |")
    md.append("")

    # ---------------------------------------------------------------- 4. verdict
    gf, gf_reason = gm[lab_false], gm_reason[lab_false]
    survives = bool(gf is not None and gf["coef"] > 0
                    and math.isfinite(gf["p"]) and gf["p"] < 0.05)
    amplified = bool(inter is not None and inter["coef"] > 0
                     and math.isfinite(inter["p"]) and inter["p"] < 0.05)

    if gf is None:
        s1 = (f"No version-effect model could be fit in the {lab_false} group "
              f"({gf_reason}).")
    elif survives:
        s1 = (f"The version effect within the {lab_false} group is "
              f"{gf['coef']:+.2f}/step (95% CI [{gf['lo']:+.2f}, {gf['hi']:+.2f}], "
              f"p={fmt_p(gf['p'])}), independently positive and significant.")
    else:
        s1 = (f"The version effect within the {lab_false} group is "
              f"{gf['coef']:+.2f}/step (95% CI [{gf['lo']:+.2f}, {gf['hi']:+.2f}], "
              f"p={fmt_p(gf['p'])}), not significantly positive on its own.")

    if inter is None:
        s2 = f"No interaction model could be fit ({inter_reason})."
    elif amplified:
        s2 = (f"The pos:recog interaction is {inter['coef']:+.2f} (95% CI "
              f"[{inter['lo']:+.2f}, {inter['hi']:+.2f}], p={fmt_p(inter['p'])}), "
              f"significantly positive: the version effect is larger on {lab_true} "
              f"papers.")
    else:
        s2 = (f"The pos:recog interaction is {inter['coef']:+.2f} (95% CI "
              f"[{inter['lo']:+.2f}, {inter['hi']:+.2f}], p={fmt_p(inter['p'])}), "
              f"not significantly positive.")

    if survives and amplified:
        verdict = (f"effect survives on {lab_false} papers AND is "
                   f"contamination-amplified")
    elif survives:
        verdict = f"effect survives on {lab_false} papers"
    elif amplified:
        verdict = "contamination-amplified"
    else:
        blockers = []
        if gf is None:
            blockers.append(f"no {lab_false}-group model ({gf_reason})")
        else:
            blockers.append(f"{lab_false}-group coefficient {gf['coef']:+.2f} "
                            f"(p={fmt_p(gf['p'])}) is not significantly positive")
        if inter is None:
            blockers.append(f"no interaction model ({inter_reason})")
        else:
            blockers.append(f"interaction {inter['coef']:+.2f} (p={fmt_p(inter['p'])}) "
                            f"is not significantly positive")
        verdict = "underpowered/indeterminate - " + "; ".join(blockers)
    s3 = f"Verdict: {verdict}."

    md.append("### 4. Verdict")
    md.append("")
    md.append(f"{s1} {s2} {s3}")
    md.append("")

    stats = {"groups": groups, "group_aids": {k: list(v) for k, v in group_aids.items()},
             "interaction": inter, "interaction_reason": inter_reason,
             "group_models": gm, "group_model_reasons": gm_reason,
             "auc": auc_stats, "survives": survives, "amplified": amplified,
             "verdict": verdict, "sentences": [s1, s2, s3],
             "labels": (lab_true, lab_false)}
    return "\n".join(md), stats


# --------------------------------------------------------------------- driver
def run_analysis(log: str, recognition: str, manifest: str, split: str):
    """Full analysis -> (markdown_report, {model: stats_dict})."""
    recs_by_model, n_total, n_used = load_reviews(log, None)
    flags, n_rec_rows, n_dup = load_recognition(recognition, split)
    man = {str(m["arxiv_id"]): m for m in read_jsonl(manifest)}
    lab_true, lab_false = SPLIT_LABELS[split]

    n_true = sum(1 for v in flags.values() if v)
    header = [
        "# Contamination split: version effect by recognition",
        "",
        f"- reviews log: `{log}` - {n_total} rows, {n_used} parsed_ok rating rows",
        f"- recognition log: `{recognition}` - {n_rec_rows} rows, {len(flags)} papers "
        f"with a boolean `{split}` flag"
        + (f" ({n_dup} duplicate arxiv_id rows ignored, first kept)" if n_dup else ""),
        f"- split `{split}`: {n_true} {lab_true} / {len(flags) - n_true} {lab_false}; "
        f"papers missing from the recognition log -> '{UNPROBED}' "
        f"(reported, excluded from tests)",
        f"- manifest: `{manifest}` ({len(man)} papers)",
        "",
    ]

    sections = []
    stats_by_model: dict[str, dict] = {}
    for model in sorted(recs_by_model):
        sec, stats = analyze_model(model, recs_by_model[model], flags, man, split)
        sections.append(sec)
        stats_by_model[model] = stats
    if not recs_by_model:
        sections.append("_No parsed_ok rating rows found in the log - nothing to "
                        "analyze._")

    return "\n".join(header + sections) + "\n", stats_by_model


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Split the version effect by model recognition of the paper.")
    ap.add_argument("--log", default="data/processed/reviews_log.jsonl")
    ap.add_argument("--recognition", default="data/processed/recognition_log.jsonl")
    ap.add_argument("--manifest", default="data/processed/pilot_manifest.jsonl")
    ap.add_argument("--out", default="results/contamination_split.md",
                    help="markdown report, OVERWRITTEN each run")
    ap.add_argument("--split", default="recognized",
                    choices=["recognized", "authors_any_correct"],
                    help="recognition field that defines the groups")
    args = ap.parse_args()

    report, stats_by_model = run_analysis(args.log, args.recognition, args.manifest,
                                          args.split)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report)

    print("\n=== analyze_contamination_split summary ===")
    print(f"split: {args.split} | models: "
          f"{', '.join(sorted(stats_by_model)) or 'none'}")
    for model, st in sorted(stats_by_model.items()):
        lab_true, lab_false = st["labels"]
        inter, gm = st["interaction"], st["group_models"]
        print(f"[{model}] interaction pos:recog "
              f"{fmt_fit(inter, st['interaction_reason'])}")
        print(f"[{model}] {lab_true}: {fmt_fit(gm[lab_true], st['group_model_reasons'][lab_true])} | "
              f"{lab_false}: {fmt_fit(gm[lab_false], st['group_model_reasons'][lab_false])}")
        print(f"[{model}] verdict: {st['verdict']}")
    print(f"report -> {out}")


if __name__ == "__main__":
    main()
