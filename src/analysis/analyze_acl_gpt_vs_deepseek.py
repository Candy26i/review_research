"""ACL 2025 version tracking: gpt-6.1-sol (one review per version) vs deepseek-chat.

The gpt-6.1-sol arm has one review per version (rep 0). To compare like with like, the
deepseek-chat arm is reduced to its rep-0 review of the same versions ("DeepSeek r0");
the mean of its three reps ("DeepSeek mean of 3") is shown for reference only. All
numbers use the versions scored by both arms ("matched versions"), and a version's
position is its index among the matched versions of its paper (0, 1, 2, ...), the same
definition as analyze_scores.py. Only papers with >= 2 matched versions enter.

Estimates per arm:
  1. Within-paper slope (paper fixed effects; pooled OLS of the within-paper centred
     rating on the within-paper centred position). 95% CI from a cluster bootstrap over
     papers (2,000 draws, numpy Generator seed 42); two-sided p from a sign-flip test on
     the per-paper contributions (10,000 flips, seed 43).
  2. The same slope with ln(word count) as a covariate (word counts from the scrub log).
  3. statsmodels MixedLM: rating ~ position + (1|paper), REML; then + log_words; then a
     random slope by paper.
  4. Binary outcome rating >= 5: within-paper linear-probability slope (bootstrap as in
     1) and a conditional (fixed-effects) logit slope (statsmodels ConditionalLogit).
  5. Standardized slope: the within-paper slope divided by the SD of that arm's ratings
     over all matched rows (both recomputed in each bootstrap draw).
Differences gpt minus DeepSeek r0 are computed on the same bootstrap draws (paired by
paper), with sign-flip p on the per-paper differences, and from a stacked MixedLM (gpt vs
DeepSeek r0; vs DeepSeek mean of 3 in three-review mode)
rating ~ position * arm with a random intercept per paper and per paper x arm (re_formula "1"
plus a paper x arm variance component; before 2026-11-18 the call omitted re_formula, and
statsmodels then drops the paper intercept when vc_formula is given).
Sensitivity (both modes): the slopes and the length-controlled slopes without versions under
1,000 words (text-extraction failures: 2412.12567v1 is a title and an empty heading, 2503.15351v1
one appendix paragraph; every other matched version has >= 1,944 words).

Adjacent pairs (consecutive matched versions) are split into: identical text
(text_sha equal), cosmetic (churn_class cosmetic, text not identical), minor, major
(churn_class from the diffs file). Identical-text pairs give each arm's repeat-call
agreement on the same input for free.

Run from the repo root:
  python3 src/analysis/analyze_acl_gpt_vs_deepseek.py
Outputs (OVERWRITTEN): results/acl_gpt61sol_vs_deepseek.md and .json

Three-review mode (--gpt-reps all): the "gpt" arm becomes the mean of gpt-6.1-sol's three
reviews (reps 0, 1, 2; only versions with exactly one parsed review per rep). The primary
contrast, fixed in the run notes before the extra reviews were called, is gpt mean of 3 vs
DeepSeek mean of 3; DeepSeek r0 stays as a reference column. Instead of identical-text pairs,
repeat-call noise comes from the within-version spread of the three reviews (all versions),
and two checks are added: the slope from each rep separately (with a within-version rep-label
permutation test, for gpt and DeepSeek), and each review's deviation from its version's
3-review mean by API key (key taken from the response's created time; reviews were called with
three different keys). The binary outcome for 3-review arms is "at least 2 of 3 reviews >= 5"
(on gpt's near-binary scale, "mean of 3 >= 5" would mean all three >= 5). Outputs (OVERWRITTEN):
results/acl_gpt61sol_3reviews_vs_deepseek.md and .json
"""
from __future__ import annotations

import argparse
import json
import math
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

N_BOOT = 2000
N_FLIP = 10000


def read_jsonl(path):
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def load_ratings(path, rep=None):
    """(arxiv_id, v) -> list of ratings (parsed_ok rows only; rep filter if given)."""
    out = defaultdict(list)
    for r in read_jsonl(path):
        if not r.get("parsed_ok") or r.get("rating") is None:
            continue
        if rep is not None and int(r["rep"]) != rep:
            continue
        out[(str(r["arxiv_id"]), int(r["v"]))].append(float(r["rating"]))
    return out


def load_text_sha(path):
    return {(str(r["arxiv_id"]), int(r["v"])): r["text_sha"] for r in read_jsonl(path)}


def build_frame(arms: dict, scrub: dict):
    """Long frame over matched versions: paper, v, pos, log_words, one rating column per arm."""
    keys = set.intersection(*[set(d) for d in arms.values()])
    by_paper = defaultdict(list)
    for aid, v in keys:
        by_paper[aid].append(v)
    rows = []
    for aid, vs in by_paper.items():
        vs = sorted(vs)
        if len(vs) < 2:
            continue
        for i, v in enumerate(vs):
            nw = scrub.get(f"{aid}v{v}")
            row = {"paper": aid, "v": v, "pos": i,
                   "log_words": math.log(nw) if nw else np.nan}
            for name, d in arms.items():
                row[name] = float(np.mean(d[(aid, v)]))
            rows.append(row)
    return pd.DataFrame(rows).sort_values(["paper", "pos"]).reset_index(drop=True)


def paper_stats(df, ycols, xcol="pos"):
    """Per-paper sufficient statistics for the within-paper (fixed-effects) slope."""
    g = df.groupby("paper", sort=True)
    xc = df[xcol] - g[xcol].transform("mean")
    out = {"Sxx": (xc * xc).groupby(df["paper"]).sum().to_numpy()}
    for y in ycols:
        yc = df[y] - g[y].transform("mean")
        out[y] = (xc * yc).groupby(df["paper"]).sum().to_numpy()
        # for the standardized slope: per-paper n, sum y, sum y^2
        out[y + "__n"] = g[y].size().to_numpy()
        out[y + "__s"] = g[y].sum().to_numpy()
        out[y + "__ss"] = (df[y] ** 2).groupby(df["paper"]).sum().to_numpy()
    return out


def sd_from(n, s, ss):
    n, s, ss = n.sum(), s.sum(), ss.sum()
    return math.sqrt((ss - s * s / n) / (n - 1))


def boot_idx(n_papers, seed=42):
    rng = np.random.default_rng(seed)
    return rng.integers(0, n_papers, size=(N_BOOT, n_papers))


def ci(a):
    lo, hi = np.percentile(a, [2.5, 97.5])
    return float(lo), float(hi)


def signflip_p(contrib, denom, seed=43):
    """Two-sided sign-flip p for sum(contrib)/denom == 0, per-paper contributions."""
    rng = np.random.default_rng(seed)
    obs = abs(contrib.sum())
    signs = rng.choice([-1.0, 1.0], size=(N_FLIP, contrib.size))
    null = np.abs(signs @ contrib)
    return float((1 + (null >= obs - 1e-12).sum()) / (N_FLIP + 1))


def fe_slope_with_cov(df, y, cov="log_words"):
    """Within-paper slope of y on pos with a within-paper covariate; returns (b_pos, b_cov)."""
    d = df.dropna(subset=[cov])
    d = d[d.groupby("paper")["pos"].transform("size") >= 2]
    g = d.groupby("paper")
    X = np.column_stack([d["pos"] - g["pos"].transform("mean"),
                         d[cov] - g[cov].transform("mean")])
    Y = (d[y] - g[y].transform("mean")).to_numpy()
    return d, X, Y


def fe_cov_boot(df, y, cov="log_words", idx_seed=42):
    d, X, Y = fe_slope_with_cov(df, y, cov)
    papers = d["paper"].to_numpy()
    uniq, inv = np.unique(papers, return_inverse=True)
    # per-paper X'X and X'Y
    XtX = np.zeros((uniq.size, 2, 2))
    XtY = np.zeros((uniq.size, 2))
    np.add.at(XtX, inv, X[:, :, None] * X[:, None, :])
    np.add.at(XtY, inv, X * Y[:, None])
    est = np.linalg.solve(XtX.sum(0), XtY.sum(0))
    idx = boot_idx(uniq.size, idx_seed)
    draws = np.array([np.linalg.solve(XtX[i].sum(0), XtY[i].sum(0)) for i in idx])
    return est, draws, uniq.size


def mixed(df, formula, re_formula=None, groups="paper", vc=None):
    import statsmodels.formula.api as smf
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = smf.mixedlm(formula, df, groups=df[groups], re_formula=re_formula,
                        vc_formula=vc)
        fit = m.fit(reml=True)
    return fit


def coef_line(fit, name):
    b, se, p = float(fit.params[name]), float(fit.bse[name]), float(fit.pvalues[name])
    return {"est": b, "lo": b - 1.96 * se, "hi": b + 1.96 * se, "p": p,
            "converged": bool(getattr(fit, "converged", True))}


def cond_logit(df, y, ycol_b=None):
    from statsmodels.discrete.conditional_models import ConditionalLogit
    d = df.copy()
    d["yb"] = (d[ycol_b] if ycol_b else (d[y] >= 5)).astype(int)
    var = d.groupby("paper")["yb"].transform("std").fillna(0)
    d = d[var > 0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = ConditionalLogit(d["yb"].to_numpy(), d[["pos"]].to_numpy(),
                               groups=d["paper"].to_numpy()).fit(disp=0)
    b, se = float(fit.params[0]), float(fit.bse[0])
    return {"est": b, "lo": b - 1.96 * se, "hi": b + 1.96 * se,
            "p": float(fit.pvalues[0]), "n_papers_informative": int(d["paper"].nunique())}


def pair_table(df, arm_cols, text_sha, diffs):
    rows = []
    for aid, g in df.groupby("paper", sort=False):
        g = g.sort_values("pos")
        vs = g["v"].tolist()
        for i in range(len(vs) - 1):
            a, b = vs[i], vs[i + 1]
            same = text_sha.get((aid, a)) == text_sha.get((aid, b))
            dr = diffs.get((aid, a, b))
            cls = "identical text" if same else (dr["churn_class"] if dr else "no diff row")
            rec = {"paper": aid, "class": cls}
            for c in arm_cols:
                rec[c] = float(g.iloc[i + 1][c] - g.iloc[i][c])
            rows.append(rec)
    return pd.DataFrame(rows)


def pair_summary(pairs, col, rng_seed=44):
    out = {}
    order = ["identical text", "cosmetic", "minor", "major", "no diff row", "ALL"]
    rng = np.random.default_rng(rng_seed)
    for cls in order:
        sub = pairs if cls == "ALL" else pairs[pairs["class"] == cls]
        if sub.empty:
            continue
        d = sub[col].to_numpy()
        # cluster bootstrap of the mean delta over papers
        papers = sub["paper"].to_numpy()
        uniq, inv = np.unique(papers, return_inverse=True)
        s = np.bincount(inv, weights=d, minlength=uniq.size)
        n = np.bincount(inv, minlength=uniq.size).astype(float)
        idx = rng.integers(0, uniq.size, size=(N_BOOT, uniq.size))
        boots = s[idx].sum(1) / n[idx].sum(1)
        out[cls] = {"n": int(d.size), "p_up": float((d > 0).mean()),
                    "p_tie": float((d == 0).mean()), "p_down": float((d < 0).mean()),
                    "mean": float(d.mean()), "mean_ci": ci(boots),
                    "mean_abs": float(np.abs(d).mean())}
    return out


def fmt_ci(e, lo, hi, nd=3):
    return f"{e:+.{nd}f} [{lo:+.{nd}f}, {hi:+.{nd}f}]"


def fmt_p(p):
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--gpt-log", default="data/processed/acl_reviews_gpt61sol_log.jsonl")
    ap.add_argument("--ds-log", default="data/processed/acl_reviews_log.jsonl")
    ap.add_argument("--scrub", default="data/processed/acl_scrub_log.jsonl")
    ap.add_argument("--diffs", default="data/processed/acl_version_diffs.jsonl")
    ap.add_argument("--out", default=None,
                    help="default: results/acl_gpt61sol_vs_deepseek.md, or "
                         "results/acl_gpt61sol_3reviews_vs_deepseek.md with --gpt-reps all")
    ap.add_argument("--gpt-reps", choices=["0", "all"], default="0",
                    help="0: one gpt review per version (rep 0); all: mean of reps 0, 1, 2")
    args = ap.parse_args()
    three = args.gpt_reps == "all"
    if args.out is None:
        args.out = ("results/acl_gpt61sol_3reviews_vs_deepseek.md" if three
                    else "results/acl_gpt61sol_vs_deepseek.md")

    if not three:
        # rep 0 only: the log also holds later reviews (reps 1, 2), which must not enter
        # the one-review-per-version arm
        gpt = load_ratings(args.gpt_log, rep=0)
    else:
        by_rep = {r: load_ratings(args.gpt_log, rep=r) for r in (0, 1, 2)}
        for r, d in by_rep.items():
            dup = [k for k, v in d.items() if len(v) != 1]
            assert not dup, f"gpt rep {r}: several parsed rows for {dup[:3]}"
        gpt = {k: [by_rep[r][k][0] for r in (0, 1, 2)] for k in by_rep[0]
               if k in by_rep[1] and k in by_rep[2]}
        n_gpt_incomplete = len(set(by_rep[0]) | set(by_rep[1]) | set(by_rep[2])) - len(gpt)
    ds_by_rep = {r: load_ratings(args.ds_log, rep=r) for r in (0, 1, 2)}
    for r, d in ds_by_rep.items():
        dup = [k for k, v in d.items() if len(v) != 1]
        assert not dup, f"DeepSeek rep {r}: several parsed rows for {dup[:3]}"
    assert set(ds_by_rep[0]) == set(ds_by_rep[1]) == set(ds_by_rep[2]), \
        "DeepSeek versions without exactly one review per rep 0, 1, 2"
    ds0 = load_ratings(args.ds_log, rep=0)
    ds3 = load_ratings(args.ds_log)
    assert all(len(v) == 3 for v in ds3.values()), "DeepSeek: a version without three reviews"
    scrub = {str(r["key"]): r["n_words"] for r in read_jsonl(args.scrub)
             if isinstance(r.get("n_words"), (int, float))}
    diffs = {(str(r["arxiv_id"]), int(r["v_from"]), int(r["v_to"])): r
             for r in read_jsonl(args.diffs)}
    text_sha = load_text_sha(args.gpt_log)

    arms = {"gpt": gpt, "ds0": ds0, "ds3": ds3}
    if not three:
        labels = {"gpt": "gpt-6.1-sol (1 review)", "ds0": "DeepSeek r0 (1 review)",
                  "ds3": "DeepSeek mean of 3 (reference)"}
        other = "ds0"            # the primary comparison arm
    else:
        labels = {"gpt": "gpt-6.1-sol mean of 3", "ds0": "DeepSeek r0 (1 review, reference)",
                  "ds3": "DeepSeek mean of 3"}
        other = "ds3"
    df = build_frame(arms, scrub)
    n_papers, n_rows = df["paper"].nunique(), len(df)
    ycols = list(arms)
    for y in ycols:
        df[y + "_b"] = (df[y] >= 5).astype(float)
    if three:
        # 3-review arms: at least 2 of the 3 reviews >= 5 (majority), comparable across scales
        for y, src in (("gpt", gpt), ("ds3", ds3)):
            df[y + "_b"] = [float(sum(x >= 5 for x in src[(p, v)]) >= 2)
                            for p, v in zip(df["paper"], df["v"])]
    st = paper_stats(df, ycols + [y + "_b" for y in ycols])
    Sxx = st["Sxx"]
    idx = boot_idx(n_papers)
    res = {"n_papers": n_papers, "n_versions": n_rows, "arms": {}, "diff": {}}

    # versions-per-paper weight
    nv = df.groupby("paper")["pos"].size().to_numpy()
    res["weight_share_ge4_versions"] = float(Sxx[nv >= 4].sum() / Sxx.sum())
    res["n_papers_ge4_versions"] = int((nv >= 4).sum())

    draws = {}
    for y in ycols:
        a = {}
        b = st[y].sum() / Sxx.sum()
        bd = st[y][idx].sum(1) / Sxx[idx].sum(1)
        draws[y] = bd
        a["fe_slope"] = {"est": float(b), "ci": ci(bd), "p": signflip_p(st[y], Sxx.sum())}
        # standardized
        sd = sd_from(st[y + "__n"], st[y + "__s"], st[y + "__ss"])
        sdd = np.array([sd_from(st[y + "__n"][i], st[y + "__s"][i], st[y + "__ss"][i])
                        for i in idx])
        draws[y + "_std"] = bd / sdd
        a["sd_rating"] = sd
        a["fe_slope_std"] = {"est": float(b / sd), "ci": ci(bd / sdd)}
        # equal weight per paper: mean of per-paper OLS slopes
        per = st[y] / Sxx
        pb = per[idx].mean(1)
        a["mean_per_paper_slope"] = {"est": float(per.mean()), "ci": ci(pb)}
        # binary
        yb = y + "_b"
        bb = st[yb].sum() / Sxx.sum()
        bbd = st[yb][idx].sum(1) / Sxx[idx].sum(1)
        draws[yb] = bbd
        a["share_ge5"] = float(df[yb].mean())
        a["fe_lpm_ge5"] = {"est": float(bb), "ci": ci(bbd), "p": signflip_p(st[yb], Sxx.sum())}
        a["cond_logit_ge5"] = cond_logit(df, y, ycol_b=yb if three else None)
        # length covariate (fixed effects)
        est, dr, npl = fe_cov_boot(df, y)
        draws[y + "_len"] = dr[:, 0]
        a["fe_slope_len"] = {"est": float(est[0]), "ci": ci(dr[:, 0]),
                             "log_words": float(est[1]), "log_words_ci": ci(dr[:, 1]),
                             "n_papers": int(npl)}
        # mixed models
        a["mixed_ri"] = coef_line(mixed(df, f"{y} ~ pos"), "pos")
        d_len = df.dropna(subset=["log_words"])
        a["mixed_ri_len"] = coef_line(mixed(d_len, f"{y} ~ pos + log_words"), "pos")
        try:
            a["mixed_rs"] = coef_line(mixed(df, f"{y} ~ pos", re_formula="~pos"), "pos")
        except Exception as exc:  # pragma: no cover
            a["mixed_rs"] = {"error": f"{type(exc).__name__}: {exc}"}
        a["rating_hist"] = {str(k): int(v) for k, v in
                            df[y].round(2).value_counts().sort_index().items()} \
            if y != "ds3" and not (three and y == "gpt") else None
        res["arms"][y] = a

    # paired differences gpt - ds0 (and gpt - ds3 for reference)
    for o in ("ds0", "ds3"):
        c = st["gpt"] - st[o]
        dd = draws["gpt"] - draws[o]
        out = {"fe_slope": {"est": float(c.sum() / Sxx.sum()), "ci": ci(dd),
                            "p": signflip_p(c, Sxx.sum())}}
        dstd = draws["gpt_std"] - draws[o + "_std"]
        out["fe_slope_std"] = {"est": res["arms"]["gpt"]["fe_slope_std"]["est"]
                               - res["arms"][o]["fe_slope_std"]["est"], "ci": ci(dstd)}
        cb = st["gpt_b"] - st[o + "_b"]
        out["fe_lpm_ge5"] = {"est": float(cb.sum() / Sxx.sum()),
                             "ci": ci(draws["gpt_b"] - draws[o + "_b"]),
                             "p": signflip_p(cb, Sxx.sum())}
        dl = draws["gpt_len"] - draws[o + "_len"]
        out["fe_slope_len"] = {"est": res["arms"]["gpt"]["fe_slope_len"]["est"]
                               - res["arms"][o]["fe_slope_len"]["est"], "ci": ci(dl)}
        res["diff"]["gpt_minus_" + o] = out

    # stacked mixed model gpt vs the primary comparison arm
    long = pd.concat([df[["paper", "pos"]].assign(rating=df["gpt"], arm="gpt"),
                      df[["paper", "pos"]].assign(rating=df[other], arm=other)],
                     ignore_index=True)
    long["is_gpt"] = (long["arm"] == "gpt").astype(float)
    # re_formula "1" is required: with vc_formula alone statsmodels fits no paper intercept
    fit = mixed(long, "rating ~ pos * is_gpt", re_formula="1", vc={"arm": "0 + C(arm)"})
    res["diff"]["gpt_minus_" + other]["mixed_stacked"] = coef_line(fit, "pos:is_gpt")

    # adjacent pairs
    pairs = pair_table(df, ycols, text_sha, diffs)
    res["pairs"] = {y: pair_summary(pairs, y) for y in ycols}

    # direction among changed pairs: share of non-zero deltas that go up
    res["share_up_among_changed"] = {}
    for y in ycols:
        out = {}
        for cls in ["identical text", "cosmetic", "minor", "major", "ALL"]:
            sub = pairs if cls == "ALL" else pairs[pairs["class"] == cls]
            ch = sub[y][sub[y] != 0]
            out[cls] = {"n_changed": int(ch.size),
                        "share_up": float((ch > 0).mean()) if ch.size else None}
        res["share_up_among_changed"][y] = out

    papers_sorted = np.array(sorted(df["paper"].unique()))
    if not three:
        # repeat-call noise of a single review from identical-text pairs (same input twice):
        # sigma = sqrt(mean(delta^2) / 2); then slope / sigma, bootstrapped on the same draws
        ident = pairs[pairs["class"] == "identical text"]
        pos_of = {p: i for i, p in enumerate(papers_sorted)}
        pi = np.array([pos_of[p] for p in ident["paper"]], dtype=int)
        n_id = np.bincount(pi, minlength=papers_sorted.size).astype(float)
        res["retest"] = {"n_pairs": int(ident.shape[0]), "n_papers": int((n_id > 0).sum())}
        sig_draws = {}
        for y in ("gpt", "ds0"):
            sq = np.bincount(pi, weights=ident[y].to_numpy() ** 2, minlength=papers_sorted.size)
            sigma = math.sqrt(sq.sum() / (2 * n_id.sum()))
            sd = np.sqrt(sq[idx].sum(1) / (2 * n_id[idx].sum(1)))
            ratio = draws[y] / sd
            sig_draws[y] = ratio
            res["retest"][y] = {"sigma": sigma, "sigma_ci": ci(sd),
                                "slope_over_sigma": float(res["arms"][y]["fe_slope"]["est"] / sigma),
                                "slope_over_sigma_ci": ci(ratio)}
        res["retest"]["diff_slope_over_sigma"] = {
            "est": res["retest"]["gpt"]["slope_over_sigma"] - res["retest"]["ds0"]["slope_over_sigma"],
            "ci": ci(sig_draws["gpt"] - sig_draws["ds0"])}
    else:
        # repeat-call noise from the three reviews of every version: pooled within-version
        # variance (ddof 1) = sigma^2 of one review; the mean of 3 has sigma / sqrt(3).
        # Noise-scaled slope = slope of the 3-review mean / sigma of one review.
        noise = {"n_versions": n_rows}
        sig_draws = {}
        for y, src in (("gpt", gpt), ("ds3", ds3)):
            wv = np.array([np.var(src[(p, v)], ddof=1) for p, v in zip(df["paper"], df["v"])])
            wsum = pd.Series(wv).groupby(df["paper"].to_numpy()).sum().reindex(papers_sorted).to_numpy()
            nver = df.groupby("paper").size().reindex(papers_sorted).to_numpy().astype(float)
            sigma = math.sqrt(wsum.sum() / nver.sum())
            sd = np.sqrt(wsum[idx].sum(1) / nver[idx].sum(1))
            ratio = draws[y] / sd
            sig_draws[y] = ratio
            noise[y] = {"sigma_one_review": sigma, "sigma_ci": ci(sd),
                        "sigma_mean_of_3": sigma / math.sqrt(3),
                        "share_versions_all_3_equal": float(np.mean(wv == 0)),
                        "slope_over_sigma": float(res["arms"][y]["fe_slope"]["est"] / sigma),
                        "slope_over_sigma_ci": ci(ratio)}
        noise["diff_slope_over_sigma"] = {
            "est": noise["gpt"]["slope_over_sigma"] - noise["ds3"]["slope_over_sigma"],
            "ci": ci(sig_draws["gpt"] - sig_draws["ds3"]),
            "ratio": noise["gpt"]["slope_over_sigma"] / noise["ds3"]["slope_over_sigma"],
            "ratio_ci": ci(sig_draws["gpt"] / sig_draws["ds3"])}
        res["noise_3reps"] = noise
        res["n_gpt_versions_incomplete"] = int(n_gpt_incomplete)

        # check 1: the slope from each gpt rep on its own (same frame and draws)
        per_rep = {}
        for r in (0, 1, 2):
            col = f"gpt_r{r}"
            df[col] = [gpt[(p, v)][r] for p, v in zip(df["paper"], df["v"])]
            s_r = paper_stats(df, [col])
            bd = s_r[col][idx].sum(1) / s_r["Sxx"][idx].sum(1)
            draws[col] = bd
            per_rep[col] = {"fe_slope": {"est": float(s_r[col].sum() / s_r["Sxx"].sum()), "ci": ci(bd)},
                            "mean_rating": float(df[col].mean()),
                            "share_ge5": float((df[col] >= 5).mean()),
                            "hist": {str(k): int(v) for k, v in
                                     df[col].value_counts().sort_index().items()}}
        for r in (1, 2):
            per_rep[f"r{r}_minus_r0"] = {
                "slope": {"est": per_rep[f"gpt_r{r}"]["fe_slope"]["est"] - per_rep["gpt_r0"]["fe_slope"]["est"],
                          "ci": ci(draws[f"gpt_r{r}"] - draws["gpt_r0"])},
                "mean": float((df[f"gpt_r{r}"] - df["gpt_r0"]).mean())}
        per_rep["r2_minus_r1"] = {
            "slope": {"est": per_rep["gpt_r2"]["fe_slope"]["est"] - per_rep["gpt_r1"]["fe_slope"]["est"],
                      "ci": ci(draws["gpt_r2"] - draws["gpt_r1"])},
            "mean": float((df["gpt_r2"] - df["gpt_r1"]).mean())}
        # DeepSeek reps on their own (same frame), for comparison
        for r in (0, 1, 2):
            col = f"ds_r{r}"
            df[col] = [ds_by_rep[r][(p, int(v))][0] for p, v in zip(df["paper"], df["v"])]
            s_r = paper_stats(df, [col])
            per_rep[col] = {"fe_slope": {"est": float(s_r[col].sum() / s_r["Sxx"].sum()),
                                         "ci": ci(s_r[col][idx].sum(1) / s_r["Sxx"][idx].sum(1))},
                            "mean_rating": float(df[col].mean())}
        # within-version rep-label permutation test: if the three reviews of a version are
        # exchangeable, shuffling their labels leaves the spread (max - min) of the three
        # single-review slopes unchanged in distribution. 10,000 permutations, seed 45.
        xc = (df["pos"] - df.groupby("paper")["pos"].transform("mean")).to_numpy()
        sxx_all = float((xc * xc).sum())
        perms = np.array([[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]])
        n_perm = 10000
        for name, cols3 in (("gpt", ["gpt_r0", "gpt_r1", "gpt_r2"]), ("ds", ["ds_r0", "ds_r1", "ds_r2"])):
            Y3 = df[cols3].to_numpy()                       # versions x 3
            obs_sl = (xc[:, None] * Y3).sum(0) / sxx_all
            obs = float(obs_sl.max() - obs_sl.min())
            rng = np.random.default_rng(45)
            null = np.empty(n_perm)
            for c0 in range(0, n_perm, 500):
                k = min(500, n_perm - c0)
                pick = perms[rng.integers(0, 6, size=(k, Y3.shape[0]))]          # k x versions x 3
                Yp = np.take_along_axis(np.broadcast_to(Y3, (k,) + Y3.shape), pick, axis=2)
                sl = np.einsum("v,kvr->kr", xc, Yp) / sxx_all
                null[c0:c0 + k] = sl.max(1) - sl.min(1)
            per_rep[f"perm_{name}"] = {"spread_of_rep_slopes": obs,
                                       "p": float((1 + (null >= obs - 1e-12).sum()) / (n_perm + 1)),
                                       "n_perm": n_perm}
        # post hoc sensitivity (rep 1 sits above reps 0 and 2): gpt mean of reps 0 and 2
        df["gpt_r02"] = (df["gpt_r0"] + df["gpt_r2"]) / 2
        s_r = paper_stats(df, ["gpt_r02"])
        bd = s_r["gpt_r02"][idx].sum(1) / s_r["Sxx"][idx].sum(1)
        per_rep["gpt_r02"] = {"fe_slope": {"est": float(s_r["gpt_r02"].sum() / s_r["Sxx"].sum()),
                                           "ci": ci(bd)},
                              "minus_ds3": {"est": float(s_r["gpt_r02"].sum() / s_r["Sxx"].sum())
                                            - res["arms"]["ds3"]["fe_slope"]["est"],
                                            "ci": ci(bd - draws["ds3"])}}
        # realized CI narrowing: width of the mean-of-3 CI over each single rep's CI width
        w3 = np.diff(res["arms"]["gpt"]["fe_slope"]["ci"])[0]
        per_rep["ci_width_ratio_mean3_over_single"] = {
            f"r{r}": float(w3 / np.diff(per_rep[f"gpt_r{r}"]["fe_slope"]["ci"])[0]) for r in (0, 1, 2)}
        res["per_rep"] = per_rep

        # check 2: API key, from the response's created time (run notes: the first key's last
        # response 2026-11-16 03:23:12 UTC, the second key's first 06:01:08; .env switched to the
        # third key 2026-11-17 07:30:45 UTC). Deviation of a review from its version's 3-review mean.
        raw_of = {}
        for r in read_jsonl(args.gpt_log):
            if r.get("parsed_ok") and r.get("rating") is not None:
                raw_of[(str(r["arxiv_id"]), int(r["v"]), int(r["rep"]))] = r["raw_path"]
        t_key2 = pd.Timestamp("2026-11-16 05:00:00", tz="UTC").timestamp()
        t_key3 = pd.Timestamp("2026-11-17 07:30:45", tz="UTC").timestamp()
        keyrec = []
        for p, v in zip(df["paper"], df["v"]):
            m = float(np.mean(gpt[(p, v)]))
            for r in (0, 1, 2):
                with open(raw_of[(p, int(v), r)]) as fh:
                    created = json.load(fh)["created"]
                k = 1 if created < t_key2 else (2 if created < t_key3 else 3)
                keyrec.append((p, k, r, gpt[(p, v)][r] - m))
        kd = pd.DataFrame(keyrec, columns=["paper", "key", "rep", "dev"])
        res["key_by_rep"] = {f"key{k}": {f"rep{r}": int(((kd["key"] == k) & (kd["rep"] == r)).sum())
                                         for r in (0, 1, 2)} for k in (1, 2, 3)}
        pidx = {p: i for i, p in enumerate(papers_sorted)}
        kd["pi"] = kd["paper"].map(pidx)
        res["by_key"] = {}
        for k in (1, 2, 3):
            sub = kd[kd["key"] == k]
            if sub.empty:
                continue
            ssum = np.bincount(sub["pi"], weights=sub["dev"], minlength=papers_sorted.size)
            scnt = np.bincount(sub["pi"], minlength=papers_sorted.size).astype(float)
            cnt = scnt[idx].sum(1)
            bd = ssum[idx].sum(1)[cnt > 0] / cnt[cnt > 0]   # draws without this key are dropped
            res["by_key"][str(k)] = {"n_reviews": int(len(sub)),
                                     "mean_dev_from_version_mean": float(sub["dev"].mean()),
                                     "ci": ci(bd), "n_draws": int(bd.size)}

    # sensitivity: drop versions whose text equals the previous matched version's text
    keep = []
    for aid, g in df.groupby("paper", sort=False):
        prev = None
        for _, r in g.sort_values("pos").iterrows():
            s = text_sha.get((aid, int(r["v"])))
            keep.append(s != prev)
            prev = s
    dfd = df[np.array(keep)].copy()
    dfd = dfd[dfd.groupby("paper")["pos"].transform("size") >= 2].copy()
    dfd["pos"] = dfd.groupby("paper").cumcount()
    std_ = paper_stats(dfd, ycols)
    idx2 = boot_idx(dfd["paper"].nunique())
    sens = {"n_papers": int(dfd["paper"].nunique()), "n_versions": int(len(dfd))}
    for y in ycols:
        b = std_[y].sum() / std_["Sxx"].sum()
        bd = std_[y][idx2].sum(1) / std_["Sxx"][idx2].sum(1)
        sens[y] = {"est": float(b), "ci": ci(bd)}
        if y == "gpt":
            gd = bd
    dd = gd - (std_[other][idx2].sum(1) / std_["Sxx"][idx2].sum(1))
    sens["gpt_minus_" + other] = {"est": sens["gpt"]["est"] - sens[other]["est"], "ci": ci(dd)}
    res["sens_dedup_identical_text"] = sens

    # sensitivity: versions under 1,000 words dropped (text-extraction failures, see docstring)
    min_words = 1000
    short = (df["log_words"] < math.log(min_words)).to_numpy()
    dfs = df[~short].copy()
    left_one = sorted(set(dfs["paper"]) - set(dfs[dfs.groupby("paper")["pos"].transform("size") >= 2]["paper"]))
    dfs = dfs[dfs.groupby("paper")["pos"].transform("size") >= 2].copy()
    dfs["pos"] = dfs.groupby("paper").cumcount()
    sts = paper_stats(dfs, ycols)
    idx3 = boot_idx(dfs["paper"].nunique())
    sx = {"min_words": min_words,
          "dropped": [f"{p}v{int(v)}" for p, v in zip(df["paper"][short], df["v"][short])],
          "dropped_ratings": {f"{r['paper']}v{int(r['v'])}": {y: float(r[y]) for y in ycols}
                              for _, r in df[short].iterrows()},
          "papers_left_with_one_version": left_one,
          "min_words_kept": int(round(math.exp(df.loc[~short, "log_words"].min()))),
          "n_papers": int(dfs["paper"].nunique()), "n_versions": int(len(dfs))}
    sdraw = {}
    for y in ycols:
        b = sts[y].sum() / sts["Sxx"].sum()
        bd = sts[y][idx3].sum(1) / sts["Sxx"][idx3].sum(1)
        est, dr, _ = fe_cov_boot(dfs, y)
        sdraw[y], sdraw[y + "_len"] = bd, dr[:, 0]
        sx[y] = {"fe_slope": {"est": float(b), "ci": ci(bd)},
                 "fe_slope_len": {"est": float(est[0]), "ci": ci(dr[:, 0])}}
    for o in ("ds0", "ds3"):
        sx["gpt_minus_" + o] = {
            "fe_slope": {"est": sx["gpt"]["fe_slope"]["est"] - sx[o]["fe_slope"]["est"],
                         "ci": ci(sdraw["gpt"] - sdraw[o])},
            "fe_slope_len": {"est": sx["gpt"]["fe_slope_len"]["est"] - sx[o]["fe_slope_len"]["est"],
                             "ci": ci(sdraw["gpt_len"] - sdraw[o + "_len"])}}
    res["sens_min_words"] = sx

    Path(args.out).with_suffix(".json").write_text(json.dumps(res, indent=2))

    # ---------- markdown ----------
    A = res["arms"]
    DS_NOTE = ("- DeepSeek: requests went to the deepseek-chat alias; every raw response reports model "
               "deepseek-flash with one system_fingerprint (e72d3c7f), and all three reps were called "
               "interleaved within one session, 2026-10-23 08:51-15:14 UTC.")
    if not three:
        head =["# ACL 2025 version tracking: gpt-6.1-sol vs deepseek-chat", "",
                f"- gpt log: `{args.gpt_log}` | DeepSeek log: `{args.ds_log}`",
                f"- matched versions (scored by both arms): {n_rows} versions of {n_papers} papers "
                "with >= 2 matched versions; position = index among the paper's matched versions.",
                "- gpt-6.1-sol: one review per version (rep 0), reasoning effort medium (API default), "
                "temperature 1.0. Its noise level is estimated below from identical-text pairs; a direct "
                "retest (a second review of 500 random versions) is in `results/acl_gpt61sol_retest.md` "
                "and agrees (SD 0.40 vs 0.41).",
                "- DeepSeek r0 = the rep-0 review of each version (same prompt_sha and text_sha), "
                "the like-for-like comparison. DeepSeek mean of 3 is for reference.",
                DS_NOTE]
    else:
        head = ["# ACL 2025 version tracking: gpt-6.1-sol vs deepseek-chat, three reviews per version", "",
                f"- gpt log: `{args.gpt_log}` | DeepSeek log: `{args.ds_log}`",
                f"- matched versions (three gpt-6.1-sol reviews and three DeepSeek reviews): {n_rows} "
                f"versions of {n_papers} papers with >= 2 matched versions; position = index among the "
                f"paper's matched versions. gpt versions without all three reviews: "
                f"{res['n_gpt_versions_incomplete']}.",
                "- gpt-6.1-sol mean of 3 = mean of reps 0, 1, 2 (rep 0: main run, 2026-11-16; "
                "rep 1: 500-version retest of 2026-11-16 + the three-review run; rep 2: the three-review "
                "run, 2026-11-17/18; reasoning_effort not sent (API default, medium), temperature 1.0, "
                "flex tier except 60 default-tier smoke/pilot rep-0 reviews; three API keys, see the "
                "key check).",
                "- Primary contrast, fixed in the run notes before the extra reviews were called (after "
                "the one-review results had been seen): gpt mean of 3 vs DeepSeek mean of 3, by the "
                "within-paper (fixed-effects) slope. Everything else is secondary. DeepSeek r0 is a "
                "reference column; the one-review comparison is `results/acl_gpt61sol_vs_deepseek.md`.",
                DS_NOTE]
    L = head + [
         f"- {res['n_papers_ge4_versions']} papers with >= 4 versions carry "
         f"{100 * res['weight_share_ge4_versions']:.0f}% of the weight of the within-paper slope.",
         "", "## Rating change per revision step", "",
         "| estimate | " + " | ".join(labels[y] for y in ycols) + " |",
         "|---|" + "---|" * len(ycols)]

    def row(name, f):
        L.append(f"| {name} | " + " | ".join(f(A[y]) for y in ycols) + " |")

    row("within-paper slope [95% CI], sign-flip p",
        lambda a: f"{fmt_ci(a['fe_slope']['est'], *a['fe_slope']['ci'])}, {fmt_p(a['fe_slope']['p'])}")
    row("same, with ln(words) covariate",
        lambda a: fmt_ci(a['fe_slope_len']['est'], *a['fe_slope_len']['ci']))
    row("mixed model, random intercept",
        lambda a: f"{fmt_ci(a['mixed_ri']['est'], a['mixed_ri']['lo'], a['mixed_ri']['hi'])}, {fmt_p(a['mixed_ri']['p'])}")
    row("mixed model + ln(words)",
        lambda a: fmt_ci(a['mixed_ri_len']['est'], a['mixed_ri_len']['lo'], a['mixed_ri_len']['hi']))
    row("mixed model, random intercept and slope",
        lambda a: (fmt_ci(a['mixed_rs']['est'], a['mixed_rs']['lo'], a['mixed_rs']['hi'])
                   + ("" if a['mixed_rs'].get('converged', True) else " (not converged)"))
        if "est" in a["mixed_rs"] else a["mixed_rs"]["error"])
    row("equal weight per paper (mean of per-paper slopes)",
        lambda a: fmt_ci(a['mean_per_paper_slope']['est'], *a['mean_per_paper_slope']['ci']))
    row("SD of ratings", lambda a: f"{a['sd_rating']:.2f}")
    row("standardized within-paper slope (slope / SD; reference column: SD of 3-review means)" if not three
        else "standardized within-paper slope (slope / SD; 3-review arms: SD of 3-review means)",
        lambda a: fmt_ci(a['fe_slope_std']['est'], *a['fe_slope_std']['ci']))
    row("share rated >= 5 (reference column: mean of 3 >= 5)" if not three
        else "share rated >= 5 (3-review arms: at least 2 of 3 reviews >= 5)",
        lambda a: f"{100 * a['share_ge5']:.1f}%")
    row("P(rating >= 5), percentage points per step (reference column: mean of 3 >= 5)" if not three
        else "P(rating >= 5), percentage points per step (3-review arms: at least 2 of 3 reviews >= 5)",
        lambda a: f"{fmt_ci(100 * a['fe_lpm_ge5']['est'], 100 * a['fe_lpm_ge5']['ci'][0], 100 * a['fe_lpm_ge5']['ci'][1], 1)}, {fmt_p(a['fe_lpm_ge5']['p'])}")
    row("conditional logit, log-odds of >= 5 per step (reference column: mean of 3 >= 5)" if not three
        else "conditional logit, log-odds of >= 5 per step (3-review arms: at least 2 of 3 reviews >= 5)",
        lambda a: f"{fmt_ci(a['cond_logit_ge5']['est'], a['cond_logit_ge5']['lo'], a['cond_logit_ge5']['hi'], 2)} "
                  f"({a['cond_logit_ge5']['n_papers_informative']} papers with variation)")

    if not three:
        R = res["retest"]
        L += ["", "## Slope relative to each model's own repeat-call noise", "",
              f"Noise of a single review, from the {R['n_pairs']} adjacent pairs whose text is "
              f"identical ({R['n_papers']} papers): sigma = sqrt(mean(delta^2) / 2). These pairs "
              "are not a random sample of papers, so treat sigma as an estimate.", "",
              "| | gpt-6.1-sol | DeepSeek r0 | gpt minus DeepSeek r0 |", "|---|---|---|---|",
              f"| repeat-call SD of one review [95% CI] | {R['gpt']['sigma']:.2f} [{R['gpt']['sigma_ci'][0]:.2f}, {R['gpt']['sigma_ci'][1]:.2f}] "
              f"| {R['ds0']['sigma']:.2f} [{R['ds0']['sigma_ci'][0]:.2f}, {R['ds0']['sigma_ci'][1]:.2f}] | |",
              f"| within-paper slope / that SD | {fmt_ci(R['gpt']['slope_over_sigma'], *R['gpt']['slope_over_sigma_ci'])} "
              f"| {fmt_ci(R['ds0']['slope_over_sigma'], *R['ds0']['slope_over_sigma_ci'])} "
              f"| {fmt_ci(R['diff_slope_over_sigma']['est'], *R['diff_slope_over_sigma']['ci'])} |"]
    else:
        N = res["noise_3reps"]
        L += ["", "## Slope relative to each model's own repeat-call noise (secondary, not in the plan)", "",
              f"Noise from the three reviews of each of the {N['n_versions']} matched versions: pooled "
              "within-version variance (ddof 1) = sigma^2 of one review; a 3-review mean has sigma / sqrt(3). "
              "Noise-scaled slope = slope of the 3-review mean / sigma of one review. Joint cluster "
              "bootstrap over papers.", "",
              "Read with care: (1) gpt's ratings are almost only 3 or 5, so its noise comes from the "
              f"{100 * (1 - N['gpt']['share_versions_all_3_equal']):.0f}% of versions whose three reviews "
              "disagree, and on a near-binary scale slope / sigma depends on how many versions sit near "
              "the 3/5 cut; (2) gpt's three reviews were called on different days with three API keys, "
              "DeepSeek's within one session; (3) the 500-version retest (`results/acl_gpt61sol_retest.md`) "
              "is a subset of these reviews, not an independent check. A ratio near 1 means no detectable "
              "difference, not equal sensitivity.", "",
              "| | gpt-6.1-sol | DeepSeek | gpt minus DeepSeek |", "|---|---|---|---|",
              f"| repeat-call SD of one review [95% CI] | {N['gpt']['sigma_one_review']:.2f} [{N['gpt']['sigma_ci'][0]:.2f}, {N['gpt']['sigma_ci'][1]:.2f}] "
              f"| {N['ds3']['sigma_one_review']:.2f} [{N['ds3']['sigma_ci'][0]:.2f}, {N['ds3']['sigma_ci'][1]:.2f}] | |",
              f"| SD of a 3-review mean (sigma / sqrt(3)) | {N['gpt']['sigma_mean_of_3']:.2f} | {N['ds3']['sigma_mean_of_3']:.2f} | |",
              f"| versions with all three reviews equal | {100 * N['gpt']['share_versions_all_3_equal']:.1f}% "
              f"| {100 * N['ds3']['share_versions_all_3_equal']:.1f}% | |",
              f"| within-paper slope (mean of 3) / SD of one review | {fmt_ci(N['gpt']['slope_over_sigma'], *N['gpt']['slope_over_sigma_ci'])} "
              f"| {fmt_ci(N['ds3']['slope_over_sigma'], *N['ds3']['slope_over_sigma_ci'])} "
              f"| {fmt_ci(N['diff_slope_over_sigma']['est'], *N['diff_slope_over_sigma']['ci'])}; ratio "
              f"{N['diff_slope_over_sigma']['ratio']:.2f} [{N['diff_slope_over_sigma']['ratio_ci'][0]:.2f}, "
              f"{N['diff_slope_over_sigma']['ratio_ci'][1]:.2f}] |"]
        P = res["per_rep"]
        L += ["", "## Check: each gpt-6.1-sol review on its own", "",
              "| | within-paper slope [95% CI] | mean rating | share >= 5 | rating counts |",
              "|---|---|---|---|---|"]
        for r in (0, 1, 2):
            q = P[f"gpt_r{r}"]
            L.append(f"| rep {r} | {fmt_ci(q['fe_slope']['est'], *q['fe_slope']['ci'])} | {q['mean_rating']:.3f} "
                     f"| {100 * q['share_ge5']:.1f}% | " + ", ".join(f"{k}: {v}" for k, v in q["hist"].items()) + " |")
        for name, lab in (("r1_minus_r0", "rep 1 minus rep 0"), ("r2_minus_r0", "rep 2 minus rep 0"),
                          ("r2_minus_r1", "rep 2 minus rep 1")):
            q = P[name]
            L.append(f"| {lab} | {fmt_ci(q['slope']['est'], *q['slope']['ci'])} | {q['mean']:+.3f} | | |")
        pg, pd_ = P["perm_gpt"], P["perm_ds"]
        L += ["", "DeepSeek's reps on the same versions: slopes "
              + ", ".join(f"rep {r} {fmt_ci(P[f'ds_r{r}']['fe_slope']['est'], *P[f'ds_r{r}']['fe_slope']['ci'])}"
                          for r in (0, 1, 2))
              + "; mean ratings " + " / ".join(f"{P[f'ds_r{r}']['mean_rating']:.3f}" for r in (0, 1, 2)) + ".", "",
              "Within-version rep-label permutation test (shuffle the three reviews' labels within each "
              f"version, {pg['n_perm']:,} permutations; statistic = largest minus smallest single-review "
              f"slope): gpt spread {pg['spread_of_rep_slopes']:.3f}, p = {pg['p']:.3f}; DeepSeek spread "
              f"{pd_['spread_of_rep_slopes']:.3f}, p = {pd_['p']:.3f}. Many rep, key and level checks were "
              "run, so a p near 0.05 is not interpreted.", "",
              "Post hoc sensitivity without gpt rep 1 (mean of reps 0 and 2): slope "
              f"{fmt_ci(P['gpt_r02']['fe_slope']['est'], *P['gpt_r02']['fe_slope']['ci'])}; minus DeepSeek "
              f"mean of 3 {fmt_ci(P['gpt_r02']['minus_ds3']['est'], *P['gpt_r02']['minus_ds3']['ci'])}.", "",
              "CI width of the mean-of-3 slope relative to a single review's CI width: "
              + ", ".join(f"vs rep {r} {100 * P['ci_width_ratio_mean3_over_single'][f'r{r}']:.0f}%" for r in (0, 1, 2))
              + "."]
        kb = res["key_by_rep"]
        L += ["", "## Check: API key", "",
              "Deviation of each review from its version's 3-review mean, by the API key it was called "
              "with (key from the response's created time: first key before 2026-11-16 05:00 UTC, third "
              "key from 2026-11-17 07:30:45 UTC; run notes). The version mean includes the review itself, "
              "so a review whose key supplies n of the version's three reviews shows (1 - n/3) of that key's "
              "level shift: about 2/3 for key 1 (always one review per version), less for keys 2 and 3, and "
              "key 2's row is near 0 partly by construction (it supplies all three reviews of many versions). "
              "Key is tied to rep and call date (matched versions; reviews by key and rep: "
              + "; ".join(f"key {k[-1]} " + "/".join(str(kb[k][f'rep{r}']) for r in (0, 1, 2)) for k in kb)
              + " for reps 0/1/2), so this check cannot separate key, rep and date.", "",
              "| key | reviews | mean deviation [95% CI] |", "|---|---|---|"]
        for k, q in res["by_key"].items():
            L.append(f"| {k} | {q['n_reviews']} | {fmt_ci(q['mean_dev_from_version_mean'], *q['ci'])} |")
        L += ["", "## Direction of changed adjacent pairs"]
    L += ["", "Share of changed adjacent pairs that go up (pairs with delta != 0):", "",
          "| pair class | " + " | ".join(labels[y] for y in ycols) + " |",
          "|---|" + "---|" * len(ycols)]
    for cls in ["identical text", "cosmetic", "minor", "major", "ALL"]:
        cells = []
        for y in ycols:
            s = res["share_up_among_changed"][y][cls]
            cells.append("n/a" if s["share_up"] is None
                         else f"{100 * s['share_up']:.0f}% of {s['n_changed']}")
        L.append(f"| {cls} | " + " | ".join(cells) + " |")
    # columns: primary comparison first
    if not three:
        cols = [("vs DeepSeek r0", "gpt_minus_ds0"), ("vs DeepSeek mean of 3", "gpt_minus_ds3")]
        std_lab = "standardized slope (mean of 3: SD of 3-review means)"
        b5_lab = "P(rating >= 5), points per step (mean of 3: mean >= 5)"
    else:
        cols = [("vs DeepSeek mean of 3 (primary)", "gpt_minus_ds3"),
                ("vs DeepSeek r0 (reference)", "gpt_minus_ds0")]
        std_lab = ("difference of standardized slopes (each arm's slope / that arm's own SD; SD of "
                   "3-review means for the 3-review arms)")
        b5_lab = "P(rating >= 5), points per step (3-review arms: at least 2 of 3 reviews >= 5)"
    L += ["", "## Difference gpt-6.1-sol minus DeepSeek (same versions, paired by paper)", "",
          f"| estimate | {cols[0][0]} | {cols[1][0]} |", "|---|---|---|"]
    DA, DB = res["diff"][cols[0][1]], res["diff"][cols[1][1]]
    L.append(f"| within-paper slope | {fmt_ci(DA['fe_slope']['est'], *DA['fe_slope']['ci'])}, {fmt_p(DA['fe_slope']['p'])} "
             f"| {fmt_ci(DB['fe_slope']['est'], *DB['fe_slope']['ci'])}, {fmt_p(DB['fe_slope']['p'])} |")
    L.append(f"| with ln(words) covariate | {fmt_ci(DA['fe_slope_len']['est'], *DA['fe_slope_len']['ci'])} "
             f"| {fmt_ci(DB['fe_slope_len']['est'], *DB['fe_slope_len']['ci'])} |")
    L.append(f"| {std_lab} | {fmt_ci(DA['fe_slope_std']['est'], *DA['fe_slope_std']['ci'])} "
             f"| {fmt_ci(DB['fe_slope_std']['est'], *DB['fe_slope_std']['ci'])} |")
    L.append(f"| {b5_lab} | "
             f"{fmt_ci(100 * DA['fe_lpm_ge5']['est'], 100 * DA['fe_lpm_ge5']['ci'][0], 100 * DA['fe_lpm_ge5']['ci'][1], 1)}, {fmt_p(DA['fe_lpm_ge5']['p'])} "
             f"| {fmt_ci(100 * DB['fe_lpm_ge5']['est'], 100 * DB['fe_lpm_ge5']['ci'][0], 100 * DB['fe_lpm_ge5']['ci'][1], 1)}, {fmt_p(DB['fe_lpm_ge5']['p'])} |")
    ms = DA["mixed_stacked"]
    L.append(f"| stacked mixed model, position x arm (random intercept per paper and per paper x arm; "
             f"common residual variance; model-based CI) | {fmt_ci(ms['est'], ms['lo'], ms['hi'])}, {fmt_p(ms['p'])} | - |")

    L += ["", "## Adjacent version pairs", "",
          "Delta = rating of the later version minus the earlier one. Mean delta CI: cluster "
          "bootstrap over papers.", ""]
    for y in ycols:
        L += [f"**{labels[y]}**", "",
              "| pair class | n | P(up) | P(tie) | P(down) | mean delta [95% CI] | mean abs delta |",
              "|---|---|---|---|---|---|---|"]
        for cls, s in res["pairs"][y].items():
            L.append(f"| {cls} | {s['n']} | {100 * s['p_up']:.1f}% | {100 * s['p_tie']:.1f}% | "
                     f"{100 * s['p_down']:.1f}% | {fmt_ci(s['mean'], *s['mean_ci'], 2)} | {s['mean_abs']:.2f} |")
        L.append("")
    S = res["sens_dedup_identical_text"]
    L += ["## Sensitivity: versions with text identical to the previous version dropped", "",
          f"{S['n_versions']} versions of {S['n_papers']} papers.", "",
          "| arm | within-paper slope [95% CI] |", "|---|---|"]
    for y in ycols:
        L.append(f"| {labels[y]} | {fmt_ci(S[y]['est'], *S[y]['ci'])} |")
    so = "gpt_minus_" + other
    L.append(f"| gpt minus {'DeepSeek r0' if not three else 'DeepSeek mean of 3'} | {fmt_ci(S[so]['est'], *S[so]['ci'])} |")
    X = res["sens_min_words"]
    L += ["", f"## Sensitivity: versions under {X['min_words']:,} words dropped (text-extraction failures)", "",
          f"Dropped: {', '.join(X['dropped'])} (every other matched version has >= {X['min_words_kept']:,} "
          "words): near-empty texts (a title with an empty heading; one appendix paragraph) that carry most "
          "of the leverage on the ln(words) coefficient. Their ratings: "
          + "; ".join(f"{k} " + ", ".join(f"{labels[y]} {v:.2f}" for y, v in d.items())
                      for k, d in X["dropped_ratings"].items()) + ". "
          + (f"Papers then left with one version leave the frame: {', '.join(X['papers_left_with_one_version'])}. "
             if X["papers_left_with_one_version"] else "")
          + f"Remaining: {X['n_versions']} versions of {X['n_papers']} papers.", "",
          "| arm | within-paper slope [95% CI] | with ln(words) covariate |", "|---|---|---|"]
    for y in ycols:
        L.append(f"| {labels[y]} | {fmt_ci(X[y]['fe_slope']['est'], *X[y]['fe_slope']['ci'])} "
                 f"| {fmt_ci(X[y]['fe_slope_len']['est'], *X[y]['fe_slope_len']['ci'])} |")
    for o in ((other, "ds3" if other == "ds0" else "ds0")):
        q = X["gpt_minus_" + o]
        L.append(f"| gpt minus {labels[o]} | {fmt_ci(q['fe_slope']['est'], *q['fe_slope']['ci'])} "
                 f"| {fmt_ci(q['fe_slope_len']['est'], *q['fe_slope_len']['ci'])} |")
    L += ["", "## Rating histograms (matched versions)", ""]
    for y in (("gpt", "ds0") if not three else ("ds0",)):
        L.append(f"- {labels[y]}: " + ", ".join(f"{k}: {v}" for k, v in A[y]["rating_hist"].items()))
    if three:
        L.append("- gpt-6.1-sol single reviews: see the per-rep check above.")
    Path(args.out).write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
