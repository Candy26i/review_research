"""Repeat-call noise of gpt-6.1-sol on ACL 2025: a second review of 500 random versions.

The main gpt-6.1-sol arm has one review per version (rep 0). For a random sample of 500
of its 3,618 scored versions (data/processed/acl_gpt61sol_retest_sample.jsonl: the set
random.Random(20261116).sample(sorted list of (arxiv_id, int v), 500), stored sorted), a
second review (rep 1) was requested with the same request settings. The same versions
already have three deepseek-chat reviews, so DeepSeek rep 0 vs rep 1 gives the
like-for-like noise comparison.

Per model, over the sampled versions with both reviews:
  - exact agreement P(r0 == r1), Cohen's kappa, agreement on rating >= 5, mean |r1 - r0|
  - repeat-call SD of one review: sigma = sqrt(mean((r1 - r0)^2) / 2)
  - ICC(2,1), two-way random, absolute agreement, k = 2 (same formula as analyze_scores.py;
    with mean r1 - r0 ~ 0 it equals ICC(1,1) to two decimals), and the Spearman-Brown
    reliability of a 3-review mean
CIs: cluster bootstrap over papers (2,000 draws, numpy Generator seed 42). Differences
gpt minus DeepSeek are paired (same versions, same draws).

Signal to noise: within-paper slope (paper fixed effects, all 1,374 papers, rep 0 of each
arm, as in analyze_acl_gpt_vs_deepseek.py) divided by sigma; joint cluster bootstrap over
all papers (papers outside the sample contribute to the slope only). DeepSeek sigma is the
like-for-like 500-sample r0 vs r1 value; the all-version 3-rep value is shown as a check.

Projection of the slope CI width with K reviews per version, two versions:
  - iid: sqrt((s2 - sigma^2 + sigma^2 / K) / s2), s2 = residual variance of the rep-0 FE fit
  - cluster-robust: only the noise part sigma^2 / Sxx of the cluster-robust slope variance
    shrinks by 1/K; the between-paper excess does not. This matches the cluster bootstrap
    that the reported CIs use, and is the one to quote.
Both are calibrated on DeepSeek, which has 3 reviews of every version: the realized ratio
(mean of 2 or 3 reviews vs a single review) is computed from the same bootstrap draws.

Run from the repo root:  python3 src/analysis/analyze_acl_gpt_retest.py
Outputs (OVERWRITTEN): results/acl_gpt61sol_retest.md and .json
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_acl_gpt_vs_deepseek import build_frame, ci, load_ratings, paper_stats, read_jsonl  # noqa: E402

N_BOOT = 2000
# rows 1-1046 of the gpt log were called with the first OpenAI API key (see the run notes)
FIRST_KEY_LAST_ROW = 1046


def icc21(mat: np.ndarray) -> float:
    n, k = mat.shape
    gm = mat.mean()
    sst = ((mat - gm) ** 2).sum()
    ssr = k * ((mat.mean(1) - gm) ** 2).sum()
    ssc = n * ((mat.mean(0) - gm) ** 2).sum()
    sse = sst - ssr - ssc
    msr, msc, mse = ssr / (n - 1), ssc / (k - 1), sse / ((n - 1) * (k - 1))
    return float((msr - mse) / (msr + (k - 1) * mse + k * (msc - mse) / n))


def kappa(a, b):
    po = float((a == b).mean())
    cats = np.union1d(a, b)
    pe = float(sum((a == c).mean() * (b == c).mean() for c in cats))
    return (po - pe) / (1 - pe)


def per_rep(path):
    """(arxiv_id, v) -> {rep: (rating, row_number, raw_path)} over parsed_ok rows."""
    out = defaultdict(dict)
    for i, r in enumerate(read_jsonl(path), start=1):
        if r.get("parsed_ok") and r.get("rating") is not None:
            k, rep = (str(r["arxiv_id"]), int(r["v"])), int(r["rep"])
            assert rep not in out[k], f"two parsed rows for {k} rep {rep} in {path}"
            out[k][rep] = (float(r["rating"]), i, r.get("raw_path"), r.get("request"))
    return out


def stats(a, b):
    d = b - a
    return {"n": int(a.size), "agree": float((d == 0).mean()), "kappa": kappa(a, b),
            "mean_abs": float(np.abs(d).mean()), "mean_diff": float(d.mean()),
            "agree_ge5": float(((a >= 5) == (b >= 5)).mean()),
            "sigma": float(math.sqrt((d ** 2).mean() / 2)),
            "icc": icc21(np.column_stack([a, b]))}


def fe_parts(df, y):
    """Within-paper slope, residual variance, iid and cluster-robust slope variance."""
    g = df.groupby("paper")
    xc = df["pos"] - g["pos"].transform("mean")
    yc = df[y] - g[y].transform("mean")
    sxx = float((xc * xc).sum())
    b = float((xc * yc).sum() / sxx)
    e = yc - b * xc
    s2 = float((e ** 2).sum() / (len(df) - df["paper"].nunique() - 1))
    u = (xc * e).groupby(df["paper"]).sum().to_numpy()
    return {"slope": b, "Sxx": sxx, "s2": s2, "v_iid": s2 / sxx, "v_cr": float((u ** 2).sum() / sxx ** 2)}


def projection(p, sig2, K):
    iid = math.sqrt(max(p["s2"] - sig2 + sig2 / K, 0) / p["s2"])
    cr = math.sqrt(max(p["v_cr"] - sig2 * (1 - 1 / K) / p["Sxx"], 0) / p["v_cr"])
    return iid, cr


def created(raw_path):
    try:
        return int(json.loads(Path(raw_path).read_text())["created"])
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--gpt-log", default="data/processed/acl_reviews_gpt61sol_log.jsonl")
    ap.add_argument("--ds-log", default="data/processed/acl_reviews_log.jsonl")
    ap.add_argument("--sample", default="data/processed/acl_gpt61sol_retest_sample.jsonl")
    ap.add_argument("--scrub", default="data/processed/acl_scrub_log.jsonl")
    ap.add_argument("--main-json", default="results/acl_gpt61sol_vs_deepseek.json")
    ap.add_argument("--out", default="results/acl_gpt61sol_retest.md")
    args = ap.parse_args()

    g, d = per_rep(args.gpt_log), per_rep(args.ds_log)
    sample = [(str(r["arxiv_id"]), int(r["v"])) for r in read_jsonl(args.sample)]
    both = [k for k in sample if 0 in g.get(k, {}) and 1 in g.get(k, {})
            and all(r in d.get(k, {}) for r in (0, 1, 2))]
    missing = [k for k in sample if k not in both]
    papers = np.array([k[0] for k in both])
    uniq, inv = np.unique(papers, return_inverse=True)
    G0 = np.array([g[k][0][0] for k in both]); G1 = np.array([g[k][1][0] for k in both])
    D0 = np.array([d[k][0][0] for k in both]); D1 = np.array([d[k][1][0] for k in both])
    D2 = np.array([d[k][2][0] for k in both])

    res = {"n_sample": len(sample), "n_used": len(both), "n_papers": int(uniq.size),
           "missing": [f"{a}v{v}" for a, v in missing]}

    # conditions of the two gpt reviews
    same_req = sum(1 for k in both if g[k][0][3] is not None and g[k][0][3] == g[k][1][3])
    first_key = sum(1 for k in both if g[k][0][1] <= FIRST_KEY_LAST_ROW)
    gaps, t1 = [], []
    for k in both:
        c0, c1 = created(g[k][0][2]), created(g[k][1][2])
        if c1 is not None:
            t1.append(c1)
        if c0 is not None and c1 is not None:
            gaps.append((c1 - c0) / 3600)
    fmt = lambda t: dt.datetime.fromtimestamp(t, dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    res["conditions"] = {"rep0_same_request_as_rep1": same_req, "rep0_first_api_key": first_key,
                         "rep1_first": fmt(min(t1)) if t1 else None,
                         "rep1_last": fmt(max(t1)) if t1 else None,
                         "gap_hours_min_median_max": [float(np.min(gaps)), float(np.median(gaps)),
                                                      float(np.max(gaps))] if gaps else None}

    res["gpt"] = stats(G0, G1)
    res["ds_r0_r1"] = stats(D0, D1)
    pairs = [stats(D0, D1), stats(D0, D2), stats(D1, D2)]
    res["ds_all_pairs_mean"] = {k: float(np.mean([p[k] for p in pairs])) for k in pairs[0]}
    res["ds_icc_3reps"] = icc21(np.column_stack([D0, D1, D2]))
    res["gpt_transitions"] = {f"{int(a)}->{int(b)}": n for (a, b), n in
                              sorted(Counter(zip(G0, G1)).items())}
    for name in ("gpt", "ds_r0_r1"):
        icc = res[name]["icc"]
        res[name]["icc_mean_of_3"] = 3 * icc / (1 + 2 * icc)
    res["ds_icc_mean_of_3_from_3reps"] = 3 * res["ds_icc_3reps"] / (1 + 2 * res["ds_icc_3reps"])

    # cluster bootstrap over sampled papers
    rng = np.random.default_rng(42)
    rows_of = [np.where(inv == i)[0] for i in range(uniq.size)]
    bs = defaultdict(list)
    for _ in range(N_BOOT):
        pick = rng.integers(0, uniq.size, uniq.size)
        ix = np.concatenate([rows_of[i] for i in pick])
        sg, sd = stats(G0[ix], G1[ix]), stats(D0[ix], D1[ix])
        for key in ("agree", "kappa", "agree_ge5", "sigma", "icc", "mean_diff"):
            bs["gpt_" + key].append(sg[key]); bs["ds_" + key].append(sd[key])
            bs["diff_" + key].append(sg[key] - sd[key])
        bs["ratio_sigma"].append(sg["sigma"] / sd["sigma"])
    res["ci"] = {k: ci(np.array(v)) for k, v in bs.items()}
    res["ratio_sigma"] = res["gpt"]["sigma"] / res["ds_r0_r1"]["sigma"]

    # ---------- whole-corpus frame (all papers, rep 0 of each arm) ----------
    gr0 = load_ratings(args.gpt_log, rep=0)
    dall = load_ratings(args.ds_log)                       # 3 reviews per version
    dr = {r: load_ratings(args.ds_log, rep=r) for r in (0, 1, 2)}
    d01 = {k: dr[0][k] + dr[1][k] for k in dr[0] if k in dr[1]}
    scrub = {str(r["key"]): r["n_words"] for r in read_jsonl(args.scrub)
             if isinstance(r.get("n_words"), (int, float))}
    df = build_frame({"gpt": gr0, "ds0": dr[0], "ds1": dr[1], "ds2": dr[2], "ds01": d01,
                      "ds3": dall}, scrub)
    df["ds_wvar"] = [float(np.var(dall[(p, v)], ddof=1)) for p, v in zip(df["paper"], df["v"])]
    ycols = ["gpt", "ds0", "ds1", "ds2", "ds01", "ds3"]
    st = paper_stats(df, ycols)
    all_papers = np.array(sorted(df["paper"].unique()))
    pos_of = {p: i for i, p in enumerate(all_papers)}
    pi = np.array([pos_of[p] for p in papers], dtype=int)
    n_s = np.bincount(pi, minlength=all_papers.size).astype(float)
    sq_g = np.bincount(pi, weights=(G1 - G0) ** 2, minlength=all_papers.size)
    sq_d = np.bincount(pi, weights=(D1 - D0) ** 2, minlength=all_papers.size)
    wv_d = df.groupby("paper", sort=True)["ds_wvar"].sum().to_numpy()
    nv = df.groupby("paper", sort=True).size().to_numpy().astype(float)
    Sxx = st["Sxx"]
    b = {y: st[y].sum() / Sxx.sum() for y in ycols}
    s_g = math.sqrt(sq_g.sum() / (2 * n_s.sum())); s_d = math.sqrt(sq_d.sum() / (2 * n_s.sum()))
    s_d_all = math.sqrt(wv_d.sum() / nv.sum())
    idx = np.random.default_rng(42).integers(0, all_papers.size, size=(N_BOOT, all_papers.size))
    bb = {y: st[y][idx].sum(1) / Sxx[idx].sum(1) for y in ycols}
    rg = bb["gpt"] / np.sqrt(sq_g[idx].sum(1) / (2 * n_s[idx].sum(1)))
    rd = bb["ds0"] / np.sqrt(sq_d[idx].sum(1) / (2 * n_s[idx].sum(1)))
    rd_all = bb["ds0"] / np.sqrt(wv_d[idx].sum(1) / nv[idx].sum(1))
    res["frame"] = {"n_papers": int(all_papers.size), "n_versions": int(len(df))}
    res["snr"] = {"gpt": {"slope": b["gpt"], "sigma": s_g, "ratio": b["gpt"] / s_g, "ci": ci(rg)},
                  "ds0": {"slope": b["ds0"], "sigma": s_d, "ratio": b["ds0"] / s_d, "ci": ci(rd)},
                  "diff": {"est": b["gpt"] / s_g - b["ds0"] / s_d, "ci": ci(rg - rd),
                           "share_draws_le0": float((rg - rd <= 0).mean())},
                  "ratio_gpt_over_ds": {"est": (b["gpt"] / s_g) / (b["ds0"] / s_d), "ci": ci(rg / rd)},
                  "ds0_sigma_all_versions": s_d_all,
                  "diff_with_ds_sigma_all_versions": {"est": b["gpt"] / s_g - b["ds0"] / s_d_all,
                                                      "ci": ci(rg - rd_all)}}

    # ---------- projection, calibrated on DeepSeek ----------
    fp = {y: fe_parts(df, y) for y in ycols}
    lo_s, hi_s = res["ci"]["gpt_sigma"]
    proj = {"gpt": {"s2": fp["gpt"]["s2"], "noise_var": s_g ** 2,
                    "noise_share": s_g ** 2 / fp["gpt"]["s2"],
                    "cr_over_iid_se": math.sqrt(fp["gpt"]["v_cr"] / fp["gpt"]["v_iid"])}}
    for K in (2, 3):
        iid, cr = projection(fp["gpt"], s_g ** 2, K)
        # range over the sigma CI (smaller sigma -> less to average away)
        rng_cr = sorted(projection(fp["gpt"], s ** 2, K)[1] for s in (lo_s, hi_s))
        rng_iid = sorted(projection(fp["gpt"], s ** 2, K)[0] for s in (lo_s, hi_s))
        proj["gpt"][f"K{K}"] = {"iid": iid, "cr": cr, "iid_range": rng_iid, "cr_range": rng_cr}
    ds = {"s2": fp["ds0"]["s2"], "sigma_sample": s_d, "sigma_all": s_d_all,
          "cr_over_iid_se": math.sqrt(fp["ds0"]["v_cr"] / fp["ds0"]["v_iid"])}
    for K, ycol in ((2, "ds01"), (3, "ds3")):
        realized = {"boot_sd_ratio": float(bb[ycol].std() / bb["ds0"].std()),
                    "ci_width_ratio": float(np.subtract(*ci(bb[ycol])[::-1]) / np.subtract(*ci(bb["ds0"])[::-1])),
                    "cr_se_ratio": math.sqrt(fp[ycol]["v_cr"] / fp["ds0"]["v_cr"])}
        if K == 3:   # realized vs each single review as the baseline
            realized["cr_se_ratio_vs_r0_r1_r2"] = [math.sqrt(fp["ds3"]["v_cr"] / fp[f"ds{r}"]["v_cr"])
                                                   for r in (0, 1, 2)]
        iid_s, cr_s = projection(fp["ds0"], s_d ** 2, K)
        iid_a, cr_a = projection(fp["ds0"], s_d_all ** 2, K)
        ds[f"K{K}"] = {"realized": realized, "proj_sigma_sample": {"iid": iid_s, "cr": cr_s},
                       "proj_sigma_all": {"iid": iid_a, "cr": cr_a}}
    proj["ds0"] = ds
    res["projection"] = proj

    try:
        main_res = json.loads(Path(args.main_json).read_text())
        res["identical_text_sigma"] = {"gpt": main_res["retest"]["gpt"], "ds0": main_res["retest"]["ds0"]}
        res["main_std_slope_diff"] = main_res["diff"]["gpt_minus_ds0"]["fe_slope_std"]
    except Exception:
        res["identical_text_sigma"] = None
        res["main_std_slope_diff"] = None

    Path(args.out).with_suffix(".json").write_text(json.dumps(res, indent=2))

    # ---------- markdown ----------
    C = res["ci"]
    f2 = lambda x: f"{x:.2f}"
    pc = lambda x: f"{100 * x:.1f}"
    cond = res["conditions"]
    gap = cond["gap_hours_min_median_max"]
    L = ["# gpt-6.1-sol repeat-call noise on ACL 2025 (500-version retest)", "",
         f"- sample: `{args.sample}`, {res['n_sample']} versions drawn at random from the 3,618 "
         f"scored versions; {res['n_used']} with a second gpt-6.1-sol review "
         f"({res['n_papers']} papers)" + (f"; missing: {', '.join(res['missing'])}" if missing else "") + ".",
         f"- gpt-6.1-sol: rep 0 (main run) vs rep 1 (retest, {cond['rep1_first']} to {cond['rep1_last']}; "
         f"{gap[0]:.0f}-{gap[2]:.0f} h after rep 0, median {gap[1]:.0f} h). Rep 1: flex tier, "
         "max_completion_tokens 16000, explicit cache mode, no reasoning_effort sent. "
         f"{same_req} of the {res['n_used']} rep-0 reviews had the identical request; the others are "
         f"default-tier smoke/pilot rows. {first_key} rep-0 reviews came from the first API key.",
         "- DeepSeek: rep 0 vs rep 1 of the same versions; the mean over its three rep pairs is "
         "shown for reference.",
         "- CIs: cluster bootstrap over papers, 2,000 draws.", "",
         "| | gpt-6.1-sol r0 vs r1 | DeepSeek r0 vs r1 | gpt minus DeepSeek | DeepSeek, mean of 3 pairs |",
         "|---|---|---|---|---|",
         f"| same rating twice | {pc(res['gpt']['agree'])}% [{pc(C['gpt_agree'][0])}, {pc(C['gpt_agree'][1])}] "
         f"| {pc(res['ds_r0_r1']['agree'])}% [{pc(C['ds_agree'][0])}, {pc(C['ds_agree'][1])}] "
         f"| {100 * (res['gpt']['agree'] - res['ds_r0_r1']['agree']):+.1f} [{100 * C['diff_agree'][0]:+.1f}, {100 * C['diff_agree'][1]:+.1f}] "
         f"| {pc(res['ds_all_pairs_mean']['agree'])}% |",
         f"| Cohen's kappa (chance-corrected) | {f2(res['gpt']['kappa'])} [{f2(C['gpt_kappa'][0])}, {f2(C['gpt_kappa'][1])}] "
         f"| {f2(res['ds_r0_r1']['kappa'])} [{f2(C['ds_kappa'][0])}, {f2(C['ds_kappa'][1])}] "
         f"| {res['gpt']['kappa'] - res['ds_r0_r1']['kappa']:+.2f} [{C['diff_kappa'][0]:+.2f}, {C['diff_kappa'][1]:+.2f}] "
         f"| {f2(res['ds_all_pairs_mean']['kappa'])} |",
         f"| same side of 5 (>= 5 or not; common cut for both scales) | {pc(res['gpt']['agree_ge5'])}% "
         f"| {pc(res['ds_r0_r1']['agree_ge5'])}% "
         f"| {100 * (res['gpt']['agree_ge5'] - res['ds_r0_r1']['agree_ge5']):+.1f} [{100 * C['diff_agree_ge5'][0]:+.1f}, {100 * C['diff_agree_ge5'][1]:+.1f}] "
         f"| {pc(res['ds_all_pairs_mean']['agree_ge5'])}% |",
         f"| mean absolute difference | {f2(res['gpt']['mean_abs'])} | {f2(res['ds_r0_r1']['mean_abs'])} | | {f2(res['ds_all_pairs_mean']['mean_abs'])} |",
         f"| mean difference r1 - r0 | {res['gpt']['mean_diff']:+.2f} [{C['gpt_mean_diff'][0]:+.2f}, {C['gpt_mean_diff'][1]:+.2f}] "
         f"| {res['ds_r0_r1']['mean_diff']:+.2f} [{C['ds_mean_diff'][0]:+.2f}, {C['ds_mean_diff'][1]:+.2f}] | | |",
         f"| repeat-call SD of one review | {f2(res['gpt']['sigma'])} [{f2(C['gpt_sigma'][0])}, {f2(C['gpt_sigma'][1])}] "
         f"| {f2(res['ds_r0_r1']['sigma'])} [{f2(C['ds_sigma'][0])}, {f2(C['ds_sigma'][1])}] "
         f"| {res['gpt']['sigma'] - res['ds_r0_r1']['sigma']:+.2f} [{C['diff_sigma'][0]:+.2f}, {C['diff_sigma'][1]:+.2f}]; "
         f"ratio {f2(res['ratio_sigma'])} [{f2(C['ratio_sigma'][0])}, {f2(C['ratio_sigma'][1])}] "
         f"| {f2(res['ds_all_pairs_mean']['sigma'])} |",
         f"| ICC(2,1), one review | {f2(res['gpt']['icc'])} [{f2(C['gpt_icc'][0])}, {f2(C['gpt_icc'][1])}] "
         f"| {f2(res['ds_r0_r1']['icc'])} [{f2(C['ds_icc'][0])}, {f2(C['ds_icc'][1])}] "
         f"| {res['gpt']['icc'] - res['ds_r0_r1']['icc']:+.2f} [{C['diff_icc'][0]:+.2f}, {C['diff_icc'][1]:+.2f}] "
         f"| ICC(2,1) over 3 reps {f2(res['ds_icc_3reps'])} |",
         f"| reliability of a 3-review mean (Spearman-Brown) | {f2(res['gpt']['icc_mean_of_3'])} "
         f"| {f2(res['ds_r0_r1']['icc_mean_of_3'])} | | {f2(res['ds_icc_mean_of_3_from_3reps'])} (from the 3-rep ICC) |",
         "",
         "ICC here is reliability across versions of different papers, so it is driven mostly by "
         "between-paper differences. The within-paper slope removes those; within papers, repeat-call "
         "noise is still most of the residual variation for both models (projection section).", ""]
    it = res["identical_text_sigma"]
    if it:
        L.append(f"Earlier indirect estimate of the repeat-call SD from 370 adjacent versions with identical "
                 f"text: gpt-6.1-sol {f2(it['gpt']['sigma'])} [{f2(it['gpt']['sigma_ci'][0])}, {f2(it['gpt']['sigma_ci'][1])}], "
                 f"DeepSeek r0 {f2(it['ds0']['sigma'])} [{f2(it['ds0']['sigma_ci'][0])}, {f2(it['ds0']['sigma_ci'][1])}]. "
                 "The direct retest agrees for both models; the two estimates share some rep-0 reviews "
                 "(versions in both sets), so they are not fully independent.")
        L.append("")
    L += ["gpt-6.1-sol rating transitions r0 -> r1: "
          + ", ".join(f"{k}: {v}" for k, v in res["gpt_transitions"].items()), "",
          "## Version slope relative to repeat-call noise", "",
          f"Slope = within-paper slope on all {res['frame']['n_papers']:,} papers (rep 0); SD from this "
          "retest (DeepSeek: same 500 versions, r0 vs r1). Joint cluster bootstrap over all papers.", "",
          "| | slope per step | repeat-call SD | slope / SD [95% CI] |", "|---|---|---|---|"]
    S = res["snr"]
    for name, lab in (("gpt", "gpt-6.1-sol"), ("ds0", "DeepSeek r0")):
        L.append(f"| {lab} | {S[name]['slope']:+.3f} | {f2(S[name]['sigma'])} | "
                 f"{S[name]['ratio']:+.3f} [{S[name]['ci'][0]:+.3f}, {S[name]['ci'][1]:+.3f}] |")
    L.append(f"| gpt minus DeepSeek | | | {S['diff']['est']:+.3f} [{S['diff']['ci'][0]:+.3f}, {S['diff']['ci'][1]:+.3f}] |")
    L.append(f"| gpt / DeepSeek | | | {S['ratio_gpt_over_ds']['est']:.2f} [{S['ratio_gpt_over_ds']['ci'][0]:.2f}, {S['ratio_gpt_over_ds']['ci'][1]:.2f}] |")
    L += ["", f"With DeepSeek's SD from all versions' three reviews ({f2(S['ds0_sigma_all_versions'])}) the "
          f"difference is {S['diff_with_ds_sigma_all_versions']['est']:+.3f} "
          f"[{S['diff_with_ds_sigma_all_versions']['ci'][0]:+.3f}, {S['diff_with_ds_sigma_all_versions']['ci'][1]:+.3f}]."]
    sd_diff = res["main_std_slope_diff"]
    if sd_diff:
        L.append(f"Scaled by each model's rating SD instead (main report), the difference is detected: "
                 f"{sd_diff['est']:+.3f} [{sd_diff['ci'][0]:+.3f}, {sd_diff['ci'][1]:+.3f}]. Whether gpt-6.1-sol "
                 "is \"less sensitive\" depends on the yardstick.")
    P = res["projection"]; PG, PD = P["gpt"], P["ds0"]
    L += ["", "## What more reviews per version would buy (projection)", "",
          "Ratio of the slope CI width with K reviews per version (averaged) to the width with one. "
          "\"iid\" shrinks the noise part of the residual variance; \"cluster-robust\" shrinks only the noise "
          "part of the cluster-robust slope variance (between-paper clustering does not average away) and "
          "matches the cluster-bootstrap CIs used throughout. DeepSeek has 3 reviews of every version, so "
          "its realized ratio checks the formulas.", "",
          "| | K | iid formula | cluster-robust formula | realized |", "|---|---|---|---|---|"]
    for K in (2, 3):
        q = PG[f"K{K}"]
        L.append(f"| gpt-6.1-sol | {K} | {100 * q['iid']:.0f}% (sigma CI: {100 * q['iid_range'][0]:.0f}-{100 * q['iid_range'][1]:.0f}%) "
                 f"| {100 * q['cr']:.0f}% ({100 * q['cr_range'][0]:.0f}-{100 * q['cr_range'][1]:.0f}%) | not observed |")
    for K in (2, 3):
        q = PD[f"K{K}"]; r = q["realized"]
        extra = ""
        if K == 3:
            extra = ("; vs r0/r1/r2 as baseline: " + "/".join(f"{100 * x:.0f}" for x in r["cr_se_ratio_vs_r0_r1_r2"]) + "%")
        L.append(f"| DeepSeek (vs r0) | {K} | {100 * q['proj_sigma_sample']['iid']:.0f}% "
                 f"| {100 * q['proj_sigma_sample']['cr']:.0f}% "
                 f"| CI width {100 * r['ci_width_ratio']:.0f}%, bootstrap SD {100 * r['boot_sd_ratio']:.0f}%, "
                 f"cluster-robust SE {100 * r['cr_se_ratio']:.0f}%{extra} |")
    L += ["", f"gpt-6.1-sol: residual variance {PG['s2']:.2f}, repeat-call noise {PG['noise_var']:.2f} "
          f"({100 * PG['noise_share']:.0f}% of it); cluster-robust SE / iid SE = {PG['cr_over_iid_se']:.2f} "
          f"(DeepSeek r0 {PD['cr_over_iid_se']:.2f}). The excess clustering is why the cluster-robust "
          "projection gains less. On DeepSeek the iid formula overstates the realized gain by a few points; "
          "expect the same or more for gpt-6.1-sol, whose sigma rests on few rating flips."]
    Path(args.out).write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
