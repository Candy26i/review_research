#!/usr/bin/env python3
"""E1-H audit: regression-based robustness of the version-slope comparison between arms.

Read-only. Usage:
    python3 regression.py --logs <main_log> <sub_log> [--repo <repo root>] [--boot 9999]

Rows: usable + de-duplicated exactly as the planned analysis defines them
  usable   = parsed_ok, and for harness_sub additionally process_ok is True and not
             reminded and not review_before_report ("clean"); --b1-all style variant is
             computed internally for the headline model.
  de-dup   = one row per key; a usable row is never replaced by a later failed row;
             several usable rows -> last one in file order.
Positions: 0-based index among the paper's versions that have a text file (manifest +
texts), never from which runs succeeded.

Sign convention everywhere: (second arm) minus (first arm), e.g. harness - direct.

Models (per pairwise comparison, on paper-versions usable in both arms, papers with at
least two such versions):
  M0   naive OLS  rating ~ arm * pos, iid errors ("independent sessions" as first described)
  P    planned estimator: Sxx-weighted pooled per-paper slope difference on cell means,
       sign-flip p, paper-bootstrap CI (re-implemented here to cross-check)
  M1c  FE regression on cell means (must equal P exactly)
  M1   FE regression on replicate rows: paper-by-arm intercepts + pos + pos:arm,
       cluster = paper; CR1 t(G-1), CR2 + Satterthwaite df, wild cluster bootstrap p
  M2   MixedLM rating ~ arm*pos, random intercept + random pos slope by paper,
       + paper-version variance component
  M2x  M2 + paper-specific random arm effect and random arm:pos slope (variance comps)
  M3   M1 with the rating recoded to its ordinal index 0..5
  M4a  first vs last common version only   M4b  position rescaled to 0..1 within paper
  M4c  calendar months since v1
  M5   leave-one-paper-out of P
  M6   level difference at the mean position (paper FE + arm + pos_c + arm:pos_c)
  M7   three-arm joint model on paper-versions usable in all three arms
"""
from __future__ import annotations

import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "1")          # keep CPU use light: one BLAS thread

import argparse
import json
import math
import warnings
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO_DEFAULT = "/Users/madili/Desktop/review papaer research"
ARMS = ("direct", "harness", "harness_sub")
COMPARISONS = (("direct", "harness", "PRIMARY"),
               ("direct", "harness_sub", "secondary"),
               ("harness", "harness_sub", "secondary"))
ORD = {1: 0, 3: 1, 5: 2, 6: 3, 8: 4, 10: 5}
SUMMARY: list[dict] = []       # rows of the final model x comparison table
PROBLEMS: list[str] = []       # modelling problems collected along the way


# ----------------------------------------------------------------------------- data
def read_log(path: str):
    rows, bad = [], 0
    for line in Path(path).read_text(encoding="utf-8").split("\n"):
        if line.strip():
            try:
                rows.append(json.loads(line))
            except ValueError:
                bad += 1
    return rows, bad


def usable(r: dict, b1_all: bool) -> bool:
    if not r.get("parsed_ok"):
        return False
    if r.get("arm") != "harness_sub":
        return True
    if r.get("process_ok") is not True:
        return False
    return b1_all or not (r.get("reminded") or r.get("review_before_report"))


def dedupe(raw: list[dict], b1_all: bool) -> list[dict]:
    latest: dict[str, dict] = {}
    for r in raw:
        if usable(r, b1_all) or not usable(latest.get(r["key"], {}), b1_all):
            latest[r["key"]] = r
    return [r for r in latest.values() if usable(r, b1_all)]


def load_design(manifest: Path, texts: Path):
    """planned positions, months since v1, per paper."""
    pos, months, nver = {}, {}, {}
    for line in manifest.read_text(encoding="utf-8").split("\n"):
        if not line.strip():
            continue
        p = json.loads(line)
        aid = p["arxiv_id"]
        sid = aid.replace("/", "_")
        allv = sorted(p["versions"], key=lambda d: d["v"])
        d0 = datetime.fromisoformat(allv[0]["date"])          # earliest listed version
        withtext = [d for d in allv if (texts / sid / f"v{d['v']}.md").exists()]
        pos[aid] = {d["v"]: i for i, d in enumerate(withtext)}
        months[aid] = {d["v"]: (datetime.fromisoformat(d["date"]) - d0).total_seconds()
                       / 86400.0 / 30.4375 for d in withtext}
        nver[aid] = len(withtext)
    return pos, months, nver


def build_frame(rows: list[dict], pos, months, nver) -> pd.DataFrame:
    recs = []
    for r in rows:
        aid, v = r["arxiv_id"], r["v"]
        if v not in pos.get(aid, {}):
            continue
        if r["rating"] not in ORD:
            PROBLEMS.append(f"rating {r['rating']} outside the scale in {r['key']} (row dropped)")
            continue
        k = nver[aid]
        recs.append(dict(paper=aid, v=v, pv=f"{aid}v{v}", arm=r["arm"], rep=r["rep"],
                         rating=float(r["rating"]), ordv=float(ORD[r["rating"]]),
                         pos=float(pos[aid][v]),
                         pos01=float(pos[aid][v]) / (k - 1) if k > 1 else 0.0,
                         months=float(months[aid][v])))
    return pd.DataFrame.from_records(recs)


def pair_frame(df: pd.DataFrame, arms: tuple[str, ...]) -> pd.DataFrame:
    """rows of the given arms, paper-versions present in all of them, papers with >= 2."""
    d = df[df["arm"].isin(arms)]
    n_arm = d.groupby("pv")["arm"].nunique()
    d = d[d["pv"].isin(n_arm[n_arm == len(arms)].index)]
    n_pv = d.groupby("paper")["pv"].nunique()
    d = d[d["paper"].isin(n_pv[n_pv >= 2].index)].copy()
    return d.sort_values(["paper", "v", "arm", "rep"]).reset_index(drop=True)


def cell_means(d: pd.DataFrame) -> pd.DataFrame:
    return (d.groupby(["paper", "v", "pv", "arm"], as_index=False)
             .agg(rating=("rating", "mean"), ordv=("ordv", "mean"), pos=("pos", "first"),
                  pos01=("pos01", "first"), months=("months", "first"),
                  nrep=("rating", "size")))


# ------------------------------------------------------------------ cluster-robust OLS
class ClusterOLS:
    """OLS with dummies kept in the design; CR1, CR2 (Bell-McCaffrey / Pustejovsky-Tipton,
    working covariance = identity, generalized inverse square root) and Satterthwaite df."""

    def __init__(self, X: np.ndarray, y: np.ndarray, cl: np.ndarray, k_focal: int):
        self.X, self.y = X, y
        self.n, self.p = X.shape
        self.rank = int(np.linalg.matrix_rank(X))
        self.M = np.linalg.pinv(X.T @ X)
        self.beta = self.M @ (X.T @ y)
        self.e = y - X @ self.beta
        codes, self.groups = pd.factorize(cl)
        self.codes = codes
        self.G = len(self.groups)
        self.idx = [np.flatnonzero(codes == g) for g in range(self.G)]
        meat1 = np.zeros((self.p, self.p))
        meat2 = np.zeros((self.p, self.p))
        self.Bg = []
        for ii in self.idx:
            Xg, eg = X[ii], self.e[ii]
            s = Xg.T @ eg
            meat1 += np.outer(s, s)
            w, U = np.linalg.eigh(np.eye(len(ii)) - Xg @ self.M @ Xg.T)
            isq = np.where(w > 1e-9, 1.0 / np.sqrt(np.clip(w, 1e-9, None)), 0.0)
            Ag = (U * isq) @ U.T
            u = Xg.T @ (Ag @ eg)
            meat2 += np.outer(u, u)
            self.Bg.append(Ag @ Xg @ self.M)
        # CR1 factor: dummies are nested in the clusters, so only focal regressors count
        c1 = self.G / (self.G - 1) * (self.n - 1) / (self.n - k_focal)
        self.V1 = c1 * self.M @ meat1 @ self.M
        self.V2 = self.M @ meat2 @ self.M
        dof = self.n - self.rank
        self.V0 = self.M * float(self.e @ self.e) / dof     # classical iid
        self.dof_iid = dof

    def satt_df(self, c: np.ndarray) -> float:
        XM = self.X @ self.M
        cols = []
        for ii, Bg in zip(self.idx, self.Bg):
            v = Bg @ c
            g = -XM @ (self.X[ii].T @ v)
            g[ii] += v
            cols.append(g)
        Gm = np.column_stack(cols)
        GG = Gm.T @ Gm
        return float(np.trace(GG) ** 2 / np.sum(GG * GG))

    def contrast(self, c: np.ndarray, kind: str = "CR2") -> dict:
        est = float(c @ self.beta)
        if kind == "CR2":
            se, df = math.sqrt(float(c @ self.V2 @ c)), self.satt_df(c)
        elif kind == "CR1":
            se, df = math.sqrt(float(c @ self.V1 @ c)), float(self.G - 1)
        else:
            se, df = math.sqrt(float(c @ self.V0 @ c)), float(self.dof_iid)
        t = est / se if se > 0 else float("nan")
        q = stats.t.ppf(0.975, df)
        return dict(est=est, se=se, df=df, t=t, p=float(2 * stats.t.sf(abs(t), df)),
                    lo=est - q * se, hi=est + q * se)


def wild_cluster_boot(X: np.ndarray, y: np.ndarray, codes: np.ndarray, test_cols: list[int],
                      B: int, seed: int) -> float:
    """Restricted wild cluster bootstrap (Rademacher) p-value for H0: beta[test_cols] = 0,
    studentized with the cluster sandwich. Works for one or several restrictions."""
    n, p = X.shape
    keep = [j for j in range(p) if j not in test_cols]
    Xr = X[:, keep]
    Ur, sr, _ = np.linalg.svd(Xr, full_matrices=False)
    Qr = Ur[:, sr > 1e-9 * sr.max()]
    er = y - Qr @ (Qr.T @ y)
    Xt = X[:, test_cols] - Qr @ (Qr.T @ X[:, test_cols])
    A = np.linalg.inv(Xt.T @ Xt)
    U, s, _ = np.linalg.svd(X, full_matrices=False)
    Q = U[:, s > 1e-9 * s.max()]
    G = int(codes.max()) + 1
    idx = [np.flatnonzero(codes == g) for g in range(G)]
    q = len(test_cols)

    def wald(Y: np.ndarray) -> np.ndarray:
        bet = A @ (Xt.T @ Y)                                   # q x B
        E = Y - Q @ (Q.T @ Y)
        S = np.stack([Xt[ii].T @ E[ii] for ii in idx])         # G x q x B
        meat = np.einsum("gib,gjb->ijb", S, S)
        V = np.einsum("ik,klb,lj->ijb", A, meat, A)
        if q == 1:
            return bet[0] ** 2 / V[0, 0]
        Vb = np.moveaxis(V, 2, 0)                              # B x q x q
        sol = np.linalg.solve(Vb, bet.T[:, :, None])[:, :, 0]
        return np.einsum("bi,bi->b", bet.T, sol)

    w_obs = wald(y[:, None])[0]
    rng = np.random.default_rng(seed)
    W = rng.choice([-1.0, 1.0], size=(G, B))
    w_star = wald(er[:, None] * W[codes])
    return float((np.sum(w_star >= w_obs - 1e-12) + 1) / (B + 1))


def dummies(*cols) -> np.ndarray:
    lab = pd.Series(["|".join(map(str, t)) for t in zip(*cols)])
    return pd.get_dummies(lab).to_numpy(dtype=float)


def fe_slope(d: pd.DataFrame, b_arm: str, ycol: str, xcol: str, B: int, seed: int) -> dict:
    """y ~ paper-by-arm intercepts + x + x:armB, clustered by paper."""
    armB = (d["arm"] == b_arm).to_numpy(dtype=float)
    x = d[xcol].to_numpy(dtype=float)
    Xf = np.column_stack([x, x * armB])
    X = np.hstack([Xf, dummies(d["paper"], d["arm"])])
    y = d[ycol].to_numpy(dtype=float)
    fit = ClusterOLS(X, y, d["paper"].to_numpy(), k_focal=2)
    c = np.zeros(X.shape[1]); c[1] = 1.0
    c0 = np.zeros(X.shape[1]); c0[0] = 1.0
    out = dict(cr2=fit.contrast(c, "CR2"), cr1=fit.contrast(c, "CR1"), iid=fit.contrast(c, "iid"),
               slope_a=fit.contrast(c0, "CR2"), slope_b=fit.contrast(c0 + c, "CR2"),
               n=fit.n, G=fit.G, rank=fit.rank, p_cols=fit.p)
    out["wcb_p"] = wild_cluster_boot(X, y, fit.codes, [1], B, seed)
    if fit.rank < fit.p:
        PROBLEMS.append(f"FE design rank-deficient ({fit.rank} < {fit.p}) for {ycol}~{xcol}")
    return out


def fe_level(d: pd.DataFrame, b_arm: str, ycol: str, B: int, seed: int) -> dict:
    """y ~ paper intercepts + armB + pos_c + armB:pos_c ; armB = level difference at the
    mean position (mean over the common paper-versions)."""
    armB = (d["arm"] == b_arm).to_numpy(dtype=float)
    mpos = d.drop_duplicates("pv")["pos"].mean()
    xc = d["pos"].to_numpy(dtype=float) - mpos
    Xf = np.column_stack([armB, xc, xc * armB])
    X = np.hstack([Xf, dummies(d["paper"])])
    y = d[ycol].to_numpy(dtype=float)
    fit = ClusterOLS(X, y, d["paper"].to_numpy(), k_focal=3)
    c = np.zeros(X.shape[1]); c[0] = 1.0
    out = dict(cr2=fit.contrast(c, "CR2"), cr1=fit.contrast(c, "CR1"), mean_pos=float(mpos),
               n=fit.n, G=fit.G)
    out["wcb_p"] = wild_cluster_boot(X, y, fit.codes, [0], B, seed)
    return out


# ----------------------------------------------------------------- planned estimator
def per_paper(cm: pd.DataFrame, a: str, b: str, ycol: str = "rating", xcol: str = "pos"):
    """per-paper within-arm slopes on cell means, Sxx, levels."""
    recs = []
    for aid, g in cm.groupby("paper", sort=True):
        ga = g[g["arm"] == a].sort_values("v")
        gb = g[g["arm"] == b].sort_values("v")
        x = ga[xcol].to_numpy(dtype=float)
        xc = x - x.mean()
        sxx = float((xc ** 2).sum())
        ya, yb = ga[ycol].to_numpy(dtype=float), gb[ycol].to_numpy(dtype=float)
        sa = float((xc * (ya - ya.mean())).sum() / sxx)
        sb = float((xc * (yb - yb.mean())).sum() / sxx)
        steps = float(x[-1] - x[0])
        recs.append(dict(paper=aid, nver=len(x), sxx=sxx, slope_a=sa, slope_b=sb,
                         diff=round(sb - sa, 9), level_a=float(ya.mean()), level_b=float(yb.mean()),
                         steps=steps, fl_a=float(ya[-1] - ya[0]), fl_b=float(yb[-1] - yb[0])))
    return pd.DataFrame.from_records(recs)


def pooled(v, w) -> float:
    v, w = np.asarray(v, float), np.asarray(w, float)
    return float((v * w).sum() / w.sum())


def boot_ci(v, w, n_boot=10000, seed=42):
    v, w = np.asarray(v, float), np.asarray(w, float)
    rng = np.random.default_rng(seed)
    ix = rng.integers(0, len(v), size=(n_boot, len(v)))
    est = (w[ix] * v[ix]).sum(axis=1) / w[ix].sum(axis=1)
    return float(np.percentile(est, 2.5)), float(np.percentile(est, 97.5))


def flip_p(v, w, n_flip=20000, seed=42) -> float:
    wd = np.asarray(w, float) * np.asarray(v, float)
    obs = abs(wd.sum())
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(n_flip, len(wd)))
    return float(((np.abs((signs * wd).sum(axis=1)) >= obs - 1e-12).sum() + 1) / (n_flip + 1))


# ------------------------------------------------------------------------ mixed model
def fit_mixed(d: pd.DataFrame, formula: str, vc: dict | None, label: str):
    """MixedLM with random intercept + random pos slope by paper (+ variance components).
    Tries several optimizers; returns (result or None, info dict)."""
    import statsmodels.formula.api as smf
    info = dict(label=label, converged=False, method=None, warnings=[], singular=[], attempts=[])
    best = None
    for method in ("lbfgs", "powell", "nm"):
        with warnings.catch_warnings(record=True) as wl:
            warnings.simplefilter("always")
            try:
                m = smf.mixedlm(formula, d, groups=d["paper"], re_formula="~pos",
                                vc_formula=vc)
                r = m.fit(reml=True, method=[method], maxiter=5000)
            except Exception as ex:                      # noqa: BLE001
                info["attempts"].append(f"{method}: {type(ex).__name__}")
                info["warnings"].append(f"{method}: {type(ex).__name__}: {ex}")
                continue
        msgs = sorted({str(w.message).split("\n")[0][:110] for w in wl})
        info["attempts"].append(f"{method}: {'converged' if r.converged else 'NOT converged'}, "
                                f"REML llf {r.llf:.3f}")
        info.setdefault("llfs", []).append((method, bool(r.converged), float(r.llf)))
        better = (best is None or (r.converged and not best[0].converged)
                  or (r.converged == best[0].converged and r.llf > best[0].llf + 1e-4))
        if better:
            best = (r, method, msgs)
    if best is None:
        PROBLEMS.append(f"{label}: MixedLM could not be fitted ({'; '.join(info['warnings'])})")
        return None, info
    if any("NOT converged" in a_ for a_ in info["attempts"]):
        PROBLEMS.append(f"{label}: optimizer non-convergence ({'; '.join(info['attempts'])}); "
                        f"reported fit = {best[1]}")
    worse = [f"{m_} {l_:.3f}" for m_, c_, l_ in info.get("llfs", []) if c_ and l_ < best[0].llf - 0.05]
    if worse:
        PROBLEMS.append(f"{label}: an optimizer reported convergence at an inferior optimum ({', '.join(worse)} "
                        f"vs best {best[0].llf:.3f} from {best[1]}); reported fit = {best[1]}")
    r, method, msgs = best
    info.update(converged=bool(r.converged), method=method)
    info["warnings"] += msgs
    cov_re = np.asarray(r.cov_re, dtype=float)
    ev = np.linalg.eigvalsh(cov_re)
    sd = np.sqrt(np.clip(np.diag(cov_re), 0, None))
    corr = cov_re[0, 1] / (sd[0] * sd[1]) if sd[0] > 0 and sd[1] > 0 else float("nan")
    if ev.min() < 1e-6 * max(ev.max(), 1e-12) or not np.isfinite(corr) or abs(corr) > 0.995:
        info["singular"].append(f"random intercept/slope covariance is (near) singular "
                                f"(sd int {sd[0]:.3f}, sd slope {sd[1]:.3f}, corr {corr:+.3f})")
    names = list(getattr(r.model, "exog_vc").names) if vc else []
    for nm, val in zip(names, np.atleast_1d(r.vcomp)):
        if val < 1e-6:
            info["singular"].append(f"variance component '{nm}' estimated at zero")
    info["re"] = dict(sd_int=float(sd[0]), sd_slope=float(sd[1]), corr=float(corr),
                      vcomp={nm: float(v) for nm, v in zip(names, np.atleast_1d(r.vcomp))},
                      resid_sd=float(math.sqrt(r.scale)))
    if not r.converged:
        PROBLEMS.append(f"{label}: MixedLM did NOT converge (best optimizer {method})")
    for s in info["singular"]:
        PROBLEMS.append(f"{label}: {s}")
    for w in info["warnings"]:
        if "boundary" in w or "Hessian" in w or "singular" in w.lower():
            PROBLEMS.append(f"{label}: statsmodels warning (reported fit): {w}")
    return r, info


def mixed_contrast(r, weights: dict) -> dict:
    names = list(r.params.index)
    L = np.zeros(len(names))
    for k, w in weights.items():
        L[names.index(k)] = w
    cov = np.asarray(r.cov_params(), dtype=float)
    est = float(L @ np.asarray(r.params))
    se = math.sqrt(float(L @ cov @ L))
    z = est / se
    return dict(est=est, se=se, t=z, p=float(2 * stats.norm.sf(abs(z))), lo=est - 1.96 * se,
                hi=est + 1.96 * se, df=float("inf"))


# -------------------------------------------------------------------------- printing
def f3(x) -> str:
    return "nan" if x is None or not np.isfinite(x) else f"{x:+.3f}"


def line(tag: str, c: dict, extra: str = "") -> str:
    dfs = "z" if not np.isfinite(c.get("df", float("inf"))) else f"t(df={c['df']:.1f})"
    return (f"  {tag:<46s} est {f3(c['est'])}  SE {c['se']:.3f}  95% CI [{f3(c['lo'])}, "
            f"{f3(c['hi'])}]  {dfs}  p = {c['p']:.4f}{extra}")


def add_summary(model: str, comp: str, c: dict | None, note: str = "", alt: dict | None = None):
    """alt = other p-values computed for the same estimate (name -> p); they enter the
    table note and the verdict check."""
    if alt:
        extra = "; ".join(f"{k} p {v:.3f}" for k, v in alt.items())
        note = f"{note}; {extra}" if note else extra
    SUMMARY.append(dict(model=model, comp=comp, c=c, note=note, alt=alt or {}))


# ---------------------------------------------------------------------- per comparison
def run_pair(df: pd.DataFrame, a: str, b: str, role: str, B: int, tag: str = "",
             headline_only: bool = False) -> None:
    comp = f"{b} - {a}"
    d = pair_frame(df, (a, b))
    cm = cell_means(d)
    n_pv, n_p = d["pv"].nunique(), d["paper"].nunique()
    print(f"\n{'=' * 100}\n{role}{tag}: {comp}\n{'=' * 100}")
    reps = Counter(zip(cm[cm.arm == a].sort_values("pv")["nrep"], cm[cm.arm == b].sort_values("pv")["nrep"]))
    print(f"  rows {len(d)} ({a} {int((d.arm == a).sum())}, {b} {int((d.arm == b).sum())}) | "
          f"paper-versions in both arms {n_pv} | papers (clusters) {n_p}")
    print(f"  replicates per cell ({a},{b}): "
          + ", ".join(f"{k}:{v}" for k, v in sorted(reps.items(), reverse=True)))

    # ---------------- M1 headline
    m1 = fe_slope(d, b, "rating", "pos", B, seed=101)
    print("\n  [M1] FE regression on replicate rows: rating ~ paper x arm intercepts + pos + pos:arm"
          f"  (n={m1['n']}, clusters={m1['G']}, design rank {m1['rank']}/{m1['p_cols']})")
    print(line(f"slope {a} (CR2)", m1["slope_a"]))
    print(line(f"slope {b} (CR2)", m1["slope_b"]))
    print(line("pos:arm  CR2 + Satterthwaite", m1["cr2"]))
    print(line("pos:arm  CR1, t(G-1)", m1["cr1"]))
    print(line("pos:arm  iid errors (for reference only)", m1["iid"]))
    print(f"  {'pos:arm  wild cluster bootstrap (restricted)':<46s} p = {m1['wcb_p']:.4f}  ({B} draws)")
    add_summary("M1 FE replicate rows, cluster=paper (CR2)" + tag, comp, m1["cr2"],
                alt={"wild-boot": m1["wcb_p"], "CR1": m1["cr1"]["p"]})
    if headline_only:
        pp = per_paper(cm, a, b)
        est = pooled(pp["diff"], pp["sxx"])
        lo, hi = boot_ci(pp["diff"], pp["sxx"])
        p = flip_p(pp["diff"], pp["sxx"])
        print(f"  [P] planned pooled Sxx-weighted difference: {f3(est)} [{f3(lo)}, {f3(hi)}], "
              f"sign-flip p = {p:.4f}")
        add_summary("P planned pooled Sxx-weighted (cell means), sign-flip" + tag, comp,
                    dict(est=est, lo=lo, hi=hi, p=p, se=float("nan")))
        return

    # ---------------- M0 naive
    armB = (d["arm"] == b).to_numpy(dtype=float)
    x = d["pos"].to_numpy(dtype=float)
    X0 = np.column_stack([np.ones(len(d)), armB, x, x * armB])
    f0 = ClusterOLS(X0, d["rating"].to_numpy(dtype=float), d["paper"].to_numpy(), k_focal=4)
    c0 = np.array([0, 0, 0, 1.0])
    m0_iid, m0_cr2 = f0.contrast(c0, "iid"), f0.contrast(c0, "CR2")
    print("\n  [M0] naive OLS rating ~ arm * pos, no paper effects")
    print(line("arm:pos  iid ('independent sessions')", m0_iid))
    print(line("arm:pos  same estimate, CR2 cluster=paper", m0_cr2))
    add_summary("M0 naive OLS arm*pos, iid errors", comp, m0_iid)
    add_summary("M0 naive OLS arm*pos, cluster=paper (CR2)", comp, m0_cr2)

    # ---------------- P planned + M1c
    pp = per_paper(cm, a, b)
    est = pooled(pp["diff"], pp["sxx"])
    lo, hi = boot_ci(pp["diff"], pp["sxx"])
    p = flip_p(pp["diff"], pp["sxx"])
    print("\n  [P] planned estimator (cell means, Sxx-weighted pooled per-paper slope difference)")
    print(f"  {'pooled slope ' + a:<46s} est {f3(pooled(pp['slope_a'], pp['sxx']))}")
    print(f"  {'pooled slope ' + b:<46s} est {f3(pooled(pp['slope_b'], pp['sxx']))}")
    print(f"  {'pooled difference':<46s} est {f3(est)}  paper-bootstrap 95% CI [{f3(lo)}, {f3(hi)}]"
          f"  sign-flip p = {p:.4f}")
    add_summary("P planned pooled Sxx-weighted (cell means), sign-flip", comp,
                dict(est=est, lo=lo, hi=hi, p=p, se=float("nan")))
    m1c = fe_slope(cm, b, "rating", "pos", B, seed=102)
    print(line("[M1c] FE regression on cell means (CR2)", m1c["cr2"],
               extra=f"   wild-boot p = {m1c['wcb_p']:.4f}"))
    print(f"  {'':<46s} |M1c - P| = {abs(m1c['cr2']['est'] - est):.2e}  (must be ~0)")
    if abs(m1c["cr2"]["est"] - est) > 1e-8:
        PROBLEMS.append(f"{comp}: cell-mean FE regression does not reproduce the planned pooled estimate")
    add_summary("M1c FE cell means, cluster=paper (CR2)", comp, m1c["cr2"],
                alt={"wild-boot": m1c["wcb_p"], "CR1": m1c["cr1"]["p"]})
    un = stats.ttest_1samp(pp["diff"], 0.0)
    ulo, uhi = boot_ci(pp["diff"], np.ones(len(pp)))
    print(f"  {'unweighted mean of per-paper differences':<46s} est {f3(pp['diff'].mean())}  "
          f"bootstrap 95% CI [{f3(ulo)}, {f3(uhi)}]  paired t({len(pp) - 1}) p = {un.pvalue:.4f}  "
          f"sign-flip p = {flip_p(pp['diff'], np.ones(len(pp))):.4f}")
    se_u = pp["diff"].std(ddof=1) / math.sqrt(len(pp))
    q = stats.t.ppf(0.975, len(pp) - 1)
    add_summary("P2 unweighted mean of per-paper slope diffs, paired t", comp,
                dict(est=float(pp["diff"].mean()), se=se_u, lo=pp["diff"].mean() - q * se_u,
                     hi=pp["diff"].mean() + q * se_u, p=float(un.pvalue)))

    # ---------------- M2 mixed
    dd = d.copy()
    dd["armB"] = (dd["arm"] == b).astype(float)
    dd["armB_pos"] = dd["armB"] * dd["pos"]
    mpos = float(d.drop_duplicates("pv")["pos"].mean())
    print("\n  [M2] MixedLM (REML): rating ~ arm * pos, random intercept + random pos slope by paper")
    variants = (("M2 mixed: RI+RS(pos) by paper + paper-version VC", {"pv": "0 + C(pv)"}),
                ("M2x mixed: M2 + random arm and arm:pos by paper", {"pv": "0 + C(pv)", "arm_paper": "0 + armB",
                                                                    "armpos_paper": "0 + armB_pos"}),
                ("M2- mixed: RI+RS(pos) by paper only", None))
    mixed_level = []
    for name, vc in variants:
        r, info = fit_mixed(dd, "rating ~ armB + pos + armB:pos", vc, f"{comp} {name.split(':')[0]}{tag}")
        if r is None:
            print(f"  {name}: could not be fitted")
            add_summary(name, comp, None, note="not fitted")
            continue
        c = mixed_contrast(r, {"armB:pos": 1.0})
        re = info["re"]
        status = ("converged" if info["converged"] else "NOT CONVERGED") + f" ({info['method']})"
        if info["singular"]:
            status += "; SINGULAR: " + "; ".join(info["singular"])
        print(line(name.split(":")[0] + " arm:pos (Wald z)", c))
        print(f"      {name.split(': ')[1]} -> {status}")
        print(f"      sd(intercept) {re['sd_int']:.3f}, sd(pos slope) {re['sd_slope']:.3f}, corr "
              f"{re['corr']:+.2f}, variance comps {', '.join(f'{k}={v:.4f}' for k, v in re['vcomp'].items()) or '-'}, "
              f"residual sd {re['resid_sd']:.3f}")
        print("      optimizers: " + "; ".join(info["attempts"]))
        if info["warnings"]:
            print("      warnings: " + " | ".join(info["warnings"]))
        note = f"{info['method']} converged" if info["converged"] else "NOT converged"
        if any("NOT converged" in a_ for a_ in info["attempts"]):
            note += ", default lbfgs failed" if "lbfgs: NOT" in info["attempts"][0] else ", some optimizers failed"
        if info["singular"]:
            note += ", singular"
        add_summary(name, comp, c, note=note)
        if not name.startswith("M2-"):
            lv = mixed_contrast(r, {"armB": 1.0, "armB:pos": mpos})
            mixed_level.append((name.split(":")[0], lv, note))

    # ---------------- M3 ordinal
    m3 = fe_slope(d, b, "ordv", "pos", B, seed=103)
    print("\n  [M3] as M1 but rating recoded to ordinal index (1,3,5,6,8,10 -> 0..5); units = index steps")
    print(line("pos:arm  CR2", m3["cr2"], extra=f"   wild-boot p = {m3['wcb_p']:.4f}"))
    cmo = cm.assign(rating=cm["ordv"])
    ppo = per_paper(cmo, a, b)
    m3_flip = flip_p(ppo["diff"], ppo["sxx"])
    print(f"  {'planned-style pooled on cell means':<46s} est {f3(pooled(ppo['diff'], ppo['sxx']))}  "
          f"sign-flip p = {m3_flip:.4f}   (CR1 p = {m3['cr1']['p']:.4f})")
    add_summary("M3 ordinal index 0..5, FE (CR2) [index units]", comp, m3["cr2"],
                alt={"wild-boot": m3["wcb_p"], "CR1": m3["cr1"]["p"], "cell-mean sign-flip": m3_flip})
    print("  [M3b] as M1 with a binary outcome 1{rating >= c} (linear probability; units = probability per step)")
    for thr in (5, 6, 8):
        dt = d.assign(thr=(d["rating"] >= thr).astype(float))
        if dt["thr"].nunique() < 2:
            continue
        mt = fe_slope(dt, b, "thr", "pos", B, seed=110 + thr)
        share = dt.groupby("arm")["thr"].mean()
        cmt = (dt.groupby(["paper", "v", "pv", "arm"], as_index=False)
                 .agg(rating=("thr", "mean"), pos=("pos", "first")))
        ppt = per_paper(cmt, a, b)
        pt_est, pt_flip = pooled(ppt["diff"], ppt["sxx"]), flip_p(ppt["diff"], ppt["sxx"])
        print(line(f"rating >= {thr}  pos:arm CR2", mt["cr2"],
                   extra=f"   wild-boot p = {mt['wcb_p']:.4f}   CR1 p = {mt['cr1']['p']:.4f}"))
        print(f"      share with rating >= {thr}: {a} {share[a]:.2f}, {b} {share[b]:.2f}; slope per step {a} "
              f"{f3(mt['slope_a']['est'])}, {b} {f3(mt['slope_b']['est'])}; planned-style pooled on cell means "
              f"{f3(pt_est)}, sign-flip p = {pt_flip:.4f}")
        add_summary(f"M3b P(rating >= {thr}), FE LPM (CR2) [probability per step]", comp, mt["cr2"],
                    alt={"wild-boot": mt["wcb_p"], "CR1": mt["cr1"]["p"], "cell-mean sign-flip": pt_flip})

    # ---------------- M4 position coding
    print("\n  [M4] position-coding sensitivity")
    fl = pp.copy()
    fl["d_step"] = (fl["fl_b"] - fl["fl_a"]) / fl["steps"]
    fl["d_tot"] = fl["fl_b"] - fl["fl_a"]
    for col, lab, key in (("d_step", "(a) first vs last, per step: mean paired diff", "M4a first-vs-last per step, paired t"),
                          ("d_tot", "(a') first vs last, total change: mean paired diff", "M4a' first-vs-last total change, paired t [points]")):
        v = fl[col].to_numpy()
        se = v.std(ddof=1) / math.sqrt(len(v))
        q = stats.t.ppf(0.975, len(v) - 1)
        t = stats.ttest_1samp(v, 0.0)
        c = dict(est=float(v.mean()), se=se, lo=v.mean() - q * se, hi=v.mean() + q * se,
                 p=float(t.pvalue), df=float(len(v) - 1))
        print(line(lab, c, extra=f"   sign-flip p = {flip_p(v, np.ones(len(v))):.4f}"))
        add_summary(key, comp, c, alt={"sign-flip": flip_p(v, np.ones(len(v)))})
    print(f"      mean (last-first)/steps: {a} {f3((fl['fl_a'] / fl['steps']).mean())}, "
          f"{b} {f3((fl['fl_b'] / fl['steps']).mean())}; step-weighted (sum change / sum steps): "
          f"{a} {f3(fl['fl_a'].sum() / fl['steps'].sum())}, {b} {f3(fl['fl_b'].sum() / fl['steps'].sum())}, "
          f"diff {f3((fl['fl_b'].sum() - fl['fl_a'].sum()) / fl['steps'].sum())}")
    m4b = fe_slope(d, b, "rating", "pos01", B, seed=104)
    print(line("(b) pos rescaled 0..1 within paper, FE CR2", m4b["cr2"], extra=f"   wild-boot p = {m4b['wcb_p']:.4f}"))
    print(f"      slopes per full history: {a} {f3(m4b['slope_a']['est'])}, {b} {f3(m4b['slope_b']['est'])}")
    add_summary("M4b position 0..1 within paper, FE (CR2) [points per full history]", comp, m4b["cr2"],
                alt={"wild-boot": m4b["wcb_p"], "CR1": m4b["cr1"]["p"]})
    m4c = fe_slope(d, b, "rating", "months", B, seed=105)
    print(line("(c) months since v1, FE CR2 (per month)", m4c["cr2"], extra=f"   wild-boot p = {m4c['wcb_p']:.4f}"))
    print(f"      slopes per month: {a} {f3(m4c['slope_a']['est'])}, {b} {f3(m4c['slope_b']['est'])}; "
          f"months range over rows 0..{d['months'].max():.1f}, median gap first->last "
          f"{d.groupby('paper')['months'].agg(lambda s: s.max() - s.min()).median():.1f}")
    add_summary("M4c calendar months since v1, FE (CR2) [points per month]", comp, m4c["cr2"],
                alt={"wild-boot": m4c["wcb_p"], "CR1": m4c["cr1"]["p"]})

    # ---------------- M5 leave one paper out
    print("\n  [M5] leave-one-paper-out of the planned pooled estimate")
    w = pp["sxx"].to_numpy(); v = pp["diff"].to_numpy()
    loo = np.array([pooled(np.delete(v, i), np.delete(w, i)) for i in range(len(v))])
    pp["loo"] = loo
    pp["share"] = w / w.sum()
    pp["shift"] = loo - est
    print(f"      full estimate {f3(est)}; leave-one-out range [{f3(loo.min())}, {f3(loo.max())}]; "
          f"effective number of papers (sum w)^2 / sum w^2 = {w.sum() ** 2 / (w ** 2).sum():.1f} of {len(w)}")
    for _, r_ in pp.sort_values("share", ascending=False).head(5).iterrows():
        print(f"      weight {r_['share']:.1%}  {r_['paper']:<12s} versions {int(r_['nver'])}  slope {a} "
              f"{f3(r_['slope_a'])}, {b} {f3(r_['slope_b'])}, diff {f3(r_['diff'])}")
    print("      largest shifts when one paper is dropped:")
    top = pp.reindex(pp["shift"].abs().sort_values(ascending=False).index).head(6)
    for _, r_ in top.iterrows():
        keep = pp["paper"] != r_["paper"]
        print(f"      drop {r_['paper']:<12s} versions {int(r_['nver'])}  weight {r_['share']:.1%}  own diff "
              f"{f3(r_['diff'])}  -> estimate {f3(r_['loo'])} (shift {f3(r_['shift'])}), "
              f"sign-flip p {flip_p(v[keep.to_numpy()], w[keep.to_numpy()]):.3f}")
    heavy = pp.sort_values("share", ascending=False).head(3)
    keep = ~pp["paper"].isin(heavy["paper"])
    print(f"      3 heaviest papers ({', '.join(heavy['paper'])}; weight {heavy['share'].sum():.0%}): "
          f"dropping all three -> {f3(pooled(v[keep.to_numpy()], w[keep.to_numpy()]))} "
          f"(sign-flip p {flip_p(v[keep.to_numpy()], w[keep.to_numpy()]):.3f}); "
          f"only these three -> {f3(pooled(heavy['diff'], heavy['sxx']))}")
    for lab, sel in (("two-version papers", pp["nver"] == 2), ("papers with 3+ versions", pp["nver"] >= 3)):
        if sel.sum() >= 2:
            print(f"      {lab} (n={int(sel.sum())}, weight {pp.loc[sel, 'share'].sum():.0%}): pooled "
                  f"{f3(pooled(pp.loc[sel, 'diff'], pp.loc[sel, 'sxx']))}, unweighted {f3(pp.loc[sel, 'diff'].mean())}")
    add_summary("M5 leave-one-paper-out range of P", comp,
                dict(est=est, lo=float(loo.min()), hi=float(loo.max()), p=float("nan"), se=float("nan")),
                note="lo/hi = LOO min/max, not a CI")

    # ---------------- M6 level
    print("\n  [M6] level difference at the mean position")
    lv = fe_level(d, b, "rating", B, seed=106)
    print(line(f"FE replicate rows (pos mean {lv['mean_pos']:.2f}) CR2", lv["cr2"], extra=f"   wild-boot p = {lv['wcb_p']:.4f}"))
    lvc = fe_level(cm, b, "rating", B, seed=107)
    print(line("FE cell means CR2", lvc["cr2"], extra=f"   wild-boot p = {lvc['wcb_p']:.4f}"))
    ld = (pp["level_b"] - pp["level_a"]).to_numpy()
    print(f"  {'planned: mean over papers of level diff':<46s} est {f3(ld.mean())}  sign-flip p = "
          f"{flip_p(ld, np.ones(len(ld))):.4f}")
    for nm_, lv_, note_ in mixed_level:
        print(line(f"{nm_}, arm effect at mean pos (Wald z)", lv_, extra=f"   ({note_})"))
    lo3 = fe_level(d, b, "ordv", B, seed=108)
    print(line("FE replicate rows, ordinal index units CR2", lo3["cr2"]))
    SUMMARY.append(dict(model="LEVEL M6 FE replicate rows at mean pos (CR2)", comp=comp, c=lv["cr2"],
                        note=f"wild-boot p {lv['wcb_p']:.4f}", alt={}, level=True))
    SUMMARY.append(dict(model="LEVEL M6 FE cell means at mean pos (CR2)", comp=comp, c=lvc["cr2"],
                        note=f"wild-boot p {lvc['wcb_p']:.4f}", alt={}, level=True))
    for nm_, lv_, note_ in mixed_level:
        SUMMARY.append(dict(model=f"LEVEL {nm_} at mean pos (Wald z)", comp=comp,
                            c=lv_, note=note_, alt={}, level=True))


# ------------------------------------------------------------------------- three arms
def run_three(df: pd.DataFrame, B: int) -> None:
    d = pair_frame(df, ARMS)
    print(f"\n{'=' * 100}\n[M7] THREE-ARM JOINT MODEL on paper-versions usable in all three arms\n{'=' * 100}")
    print(f"  rows {len(d)} ({', '.join(f'{a} {int((d.arm == a).sum())}' for a in ARMS)}) | paper-versions "
          f"{d['pv'].nunique()} | papers {d['paper'].nunique()}")
    for ycol, lab in (("rating", "rating points"), ("ordv", "ordinal index")):
        h0 = (d["arm"] == "harness").to_numpy(dtype=float)
        h1 = (d["arm"] == "harness_sub").to_numpy(dtype=float)
        x = d["pos"].to_numpy(dtype=float)
        Xf = np.column_stack([x, x * h0, x * h1])
        X = np.hstack([Xf, dummies(d["paper"], d["arm"])])
        y = d[ycol].to_numpy(dtype=float)
        fit = ClusterOLS(X, y, d["paper"].to_numpy(), k_focal=3)
        e = np.eye(X.shape[1])
        print(f"\n  FE: {lab} ~ paper x arm intercepts + pos + pos:harness + pos:harness_sub "
              f"(n={fit.n}, clusters={fit.G}, rank {fit.rank}/{fit.p})")
        res = {}
        for nm, c in (("slope direct", e[0]), ("pos:harness  (harness - direct)", e[1]),
                      ("pos:harness_sub  (harness_sub - direct)", e[2]),
                      ("harness_sub - harness", e[2] - e[1])):
            res[nm] = fit.contrast(c, "CR2")
            print(line(nm + " CR2", res[nm]))
        bsub = fit.beta[1:3]
        V1 = fit.V1[1:3, 1:3]
        Wd = float(bsub @ np.linalg.solve(V1, bsub))
        Fst = Wd / 2 * (fit.G - 2) / (fit.G - 1)
        pF = float(stats.f.sf(Fst, 2, fit.G - 2))
        pW = wild_cluster_boot(X, y, fit.codes, [1, 2], B, seed=201)
        print(f"  joint test of both interactions = 0: CR1 Wald {Wd:.3f} -> F(2,{fit.G - 2}) = {Fst:.3f}, "
              f"p = {pF:.4f}; wild cluster bootstrap p = {pW:.4f}")
        if ycol == "rating":
            add_summary("M7 three-arm joint FE (CR2), common cells", "harness - direct", res["pos:harness  (harness - direct)"],
                        note=f"joint test of both interactions: F p {pF:.3f}, wild-boot p {pW:.3f}")
            add_summary("M7 three-arm joint FE (CR2), common cells", "harness_sub - direct",
                        res["pos:harness_sub  (harness_sub - direct)"])
            add_summary("M7 three-arm joint FE (CR2), common cells", "harness_sub - harness", res["harness_sub - harness"])
            # level at mean position, paper FE
            mpos = d.drop_duplicates("pv")["pos"].mean()
            xc = x - mpos
            XL = np.hstack([np.column_stack([h0, h1, xc, xc * h0, xc * h1]), dummies(d["paper"])])
            fl = ClusterOLS(XL, y, d["paper"].to_numpy(), k_focal=5)
            eL = np.eye(XL.shape[1])
            print(f"  level at mean position ({mpos:.2f}), paper FE:")
            for nm, c in (("harness - direct", eL[0]), ("harness_sub - direct", eL[1]),
                          ("harness_sub - harness", eL[1] - eL[0])):
                cc = fl.contrast(c, "CR2")
                print(line("  level " + nm + " CR2", cc))
                SUMMARY.append(dict(model="LEVEL M7 three-arm FE at mean pos (CR2)", comp=nm, c=cc,
                                    note="", alt={}, level=True))

    dd = d.copy()
    dd["h0"] = (dd["arm"] == "harness").astype(float)
    dd["h1"] = (dd["arm"] == "harness_sub").astype(float)
    r, info = fit_mixed(dd, "rating ~ h0 + h1 + pos + h0:pos + h1:pos", {"pv": "0 + C(pv)"}, "three-arm M2")
    print("\n  MixedLM: rating ~ arm * pos, random intercept + random pos slope by paper + paper-version VC")
    if r is None:
        print("  could not be fitted")
        return
    status = ("converged" if info["converged"] else "NOT CONVERGED") + f" ({info['method']})"
    if info["singular"]:
        status += "; SINGULAR: " + "; ".join(info["singular"])
    print(f"      {status}; sd(int) {info['re']['sd_int']:.3f}, sd(slope) {info['re']['sd_slope']:.3f}, corr "
          f"{info['re']['corr']:+.2f}, vcomp {info['re']['vcomp']}, residual sd {info['re']['resid_sd']:.3f}")
    print("      optimizers: " + "; ".join(info["attempts"]))
    if info["warnings"]:
        print("      warnings: " + " | ".join(info["warnings"]))
    note = ((f"{info['method']} converged" if info["converged"] else "NOT converged")
            + (", some optimizers failed" if any("NOT converged" in a_ for a_ in info["attempts"]) else "")
            + (", singular" if info["singular"] else ""))
    for nm, wts, comp in (("h0:pos (harness - direct)", {"h0:pos": 1.0}, "harness - direct"),
                          ("h1:pos (harness_sub - direct)", {"h1:pos": 1.0}, "harness_sub - direct"),
                          ("h1:pos - h0:pos (harness_sub - harness)", {"h1:pos": 1.0, "h0:pos": -1.0},
                           "harness_sub - harness")):
        c = mixed_contrast(r, wts)
        print(line(nm + " (Wald z)", c))
        add_summary("M7 three-arm mixed (RI+RS+pv VC), common cells", comp, c, note=note)
    names = list(r.params.index)
    ix = [names.index("h0:pos"), names.index("h1:pos")]
    bb = np.asarray(r.params)[ix]
    VV = np.asarray(r.cov_params())[np.ix_(ix, ix)]
    Wd = float(bb @ np.linalg.solve(VV, bb))
    print(f"  joint Wald chi2(2) = {Wd:.3f}, p = {stats.chi2.sf(Wd, 2):.4f}")


# ------------------------------------------------------------------------------ main
def print_summary() -> None:
    comps = ["harness - direct", "harness_sub - direct", "harness_sub - harness"]
    for level in (False, True):
        rows = [s for s in SUMMARY if bool(s.get("level")) == level]
        if not rows:
            continue
        print(f"\n{'=' * 100}\nSUMMARY TABLE — {'LEVEL difference (rating points unless noted)' if level else 'SLOPE difference (rating points per revision step unless noted)'}"
              f"\n{'=' * 100}")
        models = list(dict.fromkeys(s["model"] for s in rows))
        print("| model | " + " | ".join(comps) + " |")
        print("|---|" + "---|" * len(comps))
        for m in models:
            cells = []
            for cp in comps:
                hit = [s for s in rows if s["model"] == m and s["comp"] == cp]
                if not hit:
                    cells.append("-")
                    continue
                s = hit[0]
                c = s["c"]
                if c is None:
                    cells.append(s["note"])
                    continue
                ptxt = f", p={c['p']:.3f}" if np.isfinite(c.get("p", float("nan"))) else ""
                txt = f"{f3(c['est'])} [{f3(c['lo'])}, {f3(c['hi'])}]{ptxt}"
                if s["note"]:
                    txt += f" ({s['note']})"
                cells.append(txt)
            print(f"| {m} | " + " | ".join(cells) + " |")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--logs", nargs="+", required=True)
    ap.add_argument("--repo", default=REPO_DEFAULT)
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--texts", default=None)
    ap.add_argument("--boot", type=int, default=9999, help="wild cluster bootstrap draws")
    args = ap.parse_args()
    repo = Path(args.repo)
    manifest = Path(args.manifest) if args.manifest else repo / "data/processed/pilot_manifest.jsonl"
    texts = Path(args.texts) if args.texts else repo / "data/processed/texts"
    pd.set_option("display.width", 200)

    raw, bad = [], 0
    for lg in args.logs:
        rr, b_ = read_log(lg)
        raw += [r for r in rr if r.get("arm") in ARMS]
        bad += b_
    pos, months, nver = load_design(manifest, texts)
    n_planned = sum(len(v) for v in pos.values())
    print("E1-H regression robustness report")
    print(f"logs: {', '.join(args.logs)}")
    print(f"attempt rows {len(raw)} (unparseable lines skipped: {bad}); planned paper-versions {n_planned} "
          f"in {len(pos)} papers")

    frames = {}
    for b1_all in (False, True):
        rows = dedupe(raw, b1_all)
        frames[b1_all] = build_frame(rows, pos, months, nver)
    df = frames[False]
    for a in ARMS:
        da = df[df.arm == a]
        print(f"  {a:<12s} usable de-duplicated rows {len(da):4d} | paper-versions with >=1 row "
              f"{da['pv'].nunique():3d}/{n_planned} | papers {da['paper'].nunique()} | rating counts "
              f"{dict(sorted(Counter(da['rating'].astype(int)).items()))}")
    extra = len(frames[True]) - len(df)
    print(f"  b1-all variant adds {extra} harness_sub rows (reminded / review-before-report)")
    print("  coverage by planned position (paper-versions with >=1 usable row / planned; mean usable reps per covered cell):")
    plan = pd.DataFrame([dict(paper=a_, v=v_, pos=float(i_)) for a_, dct in pos.items() for v_, i_ in dct.items()])
    plan["posg"] = plan["pos"].clip(upper=3).map({0.0: "pos 0", 1.0: "pos 1", 2.0: "pos 2", 3.0: "pos 3+"})
    for a in ARMS:
        da = df[df.arm == a].groupby(["paper", "v"]).size().rename("n").reset_index()
        mm = plan.merge(da, on=["paper", "v"], how="left")
        parts = []
        for g_, sub in mm.groupby("posg"):
            parts.append(f"{g_}: {int(sub['n'].notna().sum())}/{len(sub)} ({sub['n'].mean():.2f})")
        print(f"    {a:<12s} " + " | ".join(parts))
    dup = df.duplicated(["paper", "v", "arm", "rep"]).sum()
    if dup:
        PROBLEMS.append(f"{dup} duplicated (paper, v, arm, rep) rows after de-duplication")

    for a, b, role in COMPARISONS:
        if (df.arm == a).any() and (df.arm == b).any():
            run_pair(df, a, b, role, args.boot)
    print(f"\n{'#' * 100}\nb1-all sensitivity (every process-verified harness_sub run) — headline model only\n{'#' * 100}")
    for a, b, role in COMPARISONS:
        if "harness_sub" in (a, b):
            run_pair(frames[True], a, b, role, args.boot, tag=" [b1-all]", headline_only=True)
    if all((df.arm == a).any() for a in ARMS):
        run_three(df, args.boot)

    print_summary()
    print(f"\n{'=' * 100}\nVERDICT CHECK (95% intervals; LOO rows excluded)\n{'=' * 100}")
    sl = [s_ for s_ in SUMMARY if not s_.get("level") and s_["c"] is not None and not s_["model"].startswith("M5")]

    def all_p(s_):
        out = dict(s_.get("alt") or {})
        if np.isfinite(s_["c"].get("p", np.nan)):
            out["main"] = s_["c"]["p"]
        return out

    excl = [s_ for s_ in sl if s_["c"]["lo"] > 0 or s_["c"]["hi"] < 0 or any(v < 0.05 for v in all_p(s_).values())]
    print(f"  slope-difference rows: {len(sl)}; rows where the 95% interval excludes 0 or ANY of the p-values "
          f"computed for that row is < 0.05: {len(excl)}")
    for s_ in excl:
        print(f"    - {s_['model']} | {s_['comp']}: {f3(s_['c']['est'])} [{f3(s_['c']['lo'])}, {f3(s_['c']['hi'])}]; "
              + ", ".join(f"{k} p={v:.4f}" for k, v in all_p(s_).items()))
    prim = [s_ for s_ in sl if s_["comp"] == "harness - direct"]
    print(f"  PRIMARY (harness - direct): smallest p over {len(prim)} model rows and all tests = "
          f"{min(min(all_p(s_).values()) for s_ in prim if all_p(s_)):.3f}")
    print(f"  all comparisons: smallest p = {min(min(all_p(s_).values()) for s_ in sl if all_p(s_)):.3f} "
          f"(no multiplicity correction; {sum(len(all_p(s_)) for s_ in sl)} p-values in total)")
    lv = [s_ for s_ in SUMMARY if s_.get("level")]
    for cp in ("harness - direct", "harness_sub - direct", "harness_sub - harness"):
        rr = [s_ for s_ in lv if s_["comp"] == cp]
        if rr:
            neg = sum(s_["c"]["hi"] < 0 for s_ in rr)
            print(f"  level {cp}: estimates {f3(min(s_['c']['est'] for s_ in rr))} .. {f3(max(s_['c']['est'] for s_ in rr))}; "
                  f"{neg} of {len(rr)} model rows have a 95% CI entirely below 0")
    print(f"\n{'=' * 100}\nMODELLING PROBLEMS / WARNINGS\n{'=' * 100}")
    if PROBLEMS:
        for p in dict.fromkeys(PROBLEMS):
            print(f"  - {p}")
    else:
        print("  none")


if __name__ == "__main__":
    main()
