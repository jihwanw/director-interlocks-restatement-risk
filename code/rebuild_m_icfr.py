"""rebuild_m_icfr.py — Module M: (1) SOX 404 internal-control opinions as a second destination outcome; (2) appointment-level analysis.
Run from github/:  venv python code/rebuild_m_icfr.py --suffix _art

Data (verified 2026-10-15 from the feed's own joint distributions; the vendor dictionary was not accessible):
  audit.feed11_sox_404_internal_controls (data/sox404_icfr_wrds_2026-10-13.csv.gz): one row per opinion; ic_op_type 'm' = management
  assessment, 'a' = auditor attestation; ic_is_effective 'Y'/'N'/'ND' — every 'N' row carries count_weak >= 1 (median 2) and no 'Y' row does,
  so 'N' = internal control over financial reporting reported as not effective (material weakness); 'ND' (743 rows) is treated as missing.
  fy_ic_op is the fiscal year of the opinion (= calendar year of fye_ic_op for 99.1% of rows). A firm-fiscal-year is ADVERSE if any of its
  opinion rows is 'N'. Coverage: 2,880 firms with an opinion in FY2004, 8,170 in FY2007 (management assessments of non-accelerated filers
  begin), 6,500-7,200 from 2017; the pair analysis therefore requires an opinion in each window and reports results for ties formed 2010+.
Outputs: results_rebuild/M_icfr{suffix}.json
"""
import argparse, json, os, time
from collections import defaultdict
import numpy as np, pandas as pd
from linearmodels.panel import PanelOLS

ap = argparse.ArgumentParser(); ap.add_argument("--suffix", default="_art"); A = ap.parse_args(); SUFFIX = A.suffix
T0 = time.time()
def log(m): print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)
def norm_cik(s): return pd.Series(s).astype(str).str.replace(r"\.0$", "", regex=True).str.lstrip("0")
HORIZON = pd.Timestamp("2025-12-31"); D3 = pd.Timedelta(days=3 * 365); D5 = pd.Timedelta(days=5 * 365)

# ------------------------------------------------------------------ estimators (as in modules K/L)
def twoway_ols(y, X, g1, g2):
    X = np.asarray(X, float); y = np.asarray(y, float)
    XtX_inv = np.linalg.pinv(X.T @ X); b = XtX_inv @ X.T @ y; e = y - X @ b
    def meat(groups):
        codes = pd.factorize(groups)[0]; sums = np.zeros((codes.max() + 1, X.shape[1])); np.add.at(sums, codes, X * e[:, None]); return sums.T @ sums
    g12 = pd.factorize(pd.Series(g1).astype(str) + "#" + pd.Series(g2).astype(str))[0]
    V = XtX_inv @ (meat(g1) + meat(g2) - meat(g12)) @ XtX_inv
    return b, np.sqrt(np.maximum(np.diag(V), 0))
def gap(df, out, exp, c1="src", c2="dst"):
    X = np.column_stack([np.ones(len(df)), df[exp].values]); b, se = twoway_ols(df[out].values, X, df[c1].values, df[c2].values)
    return {"exposed_rate": round(float(df.loc[df[exp] == 1, out].mean()), 4) if (df[exp] == 1).any() else None, "clean_rate": round(float(df.loc[df[exp] == 0, out].mean()), 4),
            "diff_pp": round(100 * b[1], 3), "se_pp": round(100 * se[1], 3), "t": round(float(b[1] / se[1]), 2) if se[1] > 0 else None, "n_rows": int(len(df)), "n_exposed_rows": int(df[exp].sum())}
def did(df, pre, post, exp, c1="src", c2="dst", yfe=False):
    st = pd.concat([df.assign(y=df[pre], post=0), df.assign(y=df[post], post=1)], ignore_index=True)
    cols = [np.ones(len(st)), st[exp].values, st["post"].values, (st[exp] * st["post"]).values]
    if yfe:
        yd = pd.get_dummies(st["by"].astype(int), drop_first=True).astype(float).values; cols += [yd, yd * st["post"].values[:, None]]
    X = np.column_stack(cols); b, se = twoway_ols(st["y"].values, X, st[c1].values, st[c2].values)
    return {"pre_gap_pp": round(100 * b[1], 3), "se_pre_pp": round(100 * se[1], 3), "diff_in_gaps_pp": round(100 * b[3], 3), "se_pp": round(100 * se[3], 3), "t": round(float(b[3] / se[3]), 2) if se[3] > 0 else None,
            "controls": ["formation-year FE + formation-year x post FE"] if yfe else []}

# ------------------------------------------------------------------ data
base = pd.read_csv(f"data/pair_base_k{SUFFIX}.csv", dtype={"src": str, "dst": str}, parse_dates=["birth", "death"])
eq = base[(base.src_is_origin.astype(str) == "True") & (base.by <= 2022)].copy(); log(f"pairs: appointment direction, formed <= 2022: {len(eq):,}")
ic = pd.read_csv("data/sox404_icfr_wrds_2026-10-13.csv.gz", parse_dates=["fye_ic_op", "file_date"], low_memory=False)
ic["cik"] = norm_cik(ic.company_fkey); ic = ic[ic.ic_is_effective.isin(["Y", "N"]) & ic.fye_ic_op.notna()]
fy = ic.groupby(["cik", "fye_ic_op"]).agg(adverse=("ic_is_effective", lambda s: int((s == "N").any())), fy=("fy_ic_op", "first")).reset_index().sort_values(["cik", "fye_ic_op"])
fy["prev_adverse"] = fy.groupby("cik")["adverse"].shift(1); fy["new_adverse"] = ((fy.adverse == 1) & (fy.prev_adverse == 0)).astype(int)
FYE = defaultdict(list)   # cik -> [(fye, adverse, new_adverse)]
for c_, d_, a_, n_ in zip(fy.cik, fy.fye_ic_op, fy.adverse, fy.new_adverse): FYE[c_].append((d_, int(a_), int(n_)))
log(f"ICFR firm-fiscal-years: {len(fy):,} ({fy.cik.nunique():,} firms); adverse share {fy.adverse.mean():.3f}; new-adverse share {fy.new_adverse.mean():.3f}")
def win(cik, lo, hi, kind):
    rows = [r for r in FYE.get(cik, ()) if lo <= r[0] < hi]
    if not rows: return None
    return int(any(r[1] for r in rows)) if kind == "adverse" else int(any(r[2] for r in rows))
def n_op(cik, lo, hi): return sum(1 for r in FYE.get(cik, ()) if lo <= r[0] < hi)
b_ = eq.birth
eq["pre_adv"] = [win(d, b - D3, b, "adverse") for d, b in zip(eq.dst, b_)]; eq["post_adv"] = [win(d, b, min(b + D3, HORIZON), "adverse") for d, b in zip(eq.dst, b_)]
eq["pre_new"] = [win(d, b - D3, b, "new") for d, b in zip(eq.dst, b_)]; eq["post_new"] = [win(d, b, min(b + D3, HORIZON), "new") for d, b in zip(eq.dst, b_)]
eq["n_pre_op"] = [n_op(d, b - D3, b) for d, b in zip(eq.dst, b_)]; eq["n_post_op"] = [n_op(d, b, min(b + D3, HORIZON)) for d, b in zip(eq.dst, b_)]
# origin ICFR exposure: adverse opinion at the origin before the tie / within five years before
def any_adv_before(cik, t, years=None):
    rows = [r for r in FYE.get(cik, ()) if r[0] < t and (years is None or r[0] >= t - pd.Timedelta(days=365 * years))]; return int(any(r[1] for r in rows))
eq["icfr_exposed"] = [any_adv_before(s, b) for s, b in zip(eq.src, b_)]; eq["icfr_exposed5y"] = [any_adv_before(s, b, 5) for s, b in zip(eq.src, b_)]
eq["src_has_icfr_before"] = [int(any(r[0] < b for r in FYE.get(s, ()))) for s, b in zip(eq.src, b_)]
cov = eq[(eq.n_pre_op > 0) & (eq.n_post_op > 0)].copy()
OUT = {"suffix": SUFFIX, "coverage": {"pairs_eq_window": int(len(eq)), "pairs_with_opinion_in_both_windows": int(len(cov)), "share": round(len(cov) / len(eq), 4),
       "share_by_formation_period": {p_: round(float(((eq.n_pre_op > 0) & (eq.n_post_op > 0))[eq.by.between(*r_)].mean()), 4) for p_, r_ in (("2004-2009", (2004, 2009)), ("2010-2015", (2010, 2015)), ("2016-2022", (2016, 2022)))},
       "share_exposed_in_covered": round(float(cov.exposed.mean()), 4), "share_exposed_all": round(float(eq.exposed.mean()), 4),
       "icfr_exposure": {"share_origin_with_opinion_before_tie": round(float(eq.src_has_icfr_before.mean()), 4), "share_icfr_exposed": round(float(eq.icfr_exposed.mean()), 4), "share_icfr_exposed5y": round(float(eq.icfr_exposed5y.mean()), 4),
                         "overlap_icfr5y_with_restatement5y": round(float(eq.loc[eq.icfr_exposed5y == 1, "exposed5y"].mean()), 4)},
       "firm_fy_adverse_share": round(float(fy.adverse.mean()), 4), "firm_fy_new_adverse_share": round(float(fy.new_adverse.mean()), 4), "firm_fys": int(len(fy)), "firms_with_opinion": int(fy.cik.nunique())}}
log(f"coverage: {len(cov):,} of {len(eq):,} pairs have an ICFR opinion in both windows ({OUT['coverage']['share']:.1%}); by period {OUT['coverage']['share_by_formation_period']}")

# ------------------------------------------------------------------ M1 pair-level before/after on ICFR outcomes
def block(df):
    out = {"n_rows": int(len(df)), "pre_rate": round(float(df.pre_adv.mean()), 4), "post_rate": round(float(df.post_adv.mean()), 4), "pre_new_rate": round(float(df.pre_new.mean()), 4), "post_new_rate": round(float(df.post_new.mean()), 4)}
    for ex in ("exposed", "exposed5y", "exposed_served", "icfr_exposed", "icfr_exposed5y"):
        if df[ex].sum() < 30 or (df[ex] == 0).sum() < 30: continue
        out[ex] = {"n_exposed": int(df[ex].sum()),
                   "adverse": {"pre": gap(df, "pre_adv", ex), "post": gap(df, "post_adv", ex), "did": did(df, "pre_adv", "post_adv", ex), "did_cohortFE": did(df, "pre_adv", "post_adv", ex, yfe=True)},
                   "new_adverse": {"pre": gap(df, "pre_new", ex), "post": gap(df, "post_new", ex), "did": did(df, "pre_new", "post_new", ex), "did_cohortFE": did(df, "pre_new", "post_new", ex, yfe=True)}}
        hf = df[df.pre_adv == 0]   # destinations with no adverse opinion in the pre-formation window
        out[ex]["no_pre_adverse_post_gap"] = gap(hf, "post_adv", ex) if hf[ex].sum() >= 30 else None
        out[ex]["no_pre_adverse_post_new_gap"] = gap(hf, "post_new", ex) if hf[ex].sum() >= 30 else None
    return out
OUT["M1_pairs_covered"] = block(cov)
OUT["M1_pairs_covered_formed_2010plus"] = block(cov[cov.by >= 2010])
cov["bc"] = 0
log("M1 done: " + "; ".join(f"{ex}: adverse pre {OUT['M1_pairs_covered'][ex]['adverse']['pre']['diff_pp']:+.2f} post {OUT['M1_pairs_covered'][ex]['adverse']['post']['diff_pp']:+.2f} change {OUT['M1_pairs_covered'][ex]['adverse']['did']['diff_in_gaps_pp']:+.2f} ({OUT['M1_pairs_covered'][ex]['adverse']['did']['se_pp']:.2f}) yfe {OUT['M1_pairs_covered'][ex]['adverse']['did_cohortFE']['diff_in_gaps_pp']:+.2f}; new {OUT['M1_pairs_covered'][ex]['new_adverse']['did_cohortFE']['diff_in_gaps_pp']:+.2f}" for ex in ("exposed", "exposed5y", "icfr_exposed5y") if ex in OUT["M1_pairs_covered"]))
# restatement outcomes on the covered subsample, for comparison with the paper's main estimates on the same pairs
OUT["M1_restatement_outcomes_same_pairs"] = {ex: {"did_onset": did(cov, "pre_onset_3y", "out_onset_3y", ex), "did_onset_cohortFE": did(cov, "pre_onset_3y", "out_onset_3y", ex, yfe=True), "did_file": did(cov, "pre_file_3y", "out_file_3y", ex)} for ex in ("exposed", "exposed5y")}

# ------------------------------------------------------------------ M2 firm-year panel: adverse opinion in t on lagged exposure
pnl = pd.read_csv(f"data/integrated_panel_k{SUFFIX}.csv", dtype={"cik": str}); pnl["cik"] = norm_cik(pnl["cik"])
fy["year"] = fy["fy"].astype(int); pnl = pnl.merge(fy[["cik", "year", "adverse", "new_adverse"]], on=["cik", "year"], how="left")
pp_ = pnl[pnl.adverse.notna()].copy(); pp_["adverse"] = pp_["adverse"].astype(float); pp_["new_adverse"] = pp_["new_adverse"].fillna(0).astype(float)
ctrl = ["degree_lag", "betweenness_lag", "size", "leverage", "roa"]
pp_ = pp_[pp_.groupby("gvkey")["year"].transform("count") >= 2].set_index(["gvkey", "year"])
def fe(dep, exog, data):
    r = PanelOLS(data[dep], data[exog], entity_effects=True, time_effects=True, drop_absorbed=True).fit(cov_type="clustered", cluster_entity=True); return r
def cb(r, k): return {"coef_pp": round(100 * float(r.params[k]), 3), "se_pp": round(100 * float(r.std_errors[k]), 3), "t": round(float(r.tstats[k]), 2)}
OUT["M2_panel"] = {"n_firm_years": int(len(pp_)), "n_firms": int(pp_.index.get_level_values(0).nunique()), "adverse_rate": round(float(pp_.adverse.mean()), 4), "new_adverse_rate": round(float(pp_.new_adverse.mean()), 4), "models": {}}
for name, dep, key, controls in (("adverse_any_exposed_lag1", "adverse", "nexp_lag1", ["nnew_lag1"] + ctrl + ["had_prior_restate"]),
                                 ("adverse_recent_exposed_lag1", "adverse", "nexp5_lag1", ["nnew_lag1"] + ctrl + ["had_prior_restate"]),
                                 ("adverse_inbound_recent_lag1", "adverse", "nexp5_in_lag1", ["nnew_in_lag1", "nnew_out_lag1", "nexp5_out_lag1"] + ctrl + ["had_prior_restate"]),
                                 ("adverse_recent_stock_boy", "adverse", "nexp5_boy", ["nalive_boy"] + ctrl + ["had_prior_restate"]),
                                 ("new_adverse_recent_exposed_lag1", "new_adverse", "nexp5_lag1", ["nnew_lag1"] + ctrl + ["had_prior_restate"]),
                                 ("new_adverse_any_exposed_lag1", "new_adverse", "nexp_lag1", ["nnew_lag1"] + ctrl + ["had_prior_restate"])):
    r = fe(dep, [key] + controls, pp_); OUT["M2_panel"]["models"][name] = {"exposure_coef": cb(r, key), "n": int(r.nobs), "within_r2": round(float(r.rsquared_within), 5), "controls": controls}
# the restatement-onset specification on the same ICFR-covered firm-years, for comparison
r = fe("restate_onset", ["nexp5_lag1", "nnew_lag1"] + ctrl + ["had_prior_restate"], pp_); OUT["M2_panel"]["models"]["onset_recent_exposed_lag1_same_firm_years"] = {"exposure_coef": cb(r, "nexp5_lag1"), "n": int(r.nobs)}
log("M2 panel: " + "; ".join(f"{k}: {v['exposure_coef']['coef_pp']:+.3f} (t={v['exposure_coef']['t']})" for k, v in OUT["M2_panel"]["models"].items()))

# ------------------------------------------------------------------ M3 appointment-level analysis (referee B, r10/r11): one unit per (director, destination, appointment date)
eq["appt"] = eq.director.astype(str) + "#" + eq.dst + "#" + eq.birth.astype(str)
agg = eq.groupby("appt").agg(dst=("dst", "first"), director=("director", "first"), by=("by", "first"), n_origins=("src", "nunique"),
                             any_exposed=("exposed", "max"), all_exposed=("exposed", "min"), share_exposed=("exposed", "mean"),
                             any_recent=("exposed5y", "max"), all_recent=("exposed5y", "min"), share_recent=("exposed5y", "mean"),
                             pre_file=("pre_file_3y", "first"), post_file=("out_file_3y", "first"), pre_onset=("pre_onset_3y", "first"), post_onset=("out_onset_3y", "first"), dst_prior_any=("dst_prior_any", "first")).reset_index()
OUT["M3_appointments"] = {"n_appointments": int(len(agg)), "n_pairs": int(len(eq)), "share_multi_origin": round(float((agg.n_origins > 1).mean()), 4), "mean_origins": round(float(agg.n_origins.mean()), 3),
                          "share_mixed_exposure_among_multi_origin": round(float(((agg.any_exposed == 1) & (agg.all_exposed == 0))[agg.n_origins > 1].mean()), 4),
                          "share_mixed_recent_among_multi_origin": round(float(((agg.any_recent == 1) & (agg.all_recent == 0))[agg.n_origins > 1].mean()), 4)}
for ex in ("any_exposed", "all_exposed", "any_recent", "all_recent"):
    OUT["M3_appointments"][ex] = {"n_exposed": int(agg[ex].sum()), "pre_file": gap(agg, "pre_file", ex, "director", "dst"), "post_file": gap(agg, "post_file", ex, "director", "dst"), "did_file": did(agg, "pre_file", "post_file", ex, "director", "dst"),
                                  "pre_onset": gap(agg, "pre_onset", ex, "director", "dst"), "post_onset": gap(agg, "post_onset", ex, "director", "dst"), "did_onset": did(agg, "pre_onset", "post_onset", ex, "director", "dst"),
                                  "did_onset_cohortFE": did(agg, "pre_onset", "post_onset", ex, "director", "dst", yfe=True), "never_filed_post_onset": gap(agg[agg.dst_prior_any == 0], "post_onset", ex, "director", "dst")}
# continuous: share of origins exposed (one coefficient per unit share)
def did_cont(df, pre, post, x, c1, c2):
    st = pd.concat([df.assign(y=df[pre], post=0), df.assign(y=df[post], post=1)], ignore_index=True)
    yd = pd.get_dummies(st["by"].astype(int), drop_first=True).astype(float).values
    X = np.column_stack([np.ones(len(st)), st[x].values, st["post"].values, (st[x] * st["post"]).values, yd, yd * st["post"].values[:, None]]); b, se = twoway_ols(st["y"].values, X, st[c1].values, st[c2].values)
    return {"pre_gap_pp_per_unit": round(100 * b[1], 3), "diff_in_gaps_pp_per_unit": round(100 * b[3], 3), "se_pp": round(100 * se[3], 3), "t": round(float(b[3] / se[3]), 2)}
OUT["M3_appointments"]["share_recent_continuous_onset_cohortFE"] = did_cont(agg, "pre_onset", "post_onset", "share_recent", "director", "dst")
OUT["M3_appointments"]["share_exposed_continuous_onset_cohortFE"] = did_cont(agg, "pre_onset", "post_onset", "share_exposed", "director", "dst")
log("M3 appointments: " + "; ".join(f"{ex}: onset pre {OUT['M3_appointments'][ex]['pre_onset']['diff_pp']:+.2f} post {OUT['M3_appointments'][ex]['post_onset']['diff_pp']:+.2f} yfe change {OUT['M3_appointments'][ex]['did_onset_cohortFE']['diff_in_gaps_pp']:+.2f} ({OUT['M3_appointments'][ex]['did_onset_cohortFE']['se_pp']:.2f})" for ex in ("any_exposed", "any_recent", "all_recent")))
json.dump(OUT, open(f"results_rebuild/M_icfr{SUFFIX}.json", "w"), indent=1, default=float); log(f"saved results_rebuild/M_icfr{SUFFIX}.json")
