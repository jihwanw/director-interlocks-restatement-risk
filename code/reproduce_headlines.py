"""reproduce_headlines.py — adversarial reproduction of headline numbers with independent code.

Recomputes exposure and outcome indicators for the primary-sample pairs directly from
data/restatements_with_dates.csv (not from the columns module K wrote), re-estimates the
before/after gaps and the cohort-adjusted change with statsmodels (two-way clustered SE), and
re-estimates the refresh test and the firm-year panel row with a fresh PanelOLS call.
Prints paper value vs reproduced value; exits 1 if any headline differs beyond tolerance.
Run from github/: venv python code/reproduce_headlines.py
"""
import json, sys
import numpy as np, pandas as pd
import statsmodels.api as sm

TOL_PP = 0.06      # tolerance in percentage points for gaps/changes (rounding of both sides)
import os
PRIMARY = os.environ.get("PRIMARY_SUFFIX", "_art")
K = json.load(open(f"results_rebuild/K_rebuild{PRIMARY}.json"))
EW = K["K1_pairs"]["equal_window_3y"]
pairs = pd.read_csv(f"data/pair_base_k{PRIMARY}.csv", dtype={"src": str, "dst": str}, parse_dates=["birth", "death"])
rest = pd.read_csv("data/restatements_with_dates.csv", dtype={"company_fkey": str})
rest["cik"] = rest["company_fkey"].str.lstrip("0")
for c in ("file_date", "res_begin_date"):
    rest[c] = pd.to_datetime(rest[c], errors="coerce")
rest = rest[rest.file_date <= pd.Timestamp("2025-12-31")]   # the paper's horizon (last date with complete coverage in both feeds)
files = rest.dropna(subset=["file_date"]).groupby("cik")["file_date"].apply(lambda s: np.sort(s.values)).to_dict()
begins = rest.dropna(subset=["res_begin_date"]).groupby("cik")["res_begin_date"].apply(lambda s: np.sort(s.values)).to_dict()
D3 = np.timedelta64(3 * 365, "D"); D5 = np.timedelta64(5 * 365, "D")

def any_in(arr, lo, hi):
    if arr is None: return 0
    i = np.searchsorted(arr, lo, side="left"); j = np.searchsorted(arr, hi, side="left")
    return int(j > i)

print("src_is_origin values:", pairs.src_is_origin.astype(str).value_counts().to_dict())
# the K module's equal-window sample: appointment direction (src is the origin) and interlocks formed by 2022
eq = pairs[(pairs.by <= 2022) & (pairs.src_is_origin.astype(str) == "True")].copy()
b = eq.birth.values.astype("datetime64[ns]")
eq["r_exposed"] = [any_in(files.get(s), np.datetime64("1900-01-01"), bb) for s, bb in zip(eq.src, b)]
eq["r_exposed5y"] = [any_in(files.get(s), bb - D5, bb) for s, bb in zip(eq.src, b)]
eq["r_pre_file"] = [any_in(files.get(d), bb - D3, bb) for d, bb in zip(eq.dst, b)]
eq["r_out_file"] = [any_in(files.get(d), bb, bb + D3) for d, bb in zip(eq.dst, b)]
eq["r_pre_onset"] = [any_in(begins.get(d), bb - D3, bb) for d, bb in zip(eq.dst, b)]
eq["r_out_onset"] = [any_in(begins.get(d), bb, bb + D3) for d, bb in zip(eq.dst, b)]

print(f"equal-window rows: reproduced {len(eq):,} vs paper {EW['n_rows']:,}")
for mine, theirs in (("r_exposed", "exposed"), ("r_exposed5y", "exposed5y"), ("r_pre_file", "pre_file_3y"), ("r_out_file", "out_file_3y"), ("r_pre_onset", "pre_onset_3y"), ("r_out_onset", "out_onset_3y")):
    agree = (eq[mine] == eq[theirs]).mean()
    print(f"  agreement {mine:12s} vs K column {theirs:13s}: {agree:.4f}  (mean mine {eq[mine].mean():.4f}, K {eq[theirs].mean():.4f})")

def gap(df, y, x):
    return 100 * (df.loc[df[x] == 1, y].mean() - df.loc[df[x] == 0, y].mean())

def did_yfe(df, pre, post, x):
    st = pd.concat([df.assign(y=df[pre], post=0), df.assign(y=df[post], post=1)], ignore_index=True)
    yd = pd.get_dummies(st["by"].astype(int), prefix="by", drop_first=True).astype(float)
    X = pd.concat([pd.Series(1.0, index=st.index, name="const"), st[x].astype(float).rename("x"), st["post"].astype(float), (st[x] * st["post"]).astype(float).rename("xpost"), yd, yd.mul(st["post"].values, axis=0).add_prefix("post_")], axis=1)
    groups = np.column_stack([pd.factorize(st["src"])[0], pd.factorize(st["dst"])[0]])
    r = sm.OLS(st["y"].astype(float), X).fit(cov_type="cluster", cov_kwds={"groups": groups})
    return 100 * r.params["xpost"], 100 * r.bse["xpost"]

def did_plain(df, pre, post, x):
    st = pd.concat([df.assign(y=df[pre], post=0), df.assign(y=df[post], post=1)], ignore_index=True)
    X = pd.DataFrame({"const": 1.0, "x": st[x].astype(float), "post": st["post"].astype(float), "xpost": (st[x] * st["post"]).astype(float)})
    groups = np.column_stack([pd.factorize(st["src"])[0], pd.factorize(st["dst"])[0]])
    r = sm.OLS(st["y"].astype(float), X).fit(cov_type="cluster", cov_kwds={"groups": groups})
    return 100 * r.params["xpost"], 100 * r.bse["xpost"]

checks = []
def chk(label, paper, mine, tol=TOL_PP):
    ok = abs(paper - mine) <= tol; checks.append(ok)
    print(f"  {'OK ' if ok else 'DIFF'} {label:60s} paper {paper:+7.2f}  reproduced {mine:+7.2f}")

print("\nHeadline gaps (unadjusted means, reproduced outcomes):")
chk("pre-formation filing gap, any exposure", EW["pre_file"]["diff_pp"], gap(eq, "r_pre_file", "r_exposed"))
chk("post-formation filing gap, any exposure", EW["post_file"]["diff_pp"], gap(eq, "r_out_file", "r_exposed"))
chk("pre-formation onset gap, any exposure", EW["pre_onset"]["diff_pp"], gap(eq, "r_pre_onset", "r_exposed"))
chk("post-formation onset gap, any exposure", EW["post_onset"]["diff_pp"], gap(eq, "r_out_onset", "r_exposed"))
R5 = K["K1_pairs"]["exposure_recent_5y"]
chk("pre-formation onset gap, recent exposure", R5["pre_onset"]["diff_pp"], gap(eq, "r_pre_onset", "r_exposed5y"))
chk("post-formation onset gap, recent exposure", R5["post_onset"]["diff_pp"], gap(eq, "r_out_onset", "r_exposed5y"))
print("\nStacked changes (statsmodels, two-way clustered):")
e, s = did_plain(eq, "r_pre_onset", "r_out_onset", "r_exposed5y"); chk("unadjusted change, recent exposure (onset)", R5["did_onset"]["diff_in_gaps_pp"], e); print(f"       SE paper {R5['did_onset']['se_pp']:.2f} reproduced {s:.2f}")
e, s = did_yfe(eq, "r_pre_onset", "r_out_onset", "r_exposed5y"); chk("cohort-adjusted change, recent exposure (onset)", EW["did_onset_cohortFE_recent"]["diff_in_gaps_pp"], e); print(f"       SE paper {EW['did_onset_cohortFE_recent']['se_pp']:.2f} reproduced {s:.2f}")
e, s = did_yfe(eq, "r_pre_onset", "r_out_onset", "r_exposed"); chk("cohort-adjusted change, any exposure (onset)", EW["did_onset_cohortFE"]["diff_in_gaps_pp"], e)
e, s = did_plain(eq, "r_pre_file", "r_out_file", "r_exposed"); chk("unadjusted change, any exposure (filed)", EW["did_file"]["diff_in_gaps_pp"], e)

# refresh test on the equal-window sample (as in K1["board_refresh_test"]): destination filed in the 12 months before the appointment
D1 = np.timedelta64(365, "D")
eq["dst_filed_12m"] = [any_in(files.get(d), x - D1, x) for d, x in zip(eq.dst, b)]
BR = K["K1_pairs"]["board_refresh_test"]
print("\nRefresh test (equal-window sample):")
chk("share of pairs with a destination filing in prior 12 months, %", 100 * BR["share_dst_filed_12m_before"], 100 * eq.dst_filed_12m.mean(), tol=0.06)
chk("recent-exposure share if filed in prior 12 months, %", 100 * BR["recent_exposed_share_if_dst_filed_12m_before"], 100 * eq.loc[eq.dst_filed_12m == 1, "r_exposed5y"].mean(), tol=0.1)
chk("recent-exposure share otherwise, %", 100 * BR["recent_exposed_share_otherwise"], 100 * eq.loc[eq.dst_filed_12m == 0, "r_exposed5y"].mean(), tol=0.1)

# firm-year panel row (13): onsets in t on recently exposed interlocks formed in t-1, all new interlocks, controls and own history, firm and year effects.
# Two-way within transformation by alternating projections (independent of linearmodels), firm-clustered SE.
pnl = pd.read_csv(f"data/integrated_panel_k{PRIMARY}.csv")
cols = ["restate_onset", "nexp5_lag1", "nnew_lag1", "degree_lag", "betweenness_lag", "size", "leverage", "roa", "had_prior_restate"]
pn = pnl.dropna(subset=cols).copy()
M = K["K2_panel"]["models"]["lag1_onset_exposed5y_plus_prior"]
print(f"\nPanel row (13): rows reproduced {len(pn):,} vs paper {M['n']:,}")
def demean2(v, g1, g2, it=200):
    v = v.astype(float).copy()
    for _ in range(it):
        v0 = v.copy()
        v -= pd.Series(v).groupby(g1).transform("mean").values
        v -= pd.Series(v).groupby(g2).transform("mean").values
        if np.max(np.abs(v - v0)) < 1e-10: break
    return v
g1 = pn["gvkey"].values; g2 = pn["year"].values
Yd = demean2(pn["restate_onset"].values, g1, g2)
Xd = np.column_stack([demean2(pn[c].values, g1, g2) for c in cols[1:]])
beta = np.linalg.lstsq(Xd, Yd, rcond=None)[0]; e_ = Yd - Xd @ beta
XtX_inv = np.linalg.inv(Xd.T @ Xd); codes = pd.factorize(g1)[0]
sums = np.zeros((codes.max() + 1, Xd.shape[1])); np.add.at(sums, codes, Xd * e_[:, None])
G = codes.max() + 1; N_ = len(Yd); Kp = Xd.shape[1] + pn["gvkey"].nunique() + pn["year"].nunique()
V = XtX_inv @ (sums.T @ sums) @ XtX_inv * (G / (G - 1)) * ((N_ - 1) / (N_ - Kp))
se = np.sqrt(np.diag(V))
print(f"  coefficient on recently exposed ties: paper {100*M['exposure_coef']['coef']:+.3f} pp (t={M['exposure_coef']['t']:.2f}); reproduced {100*beta[0]:+.3f} pp (t={beta[0]/se[0]:.2f})")
chk("panel row (13) coefficient, pp", 100 * M["exposure_coef"]["coef"], 100 * beta[0], tol=0.01)

print("\nRESULT:", "PASS" if all(checks) else "FAIL", f"({sum(checks)}/{len(checks)} headline checks within {TOL_PP} pp)")
sys.exit(0 if all(checks) else 1)
