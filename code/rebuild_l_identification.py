#!/usr/bin/env python3
"""
Module L — identification, falsification and bounding layer for the dated-interlock paper.
Reads the module-K intermediates (pair base, lifecycles, panel) and the raw feeds; writes results_rebuild/L_identification{SUFFIX}.json.
Run from github/:  venv python code/rebuild_l_identification.py --suffix _le6 [--max-firms 6]

Blocks
  A  latent (unknown-at-appointment) exposure: origin misstated period overlapped the director's origin seat, ENDED before the appointment,
     and was first restated AFTER it (buffer 6 months, horizon 36 months), with no origin filing before the tie; three-class decomposition
     (latent / revealed / label-only) with formation-year x post FE, destination-industry x post, origin-demeaned version, fraud exclusion
  B  channel ordering (clean | label-only | present-at-filing-only | present-during-misstatement | latent), anticipation recruiting equation,
     detection hazard among latently misstating destinations, wave-cleaned exposure, past-outcome and future-career-link placebos,
     two-step network-neighbour placebo in the firm-year panel, symmetric forward/reverse table by destination history
  C  frozen-network origin-disclosure event study (links established >= 12 months before the origin's filing; destination x event and
     event x event-time fixed effects; outcomes: filings, new onsets, legacy filings)
  D  bounds: drift bounds with an appointment-date jump (grid), Cinelli-Hazlett robustness values and own-history benchmark for the panel
     recent-exposure coefficient, monotone-selection bound, placebo formation dates (+/-3 years), permutation inference within
     formation-year x size cells
"""
import argparse, json, os, time
from collections import defaultdict
import numpy as np, pandas as pd
from lifelines import CoxPHFitter
from linearmodels.panel import PanelOLS

ap = argparse.ArgumentParser(); ap.add_argument("--suffix", default="_le6"); ap.add_argument("--max-firms", type=int, default=6); ap.add_argument("--n-perm", type=int, default=1000)
A = ap.parse_args(); SUFFIX = A.suffix; MAX_FIRMS = A.max_firms
T0 = time.time()
def log(m): print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)
HORIZON = pd.Timestamp("2025-12-31"); D3 = pd.Timedelta(days=3 * 365); D6M = pd.Timedelta(days=183); D1Y = pd.Timedelta(days=365)
OUT = {"definition": {"latent": "origin misstated period overlapped the director's origin seat, ended before the tie, and was first restated after the tie; no origin filing within the 5 years before the tie (so no recent public label); variants: filing >= 183 days after the tie, no origin filing ever before the tie",
                      "revealed": "exposed_served (seat overlapped a misstated period of a restatement filed before the tie)",
                      "label_only": "origin filed within 5 years before the tie but the director joined the origin after every such misstated period ended",
                      "present_at_filing_only": "seat covered a filing date before the tie but overlapped no misstated period",
                      "clean": "no origin filing before the tie and not latent"}}
def norm_cik(s): return pd.Series(s).astype(str).str.replace(r"\.0$", "", regex=True).str.lstrip("0")

# ------------------------------------------------------------------ inference helpers (as in module K)
def twoway_ols(y, X, g1, g2, w=None):
    X = np.asarray(X, float); y = np.asarray(y, float)
    if w is not None:
        sw = np.sqrt(np.asarray(w, float))[:, None]; X = X * sw; y = y * sw[:, 0]
    XtX_inv = np.linalg.pinv(X.T @ X); b = XtX_inv @ X.T @ y; e = y - X @ b
    def meat(groups):
        codes = pd.factorize(groups)[0]; sums = np.zeros((codes.max() + 1, X.shape[1])); np.add.at(sums, codes, X * e[:, None]); return sums.T @ sums
    g12 = pd.factorize(pd.Series(g1).astype(str) + "#" + pd.Series(g2).astype(str))[0]
    V = XtX_inv @ (meat(g1) + meat(g2) - meat(g12)) @ XtX_inv
    return b, np.sqrt(np.maximum(np.diag(V), 0))
def gap(df, out, exp):
    X = np.column_stack([np.ones(len(df)), df[exp].values]); b, se = twoway_ols(df[out].values, X, df["src"].values, df["dst"].values)
    return {"exposed_rate": round(float(df.loc[df[exp] == 1, out].mean()), 4) if (df[exp] == 1).any() else None, "clean_rate": round(float(df.loc[df[exp] == 0, out].mean()), 4),
            "diff_pp": round(100 * b[1], 3), "se_pp": round(100 * se[1], 3), "t": round(float(b[1] / se[1]), 2) if se[1] > 0 else None, "n_rows": int(len(df)), "n_exposed_rows": int(df[exp].sum())}
def did_yfe(df, pre, post, exp):
    st = pd.concat([df.assign(y=df[pre], post=0), df.assign(y=df[post], post=1)], ignore_index=True)
    yd_ = pd.get_dummies(st["by"].astype(int), drop_first=True).astype(float).values
    X = np.column_stack([np.ones(len(st)), st[exp].values, st["post"].values, (st[exp] * st["post"]).values, yd_, yd_ * st["post"].values[:, None]])
    b, se = twoway_ols(st["y"].values, X, st["src"].values, st["dst"].values)
    return {"pre_gap_pp": round(100 * b[1], 3), "se_pre_pp": round(100 * se[1], 3), "diff_in_gaps_pp": round(100 * b[3], 3), "se_pp": round(100 * se[3], 3), "t": round(float(b[3] / se[3]), 2) if se[3] > 0 else None}
def gap_yfe(df, out, exp):
    yd_ = pd.get_dummies(df["by"].astype(int), drop_first=True).astype(float).values
    X = np.column_stack([np.ones(len(df)), df[exp].values, yd_]); b, se = twoway_ols(df[out].values, X, df["src"].values, df["dst"].values)
    return {"diff_pp": round(100 * b[1], 3), "se_pp": round(100 * se[1], 3), "t": round(float(b[1] / se[1]), 2) if se[1] > 0 else None}
def did(df, pre, post, exp):
    st = pd.concat([df.assign(y=df[pre], post=0), df.assign(y=df[post], post=1)], ignore_index=True)
    X = np.column_stack([np.ones(len(st)), st[exp], st["post"], st[exp] * st["post"]]); b, se = twoway_ols(st["y"].values, X, st["src"].values, st["dst"].values)
    return {"pre_gap_pp": round(100 * b[1], 3), "diff_in_gaps_pp": round(100 * b[3], 3), "se_pp": round(100 * se[3], 3), "t": round(float(b[3] / se[3]), 2) if se[3] > 0 else None}

# ------------------------------------------------------------------ data
base = pd.read_csv(f"data/pair_base_k{SUFFIX}.csv", dtype={"src": str, "dst": str}, parse_dates=["birth", "death"])
rest = pd.read_csv("data/restatements_with_dates.csv", dtype={"company_fkey": str}); rest["cfkey"] = norm_cik(rest["company_fkey"])
for c in ("file_date", "res_begin_date", "res_end_date"): rest[c] = pd.to_datetime(rest[c], errors="coerce")
rest = rest.dropna(subset=["file_date"]); rest = rest[rest.file_date <= HORIZON]
rest["severe"] = (rest["res_fraud"].fillna(0).astype(float) > 0) | (rest["res_sec_investigation"].fillna(0).astype(float) > 0)
RF = defaultdict(list); RP = defaultdict(list)
for c, f, b, e, sv in zip(rest.cfkey, rest.file_date, rest.res_begin_date, rest.res_end_date, rest.severe):
    RF[c].append(f)
    if pd.notna(b): RP[c].append((b, e if pd.notna(e) else f, f, bool(sv)))
sec = pd.read_csv("data/sec_sic_by_cik.csv", dtype=str).fillna(""); sec["cik"] = norm_cik(sec["cik"]); SIC2 = sec.set_index("cik")["sic"].str.strip().str[:2].to_dict()
log(f"pairs {len(base):,}; restatements {len(rest):,}")

# tenure spells (same construction as module K) for seat lookups and future-career placebo
START_ACTIONS = {"Appointed", "Returned to Position"}
END_ACTIONS = {"Resigned", "Retired", "Employment Ceased", "Dismissed", "Declined Re-election", "Not Re-elected", "Deceased", "Appointment Revoked/Not Accepted"}
dirs = pd.read_csv("data/directors_officers_full.csv", usecols=["company_fkey", "first_name", "last_name", "eff_date", "action", "is_bdmem_pers"])
dirs["eff_date"] = pd.to_datetime(dirs["eff_date"], errors="coerce"); dirs = dirs.dropna(subset=["eff_date"])
dirs["person"] = dirs["first_name"].fillna("").str.strip().str.lower() + "|" + dirs["last_name"].fillna("").str.strip().str.lower(); dirs["cfkey"] = norm_cik(dirs["company_fkey"])
bd = dirs[(dirs["is_bdmem_pers"] == 1) & dirs["action"].isin(START_ACTIONS | END_ACTIONS)].copy(); bd["is_start"] = bd["action"].isin(START_ACTIONS)
bd = bd.sort_values(["person", "cfkey", "eff_date", "is_start"])
SPELLS = defaultdict(list); PERSON_SEATS = defaultdict(list)
if os.path.exists(f"data/tenure_spells_k{SUFFIX}.csv"):
    _ten = pd.read_csv(f"data/tenure_spells_k{SUFFIX}.csv", dtype={"company": str}, parse_dates=["start", "end"]); _ten["company"] = norm_cik(_ten["company"])
    for p_, c_, s_, e_ in zip(_ten["person"], _ten["company"], _ten["start"], _ten["end"]): SPELLS[(p_, c_)].append((s_, e_)); PERSON_SEATS[p_].append((c_, s_, e_))
    bd = bd.iloc[0:0]; log(f"spells: K's file for suffix {SUFFIX} ({len(_ten):,}; person key as in the pair base)")
for (p_, c_), g in bd.groupby(["person", "cfkey"], sort=False):
    start = None
    for a_, d_ in zip(g["action"].values, g["eff_date"].values):
        d_ = pd.Timestamp(d_)
        if a_ in START_ACTIONS:
            if start is None: start = d_
        elif start is not None and d_ > start:
            SPELLS[(p_, c_)].append((start, min(d_, HORIZON))); PERSON_SEATS[p_].append((c_, start, min(d_, HORIZON))); start = None
    if start is not None and start <= HORIZON: SPELLS[(p_, c_)].append((start, HORIZON)); PERSON_SEATS[p_].append((c_, start, HORIZON))
def seat_start(person, firm, t):
    for s_, e_ in SPELLS.get((person, firm), ()):
        if s_ <= t <= e_: return s_
    return None
log("spells rebuilt")

# ------------------------------------------------------------------ A. latent exposure classes (appointment direction, formed <= 2022)
eq = base[(base.src_is_origin == True) & (base.by <= 2022)].copy()
def classify(row):
    st = seat_start(row.director, row.src, row.birth)
    if st is None: st = row.birth
    b_ = row.birth
    latent = 0; latent_severe = 0; latent_file = None; label_only = 0; latent_buffer = 0
    if row.exposed5y == 0:
        for (bb, ee, ff, sv) in RP.get(row.src, ()):
            if ee < b_ and bb <= b_ and ee >= st and ff > b_:
                latent = 1; latent_severe = max(latent_severe, int(sv)); latent_file = ff if latent_file is None else min(latent_file, ff)
                if ff >= b_ + D6M: latent_buffer = 1
    if row.exposed5y == 1 and row.exposed_served == 0:
        # all origin restatements filed within 5y before the tie have misstated periods that ended before the director's origin seat began
        label_only = 1
    latent_buffer12 = 0
    if latent:
        latent_buffer12 = int(any(ee < b_ and bb <= b_ and ee >= st and ff >= b_ + 2 * D6M for (bb, ee, ff, sv) in RP.get(row.src, ())))
    return pd.Series({"latent": latent, "latent_severe": latent_severe, "latent_file": latent_file, "label_only": label_only, "seat_start": st, "latent_buffer": latent_buffer, "latent_buffer12": latent_buffer12})
cls = eq.apply(classify, axis=1); eq = pd.concat([eq, cls], axis=1)
eq["revealed"] = eq["exposed_served"].astype(int)
eq["present_filing_only"] = ((eq["exposed_present_at_filing"] == 1) & (eq["exposed_served"] == 0)).astype(int)
eq["clean_L"] = ((eq["exposed5y"] == 0) & (eq["latent"] == 0) & (eq["present_filing_only"] == 0)).astype(int)
eq["latent_noprior"] = ((eq["latent"] == 1) & (eq["exposed"] == 0)).astype(int)
def _dst_latent(d_, b_): return int(any(bb < b_ and ff > b_ for (bb, ee, ff, sv) in RP.get(d_, ())))
eq["dst_latent_at_tie"] = [_dst_latent(d_, b_) for d_, b_ in zip(eq.dst, eq.birth)]
def _cofile(d_, of_): return int(of_ is not None and any(abs((ff - of_).days) <= 92 for (bb, ee, ff, sv) in RP.get(d_, ())))
eq["cofile_quarter"] = [_cofile(d_, of_) if l_ == 1 else 0 for d_, of_, l_ in zip(eq.dst, eq.latent_file, eq.latent)]
eq["sic2_dst"] = eq["dst"].map(SIC2).fillna("")
log(f"classes: latent {eq.latent.sum():,} (severe {eq.latent_severe.sum():,}), revealed {eq.revealed.sum():,}, label-only {eq.label_only.sum():,}, present-at-filing-only {eq.present_filing_only.sum():,}, clean {eq.clean_L.sum():,}")

def stacked_classes(df, pre, post, classes, extra_fe=("by",), demean_origin=False):
    """stacked pre/post regression with class dummies and class x post; formation-year x post FE (+ destination industry x post); returns per-class pre gap and change"""
    st = pd.concat([df.assign(y=df[pre], post=0), df.assign(y=df[post], post=1)], ignore_index=True)
    cols = [st["post"].values]
    for c in classes: cols += [st[c].values, (st[c] * st["post"]).values]
    X = np.column_stack([np.ones(len(st)), *cols])
    fe_blocks = []
    for f_ in extra_fe:
        dm = pd.get_dummies(st[f_].astype(str), drop_first=True).astype(float).values
        fe_blocks += [dm, dm * st["post"].values[:, None]]
    X = np.column_stack([X, *fe_blocks]) if fe_blocks else X
    y = st["y"].values.astype(float)
    if demean_origin:   # origin fixed effect by within transformation (FWL)
        grp = st["src"].values; keys, inv = np.unique(grp, return_inverse=True)
        def dm_(v):
            v = np.asarray(v, float); sums = np.zeros((len(keys),) + v.shape[1:]); np.add.at(sums, inv, v); cnt = np.bincount(inv).astype(float)
            return v - (sums / cnt[:, None] if v.ndim > 1 else sums / cnt)[inv]
        X = dm_(X[:, 1:]); y = dm_(y); X = np.column_stack([np.full(len(st), 1e-12), X])  # intercept absorbed by the origin means
    b, se = twoway_ols(y, X, st["src"].values, st["dst"].values)
    out = {}
    for i, c in enumerate(classes):
        j = 2 + 2 * i
        out[c] = {"pre_gap_pp": round(100 * b[j], 3), "se_pre_pp": round(100 * se[j], 3), "change_pp": round(100 * b[j + 1], 3), "se_change_pp": round(100 * se[j + 1], 3),
                  "post_gap_pp": round(100 * (b[j] + b[j + 1]), 3), "n": int(df[c].sum())}
    out["n_rows"] = int(len(df)); out["n_clean"] = int(len(df) - df[classes].sum().sum())
    return out
CL = ["latent", "revealed", "label_only", "present_filing_only"]
bc_ciks0 = set(sec.loc[(sec["sic"].str.strip() == "6770") | (sec["name"] + " | " + sec["former_names"]).str.lower().str.contains(r"acquisition\s+(?:corp|co\b|company|ltd|limited|holdings|inc|plc|partners|group)|blank\s*check", regex=True), "cik"])
eq["bc_pair"] = (eq.src.isin(bc_ciks0) | eq.dst.isin(bc_ciks0)).astype(int)
A_ = {"n_rows": int(len(eq)), "n_by_class": {c: int(eq[c].sum()) for c in CL + ["clean_L"]}, "share_latent_of_unexposed": round(float(eq.loc[eq.exposed == 0, "latent"].mean()), 4),
      "share_latent_pairs_blank_check": round(float(eq.loc[eq.latent == 1, "bc_pair"].mean()), 4), "latent_pairs_by_year": eq.loc[eq.latent == 1, "by"].value_counts().sort_index().to_dict()}
A_["n_latent_buffer_6m"] = int(eq.latent_buffer.sum()); A_["n_latent_buffer_12m"] = int(eq.latent_buffer12.sum()); A_["n_latent_no_origin_filing_ever"] = int(eq.latent_noprior.sum())
latL = eq[eq.latent == 1]
A_["latent_diagnostics"] = {"n": int(len(latL)), "share_blank_check_pair": round(float(latL.bc_pair.mean()), 4), "share_destination_also_latent": round(float(latL.dst_latent_at_tie.mean()), 4), "share_destination_also_latent_excl_blank_check": round(float(latL.loc[latL.bc_pair == 0, "dst_latent_at_tie"].mean()), 4), "destination_latent_share_among_clean_excl_blank_check": round(float(eq.loc[(eq.clean_L == 1) & (eq.bc_pair == 0), "dst_latent_at_tie"].mean()), 4),
                            "share_destination_files_within_quarter_of_origin": round(float(latL.cofile_quarter.mean()), 4), "share_same_sic2": round(float((latL.src.map(SIC2) == latL.dst.map(SIC2)).mean()), 4),
                            "share_same_auditor_ever": round(float(latL.same_auditor_ever.mean()), 4), "share_formed_2020_2022": round(float((latL.by >= 2020).mean()), 4),
                            "origin_filing_months_after_tie_quartiles": [round(float(x), 1) for x in ((latL.latent_file - latL.birth).dt.days / 30.44).quantile([0.25, 0.5, 0.75]).values],
                            "destination_latent_share_among_clean": round(float(eq.loc[eq.clean_L == 1, "dst_latent_at_tie"].mean()), 4)}
def _rates(df_): return {"n": int(len(df_)), "pre_onset_rate": round(float(df_.pre_onset_3y.mean()), 4), "post_onset_rate": round(float(df_.out_onset_3y.mean()), 4), "pre_file_rate": round(float(df_.pre_file_3y.mean()), 4), "post_file_rate": round(float(df_.out_file_3y.mean()), 4)}
clL = eq[eq.clean_L == 1]
A_["latent_rates"] = {"clean": _rates(clL), "clean_destination_not_latent": _rates(clL[clL.dst_latent_at_tie == 0]), "latent_all": _rates(latL), "latent_excl_blank_check": _rates(latL[latL.bc_pair == 0]),
                      "latent_excl_bc_and_destination_latent": _rates(latL[(latL.bc_pair == 0) & (latL.dst_latent_at_tie == 0)]), "latent_excl_bc_and_cofiling": _rates(latL[(latL.bc_pair == 0) & (latL.cofile_quarter == 0)]),
                      "latent_buffer_6m_excl_bc": _rates(latL[(latL.bc_pair == 0) & (latL.latent_buffer == 1)]), "latent_buffer_12m_excl_bc": _rates(latL[(latL.bc_pair == 0) & (latL.latent_buffer12 == 1)]),
                      "latent_buffer_6m_excl_bc_and_destination_latent": _rates(latL[(latL.bc_pair == 0) & (latL.latent_buffer == 1) & (latL.dst_latent_at_tie == 0)])}
# gaps vs clean destinations that were not themselves latent, for the cleaned latent subsets
for lab_, m_ in (("excl_bc_and_destination_latent", (latL.bc_pair == 0) & (latL.dst_latent_at_tie == 0)), ("buffer_6m_excl_bc_and_destination_latent", (latL.bc_pair == 0) & (latL.latent_buffer == 1) & (latL.dst_latent_at_tie == 0)), ("buffer_12m_excl_bc", (latL.bc_pair == 0) & (latL.latent_buffer12 == 1))):
    sub_ = pd.concat([latL[m_], clL[clL.dst_latent_at_tie == 0]]); sub_ = sub_.assign(lat=sub_.latent)
    A_[f"latent_gap_{lab_}"] = {"n_latent": int(m_.sum()), "pre_onset": gap(sub_, "pre_onset_3y", "lat"), "post_onset": gap(sub_, "out_onset_3y", "lat"), "did_onset": did(sub_, "pre_onset_3y", "out_onset_3y", "lat"),
                                "post_file": gap(sub_, "out_file_3y", "lat"), "history_free_post_onset": gap(sub_[(sub_.dst_prior_any == 0)], "out_onset_3y", "lat")}
# symmetric-comparator partition table: each restriction applied to BOTH the latent group and the clean comparator
def units(df_): return {"pairs": int(len(df_)), "appointments": int(df_.groupby(["director", "dst", "birth"]).ngroups), "destinations": int(df_.dst.nunique()), "directors": int(df_.director.nunique()), "origins": int(df_.src.nunique())}
def partition(mask_fn, lab):
    lat_ = latL[mask_fn(latL)]; cl_ = clL[mask_fn(clL)]
    sub_ = pd.concat([lat_, cl_]).assign(lat=lambda d_: d_.latent)
    hf = sub_[sub_.dst_prior_any == 0]
    return {"label": lab, "latent_units": units(lat_), "comparator_units": units(cl_),
            "rates": {"latent_pre_onset": round(float(lat_.pre_onset_3y.mean()), 4), "latent_post_onset": round(float(lat_.out_onset_3y.mean()), 4), "clean_pre_onset": round(float(cl_.pre_onset_3y.mean()), 4), "clean_post_onset": round(float(cl_.out_onset_3y.mean()), 4),
                      "latent_post_file": round(float(lat_.out_file_3y.mean()), 4), "clean_post_file": round(float(cl_.out_file_3y.mean()), 4)},
            "pre_onset": gap(sub_, "pre_onset_3y", "lat"), "post_onset": gap(sub_, "out_onset_3y", "lat"), "change_onset": did(sub_, "pre_onset_3y", "out_onset_3y", "lat"),
            "pre_onset_yfe": gap_yfe(sub_, "pre_onset_3y", "lat"), "post_onset_yfe": gap_yfe(sub_, "out_onset_3y", "lat"), "change_onset_yfe": did_yfe(sub_, "pre_onset_3y", "out_onset_3y", "lat"),
            "post_file": gap(sub_, "out_file_3y", "lat"), "post_file_yfe": gap_yfe(sub_, "out_file_3y", "lat"), "change_file_yfe": did_yfe(sub_, "pre_file_3y", "out_file_3y", "lat"),
            "history_free_post_onset": gap(hf, "out_onset_3y", "lat") if hf.lat.sum() > 0 else None, "history_free_post_onset_yfe": gap_yfe(hf, "out_onset_3y", "lat") if hf.lat.sum() > 0 else None, "history_free_n_latent": int(hf.lat.sum())}
PARTS = [("all", lambda d_: d_.index == d_.index), ("excl_blank_check", lambda d_: d_.bc_pair == 0), ("excl_bc_and_destination_latent", lambda d_: (d_.bc_pair == 0) & (d_.dst_latent_at_tie == 0)),
         ("excl_bc_buffer6m", lambda d_: (d_.bc_pair == 0) & ((d_.latent == 0) | (d_.latent_buffer == 1))), ("excl_bc_destination_latent_buffer6m", lambda d_: (d_.bc_pair == 0) & (d_.dst_latent_at_tie == 0) & ((d_.latent == 0) | (d_.latent_buffer == 1))),
         ("excl_bc_buffer12m", lambda d_: (d_.bc_pair == 0) & ((d_.latent == 0) | (d_.latent_buffer12 == 1))), ("excl_bc_destination_latent_buffer12m", lambda d_: (d_.bc_pair == 0) & (d_.dst_latent_at_tie == 0) & ((d_.latent == 0) | (d_.latent_buffer12 == 1)))]
A_["partitions"] = {lab: partition(fn, lab) for lab, fn in PARTS}
# episode clocks for the destination episodes of latent pairs filed after the appointment (indicators; a pair may have several episodes)
def clocks(d_, b_, of_):
    out_ = {"legacy_ended_before_appointment": 0, "began_before_continued_across": 0, "began_after_appointment_before_origin_filing": 0, "began_after_origin_filing": 0}
    for (bb, ee, ff, sv) in RP.get(d_, ()):
        if ff <= b_ or ff > b_ + D3: continue
        if ee < b_: out_["legacy_ended_before_appointment"] = 1
        elif bb < b_ <= ee: out_["began_before_continued_across"] = 1
        elif b_ <= bb < of_: out_["began_after_appointment_before_origin_filing"] = 1
        elif bb >= of_: out_["began_after_origin_filing"] = 1
    return out_
ck = pd.DataFrame([clocks(d_, b_, of_) for d_, b_, of_ in zip(latL.dst, latL.birth, latL.latent_file)])
A_["latent_destination_episode_clocks"] = {"n_latent_pairs": int(len(latL)), **{k: int(v) for k, v in ck.sum().items()}, "share_any_post_filing": round(float((ck.sum(axis=1) > 0).mean()), 4)}
log(f"partitions: " + "; ".join(f"{k}: n={v['latent_units']['pairs']} post onset {v['post_onset']['diff_pp']:+.2f} ({v['post_onset']['se_pp']:.2f}) yfe {v['post_onset_yfe']['diff_pp']:+.2f}, change yfe {v['change_onset_yfe']['diff_in_gaps_pp']:+.2f} ({v['change_onset_yfe']['se_pp']:.2f})" for k, v in A_["partitions"].items()))
log(f"latent diagnostics: {A_['latent_diagnostics']}; cleaned latent gap post onset {A_['latent_gap_excl_bc_and_destination_latent']['post_onset']['diff_pp']:+.2f} ({A_['latent_gap_excl_bc_and_destination_latent']['post_onset']['se_pp']:.2f})")
for lab, sub in (("all", eq), ("no_blank_check", eq[eq.bc_pair == 0]), ("formed_by_2016", eq[eq.by <= 2016]), ("latent_nonsevere", eq[(eq.latent == 0) | (eq.latent_severe == 0)]), ("latent_buffer_6m", eq[(eq.latent == 0) | (eq.latent_buffer == 1)]), ("latent_no_origin_filing_ever", eq[(eq.latent == 0) | (eq.latent_noprior == 1)])):
    A_[f"onset_{lab}"] = stacked_classes(sub, "pre_onset_3y", "out_onset_3y", CL, extra_fe=("by",))
    A_[f"file_{lab}"] = stacked_classes(sub, "pre_file_3y", "out_file_3y", CL, extra_fe=("by",))
A_["onset_all_dstindustry"] = stacked_classes(eq, "pre_onset_3y", "out_onset_3y", CL, extra_fe=("by", "sic2_dst"))
A_["onset_all_origin_demeaned"] = stacked_classes(eq, "pre_onset_3y", "out_onset_3y", CL, extra_fe=("by",), demean_origin=True)
# latent vs clean only (simple gaps) and never-filed destinations
lc = eq[((eq.latent == 1) | (eq.clean_L == 1)) & (eq.bc_pair == 0)]
A_["latent_vs_clean_no_blank_check"] = {"n_rows": int(len(lc)), "n_latent": int(lc.latent.sum()), "did_onset_yearFE": None}
lc_all = eq[(eq.latent == 1) | (eq.clean_L == 1)]
lc = lc_all
A_["latent_vs_clean"] = {"n_rows": int(len(lc)), "pre_onset": gap(lc, "pre_onset_3y", "latent"), "post_onset": gap(lc, "out_onset_3y", "latent"), "did_onset": did(lc, "pre_onset_3y", "out_onset_3y", "latent"),
                         "pre_file": gap(lc, "pre_file_3y", "latent"), "post_file": gap(lc, "out_file_3y", "latent"), "did_file": did(lc, "pre_file_3y", "out_file_3y", "latent"),
                         "never_filed_post_onset": gap(lc[lc.dst_prior_any == 0], "out_onset_3y", "latent"), "never_filed_pre_onset": gap(lc[lc.dst_prior_any == 0], "pre_onset_3y", "latent")}
# destination onsets in the post window that begin BEFORE the origin's (latent) filing — cannot be a reaction to the revelation
def onset_before(dst, b_, f_):
    return int(any(b_ <= bb < min(f_, b_ + D3) for (bb, ee, ff, sv) in RP.get(dst, ())))
lat = eq[eq.latent == 1].copy(); lat["onset_before_origin_filing"] = [onset_before(d_, b_, f_) for d_, b_, f_ in zip(lat.dst, lat.birth, lat.latent_file)]
A_["latent_onsets_before_origin_filing"] = {"share_of_latent_pairs_with_post_onset": round(float(lat.out_onset_3y.mean()), 4), "share_with_onset_before_origin_filing": round(float(lat.onset_before_origin_filing.mean()), 4),
                                            "median_months_tie_to_origin_filing": round(float(((lat.latent_file - lat.birth).dt.days / 30.44).median()), 1)}
# MDE for the latent design
p_ = float(eq.out_onset_3y.mean()); nl = int(eq.latent.sum()); nc = int(eq.clean_L.sum())
A_["mde_pp_level_80pct_power_naive"] = round(100 * 2.8 * np.sqrt(p_ * (1 - p_) * (1 / max(nl, 1) + 1 / max(nc, 1))), 2)
OUT["A_latent_exposure"] = A_
log(f"A: latent vs clean onset DiD {A_['latent_vs_clean']['did_onset']['diff_in_gaps_pp']:+.2f} ({A_['latent_vs_clean']['did_onset']['se_pp']:.2f}); classes change: " + ", ".join(f"{c} {A_['onset_all'][c]['change_pp']:+.2f}" for c in CL))

# ------------------------------------------------------------------ B. channel ordering, anticipation, detection, waves, placebos, symmetry
B_ = {}
# B1 anticipation: latent destination trouble (misstated period begun before the tie, filed after) predicts hiring an exposed director?
def dst_latent(dst, b_): return int(any(bb < b_ and ff > b_ for (bb, ee, ff, sv) in RP.get(dst, ())))
eq["dst_latent"] = [dst_latent(d_, b_) for d_, b_ in zip(eq.dst, eq.birth)]
eq["dst_filed_12m"] = [int(any(b_ - D1Y <= f_ < b_ for f_ in RF.get(d_, ()))) for d_, b_ in zip(eq.dst, eq.birth)]
yd = pd.get_dummies(eq["by"].astype(int), drop_first=True).astype(float).values
for ex_ in ("exposed", "exposed5y", "revealed"):
    X = np.column_stack([np.ones(len(eq)), eq["dst_latent"].values, eq["dst_filed_12m"].values, eq["dst_prior_any"].values, yd]); b, se = twoway_ols(eq[ex_].values, X, eq["src"].values, eq["dst"].values)
    B_[f"anticipation_{ex_}"] = {"latent_dst_coef_pp": round(100 * b[1], 3), "se_pp": round(100 * se[1], 3), "filed12m_coef_pp": round(100 * b[2], 3), "se_filed12m_pp": round(100 * se[2], 3),
                                 "own_history_coef_pp": round(100 * b[3], 3), "share_dst_latent": round(float(eq.dst_latent.mean()), 4), "exposed_share_if_dst_latent": round(float(eq.loc[eq.dst_latent == 1, ex_].mean()), 4), "exposed_share_otherwise": round(float(eq.loc[eq.dst_latent == 0, ex_].mean()), 4)}
log(f"B1 anticipation: latent-destination coef on exposed5y {B_['anticipation_exposed5y']['latent_dst_coef_pp']:+.2f} ({B_['anticipation_exposed5y']['se_pp']:.2f}); filed12m {B_['anticipation_exposed5y']['filed12m_coef_pp']:+.2f}")
# B2 detection hazard: among latently misstating destinations, time from tie to the filing that reveals it, by exposure
ld = eq[eq.dst_latent == 1].copy()
def first_reveal(dst, b_):
    fs = [ff for (bb, ee, ff, sv) in RP.get(dst, ()) if bb < b_ and ff > b_]; return min(fs)
ld["t_reveal"] = [((first_reveal(d_, b_) - b_).days / 365.25) for d_, b_ in zip(ld.dst, ld.birth)]; ld["ev"] = 1; ld["cl"] = pd.factorize(ld.dst)[0]
B_["detection_hazard"] = {"n_latent_destinations_pairs": int(len(ld)), "median_years_to_reveal": round(float(ld.t_reveal.median()), 2)}
for ex_ in ("exposed", "exposed5y", "revealed"):
    try:
        c = CoxPHFitter().fit(ld[["t_reveal", "ev", ex_, "by", "cl"]].rename(columns={ex_: "x"}), duration_col="t_reveal", event_col="ev", strata=["by"], cluster_col="cl", robust=True)
        ci = np.exp(c.confidence_intervals_.loc["x"].values)
        B_["detection_hazard"][ex_] = {"hr": round(float(np.exp(c.params_["x"])), 3), "ci95": [round(float(ci[0]), 3), round(float(ci[1]), 3)], "p": float(c.summary.loc["x", "p"])}
    except Exception as e_: B_["detection_hazard"][ex_] = {"error": type(e_).__name__}
log(f"B2 detection: HR exposed5y {B_['detection_hazard'].get('exposed5y')}")
# B3 wave-cleaned exposure: quarters where filings exceed 2x the trailing 8-quarter median, plus blank-check Apr-Jun 2021
q = rest.set_index("file_date").resample("QE").size(); med8 = q.rolling(8, min_periods=4).median().shift(1); wave_q = set(q.index[(q > 2 * med8).values])
bc_ciks = set(sec.loc[(sec["sic"].str.strip() == "6770") | (sec["name"] + " | " + sec["former_names"]).str.lower().str.contains(r"acquisition\s+(?:corp|co\b|company|ltd|limited|holdings|inc|plc|partners|group)|blank\s*check", regex=True), "cik"])
def is_wave(c_, f_): return (f_.to_period("Q").to_timestamp(how="end").normalize() in {d.normalize() for d in wave_q}) or (c_ in bc_ciks and pd.Timestamp("2021-04-12") <= f_ <= pd.Timestamp("2021-06-30"))
WQ = {d.normalize() for d in wave_q}
def exp5_split(src, b_):
    fs = [f_ for f_ in RF.get(src, ()) if b_ - pd.Timedelta(days=5 * 365) <= f_ < b_]
    if not fs: return (0, 0)
    nonwave = any(not ((f_.to_period("Q").to_timestamp(how="end").normalize() in WQ) or (src in bc_ciks and pd.Timestamp("2021-04-12") <= f_ <= pd.Timestamp("2021-06-30"))) for f_ in fs)
    return (int(nonwave), int(not nonwave))
sp = [exp5_split(s_, b_) for s_, b_ in zip(eq.src, eq.birth)]; eq["exp5_nonwave"] = [a for a, b in sp]; eq["exp5_waveonly"] = [b for a, b in sp]
B_["wave_cleaned"] = {"n_wave_quarters": len(wave_q), "wave_quarters": sorted(str(d.date()) for d in wave_q), "share_exposed5y_waveonly": round(float(eq.exp5_waveonly.sum() / max(eq.exposed5y.sum(), 1)), 4)}
for lab, col, keep in (("nonwave", "exp5_nonwave", eq[(eq.exp5_waveonly == 0)]), ("waveonly", "exp5_waveonly", eq[(eq.exp5_nonwave == 0)])):
    B_["wave_cleaned"][lab] = {"n_exposed": int(keep[col].sum()), "pre_onset": gap(keep, "pre_onset_3y", col), "post_onset": gap(keep, "out_onset_3y", col), "did_onset": did(keep, "pre_onset_3y", "out_onset_3y", col),
                               "clean_dst_post_onset": gap(keep[(keep.dst_prior_any == 0) & (keep.pre_onset == 0)], "out_onset_3y", col)}
log(f"B3 waves: {len(wave_q)} quarters; nonwave DiD onset {B_['wave_cleaned']['nonwave']['did_onset']['diff_in_gaps_pp']:+.2f}, wave-only {B_['wave_cleaned']['waveonly']['did_onset']['diff_in_gaps_pp']:+.2f}")
# B4 past-outcome placebo: destination filings/onsets in [birth-6y, birth-3y)
def any_in(dates, lo, hi): return int(any(lo <= d_ < hi for d_ in dates))
eq["pre_file_3to6y"] = [any_in(RF.get(d_, ()), b_ - 2 * D3, b_ - D3) for d_, b_ in zip(eq.dst, eq.birth)]
eq["pre_onset_3to6y"] = [int(any(b_ - 2 * D3 <= bb < b_ - D3 for (bb, ee, ff, sv) in RP.get(d_, ()))) for d_, b_ in zip(eq.dst, eq.birth)]
B_["past_outcome_placebo"] = {}
lc = eq[(eq.latent == 1) | (eq.clean_L == 1)]   # refresh with the new columns
for ex_ in ("exposed", "exposed5y", "revealed", "latent"):
    sub = eq if ex_ != "latent" else lc
    B_["past_outcome_placebo"][ex_] = {"file_6to3y_before": gap(sub, "pre_file_3to6y", ex_), "onset_6to3y_before": gap(sub, "pre_onset_3to6y", ex_), "file_3y_before": gap(sub, "pre_file_3y", ex_), "onset_3y_before": gap(sub, "pre_onset_3y", ex_)}
# B5 future-career-link placebo: director later (after birth+3y) joins a firm that had restated within 5y before that later seat start
def future_link(person, b_, exclude):
    for (c_, s_, e_) in PERSON_SEATS.get(person, ()):
        if s_ > b_ + D3 and c_ not in exclude:
            if any(s_ - pd.Timedelta(days=5 * 365) <= f_ < s_ for f_ in RF.get(c_, ())): return 1
    return 0
eq["future_exposed_link"] = [future_link(p_, b_, {s_, d_}) for p_, b_, s_, d_ in zip(eq.director, eq.birth, eq.src, eq.dst)]
eq["has_future_seat"] = [int(any(s_ > b_ + D3 and c_ not in {s0, d0} for (c_, s_, e_) in PERSON_SEATS.get(p_, ()))) for p_, b_, s0, d0 in zip(eq.director, eq.birth, eq.src, eq.dst)]
fl = eq[eq.has_future_seat == 1]
B_["future_career_link_placebo"] = {"n_pairs_with_future_seat": int(len(fl)), "share_future_exposed": round(float(fl.future_exposed_link.mean()), 4),
                                    "post_onset": gap(fl, "out_onset_3y", "future_exposed_link"), "post_file": gap(fl, "out_file_3y", "future_exposed_link"), "pre_onset": gap(fl, "pre_onset_3y", "future_exposed_link"),
                                    "did_onset": did(fl, "pre_onset_3y", "out_onset_3y", "future_exposed_link"),
                                    "clean_dst_post_onset": gap(fl[(fl.dst_prior_any == 0) & (fl.pre_onset == 0)], "out_onset_3y", "future_exposed_link")}
log(f"B5 future link: post onset gap {B_['future_career_link_placebo']['post_onset']['diff_pp']:+.2f} ({B_['future_career_link_placebo']['post_onset']['se_pp']:.2f}); n {len(fl):,}")
# B6 symmetric forward/reverse by at-risk firm's history
rv = base[(base.src_is_origin == False) & (base.by <= 2022)].copy()
def blk(df, ex_):
    return {"n_rows": int(len(df)), "pre_onset": gap(df, "pre_onset_3y", ex_), "post_onset": gap(df, "out_onset_3y", ex_), "did_onset": did(df, "pre_onset_3y", "out_onset_3y", ex_),
            "pre_file": gap(df, "pre_file_3y", ex_), "post_file": gap(df, "out_file_3y", ex_), "did_file": did(df, "pre_file_3y", "out_file_3y", ex_)}
B_["symmetry"] = {}
for ex_ in ("exposed", "exposed5y"):
    B_["symmetry"][ex_] = {"forward_all": blk(eq, ex_), "forward_history_free": blk(eq[eq.dst_prior_any == 0], ex_), "forward_with_history": blk(eq[eq.dst_prior_any == 1], ex_),
                           "reverse_all": blk(rv, ex_), "reverse_history_free": blk(rv[rv.dst_prior_any == 0], ex_), "reverse_with_history": blk(rv[rv.dst_prior_any == 1], ex_)}
# B7 two-step network placebo in the firm-year panel
if os.path.exists(f"data/edge_lifecycles_k{SUFFIX}.csv"):
    edf = pd.read_csv(f"data/edge_lifecycles_k{SUFFIX}.csv", dtype={"company_a": str, "company_b": str}, parse_dates=["edge_birth", "edge_death"]); log(f"edges: K's sample-filtered file for suffix {SUFFIX} ({len(edf):,})")
else:
    edf = pd.read_csv("data/edge_lifecycles_k.csv", dtype={"company_a": str, "company_b": str}, parse_dates=["edge_birth", "edge_death"])
    if MAX_FIRMS: edf = edf[edf.director_firms <= MAX_FIRMS]
recent = defaultdict(set)   # firm -> set of years y such that firm filed in (y-5, y]
for c_, fs in RF.items():
    for f_ in fs:
        for y in range(f_.year, f_.year + 6): recent[y].add(c_)
rows = []
for y in range(2005, 2024):
    jan1 = pd.Timestamp(f"{y}-01-01"); al = edf[(edf.edge_birth < jan1) & (edf.edge_death >= jan1)]
    nb = defaultdict(set)
    for a_, b_ in zip(al.company_a, al.company_b): nb[a_].add(b_); nb[b_].add(a_)
    rec = recent[y]
    for j, n1 in nb.items():
        n2 = set().union(*(nb[k] for k in n1)) - n1 - {j}
        rows.append((j, y, len(n1), sum(1 for k in n1 if k in rec) / len(n1), len(n2), (sum(1 for k in n2 if k in rec) / len(n2)) if n2 else 0.0))
ts = pd.DataFrame(rows, columns=["cik", "year", "deg", "direct_recent_share", "n2", "twostep_recent_share"])
pnl = pd.read_csv(f"data/integrated_panel_k{SUFFIX}.csv", dtype={"cik": str}); pnl["cik"] = norm_cik(pnl["cik"])
pnl = pnl.merge(ts.assign(year=ts.year + 1).rename(columns={"direct_recent_share": "direct_recent_share_lag1", "twostep_recent_share": "twostep_recent_share_lag1", "n2": "n2_lag1", "deg": "deg_lag1"}), on=["cik", "year"], how="left")
for c in ("direct_recent_share_lag1", "twostep_recent_share_lag1", "n2_lag1", "deg_lag1"): pnl[c] = pnl[c].fillna(0)
pp_ = pnl.set_index(["gvkey", "year"]); ctrl = ["degree_lag", "betweenness_lag", "size", "leverage", "roa"]
def fe(dep, exog, data=pp_): return PanelOLS(data[dep], data[exog], entity_effects=True, time_effects=True, drop_absorbed=True).fit(cov_type="clustered", cluster_entity=True)
def cb(r, k): return {"coef_pp": round(100 * float(r.params[k]), 3), "se_pp": round(100 * float(r.std_errors[k]), 3), "t": round(float(r.tstats[k]), 2)}
B_["two_step_placebo"] = {"mean_twostep_recent_share": round(float(pnl.twostep_recent_share_lag1.mean()), 4), "mean_direct_recent_share": round(float(pnl.direct_recent_share_lag1.mean()), 4)}
for dep in ("restate_onset", "restate_file"):
    r = fe(dep, ["nexp5_lag1", "twostep_recent_share_lag1", "nnew_lag1", "n2_lag1"] + ctrl + ["had_prior_restate"])
    B_["two_step_placebo"][dep] = {"recent_exposed_ties_lag1": cb(r, "nexp5_lag1"), "twostep_recent_share_lag1": cb(r, "twostep_recent_share_lag1"), "n": int(r.nobs)}
    r2 = fe(dep, ["direct_recent_share_lag1", "twostep_recent_share_lag1", "deg_lag1", "n2_lag1"] + ctrl + ["had_prior_restate"])
    B_["two_step_placebo"][dep]["direct_recent_share_lag1_alt"] = cb(r2, "direct_recent_share_lag1"); B_["two_step_placebo"][dep]["twostep_recent_share_lag1_alt"] = cb(r2, "twostep_recent_share_lag1")
q_ = pnl["twostep_recent_share_lag1"].quantile([0.25, 0.75]).values; qd = pnl["direct_recent_share_lag1"].quantile([0.25, 0.75]).values; qn = pnl["nexp5_lag1"].quantile([0.25, 0.75, 0.9, 0.99]).values
B_["two_step_placebo"]["scales"] = {"twostep_share_iqr": [round(float(q_[0]), 4), round(float(q_[1]), 4)], "direct_share_iqr": [round(float(qd[0]), 4), round(float(qd[1]), 4)], "nexp5_lag1_quantiles_25_75_90_99": [round(float(x), 2) for x in qn],
                                    "share_firm_years_with_recent_exposed_tie": round(float((pnl["nexp5_lag1"] > 0).mean()), 4), "share_firm_years_with_twostep_share_gt0": round(float((pnl["twostep_recent_share_lag1"] > 0).mean()), 4)}
r_ = B_["two_step_placebo"]["restate_onset"]
B_["two_step_placebo"]["scales"]["onset_effect_pp_per_10pp_twostep_share"] = round(r_["twostep_recent_share_lag1"]["coef_pp"] * 0.10, 3)
B_["two_step_placebo"]["scales"]["onset_effect_pp_twostep_iqr"] = round(r_["twostep_recent_share_lag1"]["coef_pp"] * float(q_[1] - q_[0]), 3)
B_["two_step_placebo"]["scales"]["onset_effect_pp_one_recent_exposed_tie"] = r_["recent_exposed_ties_lag1"]["coef_pp"]
log(f"B7 two-step: onset coef on two-step share {B_['two_step_placebo']['restate_onset']['twostep_recent_share_lag1']}; direct ties {B_['two_step_placebo']['restate_onset']['recent_exposed_ties_lag1']}")
OUT["B_falsification"] = B_

# ------------------------------------------------------------------ C. frozen-network origin-disclosure event study
rng = np.random.default_rng(7)
events = []
for c_, fs in RF.items():
    fs_ = sorted(fs)
    for i, f_ in enumerate(fs_):
        if f_ < pd.Timestamp("2008-01-01") or f_ > pd.Timestamp("2022-12-31"): continue
        if any(f_ - D3 <= g_ < f_ for g_ in fs_[:i]): continue   # fresh disclosure: no filing in the prior 3 years
        events.append((c_, f_))
ev = pd.DataFrame(events, columns=["origin", "f"]); log(f"C: {len(ev):,} fresh origin disclosures 2008-2022")
# links alive at the freeze date f-12m (either direction, any lifecycle in the capped set)
e2 = edf[["company_a", "company_b", "edge_birth", "edge_death", "director"]]
lk = []
ea = e2.rename(columns={"company_a": "o", "company_b": "j"}); eb = e2.rename(columns={"company_b": "o", "company_a": "j"})
und = pd.concat([ea, eb], ignore_index=True)
mg = ev.merge(und, left_on="origin", right_on="o")
mg = mg[(mg.edge_birth <= mg.f - D1Y) & (mg.edge_death >= mg.f - D1Y)]
# served during the misstated period of this disclosure?
RPd = defaultdict(list)
for c_, f_, b_, e_, sv in zip(rest.cfkey, rest.file_date, rest.res_begin_date, rest.res_end_date, rest.severe):
    RPd[(c_, f_)].append((b_, e_ if pd.notna(e_) else f_))
def served(director, origin, f_):
    per = RPd.get((origin, f_), [])
    for (s_, e_) in SPELLS.get((director, origin), ()):
        if any(pd.notna(bb) and s_ <= ee and e_ >= bb for bb, ee in per): return 1
    return 0
mg["served"] = [served(d_, o_, f_) for d_, o_, f_ in zip(mg.director, mg.origin, mg.f)]
treated = mg.groupby(["origin", "f", "j"]).agg(served=("served", "max")).reset_index()
log(f"C: {len(treated):,} linked destination-event pairs ({treated.j.nunique():,} destinations)")
# controls: firms in the feed at the freeze date, not linked to this origin, matched on prior-restatement history (any filing before freeze) — 3 per treated
FIRST_EVENT = dirs.groupby("cfkey")["eff_date"].min().to_dict(); LAST_EVENT = dirs.groupby("cfkey")["eff_date"].max().to_dict()
all_firms = np.array(list(FIRST_EVENT.keys())); first_arr = np.array([FIRST_EVENT[c] for c in all_firms]); last_arr = np.array([LAST_EVENT[c] for c in all_firms])
RFsorted = {c_: np.array(sorted(fs), dtype="datetime64[ns]") for c_, fs in RF.items()}
def filed_before(c_, t): a = RFsorted.get(c_); return int(a is not None and (a < np.datetime64(t)).any())
def hist_cell(c_, t):
    # timing-aware restatement history at the freeze: filed within the three years before it (x2) + filed earlier than that (x1)
    a = RFsorted.get(c_)
    if a is None: return 0
    t64 = np.datetime64(t); recent = int(((a < t64) & (a >= t64 - np.timedelta64(3 * 365, "D"))).any()); earlier = int((a < t64 - np.timedelta64(3 * 365, "D")).any())
    return 2 * recent + earlier
rows = []; n_treated_before_survivorship = int(len(treated)); n_dropped_treated = 0
for (o_, f_), grp in treated.groupby(["origin", "f"]):
    frz = f_ - D1Y
    # eligibility uses feed presence at the freeze only (first event at or before the freeze, last event at or after it); exits after the freeze stay in the sample for both groups
    elig = all_firms[(first_arr <= frz) & (last_arr >= frz)]
    linked = set(grp.j) | {o_}
    keep = [(FIRST_EVENT.get(j, pd.Timestamp.max) <= frz) and (LAST_EVENT.get(j, pd.Timestamp.min) >= frz) for j in grp.j]
    n_dropped_treated += int(len(grp) - sum(keep)); grp = grp[keep]
    if len(grp) == 0: continue
    elig = np.array([c for c in elig if c not in linked])
    if len(elig) < 3 * len(grp): continue
    hist_e = np.array([hist_cell(c, frz) for c in elig]); hist_e_coarse = (hist_e > 0).astype(int)
    for j_, sv_ in zip(grp.j, grp.served):
        hj = hist_cell(j_, frz); pool = elig[hist_e == hj]; exact = 1
        if len(pool) < 3: pool = elig[hist_e_coarse == int(hj > 0)]; exact = 0
        if len(pool) < 3: pool = elig
        ctrls = rng.choice(pool, size=3, replace=False)
        rows.append((o_, f_, j_, 1, sv_, hj, exact)); rows += [(o_, f_, c, 0, sv_, hj, exact) for c in ctrls]
es = pd.DataFrame(rows, columns=["origin", "f", "j", "treated", "served", "hist", "exact_match"])
es["last_event"] = es.j.map(LAST_EVENT); es["surv1y"] = (es.last_event >= es.f + D1Y).astype(int); es["surv3y"] = (es.last_event >= es.f + 3 * D1Y).astype(int)
log(f"C: event-study units {len(es):,} ({es.treated.sum():,} treated); last feed event before f+1y: treated {1 - es.loc[es.treated == 1, 'surv1y'].mean():.3f}, controls {1 - es.loc[es.treated == 0, 'surv1y'].mean():.3f}")
def y_in(dst, lo, hi, kind):
    if kind == "file": return int(any(lo <= f_ < hi for f_ in RF.get(dst, ())))
    if kind == "onset": return int(any(lo <= bb < hi for (bb, ee, ff, sv) in RP.get(dst, ())))
    if kind == "legacy": return int(any(lo <= ff < hi and bb < lo for (bb, ee, ff, sv) in RP.get(dst, ())))   # filing in the window of a period begun before the window
long = []
for k in range(-3, 3):
    lo = es.f + pd.to_timedelta(k * 365, unit="D"); hi = es.f + pd.to_timedelta((k + 1) * 365, unit="D")
    for kind in ("file", "onset", "legacy"):
        pass
    long.append(pd.DataFrame({"origin": es.origin, "f": es.f, "j": es.j, "treated": es.treated, "served": es.served, "surv1y": es.surv1y, "k": k,
                              "y_file": [y_in(d_, a_, b_, "file") for d_, a_, b_ in zip(es.j, lo, hi)], "y_onset": [y_in(d_, a_, b_, "onset") for d_, a_, b_ in zip(es.j, lo, hi)],
                              "y_legacy": [y_in(d_, a_, b_, "legacy") for d_, a_, b_ in zip(es.j, lo, hi)]}))
L = pd.concat(long, ignore_index=True); L["post"] = (L.k >= 0).astype(int); L["ev_id"] = L.origin + "#" + L.f.astype(str); L["unit"] = L.ev_id + "#" + L.j
L["window"] = L.k.isin([-1, 0]).astype(int)          # the two years around the origin's filing (the year before it and the year of it)
L["late"] = L.k.isin([1, 2]).astype(int)              # for the after-versus-before contrast that excludes the window: k in {1,2} vs k in {-3,-2}
es["wave_event"] = [int(is_wave(o_, f_)) for o_, f_ in zip(es.origin, es.f)]; es["bc_any"] = [int((o_ in bc_ciks) or (j_ in bc_ciks)) for o_, j_ in zip(es.origin, es.j)]
es["period"] = pd.cut(es.f.dt.year, [2007, 2013, 2019, 2022], labels=["2008-2013", "2014-2019", "2020-2022"]).astype(str)
for col in ("wave_event", "bc_any", "period"): L[col] = L[["origin", "f", "j"]].merge(es[["origin", "f", "j", col]].drop_duplicates(["origin", "f", "j"]), on=["origin", "f", "j"], how="left")[col].values
def es_did(df, ycol, treat_col="treated", post_col="post"):
    # balanced two-way within transformation: unit (j x event) and event x k
    y = df[ycol].values.astype(float); d = (df[treat_col] * df[post_col]).values.astype(float)
    def dm2(v):
        u = pd.factorize(df["unit"])[0]; ek = pd.factorize(df["ev_id"] + "#" + df["k"].astype(str))[0]; e_ = pd.factorize(df["ev_id"])[0]
        mu_u = np.bincount(u, v) / np.bincount(u); mu_ek = np.bincount(ek, v) / np.bincount(ek); mu_e = np.bincount(e_, v) / np.bincount(e_)
        return v - mu_u[u] - mu_ek[ek] + mu_e[e_]
    yt, dt = dm2(y), dm2(d)
    X = np.column_stack([np.ones(len(df)) * 1e-12, dt]); b, se = twoway_ols(yt, X, df["j"].values, df["ev_id"].values)
    return {"did_pp": round(100 * b[1], 3), "se_pp": round(100 * se[1], 3), "t": round(float(b[1] / se[1]), 2) if se[1] > 0 else None}
def es_eventtime(df, ycol):
    out = {}
    for k in range(-3, 3):
        if k == -1: continue
        sub = df[df.k.isin([-1, k])].copy(); sub["post"] = (sub.k == k).astype(int)
        out[str(k)] = es_did(sub, ycol)
    return out
C_ = {"n_fresh_disclosures": int(len(ev)), "n_events": int(L.ev_id.nunique()), "n_treated_units": int(es.treated.sum()), "n_control_units": int((es.treated == 0).sum()), "n_destinations_treated": int(es.loc[es.treated == 1, "j"].nunique()),
      "n_linked_pairs_before_presence_filter": n_treated_before_survivorship, "n_linked_pairs_dropped_no_feed_event_at_or_after_freeze": n_dropped_treated,
      "presence_note": "linked destinations and controls both require a feed event at or before the freeze (f-12m) and one at or after it; no condition on events after the filing (before 2026-10-12 controls alone were required to record an event at or after f+12m)",
      "matching": {"cells": "restatement filed within 3 years before the freeze x filed earlier than that (4 cells); 3 controls per linked destination", "share_treated_exact_cell_match": round(float(es.loc[es.treated == 1, "exact_match"].mean()), 4),
                   "treated_hist_cell_shares": {str(k): round(float(v), 4) for k, v in es.loc[es.treated == 1, "hist"].value_counts(normalize=True).sort_index().items()}},
      "attrition": {"share_treated_last_event_before_f_plus_1y": round(float(1 - es.loc[es.treated == 1, "surv1y"].mean()), 4), "share_control_last_event_before_f_plus_1y": round(float(1 - es.loc[es.treated == 0, "surv1y"].mean()), 4),
                    "share_treated_last_event_before_f_plus_3y": round(float(1 - es.loc[es.treated == 1, "surv3y"].mean()), 4), "share_control_last_event_before_f_plus_3y": round(float(1 - es.loc[es.treated == 0, "surv3y"].mean()), 4)},
      "share_treated_served": round(float(es.loc[es.treated == 1, "served"].mean()), 4)}
for ycol in ("y_file", "y_onset", "y_legacy"):
    C_[ycol] = {"did_post_vs_pre": es_did(L, ycol), "event_time_vs_k_minus1": es_eventtime(L, ycol),
                "pre_mean_treated": round(float(L.loc[(L.treated == 1) & (L.post == 0), ycol].mean()), 4), "pre_mean_control": round(float(L.loc[(L.treated == 0) & (L.post == 0), ycol].mean()), 4)}
    # within-origin contrast: served vs not served among treated (controls matched to each treated carry its served flag)
    C_[ycol]["served_vs_not_served_did"] = {"served": es_did(L[L.served == 1], ycol), "not_served": es_did(L[L.served == 0], ycol)}
    Lx = L[~L.k.isin([-1, 0])]
    C_[ycol]["window_minus1_0_vs_other_years"] = es_did(L, ycol, post_col="window")
    C_[ycol]["after_1_2_vs_before_minus3_minus2"] = es_did(Lx, ycol, post_col="late")
    C_[ycol]["window_robustness"] = {"survivors_1y": es_did(L[L.surv1y == 1], ycol, post_col="window"), "excluding_wave_and_blank_check": es_did(L[(L.wave_event == 0) & (L.bc_any == 0)], ycol, post_col="window"),
                                     "by_event_period": {per: es_did(L[L.period == per], ycol, post_col="window") for per in ("2008-2013", "2014-2019", "2020-2022")}}
    C_[ycol]["after_vs_before_robustness"] = {"survivors_1y": es_did(Lx[Lx.surv1y == 1], ycol, post_col="late"), "excluding_wave_and_blank_check": es_did(Lx[(Lx.wave_event == 0) & (Lx.bc_any == 0)], ycol, post_col="late"),
                                              "by_event_period": {per: es_did(Lx[Lx.period == per], ycol, post_col="late") for per in ("2008-2013", "2014-2019", "2020-2022")}}
    C_[ycol]["survivors_1y_did"] = {**es_did(L[L.surv1y == 1], ycol), "n_treated_units": int(es.loc[(es.treated == 1) & (es.surv1y == 1)].shape[0]), "n_control_units": int(es.loc[(es.treated == 0) & (es.surv1y == 1)].shape[0])}
    C_[ycol]["excluding_wave_events_did"] = {**es_did(L[L.wave_event == 0], ycol), "n_events": int(L.loc[L.wave_event == 0, "ev_id"].nunique()), "n_treated_units": int(es.loc[(es.treated == 1) & (es.wave_event == 0)].shape[0])}
    C_[ycol]["excluding_blank_check_did"] = {**es_did(L[(L.wave_event == 0) & (L.bc_any == 0)], ycol), "n_events": int(L.loc[(L.wave_event == 0) & (L.bc_any == 0), "ev_id"].nunique()), "n_treated_units": int(es.loc[(es.treated == 1) & (es.wave_event == 0) & (es.bc_any == 0)].shape[0])}
    C_[ycol]["by_event_period_did"] = {per: es_did(L[L.period == per], ycol) for per in ("2008-2013", "2014-2019", "2020-2022")}
    C_[ycol]["same_year_cofiling"] = {"treated_share_filing_in_k0": round(float(L.loc[(L.treated == 1) & (L.k == 0), "y_file"].mean()), 4), "control_share_filing_in_k0": round(float(L.loc[(L.treated == 0) & (L.k == 0), "y_file"].mean()), 4),
                                      "treated_share_filing_in_km1": round(float(L.loc[(L.treated == 1) & (L.k == -1), "y_file"].mean()), 4), "control_share_filing_in_km1": round(float(L.loc[(L.treated == 0) & (L.k == -1), "y_file"].mean()), 4)} if ycol == "y_file" else None
OUT["C_frozen_network_event_study"] = C_
log(f"C: DiD file {C_['y_file']['did_post_vs_pre']}, onset {C_['y_onset']['did_post_vs_pre']}, legacy {C_['y_legacy']['did_post_vs_pre']}")

# ------------------------------------------------------------------ D. bounds, sensitivity, placebo timing, permutation
D_ = {}
# D1 stacked event-time regression for recent exposure (onset) to get the coefficient vector and covariance
def yr_ind_onset(dst, lo, hi): return int(any(lo <= bb < hi for (bb, ee, ff, sv) in RP.get(dst, ())))
ks = list(range(-3, 3)); rowsE = []
for k in ks:
    lo = eq.birth + pd.to_timedelta(k * 365, unit="D"); hi = eq.birth + pd.to_timedelta((k + 1) * 365, unit="D")
    rowsE.append(pd.DataFrame({"src": eq.src, "dst": eq.dst, "by": eq.by.astype(int), "k": k, "x": eq.exposed5y, "y": [yr_ind_onset(d_, a_, b_) for d_, a_, b_ in zip(eq.dst, lo, hi)]}))
E = pd.concat(rowsE, ignore_index=True)
Kd = pd.get_dummies(E["k"].astype(int), prefix="k").astype(float)
def event_time_gaps(adjusted):
    # adjusted: formation-year x event-time cell means absorbed, so each gap compares exposed and clean pairs of the same cohort at the same event time
    if adjusted:
        cells = pd.get_dummies((E["by"].astype(int).astype(str) + "#" + E["k"].astype(int).astype(str))).astype(float).values
        X_ = np.column_stack([cells, (Kd.values * E["x"].values[:, None])])
    else:
        X_ = np.column_stack([Kd.values, (Kd.values * E["x"].values[:, None])])
    b_, se_ = twoway_ols(E["y"].values, X_, E["src"].values, E["dst"].values); off = X_.shape[1] - len(ks)
    return ({str(k): round(100 * b_[off + i], 3) for i, k in enumerate(ks)}, {str(k): round(100 * se_[off + i], 3) for i, k in enumerate(ks)},
            np.array([b_[off + i] for i, k in enumerate(ks) if k < 0]), np.array([b_[off + i] for i, k in enumerate(ks) if k >= 0]))
beta_u, sebeta_u, pre_u, post_u = event_time_gaps(False)
beta, sebeta, pre, post = event_time_gaps(True)
delta_hat = 100 * (post.mean() - pre.mean())
# the paper's headline for recent exposure is the cohort-adjusted change (formation-year x post effects); the unadjusted change is kept for reference
d_unadj = did(eq, "pre_onset_3y", "out_onset_3y", "exposed5y")
d_main = did_yfe(eq, "pre_onset_3y", "out_onset_3y", "exposed5y"); se_delta = d_main["se_pp"]; delta_main = d_main["diff_in_gaps_pp"]
max_pre_dev = float(100 * np.max(np.abs(np.diff(pre))))   # largest year-to-year movement of the cohort-adjusted pre-period gap
grid = {}
for M in (0.0, 0.5, 1.0, 2.0):
    for bj in (0.0, 0.5, 1.0, 2.0):
        half = M * max_pre_dev + bj
        grid[f"M={M},jump={bj}"] = {"identified_set_pp": [round(delta_main - half, 2), round(delta_main + half, 2)], "ci95_pp": [round(delta_main - half - 1.96 * se_delta, 2), round(delta_main + half + 1.96 * se_delta, 2)]}
D_["drift_bounds_recent_onset"] = {"event_time_gaps_pp": beta, "event_time_se_pp": sebeta, "event_time_gaps_unadjusted_pp": beta_u, "event_time_se_unadjusted_pp": sebeta_u,
                                   "mean_post_minus_mean_pre_pp": round(delta_hat, 3), "mean_post_minus_mean_pre_unadjusted_pp": round(100 * (post_u.mean() - pre_u.mean()), 3),
                                   "headline_change_pp": delta_main, "headline_se_pp": se_delta, "headline_adjustment": "formation-year and formation-year x post effects (did_yfe)",
                                   "unadjusted_headline_change_pp": d_unadj["diff_in_gaps_pp"], "unadjusted_headline_se_pp": d_unadj["se_pp"],
                                   "max_pre_period_year_to_year_change_pp": round(max_pre_dev, 3), "max_pre_period_year_to_year_change_unadjusted_pp": round(float(100 * np.max(np.abs(np.diff(pre_u)))), 3),
                                   "drift_to_hide_zero_effect_pp": round(delta_main, 3), "drift_to_hide_one_point_effect_pp": round(delta_main - 1.0, 3), "grid": grid,
                                   "note": "identified set = cohort-adjusted headline change +/- (M x largest cohort-adjusted pre-period year-to-year change + appointment-date jump b); a differential drift equal to the estimate would be needed for the true effect to be zero, and a drift of (estimate - 1) for a one-point effect"}
# D2 Cinelli-Hazlett robustness values for the panel recent-exposure coefficients (rows 13 and 16) with own-history benchmark
def rv(t, df_, q=1.0, alpha=None):
    f = abs(t) / np.sqrt(df_)
    if alpha is not None:
        from scipy.stats import t as tdist
        f = max(f - tdist.ppf(1 - alpha / 2, df_) / np.sqrt(df_), 0)
    fq = q * f
    return 0.5 * (np.sqrt(fq ** 4 + 4 * fq ** 2) - fq ** 2)
D_["cinelli_hazlett"] = {}
for name, dep, key, controls in (("row13_recent_onset_plus_prior", "restate_onset", "nexp5_lag1", ["nnew_lag1"] + ctrl + ["had_prior_restate"]),
                                 ("row16_inbound_recent_onset_plus_prior", "restate_onset", "nexp5_in_lag1", ["nnew_in_lag1", "nnew_out_lag1", "nexp5_out_lag1"] + ctrl + ["had_prior_restate"])):
    r = fe(dep, [key] + controls); t_ = float(r.tstats[key]); dof = float(r.df_resid)
    r2_yz = float(r.tstats["had_prior_restate"]) ** 2 / (float(r.tstats["had_prior_restate"]) ** 2 + dof)     # partial R2 of own history for Y given D, X
    rd = fe(key, controls); r2_dz = float(rd.tstats["had_prior_restate"]) ** 2 / (float(rd.tstats["had_prior_restate"]) ** 2 + float(rd.df_resid))   # partial R2 of own history for D given X
    sd_ratio = float(r.std_errors[key]) * np.sqrt(dof)
    bias_1x = sd_ratio * np.sqrt(r2_yz * r2_dz / (1 - r2_dz))
    D_["cinelli_hazlett"][name] = {"coef_pp": round(100 * float(r.params[key]), 3), "t": round(t_, 2), "df": int(dof), "RV_q1": round(rv(t_, dof), 4), "RV_q1_alpha05": round(rv(t_, dof, alpha=0.05), 4),
                                   "partial_R2_Y_ownhistory": round(r2_yz, 4), "partial_R2_D_ownhistory": round(r2_dz, 5), "bias_from_confounder_as_strong_as_own_history_pp": round(100 * bias_1x, 3),
                                   "bias_from_confounder_3x_own_history_pp": round(100 * sd_ratio * np.sqrt(min(9 * r2_yz, 0.999) * min(9 * r2_dz, 0.999) / (1 - min(9 * r2_dz, 0.999))), 3),
                                   "partial_R2_of_exposure": round(t_ ** 2 / (t_ ** 2 + dof), 5)}
log(f"D2 CH: {D_['cinelli_hazlett']['row13_recent_onset_plus_prior']}")
# D3 monotone-selection bound for recent exposure (onset): ATT <= post-level gap; under stationary selection ATT <= change
pg_u = gap(eq, "out_onset_3y", "exposed5y"); pg = gap_yfe(eq, "out_onset_3y", "exposed5y")
D_["monotone_selection_bound_recent_onset"] = {"post_level_gap_pp": pg["diff_pp"], "post_level_se_pp": pg["se_pp"], "post_level_upper95_pp": round(pg["diff_pp"] + 1.645 * pg["se_pp"], 2), "change_pp": delta_main, "change_se_pp": se_delta, "change_upper95_one_sided_pp": round(delta_main + 1.645 * se_delta, 2),
                                               "change_upper975_pp": round(delta_main + 1.96 * se_delta, 2), "adjustment": "formation-year effects (level); formation-year x post effects (change)",
                                               "unadjusted": {"post_level_gap_pp": pg_u["diff_pp"], "post_level_upper95_pp": round(pg_u["diff_pp"] + 1.645 * pg_u["se_pp"], 2), "change_pp": d_unadj["diff_in_gaps_pp"], "change_upper95_one_sided_pp": round(d_unadj["diff_in_gaps_pp"] + 1.645 * d_unadj["se_pp"], 2)},
                                               "assumption": "destinations that appoint from recently restating origins have weakly higher latent risk (positive selection on the label); stationary selection across the two windows for the change version"}
# D4 placebo formation dates: shift by -3y (pre-pre vs pre) and +3y (post vs post-post, cohorts formed by 2019)
eq["out_onset_3to6y"] = [int(any(b_ + D3 <= bb < b_ + 2 * D3 for (bb, ee, ff, sv) in RP.get(d_, ()))) for d_, b_ in zip(eq.dst, eq.birth)]
D_["placebo_formation_dates"] = {"minus_3y": {"pre_window": "[-6y,-3y)", "post_window": "[-3y,0)", "exposed5y": did(eq, "pre_onset_3to6y", "pre_onset_3y", "exposed5y"), "exposed": did(eq, "pre_onset_3to6y", "pre_onset_3y", "exposed"),
                                              "exposed5y_yfe": did_yfe(eq, "pre_onset_3to6y", "pre_onset_3y", "exposed5y"), "exposed_yfe": did_yfe(eq, "pre_onset_3to6y", "pre_onset_3y", "exposed")},
                                 "plus_3y": {"pre_window": "[0,3y)", "post_window": "[3y,6y)", "n_rows": int((eq.by <= 2019).sum()), "exposed5y": did(eq[eq.by <= 2019], "out_onset_3y", "out_onset_3to6y", "exposed5y"), "exposed": did(eq[eq.by <= 2019], "out_onset_3y", "out_onset_3to6y", "exposed"),
                                             "exposed5y_yfe": did_yfe(eq[eq.by <= 2019], "out_onset_3y", "out_onset_3to6y", "exposed5y"), "exposed_yfe": did_yfe(eq[eq.by <= 2019], "out_onset_3y", "out_onset_3to6y", "exposed")},
                                 "note": "_yfe versions include formation-year and formation-year x post effects, the adjustment used for the headline recent-exposure change"}
# D5 permutation inference within formation-year x destination-size-decile (feed-age decile where Compustat size is missing)
cw = pd.read_csv("data/cik_gvkey_crosswalk.csv", dtype=str); cw["cik"] = norm_cik(cw["company_fkey"]); cw["gvkey"] = cw["gvkey"].astype(str).str.zfill(6); cw = cw.drop_duplicates("cik")[["cik", "gvkey"]]
cp = pd.read_csv("data/compustat_financials.csv"); cp["gvkey"] = cp["gvkey"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(6); cp = cp[cp["assets"] > 0]
sz = cp.merge(cw, on="gvkey").assign(size=lambda d_: np.log(d_["assets"]))[["cik", "year", "size"]]
eqp = eq.merge(sz.rename(columns={"cik": "dst", "year": "by", "size": "dst_size"}), on=["dst", "by"], how="left")
eqp["cell_var"] = eqp["dst_size"].fillna(-100 + eqp["dst_feed_age"])   # feed age as a fallback ordering for non-Compustat destinations
eqp["cell"] = eqp["by"].astype(str) + "#" + eqp.groupby("by")["cell_var"].transform(lambda s: pd.qcut(s.rank(method="first"), 10, labels=False, duplicates="drop")).astype(str)
d_i = (eqp["out_onset_3y"] - eqp["pre_onset_3y"]).values.astype(float); x = eqp["exposed5y"].values.astype(float)
cells = pd.factorize(eqp["cell"])[0]; ncell = np.bincount(cells).astype(float)
def within_coef(xv):
    xm = xv - (np.bincount(cells, xv) / ncell)[cells]; dm = d_i - (np.bincount(cells, d_i) / ncell)[cells]
    return float((xm * dm).sum() / (xm * xm).sum())
stat = within_coef(x); perm_stats = np.empty(A.n_perm)
rng2 = np.random.default_rng(11); order = np.argsort(cells, kind="stable"); bounds = np.flatnonzero(np.diff(cells[order])) + 1; groups = np.split(order, bounds)
for i in range(A.n_perm):
    xp = x.copy()
    for gidx in groups: xp[gidx] = x[rng2.permutation(gidx)]
    perm_stats[i] = within_coef(xp)
D_["permutation_recent_onset_change"] = {"observed_pp": round(100 * stat, 3), "n_perm": A.n_perm, "p_two_sided": round(float((np.abs(perm_stats) >= abs(stat)).mean()), 4), "p_one_sided_positive": round(float((perm_stats >= stat).mean()), 4),
                                         "perm_sd_pp": round(100 * float(perm_stats.std()), 3), "n_cells": int(len(groups)), "note": "exposure labels permuted within formation-year x destination-size-decile cells (feed-age decile where size is missing); statistic = coefficient of recent exposure in a regression of the pair's post-minus-pre onset indicator with cell fixed effects"}
log(f"D5 permutation: observed {100*stat:+.2f}, p(two-sided) {D_['permutation_recent_onset_change']['p_two_sided']}, perm sd {100*perm_stats.std():.2f}")
OUT["D_bounds"] = D_

os.makedirs("results_rebuild", exist_ok=True)
json.dump(OUT, open(f"results_rebuild/L_identification{SUFFIX}.json", "w"), indent=1, default=str)
log(f"saved results_rebuild/L_identification{SUFFIX}.json")
