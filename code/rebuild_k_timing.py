#!/usr/bin/env python3
"""
rebuild_k_timing.py — JAPP re-targeting rebuild (module K), 2026-10-09
=======================================================================
Addresses the convergent must-fix items of the two JAPP-referee simulations
(2nd_revision/reviews/CONTENT_CROSS_REVIEW_20261009_163622_high.md):

 K0  Edge lifecycles rebuilt from BOARD-MEMBER events only (is_bdmem_pers == 1); a spell opens on
     'Appointed'/'Returned to Position' and closes on ANY departure action (Resigned, Retired,
     Employment Ceased, Dismissed, Declined Re-election, Not Re-elected, Deceased, Appointment
     Revoked). The old pipeline used all D&O events (officers included) and closed spells only on
     Resigned/Retired, which left dismissed/deceased/not-re-elected directors "active" to the horizon.
 K1  Directional pair base with exact-date exposure (source filed before edge birth; also a
     5-year recency-bounded version) and two outcomes: a destination restatement FILED in
     [birth, death+1y] (as before) and a destination restatement whose MISSTATED PERIOD BEGINS
     in [birth, death+1y] ("onset", Audit Analytics res_begin_date) — practice imported after
     arrival cannot be reflected in a filing that corrects pre-arrival periods.
     Pre-formation placebo for sorting: destination restatement in [birth-2y, birth).
     Two-way (source + destination firm) cluster-robust inference throughout.
 K2  Firm-year panel with exposure that PRECEDES the outcome year: exposed edges formed in t-1
     (flow) and exposed edges alive on 1 January of t (stock), control for total new/alive edges,
     outcomes filed-in-t and onset-in-t; PanelOLS two-way FE, firm-clustered; Oster inputs
     (uncontrolled model on the same sample) with beta*, delta0 per Rmax.
 K3  Cohort attenuation with a FIXED 3-year post-formation window and full follow-up.
 K4  H5 (within vs cross community) with correct exposed-tie counts, as an interaction with
     two-way clustering; Louvain communities per year on the rebuilt network.
 K5  Survival with competing risks: which seat ended the edge (origin / destination / horizon);
     cause-specific Cox, birth-year strata, origin-firm clusters, PH test; recency-bounded exposure.
 K6  Left-censoring robustness: edges formed 2008+ only.
 K7  Firm-count crosswalk (D&O universe, network, pair base, panel, estimation sample).

Run from github/:  python code/rebuild_k_timing.py
Outputs: data/edge_lifecycles_k.csv, data/integrated_panel_k.csv, data/panel_network_k.csv,
         results_rebuild/K_rebuild.json
"""
import json, os, sys, time, warnings
from collections import defaultdict

import numpy as np
import pandas as pd
import networkx as nx
import statsmodels.api as sm
from scipy import stats
from scipy.stats import norm
from linearmodels.panel import PanelOLS
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test, proportional_hazard_test
from networkx.algorithms.community import louvain_communities, modularity

warnings.filterwarnings("ignore")
t0 = time.time(); rng = np.random.default_rng(42)
def log(m): print(f"[{time.time()-t0:7.0f}s] {m}"); sys.stdout.flush()

import argparse
_ap = argparse.ArgumentParser(); _ap.add_argument("--max-firms", type=int, default=0, help="drop directors interlocked with more than this many firms (0 = no cap)")
_ap.add_argument("--suffix", default="")
_ap.add_argument("--directors-file", default="data/directors_officers_full.csv")
_ap.add_argument("--aux-file", default="", help="WRDS pull with middle_name, suffix, is_effective_date_estimate, comm_report; merged onto the extract by (firm, name, date, action)")
_ap.add_argument("--person-key", default="name", choices=["name", "name_mid"], help="name = first|last (paper); name_mid = split a name across firms when recorded middle initials conflict (unassignable blocks form no interlocks)")
_ap.add_argument("--drop-estimated-dates", action="store_true", help="drop tenure spells whose start or end event carries is_effective_date_estimate = 1")
_ap.add_argument("--sample", default="cap", choices=["cap", "artifacts"], help="cap = drop directors with more than --max-firms firms; artifacts = keep every director, drop lifecycles with a sponsor-family/fund/REIT or shell/blank-check registrant and (unless --keep-common-surnames) directors whose surname is shared by more than 150 names")
_ap.add_argument("--keep-common-surnames", action="store_true")
_args = _ap.parse_args()
MAX_FIRMS, SUFFIX = _args.max_firms, _args.suffix
HORIZON = pd.Timestamp("2025-12-31")      # last date with complete D&O and restatement coverage used here
FORM_LO, FORM_HI = 2004, 2023             # edge-formation window analysed
START_ACTIONS = {"Appointed", "Returned to Position"}
END_ACTIONS = {"Resigned", "Retired", "Employment Ceased", "Dismissed", "Declined Re-election",
               "Not Re-elected", "Deceased", "Appointment Revoked/Not Accepted"}
OUT = {}

def norm_cik(s): return pd.Series(s).astype(str).str.replace(r"\.0$", "", regex=True).str.lstrip("0")

# ======================================================================================
# K0. Director-only tenure spells and edge lifecycles
# ======================================================================================
dirs = pd.read_csv(_args.directors_file, usecols=["company_fkey", "first_name", "last_name", "eff_date", "action", "is_bdmem_pers"], dtype={"company_fkey": str})
dirs["eff_date"] = pd.to_datetime(dirs["eff_date"], errors="coerce")
dirs = dirs.dropna(subset=["eff_date"])
dirs["person"] = dirs["first_name"].fillna("").str.strip().str.lower() + "|" + dirs["last_name"].fillna("").str.strip().str.lower()
dirs["cfkey"] = norm_cik(dirs["company_fkey"])
dirs["mid_init"] = ""; dirs["est_date"] = 0; dirs["comm_report"] = None
AUX = {"file": _args.aux_file or None}
if _args.aux_file:
    _aux = pd.read_csv(_args.aux_file, usecols=["company_fkey", "first_name", "last_name", "eff_date", "action", "middle_name", "suffix", "is_effective_date_estimate", "comm_report"], dtype={"company_fkey": str}, low_memory=False)
    _aux["eff_date"] = pd.to_datetime(_aux["eff_date"], errors="coerce")
    _aux["k"] = norm_cik(_aux["company_fkey"]) + "#" + _aux["first_name"].fillna("").str.strip().str.lower() + "|" + _aux["last_name"].fillna("").str.strip().str.lower() + "#" + _aux["eff_date"].astype(str) + "#" + _aux["action"].fillna("")
    _aux = _aux.drop_duplicates("k").set_index("k")
    dirs["k"] = dirs["cfkey"] + "#" + dirs["person"] + "#" + dirs["eff_date"].astype(str) + "#" + dirs["action"].fillna("")
    dirs["mid_init"] = dirs["k"].map(_aux["middle_name"]).fillna("").astype(str).str.strip().str.lower().str[:1].str.replace(r"[^a-z]", "", regex=True)
    dirs["est_date"] = dirs["k"].map(_aux["is_effective_date_estimate"]).fillna(0).astype(int)
    dirs["comm_report"] = dirs["k"].map(_aux["comm_report"])
    AUX.update({"match_rate": round(float(dirs["k"].isin(_aux.index).mean()), 5), "middle_initial_share": round(float((dirs["mid_init"] != "").mean()), 4), "estimated_date_share": round(float(dirs["est_date"].mean()), 4), "comm_report_share": round(float(dirs["comm_report"].notna().mean()), 4)})
    dirs = dirs.drop(columns=["k"])
n_events_all = len(dirs)
bd = dirs[dirs["is_bdmem_pers"] == 1].copy()
bd = bd[bd["action"].isin(START_ACTIONS | END_ACTIONS)]
KEYSTATS = {"mode": _args.person_key}
if _args.person_key == "name_mid":
    # conflict-aware key: within a first|last name, each firm block takes the modal recorded middle initial; if the name carries two or more
    # different initials across its firm blocks, blocks are keyed by initial and blocks without an initial become unassignable (own key per firm).
    blk_init = bd[bd["mid_init"] != ""].groupby(["person", "cfkey"])["mid_init"].agg(lambda s: s.value_counts().index[0])
    bd["blk_init"] = pd.Series(list(zip(bd["person"], bd["cfkey"])), index=bd.index).map(blk_init.to_dict()).fillna("")
    n_init = bd[bd["blk_init"] != ""].groupby("person")["blk_init"].nunique()
    conflict = set(n_init[n_init > 1].index)
    firms_per_name = bd.groupby("person")["cfkey"].nunique()
    KEYSTATS.update({"names_at_more_than_one_firm": int((firms_per_name > 1).sum()), "names_with_conflicting_initials": len(conflict),
                     "share_multi_firm_names_conflicting": round(len(conflict) / max(int((firms_per_name > 1).sum()), 1), 4)})
    isc = bd["person"].isin(conflict)
    newkey = bd["person"].where(~isc, bd["person"] + "|" + bd["blk_init"].where(bd["blk_init"] != "", "?" + bd["cfkey"]))
    KEYSTATS.update({"board_rows_rekeyed": int(isc.sum()), "board_rows_unassignable": int((isc & (bd["blk_init"] == "")).sum())})
    bd["person"] = newkey
if _args.drop_estimated_dates: KEYSTATS["drop_estimated_dates"] = True
bd["is_start"] = bd["action"].isin(START_ACTIONS)
bd = bd.sort_values(["person", "cfkey", "eff_date", "is_start"])   # on the same day, an end sorts before a start
log(f"D&O events {n_events_all:,}; board-member start/end events {len(bd):,}")

spells = []           # (person, company, start, end, end_action, censored, est, start_comm)
N_DEPARTURE_WITHOUT_APPOINTMENT = 0; N_REPEAT_START_IGNORED = 0; N_DEPARTURE_SAME_DAY_IGNORED = 0; N_DEPARTURE_AFTER_HORIZON = 0; N_START_AFTER_HORIZON = 0
for (p, c), g in bd.groupby(["person", "cfkey"], sort=False):
    start = None; est_s = 0; comm_s = None
    for a, d, e_, cm_ in zip(g["action"].values, g["eff_date"].values, g["est_date"].values, g["comm_report"].values):
        d = pd.Timestamp(d)
        if a in START_ACTIONS:
            if start is None: start = d; est_s = int(e_); comm_s = cm_ if isinstance(cm_, str) else None
            else: N_REPEAT_START_IGNORED += 1
        elif start is not None and d > start:
            if d > HORIZON: N_DEPARTURE_AFTER_HORIZON += 1
            spells.append((p, c, start, min(d, HORIZON), a, d > HORIZON, int(est_s or (e_ and d <= HORIZON)), comm_s)); start = None
        elif start is None: N_DEPARTURE_WITHOUT_APPOINTMENT += 1
        else: N_DEPARTURE_SAME_DAY_IGNORED += 1
    if start is not None and start <= HORIZON:
        spells.append((p, c, start, HORIZON, None, True, int(est_s), comm_s))
    elif start is not None: N_START_AFTER_HORIZON += 1
ten = pd.DataFrame(spells, columns=["person", "company", "start", "end", "end_action", "censored", "est", "start_comm"])
N_ZERO_DURATION_SPELLS = int((ten["end"] <= ten["start"]).sum())
ten = ten[ten["end"] > ten["start"]]
KEYSTATS["spells_with_estimated_endpoint"] = int(ten["est"].sum())
if _args.drop_estimated_dates: ten = ten[ten["est"] == 0]
def _ac(cm):  # audit-committee mention in the appointment event's committee text: 1 yes, 0 committees named without audit, -1 none recorded
    if not isinstance(cm, str) or not cm.strip(): return -1
    return int("audit" in cm.lower())
ten["start_ac"] = ten["start_comm"].map(_ac)
SPELLS = defaultdict(list)
for p_, c_, s_, e_ in zip(ten["person"], ten["company"], ten["start"], ten["end"]): SPELLS[(p_, c_)].append((s_, e_))
def spell_start_at(person, company, t):
    for s_, e_ in SPELLS.get((person, company), ()):
        if s_ <= t <= e_: return s_
    return None
LAST_EVENT = bd.groupby("cfkey")["eff_date"].max().to_dict()   # last board event per firm (firm-exit proxy)
FIRST_EVENT_ALL = dirs.groupby("cfkey")["eff_date"].min().to_dict()   # first D&O event of any kind per firm (feed entry)
log(f"director tenure spells {len(ten):,} (censored at horizon {ten['censored'].mean():.1%}); persons {ten['person'].nunique():,}; firms {ten['company'].nunique():,}")

edges = []
for p, g in ten.groupby("person", sort=False):
    if len(g) < 2: continue
    rows = list(zip(g["company"], g["start"], g["end"], g["censored"], g["start_ac"]))
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            c1, s1, e1, z1, a1 = rows[i]; c2, s2, e2, z2, a2 = rows[j]
            if c1 == c2: continue
            birth, death = max(s1, s2), min(e1, e2)
            if birth < death:
                origin = c1 if s1 < s2 else (c2 if s2 < s1 else None)
                ended_by = ("both" if e1 == e2 else ("c1" if e1 < e2 else "c2"))
                ongoing = death >= HORIZON
                term = None if ongoing else (("origin" if (ended_by == "c1") == (origin == c1) else "destination") if origin and ended_by != "both" else "both")
                dst_ac = (a2 if origin == c1 else a1) if origin else -1   # committee text of the appointment that created the interlock (destination seat)
                edges.append((min(c1, c2), max(c1, c2), p, birth, death, origin, ongoing, term, dst_ac))
edf = pd.DataFrame(edges, columns=["company_a", "company_b", "director", "edge_birth", "edge_death", "origin", "ongoing", "terminated_by", "dst_ac"])
edf["birth_year"] = edf["edge_birth"].dt.year
_seats = pd.concat([edf[["director", "company_a"]].rename(columns={"company_a": "c"}), edf[["director", "company_b"]].rename(columns={"company_b": "c"})]).drop_duplicates()
_k = _seats.groupby("director")["c"].nunique(); edf["director_firms"] = edf["director"].map(_k)
OUT["director_firm_counts"] = {"directors_with_interlocks": int(len(_k)), "median_firms": float(_k.median()), "p90_firms": float(_k.quantile(0.9)), "max_firms": int(_k.max()),
                               "share_directors_gt6": round(float((_k > 6).mean()), 4), "share_directors_gt10": round(float((_k > 10).mean()), 4),
                               "share_edges_from_directors_gt6": round(float((edf.director_firms > 6).mean()), 4), "share_edges_from_directors_gt10": round(float((edf.director_firms > 10).mean()), 4),
                               "share_edges_from_directors_gt20": round(float((edf.director_firms > 20).mean()), 4), "cap_applied": MAX_FIRMS}

# registrant types (SEC EDGAR submissions): blank-check = SIC 6770 or "... Acquisition Corp" name; fund-like = SIC 6722/6726, or no SIC assigned and a fund-type name (closed-end funds and BDCs that file D&O changes)
if os.path.exists("data/sec_sic_by_cik.csv"):
    sec0 = pd.read_csv("data/sec_sic_by_cik.csv", dtype=str).fillna(""); sec0["cik"] = norm_cik(sec0["cik"])
    pat0 = r"acquisition\s+(?:corp|co\b|company|ltd|limited|holdings|inc|plc|partners|group)|blank\s*check|\bspac\b"
    nm0 = (sec0["name"] + " | " + sec0["former_names"]).str.lower()
    spac0 = set(sec0.loc[(sec0["sic"].str.strip() == "6770") | nm0.str.contains(pat0, regex=True), "cik"])
    fundpat0 = r"\bfund\b|\bfunds\b|\btrust\b|portfolio|\betf\b|income|investors|\bseries\b|\bvariable\b|municipal|tax[- ]free|\bbond\b|\bbdc\b"
    fund0 = set(sec0.loc[sec0["sic"].str.strip().isin(["6722", "6726"]) | ((sec0["sic"].str.strip() == "") & sec0["name"].str.lower().str.contains(fundpat0, regex=True)), "cik"])
    ea = edf.assign(gt6=edf["director_firms"] > 6, any_spac=edf.company_a.isin(spac0) | edf.company_b.isin(spac0), any_fund=edf.company_a.isin(fund0) | edf.company_b.isin(fund0))
    OUT["director_firm_counts"]["registrant_types"] = {"n_blank_check_ciks": len(spac0), "n_fund_like_ciks": len(fund0),
        "share_edges_with_blank_check_firm_directors_gt6": round(float(ea.loc[ea.gt6, "any_spac"].mean()), 4), "share_edges_with_blank_check_firm_directors_le6": round(float(ea.loc[~ea.gt6, "any_spac"].mean()), 4),
        "share_edges_with_fund_like_firm_directors_gt6": round(float(ea.loc[ea.gt6, "any_fund"].mean()), 4), "share_edges_with_fund_like_firm_directors_le6": round(float(ea.loc[~ea.gt6, "any_fund"].mean()), 4),
        "share_edges_with_either_directors_gt6": round(float((ea.loc[ea.gt6, "any_spac"] | ea.loc[ea.gt6, "any_fund"]).mean()), 4), "share_edges_with_either_directors_le6": round(float((ea.loc[~ea.gt6, "any_spac"] | ea.loc[~ea.gt6, "any_fund"]).mean()), 4),
        "n_edges_gt6": int(ea.gt6.sum()), "n_edges_le6": int((~ea.gt6).sum())}
    # seat-level tabulation: what kinds of registrants do high-seat directors sit on?
    st_ = ten[["person", "company"]].drop_duplicates().copy(); st_["nf"] = st_["person"].map(st_.groupby("person")["company"].nunique())
    st_["sic"] = st_["company"].map(sec0.set_index("cik")["sic"]).fillna("").str.strip(); st_["nm"] = st_["company"].map(sec0.set_index("cik")["name"]).fillna("").str.lower()
    def seat_group(sic_, nm_):
        if sic_ in ("6770", "9995") or "acquisition" in nm_ or "blank check" in nm_: return "shell_or_blank_check"
        if sic_[:2] == "65" or sic_ in ("6798", "6221", "6726", "6722") or (sic_ == "" and any(w in nm_ for w in ("fund", "trust", "partners", "properties", "investors"))): return "fund_partnership_reit_family"
        if sic_ == "": return "no_sic"
        return "operating_company"
    st_["grp"] = [seat_group(a_, b_) for a_, b_ in zip(st_["sic"], st_["nm"])]
    surn_ = pd.Series([x.split("|")[1] for x in st_["person"].unique()]).value_counts()
    st_["sur_n"] = st_["person"].str.split("|").str[1].map(surn_)
    rt = OUT["director_firm_counts"]["registrant_types"]
    for lab_, m_ in (("directors_gt6", st_["nf"] > 6), ("directors_2to6", (st_["nf"] >= 2) & (st_["nf"] <= 6))):
        vc = st_.loc[m_, "grp"].value_counts(normalize=True)
        rt[f"seat_shares_{lab_}"] = {g_: round(float(vc.get(g_, 0.0)), 4) for g_ in ("operating_company", "fund_partnership_reit_family", "shell_or_blank_check", "no_sic")}
        rt[f"n_seats_{lab_}"] = int(m_.sum())
        pers_ = st_.loc[m_].drop_duplicates("person"); rt[f"share_directors_common_surname_gt150_{lab_}"] = round(float((pers_["sur_n"] > 150).mean()), 4)
    rt["note"] = "seat shares are over distinct director-firm seats; common surname = shared by more than 150 distinct director names in the feed"
    log(f"registrant types: gt6 edges with blank-check {OUT['director_firm_counts']['registrant_types']['share_edges_with_blank_check_firm_directors_gt6']:.1%}, fund-like {OUT['director_firm_counts']['registrant_types']['share_edges_with_fund_like_firm_directors_gt6']:.1%}; le6 {OUT['director_firm_counts']['registrant_types']['share_edges_with_either_directors_le6']:.1%}")
edf.to_csv("data/edge_lifecycles_k.csv", index=False)
# artifact sets (same definitions as the artifact-flag block below): sponsor-family / fund / REIT / real-estate-partnership registrants, shell and blank-check registrants, common surnames
ART_FAM = set(); ART_SHELL = set()
if os.path.exists("data/sec_sic_by_cik.csv"):
    _s = pd.read_csv("data/sec_sic_by_cik.csv", dtype=str).fillna(""); _s["cik"] = norm_cik(_s["cik"]); _sic = _s["sic"].str.strip(); _nm = (_s["name"] + " | " + _s["former_names"]).str.lower()
    _fundpat = r"\bfund\b|\bfunds\b|\btrust\b|portfolio|\betf\b|income|investors|\bseries\b|\bvariable\b|municipal|tax[- ]free|\bbond\b|\bbdc\b"
    _spacpat = r"acquisition\s+(?:corp|co\b|company|ltd|limited|holdings|inc|plc|partners|group)|blank\s*check|\bspac\b"
    ART_FAM = set(_s.loc[_sic.str[:2].eq("65") | _sic.isin(["6798", "6221", "6722", "6726"]) | ((_sic == "") & _s["name"].str.lower().str.contains(_fundpat, regex=True)), "cik"])
    ART_SHELL = set(_s.loc[(_sic == "9995") | (_sic == "6770") | _s["name"].str.lower().str.contains("acquisition") | _nm.str.contains(_spacpat, regex=True), "cik"])
_SURN = pd.Series([x.split("|")[1] for x in ten["person"].unique()]).value_counts()
SAMPLE_DEF = {"mode": _args.sample, "max_firms": MAX_FIRMS, "keep_common_surnames": bool(_args.keep_common_surnames), "n_lifecycles_before": int(len(edf))}
if _args.sample == "artifacts":
    _fam_or_shell = edf.company_a.isin(ART_FAM | ART_SHELL) | edf.company_b.isin(ART_FAM | ART_SHELL)
    _common = (edf["director"].str.split("|").str[1].map(_SURN).fillna(0) > 150) if not _args.keep_common_surnames else pd.Series(False, index=edf.index)
    SAMPLE_DEF.update({"n_family_ciks": len(ART_FAM), "n_shell_ciks": len(ART_SHELL), "lifecycles_with_family_or_shell_registrant": int(_fam_or_shell.sum()), "lifecycles_with_common_surname_director": int(_common.sum()),
                       "lifecycles_dropped": int((_fam_or_shell | _common).sum()), "lifecycles_common_surname_only": int((_common & ~_fam_or_shell).sum()), "share_dropped": round(float((_fam_or_shell | _common).mean()), 4),
                       "share_dropped_among_directors_gt6": round(float((_fam_or_shell | _common)[edf.director_firms > 6].mean()), 4), "share_dropped_among_directors_le6": round(float((_fam_or_shell | _common)[edf.director_firms <= 6].mean()), 4)})
    edf = edf[~(_fam_or_shell | _common)].copy()
    log(f"sample=artifacts: dropped {SAMPLE_DEF['lifecycles_dropped']:,} lifecycles ({SAMPLE_DEF['share_dropped']:.1%}) -> {len(edf):,} edges; directors >6 firms kept: {int((edf.director_firms > 6).sum()):,} edges")
elif MAX_FIRMS > 0:
    edf = edf[edf.director_firms <= MAX_FIRMS].copy()
    log(f"cap: keeping directors with <= {MAX_FIRMS} firms -> {len(edf):,} edges")
SAMPLE_DEF["n_lifecycles_after"] = int(len(edf)); OUT["sample_definition"] = SAMPLE_DEF
edf.to_csv(f"data/edge_lifecycles_k{SUFFIX}.csv", index=False)
ten.to_csv(f"data/tenure_spells_k{SUFFIX}.csv", index=False)
inwin = edf[(edf.birth_year >= FORM_LO) & (edf.birth_year <= FORM_HI)]
OUT["K0_edges"] = {"spells": int(len(ten)), "departure_events_without_observed_appointment": int(N_DEPARTURE_WITHOUT_APPOINTMENT),
                   "event_flow": {"all_dno_events": int(n_events_all), "board_member_start_or_end_events": int(len(bd)), "start_events": int(bd["is_start"].sum()), "end_events": int((~bd["is_start"]).sum()),
                                  "repeat_start_events_ignored": int(N_REPEAT_START_IGNORED), "spells_closed_by_a_departure": int((~ten["censored"]).sum()), "spells_open_at_horizon": int(ten["censored"].sum()),
                                  "departures_on_the_appointment_day_ignored": int(N_DEPARTURE_SAME_DAY_IGNORED), "departures_after_horizon_spell_censored": int(N_DEPARTURE_AFTER_HORIZON),
                                  "appointments_after_horizon_no_spell": int(N_START_AFTER_HORIZON), "zero_duration_spells_dropped": int(N_ZERO_DURATION_SPELLS)}, "person_key": KEYSTATS, "aux_merge": AUX, "edge_lifecycles_all": int(len(edf)), "edge_lifecycles_2004_2023": int(len(inwin)),
                   "unique_pairs_2004_2023": int(inwin.groupby(["company_a", "company_b"]).ngroups),
                   "share_ongoing_at_horizon_2004_2023": round(float(inwin["ongoing"].mean()), 4),
                   "origin_identified_share": round(float(inwin["origin"].notna().mean()), 4),
                   "terminated_by_counts": inwin["terminated_by"].fillna("ongoing").value_counts().to_dict(),
                   "duration_years_median": round(float(((inwin.edge_death - inwin.edge_birth).dt.days / 365.25).median()), 2),
                   "share_duration_lt_30d": round(float(((inwin.edge_death - inwin.edge_birth).dt.days < 30).mean()), 4),
                   "share_duration_lt_1y": round(float(((inwin.edge_death - inwin.edge_birth).dt.days < 365).mean()), 4),
                   "share_duration_lt_30d_closed_only": round(float(((inwin.loc[~inwin.ongoing, "edge_death"] - inwin.loc[~inwin.ongoing, "edge_birth"]).dt.days < 30).mean()), 4),
                   "old_pipeline_edges_for_reference": 155663,
                   "definition": "board-member events only; spell closes on any departure action; horizon 2025-12-31"}
log(f"edges all={len(edf):,}; formed 2004-2023={len(inwin):,}; ongoing {inwin['ongoing'].mean():.1%}; terminated_by {OUT['K0_edges']['terminated_by_counts']}")

# ======================================================================================
# Restatements: filing dates and misstated-period onset
# ======================================================================================
rest = pd.read_csv("data/restatements_with_dates.csv", parse_dates=["file_date", "res_begin_date", "res_end_date"], usecols=["company_fkey", "file_date", "res_begin_date", "res_end_date", "res_fraud", "res_sec_investigation"])
rest["cfkey"] = norm_cik(rest["company_fkey"])
rest = rest[rest["file_date"].notna() & (rest["file_date"] <= HORIZON)]
rf = defaultdict(list); ro = defaultdict(list); rs = defaultdict(list)
for c, f, b, fr, sec in zip(rest["cfkey"], rest["file_date"], rest["res_begin_date"], rest["res_fraud"].fillna(0), rest["res_sec_investigation"].fillna(0)):
    rf[c].append(f)
    if pd.notna(b): ro[c].append((b, f))
    if fr == 1 or sec == 1: rs[c].append(f)
OUT["restatements"] = {"n_filings_to_horizon": int(len(rest)), "share_with_period_begin": round(float(rest["res_begin_date"].notna().mean()), 4),
                       "median_years_file_minus_begin": round(float(((rest["file_date"] - rest["res_begin_date"]).dt.days / 365.25).median()), 2)}

RP = defaultdict(list)   # (begin, end, file) per firm
for c, f, b, e_ in zip(rest["cfkey"], rest["file_date"], rest["res_begin_date"], rest["res_end_date"]):
    if pd.notna(b): RP[c].append((b, e_ if pd.notna(e_) else f, f))
def served_during_misstatement(person, src, birth):
    """source restatement filed before edge birth whose misstated period overlaps the director's tenure at the source before the birth"""
    st = spell_start_at(person, src, birth)
    if st is None: return 0
    return int(any((f < birth) and (e_ >= st) and (b <= birth) for b, e_, f in RP.get(src, ())))
def present_at_filing(person, src, birth):
    """director's seat at the source covered the filing date of a restatement filed before the tie (witnessed the correction)"""
    for s_, e_ in SPELLS.get((person, src), ()):
        if s_ <= birth <= e_ or e_ <= birth:
            if any((f < birth) and (s_ <= f <= e_) for f in rf.get(src, ())): return 1
    return 0
def present_during_misstatement_expost(person, src, birth):
    """director's seat at the source (before the tie formed) overlapped a misstated period of a restatement filed at any time"""
    st = spell_start_at(person, src, birth)
    if st is None: return 0
    return int(any((e_ >= st) and (b <= birth) for b, e_, f in RP.get(src, ())))
def any_in(dates, lo, hi): return any(lo <= d <= hi for d in dates)
def any_before(dates, t, years=None):
    return any((d < t) and (years is None or (t - d).days <= 365.25 * years) for d in dates)

# ======================================================================================
# K1. Directional pair base
# ======================================================================================
recs = []
for r in inwin.itertuples(index=False):
    ca, cb, birth, death = r.company_a, r.company_b, r.edge_birth, r.edge_death
    we = min(death + pd.Timedelta(days=365), HORIZON)
    for src, dst in ((ca, cb), (cb, ca)):
        recs.append((f"{ca}|{cb}", src, dst, r.director, birth, death, r.birth_year,
                     int(any_before(rf.get(src, ()), birth)), int(any_before(rf.get(src, ()), birth, 5)),
                     int(any_in(rf.get(dst, ()), birth, we)),
                     int(any(birth <= b <= we for b, f in ro.get(dst, ()))),
                     int(any_in(rf.get(dst, ()), birth - pd.Timedelta(days=730), birth - pd.Timedelta(days=1))),
                     int(any(birth - pd.Timedelta(days=730) <= b < birth for b, f in ro.get(dst, ()))),
                     int(any_in(rf.get(dst, ()), birth, min(birth + pd.Timedelta(days=3 * 365), HORIZON))),
                     int(any(birth <= b <= min(birth + pd.Timedelta(days=3 * 365), HORIZON) for b, f in ro.get(dst, ()))),
                     r.origin == src if r.origin else None,
                     int(any_in(rf.get(dst, ()), birth - pd.Timedelta(days=3 * 365), birth - pd.Timedelta(days=1))),
                     int(any(birth - pd.Timedelta(days=3 * 365) <= b < birth for b, f in ro.get(dst, ()))),
                     int(any_in(rs.get(dst, ()), birth, min(birth + pd.Timedelta(days=3 * 365), HORIZON))),
                     int(any_in(rs.get(dst, ()), birth - pd.Timedelta(days=3 * 365), birth - pd.Timedelta(days=1))),
                     int(any_before(rf.get(dst, ()), birth)),
                     int(any_before(rs.get(src, ()), birth)),
                     served_during_misstatement(r.director, src, birth),
                     present_during_misstatement_expost(r.director, src, birth),
                     (birth - FIRST_EVENT_ALL.get(src, birth)).days / 365.25, (birth - FIRST_EVENT_ALL.get(dst, birth)).days / 365.25,
                     present_at_filing(r.director, src, birth), int(r.dst_ac)))
base = pd.DataFrame(recs, columns=["pid", "src", "dst", "director", "birth", "death", "by", "exposed", "exposed5y",
                                   "out_file", "out_onset", "pre_file", "pre_onset", "out_file_3y", "out_onset_3y", "src_is_origin",
                                   "pre_file_3y", "pre_onset_3y", "out_severe_3y", "pre_severe_3y", "dst_prior_any", "exposed_severe", "exposed_served", "exposed_present_expost", "src_feed_age", "dst_feed_age", "exposed_present_at_filing", "dst_ac"])
# data/auditor_company.csv = audit.feed07_current_auditor (WRDS label AUDITOR KEY): ONE row per registrant, the auditor current as of the extract date (2026-06);
# it is not an auditor history, so the indicator below means "same current auditor", not "same auditor at the appointment" (verified 2026-10-15, FIELD_DEFINITIONS_2026-10-15.md).
aud = pd.read_csv("data/auditor_company.csv", usecols=["company_fkey", "auditor_key"]); aud["c"] = norm_cik(aud["company_fkey"])
aud_sets = aud.dropna().groupby("c")["auditor_key"].agg(set).to_dict()
base["same_current_auditor"] = [int(len(aud_sets.get(a, set()) & aud_sets.get(b, set())) > 0) for a, b in zip(base["src"], base["dst"])]
_cw = pd.read_csv("data/cik_gvkey_crosswalk.csv", dtype=str); _linked = set(norm_cik(_cw["company_fkey"]))
base["both_linked"] = (base["src"].isin(_linked) & base["dst"].isin(_linked)).astype(int)
base.to_csv(f"data/pair_base_k{SUFFIX}.csv", index=False)
base_all = base.copy()                                   # both directions (panel construction, comparison block)
base = base[base["src_is_origin"] == True].copy()       # PRIMARY: appointment direction — source is the origin board, destination the appointing board
log(f"appointment-direction rows {len(base):,} of {len(base_all):,}")
OUT["direction"] = {"rows_both_directions": int(len(base_all)), "rows_appointment_direction": int(len(base)), "share_origin_identified": round(len(base) / (len(base_all) / 2), 4)}
log(f"directional rows {len(base):,}; exposed share {base.exposed.mean():.3f} (5y {base.exposed5y.mean():.3f}); out_file {base.out_file.mean():.4f}, out_onset {base.out_onset.mean():.4f}")

def twoway_ols(y, X, g1, g2):
    X = np.asarray(X, float); y = np.asarray(y, float)
    XtX_inv = np.linalg.inv(X.T @ X); b = XtX_inv @ X.T @ y; e = y - X @ b
    def meat(groups):
        codes = pd.factorize(groups)[0]; sums = np.zeros((codes.max() + 1, X.shape[1])); np.add.at(sums, codes, X * e[:, None]); return sums.T @ sums
    g12 = pd.factorize(pd.Series(g1).astype(str) + "#" + pd.Series(g2).astype(str))[0]
    V = XtX_inv @ (meat(g1) + meat(g2) - meat(g12)) @ XtX_inv
    return b, np.sqrt(np.diag(V))

def oneway_ols(y, X, g):
    X = np.asarray(X, float); y = np.asarray(y, float)
    XtX_inv = np.linalg.inv(X.T @ X); b = XtX_inv @ X.T @ y; e = y - X @ b
    codes = pd.factorize(g)[0]; sums = np.zeros((codes.max() + 1, X.shape[1])); np.add.at(sums, codes, X * e[:, None])
    V = XtX_inv @ (sums.T @ sums) @ XtX_inv
    return b, np.sqrt(np.diag(V))
def gap_dircl(df, out, exp="exposed"):
    X = np.column_stack([np.ones(len(df)), df[exp].values]); b, se = oneway_ols(df[out].values, X, df["director"].values)
    return {"diff_pp": round(100 * b[1], 3), "se_pp_director_clustered": round(100 * se[1], 3)}
def did_dircl(df, pre, post, exp="exposed"):
    st = pd.concat([df.assign(y=df[pre], post=0), df.assign(y=df[post], post=1)], ignore_index=True)
    X = np.column_stack([np.ones(len(st)), st[exp], st["post"], st[exp] * st["post"]]); b, se = oneway_ols(st["y"].values, X, st["director"].values)
    return {"diff_in_gaps_pp": round(100 * b[3], 3), "se_pp_director_clustered": round(100 * se[3], 3)}
def gap(df, out, exp="exposed"):
    X = np.column_stack([np.ones(len(df)), df[exp].values]); b, se = twoway_ols(df[out].values, X, df["src"].values, df["dst"].values)
    return {"exposed_rate": round(float(df.loc[df[exp] == 1, out].mean()), 4), "clean_rate": round(float(df.loc[df[exp] == 0, out].mean()), 4),
            "diff_pp": round(100 * b[1], 3), "se_pp": round(100 * se[1], 3), "t": round(float(b[1] / se[1]), 2),
            "ci95_pp": [round(100 * (b[1] - 1.96 * se[1]), 2), round(100 * (b[1] + 1.96 * se[1]), 2)],
            "n_rows": int(len(df)), "n_exposed_rows": int(df[exp].sum())}

K1 = {"outcome_file_in_window": gap(base, "out_file"), "outcome_onset_in_window": gap(base, "out_onset"),
      "outcome_file_in_window_exposed5y": gap(base, "out_file", "exposed5y"), "outcome_onset_in_window_exposed5y": gap(base, "out_onset", "exposed5y"),
      "placebo_pre_file": gap(base, "pre_file"), "placebo_pre_onset": gap(base, "pre_onset")}
# Equal 3-year windows before and after formation (edges formed 2004-2022 so the post window is fully observed)
eq = base[base.by <= 2022].copy()
K1["equal_window_3y"] = {"n_rows": int(len(eq)),
                         "pre_file": gap(eq, "pre_file_3y"), "post_file": gap(eq, "out_file_3y"),
                         "pre_onset": gap(eq, "pre_onset_3y"), "post_onset": gap(eq, "out_onset_3y"),
                         "pre_severe": gap(eq, "pre_severe_3y"), "post_severe": gap(eq, "out_severe_3y")}
def did(df, pre, post, exp="exposed"):
    st = pd.concat([df.assign(y=df[pre], post=0), df.assign(y=df[post], post=1)], ignore_index=True)
    X = np.column_stack([np.ones(len(st)), st[exp], st["post"], st[exp] * st["post"]])
    b, se = twoway_ols(st["y"].values, X, st["src"].values, st["dst"].values)
    return {"pre_gap_pp": round(100 * b[1], 3), "post_change_clean_pp": round(100 * b[2], 3), "diff_in_gaps_pp": round(100 * b[3], 3),
            "se_pp": round(100 * se[3], 3), "t": round(float(b[3] / se[3]), 2), "ci95_pp": [round(100 * (b[3] - 1.96 * se[3]), 2), round(100 * (b[3] + 1.96 * se[3]), 2)],
            "method": "stacked pre/post (3y each) rows per directional pair; exposed x post; two-way clustered (src, dst)"}
for lab, pre, post in (("file", "pre_file_3y", "out_file_3y"), ("onset", "pre_onset_3y", "out_onset_3y"), ("severe", "pre_severe_3y", "out_severe_3y")):
    K1["equal_window_3y"][f"did_{lab}"] = did(eq, pre, post)
K1["equal_window_3y"]["did_file_exposed5y"] = did(eq, "pre_file_3y", "out_file_3y", "exposed5y")
# DiD with formation-cohort x post fixed effects and with same-auditor controls (common-shock proxy)
def did_fe(df, pre, post, extra=(), exp="exposed", cell=None):
    """stacked pre/post regression: exposure, post, exposure x post, extra controls (+ x post), formation-year FE and formation-year x post FE
    (or cell FE and cell x post FE when `cell` names a column); the post main effect is included explicitly"""
    st = pd.concat([df.assign(y=df[pre], post=0), df.assign(y=df[post], post=1)], ignore_index=True)
    grp = st[cell].astype(str) if cell else st["by"].astype(int)
    cohort = pd.get_dummies(grp, prefix="c", drop_first=True).astype(float)
    cols = [st[exp].values, st["post"].values, (st[exp] * st["post"]).values] + [st[e_].values for e_ in extra] + [(st[e_] * st["post"]).values for e_ in extra]
    X = np.column_stack([np.ones(len(st)), *cols, cohort.values, (cohort.values * st["post"].values[:, None])])
    b, se = twoway_ols(st["y"].values, X, st["src"].values, st["dst"].values)
    return {"pre_gap_pp": round(100 * b[1], 3), "se_pre_pp": round(100 * se[1], 3), "diff_in_gaps_pp": round(100 * b[3], 3), "se_pp": round(100 * se[3], 3), "t": round(float(b[3] / se[3]), 2),
            "controls": [("cell FE + cell x post FE" if cell else "formation-year FE + formation-year x post FE")] + [f"{e_} (+ x post)" for e_ in extra]}
K1["equal_window_3y"]["did_file_cohortFE"] = did_fe(eq, "pre_file_3y", "out_file_3y")
for ex_, tag in (("exposed5y", "recent"), ("exposed_served", "present"), ("exposed_present_at_filing", "present_at_filing"), ("exposed_present_expost", "present_expost")):
    K1["equal_window_3y"][f"did_onset_cohortFE_{tag}"] = did_fe(eq, "pre_onset_3y", "out_onset_3y", exp=ex_)
    K1["equal_window_3y"][f"did_file_cohortFE_{tag}"] = did_fe(eq, "pre_file_3y", "out_file_3y", exp=ex_)
K1["equal_window_3y"]["did_onset_cohortFE"] = did_fe(eq, "pre_onset_3y", "out_onset_3y")
K1["equal_window_3y"]["did_file_cohortFE_sameauditor"] = did_fe(eq, "pre_file_3y", "out_file_3y", ("same_current_auditor",))
K1["equal_window_3y"]["share_same_current_auditor"] = round(float(eq["same_current_auditor"].mean()), 4)
K1["equal_window_3y"]["director_clustered"] = {"pre_file": gap_dircl(eq, "pre_file_3y"), "post_file": gap_dircl(eq, "out_file_3y"), "did_file": did_dircl(eq, "pre_file_3y", "out_file_3y"),
                                               "pre_onset": gap_dircl(eq, "pre_onset_3y"), "post_onset": gap_dircl(eq, "out_onset_3y"), "did_onset": did_dircl(eq, "pre_onset_3y", "out_onset_3y"),
                                               "n_directors": int(eq["director"].nunique()), "max_pairs_per_director": int(eq.groupby("director").size().max())}
bl = eq[eq.both_linked == 1]
K1["both_compustat_linked"] = {"n_rows": int(len(bl)), "pre_file": gap(bl, "pre_file_3y"), "post_file": gap(bl, "out_file_3y"), "did_file": did(bl, "pre_file_3y", "out_file_3y"),
                               "post_onset": gap(bl, "out_onset_3y"), "did_onset": did(bl, "pre_onset_3y", "out_onset_3y"),
                               "never_restated_post_file": gap(bl[bl.dst_prior_any == 0], "out_file_3y"), "never_restated_post_onset": gap(bl[bl.dst_prior_any == 0], "out_onset_3y")}
# equal-window analysis restricted to interlocks formed 2008-2022 (feed coverage)
e08 = eq[eq.by >= 2008]
K1["equal_window_formed_2008plus"] = {"n_rows": int(len(e08)), "pre_file": gap(e08, "pre_file_3y"), "post_file": gap(e08, "out_file_3y"), "did_file": did(e08, "pre_file_3y", "out_file_3y"),
                                      "did_file_cohortFE": did_fe(e08, "pre_file_3y", "out_file_3y"), "pre_onset": gap(e08, "pre_onset_3y"), "post_onset": gap(e08, "out_onset_3y"), "did_onset": did(e08, "pre_onset_3y", "out_onset_3y"),
                                      "never_restated_post_file": gap(e08[e08.dst_prior_any == 0], "out_file_3y"), "never_restated_post_onset": gap(e08[e08.dst_prior_any == 0], "out_onset_3y")}
# exposure = director served on the source board during the misstated period of a restatement filed before the tie
K1["exposure_served_during_misstatement"] = {"share_exposed_served": round(float(eq["exposed_served"].mean()), 4),
                                             "pre_file": gap(eq, "pre_file_3y", "exposed_served"), "post_file": gap(eq, "out_file_3y", "exposed_served"), "did_file": did(eq, "pre_file_3y", "out_file_3y", "exposed_served"),
                                             "pre_onset": gap(eq, "pre_onset_3y", "exposed_served"), "post_onset": gap(eq, "out_onset_3y", "exposed_served"), "did_onset": did(eq, "pre_onset_3y", "out_onset_3y", "exposed_served"),
                                             "never_restated_post_file": gap(eq[eq.dst_prior_any == 0], "out_file_3y", "exposed_served"), "never_restated_post_onset": gap(eq[eq.dst_prior_any == 0], "out_onset_3y", "exposed_served"),
                                             "life_plus1y_file": gap(base, "out_file", "exposed_served"), "life_plus1y_onset": gap(base, "out_onset", "exposed_served")}
# exposure = source restated within 5 years before the tie (recent exposure), both outcomes
K1["exposure_recent_5y"] = {"share_exposed5y": round(float(eq["exposed5y"].mean()), 4),
                            "pre_file": gap(eq, "pre_file_3y", "exposed5y"), "post_file": gap(eq, "out_file_3y", "exposed5y"), "did_file": did(eq, "pre_file_3y", "out_file_3y", "exposed5y"),
                            "pre_onset": gap(eq, "pre_onset_3y", "exposed5y"), "post_onset": gap(eq, "out_onset_3y", "exposed5y"), "did_onset": did(eq, "pre_onset_3y", "out_onset_3y", "exposed5y"),
                            "never_restated_pre_onset": gap(eq[eq.dst_prior_any == 0], "pre_onset_3y", "exposed5y"), "never_restated_post_onset": gap(eq[eq.dst_prior_any == 0], "out_onset_3y", "exposed5y"),
                            "never_restated_post_file": gap(eq[eq.dst_prior_any == 0], "out_file_3y", "exposed5y")}
K1["exposure_served_during_misstatement"]["never_restated_pre_onset"] = gap(eq[eq.dst_prior_any == 0], "pre_onset_3y", "exposed_served")
# ex-post exposure: director present at the origin during a misstated period restated at any time (undetected misstatement counts as exposure)
K1["exposure_present_expost"] = {"share": round(float(eq["exposed_present_expost"].mean()), 4),
                                 "pre_file": gap(eq, "pre_file_3y", "exposed_present_expost"), "post_file": gap(eq, "out_file_3y", "exposed_present_expost"), "did_file": did(eq, "pre_file_3y", "out_file_3y", "exposed_present_expost"),
                                 "pre_onset": gap(eq, "pre_onset_3y", "exposed_present_expost"), "post_onset": gap(eq, "out_onset_3y", "exposed_present_expost"), "did_onset": did(eq, "pre_onset_3y", "out_onset_3y", "exposed_present_expost"),
                                 "never_restated_post_onset": gap(eq[eq.dst_prior_any == 0], "out_onset_3y", "exposed_present_expost"), "never_restated_pre_onset": gap(eq[eq.dst_prior_any == 0], "pre_onset_3y", "exposed_present_expost")}
# director present at the origin when the restatement was filed (witnessed the discovery and correction)
K1["exposure_present_at_filing"] = {"share": round(float(eq["exposed_present_at_filing"].mean()), 4),
                                    "pre_file": gap(eq, "pre_file_3y", "exposed_present_at_filing"), "post_file": gap(eq, "out_file_3y", "exposed_present_at_filing"), "did_file": did(eq, "pre_file_3y", "out_file_3y", "exposed_present_at_filing"),
                                    "pre_onset": gap(eq, "pre_onset_3y", "exposed_present_at_filing"), "post_onset": gap(eq, "out_onset_3y", "exposed_present_at_filing"), "did_onset": did(eq, "pre_onset_3y", "out_onset_3y", "exposed_present_at_filing"),
                                    "never_restated_pre_onset": gap(eq[eq.dst_prior_any == 0], "pre_onset_3y", "exposed_present_at_filing"), "never_restated_post_onset": gap(eq[eq.dst_prior_any == 0], "out_onset_3y", "exposed_present_at_filing"),
                                    "never_restated_post_file": gap(eq[eq.dst_prior_any == 0], "out_file_3y", "exposed_present_at_filing"), "never_restated_did_onset": did(eq[eq.dst_prior_any == 0], "pre_onset_3y", "out_onset_3y", "exposed_present_at_filing")}
# feed-age: restrict to pairs in which both firms have been in the feed for at least 3 years before formation (pre-window observable; removes age sorting through feed entry)
ob = eq[(eq.src_feed_age >= 3) & (eq.dst_feed_age >= 3)]
K1["observable_pre_window_both_firms_3y_in_feed"] = {"n_rows": int(len(ob)), "share_rows": round(len(ob) / len(eq), 4), "mean_dst_feed_age_exposed": round(float(eq.loc[eq.exposed == 1, "dst_feed_age"].mean()), 2), "mean_dst_feed_age_clean": round(float(eq.loc[eq.exposed == 0, "dst_feed_age"].mean()), 2),
                                                     "pre_file": gap(ob, "pre_file_3y"), "post_file": gap(ob, "out_file_3y"), "did_file": did(ob, "pre_file_3y", "out_file_3y"),
                                                     "pre_onset": gap(ob, "pre_onset_3y"), "post_onset": gap(ob, "out_onset_3y"), "did_onset": did(ob, "pre_onset_3y", "out_onset_3y"),
                                                     "never_restated_post_onset": gap(ob[ob.dst_prior_any == 0], "out_onset_3y")}
# board-refresh test (a sorting mechanism): do destinations that restated in the 12 months before the appointment hire from restating origins more often?
eq["dst_filed_12m_before"] = [int(any_in(rf.get(d_, ()), b_ - pd.Timedelta(days=365), b_ - pd.Timedelta(days=1))) for d_, b_ in zip(eq.dst, eq.birth)]
X_ = np.column_stack([np.ones(len(eq)), eq["dst_filed_12m_before"].values]); b_, se_ = twoway_ols(eq["exposed"].values, X_, eq["src"].values, eq["dst"].values)
X5 = np.column_stack([np.ones(len(eq)), eq["dst_filed_12m_before"].values]); b5, se5 = twoway_ols(eq["exposed5y"].values, X5, eq["src"].values, eq["dst"].values)
K1["board_refresh_test"] = {"share_dst_filed_12m_before": round(float(eq["dst_filed_12m_before"].mean()), 4),
                            "exposed_share_if_dst_filed_12m_before": round(float(eq.loc[eq.dst_filed_12m_before == 1, "exposed"].mean()), 4), "exposed_share_otherwise": round(float(eq.loc[eq.dst_filed_12m_before == 0, "exposed"].mean()), 4),
                            "diff_pp": round(100 * b_[1], 3), "se_pp": round(100 * se_[1], 3),
                            "recent_exposed_share_if_dst_filed_12m_before": round(float(eq.loc[eq.dst_filed_12m_before == 1, "exposed5y"].mean()), 4), "recent_exposed_share_otherwise": round(float(eq.loc[eq.dst_filed_12m_before == 0, "exposed5y"].mean()), 4),
                            "recent_diff_pp": round(100 * b5[1], 3), "recent_se_pp": round(100 * se5[1], 3)}
# destinations with no restatement filed and no misstated period begun before the tie: post-formation gaps under three exposure definitions
cl = eq[(eq.dst_prior_any == 0) & (eq.pre_onset == 0)]
K1["clean_destinations_no_history"] = {"n_rows": int(len(cl)), "share_of_rows": round(len(cl) / len(eq), 4)}
def gap_yearfe(df, out, exp):
    yd_ = pd.get_dummies(df["by"].astype(int), drop_first=True).astype(float).values
    X = np.column_stack([np.ones(len(df)), df[exp].values, yd_]); b, se = twoway_ols(df[out].values, X, df["src"].values, df["dst"].values)
    return {"diff_pp": round(100 * b[1], 3), "se_pp": round(100 * se[1], 3), "t": round(float(b[1] / se[1]), 2)}
for ex_ in ("exposed", "exposed5y", "exposed_served", "exposed_present_at_filing"):
    K1["clean_destinations_no_history"][ex_] = {"n_exposed": int(cl[ex_].sum()), "post_file": gap(cl, "out_file_3y", ex_), "post_onset": gap(cl, "out_onset_3y", ex_), "post_severe": gap(cl, "out_severe_3y", ex_),
                                                "post_onset_yearFE": gap_yearfe(cl, "out_onset_3y", ex_), "post_file_yearFE": gap_yearfe(cl, "out_file_3y", ex_)}
    K1.setdefault("never_restated_yearFE", {})[f"post_onset_yearFE_{ex_}"] = gap_yearfe(eq[eq.dst_prior_any == 0], "out_onset_3y", ex_)
# blank-check companies (SEC submissions: SIC 6770 now, or a current/former name of the "... Acquisition Corp" type)
SPAC = set()
if os.path.exists("data/sec_sic_by_cik.csv"):
    sec = pd.read_csv("data/sec_sic_by_cik.csv", dtype=str).fillna("")
    sec["cik"] = norm_cik(sec["cik"])
    pat = r"acquisition\s+(corp|co\b|company|ltd|limited|holdings|inc|plc|partners|group)|blank\s*check|\bspac\b"
    nm = (sec["name"] + " | " + sec["former_names"]).str.lower()
    sec["spac"] = (sec["sic"].astype(str).str.strip() == "6770") | nm.str.contains(pat, regex=True)
    SPAC = set(sec.loc[sec["spac"], "cik"])
    K1["blank_check_flag"] = {"ciks_fetched": int(len(sec)), "ciks_fetched_ok": int((sec["status"] == "ok").sum()), "n_blank_check_ciks": int(len(SPAC)),
                              "share_pairs_with_blank_check_firm": round(float((eq.src.isin(SPAC) | eq.dst.isin(SPAC)).mean()), 4),
                              "share_restatement_filings_2020_2023_by_blank_check": round(float(rest[(rest.file_date >= "2020-01-01") & (rest.file_date <= "2023-12-31")]["cfkey"].isin(SPAC).mean()), 4)}
    ns = eq[~(eq.src.isin(SPAC) | eq.dst.isin(SPAC))]
    K1["excluding_blank_check_firms"] = {"n_rows": int(len(ns)), "pre_file": gap(ns, "pre_file_3y"), "post_file": gap(ns, "out_file_3y"), "did_file": did(ns, "pre_file_3y", "out_file_3y"),
                                         "did_file_cohortFE": did_fe(ns, "pre_file_3y", "out_file_3y"), "pre_onset": gap(ns, "pre_onset_3y"), "post_onset": gap(ns, "out_onset_3y"), "did_onset": did(ns, "pre_onset_3y", "out_onset_3y"),
                                         "never_restated_post_file": gap(ns[ns.dst_prior_any == 0], "out_file_3y"), "never_restated_post_onset": gap(ns[ns.dst_prior_any == 0], "out_onset_3y"),
                                         "life_plus1y_file": gap(base[~(base.src.isin(SPAC) | base.dst.isin(SPAC))], "out_file"),
                                         "recent5y_did_onset": did(ns, "pre_onset_3y", "out_onset_3y", "exposed5y"), "recent5y_post_onset": gap(ns, "out_onset_3y", "exposed5y"), "recent5y_pre_onset": gap(ns, "pre_onset_3y", "exposed5y"),
                                         "clean_post_onset": gap(ns[(ns.dst_prior_any == 0) & (ns.pre_onset == 0)], "out_onset_3y"), "clean_post_onset_recent5y": gap(ns[(ns.dst_prior_any == 0) & (ns.pre_onset == 0)], "out_onset_3y", "exposed5y")}
    log(f"blank-check ciks {len(SPAC):,}; pairs involving one {K1['blank_check_flag']['share_pairs_with_blank_check_firm']:.1%}; excl. DiD file {K1['excluding_blank_check_firms']['did_file']['diff_in_gaps_pp']:+.2f}")
# name-collision sensitivity: drop pairs whose director's surname is shared by many distinct director names
_sur_all = pd.Series([x.split("|")[1] for x in ten["person"].unique()]).value_counts()
eq_sur = eq["director"].str.split("|").str[1].map(_sur_all).fillna(0)
K1["common_name_sensitivity"] = {}
for thr in (150, 50):
    sub_ = eq[eq_sur <= thr]; cl_ = sub_[(sub_.dst_prior_any == 0) & (sub_.pre_onset == 0)]
    K1["common_name_sensitivity"][f"surname_shared_by_at_most_{thr}_names"] = {"n_rows": int(len(sub_)), "share_rows_kept": round(len(sub_) / len(eq), 4),
        "pre_file": gap(sub_, "pre_file_3y"), "post_file": gap(sub_, "out_file_3y"), "did_file": did(sub_, "pre_file_3y", "out_file_3y"),
        "pre_onset": gap(sub_, "pre_onset_3y"), "post_onset": gap(sub_, "out_onset_3y"), "did_onset": did(sub_, "pre_onset_3y", "out_onset_3y"),
        "clean_destinations_post_onset": gap(cl_, "out_onset_3y"), "clean_destinations_post_onset_recent5y": gap(cl_, "out_onset_3y", "exposed5y")}
# artifact-flag alternative to the seat cap: remove pairs with a sponsor-family or shell registrant or a common-surname director, keep genuine high-degree directors
if os.path.exists("data/sec_sic_by_cik.csv"):
    sec1 = pd.read_csv("data/sec_sic_by_cik.csv", dtype=str).fillna(""); sec1["cik"] = norm_cik(sec1["cik"]); sic1 = sec1["sic"].str.strip()
    fundpat1 = r"\bfund\b|\bfunds\b|\btrust\b|portfolio|\betf\b|income|investors|\bseries\b|\bvariable\b|municipal|tax[- ]free|\bbond\b|\bbdc\b"
    FAM = set(sec1.loc[sic1.str[:2].eq("65") | sic1.isin(["6798", "6221", "6722", "6726"]) | ((sic1 == "") & sec1["name"].str.lower().str.contains(fundpat1, regex=True)), "cik"])
    SHELL = set(sec1.loc[(sic1 == "9995") | sec1["name"].str.lower().str.contains("acquisition"), "cik"]) | SPAC
    _sur_all2 = pd.Series([x.split("|")[1] for x in ten["person"].unique()]).value_counts()
    art = eq.src.isin(SHELL) | eq.dst.isin(SHELL) | eq.src.isin(FAM) | eq.dst.isin(FAM) | (eq["director"].str.split("|").str[1].map(_sur_all2).fillna(0) > 150)
    hi = eq["director"].map(_k) > 6   # directors interlocked with more than 6 firms (always False in a capped run)
    def blk_(df_):
        cl_ = df_[(df_.dst_prior_any == 0) & (df_.pre_onset == 0)]
        return {"n_rows": int(len(df_)), "pre_file": gap(df_, "pre_file_3y"), "post_file": gap(df_, "out_file_3y"), "did_file": did(df_, "pre_file_3y", "out_file_3y"),
                "pre_onset": gap(df_, "pre_onset_3y"), "post_onset": gap(df_, "out_onset_3y"), "did_onset": did(df_, "pre_onset_3y", "out_onset_3y"),
                "never_restated_post_file": gap(df_[df_.dst_prior_any == 0], "out_file_3y"), "never_restated_post_onset": gap(df_[df_.dst_prior_any == 0], "out_onset_3y"),
                "clean_post_onset": gap(cl_, "out_onset_3y"), "clean_post_onset_recent5y": gap(cl_, "out_onset_3y", "exposed5y"), "life_plus1y_file": None}
    K1["artifact_flags"] = {"n_family_ciks": len(FAM), "n_shell_ciks": len(SHELL), "share_pairs_flagged": round(float(art.mean()), 4),
                            "share_flagged_among_high_seat_directors": round(float(art[hi].mean()), 4) if hi.any() else None, "share_flagged_among_other_directors": round(float(art[~hi].mean()), 4)}
    K1["artifacts_removed_all_directors"] = blk_(eq[~art]); K1["artifacts_removed_all_directors"]["life_plus1y_file"] = gap(base[~(base.src.isin(SHELL) | base.dst.isin(SHELL) | base.src.isin(FAM) | base.dst.isin(FAM) | (base["director"].str.split("|").str[1].map(_sur_all2).fillna(0) > 150))], "out_file")
    if hi.any():
        K1["high_seat_directors_only"] = blk_(eq[hi]); K1["high_seat_directors_only_artifacts_removed"] = blk_(eq[hi & ~art])
    log(f"artifact flags: {art.mean():.1%} of pairs; artifacts-removed DiD file {K1['artifacts_removed_all_directors']['did_file']['diff_in_gaps_pp']:+.2f}, clean recent {K1['artifacts_removed_all_directors']['clean_post_onset_recent5y']['diff_pp']:+.2f}")
    # the most connected names: how many firms, and what share of them are shells / family registrants (no names recorded)
    top_ = _k.sort_values(ascending=False).head(5)
    K1["most_connected_names"] = [{"rank": i_ + 1, "firms_interlocked": int(n_), "share_firms_shell": round(float(pd.Series(list(_seats.loc[_seats.director == d_, "c"].unique())).isin(SHELL).mean()), 3),
                                   "share_firms_family": round(float(pd.Series(list(_seats.loc[_seats.director == d_, "c"].unique())).isin(FAM).mean()), 3)} for i_, (d_, n_) in enumerate(top_.items())]
# event-time gaps: destination outcome in each relative year from -3 to +2 around formation (parallel-trends check)
def yr_ind(dates, lo, hi): return int(any(lo <= d_ < hi for d_ in dates))
K1["event_time_gaps"] = {}
for k_ in range(-3, 3):
    lo_ = eq["birth"] + pd.to_timedelta(k_ * 365, unit="D"); hi_ = eq["birth"] + pd.to_timedelta((k_ + 1) * 365, unit="D")
    yf = np.array([yr_ind(rf.get(d_, ()), a_, b_) for d_, a_, b_ in zip(eq.dst, lo_, hi_)])
    yo = np.array([int(any(a_ <= bb_ < b_ for bb_, f_ in ro.get(d_, ()))) for d_, a_, b_ in zip(eq.dst, lo_, hi_)])
    tmp = eq.assign(yf=yf, yo=yo)
    K1["event_time_gaps"][str(k_)] = {"file": gap(tmp, "yf"), "onset": gap(tmp, "yo"), "file_recent5y": gap(tmp, "yf", "exposed5y"), "onset_recent5y": gap(tmp, "yo", "exposed5y")}
log("event-time file gaps: " + ", ".join(f"{k_}: {v['file']['diff_pp']:+.2f}" for k_, v in K1["event_time_gaps"].items()))
# appointment-weighted estimates: an appointment that creates several pairs (one per existing seat of the director) gets total weight one
def twoway_wls(y, X, g1, g2, w_):
    sw = np.sqrt(np.asarray(w_, float))[:, None]; return twoway_ols(np.asarray(y, float) * sw[:, 0], np.asarray(X, float) * sw, g1, g2)
def gap_w(df, out, exp="exposed"):
    w_ = 1.0 / df.groupby(["director", "dst", "birth"])[out].transform("size").values
    X = np.column_stack([np.ones(len(df)), df[exp].values]); b, se = twoway_wls(df[out].values, X, df["src"].values, df["dst"].values, w_)
    return {"diff_pp": round(100 * b[1], 3), "se_pp": round(100 * se[1], 3), "t": round(float(b[1] / se[1]), 2)}
def did_w(df, pre, post, exp="exposed"):
    st = pd.concat([df.assign(y=df[pre], post=0), df.assign(y=df[post], post=1)], ignore_index=True)
    w_ = 1.0 / st.groupby(["director", "dst", "birth", "post"])["y"].transform("size").values
    X = np.column_stack([np.ones(len(st)), st[exp], st["post"], st[exp] * st["post"]]); b, se = twoway_wls(st["y"].values, X, st["src"].values, st["dst"].values, w_)
    return {"diff_in_gaps_pp": round(100 * b[3], 3), "se_pp": round(100 * se[3], 3), "t": round(float(b[3] / se[3]), 2)}
K1["appointment_weighted"] = {"n_appointments": int(eq.groupby(["director", "dst", "birth"]).ngroups), "mean_pairs_per_appointment": round(float(len(eq) / eq.groupby(["director", "dst", "birth"]).ngroups), 3),
                              "pre_file": gap_w(eq, "pre_file_3y"), "post_file": gap_w(eq, "out_file_3y"), "did_file": did_w(eq, "pre_file_3y", "out_file_3y"),
                              "pre_onset": gap_w(eq, "pre_onset_3y"), "post_onset": gap_w(eq, "out_onset_3y"), "did_onset": did_w(eq, "pre_onset_3y", "out_onset_3y"),
                              "clean_post_onset": gap_w(eq[(eq.dst_prior_any == 0) & (eq.pre_onset == 0)], "out_onset_3y"), "clean_post_onset_recent5y": gap_w(eq[(eq.dst_prior_any == 0) & (eq.pre_onset == 0)], "out_onset_3y", "exposed5y")}
OUT["restatements"]["share_filings_without_period_begin"] = round(1 - OUT["restatements"]["share_with_period_begin"], 4) if "restatements" in OUT else None
# reverse direction: the origin as the firm at risk, exposure = appointing board restated before the tie
rv = base_all[(base_all.by <= 2022) & (base_all.src_is_origin == False)].copy()
rv["dst_prior_any_r"] = rv["dst_prior_any"]
K1["reverse_direction"] = {"n_rows": int(len(rv)), "life_plus1y_file": gap(base_all[base_all.src_is_origin == False], "out_file"), "life_plus1y_onset": gap(base_all[base_all.src_is_origin == False], "out_onset"), "pre_file": gap(rv, "pre_file_3y"), "post_file": gap(rv, "out_file_3y"), "did_file": did(rv, "pre_file_3y", "out_file_3y"),
                           "pre_onset": gap(rv, "pre_onset_3y"), "post_onset": gap(rv, "out_onset_3y"), "did_onset": did(rv, "pre_onset_3y", "out_onset_3y"),
                           "never_restated_post_onset": gap(rv[rv.dst_prior_any == 0], "out_onset_3y"), "never_restated_post_file": gap(rv[rv.dst_prior_any == 0], "out_file_3y"),
                           "clean_post_onset": gap(rv[(rv.dst_prior_any == 0) & (rv.pre_onset == 0)], "out_onset_3y"), "clean_post_onset_recent5y": gap(rv[(rv.dst_prior_any == 0) & (rv.pre_onset == 0)], "out_onset_3y", "exposed5y"),
                           "did_onset_cohortFE": did_fe(rv, "pre_onset_3y", "out_onset_3y"), "did_file_cohortFE": did_fe(rv, "pre_file_3y", "out_file_3y"),
                           "recent_did_onset": did(rv, "pre_onset_3y", "out_onset_3y", "exposed5y"), "recent_did_onset_cohortFE": did_fe(rv, "pre_onset_3y", "out_onset_3y", exp="exposed5y")}
if "SPAC" in globals():
    K1["reverse_direction"]["did_onset_cohortFE_excl_blank_check"] = did_fe(rv[~(rv.src.isin(SPAC) | rv.dst.isin(SPAC))], "pre_onset_3y", "out_onset_3y")
    K1["reverse_direction"]["recent_did_onset_cohortFE_excl_blank_check"] = did_fe(rv[~(rv.src.isin(SPAC) | rv.dst.isin(SPAC))], "pre_onset_3y", "out_onset_3y", exp="exposed5y")
# refresh test with formation-year effects
yd_ = pd.get_dummies(eq["by"].astype(int), prefix="y", drop_first=True).astype(float).values
for ex_, lab_ in (("exposed", "any"), ("exposed5y", "recent")):
    Xr = np.column_stack([np.ones(len(eq)), eq["dst_filed_12m_before"].values, yd_]); br_, ser_ = twoway_ols(eq[ex_].values, Xr, eq["src"].values, eq["dst"].values)
    K1["board_refresh_test"][f"{lab_}_diff_pp_yearFE"] = round(100 * br_[1], 3); K1["board_refresh_test"][f"{lab_}_se_pp_yearFE"] = round(100 * ser_[1], 3)
# concurrent seats at formation (alternative to the cumulative cap)
PSP2 = defaultdict(list)
for p_, s_, e_ in zip(ten["person"], ten["start"], ten["end"]): PSP2[p_].append((s_, e_))
eq["seats_at_birth"] = [sum(1 for s_, e_ in PSP2[d_] if s_ <= b_ <= e_) for d_, b_ in zip(eq.director, eq.birth)]
K1["cap_definition"] = {"cap_is_cumulative_firms_interlocked_2004_2025": True, "share_pairs_director_concurrent_seats_le6": round(float((eq.seats_at_birth <= 6).mean()), 4), "mean_seats_at_birth": round(float(eq.seats_at_birth.mean()), 2)}
cc_ = eq[eq.seats_at_birth <= 6]; ccl = cc_[(cc_.dst_prior_any == 0) & (cc_.pre_onset == 0)]
K1["concurrent_seats_le6_at_formation"] = {"n_rows": int(len(cc_)), "pre_file": gap(cc_, "pre_file_3y"), "post_file": gap(cc_, "out_file_3y"), "did_file": did(cc_, "pre_file_3y", "out_file_3y"),
                                           "pre_onset": gap(cc_, "pre_onset_3y"), "post_onset": gap(cc_, "out_onset_3y"), "did_onset": did(cc_, "pre_onset_3y", "out_onset_3y"),
                                           "never_restated_post_onset": gap(cc_[cc_.dst_prior_any == 0], "out_onset_3y"), "clean_post_onset": gap(ccl, "out_onset_3y"), "clean_post_onset_recent5y": gap(ccl, "out_onset_3y", "exposed5y")}
# DiD with feed-age controls (both firms' years in the feed at formation, and their interactions with post) and, in the Compustat subsample, destination size
K1["equal_window_3y"]["did_file_cohortFE_feedage"] = did_fe(eq, "pre_file_3y", "out_file_3y", ("dst_feed_age", "src_feed_age"))
K1["equal_window_3y"]["did_onset_cohortFE_feedage"] = did_fe(eq, "pre_onset_3y", "out_onset_3y", ("dst_feed_age", "src_feed_age"))
_cw = pd.read_csv("data/cik_gvkey_crosswalk.csv", dtype=str); _cw["cik"] = norm_cik(_cw["company_fkey"]); _cw["gvkey"] = _cw["gvkey"].astype(str).str.zfill(6); _cw = _cw.drop_duplicates("cik")[["cik", "gvkey"]]
_cp = pd.read_csv("data/compustat_financials.csv"); _cp["gvkey"] = _cp["gvkey"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(6); _cp = _cp[_cp["assets"] > 0]
_sz = _cp.merge(_cw, on="gvkey").assign(size=lambda d_: np.log(d_["assets"]))[["cik", "year", "size"]]
eq_sz = eq.merge(_sz.rename(columns={"cik": "dst", "year": "by", "size": "dst_size"}), on=["dst", "by"], how="left")
eq_sz = eq_sz[eq_sz.dst_size.notna()].copy()
K1["equal_window_3y"]["did_file_cohortFE_feedage_dstsize"] = {**did_fe(eq_sz, "pre_file_3y", "out_file_3y", ("dst_feed_age", "src_feed_age", "dst_size")), "n_rows": int(len(eq_sz))}
K1["equal_window_3y"]["did_onset_cohortFE_feedage_dstsize"] = {**did_fe(eq_sz, "pre_onset_3y", "out_onset_3y", ("dst_feed_age", "src_feed_age", "dst_size")), "n_rows": int(len(eq_sz))}
K1["equal_window_3y"]["mean_dst_size_exposed_clean"] = [round(float(eq_sz.loc[eq_sz.exposed == 1, "dst_size"].mean()), 2), round(float(eq_sz.loc[eq_sz.exposed == 0, "dst_size"].mean()), 2)]
# formation-year x destination-size-decile cells (feed-age decile where Compustat size is missing)
_eqc = eq.merge(_sz.rename(columns={"cik": "dst", "year": "by", "size": "dst_size"}), on=["dst", "by"], how="left")
_eqc["cell_var"] = _eqc["dst_size"].fillna(-100 + _eqc["dst_feed_age"])
_eqc["cell"] = _eqc["by"].astype(str) + "#" + _eqc.groupby("by")["cell_var"].transform(lambda s_: pd.qcut(s_.rank(method="first"), 10, labels=False, duplicates="drop")).astype(str)
for ex_, tag in (("exposed", "any"), ("exposed5y", "recent"), ("exposed_served", "present")):
    K1["equal_window_3y"][f"did_onset_cellFE_{tag}"] = did_fe(_eqc, "pre_onset_3y", "out_onset_3y", exp=ex_, cell="cell")
    K1["equal_window_3y"][f"did_file_cellFE_{tag}"] = did_fe(_eqc, "pre_file_3y", "out_file_3y", exp=ex_, cell="cell")
K1["equal_window_3y"]["did_onset_cohortFE_recent_dstsize"] = {**did_fe(eq_sz, "pre_onset_3y", "out_onset_3y", ("dst_feed_age", "src_feed_age", "dst_size"), exp="exposed5y"), "n_rows": int(len(eq_sz))}
# precision summary for the cohort-adjusted onset changes (Table 3, Panel D): two-sided 95% limits and the 80%-power two-sided minimum detectable change (2.80 x SE)
_adj = {"any": K1["equal_window_3y"]["did_onset_cohortFE"], "recent": K1["equal_window_3y"]["did_onset_cohortFE_recent"], "recent_cells": K1["equal_window_3y"]["did_onset_cellFE_recent"],
        "present": K1["equal_window_3y"].get("did_onset_cohortFE_present"), "present_at_filing": K1["equal_window_3y"].get("did_onset_cohortFE_present_at_filing"), "present_expost": K1["equal_window_3y"].get("did_onset_cohortFE_present_expost")}
_adj = {k: v for k, v in _adj.items() if v}
K1["equal_window_3y"]["adjusted_change_precision"] = {k: {"upper95_two_sided_pp": round(v["diff_in_gaps_pp"] + 1.96 * v["se_pp"], 2), "lower95_two_sided_pp": round(v["diff_in_gaps_pp"] - 1.96 * v["se_pp"], 2), "mde80_two_sided_pp": round(2.8016 * v["se_pp"], 2)} for k, v in _adj.items()}
K1["equal_window_3y"]["adjusted_change_precision"]["max_upper95_pp"] = round(max(v["upper95_two_sided_pp"] for v in K1["equal_window_3y"]["adjusted_change_precision"].values() if isinstance(v, dict)), 2)
K1["equal_window_3y"]["adjusted_change_precision"]["mde80_range_pp"] = [round(min(v["mde80_two_sided_pp"] for v in K1["equal_window_3y"]["adjusted_change_precision"].values() if isinstance(v, dict)), 2), round(max(v["mde80_two_sided_pp"] for v in K1["equal_window_3y"]["adjusted_change_precision"].values() if isinstance(v, dict)), 2)]
log(f"recent exposure onset DiD: plain {K1['equal_window_3y']['did_onset_recent' if 'did_onset_recent' in K1['equal_window_3y'] else 'did_onset']['diff_in_gaps_pp']:+.2f}; year FE {K1['equal_window_3y']['did_onset_cohortFE_recent']['diff_in_gaps_pp']:+.2f} ({K1['equal_window_3y']['did_onset_cohortFE_recent']['se_pp']:.2f}); cells {K1['equal_window_3y']['did_onset_cellFE_recent']['diff_in_gaps_pp']:+.2f}")
_cw0 = pd.read_csv("data/cik_gvkey_crosswalk.csv", dtype=str); _cw0["cik"] = norm_cik(_cw0["company_fkey"]); _cw0["gvkey"] = _cw0["gvkey"].astype(str).str.zfill(6); _cw0 = _cw0.drop_duplicates("cik")[["cik", "gvkey"]]
_cp0 = pd.read_csv("data/compustat_financials.csv"); _cp0["gvkey"] = _cp0["gvkey"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(6); _cp0 = _cp0[_cp0["assets"] > 0]
_sz = _cp0.merge(_cw0, on="gvkey").assign(size=lambda d_: np.log(d_["assets"]))[["cik", "year", "size"]]
# common-shock controls: same 2-digit SIC (EDGAR), destination-industry x post effects, same headquarters state (EDGAR business address)
if os.path.exists("data/sec_sic_by_cik.csv"):
    sic_map = pd.read_csv("data/sec_sic_by_cik.csv", dtype=str).fillna("").assign(cik=lambda d_: norm_cik(d_["cik"])).set_index("cik")["sic"].str.strip().str[:2]
    eq["sic2_src"] = eq["src"].map(sic_map).fillna(""); eq["sic2_dst"] = eq["dst"].map(sic_map).fillna("")
    eq["same_sic2"] = ((eq.sic2_src == eq.sic2_dst) & (eq.sic2_dst != "")).astype(int)
    def did_fe_dstind(df, pre, post, extra=(), exp="exposed"):
        st = pd.concat([df.assign(y=df[pre], post=0), df.assign(y=df[post], post=1)], ignore_index=True)
        cohort = pd.get_dummies(st["by"].astype(int), prefix="c", drop_first=True).astype(float)
        ind = pd.get_dummies(st["sic2_dst"].replace("", "na"), prefix="i", drop_first=True).astype(float)
        cols = [st[exp].values, st["post"].values, (st[exp] * st["post"]).values] + [st[e_].values for e_ in extra] + [(st[e_] * st["post"]).values for e_ in extra]
        X = np.column_stack([np.ones(len(st)), *cols, cohort.values, cohort.values * st["post"].values[:, None], ind.values, ind.values * st["post"].values[:, None]])
        b, se = twoway_ols(st["y"].values, X, st["src"].values, st["dst"].values)
        return {"pre_gap_pp": round(100 * b[1], 3), "se_pre_pp": round(100 * se[1], 3), "diff_in_gaps_pp": round(100 * b[3], 3), "se_pp": round(100 * se[3], 3), "t": round(float(b[3] / se[3]), 2), "n_industries": int(ind.shape[1] + 1)}
    K1["common_shock_controls"] = {"share_same_sic2": round(float(eq["same_sic2"].mean()), 4),
                                   "exposed_share_same_sic2": round(float(eq.loc[eq.same_sic2 == 1, "exposed"].mean()), 4), "exposed_share_diff_sic2": round(float(eq.loc[eq.same_sic2 == 0, "exposed"].mean()), 4),
                                   "did_file_cohortFE_sameSIC": did_fe(eq, "pre_file_3y", "out_file_3y", ("same_sic2",)), "did_onset_cohortFE_sameSIC": did_fe(eq, "pre_onset_3y", "out_onset_3y", ("same_sic2",)),
                                   "did_file_cohortFE_dstIndustryXpost": did_fe_dstind(eq, "pre_file_3y", "out_file_3y"), "did_onset_cohortFE_dstIndustryXpost": did_fe_dstind(eq, "pre_onset_3y", "out_onset_3y"),
                                   "clean_post_onset_recent5y_same_sic2": gap(eq[(eq.dst_prior_any == 0) & (eq.pre_onset == 0) & (eq.same_sic2 == 1)], "out_onset_3y", "exposed5y"),
                                   "clean_post_onset_recent5y_diff_sic2": gap(eq[(eq.dst_prior_any == 0) & (eq.pre_onset == 0) & (eq.same_sic2 == 0)], "out_onset_3y", "exposed5y")}
    if os.path.exists("data/sec_address_by_cik.csv"):
        adr = pd.read_csv("data/sec_address_by_cik.csv", dtype=str).fillna("").assign(cik=lambda d_: norm_cik(d_["cik"])).set_index("cik")["biz_state"].str.strip()
        eq["st_src"] = eq["src"].map(adr).fillna(""); eq["st_dst"] = eq["dst"].map(adr).fillna("")
        eq["same_state"] = ((eq.st_src == eq.st_dst) & (eq.st_dst != "")).astype(int)
        K1["common_shock_controls"].update({"share_same_state": round(float(eq["same_state"].mean()), 4),
                                            "exposed_share_same_state": round(float(eq.loc[eq.same_state == 1, "exposed"].mean()), 4), "exposed_share_diff_state": round(float(eq.loc[eq.same_state == 0, "exposed"].mean()), 4),
                                            "did_file_cohortFE_sameSIC_sameState": did_fe(eq, "pre_file_3y", "out_file_3y", ("same_sic2", "same_state")), "did_onset_cohortFE_sameSIC_sameState": did_fe(eq, "pre_onset_3y", "out_onset_3y", ("same_sic2", "same_state")),
                                            "did_file_cohortFE_dstIndustryXpost_sameState": did_fe_dstind(eq, "pre_file_3y", "out_file_3y", ("same_state",)), "did_onset_cohortFE_dstIndustryXpost_sameState": did_fe_dstind(eq, "pre_onset_3y", "out_onset_3y", ("same_state",)),
                                            "clean_post_onset_recent5y_same_state": gap(eq[(eq.dst_prior_any == 0) & (eq.pre_onset == 0) & (eq.same_state == 1)], "out_onset_3y", "exposed5y"),
                                            "clean_post_onset_recent5y_diff_state": gap(eq[(eq.dst_prior_any == 0) & (eq.pre_onset == 0) & (eq.same_state == 0)], "out_onset_3y", "exposed5y")})
    log(f"common shocks: same SIC2 {K1['common_shock_controls']['share_same_sic2']:.1%}; DiD file w/ sameSIC {K1['common_shock_controls']['did_file_cohortFE_sameSIC']['diff_in_gaps_pp']:+.2f}; onset w/ dst industry x post {K1['common_shock_controls']['did_onset_cohortFE_dstIndustryXpost']['diff_in_gaps_pp']:+.2f}")
# audit-committee assignment at the appointing event (committee free text of the destination's appointment; recorded for a minority of appointments)
K1["audit_committee_split"] = {"share_appointments_with_committee_text": round(float((eq.dst_ac >= 0).mean()), 4), "share_appointments_audit_committee": round(float((eq.dst_ac == 1).mean()), 4),
                               "share_of_recorded_naming_audit": round(float((eq.loc[eq.dst_ac >= 0, "dst_ac"] == 1).mean()), 4) if (eq.dst_ac >= 0).any() else None}
for lab_, m_ in (("audit_committee_named", eq.dst_ac == 1), ("committees_named_not_audit", eq.dst_ac == 0), ("no_committee_recorded", eq.dst_ac == -1)):
    s_ = eq[m_]
    blk_ = {"n_rows": int(len(s_)), "n_appointments": int(s_.groupby(["director", "dst", "birth"]).ngroups), "exposed_share": round(float(s_.exposed.mean()), 4), "exposed5y_share": round(float(s_.exposed5y.mean()), 4)}
    for ex_ in ("exposed", "exposed5y", "exposed_served"):
        if s_[ex_].sum() < 20 or (s_[ex_] == 0).sum() < 20: continue
        blk_[ex_] = {"pre_file": gap(s_, "pre_file_3y", ex_), "post_file": gap(s_, "out_file_3y", ex_), "did_file": did(s_, "pre_file_3y", "out_file_3y", ex_), "did_file_cohortFE": did_fe(s_, "pre_file_3y", "out_file_3y", exp=ex_),
                     "pre_onset": gap(s_, "pre_onset_3y", ex_), "post_onset": gap(s_, "out_onset_3y", ex_), "did_onset": did(s_, "pre_onset_3y", "out_onset_3y", ex_), "did_onset_cohortFE": did_fe(s_, "pre_onset_3y", "out_onset_3y", exp=ex_),
                     "never_restated_post_onset": gap(s_[s_.dst_prior_any == 0], "out_onset_3y", ex_) if (s_[s_.dst_prior_any == 0][ex_].sum() >= 20) else None}
    K1["audit_committee_split"][lab_] = blk_
# triple interaction: does the change in the gap differ when the appointee was named to the audit committee? stacked regression with exposure, post,
# AC, all pairwise products, the triple product, and formation-year (x post) effects; two-way (origin, destination) clustering. Comparator = committees named without audit.
def did_fe_triple(df, pre, post, exp, mod):
    st = pd.concat([df.assign(y=df[pre], post=0), df.assign(y=df[post], post=1)], ignore_index=True)
    cohort = pd.get_dummies(st["by"].astype(int), prefix="c", drop_first=True).astype(float)
    e_, p_, m_ = st[exp].values.astype(float), st["post"].values.astype(float), st[mod].values.astype(float)
    X = np.column_stack([np.ones(len(st)), e_, p_, e_ * p_, m_, m_ * p_, e_ * m_, e_ * m_ * p_, cohort.values, cohort.values * p_[:, None]])
    b_, se_ = twoway_ols(st["y"].values, X, st["src"].values, st["dst"].values)
    return {"change_base_pp": round(100 * b_[3], 3), "se_base_pp": round(100 * se_[3], 3), "triple_pp": round(100 * b_[7], 3), "se_triple_pp": round(100 * se_[7], 3), "t_triple": round(float(b_[7] / se_[7]), 2) if se_[7] > 0 else None,
            "change_in_modifier_group_pp": round(100 * (b_[3] + b_[7]), 3), "n_rows": int(len(df)), "n_modifier_rows": int(df[mod].sum())}
if (eq.dst_ac == 1).sum() >= 100:
    _e2 = eq[eq.dst_ac >= 0].copy(); _e2["ac1"] = (_e2.dst_ac == 1).astype(int)
    for ex_ in ("exposed", "exposed5y", "exposed_served"):
        K1["audit_committee_split"][f"triple_{ex_}_onset_vs_committees_not_audit"] = did_fe_triple(_e2, "pre_onset_3y", "out_onset_3y", ex_, "ac1")
        K1["audit_committee_split"][f"triple_{ex_}_file_vs_committees_not_audit"] = did_fe_triple(_e2, "pre_file_3y", "out_file_3y", ex_, "ac1")
    _e3 = eq.copy(); _e3["ac1"] = (_e3.dst_ac == 1).astype(int)   # comparator = all other appointments (committees not audit + none recorded)
    for ex_ in ("exposed", "exposed5y", "exposed_served"):
        K1["audit_committee_split"][f"triple_{ex_}_onset_vs_all_other"] = did_fe_triple(_e3, "pre_onset_3y", "out_onset_3y", ex_, "ac1")
log(f"audit committee: text for {K1['audit_committee_split']['share_appointments_with_committee_text']:.1%} of pairs, audit named {K1['audit_committee_split']['share_appointments_audit_committee']:.1%}")
# historical industry: Compustat sich at (or before) the formation year for Compustat-linked pairs, as a check on the current EDGAR code
if os.path.exists("data/comp_funda_sich_2004_2024.csv") and "same_sic2" in eq:
    _fs = pd.read_csv("data/comp_funda_sich_2004_2024.csv", dtype={"gvkey": str}); _fs = _fs.dropna(subset=["sich", "fyear"]); _fs["gvkey"] = _fs["gvkey"].str.zfill(6); _fs["fyear"] = _fs["fyear"].astype(int)
    _cwh = pd.read_csv("data/cik_gvkey_crosswalk.csv", dtype=str); _cwh["cik"] = norm_cik(_cwh["company_fkey"]); _cwh["gvkey"] = _cwh["gvkey"].astype(str).str.zfill(6); _g2c = _cwh.drop_duplicates("cik").set_index("cik")["gvkey"].to_dict()
    _sich = {(g_, y_): int(s_) // 100 for g_, y_, s_ in zip(_fs.gvkey, _fs.fyear, _fs.sich)}
    def hist_sic2(cik, year):
        g_ = _g2c.get(cik)
        if g_ is None: return ""
        for y_ in (year, year - 1, year - 2, year - 3, year + 1):
            v = _sich.get((g_, y_))
            if v is not None: return f"{v:02d}"
        return ""
    eq["hsic2_src"] = [hist_sic2(c_, int(y_)) for c_, y_ in zip(eq.src, eq.by)]; eq["hsic2_dst"] = [hist_sic2(c_, int(y_)) for c_, y_ in zip(eq.dst, eq.by)]
    _hb = eq[(eq.hsic2_src != "") & (eq.hsic2_dst != "")].copy(); _hb["same_sic2_hist"] = (_hb.hsic2_src == _hb.hsic2_dst).astype(int)
    _ov = _hb[_hb.sic2_dst != ""]
    def did_fe_dstind_h(df, pre, post, extra=(), exp="exposed"):
        d2 = df.copy(); d2["sic2_dst"] = d2["hsic2_dst"]; return did_fe_dstind(d2, pre, post, extra, exp)
    K1["common_shock_controls"]["historical_sic"] = {"n_rows_both_available": int(len(_hb)), "share_of_pairs": round(float(len(_hb) / len(eq)), 4), "share_same_sic2_hist": round(float(_hb.same_sic2_hist.mean()), 4),
        "agreement_with_current_same_sic2": round(float((_ov.same_sic2_hist == _ov.same_sic2).mean()), 4) if len(_ov) else None,
        "share_dst_2digit_differs_current_vs_historical": round(float((_ov.hsic2_dst != _ov.sic2_dst).mean()), 4) if len(_ov) else None,
        "did_file_cohortFE_sameSIChist": did_fe(_hb, "pre_file_3y", "out_file_3y", ("same_sic2_hist",)), "did_onset_cohortFE_sameSIChist": did_fe(_hb, "pre_onset_3y", "out_onset_3y", ("same_sic2_hist",)),
        "did_onset_cohortFE_sameSIChist_recent": did_fe(_hb, "pre_onset_3y", "out_onset_3y", ("same_sic2_hist",), exp="exposed5y"),
        "did_file_cohortFE_dstHistIndustryXpost": did_fe_dstind_h(_hb, "pre_file_3y", "out_file_3y"), "did_onset_cohortFE_dstHistIndustryXpost": did_fe_dstind_h(_hb, "pre_onset_3y", "out_onset_3y"),
        "did_onset_cohortFE_dstHistIndustryXpost_recent": did_fe_dstind_h(_hb, "pre_onset_3y", "out_onset_3y", exp="exposed5y"),
        "same_subsample_current_sic_did_onset": did_fe(_ov, "pre_onset_3y", "out_onset_3y", ("same_sic2",)),
        "clean_post_onset_recent5y_same_sic2_hist": gap(_hb[(_hb.dst_prior_any == 0) & (_hb.pre_onset == 0) & (_hb.same_sic2_hist == 1)], "out_onset_3y", "exposed5y"),
        "clean_post_onset_recent5y_diff_sic2_hist": gap(_hb[(_hb.dst_prior_any == 0) & (_hb.pre_onset == 0) & (_hb.same_sic2_hist == 0)], "out_onset_3y", "exposed5y")}
    log(f"historical SIC: {len(_hb):,} pairs; same-industry share {K1['common_shock_controls']['historical_sic']['share_same_sic2_hist']:.1%}; onset DiD w/ dst hist industry x post {K1['common_shock_controls']['historical_sic']['did_onset_cohortFE_dstHistIndustryXpost']['diff_in_gaps_pp']:+.2f}")
# receptivity heterogeneity: where does the recent-exposure residue among history-free destinations sit? by destination size tercile (Compustat) and feed age
clh = eq[(eq.dst_prior_any == 0) & (eq.pre_onset == 0)].copy()
clh = clh.merge(_sz.rename(columns={"cik": "dst", "year": "by", "size": "dst_size"}), on=["dst", "by"], how="left")
K1["receptivity_heterogeneity"] = {"n_history_free": int(len(clh)), "n_with_size": int(clh.dst_size.notna().sum())}
szq = clh.loc[clh.dst_size.notna(), "dst_size"].quantile([1 / 3, 2 / 3]).values
for lab_, m_ in (("size_bottom_tercile", clh.dst_size <= szq[0]), ("size_middle_tercile", (clh.dst_size > szq[0]) & (clh.dst_size <= szq[1])), ("size_top_tercile", clh.dst_size > szq[1]),
                 ("dst_in_feed_lt3y", clh.dst_feed_age < 3), ("dst_in_feed_ge3y", clh.dst_feed_age >= 3)):
    sub_ = clh[m_]
    K1["receptivity_heterogeneity"][lab_] = {"n_rows": int(len(sub_)), "post_onset_recent5y": gap(sub_, "out_onset_3y", "exposed5y"), "post_onset_any": gap(sub_, "out_onset_3y"), "post_file_recent5y": gap(sub_, "out_file_3y", "exposed5y")}
K1["receptivity_heterogeneity"]["size_tercile_cutpoints_log_assets"] = [round(float(szq[0]), 2), round(float(szq[1]), 2)]
# the same size terciles (same cutpoints) in the never-filed (ex ante) sample and in all pairs: recent-exposure pre/post onset gaps and cohort-adjusted changes
_eqs = eq.merge(_sz.rename(columns={"cik": "dst", "year": "by", "size": "dst_size"}), on=["dst", "by"], how="left")
K1["receptivity_heterogeneity"]["by_size_tercile_prepost"] = {}
for samp_, df_ in (("never_filed", _eqs[_eqs.dst_prior_any == 0]), ("all_pairs", _eqs)):
    for lab_, m_ in (("size_bottom_tercile", df_.dst_size <= szq[0]), ("size_middle_tercile", (df_.dst_size > szq[0]) & (df_.dst_size <= szq[1])), ("size_top_tercile", df_.dst_size > szq[1])):
        sub_ = df_[m_ & df_.dst_size.notna()]
        K1["receptivity_heterogeneity"]["by_size_tercile_prepost"][f"{samp_}__{lab_}"] = {
            "n_rows": int(len(sub_)), "n_exposed5y": int(sub_.exposed5y.sum()),
            "pre_onset_recent5y": gap(sub_, "pre_onset_3y", "exposed5y"), "post_onset_recent5y": gap(sub_, "out_onset_3y", "exposed5y"),
            "did_onset_recent5y": did(sub_, "pre_onset_3y", "out_onset_3y", "exposed5y"), "did_onset_recent5y_cohortFE": did_fe(sub_, "pre_onset_3y", "out_onset_3y", exp="exposed5y"),
            "post_onset_recent5y_yearFE": gap_yearfe(sub_, "out_onset_3y", "exposed5y")}
# destination's prior exposure: number of earlier exposed inbound ties at the destination (formed before this one), and the first-exposed-tie subsample
_ord = eq.sort_values(["dst", "birth"]).copy()
_ord["prior_exposed_ties"] = _ord.groupby("dst")["exposed"].cumsum() - _ord["exposed"]
_ord["prior_inbound_ties"] = _ord.groupby("dst").cumcount()
eq = eq.join(_ord[["prior_exposed_ties", "prior_inbound_ties"]])
eq["log1_prior_exposed"] = np.log1p(eq["prior_exposed_ties"]); eq["log1_prior_inbound"] = np.log1p(eq["prior_inbound_ties"])
fe_ = eq[eq.prior_exposed_ties == 0]
K1["prior_exposure_stock"] = {"share_pairs_with_prior_exposed_tie": round(float((eq.prior_exposed_ties > 0).mean()), 4), "mean_prior_exposed_ties": round(float(eq.prior_exposed_ties.mean()), 3),
                              "did_file_cohortFE_priorstock": did_fe(eq, "pre_file_3y", "out_file_3y", ("log1_prior_exposed", "log1_prior_inbound")),
                              "did_onset_cohortFE_priorstock": did_fe(eq, "pre_onset_3y", "out_onset_3y", ("log1_prior_exposed", "log1_prior_inbound")),
                              "first_exposed_tie": {"n_rows": int(len(fe_)), "pre_file": gap(fe_, "pre_file_3y"), "post_file": gap(fe_, "out_file_3y"), "did_file": did(fe_, "pre_file_3y", "out_file_3y"),
                                                    "pre_onset": gap(fe_, "pre_onset_3y"), "post_onset": gap(fe_, "out_onset_3y"), "did_onset": did(fe_, "pre_onset_3y", "out_onset_3y"),
                                                    "clean_post_onset": gap(fe_[(fe_.dst_prior_any == 0) & (fe_.pre_onset == 0)], "out_onset_3y"), "clean_post_onset_recent5y": gap(fe_[(fe_.dst_prior_any == 0) & (fe_.pre_onset == 0)], "out_onset_3y", "exposed5y")}}
log(f"prior exposure: {K1['prior_exposure_stock']['share_pairs_with_prior_exposed_tie']:.1%} of pairs have an earlier exposed tie; first-exposed-tie DiD file {K1['prior_exposure_stock']['first_exposed_tie']['did_file']['diff_in_gaps_pp']:+.2f}, onset {K1['prior_exposure_stock']['first_exposed_tie']['did_onset']['diff_in_gaps_pp']:+.2f}")
# warrant wave (SEC Staff Statement of 12 April 2021 on SPAC warrants): 2021 filings by blank-check registrants
if os.path.exists("data/sec_sic_by_cik.csv"):
    r21 = rest[(rest.file_date >= "2021-01-01") & (rest.file_date <= "2021-12-31")].copy()
    nm_map = sec1.set_index("cik")["name"].str.lower()
    r21["bc"] = (r21["cfkey"].map(sec1.set_index("cik")["sic"].str.strip()) == "6770") | r21["cfkey"].map(nm_map).fillna("").str.contains("acquisition")
    K1["warrant_wave_2021"] = {"filings_2021": int(len(r21)), "filings_2021_blank_check": int(r21["bc"].sum()), "share_blank_check": round(float(r21["bc"].mean()), 4),
                               "blank_check_filings_2021_04_12_to_06_30": int(((r21.file_date >= "2021-04-12") & (r21.file_date <= "2021-06-30") & r21["bc"]).sum()),
                               "example_registrant": "dMY Technology Group, Inc. II", "example_filing_date": "2021-04-27"}
# worked example for the definitions (firm names from EDGAR; dates from the Audit Analytics record; no director name)
_ex = base[(base.src_is_origin == True) & (base.exposed5y == 1) & (base.dst_prior_any == 0) & (base.pre_onset == 0) & (base.out_onset_3y == 1)]
_ex_src = [c for c in sec1["cik"] if "mufg americas" in str(nm_map.get(c, ""))]; _ex_dst = [c for c in sec1["cik"] if str(nm_map.get(c, "")).startswith("stride, inc")]
_row = _ex[_ex.src.isin(_ex_src) & _ex.dst.isin(_ex_dst)]
if len(_row):
    rw = _row.iloc[0]; of = sorted(f for f in rf.get(rw.src, ()) if f < rw.birth); do = sorted((b_, f_) for b_, f_ in ro.get(rw.dst, ()) if b_ >= rw.birth)
    K1["worked_example"] = {"origin": "MUFG Americas Holdings Corp", "destination": "Stride, Inc.", "tie_formed": str(rw.birth.date()), "origin_last_filing_before_tie": str(of[-1].date()),
                            "origin_filings_before_tie": len(of), "destination_first_onset_after_tie": str(do[0][0].date()), "destination_first_filing_after_tie": str(do[0][1].date()),
                            "years_from_origin_filing_to_tie": round((rw.birth - of[-1]).days / 365.25, 2), "months_from_tie_to_onset": round((do[0][0] - rw.birth).days / 30.44, 1)}
    log(f"worked example: {K1['worked_example']}")
# strict clean comparison: clean origins that also do not restate within three years after the tie (common-shock contamination of the clean group)
eq["src_files_3y_after"] = [int(any_in(rf.get(s_, ()), b_, min(b_ + pd.Timedelta(days=3 * 365), HORIZON))) for s_, b_ in zip(eq.src, eq.birth)]
sc_ = eq[(eq.exposed == 1) | (eq.src_files_3y_after == 0)]
K1["strict_clean_origins"] = {"share_clean_origins_filing_within_3y_after": round(float(eq.loc[eq.exposed == 0, "src_files_3y_after"].mean()), 4), "n_rows": int(len(sc_)),
                              "pre_file": gap(sc_, "pre_file_3y"), "post_file": gap(sc_, "out_file_3y"), "did_file": did(sc_, "pre_file_3y", "out_file_3y"),
                              "pre_onset": gap(sc_, "pre_onset_3y"), "post_onset": gap(sc_, "out_onset_3y"), "did_onset": did(sc_, "pre_onset_3y", "out_onset_3y"),
                              "recent_pre_onset": gap(sc_, "pre_onset_3y", "exposed5y"), "recent_post_onset": gap(sc_, "out_onset_3y", "exposed5y"), "recent_did_onset": did(sc_, "pre_onset_3y", "out_onset_3y", "exposed5y"),
                              "clean_post_onset_recent5y": gap(sc_[(sc_.dst_prior_any == 0) & (sc_.pre_onset == 0)], "out_onset_3y", "exposed5y"),
                              "did_onset_cohortFE": did_fe(sc_, "pre_onset_3y", "out_onset_3y"), "recent_did_onset_cohortFE": did_fe(sc_, "pre_onset_3y", "out_onset_3y", exp="exposed5y")}
log(f"strict clean: {K1['strict_clean_origins']['share_clean_origins_filing_within_3y_after']:.1%} of clean origins file within 3y after; DiD file {K1['strict_clean_origins']['did_file']['diff_in_gaps_pp']:+.2f}, onset {K1['strict_clean_origins']['did_onset']['diff_in_gaps_pp']:+.2f}")
# both directions pooled (as in the diffusion literature's undirected designs) — comparison only
eqa = base_all[base_all.by <= 2022]
K1["both_directions_pooled"] = {"n_rows": int(len(eqa)), "life_plus1y_file": gap(base_all, "out_file"), "life_plus1y_onset": gap(base_all, "out_onset"),
                                "pre_file": gap(eqa, "pre_file_3y"), "post_file": gap(eqa, "out_file_3y"), "did_file": did(eqa, "pre_file_3y", "out_file_3y"),
                                "pre_onset": gap(eqa, "pre_onset_3y"), "post_onset": gap(eqa, "out_onset_3y"), "did_onset": did(eqa, "pre_onset_3y", "out_onset_3y"),
                                "never_restated_post_file": gap(eqa[eqa.dst_prior_any == 0], "out_file_3y"), "never_restated_post_onset": gap(eqa[eqa.dst_prior_any == 0], "out_onset_3y"),
                                "reverse_direction_post_file": gap(eqa[eqa.src_is_origin == False], "out_file_3y"), "reverse_direction_pre_file": gap(eqa[eqa.src_is_origin == False], "pre_file_3y")}
# onset outcomes restricted to cohorts formed by 2019 (filing lag truncation near the horizon)
e19 = eq[eq.by <= 2019]
K1["onset_formed_by_2019"] = {"n_rows": int(len(e19)), "pre_onset": gap(e19, "pre_onset_3y"), "post_onset": gap(e19, "out_onset_3y"), "did_onset": did(e19, "pre_onset_3y", "out_onset_3y"),
                              "did_onset_cohortFE": did_fe(e19, "pre_onset_3y", "out_onset_3y"), "recent_pre_onset": gap(e19, "pre_onset_3y", "exposed5y"), "recent_post_onset": gap(e19, "out_onset_3y", "exposed5y"),
                              "recent_did_onset": did(e19, "pre_onset_3y", "out_onset_3y", "exposed5y"), "recent_did_onset_cohortFE": did_fe(e19, "pre_onset_3y", "out_onset_3y", exp="exposed5y")}
# attrition: does the destination still record board events after the three-year post-formation window? (differential exit would mechanically narrow the post gap)
eq["dst_active_after_window"] = [int(LAST_EVENT.get(d_, pd.Timestamp.min) >= b_ + pd.Timedelta(days=3 * 365)) for d_, b_ in zip(eq.dst, eq.birth)]
_aw = eq[eq.dst_active_after_window == 1]
K1["attrition_after_window"] = {"share_dst_active_after_window": round(float(eq.dst_active_after_window.mean()), 4),
                                "share_exposed": round(float(eq.loc[eq.exposed == 1, "dst_active_after_window"].mean()), 4), "share_clean": round(float(eq.loc[eq.exposed == 0, "dst_active_after_window"].mean()), 4),
                                "share_recent_exposed": round(float(eq.loc[eq.exposed5y == 1, "dst_active_after_window"].mean()), 4), "share_not_recent": round(float(eq.loc[eq.exposed5y == 0, "dst_active_after_window"].mean()), 4),
                                "n_rows": int(len(_aw)), "pre_file": gap(_aw, "pre_file_3y"), "post_file": gap(_aw, "out_file_3y"), "did_file": did(_aw, "pre_file_3y", "out_file_3y"), "did_file_cohortFE": did_fe(_aw, "pre_file_3y", "out_file_3y"),
                                "pre_onset": gap(_aw, "pre_onset_3y"), "post_onset": gap(_aw, "out_onset_3y"), "did_onset": did(_aw, "pre_onset_3y", "out_onset_3y"), "did_onset_cohortFE": did_fe(_aw, "pre_onset_3y", "out_onset_3y"),
                                "recent_did_onset": did(_aw, "pre_onset_3y", "out_onset_3y", "exposed5y"), "recent_did_onset_cohortFE": did_fe(_aw, "pre_onset_3y", "out_onset_3y", exp="exposed5y"),
                                "clean_post_onset": gap(_aw[(_aw.dst_prior_any == 0) & (_aw.pre_onset == 0)], "out_onset_3y"), "clean_post_onset_recent5y": gap(_aw[(_aw.dst_prior_any == 0) & (_aw.pre_onset == 0)], "out_onset_3y", "exposed5y"),
                                "never_restated_post_onset": gap(_aw[_aw.dst_prior_any == 0], "out_onset_3y")}
log(f"attrition: dst active after window {K1['attrition_after_window']['share_dst_active_after_window']:.1%} (exposed {K1['attrition_after_window']['share_exposed']:.1%}, clean {K1['attrition_after_window']['share_clean']:.1%}); DiD file {K1['attrition_after_window']['did_file']['diff_in_gaps_pp']:+.2f}, onset {K1['attrition_after_window']['did_onset']['diff_in_gaps_pp']:+.2f}")
K1["equal_window_3y"]["did_file_severe_source"] = did(eq, "pre_file_3y", "out_file_3y", "exposed_severe")
# destination history by exposure; never-restated destinations
K1["destination_prior_restatement_share"] = {"exposed": round(float(eq.loc[eq.exposed == 1, "dst_prior_any"].mean()), 4), "clean": round(float(eq.loc[eq.exposed == 0, "dst_prior_any"].mean()), 4)}
nv = eq[eq.dst_prior_any == 0]
K1["never_restated_destinations"] = {"n_rows": int(len(nv)), "share_of_rows": round(len(nv) / len(eq), 4), "post_file_3y": gap(nv, "out_file_3y"), "post_onset_3y": gap(nv, "out_onset_3y")}
K1["never_restated_destinations"]["pre_onset_3y"] = gap(eq[eq.dst_prior_any == 0], "pre_onset_3y")
# Panel A (never filed): pre/post change of the onset gap for each exposure definition, with SE
nvA = eq[eq.dst_prior_any == 0]
K1["never_restated_destinations"]["did_onset"] = did(nvA, "pre_onset_3y", "out_onset_3y")
K1["exposure_recent_5y"]["never_restated_did_onset"] = did(nvA, "pre_onset_3y", "out_onset_3y", "exposed5y")
K1["exposure_recent_5y"]["never_restated_did_onset_cohortFE"] = did_fe(nvA, "pre_onset_3y", "out_onset_3y", exp="exposed5y")
K1["never_restated_destinations"]["did_onset_cohortFE"] = did_fe(nvA, "pre_onset_3y", "out_onset_3y")
K1["exposure_served_during_misstatement"]["never_restated_did_onset"] = did(nvA, "pre_onset_3y", "out_onset_3y", "exposed_served")
# by cohort: pre vs post 3y
K1["equal_window_by_cohort"] = {}
for lab, yrs in (("2004-2007", range(2004, 2008)), ("2008-2012", range(2008, 2013)), ("2013-2017", range(2013, 2018)), ("2018-2022", range(2018, 2023))):
    sub = eq[eq.by.isin(list(yrs))]
    K1["equal_window_by_cohort"][lab] = {"pre_file_pp": gap(sub, "pre_file_3y")["diff_pp"], "post_file_pp": gap(sub, "out_file_3y")["diff_pp"], "did": did(sub, "pre_file_3y", "out_file_3y"), "exposed_share": round(float(sub.exposed.mean()), 4), "n_rows": int(len(sub))}
K1["left_censoring_robustness_formed_2008plus"] = {"file": gap(base[base.by >= 2008], "out_file"), "onset": gap(base[base.by >= 2008], "out_onset")}
# destination-firm level, exposure-adjusted (as module J)
base["expo_years"] = ((base["death"] - base["birth"]).dt.days / 365.25)
firm = base.groupby("dst").agg(ever_exposed=("exposed", "max"), n_in=("exposed", "size"), years_in=("expo_years", "sum"),
                               first_year=("by", "min"), ever_file=("out_file", "max"), ever_onset=("out_onset", "max")).reset_index()
Xf = sm.add_constant(pd.get_dummies(firm[["ever_exposed", "n_in", "years_in", "first_year"]].assign(n_in=np.log(firm["n_in"]), years_in=np.log1p(firm["years_in"])), columns=["first_year"], drop_first=True).astype(float))
K1["destination_firm_adjusted"] = {}
for lab, y in (("file", "ever_file"), ("onset", "ever_onset")):
    m = sm.OLS(firm[y].astype(float), Xf).fit(cov_type="HC1")
    raw = firm.loc[firm.ever_exposed == 1, y].mean() - firm.loc[firm.ever_exposed == 0, y].mean()
    _g1 = firm.loc[firm.ever_exposed == 1, y].astype(float); _g0 = firm.loc[firm.ever_exposed == 0, y].astype(float)
    K1["destination_firm_adjusted"][lab] = {"raw_gap_pp": round(100 * raw, 2), "raw_se_pp": round(100 * float(np.sqrt(_g1.var(ddof=1) / len(_g1) + _g0.var(ddof=1) / len(_g0))), 2), "adjusted_gap_pp": round(100 * m.params["ever_exposed"], 2), "se_pp": round(100 * m.bse["ever_exposed"], 2), "n_firms": int(len(firm))}
OUT["K1_pairs"] = K1
ew = K1["equal_window_3y"]
log(f"K1 life+1y gaps: file {K1['outcome_file_in_window']['diff_pp']:+.2f}, onset {K1['outcome_onset_in_window']['diff_pp']:+.2f}; equal 3y: pre {ew['pre_file']['diff_pp']:+.2f} post {ew['post_file']['diff_pp']:+.2f} DiD {ew['did_file']['diff_in_gaps_pp']:+.2f} (t={ew['did_file']['t']}); onset DiD {ew['did_onset']['diff_in_gaps_pp']:+.2f}; severe DiD {ew['did_severe']['diff_in_gaps_pp']:+.2f}; never-restated dst post gap {K1['never_restated_destinations']['post_file_3y']['diff_pp']:+.2f}")

# ======================================================================================
# K4a. Yearly network: degree, approximate betweenness, Louvain communities (rebuilt edges)
# ======================================================================================
net_rows = []; comm = {}; comm_stats = {}
for y in range(FORM_LO, FORM_HI + 1):
    ys, ye_ = pd.Timestamp(f"{y}-01-01"), pd.Timestamp(f"{y}-12-31")
    alive = edf[(edf.edge_birth <= ye_) & (edf.edge_death >= ys)]
    G = nx.Graph(); G.add_edges_from(zip(alive["company_a"], alive["company_b"]))
    if G.number_of_nodes() == 0: continue
    deg = dict(G.degree()); btw = nx.betweenness_centrality(G, k=min(1000, G.number_of_nodes()), normalized=True, seed=42)
    parts = louvain_communities(G, seed=42); part = {n: i for i, com in enumerate(parts) for n in com}
    mod = modularity(G, parts); cross = sum(1 for u, v in G.edges() if part[u] != part[v]) / max(1, G.number_of_edges())
    lcc = max(nx.connected_components(G), key=len)
    comm_stats[y] = {"nodes": G.number_of_nodes(), "edges": G.number_of_edges(), "n_communities": len(parts), "modularity": round(float(mod), 4),
                     "cross_edge_ratio": round(float(cross), 4), "lcc_share": round(len(lcc) / G.number_of_nodes(), 4)}
    for n in G.nodes(): net_rows.append((n, y, deg[n], btw[n], part[n])); comm[(n, y)] = part[n]
pn = pd.DataFrame(net_rows, columns=["company", "year", "degree", "betweenness", "community"]); pn.to_csv(f"data/panel_network_k{SUFFIX}.csv", index=False)
OUT["K4_network_by_year"] = comm_stats
log(f"network: 2004 nodes {comm_stats[2004]['nodes']:,} -> 2023 nodes {comm_stats[2023]['nodes']:,}; modularity {comm_stats[2004]['modularity']}..{comm_stats[2023]['modularity']}")

# K4b. H5: within vs cross community, exposed ties, correct units
base["ca"] = [comm.get((s, y)) for s, y in zip(base["src"], base["by"])]
base["cb"] = [comm.get((d, y)) for d, y in zip(base["dst"], base["by"])]
hb = base[base.ca.notna() & base.cb.notna()].copy(); hb["cross"] = (hb.ca != hb.cb).astype(int)
ex = hb[hb.exposed == 1]
K4 = {"n_rows_with_community": int(len(hb)), "n_exposed_rows": int(len(ex)), "n_exposed_unique_pairs": int(ex.pid.nunique()),
      "within_exposed_rate": round(float(ex.loc[ex.cross == 0, "out_file"].mean()), 4), "cross_exposed_rate": round(float(ex.loc[ex.cross == 1, "out_file"].mean()), 4),
      "n_within_exposed": int((ex.cross == 0).sum()), "n_cross_exposed": int((ex.cross == 1).sum())}
for lab, out in (("file", "out_file"), ("onset", "out_onset")):
    X = np.column_stack([np.ones(len(ex)), ex["cross"]]); b, se = twoway_ols(ex[out].values, X, ex["src"].values, ex["dst"].values)
    K4[f"cross_minus_within_exposed_{lab}"] = {"diff_pp": round(100 * b[1], 3), "se_pp": round(100 * se[1], 3), "t": round(float(b[1] / se[1]), 2)}
    X = np.column_stack([np.ones(len(hb)), hb["exposed"], hb["cross"], hb["exposed"] * hb["cross"]]); b, se = twoway_ols(hb[out].values, X, hb["src"].values, hb["dst"].values)
    K4[f"interaction_exposed_x_cross_{lab}"] = {"exposed_pp": round(100 * b[1], 3), "cross_pp": round(100 * b[2], 3), "exposed_x_cross_pp": round(100 * b[3], 3), "se_interaction_pp": round(100 * se[3], 3), "t_interaction": round(float(b[3] / se[3]), 2)}
for lab, out, sub, tag in (("file", "out_file_3y", hb[hb.by <= 2022], "3y"), ("onset", "out_onset_3y", hb[hb.by <= 2022], "3y"), ("file", "out_file_3y", hb[(hb.by <= 2022) & (hb.by >= 2012)], "3y_2012plus"), ("onset", "out_onset_3y", hb[(hb.by <= 2022) & (hb.by >= 2012)], "3y_2012plus")):
    ex_ = sub[sub.exposed == 1]
    X = np.column_stack([np.ones(len(ex_)), ex_["cross"]]); b, se = twoway_ols(ex_[out].values, X, ex_["src"].values, ex_["dst"].values)
    K4[f"cross_minus_within_exposed_{lab}_{tag}"] = {"diff_pp": round(100 * b[1], 3), "se_pp": round(100 * se[1], 3), "t": round(float(b[1] / se[1]), 2), "n_exposed": int(len(ex_)), "share_cross_exposed": round(float(ex_["cross"].mean()), 4)}
    X = np.column_stack([np.ones(len(sub)), sub["exposed"], sub["cross"], sub["exposed"] * sub["cross"]]); b, se = twoway_ols(sub[out].values, X, sub["src"].values, sub["dst"].values)
    K4[f"interaction_exposed_x_cross_{lab}_{tag}"] = {"exposed_pp": round(100 * b[1], 3), "cross_pp": round(100 * b[2], 3), "exposed_x_cross_pp": round(100 * b[3], 3), "se_interaction_pp": round(100 * se[3], 3), "t_interaction": round(float(b[3] / se[3]), 2), "n_rows": int(len(sub))}
OUT["K4_h5"] = K4
log(f"H5 exposed rows {len(ex):,}: within {K4['within_exposed_rate']} cross {K4['cross_exposed_rate']} diff {K4['cross_minus_within_exposed_file']['diff_pp']:+.2f} (SE {K4['cross_minus_within_exposed_file']['se_pp']})")

# ======================================================================================
# K3. Cohort attenuation, fixed 3-year window, edges formed 2004-2022
# ======================================================================================
K3 = {}
for lab, yrs in (("2004-2007", range(2004, 2008)), ("2008-2012", range(2008, 2013)), ("2013-2017", range(2013, 2018)), ("2018-2022", range(2018, 2023))):
    sub = base[base.by.isin(list(yrs))]
    K3[lab] = {"file_3y": gap(sub, "out_file_3y"), "onset_3y": gap(sub, "out_onset_3y"), "pre_onset_3y": gap(sub[sub.by <= 2022], "pre_onset_3y"), "exposed_share": round(float(sub.exposed.mean()), 4), "n_rows": int(len(sub))}
# year-by-year (file, 3y) for a trend regression
tr = base[base.by <= 2022].copy(); tr["yc"] = tr["by"] - 2013
X = np.column_stack([np.ones(len(tr)), tr["exposed"], tr["yc"], tr["exposed"] * tr["yc"]]); b, se = twoway_ols(tr["out_file_3y"].values, X, tr["src"].values, tr["dst"].values)
K3["linear_trend_in_gap_per_year_file_3y"] = {"pp_per_year": round(100 * b[3], 3), "se_pp": round(100 * se[3], 3), "t": round(float(b[3] / se[3]), 2)}
OUT["K3_cohorts_fixed_window"] = K3
log("K3 cohorts (file, 3y): " + ", ".join(f"{k}: {v['file_3y']['diff_pp']:+.1f} (exp share {v['exposed_share']:.2f})" for k, v in K3.items() if "file_3y" in v))

# ======================================================================================
# K2. Firm-year panel with lagged / start-of-year exposure
# ======================================================================================
cw = pd.read_csv("data/cik_gvkey_crosswalk.csv", dtype=str); cw["cik"] = norm_cik(cw["company_fkey"]); cw["gvkey"] = cw["gvkey"].astype(str).str.zfill(6)
cw = cw.drop_duplicates("cik")[["cik", "gvkey"]]
comp = pd.read_csv("data/compustat_financials.csv"); comp["gvkey"] = comp["gvkey"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(6)
comp = comp[comp["assets"] > 0].copy(); comp["size"] = np.log(comp["assets"]); comp["leverage"] = comp["liabilities"] / comp["assets"]; comp["roa"] = comp["net_income"] / comp["assets"]
comp = comp[["gvkey", "year", "size", "leverage", "roa"]].dropna(subset=["size"]).merge(cw, on="gvkey", how="inner")
if _args.sample == "artifacts": comp = comp[~comp["cik"].isin(ART_FAM | ART_SHELL)].copy()
# exposure flows (formed in year y) and stocks (alive on Jan 1 of y), per destination firm
base_all["inbound"] = (base_all["src_is_origin"] == True).astype(int)
flow = base_all.groupby(["dst", "by"]).agg(nnew=("exposed", "size"), nexp=("exposed", "sum"), nexp5=("exposed5y", "sum"),
                                          nnew_in=("inbound", "sum"), nexp_in=("exposed", lambda x: int((x * base_all.loc[x.index, "inbound"]).sum())),
                                          nexp5_in=("exposed5y", lambda x: int((x * base_all.loc[x.index, "inbound"]).sum()))).reset_index().rename(columns={"dst": "cik", "by": "year"})
flow["nnew_out"] = flow["nnew"] - flow["nnew_in"]; flow["nexp_out"] = flow["nexp"] - flow["nexp_in"]; flow["nexp5_out"] = flow["nexp5"] - flow["nexp5_in"]
stock_rows = []
for y in range(FORM_LO, FORM_HI + 2):
    jan1 = pd.Timestamp(f"{y}-01-01")
    al = base_all[(base_all.birth < jan1) & (base_all.death >= jan1)]
    s = al.groupby("dst").agg(nalive=("exposed", "size"), nexp_alive=("exposed", "sum"), nexp5_alive=("exposed5y", "sum")).reset_index(); s["year"] = y; stock_rows.append(s)
stock = pd.concat(stock_rows).rename(columns={"dst": "cik"})
p = comp.copy()
p["restate_file"] = [int(any(f.year == y for f in rf.get(c, ()))) for c, y in zip(p["cik"], p["year"])]
p["restate_onset"] = [int(any(b.year == y for b, f in ro.get(c, ()))) for c, y in zip(p["cik"], p["year"])]
p["had_prior_restate"] = [int(any(f.year < y for f in rf.get(c, ()))) for c, y in zip(p["cik"], p["year"])]
p["restated_before_2005"] = [int(any(f.year < 2005 for f in rf.get(c, ()))) for c in p["cik"]]   # pre-sample history (time-invariant; absorbed by firm FE, used in pooled models)
pn2 = pn.rename(columns={"company": "cik"}); pn2["cik"] = norm_cik(pn2["cik"])
p = p.merge(pn2[["cik", "year", "degree", "betweenness"]].assign(year=lambda d: d["year"] + 1).rename(columns={"degree": "degree_lag", "betweenness": "betweenness_lag"}), on=["cik", "year"], how="left")
p = p.merge(flow.assign(year=flow["year"] + 1).rename(columns={"nnew": "nnew_lag1", "nexp": "nexp_lag1", "nexp5": "nexp5_lag1", "nnew_in": "nnew_in_lag1", "nexp_in": "nexp_in_lag1", "nnew_out": "nnew_out_lag1", "nexp_out": "nexp_out_lag1", "nexp5_in": "nexp5_in_lag1", "nexp5_out": "nexp5_out_lag1"}), on=["cik", "year"], how="left")
p = p.merge(flow.rename(columns={"nnew": "nnew_t", "nexp": "nexp_t"}), on=["cik", "year"], how="left")
p = p.merge(stock.rename(columns={"nalive": "nalive_boy", "nexp_alive": "nexp_boy", "nexp5_alive": "nexp5_boy"}), on=["cik", "year"], how="left")
for c in ["degree_lag", "betweenness_lag", "nnew_lag1", "nexp_lag1", "nexp5_lag1", "nnew_t", "nexp_t", "nalive_boy", "nexp_boy", "nnew_in_lag1", "nexp_in_lag1", "nnew_out_lag1", "nexp_out_lag1", "nexp5_boy", "nexp5_in_lag1", "nexp5_out_lag1"]: p[c] = p[c].fillna(0)
p = p[(p.year >= 2005) & (p.year <= FORM_HI)].replace([np.inf, -np.inf], np.nan).dropna(subset=["size", "leverage", "roa"])
p = p[p.groupby("gvkey")["year"].transform("count") >= 2].copy()
p.to_csv(f"data/integrated_panel_k{SUFFIX}.csv", index=False)
pp_ = p.set_index(["gvkey", "year"])
def fe(dep, exog):
    r = PanelOLS(pp_[dep], pp_[exog], entity_effects=True, time_effects=True, drop_absorbed=True).fit(cov_type="clustered", cluster_entity=True)
    return r
def cb(r, k): return {"coef": round(float(r.params[k]), 6), "se": round(float(r.std_errors[k]), 6), "t": round(float(r.tstats[k]), 2), "p": round(float(r.pvalues[k]), 5)}
ctrl = ["degree_lag", "betweenness_lag", "size", "leverage", "roa"]
K2 = {"estimation_sample": {"firm_years": int(len(p)), "firms": int(p.gvkey.nunique()), "years": [int(p.year.min()), int(p.year.max())],
                            "rate_file": round(float(p.restate_file.mean()), 4), "rate_onset": round(float(p.restate_onset.mean()), 4),
                            "mean_nexp_lag1": round(float(p.nexp_lag1.mean()), 4), "mean_nexp_boy": round(float(p.nexp_boy.mean()), 4), "share_nexp_lag1_pos": round(float((p.nexp_lag1 > 0).mean()), 4)},
      "models": {}}
specs = {"contemporaneous_file_no_turnover (old timing, old controls)": ("restate_file", "nexp_t", ctrl),
         "contemporaneous_file (old timing)": ("restate_file", "nexp_t", ["nnew_t"] + ctrl),
         "lag1_file": ("restate_file", "nexp_lag1", ["nnew_lag1"] + ctrl),
         "lag1_file_plus_prior": ("restate_file", "nexp_lag1", ["nnew_lag1"] + ctrl + ["had_prior_restate"]),
         "lag1_onset": ("restate_onset", "nexp_lag1", ["nnew_lag1"] + ctrl),
         "lag1_onset_plus_prior": ("restate_onset", "nexp_lag1", ["nnew_lag1"] + ctrl + ["had_prior_restate"]),   # 2026-10-15 r12: Table 8 row 1 with own history held fixed
         "boy_stock_file": ("restate_file", "nexp_boy", ["nalive_boy"] + ctrl),
         "boy_stock_onset": ("restate_onset", "nexp_boy", ["nalive_boy"] + ctrl),
         "lag1_file_exposed5y": ("restate_file", "nexp5_lag1", ["nnew_lag1"] + ctrl),
         "lag1_file_exposed5y_plus_prior": ("restate_file", "nexp5_lag1", ["nnew_lag1"] + ctrl + ["had_prior_restate"]),
         "lag1_onset_exposed5y_plus_prior": ("restate_onset", "nexp5_lag1", ["nnew_lag1"] + ctrl + ["had_prior_restate"]),
         "boy_stock_onset_exposed5y": ("restate_onset", "nexp5_boy", ["nalive_boy"] + ctrl),
         "boy_stock_onset_exposed5y_plus_prior": ("restate_onset", "nexp5_boy", ["nalive_boy"] + ctrl + ["had_prior_restate"]),   # 2026-10-15 r12: Table 8 stock row with own history held fixed
         "lag1_onset_inbound_recent5y_plus_prior": ("restate_onset", "nexp5_in_lag1", ["nnew_in_lag1", "nnew_out_lag1", "nexp5_out_lag1"] + ctrl + ["had_prior_restate"]),
         "lag1_onset_exposed5y_no_prior_control": ("restate_onset", "nexp5_lag1", ["nnew_lag1"] + ctrl),
         "lag1_file_inbound_recent5y_plus_prior": ("restate_file", "nexp5_in_lag1", ["nnew_in_lag1", "nnew_out_lag1", "nexp5_out_lag1"] + ctrl + ["had_prior_restate"]),
         "lag1_onset_exposed5y": ("restate_onset", "nexp5_lag1", ["nnew_lag1"] + ctrl),
         "lag1_file_no_turnover_control": ("restate_file", "nexp_lag1", ctrl),
         "lag1_file_inbound_only": ("restate_file", "nexp_in_lag1", ["nnew_in_lag1", "nnew_out_lag1", "nexp_out_lag1"] + ctrl),
         "lag1_onset_inbound_only": ("restate_onset", "nexp_in_lag1", ["nnew_in_lag1", "nnew_out_lag1", "nexp_out_lag1"] + ctrl)}
if SPAC:
    pp_ns = p[~p["cik"].isin(SPAC)].set_index(["gvkey", "year"])
    def fe_ns(dep, exog): return PanelOLS(pp_ns[dep], pp_ns[exog], entity_effects=True, time_effects=True, drop_absorbed=True).fit(cov_type="clustered", cluster_entity=True)
    for name, (dep, key, controls) in (("lag1_file_excl_blank_check", ("restate_file", "nexp_lag1", ["nnew_lag1"] + ctrl)), ("lag1_onset_excl_blank_check", ("restate_onset", "nexp_lag1", ["nnew_lag1"] + ctrl))):
        r = fe_ns(dep, [key] + controls)
        K2["models"][name] = {"dep": dep, "exposure": key, "controls": controls, "exposure_coef": cb(r, key), "turnover_coef": cb(r, controls[0]), "r2_within": round(float(r.rsquared), 5), "n": int(r.nobs),
                              "oster": {"beta_uncontrolled": round(float(fe_ns(dep, [key]).params[key]), 6)}}
    K2["estimation_sample"]["firm_years_excl_blank_check"] = int(len(pp_ns)); K2["estimation_sample"]["blank_check_firms_in_panel"] = int(p.loc[p["cik"].isin(SPAC), "gvkey"].nunique())
for name, (dep, key, controls) in specs.items():
    r = fe(dep, [key] + controls); r0 = fe(dep, [key])
    bu, Ru, bc_, Rc = float(r0.params[key]), float(r0.rsquared), float(r.params[key]), float(r.rsquared)
    oster = {"beta_uncontrolled": round(bu, 6), "R2_uncontrolled": round(Ru, 5), "beta_controlled": round(bc_, 6), "R2_controlled": round(Rc, 5)}
    for mult in (1.3, 2.0):
        Rmax = min(mult * Rc, 1.0); den = (Rc - Ru)
        bs = bc_ - 1.0 * (bu - bc_) * (Rmax - Rc) / den if den > 0 else None
        d0 = (bc_ * den) / ((bu - bc_) * (Rmax - Rc)) if (bu != bc_ and den > 0 and Rmax > Rc) else None
        oster[f"Rmax_{mult}x"] = {"beta_star_delta1": round(bs, 6) if bs is not None else None, "delta_for_beta0": round(d0, 3) if d0 is not None else None}
    K2["models"][name] = {"dep": dep, "exposure": key, "controls": controls, "exposure_coef": cb(r, key),
                          "turnover_coef": cb(r, controls[0]) if controls and controls[0] in r.params.index else None,
                          "r2_within": round(float(r.rsquared), 5), "n": int(r.nobs), "oster": oster}
    log(f"K2 {name}: {key} -> {dep}: {r.params[key]*100:+.3f}pp (t={r.tstats[key]:.2f}); uncontrolled {bu*100:+.3f}pp")
# between-firm screening test: pooled logit with year effects; baseline = own prior restatement + firm characteristics + all new interlocks; add exposure
import statsmodels.api as sm
def auc_(y, s_):
    r_ = pd.Series(s_).rank().values; n1 = y.sum(); n0 = len(y) - n1
    return float((r_[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
yd = pd.get_dummies(p["year"].astype(int), prefix="y", drop_first=True).astype(float)
basecols = ["had_prior_restate", "size", "leverage", "roa", "degree_lag", "nnew_lag1"]
K2["between_firm_screening"] = {}
for dep_ in ("restate_file", "restate_onset"):
    y_ = p[dep_].values.astype(int)
    X0 = sm.add_constant(pd.concat([p[basecols].reset_index(drop=True), yd.reset_index(drop=True)], axis=1))
    m0 = sm.Logit(y_, X0).fit(disp=0, maxiter=200); a0 = auc_(y_, m0.predict(X0))
    out_ = {"baseline_auc": round(a0, 4), "baseline_coef_had_prior_restate": round(float(m0.params["had_prior_restate"]), 4)}
    for ex_ in ("nexp_lag1", "nexp_boy", "nexp5_lag1", "nexp5_boy"):
        X1 = sm.add_constant(pd.concat([p[basecols + [ex_]].reset_index(drop=True), yd.reset_index(drop=True)], axis=1))
        m1 = sm.Logit(y_, X1).fit(disp=0, maxiter=200); a1 = auc_(y_, m1.predict(X1))
        out_[ex_] = {"coef": round(float(m1.params[ex_]), 4), "se": round(float(m1.bse[ex_]), 4), "z": round(float(m1.tvalues[ex_]), 2), "auc": round(a1, 4), "auc_gain": round(a1 - a0, 4),
                     "lr_stat": round(float(2 * (m1.llf - m0.llf)), 2)}
    K2["between_firm_screening"][dep_] = out_
    log(f"screening {dep_}: AUC base {a0:.4f}; +nexp_lag1 {out_['nexp_lag1']['auc']:.4f} (z {out_['nexp_lag1']['z']}); +nexp_boy {out_['nexp_boy']['auc']:.4f} (z {out_['nexp_boy']['z']})")
# temporal out-of-sample screening test: fit on 2005-2016 (no year effects), evaluate AUC on 2017-2023 with and without exposure counts
tr_ = p[p.year <= 2016]; te_ = p[p.year >= 2017]
K2["between_firm_screening_out_of_sample"] = {"train_years": [2005, 2016], "test_years": [2017, int(p.year.max())], "n_train": int(len(tr_)), "n_test": int(len(te_))}
for dep_ in ("restate_file", "restate_onset"):
    ytr = tr_[dep_].values.astype(int); yte = te_[dep_].values.astype(int)
    X0tr = sm.add_constant(tr_[basecols].reset_index(drop=True)); X0te = sm.add_constant(te_[basecols].reset_index(drop=True), has_constant="add")
    m0 = sm.Logit(ytr, X0tr).fit(disp=0, maxiter=200); a0 = auc_(yte, m0.predict(X0te))
    out_ = {"baseline_auc_test": round(a0, 4), "test_event_rate": round(float(yte.mean()), 4)}
    for ex_ in ("nexp_lag1", "nexp_boy", "nexp5_lag1", "nexp5_boy"):
        X1tr = sm.add_constant(tr_[basecols + [ex_]].reset_index(drop=True)); X1te = sm.add_constant(te_[basecols + [ex_]].reset_index(drop=True), has_constant="add")
        m1 = sm.Logit(ytr, X1tr).fit(disp=0, maxiter=200); a1 = auc_(yte, m1.predict(X1te))
        out_[ex_] = {"coef_train": round(float(m1.params[ex_]), 4), "auc_test": round(a1, 4), "auc_gain_test": round(a1 - a0, 4)}
    K2["between_firm_screening_out_of_sample"][dep_] = out_
    log(f"OOS screening {dep_}: base AUC {a0:.4f}; gains " + ", ".join(f"{k}:{out_[k]['auc_gain_test']:+.4f}" for k in ("nexp_lag1", "nexp_boy", "nexp5_lag1", "nexp5_boy")))
# Nickell check: the own-history indicator is a lagged outcome inside firm fixed effects; re-estimate the recent-exposure onset rows with year effects only (firm-clustered) and with history measured before the sample
def fe_year_only(dep, exog):
    return PanelOLS(pp_[dep], pp_[exog], entity_effects=False, time_effects=True, drop_absorbed=True).fit(cov_type="clustered", cluster_entity=True)
K2["nickell_check"] = {}
for name, (dep, key, controls) in (("lag1_onset_exposed5y_plus_prior_yearFE_only", ("restate_onset", "nexp5_lag1", ["nnew_lag1"] + ctrl + ["had_prior_restate"])),
                                   ("lag1_onset_exposed5y_plus_pre2005_history_yearFE_only", ("restate_onset", "nexp5_lag1", ["nnew_lag1"] + ctrl + ["restated_before_2005"])),
                                   ("lag1_onset_inbound_recent5y_plus_prior_yearFE_only", ("restate_onset", "nexp5_in_lag1", ["nnew_in_lag1", "nnew_out_lag1", "nexp5_out_lag1"] + ctrl + ["had_prior_restate"])),
                                   ("lag1_file_exposed5y_plus_prior_yearFE_only", ("restate_file", "nexp5_lag1", ["nnew_lag1"] + ctrl + ["had_prior_restate"]))):
    r = fe_year_only(dep, [key] + controls)
    K2["nickell_check"][name] = {"dep": dep, "exposure": key, "exposure_coef": cb(r, key), "history_coef": cb(r, controls[-1]), "n": int(r.nobs)}
    log(f"K2 nickell {name}: {r.params[key]*100:+.3f}pp (t={r.tstats[key]:.2f})")
# right-truncation check: onset rows on 2005-2020 only (onsets after 2020 may not yet have been filed)
pp20 = p[p.year <= 2020].set_index(["gvkey", "year"])
def fe20(dep, exog): return PanelOLS(pp20[dep], pp20[exog], entity_effects=True, time_effects=True, drop_absorbed=True).fit(cov_type="clustered", cluster_entity=True)
K2["onset_rows_2005_2020"] = {}
for name, (dep, key, controls) in (("lag1_onset_exposed5y", ("restate_onset", "nexp5_lag1", ["nnew_lag1"] + ctrl)),
                                   ("lag1_onset_exposed5y_plus_prior", ("restate_onset", "nexp5_lag1", ["nnew_lag1"] + ctrl + ["had_prior_restate"])),
                                   ("lag1_onset_inbound_recent5y_plus_prior", ("restate_onset", "nexp5_in_lag1", ["nnew_in_lag1", "nnew_out_lag1", "nexp5_out_lag1"] + ctrl + ["had_prior_restate"])),
                                   ("lag1_onset", ("restate_onset", "nexp_lag1", ["nnew_lag1"] + ctrl))):
    r = fe20(dep, [key] + controls)
    K2["onset_rows_2005_2020"][name] = {"exposure_coef": cb(r, key), "n": int(r.nobs)}
    log(f"K2 2005-2020 {name}: {r.params[key]*100:+.3f}pp (t={r.tstats[key]:.2f})")
OUT["K2_panel"] = K2

# ======================================================================================
# K5. Survival with competing risks (origin-based exposure), PH diagnostics
# ======================================================================================
sv = inwin[inwin.origin.notna()].copy()
sv["dest"] = np.where(sv.origin == sv.company_a, sv.company_b, sv.company_a)
sv["prior_any"] = [int(any_before(rf.get(o, ()), b)) for o, b in zip(sv.origin, sv.edge_birth)]
sv["prior_5y"] = [int(any_before(rf.get(o, ()), b, 5)) for o, b in zip(sv.origin, sv.edge_birth)]
sv["dur"] = ((sv.edge_death - sv.edge_birth).dt.days / 365.25).clip(lower=1 / 365.25)
sv["ev_any"] = (~sv.ongoing).astype(int)
sv["ev_dest"] = ((~sv.ongoing) & (sv.terminated_by == "destination")).astype(int)
sv["ev_orig"] = ((~sv.ongoing) & (sv.terminated_by == "origin")).astype(int)
sv["origin_last_event"] = sv["origin"].map(LAST_EVENT)
sv["origin_firm_continues"] = (sv["origin_last_event"] > sv["edge_death"] + pd.Timedelta(days=365)).astype(int)   # any board event at the origin >1y after the edge ended
sv["ev_orig_firm_continues"] = (sv["ev_orig"].astype(bool) & sv["origin_firm_continues"].astype(bool)).astype(int)
sv["ev_orig_firm_exit"] = (sv["ev_orig"].astype(bool) & ~sv["origin_firm_continues"].astype(bool)).astype(int)
sv["cl"] = pd.factorize(sv.origin)[0]
PSPELLS = defaultdict(list)
for p_, s_, e_ in zip(ten["person"], ten["start"], ten["end"]): PSPELLS[p_].append((s_, e_))
FIRST_EVENT = bd.groupby("cfkey")["eff_date"].min().to_dict()
sv["seats_at_birth"] = [sum(1 for s_, e_ in PSPELLS[d] if s_ <= b <= e_) for d, b in zip(sv.director, sv.edge_birth)]
sv["dest_prior_any"] = [int(any_before(rf.get(d, ()), b)) for d, b in zip(sv.dest, sv.edge_birth)]
sv["origin_feed_age"] = [(b - FIRST_EVENT[o]).days / 365.25 for o, b in zip(sv.origin, sv.edge_birth)]
sv["dest_feed_age"] = [(b - FIRST_EVENT[d]).days / 365.25 for d, b in zip(sv.dest, sv.edge_birth)]
sv["origin_blank_check"] = sv.origin.isin(SPAC).astype(int) if SPAC else 0
def _tenure(person, firm, t):
    st_ = spell_start_at(person, firm, t); return (t - st_).days / 365.25 if st_ is not None else 0.0
sv["origin_tenure_years"] = [_tenure(d_, o_, b_) for d_, o_, b_ in zip(sv.director, sv.origin, sv.edge_birth)]
COV = ["seats_at_birth", "dest_prior_any", "origin_feed_age", "dest_feed_age", "origin_tenure_years"]
K5 = {"n_edges": int(len(sv)), "share_ongoing": round(float(sv.ongoing.mean()), 4), "share_prior_any": round(float(sv.prior_any.mean()), 4), "share_prior_5y": round(float(sv.prior_5y.mean()), 4),
      "terminated_by": sv.terminated_by.fillna("ongoing").value_counts().to_dict()}
for expo in ("prior_any", "prior_5y"):
    blk = {}
    for lab, m in (("prior", sv[expo] == 1), ("clean", sv[expo] == 0)):
        k = KaplanMeierFitter().fit(sv.loc[m, "dur"], sv.loc[m, "ev_any"])
        blk[f"km_{lab}"] = {"n": int(m.sum()), "events": int(sv.loc[m, "ev_any"].sum()), "S5": round(float(k.predict(5.0)), 3), "S10": round(float(k.predict(10.0)), 3),
                            "median": None if np.isinf(k.median_survival_time_) else round(float(k.median_survival_time_), 2)}
    blk["logrank_p"] = float(logrank_test(sv.loc[sv[expo] == 1, "dur"], sv.loc[sv[expo] == 0, "dur"], sv.loc[sv[expo] == 1, "ev_any"], sv.loc[sv[expo] == 0, "ev_any"]).p_value)
    blk["share_origin_seat_ends_with_firm_continuing"] = round(float(sv.loc[sv.ev_orig == 1, "origin_firm_continues"].mean()), 4)
    for ev, lab in (("ev_any", "any_dissolution"), ("ev_dest", "destination_seat_ended"), ("ev_orig", "origin_seat_ended"), ("ev_orig_firm_continues", "origin_seat_ended_firm_continues"), ("ev_orig_firm_exit", "origin_seat_ended_firm_exit")):
        df_ = sv[["dur", ev, expo, "birth_year", "cl"]].rename(columns={expo: "x", ev: "e"})
        c = CoxPHFitter().fit(df_, duration_col="dur", event_col="e", strata=["birth_year"], cluster_col="cl", robust=True)
        hr = float(np.exp(c.params_["x"])); ci = np.exp(c.confidence_intervals_.loc["x"].values)
        res = {"hr": round(hr, 3), "ci95": [round(float(ci[0]), 3), round(float(ci[1]), 3)], "p": float(c.summary.loc["x", "p"]), "events": int(df_["e"].sum())}
        try:
            pht = proportional_hazard_test(c, df_, time_transform="rank"); res["ph_test_p"] = float(pht.summary["p"].iloc[0])
        except Exception as e_: res["ph_test_p"] = f"n/a ({type(e_).__name__})"
        blk[f"cox_{lab}"] = res
        if lab in ("destination_seat_ended", "origin_seat_ended"):   # covariate-adjusted cause-specific model
            dfc = sv[["dur", ev, expo, "birth_year", "cl"] + COV].rename(columns={expo: "x", ev: "e"})
            c2 = CoxPHFitter().fit(dfc, duration_col="dur", event_col="e", strata=["birth_year"], cluster_col="cl", robust=True)
            ci2 = np.exp(c2.confidence_intervals_.loc["x"].values)
            blk[f"cox_{lab}_covariates"] = {"hr": round(float(np.exp(c2.params_["x"])), 3), "ci95": [round(float(ci2[0]), 3), round(float(ci2[1]), 3)], "p": float(c2.summary.loc["x", "p"]), "events": int(dfc["e"].sum()),
                                            "covariate_hr": {k: round(float(np.exp(c2.params_[k])), 3) for k in COV}}
            # interval-specific hazards (PH relaxed): subjects at risk at the interval start, censored at its end
            pw = {}
            for lo_, hi_, ilab in ((0.0, 2.0, "0_2y"), (2.0, 5.0, "2_5y"), (5.0, np.inf, "5y_plus")):
                sub_ = sv[sv.dur > lo_].copy()
                sub_["dur_i"] = np.minimum(sub_["dur"], hi_) - lo_; sub_["e_i"] = ((sub_[ev] == 1) & (sub_["dur"] <= hi_)).astype(int)
                dfi = sub_[["dur_i", "e_i", expo, "birth_year", "cl"] + COV].rename(columns={expo: "x"})
                try:
                    ci_ = CoxPHFitter().fit(dfi, duration_col="dur_i", event_col="e_i", strata=["birth_year"], cluster_col="cl", robust=True)
                    cii = np.exp(ci_.confidence_intervals_.loc["x"].values)
                    pw[ilab] = {"hr": round(float(np.exp(ci_.params_["x"])), 3), "ci95": [round(float(cii[0]), 3), round(float(cii[1]), 3)], "p": float(ci_.summary.loc["x", "p"]), "events": int(dfi["e_i"].sum()), "at_risk": int(len(dfi))}
                except Exception as e_: pw[ilab] = {"error": type(e_).__name__}
            blk[f"cox_{lab}_piecewise"] = pw
    K5[expo] = blk
    log(f"K5 {expo}: any HR {blk['cox_any_dissolution']['hr']} p={blk['cox_any_dissolution']['p']:.3g}; dest-seat HR {blk['cox_destination_seat_ended']['hr']} p={blk['cox_destination_seat_ended']['p']:.3g}; origin-seat HR {blk['cox_origin_seat_ended']['hr']}")
OUT["K5_survival"] = K5

# ======================================================================================
# K7. Firm-count crosswalk
# ======================================================================================
OUT["K7_firm_counts"] = {"D&O universe firms (any board-member event)": int(bd.cfkey.nunique()),
                         "firms with >=1 tenure spell": int(ten.company.nunique()),
                         "firms in >=1 edge formed 2004-2023 (pair base firms)": int(pd.unique(base[["src", "dst"]].values.ravel()).size),
                         "firms in yearly networks (any year)": int(pn.company.nunique()),
                         "panel firms (gvkey, Compustat-linked, 2005-2023)": int(p.gvkey.nunique()),
                         "panel firm-years": int(len(p))}
json.dump(OUT, open(f"results_rebuild/K_rebuild{SUFFIX}.json", "w"), indent=2, default=float)
log(f"saved results_rebuild/K_rebuild{SUFFIX}.json")
