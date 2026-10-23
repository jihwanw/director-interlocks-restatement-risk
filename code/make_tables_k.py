#!/usr/bin/env python3
"""
make_tables_k.py — exhibits for the JAPP version ("Sorting before transmission") from results_rebuild/K_rebuild_{le6,full}.json.
Every cell is read from JSON. Writes 3rd_revision/source/tables_auto/*.tex and figures_k/*.pdf.
Primary sample: directors interlocked with at most 6 firms; pairs in the appointment direction (origin board -> appointing board).
Run from github/:  python code/make_tables_k.py
"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
import os
PRIMARY = os.environ.get("PRIMARY_SUFFIX", "_art")   # primary: all directors, fund/partnership/REIT-family and shell/blank-check registrants and common-surname directors removed, conflict-aware name key
K = json.load(open(HERE.parent / "results_rebuild" / f"K_rebuild{PRIMARY}.json"))
KF = json.load(open(HERE.parent / "results_rebuild" / "K_rebuild_full.json"))   # all directors, first|last key, no exclusions (comparison)
KC = json.load(open(HERE.parent / "results_rebuild" / "K_rebuild_le6.json"))    # previous definition: directors with <= 6 firms, first|last key (comparison)
SD = K.get("sample_definition", {})
DC = K["director_firm_counts"]; DIR = K["direction"]; RT = DC["registrant_types"]
SRC = HERE.parent.parent / "3rd_revision" / "source"
OUT = SRC / "tables_auto"; FIG = SRC / "figures_k"
OUT.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 10.5, "figure.dpi": 150, "savefig.bbox": "tight"})

def w(name, s):
    s = s.replace("\\begin{table}[htbp]\\centering", "\\begin{table}[htbp]\\centering\\def\\baselinestretch{1}")  # elsarticle review stretch (1.5) leaks into p-cells
    (OUT / name).write_text(s); print("wrote", name)
def pc(x, d=1): return f"{100*x:.{d}f}"
def g(d): return f"${d['diff_pp']:+.2f}$ ({d['se_pp']:.2f})"                       # gap cell: diff (SE)
def dd(d): return f"${d['diff_in_gaps_pp']:+.2f}$ ({d['se_pp']:.2f}){star(d['t'])}"   # DiD cell
def star(t): return "***" if abs(t) > 2.576 else "**" if abs(t) > 1.96 else "*" if abs(t) > 1.645 else ""

K0, R, K1, K2, K3, K4, K5, K7 = (K["K0_edges"], K["restatements"], K["K1_pairs"], K["K2_panel"], K["K3_cohorts_fixed_window"], K["K4_h5"], K["K5_survival"], K["K7_firm_counts"])
NET = K["K4_network_by_year"]; EW = K1["equal_window_3y"]; NV = K1["never_restated_destinations"]; CL = K1["clean_destinations_no_history"]
R5 = K1["exposure_recent_5y"]; SV = K1["exposure_served_during_misstatement"]; E08 = K1["equal_window_formed_2008plus"]; BD = K1["both_directions_pooled"]
BC = K1.get("blank_check_flag"); XS = K1.get("excluding_blank_check_firms"); EP = K1["exposure_present_expost"]; OB = K1["observable_pre_window_both_firms_3y_in_feed"]; AW = K1["appointment_weighted"]; ET = K1["event_time_gaps"]; CS = K1.get("common_shock_controls", {}); RH = K1["receptivity_heterogeneity"]; PF = K1["exposure_present_at_filing"]; PX = K1["prior_exposure_stock"]; SCO = K1["strict_clean_origins"]; NK = K2["nickell_check"]; T20 = K2["onset_rows_2005_2020"]; E19 = K1["onset_formed_by_2019"]; ATW = K1["attrition_after_window"]
def fe_row(lbl, d_):
    pre_ = f"${d_['pre_gap_pp']:+.2f}$" + (f" ({d_['se_pre_pp']:.2f})" if 'se_pre_pp' in d_ else "")
    return f"{lbl} & {pre_} & ${d_['pre_gap_pp']+d_['diff_in_gaps_pp']:+.2f}$ & {dd(d_)} \\\\\n"

# ── T1 sample ──────────────────────────────────────────────────────────────────────────
w("tab_sample.tex", rf"""\begin{{table}}[htbp]\centering\footnotesize\renewcommand{{\arraystretch}}{{0.9}}
\caption{{Sample construction: director tenure spells, interlock lifecycles, directed pairs, restatements, and firm counts.}}
\label{{tab:sample}}
\adjustbox{{max width=\textwidth}}{{\begin{{tabular}}{{p{{13.6cm}}r}}\toprule
Item & Count \\\midrule
Director and officer events in the Audit Analytics feed, 2004--2025 & {K0['event_flow']['all_dno_events']:,} \\
\quad board-member appointment or departure events & {K0['event_flow']['board_member_start_or_end_events']:,} \\
\quad\quad appointments / departures & {K0['event_flow']['start_events']:,} / {K0['event_flow']['end_events']:,} \\
\quad\quad departures with no observed appointment (incumbents before 2004; open no spell) & {K0['departure_events_without_observed_appointment']:,} \\
\quad\quad repeat appointments within an open spell (ignored) & {K0['event_flow']['repeat_start_events_ignored']:,} \\
\quad\quad appointments after the horizon (no spell) / zero-duration spells dropped & {K0['event_flow']['appointments_after_horizon_no_spell']:,} / {K0['event_flow']['zero_duration_spells_dropped']:,} \\
\quad\quad departures after the horizon (spell counted as open) & {K0['event_flow']['departures_after_horizon_spell_censored']:,} \\
Firms with at least one board-member event & {K7['D&O universe firms (any board-member event)']:,} \\
Director tenure spells (board seats) & {K0['spells']:,} \\
\quad closed by a departure / open at the horizon & {K0['event_flow']['spells_closed_by_a_departure']:,} / {K0['event_flow']['spells_open_at_horizon']:,} \\
Directors holding at least one interlock & {DC['directors_with_interlocks']:,} \\
\quad median / 90th percentile / maximum number of firms per director & {DC['median_firms']:.0f} / {DC['p90_firms']:.0f} / {DC['max_firms']} \\
\quad share of directors interlocked with more than 6 firms & {pc(DC['share_directors_gt6'])}\% \\
\quad share of all interlock lifecycles they generate & {pc(DC['share_edges_from_directors_gt6'])}\% \\
\quad share of their seats in fund, partnership or REIT-family registrants (directors with 2--6 firms in parentheses) & {pc(RT['seat_shares_directors_gt6']['fund_partnership_reit_family'])}\% ({pc(RT['seat_shares_directors_2to6']['fund_partnership_reit_family'])}\%) \\
\quad share of their seats in shell or blank-check registrants & {pc(RT['seat_shares_directors_gt6']['shell_or_blank_check'])}\% ({pc(RT['seat_shares_directors_2to6']['shell_or_blank_check'])}\%) \\
\quad share with a surname shared by more than 150 distinct director names & {pc(RT['share_directors_common_surname_gt150_directors_gt6'])}\% ({pc(RT['share_directors_common_surname_gt150_directors_2to6'])}\%) \\
Interlock lifecycles, all directors, first-and-last-name key, all years & {KF['K0_edges']['edge_lifecycles_all']:,} \\
\quad with names split where recorded middle initials conflict across firms & {SD.get('n_lifecycles_before', K0['edge_lifecycles_all']):,} \\
\quad\quad involving a fund, partnership or REIT-family registrant or a shell or blank-check registrant (removed) & {SD.get('lifecycles_with_family_or_shell_registrant', 0):,} \\
\quad\quad director's surname shared by more than 150 distinct director names (removed; overlaps with the row above; {SD.get('lifecycles_dropped', 0):,} removed in total) & {SD.get('lifecycles_with_common_surname_director', 0):,} \\
Interlock lifecycles, primary sample, all years & {K0['edge_lifecycles_all']:,} \\
\quad formed 2004--2023 (analysis sample) & {K0['edge_lifecycles_2004_2023']:,} \\
\quad unique firm pairs among them & {K0['unique_pairs_2004_2023']:,} \\
\quad still open at the horizon (31 Dec 2025) & {pc(K0['share_ongoing_at_horizon_2004_2023'])}\% \\
\quad with an identifiable origin firm (the director's earlier seat) & {pc(K0['origin_identified_share'])}\% \\
\quad median duration in years / share shorter than 30 days / shorter than one year & {K0['duration_years_median']:.1f} / {pc(K0['share_duration_lt_30d'])}\% / {pc(K0['share_duration_lt_1y'])}\% \\
Directed pairs in the appointment direction, origin $\rightarrow$ appointing board (primary pair sample) & {DIR['rows_appointment_direction']:,} \\
\quad formed 2004--2022, so that a three-year post-formation window is observed & {EW['n_rows']:,} \\
\quad both firms Compustat-linked & {K1['both_compustat_linked']['n_rows']:,} \\
Directed pairs counting each interlock in both directions, primary sample, formed 2004--2023 (comparison only) & {DIR['rows_both_directions']:,} \\
Firms appearing in at least one analysed interlock & {K7['firms in >=1 edge formed 2004-2023 (pair base firms)']:,} \\
Restatement filings through the horizon & {R['n_filings_to_horizon']:,} \\
\quad with a recorded misstated-period start date & {pc(R['share_with_period_begin'])}\% \\
\quad median years from misstated-period start to filing & {R['median_years_file_minus_begin']:.2f} \\
Firm-year panel (Compustat-linked, 2005--2023): firm-years / firms & {K7['panel firm-years']:,} / {K7['panel firms (gvkey, Compustat-linked, 2005-2023)']:,} \\
\bottomrule\end{{tabular}}}}\par\vspace{{3pt}}
\begin{{minipage}}{{\textwidth}}\scriptsize \emph{{Notes.}} Events are restricted to board members (\texttt{{is\_bdmem\_pers}} $=1$); a spell opens on an appointment or return and closes on any of
resignation, retirement, employment ceased, dismissal, declined or failed re-election, death, or revoked appointment. Directors are identified by name,
so distinct individuals with identical names may be merged; Table~\ref{{tab:samples}} and Section~\ref{{sec:results}} report the sensitivity of the main estimates to dropping common names. Registrant types come from the SEC EDGAR
submissions record: ``shell or blank-check'' is SIC 6770 or 9995 or a name containing ``acquisition'' or ``blank check''; ``fund, partnership or REIT family'' is SIC 65xx, 6798, 6221, 6722 or 6726, or no SIC
and a fund-type name; shells that later acquired an operating business carry that business's SIC code and are counted as operating companies. An interlock lifecycle is the period during which one director holds
seats at two firms simultaneously; it is dated from the start of the second seat to the end of either seat. The firm at which the director's seat began
earlier is the origin and the firm whose appointment created the interlock is the destination; the {pc(1-K0['origin_identified_share'])}\% of lifecycles whose two seats began
on the same day have no direction and enter only the undirected comparison. The {pc(DC['share_directors_gt6'])}\% of directors who sit on more than six registrant boards
generate {pc(DC['share_edges_from_directors_gt6'])}\% of all lifecycles under the conflict-aware key; the primary sample keeps them and removes instead the lifecycles that involve a fund, partnership, REIT-family, shell or blank-check registrant or a director whose surname is shared by more than 150 distinct names ({pc(SD.get('share_dropped_among_directors_gt6', 0))}\% of the lifecycles of directors with more than six firms, {pc(SD.get('share_dropped_among_directors_le6', 0))}\% of the rest); Table~\ref{{tab:samples}} reports the unrestricted sample and the alternative six-firm cap for comparison. The horizon is the last date with complete coverage in both feeds.\end{{minipage}}\end{{table}}""")

# ── T2 network by year ──────────────────────────────────────────────────────────────────
rows = "".join(f"{y} & {NET[str(y)]['nodes']:,} & {NET[str(y)]['edges']:,} & {pc(NET[str(y)]['lcc_share'])}\\% & {NET[str(y)]['n_communities']} & {NET[str(y)]['modularity']:.3f} & {pc(NET[str(y)]['cross_edge_ratio'])}\\% \\\\\n" for y in (2004, 2008, 2012, 2016, 2020, 2023))
w("tab_network.tex", rf"""\begin{{table}}[htbp]\centering\footnotesize
\caption{{The board interlock network by year (interlocks alive in the year; primary sample).}}
\label{{tab:network}}
\begin{{threeparttable}}\begin{{tabular}}{{lrrrrrr}}\toprule
Year & Firms & Interlocks & Largest component & Communities & Modularity & Cross-community ties \\\midrule
{rows}\bottomrule\end{{tabular}}
\begin{{tablenotes}}\footnotesize\item Communities from the Louvain algorithm (seed 42). Because tenure spells are observed only from the start of the Audit Analytics feed,
directors already in office before 2004 enter the network only when a subsequent event is recorded; the early years therefore understate the
network, and the growth in the first rows is partly a coverage effect. Analyses that could be affected are repeated on interlocks formed in 2008 or later.\end{{tablenotes}}
\end{{threeparttable}}\end{{table}}""")

# ── T3 clustering (P1) ──────────────────────────────────────────────────────────────────
a = K1; lc = a["left_censoring_robustness_formed_2008plus"]; da = a["destination_firm_adjusted"]
w("tab_cluster.tex", rf"""\begin{{table}}[htbp]\centering\footnotesize
\caption{{Does restatement risk cluster along appointment ties? Destination (appointing-board) restatement rates by origin exposure.}}
\label{{tab:cluster}}
\begin{{threeparttable}}\begin{{tabular}}{{p{{7.2cm}}ccc}}\toprule
Outcome (destination firm) & Exposed & Clean & Gap, pp (SE) \\\midrule
\multicolumn{{4}}{{@{{}}p{{15.8cm}}@{{}}}}{{\emph{{Panel A. Window: life of the interlock plus one year; appointment direction, $N={a['outcome_file_in_window']['n_rows']:,}$ pairs}}}} \\
Restatement filed in window & {pc(a['outcome_file_in_window']['exposed_rate'],2)}\% & {pc(a['outcome_file_in_window']['clean_rate'],2)}\% & {g(a['outcome_file_in_window'])} \\
Misstated period begins in window & {pc(a['outcome_onset_in_window']['exposed_rate'],2)}\% & {pc(a['outcome_onset_in_window']['clean_rate'],2)}\% & {g(a['outcome_onset_in_window'])} \\
\quad exposure $=$ origin restated within 5 years before the tie: filed & {pc(a['outcome_file_in_window_exposed5y']['exposed_rate'],2)}\% & {pc(a['outcome_file_in_window_exposed5y']['clean_rate'],2)}\% & {g(a['outcome_file_in_window_exposed5y'])} \\
\quad interlocks formed 2008 or later: filed & {pc(lc['file']['exposed_rate'],2)}\% & {pc(lc['file']['clean_rate'],2)}\% & {g(lc['file'])} \\
\quad each interlock counted in both directions (undirected design): filed & {pc(BD['life_plus1y_file']['exposed_rate'],2)}\% & {pc(BD['life_plus1y_file']['clean_rate'],2)}\% & {g(BD['life_plus1y_file'])} \\
\midrule
\multicolumn{{4}}{{@{{}}p{{15.8cm}}@{{}}}}{{\emph{{Panel B. Destination (appointing) firms, one row per firm; $N={da['file']['n_firms']:,}$}}}} \\
Ever restated after an inbound interlock, unadjusted & & & ${da['file']['raw_gap_pp']:+.2f}$ ({da['file']['raw_se_pp']:.2f}) \\
\quad adjusted for inbound interlocks, exposure-years, first-tie year & & & ${da['file']['adjusted_gap_pp']:+.2f}$ ({da['file']['se_pp']:.2f}) \\
Misstated period ever begins after an inbound interlock, adjusted & & & ${da['onset']['adjusted_gap_pp']:+.2f}$ ({da['onset']['se_pp']:.2f}) \\
\bottomrule\end{{tabular}}
\begin{{tablenotes}}\footnotesize\item Exposed $=$ the origin firm (the director's earlier board) filed a restatement before the interlock formed. Standard errors in Panel A are cluster-robust
in two dimensions, origin firm and destination firm \citep{{cameron2011}}. Panel B uses HC1 standard errors in a linear probability model with the log number of inbound
interlocks, log(1 $+$ inbound exposure-years) and first-tie-year dummies. ``Clean'' origins include firms that restated only after the tie formed. The undirected row counts
each interlock twice, once in each direction, as static designs that cannot tell origin from destination implicitly do.\end{{tablenotes}}\end{{threeparttable}}\end{{table}}""")

# ── T4 before and after (P2): core panels ────────────────────────────────────────────────
dp = a["destination_prior_restatement_share"]; o19 = a["onset_formed_by_2019"]
def row(lbl, pre, post, d_): return f"{lbl} & {g(pre)} & {g(post)} & {dd(d_)} \\\\\n"
w("tab_prepost.tex", rf"""\begin{{table}}[htbp]\centering
\caption{{Before and after the tie: the exposed--clean gap in destination restatements over equal three-year windows.}}
\label{{tab:prepost}}
\footnotesize\adjustbox{{max width=\textwidth}}{{\begin{{tabular}}{{p{{6.6cm}}ccc}}\toprule
 & 3 years before & 3 years after & Difference \\
Outcome & formation & formation & (after $-$ before) \\
 & gap, pp (SE) & gap, pp (SE) & pp (SE) \\\midrule
\multicolumn{{4}}{{@{{}}p{{15.8cm}}@{{}}}}{{\emph{{Panel A. Exposure $=$ origin filed a restatement at any time before the tie ({pc(EW['pre_file']['n_exposed_rows']/EW['pre_file']['n_rows'])}\% of pairs exposed)}}}} \\
{row("Restatement filed", EW["pre_file"], EW["post_file"], EW["did_file"])}{row("Misstated period begins", EW["pre_onset"], EW["post_onset"], EW["did_onset"])}{row("Severe restatement filed (fraud or SEC investigation)", EW["pre_severe"], EW["post_severe"], EW["did_severe"])}\midrule
\multicolumn{{4}}{{@{{}}p{{15.8cm}}@{{}}}}{{\emph{{Panel B. Exposure $=$ origin filed a restatement within 5 years before the tie ({pc(R5['share_exposed5y'])}\% of pairs)}}}} \\
{row("Restatement filed", R5["pre_file"], R5["post_file"], R5["did_file"])}{row("Misstated period begins", R5["pre_onset"], R5["post_onset"], R5["did_onset"])}\midrule
\multicolumn{{4}}{{@{{}}p{{15.8cm}}@{{}}}}{{\emph{{Panel C. Exposure $=$ the director sat on the origin board during a misstated period restated before the tie ({pc(SV['share_exposed_served'])}\% of pairs)}}}} \\
{row("Restatement filed", SV["pre_file"], SV["post_file"], SV["did_file"])}{row("Misstated period begins", SV["pre_onset"], SV["post_onset"], SV["did_onset"])}\midrule
\multicolumn{{4}}{{@{{}}p{{15.8cm}}@{{}}}}{{\emph{{Panel C$'$. Exposure $=$ the director sat on the origin board during a misstated period restated at any time, including after the tie (retrospective classification; {pc(EP['share'])}\% of pairs)}}}} \\
{row("Restatement filed", EP["pre_file"], EP["post_file"], EP["did_file"])}{row("Misstated period begins", EP["pre_onset"], EP["post_onset"], EP["did_onset"])}\midrule
\multicolumn{{4}}{{@{{}}p{{15.8cm}}@{{}}}}{{\emph{{Panel C$''$. Exposure $=$ the director sat on the origin board when a restatement was filed, before the tie ({pc(PF['share'])}\% of pairs)}}}} \\
{row("Restatement filed", PF["pre_file"], PF["post_file"], PF["did_file"])}{row("Misstated period begins", PF["pre_onset"], PF["post_onset"], PF["did_onset"])}\midrule
\multicolumn{{4}}{{@{{}}p{{15.8cm}}@{{}}}}{{\emph{{Panel D. Change in the misstated-period-onset gap with formation-year and formation-year $\times$ post effects, by exposure definition}}}} \\
Any prior restatement at the origin & ${EW['did_onset_cohortFE']['pre_gap_pp']:+.2f}$ ({EW['did_onset_cohortFE']['se_pre_pp']:.2f}) & ${EW['did_onset_cohortFE']['pre_gap_pp']+EW['did_onset_cohortFE']['diff_in_gaps_pp']:+.2f}$ & {dd(EW['did_onset_cohortFE'])} \\
\quad same, formation-year $\times$ destination-size-decile cells & ${EW['did_onset_cellFE_any']['pre_gap_pp']:+.2f}$ ({EW['did_onset_cellFE_any']['se_pre_pp']:.2f}) & ${EW['did_onset_cellFE_any']['pre_gap_pp']+EW['did_onset_cellFE_any']['diff_in_gaps_pp']:+.2f}$ & {dd(EW['did_onset_cellFE_any'])} \\
Origin restated within 5 years & ${EW['did_onset_cohortFE_recent']['pre_gap_pp']:+.2f}$ ({EW['did_onset_cohortFE_recent']['se_pre_pp']:.2f}) & ${EW['did_onset_cohortFE_recent']['pre_gap_pp']+EW['did_onset_cohortFE_recent']['diff_in_gaps_pp']:+.2f}$ & {dd(EW['did_onset_cohortFE_recent'])} \\
\quad same, formation-year $\times$ destination-size-decile cells & ${EW['did_onset_cellFE_recent']['pre_gap_pp']:+.2f}$ ({EW['did_onset_cellFE_recent']['se_pre_pp']:.2f}) & ${EW['did_onset_cellFE_recent']['pre_gap_pp']+EW['did_onset_cellFE_recent']['diff_in_gaps_pp']:+.2f}$ & {dd(EW['did_onset_cellFE_recent'])} \\
Director present during the origin's misstatement & ${EW['did_onset_cohortFE_present']['pre_gap_pp']:+.2f}$ ({EW['did_onset_cohortFE_present']['se_pre_pp']:.2f}) & ${EW['did_onset_cohortFE_present']['pre_gap_pp']+EW['did_onset_cohortFE_present']['diff_in_gaps_pp']:+.2f}$ & {dd(EW['did_onset_cohortFE_present'])} \\
Director present when the restatement was filed & ${EW['did_onset_cohortFE_present_at_filing']['pre_gap_pp']:+.2f}$ ({EW['did_onset_cohortFE_present_at_filing']['se_pre_pp']:.2f}) & ${EW['did_onset_cohortFE_present_at_filing']['pre_gap_pp']+EW['did_onset_cohortFE_present_at_filing']['diff_in_gaps_pp']:+.2f}$ & {dd(EW['did_onset_cohortFE_present_at_filing'])} \\
Director present during a misstatement restated at any time & ${EW['did_onset_cohortFE_present_expost']['pre_gap_pp']:+.2f}$ ({EW['did_onset_cohortFE_present_expost']['se_pre_pp']:.2f}) & ${EW['did_onset_cohortFE_present_expost']['pre_gap_pp']+EW['did_onset_cohortFE_present_expost']['diff_in_gaps_pp']:+.2f}$ & {dd(EW['did_onset_cohortFE_present_expost'])} \\
\bottomrule\end{{tabular}}}}\par\vspace{{3pt}}
\begin{{minipage}}{{\textwidth}}\footnotesize \emph{{Notes.}} Appointment-direction pairs formed 2004--2022 ($N={EW['n_rows']:,}$) so that the three-year post-formation window is fully observed.
Rates and gaps are measured in a fixed three-year window ending the day before formation and a fixed three-year window starting on the formation date. The difference column is the
coefficient on exposed $\times$ post in a stacked regression of the two windows with two-way (origin, destination) clustered standard errors; *** $|t|>2.58$, ** $|t|>1.96$, * $|t|>1.645$.
In Panel C$'$ the filing outcome is confounded by timing, because an origin restatement filed after the tie and a destination filing after the tie respond to the same calendar-time waves;
the onset rows are the informative ones. Panel C requires that the director's seat overlapped the misstated period; Panel C$''$ requires that it covered the filing date, so that the director
witnessed the discovery and correction rather than only the misstated reporting. Panel D re-estimates the onset change for each definition with formation-year effects and formation-year $\times$ post effects (the post
main effect is included), because the share of exposed pairs and the onset base rate both vary across formation years; the first column of Panel D is the adjusted pre-formation gap. Among destination firms, {pc(dp['exposed'])}\% of those appointing a director from a prior-restatement firm had themselves restated before the tie,
against {pc(dp['clean'])}\% of those appointing a director from a clean firm. Table~\ref{{tab:prepost_robust}} reports the robustness of Panel A.\end{{minipage}}\end{{table}}""")

# ── T4b robustness of the before/after comparison (appendix) ──────────────────────────
def fe_row(lbl, d_):
    pre_ = f"${d_['pre_gap_pp']:+.2f}$" + (f" ({d_['se_pre_pp']:.2f})" if 'se_pre_pp' in d_ else "")
    return f"{lbl} & {pre_} & ${d_['pre_gap_pp']+d_['diff_in_gaps_pp']:+.2f}$ & {dd(d_)} \\\\\n"
cs_rows = ""
if CS:
    cs_rows += fe_row("Filed, formation-year effects $+$ same-industry indicator (2-digit SIC) and its interaction with post", CS["did_file_cohortFE_sameSIC"])
    cs_rows += fe_row("Misstated period begins, same controls", CS["did_onset_cohortFE_sameSIC"])
    cs_rows += fe_row("Filed, formation-year effects $+$ destination-industry $\\times$ post effects", CS["did_file_cohortFE_dstIndustryXpost"])
    cs_rows += fe_row("Misstated period begins, same controls", CS["did_onset_cohortFE_dstIndustryXpost"])
    if "did_file_cohortFE_sameSIC_sameState" in CS:
        cs_rows += fe_row("Filed, formation-year effects $+$ same-industry and same-state indicators", CS["did_file_cohortFE_sameSIC_sameState"])
        cs_rows += fe_row("Misstated period begins, same controls", CS["did_onset_cohortFE_sameSIC_sameState"])
    if "historical_sic" in CS:
        HS = CS["historical_sic"]
        cs_rows += fe_row(f"Filed, formation-year effects $+$ same-industry indicator from historical Compustat SIC at formation ($N={HS['n_rows_both_available']:,}$)", HS["did_file_cohortFE_sameSIChist"])
        cs_rows += fe_row("Misstated period begins, same controls", HS["did_onset_cohortFE_sameSIChist"])
        cs_rows += fe_row("Misstated period begins, destination historical industry $\\times$ post effects", HS["did_onset_cohortFE_dstHistIndustryXpost"])
        cs_rows += fe_row("Misstated period begins, recent exposure, destination historical industry $\\times$ post effects", HS["did_onset_cohortFE_dstHistIndustryXpost_recent"])
hs_note = ((f" Historical industry uses the Compustat SIC code of the fiscal year of formation (or the nearest earlier year within three) for the {pc(CS['historical_sic']['share_of_pairs'])}\\% of pairs in which both firms are Compustat-linked with a code; "
            f"{pc(CS['historical_sic']['share_same_sic2_hist'])}\\% of these pairs share a historical 2-digit industry, the historical and current same-industry indicators agree for {pc(CS['historical_sic']['agreement_with_current_same_sic2'])}\\% of them, and the destination's historical 2-digit code differs from its current EDGAR code in {pc(CS['historical_sic']['share_dst_2digit_differs_current_vs_historical'])}\\% of pairs.") if CS and "historical_sic" in CS else "")
cs_note = ((" " + pc(CS['share_same_sic2']) + "\\% of pairs share a 2-digit SIC industry (EDGAR)" + ((" and " + pc(CS['share_same_state']) + "\\% a headquarters state (EDGAR business address)") if 'share_same_state' in CS else "") +
           "; exposed shares are " + pc(CS['exposed_share_same_sic2']) + "\\% among same-industry and " + pc(CS['exposed_share_diff_sic2']) + "\\% among other pairs" +
           ((", and " + pc(CS['exposed_share_same_state']) + "\\% among same-state and " + pc(CS['exposed_share_diff_state']) + "\\% among other pairs") if 'exposed_share_same_state' in CS else "") + ".") if CS else "") + hs_note
w("tab_prepost_robust.tex", rf"""\begin{{table}}[htbp]\centering\renewcommand{{\arraystretch}}{{0.95}}
\caption{{Robustness of the before--after comparison: additional controls and the narrower exposure definitions (exposure as in Panel A of Table~\ref{{tab:prepost}} unless stated).}}
\label{{tab:prepost_robust}}
\footnotesize\adjustbox{{max width=\textwidth}}{{\begin{{tabular}}{{p{{8.4cm}}ccc}}\toprule
 & 3 years before & 3 years after & Difference (after $-$ before) \\
Specification & gap, pp (SE) & gap, pp & pp (SE) \\\midrule
\multicolumn{{4}}{{@{{}}p{{15.8cm}}@{{}}}}{{\emph{{Panel A. Additional controls in the stacked regression ($N={EW['n_rows']:,}$ unless stated)}}}} \\
{fe_row("Filed, formation-year and year $\\times$ post fixed effects", EW["did_file_cohortFE"])}{fe_row("Misstated period begins, formation-year and year $\\times$ post fixed effects", EW["did_onset_cohortFE"])}{fe_row("Filed, formation-year effects $+$ shared-auditor indicator and its interaction with post", EW["did_file_cohortFE_sameauditor"])}Misstated period begins, interlocks formed by 2019 ($N={o19['n_rows']:,}$) & {g(o19['pre_onset'])} & {g(o19['post_onset'])} & {dd(o19['did_onset'])} \\
{fe_row("Filed, formation-year effects $+$ both firms' years in the feed (and their interactions with post)", EW["did_file_cohortFE_feedage"])}{fe_row("Misstated period begins, same controls", EW["did_onset_cohortFE_feedage"])}{fe_row("Filed, same $+$ destination log assets (destinations with Compustat assets in the formation year, $N=" + f"{EW['did_file_cohortFE_feedage_dstsize']['n_rows']:,}" + "$)", EW["did_file_cohortFE_feedage_dstsize"])}{fe_row("Misstated period begins, same controls", EW["did_onset_cohortFE_feedage_dstsize"])}{cs_rows}{fe_row("Filed, formation-year effects $+$ destination's earlier exposed and earlier inbound ties (logs, and interactions with post)", PX["did_file_cohortFE_priorstock"])}{fe_row("Misstated period begins, same controls", PX["did_onset_cohortFE_priorstock"])}\midrule
\multicolumn{{4}}{{@{{}}p{{15.8cm}}@{{}}}}{{\emph{{Panel A$'$. The narrower exposure definitions with formation-year and year $\times$ post effects}}}} \\
{fe_row("Recent exposure (origin restated within 5 years): filed", EW["did_file_cohortFE_recent"])}{fe_row("Recent exposure: misstated period begins", EW["did_onset_cohortFE_recent"])}{fe_row("Recent exposure: misstated period begins, formation-year $\\times$ destination-size-decile cells", EW["did_onset_cellFE_recent"])}{fe_row("Recent exposure: misstated period begins, $+$ feed ages and destination log assets (Compustat destinations)", EW["did_onset_cohortFE_recent_dstsize"])}{fe_row("Director present during the origin's misstatement: misstated period begins", EW["did_onset_cohortFE_present"])}{fe_row("Director present when the restatement was filed: misstated period begins", EW["did_onset_cohortFE_present_at_filing"])}{fe_row("Any exposure: misstated period begins, formation-year $\\times$ destination-size-decile cells", EW["did_onset_cellFE_any"])}\midrule
Severe exposure (origin filing flagged for fraud or SEC investigation): filed, unadjusted & ${EW['did_file_severe_source']['pre_gap_pp']:+.2f}$ & ${EW['did_file_severe_source']['pre_gap_pp']+EW['did_file_severe_source']['diff_in_gaps_pp']:+.2f}$ & {dd(EW['did_file_severe_source'])} \\
\bottomrule\end{{tabular}}}}\par\vspace{{3pt}}
\begin{{minipage}}{{\textwidth}}\scriptsize \emph{{Notes.}} Formation-year effects are dummies for each formation year interacted with the post indicator. Because the median restatement is filed
{R['median_years_file_minus_begin']:.1f} years after the misstated period begins, onsets late in the post-formation window of the most recent cohorts may not yet have been filed; the row restricted to
interlocks formed by 2019 removes this truncation. {pc(EW['share_same_current_auditor'])}\% of pairs share the auditor recorded as current for both registrants in the Audit Analytics auditor table (one auditor per registrant as of the extract, not an auditor history). Exposed destinations have been in the feed longer than clean ones
({OB['mean_dst_feed_age_exposed']:.1f} against {OB['mean_dst_feed_age_clean']:.1f} years at formation) and are larger (mean log assets {EW['mean_dst_size_exposed_clean'][0]:.2f} against {EW['mean_dst_size_exposed_clean'][1]:.2f} among Compustat-linked
destinations); the rows with feed-age and size controls add these variables and their interactions with post, and the subsample of Table~\ref{{tab:prepost_robust2}} that requires both firms in the feed for three years removes pairs whose pre-formation window
precedes either firm's entry into the feed. Standard errors in parentheses; stars as in Table~\ref{{tab:prepost}}.\end{{minipage}}\end{{table}}""")

w("tab_prepost_robust2.tex", rf"""\begin{{table}}[htbp]\centering\renewcommand{{\arraystretch}}{{0.95}}
\caption{{Robustness of the before--after comparison: subsamples, weighting, and clustering (exposure as in Panel A of Table~\ref{{tab:prepost}} unless stated).}}
\label{{tab:prepost_robust2}}
\footnotesize\adjustbox{{max width=\textwidth}}{{\begin{{tabular}}{{p{{7.0cm}}ccc}}\toprule
 & 3 years before & 3 years after & Difference (after $-$ before) \\
Specification & gap, pp (SE) & gap, pp & pp (SE) \\\midrule
\multicolumn{{4}}{{@{{}}p{{15.8cm}}@{{}}}}{{\emph{{Panel A. Subsamples}}}} \\
Filed, destination's first exposed inbound tie ($N={PX['first_exposed_tie']['n_rows']:,}$) & {g(PX['first_exposed_tie']['pre_file'])} & {g(PX['first_exposed_tie']['post_file'])} & {dd(PX['first_exposed_tie']['did_file'])} \\
Misstated period begins, destination's first exposed inbound tie & {g(PX['first_exposed_tie']['pre_onset'])} & {g(PX['first_exposed_tie']['post_onset'])} & {dd(PX['first_exposed_tie']['did_onset'])} \\
Filed, clean origins restricted to those not restating within 3 years after the tie ($N={SCO['n_rows']:,}$) & {g(SCO['pre_file'])} & {g(SCO['post_file'])} & {dd(SCO['did_file'])} \\
Misstated period begins, clean origins not restating within 3 years after the tie & {g(SCO['pre_onset'])} & {g(SCO['post_onset'])} & {dd(SCO['did_onset'])} \\
Misstated period begins, same restriction, recent exposure & {g(SCO['recent_pre_onset'])} & {g(SCO['recent_post_onset'])} & {dd(SCO['recent_did_onset'])} \\
{fe_row('Misstated period begins, same restriction, formation-year effects', SCO['did_onset_cohortFE'])}{fe_row('Misstated period begins, same restriction, recent exposure, formation-year effects', SCO['recent_did_onset_cohortFE'])}
Filed, interlocks formed 2008--2022 ($N={E08['n_rows']:,}$) & {g(E08['pre_file'])} & {g(E08['post_file'])} & {dd(E08['did_file'])} \\
{fe_row('Misstated period begins, formed by 2019, formation-year effects', E19['did_onset_cohortFE'])}Misstated period begins, formed by 2019, recent exposure & {g(E19['recent_pre_onset'])} & {g(E19['recent_post_onset'])} & {dd(E19['recent_did_onset'])} \\
{fe_row('Misstated period begins, formed by 2019, recent exposure, formation-year effects', E19['recent_did_onset_cohortFE'])}Filed, destination records a board event after the post-formation window ($N={ATW['n_rows']:,}$; {pc(ATW['share_dst_active_after_window'])}\% of pairs) & {g(ATW['pre_file'])} & {g(ATW['post_file'])} & {dd(ATW['did_file'])} \\
Misstated period begins, destination records a board event after the window & {g(ATW['pre_onset'])} & {g(ATW['post_onset'])} & {dd(ATW['did_onset'])} \\
{fe_row('Misstated period begins, same restriction, formation-year effects', ATW['did_onset_cohortFE'])}{fe_row('Misstated period begins, same restriction, recent exposure, formation-year effects', ATW['recent_did_onset_cohortFE'])}
Misstated period begins, interlocks formed 2008--2022 & {g(E08['pre_onset'])} & {g(E08['post_onset'])} & {dd(E08['did_onset'])} \\
Filed, both firms in the feed for at least 3 years before formation ($N={OB['n_rows']:,}$, {pc(OB['share_rows'])}\% of pairs) & {g(OB['pre_file'])} & {g(OB['post_file'])} & {dd(OB['did_file'])} \\
Misstated period begins, same restriction & {g(OB['pre_onset'])} & {g(OB['post_onset'])} & {dd(OB['did_onset'])} \\
\midrule
\multicolumn{{4}}{{@{{}}p{{15.8cm}}@{{}}}}{{\emph{{Panel B. Weighting and clustering}}}} \\
Filed, each appointment weighted equally ({AW['n_appointments']:,} appointments; {AW['mean_pairs_per_appointment']:.2f} pairs per appointment) & ${AW['pre_file']['diff_pp']:+.2f}$ ({AW['pre_file']['se_pp']:.2f}) & ${AW['post_file']['diff_pp']:+.2f}$ ({AW['post_file']['se_pp']:.2f}) & ${AW['did_file']['diff_in_gaps_pp']:+.2f}$ ({AW['did_file']['se_pp']:.2f}) \\
Misstated period begins, same restriction & ${AW['pre_onset']['diff_pp']:+.2f}$ ({AW['pre_onset']['se_pp']:.2f}) & ${AW['post_onset']['diff_pp']:+.2f}$ ({AW['post_onset']['se_pp']:.2f}) & ${AW['did_onset']['diff_in_gaps_pp']:+.2f}$ ({AW['did_onset']['se_pp']:.2f}) \\
Filed, standard errors clustered by director ({EW['director_clustered']['n_directors']:,} directors) & ${EW['director_clustered']['pre_file']['diff_pp']:+.2f}$ ({EW['director_clustered']['pre_file']['se_pp_director_clustered']:.2f}) & ${EW['director_clustered']['post_file']['diff_pp']:+.2f}$ ({EW['director_clustered']['post_file']['se_pp_director_clustered']:.2f}) & ${EW['director_clustered']['did_file']['diff_in_gaps_pp']:+.2f}$ ({EW['director_clustered']['did_file']['se_pp_director_clustered']:.2f}) \\
Misstated period begins, director-clustered standard errors & ${EW['director_clustered']['pre_onset']['diff_pp']:+.2f}$ ({EW['director_clustered']['pre_onset']['se_pp_director_clustered']:.2f}) & ${EW['director_clustered']['post_onset']['diff_pp']:+.2f}$ ({EW['director_clustered']['post_onset']['se_pp_director_clustered']:.2f}) & ${EW['director_clustered']['did_onset']['diff_in_gaps_pp']:+.2f}$ ({EW['director_clustered']['did_onset']['se_pp_director_clustered']:.2f}) \\
\bottomrule\end{{tabular}}}}\par\vspace{{3pt}}
\begin{{minipage}}{{\textwidth}}\scriptsize \emph{{Notes.}} Specifications as in Table~\ref{{tab:prepost_robust}}, without the additional controls unless stated.{cs_note} An appointment that gives the director a seat on a board while holding several other seats creates one pair per existing seat;
The destination's earlier exposed and earlier inbound ties count appointment-direction pairs formed at the same destination before the focal one ({pc(PX['share_pairs_with_prior_exposed_tie'])}\% of pairs have at least one earlier exposed tie); the first-exposed-tie subsample keeps pairs with none. The strict-clean rows drop the {pc(SCO['share_clean_origins_filing_within_3y_after'])}\% of clean origins that file a restatement within three years after the tie, so that a common shock hitting both firms cannot place an origin in the clean group. The attrition rows keep pairs whose destination records a board event in the feed after the three-year post-formation window ({pc(ATW['share_exposed'])}\% of exposed and {pc(ATW['share_clean'])}\% of clean pairs), so that differential exit from the feed cannot narrow the post-formation gap. Panel B weights each appointment equally and, in the last two rows, clusters by director instead of by the two firms (at most {EW['director_clustered']['max_pairs_per_director']} pairs per director, more than the 15 that six seats allow because a director who leaves and returns creates a new lifecycle). Standard errors in parentheses; stars as in Table~\ref{{tab:prepost}}.\end{{minipage}}\end{{table}}""")

# ── T5 destinations without prior history ───────────────────────────────────────────────
def rt(d_): return f"{pc(d_['exposed_rate'])}\\% / {pc(d_['clean_rate'])}\\%"
def crow(lbl, key):
    c = CL[key]; return f"{lbl} & {c['n_exposed']:,} & {g(c['post_file'])} & {g(c['post_onset'])} & {g(c['post_onset_yearFE'])} & {rt(c['post_onset'])} \\\\\n"
cs_split_rows = ""
if CS:
    for lbl_, k_ in (("Origin and destination in the same 2-digit SIC industry", "clean_post_onset_recent5y_same_sic2"), ("Different industries", "clean_post_onset_recent5y_diff_sic2"),
                     ("Headquarters in the same state", "clean_post_onset_recent5y_same_state"), ("Different states", "clean_post_onset_recent5y_diff_state")):
        if k_ in CS: cs_split_rows += f"{lbl_} & {CS[k_]['n_rows']:,} & & {g(CS[k_])} & & {rt(CS[k_])} \\\\\n"
NY = K1["never_restated_yearFE"]
w("tab_clean.tex", rf"""\begin{{table}}[htbp]\centering
\caption{{Destinations with no restatement history at the time of the appointment: the post-formation gap under three exposure definitions.}}
\label{{tab:clean}}
\footnotesize\setlength{{\tabcolsep}}{{3pt}}\adjustbox{{max width=\textwidth}}{{\begin{{tabular}}{{@{{}}p{{5.0cm}}ccccc@{{}}}}\toprule
 & & \multicolumn{{3}}{{c}}{{Gap in the 3 years after formation, pp (SE)}} & Onset rate, \\
Exposure definition & Exposed pairs & Filed & Onset & Onset, year-adjusted & exposed / clean \\\midrule
\multicolumn{{6}}{{@{{}}p{{15.8cm}}@{{}}}}{{\emph{{Panel A. Destination never filed a restatement before the tie ($N={NV['n_rows']:,}$ pairs, {pc(NV['share_of_rows'])}\% of the sample)}}}} \\
Origin restated at any time before the tie & {NV['post_file_3y']['n_exposed_rows']:,} & {g(NV['post_file_3y'])} & {g(NV['post_onset_3y'])} & {g(NY['post_onset_yearFE_exposed'])} & {rt(NV['post_onset_3y'])} \\
\quad pre-formation onset gap, same pairs & & & {g(NV['pre_onset_3y'])} & & \\
\quad change, after $-$ before & & & {dd(NV['did_onset'])} & & \\
Origin restated within 5 years before the tie & {R5['never_restated_post_file']['n_exposed_rows']:,} & {g(R5['never_restated_post_file'])} & {g(R5['never_restated_post_onset'])} & {g(NY['post_onset_yearFE_exposed5y'])} & {rt(R5['never_restated_post_onset'])} \\
\quad pre-formation onset gap, same pairs & & & {g(R5['never_restated_pre_onset'])} & & \\
\quad change, after $-$ before & & & {dd(R5['never_restated_did_onset'])} & {dd(R5['never_restated_did_onset_cohortFE'])} & \\
Director present at the origin during the misstatement & {SV['never_restated_post_file']['n_exposed_rows']:,} & {g(SV['never_restated_post_file'])} & {g(SV['never_restated_post_onset'])} & {g(NY['post_onset_yearFE_exposed_served'])} & {rt(SV['never_restated_post_onset'])} \\
\quad pre-formation onset gap, same pairs & & & {g(SV['never_restated_pre_onset'])} & & \\
\quad change, after $-$ before & & & {dd(SV['never_restated_did_onset'])} & & \\
Director present at the origin when the restatement was filed & {PF['never_restated_post_file']['n_exposed_rows']:,} & {g(PF['never_restated_post_file'])} & {g(PF['never_restated_post_onset'])} & {g(NY['post_onset_yearFE_exposed_present_at_filing'])} & {rt(PF['never_restated_post_onset'])} \\
\quad pre-formation onset gap, same pairs & & & {g(PF['never_restated_pre_onset'])} & & \\
\quad change, after $-$ before & & & {dd(PF['never_restated_did_onset'])} & & \\
\midrule
\multicolumn{{6}}{{@{{}}p{{15.8cm}}@{{}}}}{{\emph{{Panel B. Destination never filed a restatement and no misstated period had begun before the tie ($N={CL['n_rows']:,}$ pairs, {pc(CL['share_of_rows'])}\%)}}}} \\
{crow("Origin restated at any time before the tie", "exposed")}{crow("Origin restated within 5 years before the tie", "exposed5y")}{crow("Director present at the origin during the misstatement", "exposed_served")}{crow("Director present at the origin when the restatement was filed", "exposed_present_at_filing")}\midrule
\multicolumn{{6}}{{@{{}}p{{15.8cm}}@{{}}}}{{\emph{{Panel C. Where the recent-exposure gap of Panel B sits (exposure $=$ origin restated within 5 years; column 2 is the number of pairs; the pre/post treatment by size tercile is in Appendix Table~\ref{{tab:sizetercile}})}}}} \\
Destination in the smallest third by assets (Compustat-linked; log assets $\le {RH['size_tercile_cutpoints_log_assets'][0]:.2f}$) & {RH['size_bottom_tercile']['n_rows']:,} & {g(RH['size_bottom_tercile']['post_file_recent5y'])} & {g(RH['size_bottom_tercile']['post_onset_recent5y'])} & & {rt(RH['size_bottom_tercile']['post_onset_recent5y'])} \\
Destination in the middle third & {RH['size_middle_tercile']['n_rows']:,} & {g(RH['size_middle_tercile']['post_file_recent5y'])} & {g(RH['size_middle_tercile']['post_onset_recent5y'])} & & {rt(RH['size_middle_tercile']['post_onset_recent5y'])} \\
Destination in the largest third (log assets $> {RH['size_tercile_cutpoints_log_assets'][1]:.2f}$) & {RH['size_top_tercile']['n_rows']:,} & {g(RH['size_top_tercile']['post_file_recent5y'])} & {g(RH['size_top_tercile']['post_onset_recent5y'])} & & {rt(RH['size_top_tercile']['post_onset_recent5y'])} \\
Destination in the feed for less than 3 years & {RH['dst_in_feed_lt3y']['n_rows']:,} & {g(RH['dst_in_feed_lt3y']['post_file_recent5y'])} & {g(RH['dst_in_feed_lt3y']['post_onset_recent5y'])} & & {rt(RH['dst_in_feed_lt3y']['post_onset_recent5y'])} \\
Destination in the feed for 3 years or more & {RH['dst_in_feed_ge3y']['n_rows']:,} & {g(RH['dst_in_feed_ge3y']['post_file_recent5y'])} & {g(RH['dst_in_feed_ge3y']['post_onset_recent5y'])} & & {rt(RH['dst_in_feed_ge3y']['post_onset_recent5y'])} \\
{cs_split_rows}\bottomrule\end{{tabular}}}}\par\vspace{{3pt}}
\begin{{minipage}}{{\textwidth}}\scriptsize \emph{{Notes.}} Subsamples of the pairs in Table~\ref{{tab:prepost}}; ``onset'' is a misstated period that begins in the window; ``year-adjusted'' adds formation-year effects (for the change rows, formation-year $\times$ post effects). In Panel B the gap in severe restatements filed after formation is {g(CL['exposed']['post_severe'])} for any exposure, {g(CL['exposed5y']['post_severe'])} for recent exposure, {g(CL['exposed_served']['post_severe'])} for directors present during the misstatement and {g(CL['exposed_present_at_filing']['post_severe'])} for directors present at the filing. Panel A is defined on information available at the appointment. In it the pre-formation \emph{{filing}} gap is zero by construction; the pre-formation
onset gap is not, because a misstated period can begin before the tie and be restated after it; the change rows are the coefficient on exposed $\times$ post in the stacked regression of Table~\ref{{tab:prepost}} on these pairs.
Panel B also removes pairs with a misstated period that began before the tie, which is known only after the fact, so it is a retrospective selection in which any post-formation gap
concerns misstatements that began after the director arrived; ``year-adjusted'' adds formation-year effects to the onset gap, because the exposure definitions and the onset base rate both vary across formation years. Panel C splits the Panel B pairs by destination characteristics and by whether the two firms share an industry or a headquarters state (EDGAR), and reports the gap for recent exposure (origin restated within 5 years); the last column gives the onset rates of exposed and clean destinations. Two-way (origin, destination) clustered standard errors.\end{{minipage}}\end{{table}}""")

# ── T6 cohorts ─────────────────────────────────────────────────────────────────────────
bc = a["equal_window_by_cohort"]
rows = "".join(f"{k} & {pc(v['exposed_share'])}\\% & ${v['pre_file_pp']:+.2f}$ & ${v['post_file_pp']:+.2f}$ & ${v['did']['diff_in_gaps_pp']:+.2f}$ ({v['did']['se_pp']:.2f}){star(v['did']['t'])} & {v['n_rows']:,} \\\\\n" for k, v in bc.items())
rows3 = "".join(f"{k} & {g(v['pre_onset_3y'])} & {g(v['file_3y'])} & {g(v['onset_3y'])} \\\\\n" for k, v in K3.items() if "file_3y" in v)
tr = K3["linear_trend_in_gap_per_year_file_3y"]
w("tab_cohort.tex", rf"""\begin{{table}}[htbp]\centering\footnotesize
\caption{{By formation cohort: exposed share, the gap before and after formation, and the post-formation gap by outcome (three-year windows).}}
\label{{tab:cohort}}
\adjustbox{{max width=\textwidth}}{{\begin{{tabular}}{{p{{2.4cm}}ccccr}}\toprule
 & & & & After $-$ before, & \\
Formation cohort & Exposed share & Gap before, pp & Gap after, pp & pp (SE) & $N$ pairs \\\midrule
{rows}\midrule
 & Pre gap, & Post gap, & \multicolumn{{2}}{{c}}{{Post gap,}} & \\
 & onset (SE) & filed (SE) & \multicolumn{{2}}{{c}}{{onset (SE)}} & \\
{rows3}\bottomrule\end{{tabular}}}}\par\vspace{{3pt}}
\begin{{minipage}}{{\textwidth}}\footnotesize \emph{{Notes.}} Outcome in the upper block: restatement filed; exposure as in Panel A of Table~\ref{{tab:prepost}}. Linear trend in the post-formation filed gap across formation years (interaction of
exposed with formation year, two-way clustered): ${tr['pp_per_year']:+.3f}$ pp per year (SE {tr['se_pp']:.3f}). The 2004--2007 cohort is affected by the left-censoring of tenure spells described in
Table~\ref{{tab:network}}; Panel A of Table~\ref{{tab:prepost_robust2}} repeats the pooled estimates without it. Onsets in the 2018--2022 cohort are affected by the filing lag (Table~\ref{{tab:prepost_robust}}, note). Other notes as in Table~\ref{{tab:prepost}}.\end{{minipage}}\end{{table}}""")

# ── Figure A.1 (prepost by cohort) with a generated caption (2026-10-15, r12: pooled vs cohort-weighted change) ─────────
_wsum = sum(v["n_rows"] for v in bc.values()); _wmean = sum(v["did"]["diff_in_gaps_pp"] * v["n_rows"] for v in bc.values()) / _wsum
_first = next(iter(bc)); _firstd = bc[_first]["did"]["diff_in_gaps_pp"]
w("fig_prepost.tex", rf"""\begin{{figure}}[htbp]\centering
\includegraphics[width=0.85\textwidth]{{figures_k/figK1_prepost.pdf}}
\caption{{The exposed--clean gap in destination restatement filings in the three years before and after interlock formation, by formation cohort and pooled (primary sample, appointment direction; unadjusted). $\Delta$ is the post-minus-pre difference from the stacked regression (pooled: Table~\ref{{tab:prepost}}, Panel A; by cohort: Table~\ref{{tab:cohort}}), with the pooled standard error in parentheses (cohort standard errors in Table~\ref{{tab:cohort}}); the bars show the separate pre- and post-formation gaps, not $\Delta$. The pooled $\Delta$ of ${EW['did_file']['diff_in_gaps_pp']:+.1f}$ is not the pair-weighted mean of the cohort $\Delta$s (${_wmean:+.1f}$), because exposure shares and base rates differ across cohorts; Panel D of Table~\ref{{tab:prepost}} holds cohorts fixed. The {_first} bar ($\Delta={_firstd:+.1f}$) is affected by the left-censoring of tenure spells at the start of the feed (Table~\ref{{tab:network}}); Panel A of Table~\ref{{tab:prepost_robust2}} repeats the pooled estimates without that cohort.}}
\label{{fig:prepost}}\end{{figure}}""")

# ── T7 panel (rows = specifications) ───────────────────────────────────────────────────
M = K2["models"]; S = K2["estimation_sample"]; SC = K2["between_firm_screening"]; SO = K2["between_firm_screening_out_of_sample"]
specs = [("contemporaneous_file_no_turnover (old timing, old controls)", "(1$'$) Filed in $t$", "formed in $t$", "no"),
         ("contemporaneous_file (old timing)", "(1) Filed in $t$", "formed in $t$", "yes"),
         ("lag1_file_no_turnover_control", "(2) Filed in $t$", "formed in $t-1$", "no"),
         ("lag1_file", "(3) Filed in $t$", "formed in $t-1$", "yes"),
         ("lag1_file_plus_prior", "(4) Filed in $t$", "formed in $t-1$", "yes$^{\\dagger}$"),
         ("lag1_onset", "(5) Onset in $t$", "formed in $t-1$", "yes"),
         ("lag1_onset_plus_prior", "(5$^{\\dagger}$) Onset in $t$", "formed in $t-1$", "yes$^{\\dagger}$"),
         ("boy_stock_file", "(6) Filed in $t$", "alive on 1 Jan $t$", "yes"),
         ("boy_stock_onset", "(7) Onset in $t$", "alive on 1 Jan $t$", "yes"),
         ("lag1_file_inbound_only", "(8) Filed in $t$", "inbound, formed in $t-1$", "yes$^{\\ddagger}$"),
         ("lag1_onset_inbound_only", "(9) Onset in $t$", "inbound, formed in $t-1$", "yes$^{\\ddagger}$"),
         ("lag1_file_exposed5y", "(10) Filed in $t$", "recent, formed in $t-1$", "yes"),
         ("lag1_file_exposed5y_plus_prior", "(11) Filed in $t$", "recent, formed in $t-1$", "yes$^{\\dagger}$"),
         ("lag1_onset_exposed5y", "(12) Onset in $t$", "recent, formed in $t-1$", "yes"),
         ("lag1_onset_exposed5y_plus_prior", "(13) Onset in $t$", "recent, formed in $t-1$", "yes$^{\\dagger}$"),
         ("boy_stock_onset_exposed5y", "(14) Onset in $t$", "recent, alive on 1 Jan $t$", "yes"),
         ("boy_stock_onset_exposed5y_plus_prior", "(14$^{\\dagger}$) Onset in $t$", "recent, alive on 1 Jan $t$", "yes$^{\\dagger}$"),
         ("lag1_file_inbound_recent5y_plus_prior", "(15) Filed in $t$", "inbound recent, formed in $t-1$", "yes$^{\\dagger\\ddagger}$"),
         ("lag1_onset_inbound_recent5y_plus_prior", "(16) Onset in $t$", "inbound recent, formed in $t-1$", "yes$^{\\dagger\\ddagger}$")]
if "lag1_file_excl_blank_check" in M and S.get("blank_check_firms_in_panel", 0) > 0:
    specs += [("lag1_file_excl_blank_check", "(17) Filed in $t$$^{\\S}$", "formed in $t-1$", "yes"),
              ("lag1_onset_excl_blank_check", "(18) Onset in $t$$^{\\S}$", "formed in $t-1$", "yes")]
def cc(m): c = m["exposure_coef"]; return f"${100*c['coef']:+.3f}${star(c['t'])} ({c['t']:.2f})"
def tc(m, has):
    if has == "no" or m["turnover_coef"] is None: return "---"
    c = m["turnover_coef"]; return f"${100*c['coef']:+.3f}${star(c['t'])} ({c['t']:.2f})"
rows = "".join(f"{lab} & {exp} & {has} & {cc(M[k])} & {tc(M[k], has)} & ${100*M[k]['oster']['beta_uncontrolled']:+.3f}$ & {M[k]['r2_within']:.5f} \\\\\n" for k, lab, exp, has in specs)
xbc = (f" Rows (17)--(18) drop the {S.get('blank_check_firms_in_panel', 0):,} firms flagged as blank-check companies ($N={S.get('firm_years_excl_blank_check', 0):,}$ firm-years)." if S.get("blank_check_firms_in_panel", 0) > 0 and "lag1_file_excl_blank_check" in M else "")
w("tab_panel.tex", rf"""\begin{{table}}[htbp]\centering\small\renewcommand{{\arraystretch}}{{0.95}}
\caption{{Firm-year panel: exposure that precedes the outcome year. Two-way (firm and year) fixed effects, firm-clustered standard errors.}}
\label{{tab:panel}}
\footnotesize\setlength{{\tabcolsep}}{{3pt}}\adjustbox{{max width=\textwidth}}{{\begin{{tabular}}{{@{{}}p{{2.1cm}}p{{5.2cm}}p{{1.5cm}}cccc@{{}}}}\toprule
 & Exposed interlocks & New-ties & Exposed, & All new, & No & Within \\
Outcome & measured on & control & pp ($t$) & pp ($t$) & controls & $R^2$ \\\midrule
{rows}\bottomrule\end{{tabular}}}}\par\vspace{{3pt}}
\begin{{minipage}}{{\textwidth}}\scriptsize \emph{{Notes.}} $N={S['firm_years']:,}$ firm-years, {S['firms']:,} firms, {S['years'][0]}--{S['years'][1]}; annual rate of restatement filings {pc(S['rate_file'],2)}\%,
of misstated-period onsets {pc(S['rate_onset'],2)}\%. ``Exposed interlocks'' counts new (or alive) interlocks whose other firm had filed a restatement before the tie formed;
``all new interlocks'' counts every new (or alive) interlock of the firm in the same window, whether created by a director joining the focal board (inbound) or by one of its
directors joining another board (outbound). Rows (8)--(9) use inbound exposed interlocks only and control separately for inbound new interlocks, outbound new interlocks and outbound
exposed interlocks ($^{{\ddagger}}$; the coefficient shown in the fifth column is on inbound new interlocks); rows (15)--(16) do the same for recent exposure with the own-history indicator. ``Recent exposure'' counts interlocks whose other firm filed a restatement within the five years before the tie formed. All rows include lagged degree and approximate betweenness,
log assets, leverage and ROA; $^{{\dagger}}$ rows (4), (11), (13), (15) and (16) add an indicator for any restatement filed before $t$. The two-way fixed effects are absorbed with the
within transformation of \texttt{{linearmodels.PanelOLS}}, which handles an unbalanced panel exactly rather than by one-pass double demeaning. Because the own-history indicator is a lagged outcome inside firm fixed effects,
rows (13) and (16) are re-estimated with year effects only (firm-clustered): recent exposure on onsets ${100*NK['lag1_onset_exposed5y_plus_prior_yearFE_only']['exposure_coef']['coef']:+.3f}$ ($t={NK['lag1_onset_exposed5y_plus_prior_yearFE_only']['exposure_coef']['t']:.2f}$), ${100*NK['lag1_onset_exposed5y_plus_pre2005_history_yearFE_only']['exposure_coef']['coef']:+.3f}$ ($t={NK['lag1_onset_exposed5y_plus_pre2005_history_yearFE_only']['exposure_coef']['t']:.2f}$) with history measured before 2005 instead,
and inbound recent exposure ${100*NK['lag1_onset_inbound_recent5y_plus_prior_yearFE_only']['exposure_coef']['coef']:+.3f}$ ($t={NK['lag1_onset_inbound_recent5y_plus_prior_yearFE_only']['exposure_coef']['t']:.2f}$); for filings the year-effects-only coefficient is ${100*NK['lag1_file_exposed5y_plus_prior_yearFE_only']['exposure_coef']['coef']:+.3f}$ ($t={NK['lag1_file_exposed5y_plus_prior_yearFE_only']['exposure_coef']['t']:.2f}$). Because onsets after 2020 may not yet have been
filed, the onset rows are also estimated on 2005--2020: rows (12), (13) and (16) give ${100*T20['lag1_onset_exposed5y']['exposure_coef']['coef']:+.3f}$ ($t={T20['lag1_onset_exposed5y']['exposure_coef']['t']:.2f}$), ${100*T20['lag1_onset_exposed5y_plus_prior']['exposure_coef']['coef']:+.3f}$ ($t={T20['lag1_onset_exposed5y_plus_prior']['exposure_coef']['t']:.2f}$) and ${100*T20['lag1_onset_inbound_recent5y_plus_prior']['exposure_coef']['coef']:+.3f}$ ($t={T20['lag1_onset_inbound_recent5y_plus_prior']['exposure_coef']['t']:.2f}$), and row (5) ${100*T20['lag1_onset']['exposure_coef']['coef']:+.3f}$ ($t={T20['lag1_onset']['exposure_coef']['t']:.2f}$). ``Exposed'' and ``all new'' coefficients are in percentage points
per interlock with firm-clustered $t$ statistics in parentheses; *** $|t|>2.58$, ** $|t|>1.96$, * $|t|>1.645$. Rows (1$'$) and (1) use the contemporaneous timing of static designs, in which exposure and outcome are measured in the same year, without and with the new-interlocks control; they are shown for comparison only. ``Exposure coefficient without
controls'' is from the same two-way fixed-effects model with no other regressors.{xbc} Between firms, a pooled logit of a restatement filed in $t$ on own prior restatement, size, leverage, ROA, lagged degree, all new
interlocks and year effects has an area under the ROC curve of {SC['restate_file']['baseline_auc']:.3f}; adding exposed interlocks formed in $t-1$ changes it by {SC['restate_file']['nexp_lag1']['auc_gain']:+.4f} (coefficient {SC['restate_file']['nexp_lag1']['coef']:+.3f}, $z={SC['restate_file']['nexp_lag1']['z']:.2f}$) and adding the stock of exposed
interlocks alive on 1 January by {SC['restate_file']['nexp_boy']['auc_gain']:+.4f} (coefficient {SC['restate_file']['nexp_boy']['coef']:+.3f}, $z={SC['restate_file']['nexp_boy']['z']:.2f}$); for onsets the baseline is {SC['restate_onset']['baseline_auc']:.3f} and the changes are {SC['restate_onset']['nexp_lag1']['auc_gain']:+.4f} and {SC['restate_onset']['nexp_boy']['auc_gain']:+.4f}. With recent-exposure counts instead, the changes are {SC['restate_file']['nexp5_lag1']['auc_gain']:+.4f} and {SC['restate_file']['nexp5_boy']['auc_gain']:+.4f} for filings
(coefficients {SC['restate_file']['nexp5_lag1']['coef']:+.3f}, $z={SC['restate_file']['nexp5_lag1']['z']:.2f}$, and {SC['restate_file']['nexp5_boy']['coef']:+.3f}, $z={SC['restate_file']['nexp5_boy']['z']:.2f}$) and {SC['restate_onset']['nexp5_lag1']['auc_gain']:+.4f} and {SC['restate_onset']['nexp5_boy']['auc_gain']:+.4f} for onsets (coefficients {SC['restate_onset']['nexp5_lag1']['coef']:+.3f}, $z={SC['restate_onset']['nexp5_lag1']['z']:.2f}$, and {SC['restate_onset']['nexp5_boy']['coef']:+.3f}, $z={SC['restate_onset']['nexp5_boy']['z']:.2f}$). These comparisons are in sample; fitting the same models without year effects on {SO['train_years'][0]}--{SO['train_years'][1]} and scoring {SO['test_years'][0]}--{SO['test_years'][1]} ($N={SO['n_test']:,}$ firm-years) gives a test-sample area under the curve of {SO['restate_file']['baseline_auc_test']:.3f} for filings and {SO['restate_onset']['baseline_auc_test']:.3f} for onsets, which the four exposure counts change by between {min(SO[d][k]['auc_gain_test'] for d in ('restate_file','restate_onset') for k in ('nexp_lag1','nexp_boy','nexp5_lag1','nexp5_boy')):+.4f} and {max(SO[d][k]['auc_gain_test'] for d in ('restate_file','restate_onset') for k in ('nexp_lag1','nexp_boy','nexp5_lag1','nexp5_boy')):+.4f}.
\end{{minipage}}\end{{table}}""")

# ── T8 community (P4) ──────────────────────────────────────────────────────────────────
def crow_(lbl, kf, ko):
    return f"{lbl} & ${K4[kf]['diff_pp']:+.2f}$ ({K4[kf]['se_pp']:.2f}) & ${K4[ko]['diff_pp']:+.2f}$ ({K4[ko]['se_pp']:.2f}) \\\\\n"
def irow_(lbl, kf, ko):
    return f"{lbl} & ${K4[kf]['exposed_x_cross_pp']:+.2f}$ ({K4[kf]['se_interaction_pp']:.2f}) & ${K4[ko]['exposed_x_cross_pp']:+.2f}$ ({K4[ko]['se_interaction_pp']:.2f}) \\\\\n"
w("tab_community.tex", rf"""\begin{{table}}[htbp]\centering\footnotesize
\caption{{Bridging versus within-community interlocks: destination rates among exposed ties and the exposed--clean gap by tie type.}}
\label{{tab:community}}
\adjustbox{{max width=\textwidth}}{{\begin{{tabular}}{{p{{9.0cm}}cc}}\toprule
 & Restatement filed & Misstated period begins \\\midrule
\multicolumn{{3}}{{@{{}}p{{15.8cm}}@{{}}}}{{\emph{{Panel A. Window: life of the interlock plus one year; all formation years ($N={K4['n_rows_with_community']:,}$ pairs, {K4['n_exposed_rows']:,} exposed)}}}} \\
Within-community exposed ties: destination rate ($N={K4['n_within_exposed']:,}$) & {pc(K4['within_exposed_rate'],2)}\% & \\
Cross-community exposed ties: destination rate ($N={K4['n_cross_exposed']:,}$) & {pc(K4['cross_exposed_rate'],2)}\% & \\
{crow_("Cross $-$ within among exposed ties, pp (SE)", "cross_minus_within_exposed_file", "cross_minus_within_exposed_onset")}Exposed $-$ clean gap along within-community ties, pp & ${K4['interaction_exposed_x_cross_file']['exposed_pp']:+.2f}$ & ${K4['interaction_exposed_x_cross_onset']['exposed_pp']:+.2f}$ \\
Cross-community main effect among clean ties, pp & ${K4['interaction_exposed_x_cross_file']['cross_pp']:+.2f}$ & ${K4['interaction_exposed_x_cross_onset']['cross_pp']:+.2f}$ \\
{irow_("Exposed $\\times$ cross-community interaction (gap along bridges $-$ gap within), pp (SE)", "interaction_exposed_x_cross_file", "interaction_exposed_x_cross_onset")}\midrule
\multicolumn{{3}}{{@{{}}p{{15.8cm}}@{{}}}}{{\emph{{Panel B. Window: three years after formation; interlocks formed 2004--2022 ($N={K4['interaction_exposed_x_cross_file_3y']['n_rows']:,}$ pairs, {K4['cross_minus_within_exposed_file_3y']['n_exposed']:,} exposed)}}}} \\
{crow_("Cross $-$ within among exposed ties, pp (SE)", "cross_minus_within_exposed_file_3y", "cross_minus_within_exposed_onset_3y")}{irow_("Exposed $\\times$ cross-community interaction, pp (SE)", "interaction_exposed_x_cross_file_3y", "interaction_exposed_x_cross_onset_3y")}\midrule
\multicolumn{{3}}{{@{{}}p{{15.8cm}}@{{}}}}{{\emph{{Panel C. Three-year window; interlocks formed 2012--2022 ($N={K4['interaction_exposed_x_cross_file_3y_2012plus']['n_rows']:,}$ pairs, {K4['cross_minus_within_exposed_file_3y_2012plus']['n_exposed']:,} exposed; {pc(K4['cross_minus_within_exposed_file_3y_2012plus']['share_cross_exposed'])}\% of exposed ties cross communities)}}}} \\
{crow_("Cross $-$ within among exposed ties, pp (SE)", "cross_minus_within_exposed_file_3y_2012plus", "cross_minus_within_exposed_onset_3y_2012plus")}{irow_("Exposed $\\times$ cross-community interaction, pp (SE)", "interaction_exposed_x_cross_file_3y_2012plus", "interaction_exposed_x_cross_onset_3y_2012plus")}\bottomrule\end{{tabular}}}}\par\vspace{{3pt}}
\begin{{minipage}}{{\textwidth}}\footnotesize \emph{{Notes.}} Appointment-direction pairs. Communities from Louvain detection on the network of the formation year; a tie is cross-community when origin and destination belong to
different communities in that year. Because cross-community ties are {pc(NET['2008']['cross_edge_ratio'])}\% of interlocks in 2008 and {pc(NET['2012']['cross_edge_ratio'])}\% in 2012 (Table~\ref{{tab:network}}), the comparison is
effectively identified from 2012 onward, which Panel C isolates; Panel B uses the fixed window because the life-plus-one-year window is truncated at the horizon for open ties and its length is
correlated with exposure (Table~\ref{{tab:survival}}). The interaction rows are from a linear probability model with exposed, cross-community and their product; two-way clustered standard errors.\end{{minipage}}\end{{table}}""")

# ── T9 survival (P5) ───────────────────────────────────────────────────────────────────
def hr(c): return f"{c['hr']:.3f} [{c['ci95'][0]:.3f}, {c['ci95'][1]:.3f}]"
def sv_rows(key, lbl):
    b = K5[key]
    out = (f"\\multicolumn{{4}}{{l}}{{\\emph{{{lbl}}}}} \\\\\n"
           f"Kaplan--Meier $S(5)$ / $S(10)$ / median years: exposed origin ($N={b['km_prior']['n']:,}$) & {b['km_prior']['S5']:.3f} / {b['km_prior']['S10']:.3f} / {b['km_prior']['median']} & & \\\\\n"
           f"\\quad clean origin ($N={b['km_clean']['n']:,}$) & {b['km_clean']['S5']:.3f} / {b['km_clean']['S10']:.3f} / {b['km_clean']['median']} & & log-rank $p={b['logrank_p']:.3f}$ \\\\\n")
    for lab, ck in (("any dissolution", "cox_any_dissolution"), ("destination seat ended", "cox_destination_seat_ended"), ("\\quad with covariates", "cox_destination_seat_ended_covariates"),
                    ("origin seat ended", "cox_origin_seat_ended"), ("\\quad with covariates", "cox_origin_seat_ended_covariates"),
                    ("\\quad origin seat ended, origin firm continues", "cox_origin_seat_ended_firm_continues"), ("\\quad origin seat ended, origin firm exits the feed", "cox_origin_seat_ended_firm_exit")):
        c = b[ck]; ph = f"PH $p={c['ph_test_p']:.3f}$" if isinstance(c.get("ph_test_p"), float) else ""
        out += f"Cox: {lab} & & {hr(c)} & $p={c['p']:.3f}$; {c['events']:,} events{'; ' + ph if ph else ''} \\\\\n"
    for lab, ck in (("destination seat ended, covariates", "cox_destination_seat_ended_piecewise"), ("origin seat ended, covariates", "cox_origin_seat_ended_piecewise")):
        pw = b[ck]
        for il, nm_ in (("0_2y", "years 0--2 after formation"), ("2_5y", "years 2--5"), ("5y_plus", "beyond year 5")):
            if "hr" in pw[il]:
                out += f"\\quad {lab}, {nm_} & & {pw[il]['hr']:.3f} [{pw[il]['ci95'][0]:.3f}, {pw[il]['ci95'][1]:.3f}] & $p={pw[il]['p']:.3f}$; {pw[il]['events']:,} events; {pw[il]['at_risk']:,} at risk \\\\\n"
    return out
w("tab_survival.tex", rf"""\begin{{table}}[htbp]\centering\footnotesize
\caption{{How interlocks end: survival of interlocks by origin-firm restatement history, with cause-specific hazards (origin restated at any time before the tie).}}
\label{{tab:survival}}
\adjustbox{{max width=\textwidth}}{{\begin{{tabular}}{{p{{5.6cm}}ccp{{4.0cm}}}}\toprule
 & Kaplan--Meier & Hazard ratio [95\% CI] & \\\midrule
{sv_rows("prior_any", "Origin filed a restatement at any time before the tie (" + pc(K5['share_prior_any']) + "\\% of interlocks)")}\bottomrule\end{{tabular}}}}\par\vspace{{3pt}}
\begin{{minipage}}{{\textwidth}}\footnotesize \emph{{Notes.}} Interlocks formed 2004--2023 with an identifiable origin, $N={K5['n_edges']:,}$; {pc(K5['share_ongoing'])}\% are open at the horizon and
right-censored. The origin is the firm at which the director's seat began earlier; the destination is the firm whose later appointment created the interlock.
An interlock ends when either seat ends; cause-specific hazards treat dissolution through the other seat as censoring. Cox models are stratified by
formation year with standard errors clustered by origin firm; ``PH'' is the Schoenfeld-residual test of proportional hazards (rank transform). ``With covariates'' adds the
director's number of concurrent seats at formation and tenure at the origin, an indicator for a destination restatement before the tie, and the years since each firm first appears in the feed. Because the
proportional-hazards test rejects for the origin-seat models, the ``by interval'' rows re-estimate the covariate model separately for interlocks at risk in years 0--2, 2--5 and beyond 5 after formation.
``Origin firm continues'' means a board event is recorded at the origin more than one year after the interlock ended ({pc(K5['prior_any']['share_origin_seat_ends_with_firm_continuing'])}\% of origin-seat
endings); ``exits the feed'' covers the rest, which include delistings and acquisitions. Interlocks that end on the same day at both firms ({K5['terminated_by']['both']:,}) count as any
dissolution but as neither cause.\end{{minipage}}\end{{table}}""")
w("tab_survival2.tex", rf"""\begin{{table}}[htbp]\centering\footnotesize
\caption{{How interlocks end: survival of interlocks by origin-firm restatement history, with cause-specific hazards (origin restated within five years before the tie).}}
\label{{tab:survival2}}
\adjustbox{{max width=\textwidth}}{{\begin{{tabular}}{{p{{5.6cm}}ccp{{4.0cm}}}}\toprule
 & Kaplan--Meier & Hazard ratio [95\% CI] & \\\midrule
{sv_rows("prior_5y", "Origin filed a restatement within 5 years before the tie (" + pc(K5['share_prior_5y']) + "\\% of interlocks)")}\bottomrule\end{{tabular}}}}\par\vspace{{3pt}}
\begin{{minipage}}{{\textwidth}}\footnotesize \emph{{Notes.}} Definitions, models and standard errors as in Table~\ref{{tab:survival}}.\end{{minipage}}\end{{table}}""")

# ── T10 sample and design comparison ─────────────────────────────────────────────────
def c_(d): return g(d) if d else "---"
def s10(lbl, b_, did_onset=True, never_onset="never_restated_post_onset", clean=True):
    do = dd(b_["did_onset"]) if (did_onset and "did_onset" in b_) else "---"
    nv_ = c_(b_.get(never_onset)) if never_onset else "---"
    ca = c_(b_.get("clean_post_onset")) if clean else "---"; cr = c_(b_.get("clean_post_onset_recent5y")) if clean else "---"
    return f"{lbl} & {b_['n_rows']:,} & {g(b_['pre_file'])} & {g(b_['post_file'])} & {dd(b_['did_file'])} & {do} & {nv_} & {ca} & {cr} \\\\\n"
def main_blk(J):
    a_ = J["K1_pairs"]; ew_ = a_["equal_window_3y"]; nv_ = a_["never_restated_destinations"]; cl_ = a_["clean_destinations_no_history"]
    return {"n_rows": ew_["n_rows"], "pre_file": ew_["pre_file"], "post_file": ew_["post_file"], "did_file": ew_["did_file"], "did_onset": ew_["did_onset"],
            "never_restated_post_onset": nv_["post_onset_3y"], "clean_post_onset": cl_["exposed"]["post_onset"], "clean_post_onset_recent5y": cl_["exposed5y"]["post_onset"]}
def bl_blk(J):
    b_ = J["K1_pairs"]["both_compustat_linked"]; return {"n_rows": b_["n_rows"], "pre_file": b_["pre_file"], "post_file": b_["post_file"], "did_file": b_["did_file"], "never_restated_post_onset": b_["never_restated_post_onset"]}
xs_note = (f" ``Blank-check companies'' are the {BC['n_blank_check_ciks']:,} registrants with SIC code 6770 in SEC EDGAR or a current or former name of the ``\\ldots{{}} Acquisition Corp'' type; "
           f"{pc(BC['share_pairs_with_blank_check_firm'])}\\% of primary pairs involve one, and they account for {pc(BC['share_restatement_filings_2020_2023_by_blank_check'])}\\% of restatement filings in 2020--2023." if BC else "")
AF = K1["artifact_flags"]; AFF = KF["K1_pairs"]["artifact_flags"]
rows10 = (s10("All directors, first-and-last-name key, no exclusions: both directions (undirected)", KF["K1_pairs"]["both_directions_pooled"], clean=False)
        + s10("\\quad appointment direction", main_blk(KF))
        + s10("\\quad appointment direction, artifact flags removed", KF["K1_pairs"]["artifacts_removed_all_directors"])
        + s10("\\quad reverse direction", KF["K1_pairs"]["reverse_direction"])
        + "\\midrule\n"
        + s10("Primary sample: all directors, artifact registrants and common-surname directors removed, conflict-aware key, appointment direction", main_blk(K))
        + s10("\\quad directors with $>6$ firms only", K1["high_seat_directors_only"])
        + s10("\\quad reverse direction", K1["reverse_direction"])
        + "\\midrule\n"
        + s10("Alternative definition: directors with $\\le 6$ firms, first-and-last-name key, appointment direction", main_blk(KC))
        + s10("\\quad artifact flags removed", KC["K1_pairs"]["artifacts_removed_all_directors"])
        + s10("\\quad reverse direction", KC["K1_pairs"]["reverse_direction"]))
import os as _os
KV = {k: json.load(open(HERE.parent / "results_rebuild" / f)) for k, f in (("artns", f"K_rebuild{PRIMARY}ns.json"), ("name", f"K_rebuild{PRIMARY}_name.json"), ("noest", f"K_rebuild{PRIMARY}_noest.json")) if _os.path.exists(HERE.parent / "results_rebuild" / f)}
var_rows = ""
if "artns" in KV:
    var_rows += s10(f"Primary sample, common-surname directors kept ({KV['artns']['sample_definition']['n_lifecycles_after'] - SD.get('n_lifecycles_after', 0):,} more lifecycles)", main_blk(KV["artns"]))
if "name" in KV:
    var_rows += s10(f"Primary sample with the first-and-last-name key (no middle-initial split; {KV['name']['sample_definition']['n_lifecycles_after']:,} lifecycles)", main_blk(KV["name"]))
if "noest" in KV:
    var_rows += s10(f"Primary sample, spells with an estimated effective date dropped ({KV['noest']['K0_edges']['person_key']['spells_with_estimated_endpoint']:,} spells)", main_blk(KV["noest"]))
_pk = K["K0_edges"].get("person_key", {})
rows10b = (s10("All directors, no exclusions, both firms Compustat-linked", bl_blk(KF), did_onset=False, clean=False)
        + s10("All directors, no exclusions, directors with $>6$ firms only", KF["K1_pairs"]["high_seat_directors_only"])
        + s10("All directors, no exclusions, $\\le 6$ concurrent seats at formation", KF["K1_pairs"]["concurrent_seats_le6_at_formation"])
        + "\\midrule\n"
        + s10("Primary sample, both directions", K1["both_directions_pooled"], clean=False)
        + s10("Primary sample, both firms Compustat-linked", bl_blk(K), did_onset=False, clean=False)
        + (s10("Primary sample, excluding blank-check companies", XS) if (XS and XS["n_rows"] != EW["n_rows"]) else "")
        + s10("Primary sample, surname shared by $\\le 50$ names", {**K1["common_name_sensitivity"]["surname_shared_by_at_most_50_names"], "clean_post_onset": K1["common_name_sensitivity"]["surname_shared_by_at_most_50_names"]["clean_destinations_post_onset"], "clean_post_onset_recent5y": K1["common_name_sensitivity"]["surname_shared_by_at_most_50_names"]["clean_destinations_post_onset_recent5y"]}, never_onset=None)
        + var_rows)
w("tab_samples.tex", rf"""\begin{{sidewaystable}}[htbp]\centering\def\baselinestretch{{1}}\footnotesize\setlength{{\tabcolsep}}{{3pt}}
\caption{{The main estimates across sample and design definitions.}}
\label{{tab:samples}}
\adjustbox{{max width=\linewidth}}{{\begin{{tabular}}{{p{{5.0cm}}cccccccc}}\toprule
 & & \multicolumn{{3}}{{c}}{{Filed, 3y windows}} & Onset & Never-filed dest., & \multicolumn{{2}}{{c}}{{History-free dest., onset after}} \\
Sample & $N$ & before & after & after $-$ before & after $-$ before & onset after & any exposure & recent \\\midrule
{rows10}\bottomrule\end{{tabular}}}}\par\vspace{{3pt}}
\begin{{minipage}}{{\linewidth}}\footnotesize \emph{{Notes.}} Each row applies the definitions of Tables~\ref{{tab:prepost}} and \ref{{tab:clean}} to a different sample or design; cells are gaps in percentage points (SE) with two-way
(origin, destination) clustered standard errors. ``All directors, no exclusions'' uses the first-and-last-name key and includes every registrant and every director; under that key {pc(KF['director_firm_counts']['share_directors_gt6'])}\% of directors are interlocked with more than six registrant boards and generate {pc(KF['director_firm_counts']['share_edges_from_directors_gt6'])}\% of all lifecycles.
Over the life of the tie plus one year the undirected filing gap is ${KF['K1_pairs']['both_directions_pooled']['life_plus1y_file']['diff_pp']:+.2f}$ ({KF['K1_pairs']['both_directions_pooled']['life_plus1y_file']['se_pp']:.2f}) with no exclusions and ${K1['both_directions_pooled']['life_plus1y_file']['diff_pp']:+.2f}$ ({K1['both_directions_pooled']['life_plus1y_file']['se_pp']:.2f}) in the primary sample. ``Both directions'' counts each interlock twice, once with each firm as the destination, and includes the interlocks without an identifiable origin; ``appointment direction'' keeps only the pair in which the destination is the board
that made the appointment. ``Artifact flags'' remove pairs in which either firm is a fund, partnership or REIT-family registrant ({AFF['n_family_ciks']:,} registrants) or a shell or blank-check company ({AFF['n_shell_ciks']:,}), or the director's
surname is shared by more than 150 distinct director names; the flags apply to {pc(AFF['share_pairs_flagged'])}\% of all-director pairs ({pc(AFF['share_flagged_among_high_seat_directors'])}\% among directors with more than six firms, {pc(AFF['share_flagged_among_other_directors'])}\% among the rest).
The {AFF['n_shell_ciks']:,} ``shell or blank-check'' registrants of the artifact flags are the {BC['n_blank_check_ciks']:,} blank-check registrants (SIC 6770 or an ``Acquisition Corp''-type name) plus registrants with SIC code 9995 or ``acquisition'' in their current name; {K2['estimation_sample'].get('blank_check_firms_in_panel', 0):,} of the blank-check registrants are Compustat-linked and appear in the firm-year panel of Table~\ref{{tab:panel}}. The reverse-direction rows treat the director's earlier board as the firm at risk and the appointing board's history as the exposure; they use the same {EW['n_rows']:,} direction-identifiable lifecycles as the primary rows, whereas the undirected row also includes lifecycles whose two seats began on the same day. Over the life of the tie plus one year, the reverse-direction filing gap is ${K1['reverse_direction']['life_plus1y_file']['diff_pp']:+.2f}$ ({K1['reverse_direction']['life_plus1y_file']['se_pp']:.2f}) in the primary sample. Table~\ref{{tab:samples_more}} in \ref{{app:tables}} reports further variants.\end{{minipage}}\end{{sidewaystable}}""")

var_note = (f" Under the first-and-last-name key and the estimated-date exclusion the within-firm coefficient on recently exposed ties formed in $t-1$ with own history held fixed (Table~\\ref{{tab:residual}}) is "
            f"${100*KV['name']['K2_panel']['models']['lag1_onset_exposed5y_plus_prior']['exposure_coef']['coef']:+.2f}$ ($t={KV['name']['K2_panel']['models']['lag1_onset_exposed5y_plus_prior']['exposure_coef']['t']:.2f}$) and "
            f"${100*KV['noest']['K2_panel']['models']['lag1_onset_exposed5y_plus_prior']['exposure_coef']['coef']:+.2f}$ ($t={KV['noest']['K2_panel']['models']['lag1_onset_exposed5y_plus_prior']['exposure_coef']['t']:.2f}$); under the six-firm cap it is "
            f"${100*KC['K2_panel']['models']['lag1_onset_exposed5y_plus_prior']['exposure_coef']['coef']:+.2f}$ ($t={KC['K2_panel']['models']['lag1_onset_exposed5y_plus_prior']['exposure_coef']['t']:.2f}$).") if ("name" in KV and "noest" in KV) else ""
w("tab_samples_more.tex", rf"""\begin{{sidewaystable}}[htbp]\centering\def\baselinestretch{{1}}\footnotesize\setlength{{\tabcolsep}}{{3pt}}
\caption{{Further sample and design variants of the main estimates.}}
\label{{tab:samples_more}}
\adjustbox{{max width=\linewidth}}{{\begin{{tabular}}{{p{{5.0cm}}cccccccc}}\toprule
 & & \multicolumn{{3}}{{c}}{{Filed, 3y windows}} & Onset & Never-filed dest., & \multicolumn{{2}}{{c}}{{History-free dest., onset after}} \\
Sample & $N$ & before & after & after $-$ before & after $-$ before & onset after & any exposure & recent \\\midrule
{rows10b}\bottomrule\end{{tabular}}}}\par\vspace{{3pt}}
\begin{{minipage}}{{\linewidth}}\footnotesize \emph{{Notes.}} Definitions as in Table~\ref{{tab:samples}}. The alternative definition caps the firms a director is interlocked with over 2004--2025 at six; the concurrent-seat row keeps pairs whose director held at most six seats on the formation date ({pc(KF['K1_pairs']['cap_definition']['share_pairs_director_concurrent_seats_le6'])}\% of all-director pairs).{xs_note} The surname row drops, in addition, directors whose surname is shared by more than 50 distinct director names in the feed (the primary sample already drops those above 150). The reverse-direction onset change of ${K1['reverse_direction']['did_onset']['diff_in_gaps_pp']:+.2f}$ ({K1['reverse_direction']['did_onset']['se_pp']:.2f}) is ${K1['reverse_direction']['did_onset_cohortFE']['diff_in_gaps_pp']:+.2f}$ ({K1['reverse_direction']['did_onset_cohortFE']['se_pp']:.2f}) with formation-year effects and ${K1['reverse_direction']['did_onset_cohortFE_excl_blank_check']['diff_in_gaps_pp']:+.2f}$ ({K1['reverse_direction']['did_onset_cohortFE_excl_blank_check']['se_pp']:.2f}) when blank-check pairs are also excluded (recent exposure: ${K1['reverse_direction']['recent_did_onset']['diff_in_gaps_pp']:+.2f}$, ${K1['reverse_direction']['recent_did_onset_cohortFE']['diff_in_gaps_pp']:+.2f}$ and ${K1['reverse_direction']['recent_did_onset_cohortFE_excl_blank_check']['diff_in_gaps_pp']:+.2f}$). The identification variants use the middle names and effective-date flags of the Audit Analytics feed (merged onto the extract by firm, name, date and action): the primary sample's conflict-aware key splits a first-and-last name across firms when the recorded middle initials conflict ({_pk.get('names_with_conflicting_initials', 0):,} of {_pk.get('names_at_more_than_one_firm', 0):,} names at more than one firm), treating firm blocks without an initial under such a name as unassignable (no interlocks); the first-and-last-name row removes that split, and the estimated-date row drops tenure spells whose opening or closing event carries the feed's estimated-effective-date flag.{var_note}\end{{minipage}}\end{{sidewaystable}}""")


# ── T11 board-refresh test ─────────────────────────────────────────────────────────────
RF = K1["board_refresh_test"]
w("tab_refresh.tex", rf"""\begin{{table}}[htbp]\centering\footnotesize
\caption{{Board refresh after a restatement: recruitment from restating firms by boards that have just restated.}}
\label{{tab:refresh}}
\adjustbox{{max width=\textwidth}}{{\begin{{tabular}}{{p{{5.0cm}}cccc}}\toprule
 & \multicolumn{{2}}{{c}}{{Share of appointments from an exposed origin}} & & Difference with \\
 & destination filed & other & Difference & formation-year \\
Exposure definition & in prior 12 months & destinations & pp (SE) & effects, pp (SE) \\\midrule
Origin restated at any time before the tie & {pc(RF['exposed_share_if_dst_filed_12m_before'])}\% & {pc(RF['exposed_share_otherwise'])}\% & ${RF['diff_pp']:+.2f}$ ({RF['se_pp']:.2f}) & ${RF['any_diff_pp_yearFE']:+.2f}$ ({RF['any_se_pp_yearFE']:.2f}) \\
Origin restated within 5 years before the tie & {pc(RF['recent_exposed_share_if_dst_filed_12m_before'])}\% & {pc(RF['recent_exposed_share_otherwise'])}\% & ${RF['recent_diff_pp']:+.2f}$ ({RF['recent_se_pp']:.2f}) & ${RF['recent_diff_pp_yearFE']:+.2f}$ ({RF['recent_se_pp_yearFE']:.2f}) \\
\bottomrule\end{{tabular}}}}\par\vspace{{3pt}}
\begin{{minipage}}{{\textwidth}}\footnotesize \emph{{Notes.}} Appointment-direction pairs formed 2004--2022 ($N={EW['n_rows']:,}$); {pc(RF['share_dst_filed_12m_before'])}\% of destinations filed a restatement in the twelve months before the appointment.
The difference is the coefficient on that indicator in a linear probability model for the exposure indicator, without and with formation-year dummies; two-way (origin, destination) clustered standard errors.\end{{minipage}}\end{{table}}""")

# ── Figures ────────────────────────────────────────────────────────────────────────────
# F1: pre vs post gaps by cohort + pooled (axis extended below zero)
fig, ax = plt.subplots(figsize=(6.6, 3.9))
labels = list(bc.keys()) + ["All 2004--2022"]
pre = [bc[k]["pre_file_pp"] for k in bc] + [EW["pre_file"]["diff_pp"]]
post = [bc[k]["post_file_pp"] for k in bc] + [EW["post_file"]["diff_pp"]]
x = np.arange(len(labels)); wd = 0.38
ax.bar(x - wd / 2, pre, wd, color="#7f7f7f", label="3 years before the tie formed")
ax.bar(x + wd / 2, post, wd, color="#C44E52", label="3 years after the tie formed")
ax.errorbar([x[-1] - wd / 2], [pre[-1]], yerr=[1.96 * EW["pre_file"]["se_pp"]], fmt="none", ecolor="k", capsize=3)
ax.errorbar([x[-1] + wd / 2], [post[-1]], yerr=[1.96 * EW["post_file"]["se_pp"]], fmt="none", ecolor="k", capsize=3)
for i, k in enumerate(bc): ax.text(x[i], max(pre[i], post[i]) + 0.6, f"Δ = {bc[k]['did']['diff_in_gaps_pp']:+.1f}", ha="center", fontsize=8.5)
ax.text(x[-1], max(pre[-1], post[-1]) + 2.4, f"Δ = {EW['did_file']['diff_in_gaps_pp']:+.1f} (SE {EW['did_file']['se_pp']:.2f})", ha="center", fontsize=8.5)
ax.set_xticks(x); ax.set_xticklabels([l.replace("--", "–") for l in labels]); ax.set_ylabel("Exposed − clean gap in destination\nrestatement filing rate (pp)")
ax.axhline(0, color="k", lw=0.6); ax.legend(frameon=False, fontsize=9, loc="upper right"); ax.set_ylim(min(min(pre), min(post)) - 1.5, max(pre) * 1.25)
plt.tight_layout(); plt.savefig(FIG / "figK1_prepost.pdf"); plt.close(); print("wrote figK1_prepost.pdf")

# F2: panel forest plot
fig, ax = plt.subplots(figsize=(7.4, 6.6))
order = [("contemporaneous_file (old timing)", "(1) Filed in t ~ exposed formed in t\n(old timing; new-interlocks control)"), ("lag1_file_no_turnover_control", "(2) Filed in t ~ exposed formed in t−1\n(no new-interlocks control)"),
         ("lag1_file", "(3) Filed in t ~ exposed formed in t−1"), ("lag1_file_plus_prior", "(4) Filed in t ~ exposed formed in t−1\n(+ own prior restatement)"), ("lag1_onset", "(5) Onset in t ~ exposed formed in t−1"), ("boy_stock_file", "(6) Filed in t ~ exposed alive 1 Jan t"), ("boy_stock_onset", "(7) Onset in t ~ exposed alive 1 Jan t"),
         ("lag1_file_inbound_only", "(8) Filed in t ~ inbound exposed formed in t−1"), ("lag1_onset_inbound_only", "(9) Onset in t ~ inbound exposed formed in t−1"),
         ("lag1_file_exposed5y", "(10) Filed in t ~ recent exposed formed in t−1"), ("lag1_file_exposed5y_plus_prior", "(11) Filed in t ~ recent exposed formed in t−1\n(+ own prior restatement)"),
         ("lag1_onset_exposed5y", "(12) Onset in t ~ recent exposed formed in t−1"), ("lag1_onset_exposed5y_plus_prior", "(13) Onset in t ~ recent exposed formed in t−1\n(+ own prior restatement)"),
         ("boy_stock_onset_exposed5y", "(14) Onset in t ~ recent exposed alive 1 Jan t"), ("lag1_file_inbound_recent5y_plus_prior", "(15) Filed in t ~ inbound recent exposed formed in t−1\n(+ own prior restatement)"),
         ("lag1_onset_inbound_recent5y_plus_prior", "(16) Onset in t ~ inbound recent exposed formed in t−1\n(+ own prior restatement)")]
ys = np.arange(len(order))[::-1]
for yy, (k, lab) in zip(ys, order):
    c = M[k]["exposure_coef"]; ax.errorbar(100 * c["coef"], yy, xerr=1.96 * 100 * c["se"], fmt="o", color="#C44E52" if "no new" in lab or "old timing" in lab else "#4C72B0", capsize=3)
ax.axvline(0, color="k", lw=0.7); ax.set_yticks(ys); ax.set_yticklabels([l for _, l in order], fontsize=8.5)
ax.set_xlabel("Coefficient, pp per exposed interlock (95% CI)")
plt.tight_layout(); plt.savefig(FIG / "figK2_panel.pdf"); plt.close(); print("wrote figK2_panel.pdf")

# F3: survival hazard ratios forest
fig, ax = plt.subplots(figsize=(7.0, 3.6))
items = []
for key, lbl in (("prior_any", "origin restated ever before"), ("prior_5y", "origin restated within 5y before")):
    for ck, cl in (("cox_any_dissolution", "any dissolution"), ("cox_destination_seat_ended_covariates", "destination seat ended (covariates)"), ("cox_origin_seat_ended_covariates", "origin seat ended (covariates)"), ("cox_origin_seat_ended_firm_continues", "origin seat ended, firm continues")):
        c = K5[key][ck]; items.append((f"{lbl}: {cl}", c["hr"], c["ci95"]))
ys = np.arange(len(items))[::-1]
for yy, (lab, hr_, ci) in zip(ys, items):
    ax.errorbar(hr_, yy, xerr=[[hr_ - ci[0]], [ci[1] - hr_]], fmt="o", color="#4C72B0" if "destination" in lab else ("#C44E52" if "origin seat" in lab else "#555555"), capsize=3)
ax.axvline(1, color="k", lw=0.7); ax.set_yticks(ys); ax.set_yticklabels([i[0] for i in items], fontsize=8.5); ax.set_xlabel("Hazard ratio of dissolution (95% CI)")
plt.tight_layout(); plt.savefig(FIG / "figK3_hazards.pdf"); plt.close(); print("wrote figK3_hazards.pdf")

# F4: network by year
yrs = sorted(int(y) for y in NET); fig, ax1 = plt.subplots(figsize=(6.4, 3.4)); ax2 = ax1.twinx()
ax1.plot(yrs, [NET[str(y)]["nodes"] for y in yrs], "-o", ms=3, color="#4C72B0", label="firms in network")
ax2.plot(yrs, [NET[str(y)]["lcc_share"] for y in yrs], "-s", ms=3, color="#C44E52", label="largest component share")
ax2.plot(yrs, [NET[str(y)]["modularity"] for y in yrs], "-^", ms=3, color="#55A868", label="modularity")
ax1.set_ylabel("Firms"); ax2.set_ylabel("Share / modularity"); ax1.set_xlabel("Year"); ax1.set_xticks(list(range(2004, 2024, 3)))
h1, l1 = ax1.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels(); ax1.legend(h1 + h2, l1 + l2, frameon=False, fontsize=8.5, loc="center right")
plt.tight_layout(); plt.savefig(FIG / "figK4_network.pdf"); plt.close(); print("wrote figK4_network.pdf")

# F5: event-time gaps
fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.3), sharey=True)
ks = [str(k) for k in range(-3, 3)]; xs = np.arange(-3, 3) + 0.5
for ax, (lab, key, key5) in zip(axes, (("Restatement filed", "file", "file_recent5y"), ("Misstated period begins", "onset", "onset_recent5y"))):
    v = [ET[k][key]["diff_pp"] for k in ks]; e = [1.96 * ET[k][key]["se_pp"] for k in ks]
    v5 = [ET[k][key5]["diff_pp"] for k in ks]; e5 = [1.96 * ET[k][key5]["se_pp"] for k in ks]
    ax.errorbar(xs - 0.08, v, yerr=e, fmt="o-", color="#4C72B0", capsize=3, label="origin restated at any time before the tie")
    ax.errorbar(xs + 0.08, v5, yerr=e5, fmt="s--", color="#C44E52", capsize=3, label="origin restated within 5 years before the tie")
    ax.axvline(0, color="k", lw=0.8); ax.axhline(0, color="k", lw=0.5); ax.set_title(lab, fontsize=10); ax.set_xlabel("Years relative to formation")
    ax.set_xticks(xs); ax.set_xticklabels(["−3", "−2", "−1", "0", "+1", "+2"])
axes[0].set_ylabel("Exposed − clean gap (pp)"); axes[0].legend(frameon=False, fontsize=7.5, loc="upper right"); fig.text(0.5, -0.02, "Year 0 begins on the formation date", ha="center", fontsize=8)
plt.tight_layout(); plt.savefig(FIG / "figK5_eventtime.pdf"); plt.close(); print("wrote figK5_eventtime.pdf")

# ── compact survival (recent exposure) for the main text ─────────────────────────────
b5 = K5["prior_5y"]
def hrr(c): return f"{c['hr']:.2f} [{c['ci95'][0]:.2f}, {c['ci95'][1]:.2f}]"
w("tab_survival_compact.tex", rf"""\begin{{table}}[htbp]\centering\footnotesize
\caption{{Where active interlocks end when the origin restated within the five years before the tie: cause-specific hazards (Cox, formation-year strata, origin clusters).}}
\label{{tab:survival_compact}}
\begin{{tabular}}{{@{{}}p{{8.6cm}}cc@{{}}}}\toprule
Event & Hazard ratio [95\% CI] & Events \\\midrule
Any dissolution & {hrr(b5['cox_any_dissolution'])} & {b5['cox_any_dissolution']['events']:,} \\
Destination (appointing-board) seat ended, with covariates & {hrr(b5['cox_destination_seat_ended_covariates'])} & {b5['cox_destination_seat_ended_covariates']['events']:,} \\
\quad years 0--2 / 2--5 / beyond 5 after formation & {b5['cox_destination_seat_ended_piecewise']['0_2y']['hr']:.2f} / {b5['cox_destination_seat_ended_piecewise']['2_5y']['hr']:.2f} / {b5['cox_destination_seat_ended_piecewise']['5y_plus']['hr']:.2f} & \\
Origin seat ended, with covariates & {hrr(b5['cox_origin_seat_ended_covariates'])} & {b5['cox_origin_seat_ended_covariates']['events']:,} \\
\quad years 0--2 / 2--5 / beyond 5 after formation & {b5['cox_origin_seat_ended_piecewise']['0_2y']['hr']:.2f} / {b5['cox_origin_seat_ended_piecewise']['2_5y']['hr']:.2f} / {b5['cox_origin_seat_ended_piecewise']['5y_plus']['hr']:.2f} & \\
\quad origin firm continues to record board events / leaves the feed & {b5['cox_origin_seat_ended_firm_continues']['hr']:.2f} / {b5['cox_origin_seat_ended_firm_exit']['hr']:.2f} & {b5['cox_origin_seat_ended_firm_continues']['events']:,} / {b5['cox_origin_seat_ended_firm_exit']['events']:,} \\
\bottomrule\end{{tabular}}\par\vspace{{3pt}}
\begin{{minipage}}{{\textwidth}}\footnotesize \emph{{Notes.}} Interlocks formed 2004--2023 with an identifiable origin ($N={K5['n_edges']:,}$; {pc(K5['share_prior_5y'])}\% exposed, against {pc(K1['exposure_recent_5y']['share_exposed5y'])}\% in the 2004--2022 equal-window sample of Table~\ref{{tab:prepost}}; {pc(K5['share_ongoing'])}\% open at the horizon). The clock starts at formation; both seats are
active at entry. Kaplan--Meier median duration {b5['km_prior']['median']} years for exposed against {b5['km_clean']['median']} for clean interlocks (the {K0['duration_years_median']:.1f}-year median of Table~\ref{{tab:sample}} is the observed duration including censored lifecycles). Covariates: the director's concurrent seats and tenure at the origin, a destination restatement before the tie,
and both firms' years in the feed. Cause-specific hazards treat the other seat's ending as censoring ({K5['terminated_by']['both']:,} interlocks ending on the same day at both seats count as dissolutions but as neither cause); seat endings include resignations, retirements, dismissals, failed re-elections, deaths, and
firm disappearance, so they describe where and when interlocks end, not decisions by appointing boards. Full results, the any-exposure definition and the proportional-hazards tests are in Appendix Tables~\ref{{tab:survival}} and \ref{{tab:survival2}}.\end{{minipage}}\end{{table}}""")

# ── size terciles: recent exposure pre/post (referee r9 must-fix 5) ─────────────────────
TZ = K1["receptivity_heterogeneity"]["by_size_tercile_prepost"]; cut_ = K1["receptivity_heterogeneity"]["size_tercile_cutpoints_log_assets"]
def tzrow(lbl, k):
    v = TZ[k]; return (f"{lbl} & {v['n_rows']:,} & {v['n_exposed5y']:,} & {g(v['pre_onset_recent5y'])} & {g(v['post_onset_recent5y'])} & {g(v['post_onset_recent5y_yearFE'])} & {dd(v['did_onset_recent5y'])} & {dd(v['did_onset_recent5y_cohortFE'])} \\\\" + "\n")
w("tab_sizetercile.tex", rf"""\begin{{table}}[htbp]\centering\footnotesize\setlength{{\tabcolsep}}{{4pt}}
\caption{{Recent exposure by destination size: the exposed--clean onset gap before and after formation and its change, by tercile of the destination's log assets.}}
\label{{tab:sizetercile}}
\adjustbox{{max width=\textwidth}}{{\begin{{tabular}}{{p{{4.2cm}}ccccccc}}\toprule
 & & Recently & Pre-formation & Post-formation & Post, year- & Change & Change, cohort- \\
Destination size tercile & Pairs & exposed & gap, pp (SE) & gap, pp (SE) & adjusted (SE) & unadjusted (SE) & adjusted (SE) \\\midrule
\multicolumn{{8}}{{@{{}}p{{15.8cm}}@{{}}}}{{\emph{{Panel A. Destinations that had never filed a restatement before the appointment}}}} \\
{tzrow(f"Smallest third (log assets $\\le {cut_[0]:.2f}$)", "never_filed__size_bottom_tercile")}{tzrow("Middle third", "never_filed__size_middle_tercile")}{tzrow(f"Largest third (log assets $> {cut_[1]:.2f}$)", "never_filed__size_top_tercile")}\midrule
\multicolumn{{8}}{{@{{}}p{{15.8cm}}@{{}}}}{{\emph{{Panel B. All pairs}}}} \\
{tzrow("Smallest third", "all_pairs__size_bottom_tercile")}{tzrow("Middle third", "all_pairs__size_middle_tercile")}{tzrow("Largest third", "all_pairs__size_top_tercile")}\bottomrule\end{{tabular}}}}\par\vspace{{3pt}}
\begin{{minipage}}{{\textwidth}}\footnotesize \emph{{Notes.}} Compustat-linked destinations in the equal-window sample (interlocks formed 2004--2022), split at the terciles of log assets used in Table~\ref{{tab:clean}}, Panel C. Exposure is a restatement filed by the origin within the five years before the tie; the outcome is a misstated period beginning in the three-year window. ``Year-adjusted'' adds formation-year effects; the cohort-adjusted change adds formation-year and formation-year $\times$ post effects to the stacked regression. Two-way (origin, destination) clustered standard errors; stars as in Table~\ref{{tab:prepost}}.\end{{minipage}}\end{{table}}""")

# ── between-firm screening (referee r9 must-fix 7) ─────────────────────────────────────
BS = K["K2_panel"]["between_firm_screening"]; BO = K["K2_panel"]["between_firm_screening_out_of_sample"]
def srow(lbl, key):
    cells = []
    for dep in ("restate_file", "restate_onset"):
        a_ = BS[dep][key]; b_ = BO[dep][key]
        cells += [f"${a_['coef']:+.3f}$ ({a_['se']:.3f})", f"{a_['auc']:.4f}", f"{a_['auc_gain']:+.4f}", f"{b_['auc_test']:.4f}", f"{b_['auc_gain_test']:+.4f}"]
    return lbl + " & " + " & ".join(cells) + " \\\\\n"
w("tab_screening.tex", rf"""\begin{{table}}[htbp]\centering\footnotesize\setlength{{\tabcolsep}}{{3pt}}
\caption{{Between-firm screening: does an exposure count add to a prediction built on the registrant's own record?}}
\label{{tab:screening}}
\adjustbox{{max width=\textwidth}}{{\begin{{tabular}}{{p{{4.6cm}}cccccccccc}}\toprule
 & \multicolumn{{5}}{{c}}{{Restatement filed in $t$}} & \multicolumn{{5}}{{c}}{{Misstated period begins in $t$}} \\
 & Logit coef. & AUC & $\Delta$AUC & AUC, & $\Delta$AUC, & Logit coef. & AUC & $\Delta$AUC & AUC, & $\Delta$AUC, \\
Exposure count added & (SE) & in sample & in sample & 2017--2023 & 2017--2023 & (SE) & in sample & in sample & 2017--2023 & 2017--2023 \\\midrule
Baseline (no exposure count) & & {BS['restate_file']['baseline_auc']:.4f} & & {BO['restate_file']['baseline_auc_test']:.4f} & & & {BS['restate_onset']['baseline_auc']:.4f} & & {BO['restate_onset']['baseline_auc_test']:.4f} & \\
{srow("Exposed interlocks formed in $t-1$", "nexp_lag1")}{srow("Exposed interlocks alive on 1 January", "nexp_boy")}{srow("Recently exposed interlocks formed in $t-1$", "nexp5_lag1")}{srow("Recently exposed interlocks alive on 1 January", "nexp5_boy")}\bottomrule\end{{tabular}}}}\par\vspace{{3pt}}
\begin{{minipage}}{{\textwidth}}\footnotesize \emph{{Notes.}} Pooled logit of the outcome on an indicator for any restatement filed before $t$ (coefficient {BS['restate_file']['baseline_coef_had_prior_restate']:.3f} for filings, {BS['restate_onset']['baseline_coef_had_prior_restate']:.3f} for onsets), log assets, leverage, ROA, lagged degree, all new interlocks in $t-1$ and year effects, on the firm-year panel of Table~\ref{{tab:panel}} ($N={K['K2_panel']['estimation_sample']['firm_years']:,}$). ``In sample'' fits and scores on 2005--2023; the out-of-sample columns fit the same model without year effects on 2005--2016 ($N={BO['n_train']:,}$) and score 2017--2023 ($N={BO['n_test']:,}$; event rates {pc(BO['restate_file']['test_event_rate'])}\% and {pc(BO['restate_onset']['test_event_rate'])}\%). $\Delta$AUC is the change in the area under the ROC curve from adding the exposure count. These are associational comparisons in one sample.\end{{minipage}}\end{{table}}""")

# ── audit-committee assignment at the appointing event (WRDS comm_report; 2026-10-13) ──────
AC = K1.get("audit_committee_split")
if AC:
    def acrow(lbl, grp, ex):
        v = AC[grp].get(ex)
        if not v: return f"{lbl} & {AC[grp]['n_rows']:,} & --- & --- & --- & --- & --- \\\\\n"
        nv = g(v["never_restated_post_onset"]) if v["never_restated_post_onset"] else "---"
        return f"{lbl} & {AC[grp]['n_rows']:,} & {g(v['pre_onset'])} & {g(v['post_onset'])} & {dd(v['did_onset'])} & {dd(v['did_onset_cohortFE'])} & {nv} \\\\\n"
    def trow(lbl, key):
        v = AC[key]; return f"{lbl} & {v['n_rows']:,} & & & ${v['change_base_pp']:+.2f}$ ({v['se_base_pp']:.2f}) & ${v['triple_pp']:+.2f}$ ({v['se_triple_pp']:.2f}){star(v['t_triple'])} & \\\\\n"
    ac_rows = ""
    for ex, exl in (("exposed", "Origin restated at any time before the tie"), ("exposed5y", "Origin restated within 5 years before the tie"), ("exposed_served", "Director present at the origin during the misstatement")):
        ac_rows += f"\\multicolumn{{7}}{{@{{}}p{{15.8cm}}@{{}}}}{{\\emph{{Exposure: {exl}}}}} \\\\\n"
        ac_rows += acrow("\\quad audit committee named", "audit_committee_named", ex) + acrow("\\quad other committees named", "committees_named_not_audit", ex) + acrow("\\quad no committee recorded", "no_committee_recorded", ex)
        if f"triple_{ex}_onset_vs_committees_not_audit" in AC: ac_rows += trow("\\quad difference, audit vs other committees$^{\\dagger}$", f"triple_{ex}_onset_vs_committees_not_audit")
        if f"triple_{ex}_onset_vs_all_other" in AC: ac_rows += trow("\\quad difference, audit vs all other appointments$^{\\dagger}$", f"triple_{ex}_onset_vs_all_other")
    AC6 = KC["K1_pairs"].get("audit_committee_split") if KC else None
    if AC6:
        def trow6(lbl, key):
            v = AC6[key]; return f"{lbl} & {v['n_rows']:,} & & & ${v['change_base_pp']:+.2f}$ ({v['se_base_pp']:.2f}) & ${v['triple_pp']:+.2f}$ ({v['se_triple_pp']:.2f}){star(v['t_triple'])} & \\\\\n"
        ac_rows += "\\multicolumn{7}{@{}p{15.8cm}@{}}{\\emph{Alternative definition (six-board cap, first-and-last-name key, no registrant exclusions), for comparison}} \\\\\n"
        for ex, exl in (("exposed", "any prior restatement"), ("exposed5y", "restatement within 5 years"), ("exposed_served", "director present")):
            if f"triple_{ex}_onset_vs_committees_not_audit" in AC6: ac_rows += trow6(f"\\quad {exl}: audit vs other committees$^{{\\dagger}}$", f"triple_{ex}_onset_vs_committees_not_audit")
            if f"triple_{ex}_onset_vs_all_other" in AC6: ac_rows += trow6(f"\\quad {exl}: audit vs all other appointments$^{{\\dagger}}$", f"triple_{ex}_onset_vs_all_other")
    w("tab_auditcomm.tex", rf"""\begin{{table}}[htbp]\centering\footnotesize\setlength{{\tabcolsep}}{{3pt}}
\caption{{Audit-committee assignment of the appointee: the exposed--clean gap in destination misstated-period onsets before and after formation, by what the appointing event's committee text records.}}
\label{{tab:auditcomm}}
\adjustbox{{max width=\textwidth}}{{\begin{{tabular}}{{p{{4.4cm}}cccccc}}\toprule
 & & Gap before, & Gap after, & Change, & Change, cohort- & Never-filed, \\
Appointment group & Pairs & pp (SE) & pp (SE) & pp (SE) & adjusted (SE) & gap after (SE) \\\midrule
{ac_rows}\bottomrule\end{{tabular}}}}\par\vspace{{3pt}}
\begin{{minipage}}{{\textwidth}}\footnotesize \emph{{Notes.}} Appointment-direction pairs formed 2004--2022. The committee text (\texttt{{comm\_report}}) of the Audit Analytics event that created the interlock is recorded for {pc(AC['share_appointments_with_committee_text'])}\% of pairs; {pc(AC['share_of_recorded_naming_audit'])}\% of those name the audit committee ({pc(AC['share_appointments_audit_committee'])}\% of all pairs). Whether a committee is named depends on what the filing disclosed, so the three groups are not a random partition, and the comparison is descriptive. ``Change'' is the coefficient on exposed $\times$ post in the stacked regression of Table~\ref{{tab:prepost}} within the group; ``cohort-adjusted'' adds formation-year and formation-year $\times$ post effects. $^{{\dagger}}$ Pooled regressions adding audit-committee $\times$ post and exposure $\times$ audit-committee $\times$ post terms to the cohort-adjusted specification, on the two groups with committee text (``other committees'') or on all pairs (``all other appointments''): the ``change'' column shows the pooled change for the comparator appointments and the ``cohort-adjusted'' column the difference for audit-committee appointments (the triple interaction). Two-way (origin, destination) clustered standard errors; stars as in Table~\ref{{tab:prepost}}.\end{{minipage}}\end{{table}}""")
