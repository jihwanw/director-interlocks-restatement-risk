#!/usr/bin/env python3
"""make_tables_l.py — tables for the identification layer from results_rebuild/L_identification_le6.json (+ K for cross-references).
Run from github/: venv python code/make_tables_l.py"""
import json
from pathlib import Path
HERE = Path(__file__).resolve().parent
import os
PRIMARY = os.environ.get("PRIMARY_SUFFIX", "_art")
L = json.load(open(HERE.parent / "results_rebuild" / f"L_identification{PRIMARY}.json"))
MM = json.load(open(HERE.parent / "results_rebuild" / f"M_icfr{PRIMARY}.json")) if (HERE.parent / "results_rebuild" / f"M_icfr{PRIMARY}.json").exists() else None
K = json.load(open(HERE.parent / "results_rebuild" / f"K_rebuild{PRIMARY}.json"))
OUT = HERE.parent.parent / "3rd_revision" / "source" / "tables_auto"
def w(name, s):
    s = s.replace("\\begin{table}[htbp]\\centering", "\\begin{table}[htbp]\\centering\\def\\baselinestretch{1}")  # elsarticle review stretch (1.5) leaks into p-cells
    (OUT / name).write_text(s); print("wrote", name)
def pc(x, d=1): return f"{100*x:.{d}f}"
def st(t): return "***" if t is not None and abs(t) > 2.576 else "**" if t is not None and abs(t) > 1.96 else "*" if t is not None and abs(t) > 1.645 else ""
def g(d): return f"${d['diff_pp']:+.2f}$ ({d['se_pp']:.2f})"
def sw(x): return "near zero" if abs(x) < 0.1 else ("negative" if x < 0 else "positive")   # sign word for generated prose cells
def dd(d): return f"${d['diff_in_gaps_pp']:+.2f}$ ({d['se_pp']:.2f}){st(d.get('t'))}"
A, B, C, D = L["A_latent_exposure"], L["B_falsification"], L["C_frozen_network_event_study"], L["D_bounds"]
CLS = [("latent", "Latent: origin misstatement completed before the tie, restated only after it (no recent public label)"),
       ("revealed", "Revealed: director sat through a misstated period restated before the tie"),
       ("label_only", "Label only: origin restated within 5 years before the tie; director joined the origin after the misstated period"),
       ("present_filing_only", "Present at filing only: director's seat covered a filing but overlapped no misstated period")]

# ── L1 unrevealed-source partitions (symmetric comparator) ────────────────────────────
P = A["partitions"]; lo = A["latent_onsets_before_origin_filing"]; dg = A["latent_diagnostics"]; ck = A["latent_destination_episode_clocks"]
LAB = {"all": "All pairs", "excl_blank_check": "Excluding pairs with a blank-check company", "excl_bc_and_destination_latent": "\\quad and excluding destinations with an unrevealed pre-appointment error of their own",
       "excl_bc_buffer6m": "Excluding blank-check pairs; origin filing at least 6 months after the appointment", "excl_bc_destination_latent_buffer6m": "\\quad and excluding destinations with an unrevealed pre-appointment error",
       "excl_bc_buffer12m": "Excluding blank-check pairs; origin filing at least 12 months after the appointment", "excl_bc_destination_latent_buffer12m": "\\quad and excluding destinations with an unrevealed pre-appointment error"}
def prow(k):
    v = P[k]; u = v["latent_units"]; c = v["comparator_units"]; r = v["rates"]
    stk = lambda e_, s_: f"\\begin{{tabular}}[t]{{@{{}}c@{{}}}}${e_:+.1f}$\\\\{{\\scriptsize[{e_-1.96*s_:+.1f}, {e_+1.96*s_:+.1f}]}}\\end{{tabular}}"
    ci = lambda d_: stk(d_['diff_pp'], d_['se_pp'])
    cc = lambda d_: stk(d_['diff_in_gaps_pp'], d_['se_pp'])
    hf = v["history_free_post_onset_yfe"]; hfc = (stk(hf["diff_pp"], hf["se_pp"]).replace("\\end{tabular}", "\\\\{\\scriptsize $n=" + str(v["history_free_n_latent"]) + "$}\\end{tabular}")) if hf else "---"
    return (f"{LAB[k]} & {u['pairs']:,} / {u['appointments']:,} / {u['destinations']:,} & {c['pairs']:,} & {pc(r['latent_pre_onset'])} / {pc(r['clean_pre_onset'])} & {pc(r['latent_post_onset'])} / {pc(r['clean_post_onset'])} & "
            f"{ci(v['pre_onset_yfe'])} & {ci(v['post_onset_yfe'])} & {cc(v['change_onset_yfe'])} & {hfc} \\\\\\addlinespace[2pt]\n")
NO_BC = P["all"]["latent_units"]["pairs"] == P["excl_blank_check"]["latent_units"]["pairs"]   # blank-check registrants already excluded by the sample definition
if NO_BC:
    LAB.update({"excl_bc_and_destination_latent": "Excluding destinations with an unrevealed pre-appointment error of their own", "excl_bc_buffer6m": "Origin filing at least 6 months after the appointment",
                "excl_bc_buffer12m": "Origin filing at least 12 months after the appointment"})
rows = "".join(prow(k) for k in ("all", "excl_blank_check", "excl_bc_and_destination_latent", "excl_bc_buffer6m", "excl_bc_destination_latent_buffer6m", "excl_bc_buffer12m", "excl_bc_destination_latent_buffer12m") if not (NO_BC and k == "excl_blank_check"))
w("tab_latent.tex", rf"""\begin{{sidewaystable}}[htbp]\centering\def\baselinestretch{{1}}\footnotesize\setlength{{\tabcolsep}}{{4pt}}
\caption{{Unrevealed pre-existing source-error exposure: destination misstated-period onsets by partition, against a symmetrically restricted clean comparator.}}
\label{{tab:latent}}
\begin{{tabular}}{{p{{4.9cm}}cccccccc}}\toprule
Partition (applied to both groups) & Unrevealed pairs / & & \multicolumn{{2}}{{c}}{{Onset rate, \%}} & \multicolumn{{3}}{{c}}{{Year-adjusted gap, pp}} & Never-filed \\
 & appointments / & Clean & \multicolumn{{2}}{{c}}{{(unrevealed / clean)}} & \multicolumn{{3}}{{c}}{{[95\% interval]}} & gap after \\
 & destinations & pairs & before & after & before & after & change & \\\midrule
{rows}\bottomrule\end{{tabular}}\par\vspace{{3pt}}
\begin{{minipage}}{{\linewidth}}\footnotesize \emph{{Notes.}} ``Unrevealed'' pairs: the director's seat at the origin overlapped a misstated period that ended before the appointment and was first restated after it, while the origin
had filed no restatement in the five years before the appointment; this records the absence of a qualifying filed origin restatement, not the absence of public warning signs or private knowledge (the origin's filing
follows the appointment by a median of {dg['origin_filing_months_after_tie_quartiles'][1]:.0f} months, quartiles {dg['origin_filing_months_after_tie_quartiles'][0]:.1f} to {dg['origin_filing_months_after_tie_quartiles'][2]:.1f}). The comparator is clean pairs (no origin filing within five years, not unrevealed,
director not present at a filing); the {"destination-error restriction is" if NO_BC else "blank-check and destination-error restrictions are"} applied to both groups, while the filing-delay restrictions concern the origin's filing after the appointment and therefore apply to unrevealed pairs only.{" Blank-check registrants are excluded by the sample definition." if NO_BC else ""} ``Destination with an unrevealed pre-appointment error'' means a destination misstated period that began before the appointment and
was filed after it ({pc(dg['share_destination_also_latent'])}\% of unrevealed pairs; {pc(dg['destination_latent_share_among_clean'])}\% of clean pairs); membership therefore depends on subsequent revelation, and the partitions are descriptive, not a cleaned causal comparison.
{pc(dg['share_destination_files_within_quarter_of_origin'])}\% of unrevealed pairs have a destination filing within one quarter of the origin's filing; {pc(dg['share_same_sic2'])}\% share a two-digit industry and {pc(dg['share_same_current_auditor'])}\% the current auditor. Among the {ck['n_latent_pairs']:,} unrevealed pairs,
destination episodes filed within three years after the appointment comprise {ck['legacy_ended_before_appointment']:,} legacy errors that ended before the appointment, {ck['began_before_continued_across']:,} that began before and continued across it, {ck['began_after_appointment_before_origin_filing']:,} that began after the appointment but before the origin's filing, and {ck['began_after_origin_filing']:,} that began after the origin's filing (a pair can have several).
Year-adjusted gaps include formation-year effects (and formation-year $\times$ post effects for the change); two-way (origin, destination) clustered standard errors. ``Never-filed gap after'' (destinations with no restatement filed before the appointment, the definition of Panel A of Table~\ref{{tab:clean}}): the year-adjusted post-appointment gap among destinations with no restatement filed before the appointment.\end{{minipage}}\end{{sidewaystable}}""")

# ── L2 falsification tests ─────────────────────────────────────────────────────────
an = B["anticipation_exposed5y"]; an_any = B["anticipation_exposed"]; dh = B["detection_hazard"]; wv = B["wave_cleaned"]; pp = B["past_outcome_placebo"]; fl = B["future_career_link_placebo"]; ts = B["two_step_placebo"]; sy = B["symmetry"]
def hr(d): return f"{d['hr']:.3f} [{d['ci95'][0]:.3f}, {d['ci95'][1]:.3f}]" if "hr" in d else "---"
def cb(d): return f"${d['coef_pp']:+.3f}$ ({d['se_pp']:.3f}){st(d['t'])}"
w("tab_falsification.tex", rf"""\begin{{table}}[htbp]\centering\footnotesize\renewcommand{{\arraystretch}}{{0.95}}
\caption{{Falsification tests: anticipatory recruiting, detection, template waves, placebo exposures, network distance, and direction.}}
\label{{tab:falsification}}
\begin{{tabular}}{{@{{}}p{{6.2cm}}ccp{{4.8cm}}@{{}}}}\toprule
Test & Estimate & SE & Comparison or note \\\midrule
\multicolumn{{4}}{{@{{}}p{{15.8cm}}@{{}}}}{{\emph{{Panel A. Anticipatory recruiting: does latent trouble at the destination predict appointing an exposed director?}}}} \\
Destination latently misstating at the appointment ({pc(an['share_dst_latent'])}\% of pairs) $\rightarrow$ appoints a director from an origin that restated within 5 years, pp & ${an['latent_dst_coef_pp']:+.2f}$ & {an['se_pp']:.2f} & own prior restatement ${an['own_history_coef_pp']:+.2f}$; filed in prior 12 months ${an['filed12m_coef_pp']:+.2f}$ ({an['se_filed12m_pp']:.2f}) \\
\quad same, origin restated at any time & ${an_any['latent_dst_coef_pp']:+.2f}$ & {an_any['se_pp']:.2f} & own prior restatement ${an_any['own_history_coef_pp']:+.2f}$ \\
\midrule
\multicolumn{{4}}{{@{{}}p{{15.8cm}}@{{}}}}{{\emph{{Panel B. Detection: hazard of the revealing filing among latently misstating destinations ($N={dh['n_latent_destinations_pairs']:,}$ pairs)}}}} \\
Origin restated within 5 years before the tie (Cox, formation-year strata, destination clusters) & \multicolumn{{2}}{{c}}{{{hr(dh['exposed5y'])}}} & median years to filing {dh['median_years_to_reveal']:.2f} \\
Director sat through the origin's misstated period & \multicolumn{{2}}{{c}}{{{hr(dh['revealed'])}}} & $>1$ would indicate faster detection (learning) \\
\midrule
\multicolumn{{4}}{{@{{}}p{{15.8cm}}@{{}}}}{{\emph{{Panel C. Template waves: recent exposure by whether the origin filed in a wave ({wv['n_wave_quarters']} quarters; {pc(wv['share_exposed5y_waveonly'])}\% wave-only)}}}} \\
Origin filed outside waves ($N={wv['nonwave']['n_exposed']:,}$ exposed): change in onset gap, pp & ${wv['nonwave']['did_onset']['diff_in_gaps_pp']:+.2f}$ & {wv['nonwave']['did_onset']['se_pp']:.2f} & never-filed post gap {g(wv['nonwave']['clean_dst_post_onset'])} \\
Origin filed only in waves ($N={wv['waveonly']['n_exposed']:,}$ exposed): change in onset gap, pp & ${wv['waveonly']['did_onset']['diff_in_gaps_pp']:+.2f}$ & {wv['waveonly']['did_onset']['se_pp']:.2f} & never-filed post gap {g(wv['waveonly']['clean_dst_post_onset'])} \\
\midrule
\multicolumn{{4}}{{@{{}}p{{15.8cm}}@{{}}}}{{\emph{{Panel D. Past-outcome placebo: exposed--clean gap in destination onsets 6 to 3 years before the appointment}}}} \\
Origin restated within 5 years before the tie & ${pp['exposed5y']['onset_6to3y_before']['diff_pp']:+.2f}$ & {pp['exposed5y']['onset_6to3y_before']['se_pp']:.2f} & 3 to 0 years before: {g(pp['exposed5y']['onset_3y_before'])} \\
Origin restated at any time before the tie & ${pp['exposed']['onset_6to3y_before']['diff_pp']:+.2f}$ & {pp['exposed']['onset_6to3y_before']['se_pp']:.2f} & 3 to 0 years before: {g(pp['exposed']['onset_3y_before'])} \\
\midrule
\multicolumn{{4}}{{@{{}}p{{15.8cm}}@{{}}}}{{\emph{{Panel E. Future-career-link placebo: the director later joins a recently restating firm ($N={fl['n_pairs_with_future_seat']:,}$ pairs with a later seat)}}}} \\
Post-formation onset gap by future exposure, pp & ${fl['post_onset']['diff_pp']:+.2f}$ & {fl['post_onset']['se_pp']:.2f} & pre-formation gap {g(fl['pre_onset'])}; never-filed destinations {g(fl['clean_dst_post_onset'])} \\
\midrule
\multicolumn{{4}}{{@{{}}p{{15.8cm}}@{{}}}}{{\emph{{Panel F. Network distance: next-year onsets in the firm-year panel (firm and year effects)}}}} \\
Recent exposed interlocks formed in $t-1$, pp per tie & {cb(ts['restate_onset']['recent_exposed_ties_lag1'])} & & same model, share of two-step neighbours (no shared director) that restated within 5 years: {cb(ts['restate_onset']['twostep_recent_share_lag1'])} \\
Share of direct neighbours that restated within 5 years & {cb(ts['restate_onset']['direct_recent_share_lag1_alt'])} & & two-step share in the same model: {cb(ts['restate_onset']['twostep_recent_share_lag1_alt'])} \\
\midrule
\multicolumn{{4}}{{@{{}}p{{15.8cm}}@{{}}}}{{\emph{{Panel G. Direction: change in the onset gap for recent exposure, appointment versus reverse direction}}}} \\
All pairs & ${sy['exposed5y']['forward_all']['did_onset']['diff_in_gaps_pp']:+.2f}$ & {sy['exposed5y']['forward_all']['did_onset']['se_pp']:.2f} & reverse ${sy['exposed5y']['reverse_all']['did_onset']['diff_in_gaps_pp']:+.2f}$ ({sy['exposed5y']['reverse_all']['did_onset']['se_pp']:.2f}) \\
At-risk firm never filed before the tie & ${sy['exposed5y']['forward_history_free']['did_onset']['diff_in_gaps_pp']:+.2f}$ & {sy['exposed5y']['forward_history_free']['did_onset']['se_pp']:.2f} & reverse ${sy['exposed5y']['reverse_history_free']['did_onset']['diff_in_gaps_pp']:+.2f}$ ({sy['exposed5y']['reverse_history_free']['did_onset']['se_pp']:.2f}) \\
At-risk firm with a prior filing & ${sy['exposed5y']['forward_with_history']['did_onset']['diff_in_gaps_pp']:+.2f}$ & {sy['exposed5y']['forward_with_history']['did_onset']['se_pp']:.2f} & reverse ${sy['exposed5y']['reverse_with_history']['did_onset']['diff_in_gaps_pp']:+.2f}$ ({sy['exposed5y']['reverse_with_history']['did_onset']['se_pp']:.2f}) \\
\bottomrule\end{{tabular}}\par\vspace{{3pt}}
\begin{{minipage}}{{\textwidth}}\scriptsize \emph{{Notes.}} Appointment-direction pairs formed 2004--2022 unless stated. Panel A is a linear probability model with formation-year effects; ``latently misstating'' means a misstated period had begun but
had not been filed at the appointment. Panel B is a Cox model with formation-year strata and destination clusters; a hazard ratio above one would indicate faster detection. Panel E uses seats that begin more than three years
after the appointment. Waves are calendar quarters in which restatement filings exceed twice the trailing eight-quarter median,
plus blank-check filings between 12 April and 30 June 2021. In Panel F the two-step share is computed on the interlock network alive on 1 January of $t-1$ among firms reachable through one intermediary
and not directly connected; both models include the count of new interlocks, the number of two-step neighbours, lagged degree and betweenness, size, leverage, ROA and the own-history indicator. Panels A--E and G use
two-way (origin, destination) clustered standard errors; Panel F clusters by firm. Stars as in Table~\ref{{tab:prepost}}.\end{{minipage}}\end{{table}}""")

# ── L3 frozen-network event study ──────────────────────────────────────────────────
def et(y): return " & ".join(f"${C[y]['event_time_vs_k_minus1'][k]['did_pp']:+.2f}$ ({C[y]['event_time_vs_k_minus1'][k]['se_pp']:.2f})" for k in ("-3", "-2", "0", "1", "2"))
def cs(d): return f"${d['did_pp']:+.2f}$ ({d['se_pp']:.2f}){st(d['t'])}"
def rob_row(lbl, key, sub=None):
    cells = []
    for y in ("y_file", "y_onset", "y_legacy"):
        w_ = C[y]["window_robustness"][key] if sub is None else C[y]["window_robustness"][key][sub]
        a_ = C[y]["after_vs_before_robustness"][key] if sub is None else C[y]["after_vs_before_robustness"][key][sub]
        cells += [cs(w_), cs(a_)]
    return lbl + " & " + " & ".join(cells) + " \\\\\n"
w("tab_eventstudy.tex", rf"""\begin{{sidewaystable}}[htbp]\centering\def\baselinestretch{{1}}\footnotesize\setlength{{\tabcolsep}}{{4pt}}
\caption{{Frozen-network event study: outcomes at destinations linked to an origin a year before the origin's restatement filing, relative to matched unlinked registrants.}}
\label{{tab:eventstudy}}
\adjustbox{{max width=\linewidth}}{{\begin{{tabular}}{{p{{4.6cm}}ccccccccc}}\toprule
 & Years $-1$ and $0$ & Years $+1$, $+2$ & Post $-$ pre & \multicolumn{{5}}{{c}}{{Year relative to the origin's filing, versus year $-1$ (pp, SE)}} \\
Destination outcome & vs other years & vs $-3$, $-2$ & (years $\ge 0$ vs $<0$) & $-3$ & $-2$ & $0$ & $+1$ & $+2$ \\\midrule
Restatement filed & {cs(C['y_file']['window_minus1_0_vs_other_years'])} & {cs(C['y_file']['after_1_2_vs_before_minus3_minus2'])} & {cs(C['y_file']['did_post_vs_pre'])} & {et('y_file')} \\
Misstated period begins & {cs(C['y_onset']['window_minus1_0_vs_other_years'])} & {cs(C['y_onset']['after_1_2_vs_before_minus3_minus2'])} & {cs(C['y_onset']['did_post_vs_pre'])} & {et('y_onset')} \\
Filing of a period begun before the window (legacy error) & {cs(C['y_legacy']['window_minus1_0_vs_other_years'])} & {cs(C['y_legacy']['after_1_2_vs_before_minus3_minus2'])} & {cs(C['y_legacy']['did_post_vs_pre'])} & {et('y_legacy')} \\
\bottomrule\end{{tabular}}}}\par\vspace{{4pt}}
\adjustbox{{max width=\linewidth}}{{\begin{{tabular}}{{p{{5.4cm}}cccccc}}\toprule
 & \multicolumn{{2}}{{c}}{{Restatement filed}} & \multicolumn{{2}}{{c}}{{Misstated period begins}} & \multicolumn{{2}}{{c}}{{Legacy error filed}} \\
Robustness of the two contrasts & years $-1$, $0$ & years $+1$, $+2$ & years $-1$, $0$ & years $+1$, $+2$ & years $-1$, $0$ & years $+1$, $+2$ \\\midrule
{rob_row("Destinations and controls with a feed event after year $+1$", "survivors_1y")}{rob_row("Excluding template-wave quarters and blank-check firms", "excluding_wave_and_blank_check")}{rob_row("Events filed 2008--2013", "by_event_period", "2008-2013")}{rob_row("Events filed 2014--2019", "by_event_period", "2014-2019")}{rob_row("Events filed 2020--2022", "by_event_period", "2020-2022")}\bottomrule\end{{tabular}}}}\par\vspace{{3pt}}
\begin{{minipage}}{{\linewidth}}\footnotesize \emph{{Notes.}} Events are the {C['n_events']:,} restatement filings in 2008--2022 by origins with no filing in the preceding three years that have at least one linked destination and enough controls. A destination is linked if an interlock with the origin
(primary sample, either direction) was alive twelve months before the filing; the assignment is kept fixed thereafter. Linked destinations and controls must have a board event in the feed at or before that date and one at or after it, with no condition on events after the filing; the share whose last feed event precedes year $+1$ is {pc(C['attrition']['share_treated_last_event_before_f_plus_1y'])}\% for linked destinations and {pc(C['attrition']['share_control_last_event_before_f_plus_1y'])}\% for controls. Each of the {C['n_treated_units']:,} linked destination-event pairs ({C['n_destinations_treated']:,} destinations) is matched to
three registrants not linked to the origin with the same restatement history at the freeze date (filed within the three years before it $\times$ filed earlier; {C['n_control_units']:,} control units). Outcomes are measured in yearly windows from three years
before to two years after the filing; the regressions include destination $\times$ event and event $\times$ event-time fixed effects (balanced two-way within transformation). The first contrast compares the two years around the origin's filing (the year before it and the year of it) with the other four; the second compares years $+1$ and $+2$ with years $-3$ and $-2$, leaving the two years around the filing out. Pre-period means for linked destinations:
filing {pc(C['y_file']['pre_mean_treated'])}\%, onset {pc(C['y_onset']['pre_mean_treated'])}\% (controls {pc(C['y_file']['pre_mean_control'])}\% and {pc(C['y_onset']['pre_mean_control'])}\%). The linking director sat through the origin's misstated period for {pc(C['share_treated_served'])}\% of linked destinations. Standard errors clustered by destination and by event; stars as in Table~\ref{{tab:prepost}}.\end{{minipage}}\end{{sidewaystable}}""")

# ── L4 bounds ───────────────────────────────────────────────────────────────────────
db = D["drift_bounds_recent_onset"]; ch = D["cinelli_hazlett"]; mo = D["monotone_selection_bound_recent_onset"]; pd_ = D["placebo_formation_dates"]; pm = D["permutation_recent_onset_change"]
grid_rows = "".join(f"$M={M}$ & " + " & ".join(f"[{db['grid'][f'M={M},jump={b}']['identified_set_pp'][0]:+.1f}, {db['grid'][f'M={M},jump={b}']['identified_set_pp'][1]:+.1f}]" for b in (0.0, 0.5, 1.0, 2.0)) + " \\\\\n" for M in (0.0, 0.5, 1.0, 2.0))
ci_rows = "".join(f"$M={M}$ & " + " & ".join(f"[{db['grid'][f'M={M},jump={b}']['ci95_pp'][0]:+.1f}, {db['grid'][f'M={M},jump={b}']['ci95_pp'][1]:+.1f}]" for b in (0.0, 0.5, 1.0, 2.0)) + " \\\\\n" for M in (0.0, 0.5, 1.0, 2.0))
ew = K["K1_pairs"]["equal_window_3y"]
w("tab_bounds.tex", rf"""\begin{{table}}[htbp]\centering\footnotesize
\caption{{Sensitivity ranges for the cohort-adjusted change in the onset gap under an assumed differential drift and an appointment-date selection jump (recent exposure).}}
\label{{tab:bounds}}
\begin{{tabular}}{{lcccc}}\toprule
 & \multicolumn{{4}}{{c}}{{Appointment-date selection jump $b$ (pp)}} \\
Differential-trend multiple $M$ & $b=0$ & $b=0.5$ & $b=1$ & $b=2$ \\\midrule
\multicolumn{{5}}{{@{{}}p{{15.8cm}}@{{}}}}{{\emph{{Panel A. Point-estimate range, pp (no sampling error)}}}} \\
{grid_rows}\midrule
\multicolumn{{5}}{{@{{}}p{{15.8cm}}@{{}}}}{{\emph{{Panel B. The same ranges widened by $\pm 1.96$ SE}}}} \\
{ci_rows}\bottomrule\end{{tabular}}\par\vspace{{3pt}}
\begin{{minipage}}{{\textwidth}}\footnotesize \emph{{Notes.}} The sets are centred on the change in the onset gap for recent exposure with formation-year and formation-year $\times$ post effects, ${db['headline_change_pp']:+.2f}$ (SE {db['headline_se_pp']:.2f}); the unadjusted change is ${db['unadjusted_headline_change_pp']:+.2f}$ (SE {db['unadjusted_headline_se_pp']:.2f}).
The range is the adjusted change $\pm$ ($M \times$ the largest year-to-year movement of the cohort-adjusted pre-formation gap, {db['max_pre_period_year_to_year_change_pp']:.2f} pp, plus an appointment-date jump $b$). The yearly
cohort-adjusted gaps (exposed minus clean, within formation-year $\times$ event-time cells) are {", ".join(f"{k}: {v:+.2f}" for k, v in db['event_time_gaps_pp'].items())} (years relative to formation; mean after minus mean before ${db['mean_post_minus_mean_pre_pp']:+.2f}$);
the unadjusted yearly gaps are {", ".join(f"{k}: {v:+.2f}" for k, v in db['event_time_gaps_unadjusted_pp'].items())}. A differential drift of ${db['drift_to_hide_zero_effect_pp']:+.2f}$ pp over the window would be needed for the true effect to be zero, and of ${db['drift_to_hide_one_point_effect_pp']:+.2f}$ pp for a one-point increase to be hidden.\end{{minipage}}\end{{table}}""")

w("tab_bounds2.tex", rf"""\begin{{table}}[htbp]\centering\footnotesize
\caption{{Omitted-variable robustness values, conditional bounds under monotone selection, placebo formation dates, and permutation inference for the recent-exposure estimates.}}
\label{{tab:bounds2}}
\begin{{tabular}}{{@{{}}p{{8.6cm}}p{{7.2cm}}@{{}}}}\toprule
\multicolumn{{2}}{{@{{}}p{{15.8cm}}@{{}}}}{{\emph{{Panel A. Within-firm recent-exposure coefficients on next-year onsets (Table~\ref{{tab:panel}}), Cinelli--Hazlett robustness}}}} \\\midrule
Row (13): recent exposed ties, own history held fixed; coefficient {ch['row13_recent_onset_plus_prior']['coef_pp']:+.3f} pp ($t={ch['row13_recent_onset_plus_prior']['t']:.2f}$) & RV$_{{q=1}}$ = {ch['row13_recent_onset_plus_prior']['RV_q1']:.4f}; RV$_{{q=1,\alpha=0.05}}$ = {ch['row13_recent_onset_plus_prior']['RV_q1_alpha05']:.4f}; bias from a confounder as strong as own history {ch['row13_recent_onset_plus_prior']['bias_from_confounder_as_strong_as_own_history_pp']:.2f} pp, three times as strong {ch['row13_recent_onset_plus_prior']['bias_from_confounder_3x_own_history_pp']:.2f} pp \\
Row (16): inbound recent exposed ties, own history held fixed; coefficient {ch['row16_inbound_recent_onset_plus_prior']['coef_pp']:+.3f} pp ($t={ch['row16_inbound_recent_onset_plus_prior']['t']:.2f}$) & RV$_{{q=1}}$ = {ch['row16_inbound_recent_onset_plus_prior']['RV_q1']:.4f}; RV$_{{q=1,\alpha=0.05}}$ = {ch['row16_inbound_recent_onset_plus_prior']['RV_q1_alpha05']:.4f}; bias {ch['row16_inbound_recent_onset_plus_prior']['bias_from_confounder_as_strong_as_own_history_pp']:.2f} pp, three times as strong {ch['row16_inbound_recent_onset_plus_prior']['bias_from_confounder_3x_own_history_pp']:.2f} pp \\
\midrule
\multicolumn{{2}}{{@{{}}p{{15.8cm}}@{{}}}}{{\emph{{Panel B. Conditional bounds on the recent-exposure effect on onsets under monotone selection (assumption-dependent, not identified)}}}} \\\midrule
If destinations that appoint from recently restating origins have weakly higher latent risk: the year-adjusted post-formation level gap bounds the effect from above & ${mo['post_level_gap_pp']:+.2f}$ pp (one-sided 95\% limit {mo['post_level_upper95_pp']:.2f}); unadjusted ${mo['unadjusted']['post_level_gap_pp']:+.2f}$ (limit {mo['unadjusted']['post_level_upper95_pp']:.2f}) \\
If selection is also stationary across the two windows: the cohort-adjusted change bounds it & ${mo['change_pp']:+.2f}$ pp (one-sided 95\% limit {mo['change_upper95_one_sided_pp']:.2f}); unadjusted ${mo['unadjusted']['change_pp']:+.2f}$ (limit {mo['unadjusted']['change_upper95_one_sided_pp']:.2f}) \\
\midrule
\multicolumn{{2}}{{@{{}}p{{15.8cm}}@{{}}}}{{\emph{{Panel C. Placebo formation dates and permutation inference}}}} \\\midrule
Formation date shifted 3 years earlier (windows $[-6,-3)$ vs $[-3,0)$): change in onset gap, cohort-adjusted & {dd(pd_['minus_3y']['exposed5y_yfe'])}; unadjusted {dd(pd_['minus_3y']['exposed5y'])} \\
Formation date shifted 3 years later (windows $[0,3)$ vs $[3,6)$; interlocks formed by 2019), cohort-adjusted & {dd(pd_['plus_3y']['exposed5y_yfe'])}; unadjusted {dd(pd_['plus_3y']['exposed5y'])} \\
Recent exposure permuted within formation-year $\times$ destination-size-decile cells ({pm['n_perm']:,} draws, {pm['n_cells']} cells) & within-cell estimate ${pm['observed_pp']:+.2f}$ pp; permutation SD {pm['perm_sd_pp']:.2f}; two-sided $p={pm['p_two_sided']:.3f}$ \\
\bottomrule\end{{tabular}}\par\vspace{{3pt}}
\begin{{minipage}}{{\textwidth}}\footnotesize \emph{{Notes.}} RV$_q$ is the share of residual variance that an omitted variable would need to explain in both the exposure and the outcome to reduce the coefficient by
100$q$ percent (to zero for $q=1$); the own-history benchmark uses the partial $R^2$ of the firm's own prior-restatement indicator, computed as the squared correlation of residuals after partialling out the other regressors and both fixed effects (Frisch--Waugh--Lovell), not from the cluster-robust $t$ statistic ({ch['row13_recent_onset_plus_prior']['partial_R2_Y_ownhistory']:.4f} for the outcome,
{ch['row13_recent_onset_plus_prior']['partial_R2_D_ownhistory']:.5f} for exposure). The permutation reassigns recent exposure within cells and recomputes the within-cell coefficient of the pair's post-minus-pre onset indicator;
size deciles use Compustat log assets, or years in the feed where assets are missing. Cohort-adjusted placebo changes include formation-year and formation-year $\times$ post effects, as the headline change does. Stars as in Table~\ref{{tab:prepost}}.\end{{minipage}}\end{{table}}""")

# ── L5 evidence map ─────────────────────────────────────────────────────────────────
EW = K["K1_pairs"]["equal_window_3y"]; CLN = K["K1_pairs"]["clean_destinations_no_history"]; M = K["K2_panel"]["models"]; S5 = K["K5_survival"]["prior_5y"]
def ci(e, se): return f"${e:+.2f}$ [{e-1.96*se:+.2f}, {e+1.96*se:+.2f}]"
pr = P["excl_bc_and_destination_latent"]; ts = B["two_step_placebo"]; sc = ts["scales"]
rows = [
 ("Pre-appointment filing gap, any exposure (3-year windows)", ci(EW["pre_file"]["diff_pp"], EW["pre_file"]["se_pp"]), "Cross-pair comparison of appointing boards before the tie", "Two-way clustering; formation-year effects in the robustness table", "Destinations differ in observed risk before the tie; age, size and industry composition explain part of it", "Boards that appoint from restating firms were already restating more often; limits accounts in which the appointment initiates the association"),
 ("Change in the onset gap, any exposure, formation-year $\\times$ post effects", ci(EW["did_onset_cohortFE"]["diff_in_gaps_pp"], EW["did_onset_cohortFE"]["se_pp"]), "Within-pair pre/post contrast, cohorts held fixed", "Cohort-common post changes removed; pair-level selection not", "Parallel counterfactual drift of exposed and clean destinations", "No widening of misstatement onsets after the tie"),
 ("Change in the onset gap, recent exposure: unadjusted / formation-year / year $\\times$ size cells", f"{ci(EW['did_onset_recent']['diff_in_gaps_pp'], EW['did_onset_recent']['se_pp']) if 'did_onset_recent' in EW else ci(K['K1_pairs']['exposure_recent_5y']['did_onset']['diff_in_gaps_pp'], K['K1_pairs']['exposure_recent_5y']['did_onset']['se_pp'])} / {ci(EW['did_onset_cohortFE_recent']['diff_in_gaps_pp'], EW['did_onset_cohortFE_recent']['se_pp'])} / {ci(EW['did_onset_cellFE_recent']['diff_in_gaps_pp'], EW['did_onset_cellFE_recent']['se_pp'])}", "Same contrast; cohort composition differs by exposure definition", "Formation-year and size-cell effects", "As above; the unadjusted estimate mixes cohorts", f"The apparent widening is not distinguishable from zero once cohorts are held fixed; adjusted estimates are {sw(EW['did_onset_cohortFE_recent']['diff_in_gaps_pp'])} (formation-year effects) and {sw(EW['did_onset_cellFE_recent']['diff_in_gaps_pp'])} (size cells)"),
 ("Never-filed destinations, recent exposure: post gap unadjusted / year-adjusted", f"{ci(CLN['exposed5y']['post_onset']['diff_pp'], CLN['exposed5y']['post_onset']['se_pp'])} / {ci(CLN['exposed5y']['post_onset_yearFE']['diff_pp'], CLN['exposed5y']['post_onset_yearFE']['se_pp'])}", "Level comparison in a retrospectively selected subsample", "Formation-year effects", "Selection on subsequent revelation; latent risk", "Level difference is a cohort effect"),
 ("Unrevealed-source exposure, non-blank-check, destination without its own unrevealed error: year-adjusted post gap / change", f"{ci(pr['post_onset_yfe']['diff_pp'], pr['post_onset_yfe']['se_pp'])} / {ci(pr['change_onset_yfe']['diff_in_gaps_pp'], pr['change_onset_yfe']['se_pp'])}", f"Origin restatement not yet filed at the appointment (median {lo['median_months_tie_to_origin_filing']:.1f} months later)", "Symmetric comparator; formation-year effects", "Private knowledge; shared conditions; selection on subsequent revelation", f"{sw(pr['post_onset_yfe']['diff_pp']).capitalize()} but imprecise level; {'no estimated widening' if pr['change_onset_yfe']['diff_in_gaps_pp'] <= 0 else 'estimated widening of ' + format(pr['change_onset_yfe']['diff_in_gaps_pp'], '+.1f')}; descriptive partition"),
 ("Within-firm panel: recently exposed ties formed in $t-1$ $\\rightarrow$ onsets in $t$ (own history held fixed), per tie", ci(100 * M["lag1_onset_exposed5y_plus_prior"]["exposure_coef"]["coef"], 100 * M["lag1_onset_exposed5y_plus_prior"]["exposure_coef"]["se"]), "Within-firm variation in new ties, all new ties held fixed", "Firm and year effects; own prior restatement", "Time-varying recruitment and shared conditions within the firm's community", "Positive association, not director-specific (see next row)"),
 ("Share of two-step neighbours (no shared director) that restated within 5 years $\\rightarrow$ onsets, per 10 points of the share", ci(sc["onset_effect_pp_per_10pp_twostep_share"], ts["restate_onset"]["twostep_recent_share_lag1"]["se_pp"] * 0.10), "Same panel and controls as the row above", "Firm and year effects", "Community-level correlation of restatement risk", "Firms with no director link to a recent restater show an association of the same sign"),
 ("Frozen-network event study, destinations linked a year before the origin's filing: filings in years $-1$ and $0$ vs other years / years $+1$, $+2$ vs $-3$, $-2$; onsets, same two contrasts", f"{ci(C['y_file']['window_minus1_0_vs_other_years']['did_pp'], C['y_file']['window_minus1_0_vs_other_years']['se_pp'])} / {ci(C['y_file']['after_1_2_vs_before_minus3_minus2']['did_pp'], C['y_file']['after_1_2_vs_before_minus3_minus2']['se_pp'])}; onsets {ci(C['y_onset']['window_minus1_0_vs_other_years']['did_pp'], C['y_onset']['window_minus1_0_vs_other_years']['se_pp'])} / {ci(C['y_onset']['after_1_2_vs_before_minus3_minus2']['did_pp'], C['y_onset']['after_1_2_vs_before_minus3_minus2']['se_pp'])}", "Timing of linked destinations' outcomes around an origin's filing, against matched registrants", "Destination $\\times$ event and event $\\times$ year effects; history-matched controls", "A common trigger reaching both firms; the rise begins before the origin files; differential exit from the feed", "Revelations are simultaneous: filings of existing errors cluster around the origin's filing and no new misstatements begin; the design cannot attribute the clustering to the disclosure"),
 *([("Ineffective internal control at the appointing board, origin reported a material weakness within 5 years: gap before / change (cohort-adjusted)", f"{ci(MM['M1_pairs_covered']['icfr_exposed5y']['adverse']['pre']['diff_pp'], MM['M1_pairs_covered']['icfr_exposed5y']['adverse']['pre']['se_pp'])} / {ci(MM['M1_pairs_covered']['icfr_exposed5y']['adverse']['did_cohortFE']['diff_in_gaps_pp'], MM['M1_pairs_covered']['icfr_exposed5y']['adverse']['did_cohortFE']['se_pp'])}", "Second outcome with a ten-week revelation lag; pairs with an opinion in both windows", "Formation-year $\\times$ post effects", "Coverage limited to filers with SOX 404 opinions; exposure and outcome share a measurement", "The pre-existing difference is large and unchanged by the appointment")] if MM else []),
 ("How ties end, recent exposure: origin-seat / destination-seat hazard (covariates)", f"{S5['cox_origin_seat_ended_covariates']['hr']:.2f} [{S5['cox_origin_seat_ended_covariates']['ci95'][0]:.2f}, {S5['cox_origin_seat_ended_covariates']['ci95'][1]:.2f}] / {S5['cox_destination_seat_ended_covariates']['hr']:.2f} [{S5['cox_destination_seat_ended_covariates']['ci95'][0]:.2f}, {S5['cox_destination_seat_ended_covariates']['ci95'][1]:.2f}]", "Which seat ends an active interlock", "Formation-year strata; origin clustering; director and firm covariates", "Seat endings reflect resignations, mergers and dissolutions as well as board decisions", "Ties end at the restating firm's seat, not the appointing board's; describes where ties end, not what boards decide"),
]
body = "".join(f"{a} & {b} & {c} & {d} & {e} & {f} \\\\\n" for a, b, c, d, e, f in rows)
w("tab_evidence.tex", rf"""\begin{{sidewaystable}}[htbp]\centering\def\baselinestretch{{1}}\footnotesize\setlength{{\tabcolsep}}{{4pt}}
\caption{{Evidence map: what each headline estimate measures, what the adjustment removes, what remains unresolved, and the interpretation the result supports.}}
\label{{tab:evidence}}
\begin{{tabular}}{{p{{4.9cm}}p{{3.5cm}}p{{3.1cm}}p{{2.8cm}}p{{3.1cm}}p{{3.6cm}}}}\toprule
Estimate (pp unless stated) & Estimate [95\% interval] & Source of variation & Adjustment removes & Unresolved assumption & Narrow interpretation \\\midrule
{body}\bottomrule\end{{tabular}}\par\vspace{{3pt}}
\begin{{minipage}}{{\linewidth}}\footnotesize \emph{{Notes.}} Intervals are $\pm 1.96$ standard errors with two-way (origin, destination) clustering for pair-level estimates, firm clustering for the panel, and destination and event clustering for the event study.
None of the rows identifies the causal effect of appointing a restatement-exposed director; the table records what each design holds fixed and what it leaves open. Minimum detectable effects and support counts are in the online appendix.\end{{minipage}}\end{{sidewaystable}}""")

# ── residual specificity (main text) ────────────────────────────────────────────────
Mk = K["K2_panel"]["models"]
def mrow(lbl, k):
    m = Mk[k]["exposure_coef"]; return f"{lbl} & ${100*m['coef']:+.3f}$ ({100*m['se']:.3f}){st(m['t'])} & {Mk[k]['n']:,} \\\\\n"
r_on = ts["restate_onset"]; r_fi = ts["restate_file"]
def tsr(lbl, d, n): return f"{lbl} & ${d['coef_pp']:+.3f}$ ({d['se_pp']:.3f}){st(d['t'])} & {n:,} \\\\\n"
def esr(lbl, y):
    d = C[y]["did_post_vs_pre"]; return f"{lbl} & ${d['did_pp']:+.2f}$ ({d['se_pp']:.2f}){st(d['t'])} & {C['n_treated_units']:,} / {C['n_control_units']:,} \\\\\n"
def esw(lbl, y, key):
    d = C[y][key]; return f"{lbl} & ${d['did_pp']:+.2f}$ ({d['se_pp']:.2f}){st(d['t'])} & {C['n_treated_units']:,} / {C['n_control_units']:,} \\\\\n"
w("tab_residual.tex", rf"""\begin{{table}}[htbp]\centering\footnotesize
\caption{{Is the remaining association specific to the director's seat? Within-firm exposure, network distance, and the timing of linked destinations' outcomes around an origin's disclosure.}}
\label{{tab:residual}}
\begin{{tabular}}{{@{{}}p{{10.2cm}}cc@{{}}}}\toprule
 & Coefficient (SE) & $N$ \\\midrule
\multicolumn{{3}}{{@{{}}p{{15.8cm}}@{{}}}}{{\emph{{Panel A. Firm-year panel, next-year misstated-period onsets (pp); firm and year effects and all new interlocks held fixed; the firm's own restatement history held fixed in every row}}}} \\
{mrow("Exposed interlocks formed in $t-1$, any prior restatement at the other firm, per tie", "lag1_onset_plus_prior")}{mrow("Recently (5-year) exposed interlocks formed in $t-1$, per tie", "lag1_onset_exposed5y_plus_prior")}{mrow("\\quad inbound only (director joined the focal board), per tie", "lag1_onset_inbound_recent5y_plus_prior")}{mrow("\\quad stock of recently exposed interlocks alive on 1 January, per tie", "boy_stock_onset_exposed5y_plus_prior")}{tsr("Share of two-step neighbours (no shared director) that restated within 5 years, per unit share", r_on["twostep_recent_share_lag1"], r_on["n"])}{tsr("\\quad recently exposed interlocks formed in $t-1$ in the same regression, per tie", r_on["recent_exposed_ties_lag1"], r_on["n"])}\midrule
\multicolumn{{3}}{{@{{}}p{{15.8cm}}@{{}}}}{{\emph{{Panel B. Frozen-network event study: linked destinations versus matched controls (pp); $N$ = linked / control units}}}} \\
{esw("Restatement filed: years $-1$ and $0$ around the origin's filing vs other years", "y_file", "window_minus1_0_vs_other_years")}{esw("\\quad years $+1$ and $+2$ vs years $-3$ and $-2$", "y_file", "after_1_2_vs_before_minus3_minus2")}{esw("Misstated period begins: years $-1$ and $0$ vs other years", "y_onset", "window_minus1_0_vs_other_years")}{esw("\\quad years $+1$ and $+2$ vs years $-3$ and $-2$", "y_onset", "after_1_2_vs_before_minus3_minus2")}{esw("Filing of a period begun before the window (legacy error): years $-1$ and $0$ vs other years", "y_legacy", "window_minus1_0_vs_other_years")}{esw("\\quad years $+1$ and $+2$ vs years $-3$ and $-2$", "y_legacy", "after_1_2_vs_before_minus3_minus2")}\bottomrule\end{{tabular}}\par\vspace{{3pt}}
\begin{{minipage}}{{\textwidth}}\footnotesize \emph{{Notes.}} Panel A: $N={Mk['lag1_onset_plus_prior']['n']:,}$ firm-years, 2005--2023; onset rate {pc(K['K2_panel']['estimation_sample']['rate_onset'],2)}\%; all rows include lagged degree and betweenness, log assets, leverage, ROA,
the count of new interlocks (inbound and outbound separately where stated) and an indicator for any restatement filed before $t$; firm-clustered standard errors. The two-step share is computed on the interlock
network alive on 1 January of $t-1$ among firms reachable through one intermediary and not directly connected; {pc(sc['share_firm_years_with_twostep_share_gt0'])}\% of firm-years have a positive share and {pc(sc['share_firm_years_with_recent_exposed_tie'])}\% a recently exposed new tie,
so the coefficient per unit share corresponds to {sc['onset_effect_pp_per_10pp_twostep_share']:.2f} pp per ten points of the share. Panel B: {C['n_fresh_disclosures']:,} origin filings in 2008--2022 after at least three clean years ({C['n_events']:,} with a linked destination and controls); a destination is linked if an interlock with the origin
was alive twelve months before the filing (assignment kept fixed); linked destinations and controls must have a feed event at or before that date and one at or after it, with no condition on later events; each linked destination-event is matched to three registrants not linked to the origin with the same restatement history
(filed within the three years before the freeze $\times$ filed earlier); destination $\times$ event and event $\times$ event-time effects over years $-3$ to $+2$; standard errors clustered by destination and event. The first contrast compares the year before and the year of the origin's filing with the other four years; the second leaves those two years out. Yearly estimates, the post-minus-pre summary, and
robustness to attrition, template waves, blank-check firms and event period are in Appendix Table~\ref{{tab:eventstudy}}. Stars as in Table~\ref{{tab:prepost}}.\end{{minipage}}\end{{table}}""")
