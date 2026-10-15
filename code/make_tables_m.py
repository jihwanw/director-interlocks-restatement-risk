"""make_tables_m.py — appendix tables for module M (SOX 404 internal-control outcome; appointment-level analysis) from results_rebuild/M_icfr{PRIMARY}.json.
Run from github/: venv python code/make_tables_m.py"""
import json, os
from pathlib import Path
HERE = Path(__file__).resolve().parent
PRIMARY = os.environ.get("PRIMARY_SUFFIX", "_art")
M = json.load(open(HERE.parent / "results_rebuild" / f"M_icfr{PRIMARY}.json"))
OUT = HERE.parent.parent / "3rd_revision" / "source" / "tables_auto"
def w(name, s):
    s = s.replace("\\begin{table}[htbp]\\centering", "\\begin{table}[htbp]\\centering\\def\\baselinestretch{1}")
    (OUT / name).write_text(s); print("wrote", name)
def pc(x, d=1): return f"{100*x:.{d}f}"
def st(t): return "***" if t is not None and abs(t) > 2.576 else "**" if t is not None and abs(t) > 1.96 else "*" if t is not None and abs(t) > 1.645 else ""
def g(d): return f"${d['diff_pp']:+.2f}$ ({d['se_pp']:.2f})" if d else "---"
def dd(d): return f"${d['diff_in_gaps_pp']:+.2f}$ ({d['se_pp']:.2f}){st(d.get('t'))}"
C = M["coverage"]; B = M["M1_pairs_covered"]; B10 = M["M1_pairs_covered_formed_2010plus"]; P = M["M2_panel"]; A3 = M["M3_appointments"]
LAB = {"exposed": "Origin filed a restatement at any time before the tie", "exposed5y": "Origin filed a restatement within 5 years before the tie", "exposed_served": "Director present at the origin during the misstatement",
       "icfr_exposed": "Origin reported ineffective internal control at any time before the tie", "icfr_exposed5y": "Origin reported ineffective internal control within 5 years before the tie"}
def rows(blk, outcome):
    r = ""
    for ex in ("exposed", "exposed5y", "exposed_served", "icfr_exposed", "icfr_exposed5y"):
        if ex not in blk: continue
        v = blk[ex][outcome]; npre = blk[ex]["no_pre_adverse_post_gap" if outcome == "adverse" else "no_pre_adverse_post_new_gap"]
        r += f"{LAB[ex]} & {blk[ex]['n_exposed']:,} & {g(v['pre'])} & {g(v['post'])} & {dd(v['did'])} & {dd(v['did_cohortFE'])} & {g(npre)} \\\\\n"
    return r
def prow(lbl, k):
    m = P["models"][k]["exposure_coef"]; return f"{lbl} & & & & & ${m['coef_pp']:+.3f}$ ({m['se_pp']:.3f}){st(m['t'])} & {P['models'][k]['n']:,} \\\\\n"
w("tab_icfr.tex", rf"""\begin{{table}}[htbp]\centering\footnotesize\setlength{{\tabcolsep}}{{3pt}}
\caption{{A second outcome: ineffective internal control over financial reporting (SOX 404 opinions) at the appointing board, before and after formation, and in the firm-year panel.}}
\label{{tab:icfr}}
\adjustbox{{max width=\textwidth}}{{\begin{{tabular}}{{p{{5.6cm}}cccccc}}\toprule
 & Exposed & Gap before, & Gap after, & Change, & Change, cohort- & Post gap, no adverse \\
Exposure & pairs & pp (SE) & pp (SE) & pp (SE) & adjusted (SE) & opinion before (SE) \\\midrule
\multicolumn{{7}}{{@{{}}p{{15.8cm}}@{{}}}}{{\emph{{Panel A. Destination reports ineffective internal control in the three-year window ($N={B['n_rows']:,}$ pairs with an opinion in both windows; rates {pc(B['pre_rate'])}\% before, {pc(B['post_rate'])}\% after)}}}} \\
{rows(B, "adverse")}\multicolumn{{7}}{{@{{}}p{{15.8cm}}@{{}}}}{{\emph{{Panel B. Destination newly reports ineffective internal control (an adverse opinion following an effective one; rates {pc(B['pre_new_rate'])}\% before, {pc(B['post_new_rate'])}\% after)}}}} \\
{rows(B, "new_adverse")}\multicolumn{{7}}{{@{{}}p{{15.8cm}}@{{}}}}{{\emph{{Panel C. As Panel A, interlocks formed 2010--2022 ($N={B10['n_rows']:,}$)}}}} \\
{rows(B10, "adverse")}\midrule
\multicolumn{{7}}{{@{{}}p{{15.8cm}}@{{}}}}{{\emph{{Panel D. Firm-year panel: ineffective internal control in $t$ on exposed interlocks formed in $t-1$ (pp per interlock; firm and year effects; $N={P['n_firm_years']:,}$ firm-years with an opinion, {P['n_firms']:,} firms; rate {pc(P['adverse_rate'])}\%)}}}} \\
 & & & & & Coefficient (SE) & $N$ \\
{prow("Exposed interlocks (any prior restatement at the other firm), own history held fixed", "adverse_any_exposed_lag1")}{prow("Recently exposed interlocks, own history held fixed", "adverse_recent_exposed_lag1")}{prow("\\quad inbound only", "adverse_inbound_recent_lag1")}{prow("\\quad stock alive on 1 January", "adverse_recent_stock_boy")}{prow("Newly ineffective: recently exposed interlocks", "new_adverse_recent_exposed_lag1")}{prow("Misstated period begins in $t$ (same firm-years, for comparison): recently exposed interlocks", "onset_recent_exposed_lag1_same_firm_years")}\bottomrule\end{{tabular}}}}\par\vspace{{3pt}}
\begin{{minipage}}{{\textwidth}}\footnotesize \emph{{Notes.}} Internal-control opinions are from the Audit Analytics SOX 404 feed (management assessments and auditor attestations); a firm-fiscal-year is ``ineffective'' when any opinion for it reports internal control as not effective (every such opinion row carries at least one material weakness and no effective-opinion row does). Opinions are available for {C['firms_with_opinion']:,} registrants and {C['firm_fys']:,} firm-fiscal-years, {pc(C['firm_fy_adverse_share'])}\% of them ineffective; coverage begins with accelerated filers in FY2004 and extends to management assessments of other filers from FY2007, so Panels A--C keep the {pc(C['share'])}\% of equal-window pairs whose destination has an opinion in each window ({pc(C['share_by_formation_period']['2004-2009'])}\% of pairs formed 2004--2009, {pc(C['share_by_formation_period']['2010-2015'])}\% for 2010--2015, {pc(C['share_by_formation_period']['2016-2022'])}\% for 2016--2022). The internal-control exposure definitions use the origin's own opinions before the tie ({pc(C['icfr_exposure']['share_origin_with_opinion_before_tie'])}\% of origins have one); {pc(C['icfr_exposure']['overlap_icfr5y_with_restatement5y'])}\% of pairs whose origin reported ineffective control within five years also had a restatement filed within five years. Windows, cohort adjustment and standard errors as in Table~\ref{{tab:prepost}}; the last column is the post-formation gap among destinations with no ineffective opinion in the pre-formation window. Panel D uses the controls of Table~\ref{{tab:residual}} on the firm-years with an opinion; standard errors clustered by firm. Stars as in Table~\ref{{tab:prepost}}.\end{{minipage}}\end{{table}}""")

def arow(lbl, ex):
    v = A3[ex]; return f"{lbl} & {v['n_exposed']:,} & {g(v['pre_file'])} & {g(v['post_file'])} & {dd(v['did_file'])} & {g(v['pre_onset'])} & {g(v['post_onset'])} & {dd(v['did_onset'])} & {dd(v['did_onset_cohortFE'])} & {g(v['never_filed_post_onset'])} \\\\\n"
cs = A3["share_recent_continuous_onset_cohortFE"]; ce = A3["share_exposed_continuous_onset_cohortFE"]
w("tab_appointments.tex", rf"""\begin{{table}}[htbp]\centering\footnotesize\setlength{{\tabcolsep}}{{3pt}}
\caption{{Appointment-level analysis: one unit per appointment, with exposure defined over all of the appointee's origin boards.}}
\label{{tab:appointments}}
\adjustbox{{max width=\textwidth}}{{\begin{{tabular}}{{p{{4.4cm}}ccccccccc}}\toprule
 & & \multicolumn{{3}}{{c}}{{Filed, 3-year windows}} & \multicolumn{{4}}{{c}}{{Misstated period begins}} & Never-filed, \\
Exposure of the appointment & Exposed & before & after & change & before & after & change & cohort-adjusted & onset after \\\midrule
{arow("Any origin restated before the tie", "any_exposed")}{arow("Every origin restated before the tie", "all_exposed")}{arow("Any origin restated within 5 years", "any_recent")}{arow("Every origin restated within 5 years", "all_recent")}\bottomrule\end{{tabular}}}}\par\vspace{{3pt}}
\begin{{minipage}}{{\textwidth}}\footnotesize \emph{{Notes.}} The {A3['n_pairs']:,} appointment-direction pairs of Table~\ref{{tab:prepost}} collapse to {A3['n_appointments']:,} appointments (director, appointing board, date); {pc(A3['share_multi_origin'])}\% of appointments create more than one pair because the appointee already sat on several boards ({A3['mean_origins']:.2f} origins per appointment on average), and among those {pc(A3['share_mixed_exposure_among_multi_origin'])}\% combine an exposed and a clean origin ({pc(A3['share_mixed_recent_among_multi_origin'])}\% for recent exposure). Each appointment enters once; its outcomes are the appointing board's, so the comparison is not affected by one appointment contributing exposed and clean pairs with identical outcomes. Treating the share of the appointee's origins that restated within five years as a continuous exposure gives a cohort-adjusted change of ${cs['diff_in_gaps_pp_per_unit']:+.2f}$ pp per unit share (SE {cs['se_pp']:.2f}); for the share with any prior restatement, ${ce['diff_in_gaps_pp_per_unit']:+.2f}$ (SE {ce['se_pp']:.2f}). Standard errors clustered two-way by director and appointing board; cohort adjustment as in Table~\ref{{tab:prepost}}; stars as in Table~\ref{{tab:prepost}}.\end{{minipage}}\end{{table}}""")
