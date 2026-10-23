#!/usr/bin/env python3
"""
split_table_notes.py — run AFTER make_tables_k.py / make_tables_l.py / make_tables_m.py / make_definitions.py (added 2026-10-23).

Author instruction (2026-10-23): a table whose notes run to more than three lines is hard to read; the explanatory
content belongs in the text that introduces the table, and the note should keep only what is needed to read the cells.

For every table in CONFIG this script
  1. extracts the generated Notes block (minipage or tablenotes form),
  2. replaces it with the SHORT note given here (sample/unit, standard errors, stars, pointer to the definitions), and
  3. for appendix tables (lead=True) writes tables_auto/lead_<name>.tex containing the original note text as a
     paragraph headed by the table reference, which main.tex inputs immediately before the table.
Main-text tables (lead=False) have their explanatory content written into the section prose by hand.
Nothing between \\toprule and \\bottomrule is touched; verify_bodies() asserts that.
"""
import re, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TA = HERE.parent.parent / "3rd_revision" / "source" / "tables_auto"
STARS = r"*** $|t|>2.58$, ** $|t|>1.96$, * $|t|>1.645$"
APP = r"Sample, definitions and standard errors as described in the paragraph preceding the table; " + STARS + " where stars appear."

CONFIG = {
    # ---------------- main-text tables: short note here, explanation in the section prose ----------------
    "tab_sample": dict(lead=False, short=r"Horizon: 31 December 2025, the last date with complete coverage in both feeds. Board-member events only (\texttt{is\_bdmem\_pers} $=1$). The spell rules, the name key, the registrant-type definitions and the removal rules are given in Section~\ref{sec:data}; the unrestricted sample and the six-firm cap are in Appendix Table~\ref{tab:samples}."),
    "tab_prepost": dict(lead=False, short=r"$N=34{,}211$ appointment-direction pairs formed 2004--2022; fixed three-year windows before and after formation. The difference is $\delta$ in equation~\ref{eq:stacked}, two-way (origin, destination) clustered standard errors; " + STARS + r". Panel D adds formation-year ($\times$ post) effects; its first column is the adjusted pre-formation gap."),
    "tab_clean": dict(lead=False, short=r"Subsamples of Table~\ref{tab:prepost}; onset $=$ a misstated period begins in the window; year-adjusted adds formation-year ($\times$ post for change rows) effects. Panel C splits the Panel B pairs by destination size tercile, years in the feed, and shared two-digit SIC or headquarters state, for recent exposure. Two-way clustered standard errors."),
    "tab_latent": dict(lead=False, short=r"Unrevealed pairs: $U_p=1$ in equation~\ref{eq:exposure}; comparator and partitions as defined in Section~\ref{sec:latent}. Year-adjusted gaps add formation-year ($\times$ post) effects; two-way (origin, destination) clustered standard errors; 95 percent intervals in brackets."),
    "tab_residual": dict(lead=False, short=r"Panel A: $N=133{,}231$ firm-years, 2005--2023, equation~\ref{eq:panel} with firm-clustered standard errors; onset rate 7.10\%. Panel B: equation~\ref{eq:eventstudy}; 8,666 origin filings in 2008--2022 after three clean years, 2,217 with a linked destination; three matched controls per linked destination; standard errors clustered by destination and event. " + STARS + "."),
    # ---------------- appendix tables: original note becomes the lead paragraph ----------------
    "tab_samples": dict(lead=True, short=APP), "tab_samples_more": dict(lead=True, short=APP),
    "tab_prepost_robust": dict(lead=True, short=APP), "tab_prepost_robust2": dict(lead=True, short=APP),
    "tab_cohort": dict(lead=True, short=APP), "tab_sizetercile": dict(lead=True, short=APP),
    "tab_appointments": dict(lead=True, short=APP), "tab_icfr": dict(lead=True, short=APP),
    "tab_falsification": dict(lead=True, short=APP), "tab_eventstudy": dict(lead=True, short=APP),
    "tab_community": dict(lead=True, short=APP), "tab_panel": dict(lead=True, short=APP),
    "tab_screening": dict(lead=True, short=APP), "tab_auditcomm": dict(lead=True, short=APP),
    "tab_survival": dict(lead=True, short=APP), "tab_survival_compact": dict(lead=True, short=APP),
    "tab_bounds": dict(lead=True, short=APP), "tab_bounds2": dict(lead=True, short=APP), "tab_evidence": dict(lead=True, short=APP),
}

PAT_MINI = re.compile(r"(\\begin\{minipage\}\{\\(?:textwidth|linewidth)\}\\(?:scriptsize|footnotesize|small)\s*\\emph\{Notes\.\})(.*?)(\\end\{minipage\})", re.S)
PAT_TN = re.compile(r"(\\begin\{tablenotes\}\\(?:footnotesize|scriptsize)\\item\s*)(.*?)(\\end\{tablenotes\})", re.S)

def body(s):
    return re.findall(r"\\toprule(.*?)\\bottomrule", s, re.S)

def process(name, cfg):
    p = TA / f"{name}.tex"; s = p.read_text()
    m = PAT_MINI.search(s) or PAT_TN.search(s)
    if not m:
        print(f"  {name}: no notes block found — skipped"); return
    old_note = re.sub(r"\s+", " ", m.group(2)).strip()
    label = re.search(r"\\label\{(tab:[^}]+)\}", s).group(1)
    new = s[:m.start(2)] + " " + cfg["short"] + s[m.end(2):]
    assert body(new) == body(s), f"{name}: table body changed"
    p.write_text(new)
    if cfg["lead"]:
        lead = re.sub(r"\(Table~\\ref\{(tab:[^}]+)\},? note\)", r"(see the paragraph on Table~\\ref{\1})", old_note)
        lead = lead.replace("Other notes as in Table~\\ref{tab:prepost}.", "Standard errors and stars as in Table~\\ref{tab:prepost}.").replace("Stars as in Table~\\ref{tab:prepost}.", "Stars as in Table~\\ref{tab:prepost}.")
        (TA / f"lead_{name}.tex").write_text("\\noindent\\textbf{Table~\\ref{" + label + "}.}~" + lead + "\\par\\medskip\n")
    print(f"  {name}: note {len(old_note)} -> {len(cfg['short'])} chars" + (f"; lead_{name}.tex written" if cfg["lead"] else ""))

if __name__ == "__main__":
    for name, cfg in CONFIG.items():
        process(name, cfg)
