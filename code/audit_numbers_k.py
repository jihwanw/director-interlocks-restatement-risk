#!/usr/bin/env python3
"""audit_numbers_k.py — every number in the manuscript prose must trace to a value in the results JSONs.
Usage (from github/): venv python code/audit_numbers_k.py
Exit 1 if any number is unmatched."""
import json, re, sys
from decimal import Decimal, ROUND_HALF_UP
def rhu(x, d): return float(Decimal(str(x)).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP))
from pathlib import Path
HERE = Path(__file__).resolve().parent
SRC = HERE.parent.parent / "3rd_revision" / "source"
import os
PRIMARY = os.environ.get("PRIMARY_SUFFIX", "_art")
SRC_FILES = [("primary_K", f"K_rebuild{PRIMARY}.json"), ("primary_L", f"L_identification{PRIMARY}.json"), ("primary_M", f"M_icfr{PRIMARY}.json"),
             ("full", "K_rebuild_full.json"), ("le6", "K_rebuild_le6.json"), ("artns", "K_rebuild_artns.json"), ("art_name", "K_rebuild_art_name.json"), ("art_noest", "K_rebuild_art_noest.json"),
             ("inventory", "wrds_inventory/inventory_2026-10-13.json")]
JS = {name: json.load(open(HERE.parent / "results_rebuild" / f)) for name, f in SRC_FILES if (HERE.parent / "results_rebuild" / f).exists()}
J = list(JS.values())
# A number matched ONLY by a non-primary source must appear in a sentence that names that variant (2026-10-15: a stale six-board-cap
# share, 94 percent, had passed the audit because the le6 file is a legitimate comparison source).
CONTEXT = {"full": ("no exclusions", "all directors", "before exclusion", "first-and-last-name", "full sample", "every registrant"),
           "le6": ("six", "alternative definition", "cap"),
           "artns": ("surname",), "art_name": ("first-and-last", "name key", "middle"), "art_noest": ("estimated",),
           "inventory": ("feed", "WRDS", "middle", "committee", "extract", "person identifier", "sich", "SIC", "coverage", "names"),
           "log": ("six", "alternative definition", "cap")}
LOG = (HERE / "K_le6.log").read_text() if (HERE / "K_le6.log").exists() else ""

def collect(j):
    vals = set()
    def walk(o):
        if isinstance(o, dict):
            for v in o.values(): walk(v)
        elif isinstance(o, list):
            for v in o: walk(v)
        elif isinstance(o, bool): pass
        elif isinstance(o, (int, float)): vals.add(float(o))
        elif isinstance(o, str):
            m_ = re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})", o)
            if m_: vals.update(float(x) for x in m_.groups())
    walk(j); return vals
def renderings(v):
    out = set()
    for d in (0, 1, 2, 3):
        out.add(round(v, d)); out.add(round(100 * v, d)); out.add(round(v / 100, d))
        out.add(rhu(v, d)); out.add(rhu(100 * v, d)); out.add(rhu(abs(v), d)); out.add(rhu(100 * abs(v), d))
    out.add(round(100 * (1 - v), 1)); out.add(round(100 * (1 - v), 0))
    return out
def derived(o, R):
    if isinstance(o, dict):
        if "diff_pp" in o and "se_pp" in o:
            for s in (1.96,):
                R.add(round(o["diff_pp"] - s * o["se_pp"], 1)); R.add(round(o["diff_pp"] + s * o["se_pp"], 1))
                R.add(round(abs(o["diff_pp"] - s * o["se_pp"]), 1)); R.add(round(abs(o["diff_pp"] + s * o["se_pp"]), 1))
        if "diff_in_gaps_pp" in o and "se_pp" in o:
            R.add(round(o["diff_in_gaps_pp"] - 1.96 * o["se_pp"], 1)); R.add(round(o["diff_in_gaps_pp"] + 1.96 * o["se_pp"], 1))
            R.add(round(abs(o["diff_in_gaps_pp"] - 1.96 * o["se_pp"]), 1)); R.add(round(abs(o["diff_in_gaps_pp"] + 1.96 * o["se_pp"]), 1))
            if "pre_gap_pp" in o: R.add(round(o["pre_gap_pp"] + o["diff_in_gaps_pp"], 1)); R.add(round(o["pre_gap_pp"] + o["diff_in_gaps_pp"], 2))
        if "did_pp" in o and "se_pp" in o:
            R.add(round(o["did_pp"] - 1.96 * o["se_pp"], 1)); R.add(round(o["did_pp"] + 1.96 * o["se_pp"], 1))
        if "change_pp" in o and "se_change_pp" in o:
            R.add(round(o["change_pp"] - 1.96 * o["se_change_pp"], 1)); R.add(round(o["change_pp"] + 1.96 * o["se_change_pp"], 1))
        if "coef_pp" in o and "se_pp" in o:
            R.add(round(o["coef_pp"] - 1.96 * o["se_pp"], 2)); R.add(round(o["coef_pp"] + 1.96 * o["se_pp"], 2))
        if "coef" in o and "se" in o:
            R.add(round(100 * 1.96 * o["se"], 1)); R.add(round(abs(100 * o["coef"]), 1)); R.add(round(abs(100 * o["coef"]), 2))
        for v in o.values(): derived(v, R)
    elif isinstance(o, list):
        for v in o: derived(v, R)
RENDER_BY = {}
for name, j in JS.items():
    R = set()
    for v in collect(j):
        R |= renderings(v); R.add(abs(v))
        for d in (1, 2): R.add(round(abs(v), d)); R.add(round(100 * abs(v), d))
    derived(j, R); RENDER_BY[name] = R
if LOG:
    R = set()
    for m in re.finditer(r"\d[\d,]*\.?\d*", LOG):
        try: R.add(float(m.group(0).replace(",", "")))
        except ValueError: pass
    RENDER_BY["log"] = R
PRIMARY_SET = set().union(*(R for n, R in RENDER_BY.items() if n.startswith("primary")))
KEYED = {n: {round(x, 3) for x in R} for n, R in RENDER_BY.items()}
def sources_of(v):
    a = round(abs(v), 3); return [n for n, R in KEYED.items() if a in R or any(abs(x - a) < 0.0015 for x in R if abs(x - a) < 0.0015)]
ALLOW = {2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2025, 1500,
         1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 25, 31, 95, 100, 401, 404, 42, 1.96, 2.58, 1.645, 0.01, 0.05, 0.001, 155663, 568332, 279354, 14672,
         0.0, 2.0, 3.0, 5.0, 6.0, 1.0, 6770, 1996, 1997, 1998, 1999, 2000, 2001, 2002, 2003}
# structural: section numbers, years, Regulation S-K items, hypothesis labels (P1..P5), thresholds

text = ""
for f in sorted((SRC / "sections").glob("*.tex")): text += (SRC / "sections" / f.name).read_text() + "\n"
m = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", (SRC / "main.tex").read_text(), re.S); text += m.group(1)
text += (SRC / "highlights.tex").read_text()
text = re.sub(r"\\(cite[tp]?|ref|label|input|includegraphics)\{[^}]*\}", " ", text)
text = re.sub(r"\\(section|subsection)\*?\{[^}]*\}", " ", text)
text = re.sub(r"(?<!\\)%.*", "", text)
text = text.replace("--", " ").replace("{,}", "")
# numbers rendered in the generated tables (for the table-presence list)
TABLE_TEXT = " ".join(f.read_text() for f in (SRC / "tables_auto").glob("*.tex"))
TABLE_NUMS = set()
for m_ in re.finditer(r"\d[\d,]*(?:\.\d+)?", TABLE_TEXT.replace("{,}", "")):
    try: TABLE_NUMS.add(round(float(m_.group(0).replace(",", "")), 3))
    except ValueError: pass
def sentence_of(pos):
    a = max(text.rfind(". ", 0, pos), text.rfind("\n", 0, pos)) + 1; b = text.find(". ", pos); b = len(text) if b < 0 else b + 1
    return text[a:b].strip().replace("\n", " ")
bad = []; ctx_bad = []; nontable = []; n = 0; pos = 0
for mm in re.finditer(r"(?<![A-Za-z_\\])[-+−]?\$?\d[\d,]*(?:\.\d+)?", text):
    raw = mm.group(0); pos = mm.start()
    s = raw.replace("$", "").replace(",", "").replace("−", "-").lstrip("+")
    try: v = float(s)
    except ValueError: continue
    n += 1
    if v in ALLOW or abs(v) in ALLOW: continue
    srcs = sources_of(v)
    if not srcs: bad.append((raw, sentence_of(pos))); continue
    if not any(x.startswith("primary") for x in srcs):
        sent = sentence_of(pos); low = sent.lower()
        if not any(kw.lower() in low for x in srcs for kw in CONTEXT.get(x, ())): ctx_bad.append((raw, srcs, sent))
    if abs(v) >= 10 or "." in s:   # skip small integers (counts in prose such as "three windows")
        if round(abs(v), 3) not in TABLE_NUMS and round(abs(v) / 100, 3) not in TABLE_NUMS and round(abs(v) * 100, 3) not in TABLE_NUMS: nontable.append((raw, sentence_of(pos)))
print(f"numbers checked: {n}; unmatched: {len(bad)}; matched only by a non-primary source without naming it: {len(ctx_bad)}; prose numbers absent from every generated table: {len(nontable)} (list in audit_prose_not_in_tables.txt)")
(HERE / "audit_prose_not_in_tables.txt").write_text("\n".join(f"{r}\t{s[:220]}" for r, s in nontable))
for b, sent in bad: print("  UNMATCHED", b, "…", sent[:200])
for b, srcs, sent in ctx_bad: print("  NON-PRIMARY ONLY", b, srcs, "…", sent[:220])
ok = not bad and not ctx_bad
print("RESULT: PASS" if ok else "RESULT: FAIL")
sys.exit(0 if ok else 1)
