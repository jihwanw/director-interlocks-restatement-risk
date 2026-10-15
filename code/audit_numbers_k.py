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
J = [json.load(open(HERE.parent / "results_rebuild" / f)) for f in (f"K_rebuild{PRIMARY}.json", f"L_identification{PRIMARY}.json", f"M_icfr{PRIMARY}.json", "K_rebuild_full.json", "K_rebuild_le6.json", f"K_rebuild{PRIMARY}ns.json", f"K_rebuild{PRIMARY}_name.json", f"K_rebuild{PRIMARY}_noest.json", "wrds_inventory/inventory_2026-10-13.json") if (HERE.parent / "results_rebuild" / f).exists()]
LOG = (HERE / "K_le6.log").read_text() if (HERE / "K_le6.log").exists() else ""

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
for j in J: walk(j)
# numbers appearing in the run log (event counts etc.)
for m in re.finditer(r"\d[\d,]*\.?\d*", LOG): 
    try: vals.add(float(m.group(0).replace(",", "")))
    except ValueError: pass

def renderings(v):
    out = set()
    for d in (0, 1, 2, 3):
        out.add(round(v, d)); out.add(round(100 * v, d)); out.add(round(v / 100, d))
        out.add(rhu(v, d)); out.add(rhu(100 * v, d)); out.add(rhu(abs(v), d)); out.add(rhu(100 * abs(v), d))
    out.add(round(100 * (1 - v), 1)); out.add(round(100 * (1 - v), 0))
    return out
RENDER = set()
for v in vals:
    RENDER |= renderings(v); RENDER.add(abs(v))
    for d in (1, 2): RENDER.add(round(abs(v), d)); RENDER.add(round(100 * abs(v), d))
# derived: differences of pre/post gaps etc. are in JSON already (did). CI bounds: diff ± 1.96 SE
for j in J:
    def walk2(o):
        if isinstance(o, dict):
            if "diff_pp" in o and "se_pp" in o:
                for s in (1.96,): 
                    RENDER.add(round(o["diff_pp"] - s * o["se_pp"], 1)); RENDER.add(round(o["diff_pp"] + s * o["se_pp"], 1))
                    RENDER.add(round(abs(o["diff_pp"] - s * o["se_pp"]), 1)); RENDER.add(round(abs(o["diff_pp"] + s * o["se_pp"]), 1))
            if "diff_in_gaps_pp" in o and "se_pp" in o:
                RENDER.add(round(o["diff_in_gaps_pp"] - 1.96 * o["se_pp"], 1)); RENDER.add(round(o["diff_in_gaps_pp"] + 1.96 * o["se_pp"], 1))
                RENDER.add(round(abs(o["diff_in_gaps_pp"] - 1.96 * o["se_pp"]), 1)); RENDER.add(round(abs(o["diff_in_gaps_pp"] + 1.96 * o["se_pp"]), 1))
                if "pre_gap_pp" in o: RENDER.add(round(o["pre_gap_pp"] + o["diff_in_gaps_pp"], 1)); RENDER.add(round(o["pre_gap_pp"] + o["diff_in_gaps_pp"], 2))
            if "did_pp" in o and "se_pp" in o:
                RENDER.add(round(o["did_pp"] - 1.96 * o["se_pp"], 1)); RENDER.add(round(o["did_pp"] + 1.96 * o["se_pp"], 1))
            if "change_pp" in o and "se_change_pp" in o:
                RENDER.add(round(o["change_pp"] - 1.96 * o["se_change_pp"], 1)); RENDER.add(round(o["change_pp"] + 1.96 * o["se_change_pp"], 1))
            if "coef_pp" in o and "se_pp" in o:
                RENDER.add(round(o["coef_pp"] - 1.96 * o["se_pp"], 2)); RENDER.add(round(o["coef_pp"] + 1.96 * o["se_pp"], 2))
            if "coef" in o and "se" in o:
                RENDER.add(round(100 * 1.96 * o["se"], 1)); RENDER.add(round(abs(100 * o["coef"]), 1)); RENDER.add(round(abs(100 * o["coef"]), 2))
            for v in o.values(): walk2(v)
        elif isinstance(o, list):
            for v in o: walk2(v)
    walk2(j)

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
nums = re.findall(r"(?<![A-Za-z_\\])[-+−]?\$?\d[\d,]*(?:\.\d+)?", text)
bad = []; n = 0
for raw in nums:
    s = raw.replace("$", "").replace(",", "").replace("−", "-").lstrip("+")
    try: v = float(s)
    except ValueError: continue
    n += 1
    if v in ALLOW or abs(v) in ALLOW: continue
    if round(abs(v), 3) in {round(x, 3) for x in RENDER if abs(x - abs(v)) < 0.0015}: continue
    bad.append(raw)
print(f"numbers checked: {n}; unmatched: {len(bad)}"); print("RESULT: PASS" if not bad else "RESULT: FAIL")
for b in bad: 
    i = text.find(b); print("  ", b, "…", text[max(0, i - 70): i + 40].replace("\n", " "))
sys.exit(1 if bad else 0)
