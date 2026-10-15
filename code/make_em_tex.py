#!/usr/bin/env python3
"""Build a single, self-contained, EDITABLE .tex for Editorial Manager (Elsevier).

EM requirements at revision stage (confirmed on sibling JBR submission
JOBR-D-26-01739, 2026-07-22):
- manuscript must be an editable source file (.tex), not PDF;
- tables must be editable LaTeX tabulars (already true here: tables_auto/*.tex);
- EM flattens all uploads into ONE directory, so figure paths must have no
  subdirectory prefix;
- the bibliography (.bbl) is inlined so EM does not need to run BibTeX.

This script merges main_blinded.tex + sections/*.tex + tables_auto/*.tex,
flattens figure paths (figures/, figures_rebuild/ and figures_k/ prefixes), and inlines
main_blinded.bbl.

Output: ../01_Manuscript_blinded.tex  (upload as the Manuscript item)
Also copies flat-named figure PDFs to ../EM_upload_figures/.

Requires: main_blinded.bbl next to this script (copy from
../../manuscript/main_blinded.bbl, same 7/22 build).
"""
import re, shutil
from pathlib import Path

HERE = Path(__file__).parent
OUT_TEX = HERE.parent / "01_Manuscript_blinded.tex"
FIG_OUT = HERE.parent / "EM_upload_figures"

main = (HERE / "main_blinded.tex").read_text()

# 1. inline \input{sections/X} and \input{tables_auto/tab_X}
def inline_inputs(text: str) -> str:
    def repl(m):
        rel = m.group(1)
        p = HERE / (rel if rel.endswith(".tex") else rel + ".tex")
        return ("\n% ==== BEGIN " + rel + " ====\n" + p.read_text()
                + "\n% ==== END " + rel + " ====\n")
    return re.sub(r"\\input\{([^}]+)\}", repl, text)

main = inline_inputs(main)   # sections
main = inline_inputs(main)   # tables_auto referenced inside sections

# 2. flatten figure paths (EM puts every upload in one directory)
main, n_fig = re.subn(r"\{(?:figures|figures_rebuild|figures_k)/([\w\-]+\.(?:pdf|png|eps))\}",
                      r"{\1}", main)

# 3. inline the bibliography (.bbl) so EM need not run BibTeX
bbl = (HERE / "main_blinded.bbl").read_text()
main, n_bib = re.subn(
    r"\\bibliographystyle\{elsarticle-harv\}\s*\\bibliography\{references\}",
    lambda _m: "% Bibliography inlined from main_blinded.bbl (elsarticle-harv)\n" + bbl,
    main)

assert n_bib == 1, "bibliography block not found/replaced"
assert "\\input{" not in main, "unresolved \\input remains"

OUT_TEX.write_text(main)
print(f"wrote {OUT_TEX}  (figure paths flattened: {n_fig}, bbl inlined: {n_bib})")

# 4. flat figure copies for the EM Figure slots
FIG_OUT.mkdir(exist_ok=True)
used = set(re.findall(r"\\includegraphics\[[^\]]*\]\{([\w\-]+\.pdf)\}", main))
for d in ("figures", "figures_rebuild", "figures_k"):
    if not (HERE / d).exists(): continue
    for f in sorted((HERE / d).glob("fig*.pdf")):
        if f.name in used:
            shutil.copy(f, FIG_OUT / f.name); print("copied", f.name)
