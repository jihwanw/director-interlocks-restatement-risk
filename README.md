# Director Interlocks and Restatement Risk: Appointment, Disclosure, and Departure

Replication code for the manuscript of that title (Kim and Woo, 2026; submitted to *The British Accounting Review*).

This repository contains the code that builds the dated, directed board-interlock network from Audit Analytics board-member events,
estimates every number in the manuscript, generates every table and figure from the results JSON files, and audits the manuscript's
numbers against those files. Aggregate results (`results_rebuild/*.json`) are included; the licensed microdata are not.

## Data (licensed; not redistributed)
- Audit Analytics via WRDS: `audit.feed17_director_and_officer_chan` (board-member events; extract 2026-06-09, middle names and committee
  text 2026-10-13), `audit.feed39_financial_restatements` (2026-06-09), `audit.feed11_sox_404_internal_controls` (2026-10-13).
- Compustat via WRDS: `comp.funda` (fundamentals; historical SIC `sich`), `comp.company`.
- SEC EDGAR submissions API: SIC codes, current and former names, business address (fetched 2026-10-09/10 by `code/fetch_sec_sic.py`, `code/fetch_sec_address.py`).
Place the extracts in `data/` with the file names expected by `code/rebuild_k_timing.py` (see its header docstring).

## Pipeline (run from the repository root; Python 3.13, pandas, numpy, scipy, statsmodels, linearmodels, lifelines, networkx)
```
python code/rebuild_k_timing.py --sample artifacts --suffix _art  --aux-file data/directors_officers_wrds_2026-10-13.csv.gz --person-key name_mid   # primary sample
python code/rebuild_k_timing.py --suffix _full                                                                                                   # all directors, no exclusions
python code/rebuild_k_timing.py --max-firms 6 --suffix _le6                                                                                       # alternative: six-firm cap
python code/rebuild_k_timing.py --sample artifacts --keep-common-surnames --suffix _artns --aux-file ... --person-key name_mid                   # variant
python code/rebuild_k_timing.py --sample artifacts --suffix _art_name  --aux-file ... --person-key name                                           # variant
python code/rebuild_k_timing.py --sample artifacts --suffix _art_noest --aux-file ... --person-key name_mid --drop-estimated-dates               # variant
python code/rebuild_l_identification.py --suffix _art --max-firms 0      # identification layer (latent exposure, falsification, event study, bounds)
python code/rebuild_m_icfr.py --suffix _art                              # SOX 404 outcome; appointment-level analysis
python code/make_tables_k.py; python code/make_tables_l.py; python code/make_tables_m.py; python code/make_definitions.py   # tables and figures
python code/audit_numbers_k.py                                           # every number in the prose traced to a JSON value
python code/reproduce_headlines.py                                       # independent re-estimation of the headline numbers
```
`PRIMARY_SUFFIX` (default `_art`) selects the primary sample for the generators and audits.

- `code/make_figures_design_network.py` draws the design schematic (Figure 3) and the network snapshot and ego networks (Figure 2) from the primary-sample edge lifecycles, tenure spells and restatement records, and writes the counts it prints to `results_rebuild/N_figures_design_network.json`.
- `code/split_table_notes.py` runs after the table generators: it replaces each long table note with a short one and writes the explanatory text to `tables_auto/lead_<table>.tex`, which the manuscript inputs before the table (appendix tables); table bodies are left unchanged.

## Releases and archive
- `v1.0` (2026-03-05): code of a superseded version of the paper, submitted to another journal under a different title; archived at Zenodo 10.5281/zenodo.18872817.
- `v2.0-results` (2026-10-15): the code state that produced every result in the present manuscript.
- `v2.0` (2026-10-23): the archived submission release — the code of `v2.0-results` plus the final manuscript generators (`make_tables_*.py`, `make_definitions.py`, `split_table_notes.py`, `make_figures_design_network.py`), the numeric audit and documentation; Zenodo version DOI: 10.5281/zenodo.23584371 (https://doi.org/10.5281/zenodo.23584371). Concept DOI (always resolves to the latest version): 10.5281/zenodo.18872816.

## Licence
MIT (see `LICENSE`). The data are licensed by their providers and are not included.
