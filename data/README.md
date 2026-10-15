# Data directory (intentionally empty — licensed data are not redistributed)

All inputs come from WRDS (institutional subscription required) and from the SEC EDGAR submissions API.

| File expected by the code | Source | Script |
|---|---|---|
| `directors_officers_full.csv` | Audit Analytics `audit.feed17_director_and_officer_chan` (board-member and officer events; June 2026 extract) | `code/00_download_wrds_data.py` |
| `restatements_with_dates.csv` | Audit Analytics `audit.feed39_financial_restatements` | `code/00_download_wrds_data.py` |
| `auditor_company.csv` | Audit Analytics `audit.feed01_audit_opinions` (auditor identity) | `code/00_download_wrds_data.py` |
| `compustat_financials.csv`, `cik_gvkey_crosswalk.csv` | Compustat `comp.funda`, `comp.company` | `code/00_download_wrds_data.py` |
| `directors_officers_wrds_2026-10-13.csv.gz` | `audit.feed17_director_and_officer_chan` (middle names, suffixes, estimated-date flag, committee text) | `code/fetch_wrds_extracts_2026-10-13.py` |
| `comp_funda_sich_2004_2024.csv` | `comp.funda` historical SIC | `code/fetch_wrds_extracts_2026-10-13.py` |
| `sox404_icfr_wrds_2026-10-13.csv.gz` | `audit.feed11_sox_404_internal_controls` | `code/fetch_wrds_extracts_2026-10-13.py` |
| `sec_sic_by_cik.csv`, `sec_address_by_cik.csv` | SEC EDGAR submissions API (SIC, names, business address) | `code/fetch_sec_sic.py`, `code/fetch_sec_address.py` |

Intermediate files (`edge_lifecycles_k*.csv`, `tenure_spells_k*.csv`, `pair_base_k*.csv`, `integrated_panel_k*.csv`, `panel_network_k*.csv`) are written by `code/rebuild_k_timing.py`.
