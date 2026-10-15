"""fetch_wrds_extracts_2026-10-13.py — the three WRDS pulls added on 2026-10-13 (run with a WRDS account; credentials are read from a local ini file, never stored here).
1. audit.feed17_director_and_officer_chan: the columns merged onto the June extract (middle names, suffixes, estimated-date flag, committee text)
   -> data/directors_officers_wrds_2026-10-13.csv.gz
2. comp.funda historical SIC (sich) for the CIK-linked gvkeys, fiscal years 2004-2024 -> data/comp_funda_sich_2004_2024.csv
3. audit.feed11_sox_404_internal_controls (opinion-level) -> data/sox404_icfr_wrds_2026-10-13.csv.gz
Verification (2026-10-15): the column lists equal the headers of the three files used in the paper; the comp.funda query with the standard
indfmt/datafmt/popsrc/consol filters returned 174,849 rows against 174,842 in the file (15 firm-year keys differ, consistent with Compustat's
ongoing revisions; the unfiltered query returns 325,242 rows and is not what was used).
Usage: python code/fetch_wrds_extracts_2026-10-13.py --ini path/to/api_keys.ini   (ini section [wrds] with username and password)
"""
import argparse, builtins, configparser, getpass, os
import pandas as pd, wrds
ap = argparse.ArgumentParser(); ap.add_argument("--ini", required=True); a = ap.parse_args()
cfg = configparser.ConfigParser(); cfg.read(a.ini); u, pw = cfg["wrds"]["username"], cfg["wrds"]["password"]
os.environ["PGPASSWORD"] = pw; builtins.input = lambda p="": u; getpass.getpass = lambda p="": pw
db = wrds.Connection(wrds_username=u)
df = db.raw_sql("""select do_off_pers_key, do_change_key, company_fkey, first_name, middle_name, last_name, suffix, eff_date, is_effective_date_estimate, eff_date_unspec,
                          action, reasons, is_bdmem_pers, is_ceo, is_cfo, is_chair, title_standard, comm_report, comm_standard, interim, do_off_remains, file_date, form_fkey
                   from audit.feed17_director_and_officer_chan""", date_cols=["eff_date", "file_date"])
df.to_csv("data/directors_officers_wrds_2026-10-13.csv.gz", index=False, compression="gzip")
cw = pd.read_csv("data/cik_gvkey_crosswalk.csv", dtype=str); gv = sorted(set(cw["gvkey"].dropna().astype(str).str.zfill(6)))
fu = db.raw_sql(f"""select gvkey, fyear, sich, cik from comp.funda where indfmt='INDL' and datafmt='STD' and popsrc='D' and consol='C'
                    and fyear between 2004 and 2024 and gvkey in ({",".join("'" + g + "'" for g in gv)})""")
fu.to_csv("data/comp_funda_sich_2004_2024.csv", index=False)
ic = db.raw_sql("""select company_fkey, fy_ic_op, fye_ic_op, ic_is_effective, count_weak, ic_op_type, auditor_agrees, noteff_fin_fraud, notefferrors, file_date
                   from audit.feed11_sox_404_internal_controls""", date_cols=["fye_ic_op", "file_date"])
ic.to_csv("data/sox404_icfr_wrds_2026-10-13.csv.gz", index=False, compression="gzip")
db.close(); print("done")
