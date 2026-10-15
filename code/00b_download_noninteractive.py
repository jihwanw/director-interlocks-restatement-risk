#!/usr/bin/env python3
"""
00b_download_noninteractive.py
Non-interactive WRDS download for the JBR board-interlock rebuild.
Same queries as 00_download_wrds_data.py but connects without prompting,
reading credentials from api_keys.ini (see ~/.kiro/steering/wrds_access.md).
Writes a single fresh source to data/ so every downstream number comes from
one run (addresses Reviewer 1's cross-exhibit inconsistencies).
"""
import os, configparser, builtins, getpass, time
import pandas as pd

INI = "/Users/jihwanw/PhD/code/api_info/api_keys.ini"
OUTPUT = os.path.join(os.path.dirname(__file__), "..", "data")
OUTPUT = os.path.abspath(OUTPUT)
os.makedirs(OUTPUT, exist_ok=True)

cfg = configparser.ConfigParser(); cfg.read(INI)
u, pw = cfg["wrds"]["username"], cfg["wrds"]["password"]
os.environ["PGPASSWORD"] = pw
builtins.input = lambda p="": u
getpass.getpass = lambda p="": pw

import wrds
t0 = time.time()
print("Connecting to WRDS (non-interactive)...", flush=True)
db = wrds.Connection(wrds_username=u)
print("Connected.", flush=True)

def pull(name, sql, fname):
    s = time.time()
    print(f"[{name}] querying...", flush=True)
    df = db.raw_sql(sql)
    path = os.path.join(OUTPUT, fname)
    df.to_csv(path, index=False)
    print(f"[{name}] {len(df):,} rows -> {fname} ({time.time()-s:.0f}s)", flush=True)
    return len(df)

counts = {}
counts["directors"] = pull("1/4 directors", """
    SELECT company_fkey, first_name, last_name, eff_date, action,
           is_ceo, is_cfo, is_bdmem_pers, is_chair, is_c_level,
           is_fin_pers, is_op_pers, is_legal, is_exec_vp,
           is_president, is_coo, is_secretary, is_cont
    FROM audit.feed17_director_and_officer_chan
    ORDER BY company_fkey, eff_date
""", "directors_officers_full.csv")

counts["restatements"] = pull("2/4 restatements", """
    SELECT company_fkey, file_date, res_begin_date, res_end_date,
           res_accounting, res_adverse, res_fraud, res_sec_investigation,
           matchqu_balsh_assets, matchqu_incmst_netinc_ttm,
           matchqu_incmst_rev_ttm
    FROM audit.feed39_financial_restatements
    WHERE file_date IS NOT NULL
    ORDER BY company_fkey, file_date
""", "restatements_with_dates.csv")

counts["auditors"] = pull("3/4 auditors", """
    SELECT company_fkey, auditor_key, auditor_name, event_type
    FROM audit.feed07_current_auditor
    WHERE auditor_key IS NOT NULL
    ORDER BY company_fkey
""", "auditor_company.csv")

counts["compustat"] = pull("4/4 compustat", """
    SELECT gvkey, fyear AS year, at AS assets, lt AS liabilities,
           ceq AS equity, revt AS revenue, ni AS net_income,
           oancf, csho * prcc_f AS mktcap, csho AS shares,
           prcc_f AS price
    FROM comp.funda
    WHERE fyear BETWEEN 2004 AND 2023
      AND indfmt = 'INDL' AND datafmt = 'STD'
      AND popsrc = 'D' AND consol = 'C'
      AND at > 0
    ORDER BY gvkey, fyear
""", "compustat_financials.csv")

db.close()
import json
with open(os.path.join(OUTPUT, "download_manifest.json"), "w") as f:
    json.dump({"counts": counts, "elapsed_sec": round(time.time()-t0,1),
               "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
               "source": "WRDS Audit Analytics + Compustat"}, f, indent=2)
print(f"\nDONE in {time.time()-t0:.0f}s. counts={counts}", flush=True)
