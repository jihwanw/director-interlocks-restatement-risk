"""Fetch SIC, name, former names, state for each CIK in the D&O universe from data.sec.gov/submissions (public).
Rate-limited to ~8 requests/s as the SEC asks; checkpoints every 500 firms; resumable."""
import pandas as pd, requests, time, json, os, sys
OUT = "data/sec_sic_by_cik.csv"
ciks = pd.read_csv("data/directors_officers_full.csv", usecols=["company_fkey"], dtype=str)["company_fkey"].str.lstrip("0").dropna().unique().tolist()
done = {}
if os.path.exists(OUT):
    d0 = pd.read_csv(OUT, dtype=str); done = {r.cik: None for r in d0.itertuples()}
    rows = d0.to_dict("records")
else: rows = []
hdr = {"User-Agent": "academic research jihwan.woo@gmail.com", "Accept-Encoding": "gzip, deflate"}
todo = [c for c in ciks if c and c not in done]
print(f"ciks {len(ciks)}, done {len(done)}, todo {len(todo)}", flush=True)
t0 = time.time()
from concurrent.futures import ThreadPoolExecutor
import threading
lock = threading.Lock()
def fetch(c):
    s = requests.Session(); url = f"https://data.sec.gov/submissions/CIK{int(c):010d}.json"
    rec = {"cik": c, "name": None, "sic": None, "sic_desc": None, "state_inc": None, "former_names": None, "entity_type": None, "status": None}
    for attempt in range(3):
        try:
            r = s.get(url, headers=hdr, timeout=20)
            if r.status_code == 200:
                j = r.json(); rec.update({"name": j.get("name"), "sic": j.get("sic"), "sic_desc": j.get("sicDescription"), "state_inc": j.get("stateOfIncorporation"),
                                          "former_names": " | ".join(f.get("name", "") for f in j.get("formerNames", [])), "entity_type": j.get("entityType"), "status": "ok"}); break
            elif r.status_code == 404: rec["status"] = "404"; break
            elif r.status_code == 429: time.sleep(15)
            else: rec["status"] = f"http{r.status_code}"; time.sleep(1)
        except Exception as e: rec["status"] = type(e).__name__; time.sleep(2)
    time.sleep(0.45)   # 4 workers x ~2/s = ~8 requests/s
    return rec
with ThreadPoolExecutor(max_workers=4) as ex:
    for i, rec in enumerate(ex.map(fetch, todo)):
        rows.append(rec)
        if (i + 1) % 500 == 0:
            pd.DataFrame(rows).to_csv(OUT, index=False); print(f"[{time.time()-t0:6.0f}s] {i+1}/{len(todo)}", flush=True)
pd.DataFrame(rows).to_csv(OUT, index=False); print("done", len(rows), flush=True)
