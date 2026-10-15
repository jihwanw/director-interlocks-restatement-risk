"""Fetch business-address state/city (headquarters) and SIC for each CIK from data.sec.gov/submissions (public); 4 workers, ~8 req/s; resumable."""
import pandas as pd, requests, time, os, threading
from concurrent.futures import ThreadPoolExecutor
OUT = "data/sec_address_by_cik.csv"
ciks = pd.read_csv("data/directors_officers_full.csv", usecols=["company_fkey"], dtype=str)["company_fkey"].str.lstrip("0").dropna().unique().tolist()
rows = pd.read_csv(OUT, dtype=str).to_dict("records") if os.path.exists(OUT) else []
done = {r["cik"] for r in rows}; todo = [c for c in ciks if c and c not in done]
print(f"ciks {len(ciks)}, done {len(done)}, todo {len(todo)}", flush=True)
hdr = {"User-Agent": "academic research jihwan.woo@gmail.com", "Accept-Encoding": "gzip, deflate"}
t0 = time.time()
def fetch(c):
    s = requests.Session(); url = f"https://data.sec.gov/submissions/CIK{int(c):010d}.json"
    rec = {"cik": c, "biz_state": None, "biz_city": None, "mail_state": None, "fye": None, "status": None}
    for attempt in range(3):
        try:
            r = s.get(url, headers=hdr, timeout=20)
            if r.status_code == 200:
                j = r.json(); b = (j.get("addresses") or {}).get("business") or {}; m = (j.get("addresses") or {}).get("mailing") or {}
                rec.update({"biz_state": b.get("stateOrCountry"), "biz_city": b.get("city"), "mail_state": m.get("stateOrCountry"), "fye": j.get("fiscalYearEnd"), "status": "ok"}); break
            elif r.status_code == 404: rec["status"] = "404"; break
            elif r.status_code == 429: time.sleep(15)
            else: rec["status"] = f"http{r.status_code}"; time.sleep(1)
        except Exception as e: rec["status"] = type(e).__name__; time.sleep(2)
    time.sleep(0.45); return rec
with ThreadPoolExecutor(max_workers=4) as ex:
    for i, rec in enumerate(ex.map(fetch, todo)):
        rows.append(rec)
        if (i + 1) % 500 == 0: pd.DataFrame(rows).to_csv(OUT, index=False); print(f"[{time.time()-t0:6.0f}s] {i+1}/{len(todo)}", flush=True)
pd.DataFrame(rows).to_csv(OUT, index=False); print("done", len(rows), flush=True)
