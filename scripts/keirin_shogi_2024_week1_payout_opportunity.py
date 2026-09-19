#!/usr/bin/env python3
import csv,re,unicodedata,json
from pathlib import Path
from collections import defaultdict

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/"data/2024/s_class_f1_all_parts/2024_q1"
START,END="2024-01-01","2024-01-07"

def rows(p):
    with p.open("r",encoding="utf-8-sig",newline="") as f:return list(csv.DictReader(f))
def ino(v):
    try:return int(float(v))
    except:return 0
def norm(v):return unicodedata.normalize("NFKC",str(v or "")).strip()

races=rows(BASE/"races.csv")
payouts=rows(BASE/"payouts.csv")
targets={r["race_id"]:r for r in races if START<=r.get("race_date","")<=END and ino(r.get("entry_count"))==7 and r.get("meeting_grade")=="F1" and "Ｓ級" in str(r.get("race_type",""))}
out=[]
for p in payouts:
    rid=str(p.get("race_id",""))
    if rid not in targets:continue
    if norm(p.get("ticket_type")) not in {"3連単","三連単"}:continue
    if norm(p.get("status","")).lower() not in {"","paid"}:continue
    py=re.sub(r"[^0-9]","",norm(p.get("payout_yen","")))
    if not py:continue
    pay=int(py)
    r=targets[rid]
    out.append({"race_id":rid,"race_date":r.get("race_date"),"track":r.get("track"),"race_no":ino(r.get("race_no")),"combination":norm(p.get("combination")),"payout_yen":pay})
out.sort(key=lambda x:(x["race_date"],x["track"],x["race_no"]))
kings=[x for x in out if x["payout_yen"]>=10000]
slaves=[x for x in out if x["payout_yen"]<10000]
print(json.dumps({
 "target_races":len(targets),
 "trifecta_payout_rows":len(out),
 "king_opportunities":len(kings),
 "slave_outcomes":len(slaves),
 "king_rate_pct":100*len(kings)/len(out) if out else None,
 "max_actual_trifecta_yen":max((x["payout_yen"] for x in out),default=0),
 "median_actual_trifecta_yen":sorted([x["payout_yen"] for x in out])[len(out)//2] if out else 0,
 "kings":sorted(kings,key=lambda x:-x["payout_yen"])
},ensure_ascii=False,indent=2))
