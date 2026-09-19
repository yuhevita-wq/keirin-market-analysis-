#!/usr/bin/env python3
import csv, json, re, unicodedata
from pathlib import Path

BASE=Path("data/2024/s_class_f1_all_parts/2024_q1")
START,END="2024-01-01","2024-01-07"
KING=10000

def ino(v):
    try:return int(float(v))
    except:return 0

def read(p):
    with p.open("r",encoding="utf-8-sig",newline="") as f:
        return list(csv.DictReader(f))

def norm(v):
    return unicodedata.normalize("NFKC",str(v or "")).strip()

races=read(BASE/"races.csv")
payouts=read(BASE/"payouts.csv")
targets={r["race_id"]:r for r in races
         if START<=r.get("race_date","")<=END
         and ino(r.get("entry_count"))==7
         and r.get("meeting_grade")=="F1"
         and "Ｓ級" in r.get("race_type","")}
rows=[]
for p in payouts:
    rid=str(p.get("race_id",""))
    if rid not in targets: continue
    if norm(p.get("ticket_type")) not in {"3連単","三連単"}: continue
    if norm(p.get("status")).lower() not in {"","paid"}: continue
    y=re.sub(r"[^0-9]","",norm(p.get("payout_yen")))
    if not y: continue
    payout=int(y)
    r=targets[rid]
    rows.append({
      "race_id":rid,"race_date":r.get("race_date"),"track":r.get("track"),
      "race_no":ino(r.get("race_no")),"combination":norm(p.get("combination")),
      "payout_yen":payout,"label":"KING" if payout>=KING else "SLAVE"
    })
rows.sort(key=lambda x:(x["race_date"],x["track"],x["race_no"]))
print(json.dumps({
  "target_races":len(targets),
  "trifecta_paid_rows":len(rows),
  "actual_kings":sum(r["payout_yen"]>=KING for r in rows),
  "actual_slaves":sum(r["payout_yen"]<KING for r in rows),
  "max_actual_payout_yen":max((r["payout_yen"] for r in rows),default=0),
  "kings":[r for r in rows if r["payout_yen"]>=KING]
},ensure_ascii=False,indent=2))
