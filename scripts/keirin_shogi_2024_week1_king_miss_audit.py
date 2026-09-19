#!/usr/bin/env python3
from __future__ import annotations
import csv, importlib.util, json, re, sys, unicodedata
from collections import Counter, defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/"data/2024/s_class_f1_all_parts/2024_q1"
START,END="2024-01-01","2024-01-07"
KING=10000

def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name]=mod
    spec.loader.exec_module(mod)
    return mod

runtime=load_module("king_miss_runtime",ROOT/"scripts/keirin_shogi_v37_auto_place_runtime.py")

def read(p):
    with p.open("r",encoding="utf-8-sig",newline="") as f:
        return list(csv.DictReader(f))
def ino(v):
    try:return int(float(v))
    except:return 0
def norm(v):
    return unicodedata.normalize("NFKC",str(v or "")).strip()
def parse_triple(v):
    xs=[int(x) for x in re.findall(r"[1-9]",norm(v))]
    if len(xs)>=3 and len(set(xs[:3]))==3:return tuple(xs[:3])
    return None
def rank_of(rows,no,key):
    for i,r in enumerate(rows,1):
        if int(r["no"])==no:return i,float(r[key])
    return None,None

races=read(BASE/"races.csv"); entries=read(BASE/"entries.csv"); payouts=read(BASE/"payouts.csv")
targets={r["race_id"]:r for r in races if START<=r.get("race_date","")<=END and ino(r.get("entry_count"))==7 and r.get("meeting_grade")=="F1" and "Ｓ級" in r.get("race_type","")}
eb=defaultdict(list)
for e in entries:
    if e.get("race_id") in targets: eb[e["race_id"]].append(e)

kings={}
for p in payouts:
    rid=str(p.get("race_id",""))
    if rid not in targets: continue
    if norm(p.get("ticket_type")) not in {"3連単","三連単"}: continue
    if norm(p.get("status")).lower() not in {"","paid"}: continue
    py=re.sub(r"[^0-9]","",norm(p.get("payout_yen")))
    comb=parse_triple(p.get("combination",""))
    if py and comb and int(py)>=KING:
        kings[rid]={"combo":comb,"payout_yen":int(py)}

engine=runtime.load_base_engine()
nine=runtime.load_ninecar_engine()
overlay=runtime.load_sevencar_overlay_engine()
v21_state=engine.build_v21_state()
pair_model=engine.build_pair_model()
third_model,feature_names,freeze=engine.build_third_model()

rows=[]
for rid,k in sorted(kings.items(), key=lambda kv:(targets[kv[0]].get("race_date",""),targets[kv[0]].get("track",""),ino(targets[kv[0]].get("race_no")))):
    race=targets[rid]
    erows=sorted(eb[rid],key=lambda x:ino(x.get("car_no")))
    live={"race_id":rid,"race_date":race.get("race_date",""),"track":race.get("track",""),"race_no":race.get("race_no",""),"race_type":race.get("race_type",""),"meeting_grade":race.get("meeting_grade",""),"entries":[runtime.normalize_entry(e) for e in erows]}
    p=runtime.runtime_place_one(engine,nine,overlay,live,v21_state,pair_model,third_model,feature_names,freeze)
    if not p.get("board_generated"): continue
    a,b,c=k["combo"]
    h1=a in p["first_candidates"]; h2=b in p["second_candidates"]; h3=c in p["third_candidates"]
    r1,s1=rank_of(p["first_ranking"],a,"probability")
    r2,m2=rank_of(p["second_membership"],b,"mass")
    r3,s3=rank_of(p["third_ranking"],c,"probability")
    rows.append({
      "race_id":rid,"date":race.get("race_date"),"track":race.get("track"),"race_no":ino(race.get("race_no")),
      "combo":list(k["combo"]),"payout_yen":k["payout_yen"],"participate":bool(p.get("participate")),
      "board":{"first":p["first_candidates"],"second":p["second_candidates"],"third":p["third_candidates"]},
      "hit_first":h1,"hit_second":h2,"hit_third":h3,
      "missing_rows":[x for x,h in [("1着",h1),("2着",h2),("3着",h3)] if not h],
      "actual_first_rank":r1,"actual_first_prob":s1,
      "actual_second_rank":r2,"actual_second_mass":m2,
      "actual_third_rank":r3,"actual_third_prob":s3,
      "first_candidate_count":len(p["first_candidates"]),
      "second_candidate_count":len(p["second_candidates"]),
      "third_candidate_count":len(p["third_candidates"]),
      "overlay_action":p.get("sevencar_state_overlay_action"),
      "overlay_added":bool(p.get("sevencar_state_added_third",False)),
    })

miss_pattern=Counter("+".join(r["missing_rows"]) if r["missing_rows"] else "CAPTURE" for r in rows)
row_miss={
 "1着":sum(not r["hit_first"] for r in rows),
 "2着":sum(not r["hit_second"] for r in rows),
 "3着":sum(not r["hit_third"] for r in rows),
}
just_out={
 "1着_rank<=3_but_cut":sum((not r["hit_first"]) and (r["actual_first_rank"] or 99)<=3 for r in rows),
 "2着_rank<=3_but_cut":sum((not r["hit_second"]) and (r["actual_second_rank"] or 99)<=3 for r in rows),
 "3着_rank<=4_but_cut":sum((not r["hit_third"]) and (r["actual_third_rank"] or 99)<=4 for r in rows),
}
summary={
 "actual_kings":len(rows),
 "captured_kings":sum(not r["missing_rows"] for r in rows),
 "row_miss_counts":row_miss,
 "miss_patterns":dict(miss_pattern),
 "near_cut_misses":just_out,
 "avg_candidate_counts":{
   "first":sum(r["first_candidate_count"] for r in rows)/len(rows),
   "second":sum(r["second_candidate_count"] for r in rows)/len(rows),
   "third":sum(r["third_candidate_count"] for r in rows)/len(rows),
 },
 "participate_kings":sum(r["participate"] for r in rows),
 "overlay_added_on_kings":sum(r["overlay_added"] for r in rows),
 "rows":rows
}
print(json.dumps(summary,ensure_ascii=False,indent=2))
