#!/usr/bin/env python3
from __future__ import annotations

"""
True next-week holdout for v38 KING SEAT/GATE.
Train/freeze source: 2024-01-01..2024-01-07
Validation only: 2024-01-08..2024-01-14

Labels requested by project language:
- KING: bought, exact KING seat hit, trifecta payout >= 10,000 yen
- IMPUDENT: bought, exact KING seat hit, payout < 10,000 yen
- BEHEADED: bought, exact KING seat miss
- SKIP: KING gate says do not buy

No retraining or threshold adjustment is allowed here.
"""

import csv, importlib.util, itertools, json, re, sys, unicodedata
from collections import defaultdict, Counter
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/"data/2024/s_class_f1_all_parts/2024_q1"
START,END="2024-01-08","2024-01-14"
KING=10000
SEAT_COST=100
OUT=ROOT/"results/keirin_shogi/v38_king_seat_2024w2_holdout"
OUT.mkdir(parents=True,exist_ok=True)

def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name]=mod
    spec.loader.exec_module(mod)
    return mod

runtime=load_module("w2_runtime",ROOT/"scripts/keirin_shogi_v37_auto_place_runtime.py")
king=load_module("w2_king",ROOT/"scripts/keirin_shogi_v38_king_runtime.py")

def read(path):
    with path.open("r",encoding="utf-8-sig",newline="") as f:
        return list(csv.DictReader(f))
def ino(v):
    try:return int(float(v))
    except:return 0
def norm(v): return unicodedata.normalize("NFKC",str(v or "")).strip()
def parse_triple(v):
    xs=[int(x) for x in re.findall(r"[1-9]",norm(v))]
    if len(xs)>=3 and len(set(xs[:3]))==3:return tuple(xs[:3])
    return None

races=read(BASE/"races.csv")
entries=read(BASE/"entries.csv")
results=read(BASE/"results.csv")
payouts=read(BASE/"payouts.csv")

targets={r["race_id"]:r for r in races if START<=r.get("race_date","")<=END and ino(r.get("entry_count"))==7 and r.get("meeting_grade")=="F1" and "Ｓ級" in r.get("race_type","")}
eb=defaultdict(list); rb=defaultdict(list)
for e in entries:
    if e.get("race_id") in targets: eb[e["race_id"]].append(e)
for x in results:
    if x.get("race_id") in targets: rb[x["race_id"]].append(x)

paid=defaultdict(dict)
for p in payouts:
    rid=str(p.get("race_id",""))
    if rid not in targets: continue
    if norm(p.get("ticket_type")) not in {"3連単","三連単"}: continue
    if norm(p.get("status")).lower() not in {"","paid"}: continue
    comb=parse_triple(p.get("combination",""))
    yen=re.sub(r"[^0-9]","",norm(p.get("payout_yen")))
    if comb and yen: paid[rid][comb]=int(yen)

def actual_order(rows):
    top={}
    for x in rows:
        pos=ino(x.get("finish_position"))
        if pos in (1,2,3): top[pos]=ino(x.get("car_no"))
    return (top.get(1),top.get(2),top.get(3)) if len(top)==3 else None

engine=runtime.load_base_engine()
nine=runtime.load_ninecar_engine()
overlay=runtime.load_sevencar_overlay_engine()
king_engine=runtime.load_king_engine()
v21=engine.build_v21_state()
pair=engine.build_pair_model()
third,features,freeze=engine.build_third_model()

logs=[]; failures=[]
for rid,r in sorted(targets.items(),key=lambda kv:(kv[1].get("race_date",""),kv[1].get("track",""),ino(kv[1].get("race_no")))):
    ers=sorted(eb[rid],key=lambda x:ino(x.get("car_no")))
    live={
      "race_id":rid,"race_date":r.get("race_date",""),"track":r.get("track",""),"race_no":r.get("race_no",""),
      "race_type":r.get("race_type",""),"meeting_grade":r.get("meeting_grade",""),
      "entries":[runtime.normalize_entry(e) for e in ers],
    }
    placed=runtime.runtime_place_one(engine,nine,overlay,king_engine,live,v21,pair,third,features,freeze)
    if not placed.get("board_generated"):
        failures.append({"race_id":rid,"date":r.get("race_date"),"track":r.get("track"),"race_no":ino(r.get("race_no")),"error":placed.get("error"),"stage":placed.get("failure_stage")})
        continue
    actual=actual_order(rb[rid])
    payout=paid[rid].get(actual,0) if actual else 0
    seats=[(int(x["first"]),int(x["second"]),int(x["third"])) for x in placed.get("king_top_tickets",[])]
    participate=bool(placed.get("participate"))
    hit=bool(participate and actual in seats)
    if not participate:
        verdict="SKIP"
    elif hit and payout>=KING:
        verdict="KING"
    elif hit:
        verdict="IMPUDENT"
    else:
        verdict="BEHEADED"
    stake=len(seats)*SEAT_COST if participate else 0
    ret=payout if hit else 0
    board_hit=bool(actual and actual[0] in placed["first_candidates"] and actual[1] in placed["second_candidates"] and actual[2] in placed["third_candidates"])
    logs.append({
      "race_id":rid,"date":r.get("race_date"),"track":r.get("track"),"race_no":ino(r.get("race_no")),
      "actual":list(actual) if actual else None,"payout_yen":payout,
      "participate":participate,"king_gate_score":placed.get("king_gate_score"),"king_gate_threshold":placed.get("king_gate_threshold"),
      "seat_count":len(seats),"exact_seat_hit":hit,"board_capture":board_hit,"verdict":verdict,
      "stake_yen":stake,"return_yen":ret,"profit_yen":ret-stake,
      "first_candidates":placed["first_candidates"],"second_candidates":placed["second_candidates"],"third_candidates":placed["third_candidates"],
      "king_seats":[list(x) for x in seats],
    })

bought=[x for x in logs if x["participate"]]
actual_kings=[x for x in logs if x["payout_yen"]>=KING]
bought_actual_kings=[x for x in bought if x["payout_yen"]>=KING]
king_hits=[x for x in logs if x["verdict"]=="KING"]
impudent=[x for x in logs if x["verdict"]=="IMPUDENT"]
beheaded=[x for x in logs if x["verdict"]=="BEHEADED"]
skips=[x for x in logs if x["verdict"]=="SKIP"]
stake=sum(x["stake_yen"] for x in logs); ret=sum(x["return_yen"] for x in logs)
summary={
  "algorithm":"v38_king_seat_gate_2024w1_frozen",
  "training_period":["2024-01-01","2024-01-07"],
  "holdout_period":[START,END],
  "retrained_on_holdout":False,
  "target_races":len(targets),"evaluated_races":len(logs),"failures":len(failures),
  "buy_races":len(bought),"skip_races":len(skips),"buy_rate":len(bought)/len(logs) if logs else 0,
  "verdict_counts":dict(Counter(x["verdict"] for x in logs)),
  "KING":len(king_hits),"IMPUDENT":len(impudent),"BEHEADED":len(beheaded),
  "hit_races":len(king_hits)+len(impudent),
  "hit_rate_on_bought":(len(king_hits)+len(impudent))/len(bought) if bought else 0,
  "actual_king_races":len(actual_kings),
  "actual_king_bought":len(bought_actual_kings),
  "actual_king_gate_recall":len(bought_actual_kings)/len(actual_kings) if actual_kings else 0,
  "exact_king_hits":len(king_hits),
  "exact_king_recall_all":len(king_hits)/len(actual_kings) if actual_kings else 0,
  "board_capture_actual_kings":sum(x["board_capture"] for x in actual_kings),
  "board_capture_actual_king_rate":sum(x["board_capture"] for x in actual_kings)/len(actual_kings) if actual_kings else 0,
  "stake_yen":stake,"payout_yen":ret,"profit_yen":ret-stake,"roi":ret/stake if stake else 0,
  "max_hit_payout_yen":max([x["payout_yen"] for x in logs if x["exact_seat_hit"]],default=0),
  "max_actual_payout_yen":max([x["payout_yen"] for x in logs],default=0),
  "king_hit_details":[{k:x[k] for k in ("date","track","race_no","actual","payout_yen","profit_yen")} for x in king_hits],
  "impudent_details":[{k:x[k] for k in ("date","track","race_no","actual","payout_yen","profit_yen")} for x in impudent],
  "beheaded_top_payouts":[{k:x[k] for k in ("date","track","race_no","actual","payout_yen")} for x in sorted(beheaded,key=lambda z:z["payout_yen"],reverse=True)[:15]],
  "skipped_kings":[{k:x[k] for k in ("date","track","race_no","actual","payout_yen","king_gate_score")} for x in skips if x["payout_yen"]>=KING],
  "failures_detail":failures,
}
(OUT/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
(OUT/"race_log.json").write_text(json.dumps(logs,ensure_ascii=False,indent=2),encoding="utf-8")
print(json.dumps(summary,ensure_ascii=False,indent=2))
