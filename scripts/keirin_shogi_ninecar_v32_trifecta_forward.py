#!/usr/bin/env python3
from __future__ import annotations
import csv, io, json, re, zipfile, importlib.util, sys
from collections import defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/"scripts/keirin_shogi_ninecar_wide_study.py"
OUT=ROOT/"results/keirin_shogi/ninecar_v32_trifecta_forward/summary.json"
STAKE=100

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name]=mod
    spec.loader.exec_module(mod)
    return mod

w=load("wide_base_for_trifecta",BASE)
v32=w.v32
v2=w.v2

def parse_trifecta(value):
    xs=[int(x) for x in re.findall(r"[1-9]",str(value))]
    if len(xs)!=3 or len(set(xs))!=3:
        return None
    return tuple(xs)

def load_trifecta_payouts():
    by_race={}
    stats=[]
    for path in v2.archive_paths():
        rows=paid=0
        try:
            with zipfile.ZipFile(path) as z:
                try: raw=z.read("payouts.csv")
                except KeyError:
                    stats.append({"archive":str(path.relative_to(ROOT)),"rows":0,"paid_trifecta":0});continue
                text=raw.decode("utf-8-sig",errors="replace")
                for row in csv.DictReader(io.StringIO(text)):
                    rows+=1
                    if row.get("ticket_type")!="3連単" or row.get("status")!="paid":
                        continue
                    combo=parse_trifecta(row.get("combination",""))
                    payout=str(row.get("payout_yen","")).replace(",","").strip()
                    rid=str(row.get("race_id",""))
                    if combo and rid and payout.isdigit():
                        by_race[rid]=(combo,int(payout));paid+=1
        except zipfile.BadZipFile:
            pass
        stats.append({"archive":str(path.relative_to(ROOT)),"rows":rows,"paid_trifecta":paid})
    return by_race,stats

def tickets(board):
    out=set()
    for a in board[0]:
        for b in board[1]:
            for c in board[2]:
                if len({a,b,c})==3:
                    out.add((int(a),int(b),int(c)))
    return out

def summarize(rows):
    if not rows:return {}
    stake=sum(r["ticket_count"]*STAKE for r in rows)
    payout=sum(r["payout_yen"] for r in rows)
    hits=sum(r["hit"] for r in rows)
    hit_payouts=sorted([r["payout_yen"] for r in rows if r["hit"]],reverse=True)
    return {
      "races":len(rows),
      "tickets":sum(r["ticket_count"] for r in rows),
      "avg_tickets":sum(r["ticket_count"] for r in rows)/len(rows),
      "stake_yen":stake,
      "payout_yen":payout,
      "profit_yen":payout-stake,
      "roi_pct":100*payout/stake if stake else None,
      "hit_races":hits,
      "hit_rate_pct":100*hits/len(rows),
      "max_hit_payout_yen":max(hit_payouts,default=0),
      "top1_payout_share":(hit_payouts[0]/payout if payout and hit_payouts else None),
      "top3_payout_share":(sum(hit_payouts[:3])/payout if payout and hit_payouts else None),
    }

def main():
    races=v2.load_races()
    payouts,stats=load_trifecta_payouts()
    all_years={}
    all_records=[]
    for year in (2024,2025,2026):
        scored,rules,cal_n=w.build_forward_scored(races,year)
        rec=[]
        for row in scored:
            race=row["race"]
            if year==2026 and race.race_date>"2026-06-30":
                continue
            paid=payouts.get(race.race_id)
            if not paid:
                continue
            board,mass,participate,dominant,action=v32.apply_overlay(row,rules)
            if not participate:
                continue
            t=tickets(board)
            actual,payout=paid
            hit=actual in t
            r={
              "year":year,"race_id":race.race_id,"race_date":race.race_date,
              "grade":race.grade,"race_type":race.race_type,
              "board":[list(x) for x in board],"ticket_count":len(t),
              "actual":list(actual),"hit":hit,
              "payout_yen":payout if hit else 0,
              "actual_payout_yen":payout,
              "board_mass":float(mass),"overlay_action":action,
            }
            rec.append(r);all_records.append(r)
        all_years[str(year)]={"calibration_rows":cal_n,"summary":summarize(rec)}
        by_grade={}
        for g in ("G1","G2","G3"):
            by_grade[g]=summarize([r for r in rec if r["grade"]==g])
        all_years[str(year)]["by_grade"]=by_grade

    report={
      "study":"v32 forward ordered trifecta board conversion",
      "rule":"participating races only; all legal ordered triples implied by board; 100 yen each",
      "guards":{"odds_used_for_selection":False,"popularity_used":False,"target_results_used_at_inference":False,"payout_used_for_selection":False},
      "years":all_years,
      "overall":summarize(all_records),
      "records":all_records,
      "payout_archive_stats":stats,
    }
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({k:v for k,v in report.items() if k!="records" and k!="payout_archive_stats"},ensure_ascii=False,indent=2))

if __name__=="__main__": main()
