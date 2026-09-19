#!/usr/bin/env python3
from __future__ import annotations
import importlib.util, json, sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/"scripts/keirin_shogi_ninecar_wide_study.py"
OUT=ROOT/"results/keirin_shogi/kyodo_2024_strong2_cut.json"

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name]=mod
    spec.loader.exec_module(mod)
    return mod

w=load("wide_base_strong2_cut",BASE)
START,END="2024-09-13","2024-09-14"
STAKE=100

def top3_inclusion(joint):
    p1,p2,p3=w.v2.marginals(joint)
    return {n:p1[n]+p2[n]+p3[n] for n in p1}

def summarize(rows,key):
    bets=[r for r in rows if r[key+"_tickets"]>0]
    tickets=sum(r[key+"_tickets"] for r in bets)
    payout=sum(r[key+"_payout"] for r in bets)
    hits=sum(r[key+"_payout"]>0 for r in bets)
    return {
        "races":len(bets),"tickets":tickets,"stake_yen":tickets*STAKE,
        "payout_yen":payout,"profit_yen":payout-tickets*STAKE,
        "roi_pct":100*payout/(tickets*STAKE) if tickets else None,
        "hit_races":hits,"hit_rate_pct":100*hits/len(bets) if bets else None,
        "avg_tickets":tickets/len(bets) if bets else None,
    }

def main():
    races=w.v2.load_races()
    payouts,_=w.load_wide_payouts()
    scored,rules,cal_n=w.build_forward_scored(races,2024)
    rows=[]
    for r in scored:
        race=r["race"]
        if race.grade!="G2" or not (START<=race.race_date<=END):
            continue
        paid=payouts.get(race.race_id)
        if not paid: continue
        board,mass,participate,dominant,action=w.v32.apply_overlay(r,rules)
        if not participate: continue
        base=w.union_pairs(board)
        inc=top3_inclusion(r["joint"])
        ranked=sorted(inc,key=lambda n:(-inc[n],n))
        strong_pair=tuple(sorted((ranked[0],ranked[1])))
        cut=set(base)
        cut.discard(strong_pair)
        def pay(ts):
            hp=sorted(set(ts)&set(paid))
            return sum(paid[p] for p in hp),hp
        bp,bh=pay(base); cp,ch=pay(cut)
        rows.append({
            "race_id":race.race_id,"date":race.race_date,"race_type":race.race_type,
            "result":list(race.order),"board":[list(x) for x in board],
            "strongest_two":[ranked[0],ranked[1]],
            "strongest_two_top3_prob":[inc[ranked[0]],inc[ranked[1]]],
            "cut_pair":list(strong_pair),"cut_pair_was_candidate":strong_pair in base,
            "cut_pair_was_winner":strong_pair in paid,
            "cut_pair_payout_yen":paid.get(strong_pair,0),
            "base_tickets":len(base),"base_payout":bp,"base_hits":[list(x) for x in bh],
            "cut_tickets":len(cut),"cut_payout":cp,"cut_hits":[list(x) for x in ch],
        })
    report={
        "study":"2024 Kyodo days1-2 board wide minus strongest-two pair",
        "dates":[START,END],
        "definition":{
            "board":"v3.2 forward board, calibrated only on prior data",
            "base_tickets":"all unique wide pairs among riders appearing anywhere on the 7-piece board",
            "strongest_two":"top two riders by model top3 inclusion probability = P1+P2+P3 from joint504",
            "cut":"remove exactly that strongest-two wide pair if it exists in base tickets",
            "stake":"100 yen per remaining pair","odds_used":False,"popularity_used":False,
        },
        "calibration_rows":cal_n,
        "baseline":summarize(rows,"base"),
        "strong2_cut":summarize(rows,"cut"),
        "cut_effect":{
            "candidate_cut_races":sum(r["cut_pair_was_candidate"] for r in rows),
            "winning_cut_pair_races":sum(r["cut_pair_was_winner"] for r in rows),
            "removed_stake_yen":sum(r["base_tickets"]-r["cut_tickets"] for r in rows)*100,
            "removed_winning_payout_yen":sum(r["cut_pair_payout_yen"] for r in rows if r["cut_pair_was_candidate"] and r["cut_pair_was_winner"]),
        },
        "races":rows,
    }
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({k:report[k] for k in ("baseline","strong2_cut","cut_effect")},ensure_ascii=False,indent=2))

if __name__=="__main__": main()
