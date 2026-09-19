#!/usr/bin/env python3
from __future__ import annotations
import importlib.util, json, sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/"scripts/keirin_shogi_ninecar_wide_study.py"
OUT=ROOT/"results/keirin_shogi/kyodo_2026_day1_h5_compression.json"
DAY="2026-09-18"
YEAR=2026
STAKE=100

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name]=mod
    spec.loader.exec_module(mod)
    return mod

w=load("wide_base_2026_day1_h5",BASE)

def top3_inclusion(joint):
    p1,p2,p3=w.v2.marginals(joint)
    return {n:p1[n]+p2[n]+p3[n] for n in p1}

def summarize(rows,key):
    bets=[r for r in rows if r[key+"_tickets"]>0]
    tickets=sum(r[key+"_tickets"] for r in bets)
    payout=sum(r[key+"_payout"] for r in bets)
    hits=sum(r[key+"_payout"]>0 for r in bets)
    multi=sum(r[key+"_hit_count"]>=2 for r in bets)
    triple=sum(r[key+"_hit_count"]>=3 for r in bets)
    return {
        "races":len(bets),"tickets":tickets,"stake_yen":tickets*STAKE,
        "payout_yen":payout,"profit_yen":payout-tickets*STAKE,
        "roi_pct":100*payout/(tickets*STAKE) if tickets else None,
        "hit_races":hits,"hit_rate_pct":100*hits/len(bets) if bets else None,
        "multi_hit_races":multi,"triple_hit_races":triple,
        "avg_tickets":tickets/len(bets) if bets else None,
        "max_race_payout_yen":max((r[key+"_payout"] for r in bets),default=0),
    }

def eval_tickets(tickets,paid):
    hits=sorted(set(tickets)&set(paid))
    return len(tickets),sum(paid[p] for p in hits),len(hits),hits

def main():
    races=w.v2.load_races()
    payouts,_=w.load_wide_payouts()
    scored,rules,cal_n=w.build_forward_scored(races,YEAR)
    rows=[]
    for r in scored:
        race=r["race"]
        if race.grade!="G2" or race.race_date!=DAY:
            continue
        paid=payouts.get(race.race_id)
        if not paid:
            continue

        board,mass,participate,dominant,action=w.v32.apply_overlay(r,rules)
        if not participate:
            continue

        base=w.union_pairs(board)
        pair_probs=w.joint_pair_probabilities(r["joint"])
        inc=top3_inclusion(r["joint"])
        ranked=sorted(inc,key=lambda n:(-inc[n],n))
        strong_pair=tuple(sorted((ranked[0],ranked[1])))

        line_of={int(e["car_no"]):int(float(e.get("line_id") or 0)) for e in race.entries}
        same_line=[
            p for p in base
            if line_of.get(p[0],0)>0 and line_of.get(p[0])==line_of.get(p[1])
        ]
        strong_line_pair=max(
            same_line,
            key=lambda p:(pair_probs.get(p,0.0),-p[0],-p[1]),
            default=None,
        )

        h1=set(base); h1.discard(strong_pair)
        h2=set(h1)
        if strong_line_pair is not None:
            h2.discard(strong_line_pair)

        h3_pair=max(h2,key=lambda p:(pair_probs.get(p,0.0),-p[0],-p[1]),default=None)
        h3=set(h2)
        if h3_pair is not None: h3.discard(h3_pair)

        h4_pair=max(h3,key=lambda p:(pair_probs.get(p,0.0),-p[0],-p[1]),default=None)
        h4=set(h3)
        if h4_pair is not None: h4.discard(h4_pair)

        actual_centers=[]
        if strong_pair in base: actual_centers.append(("H1",strong_pair))
        if strong_line_pair is not None and strong_line_pair in h1:
            actual_centers.append(("H2",strong_line_pair))
        if h3_pair is not None and h3_pair in h2:
            actual_centers.append(("H3",h3_pair))
        if h4_pair is not None and h4_pair in h3:
            actual_centers.append(("H4",h4_pair))

        touch={n:0 for n in inc}
        for _,pair in actual_centers:
            touch[pair[0]]+=1; touch[pair[1]]+=1

        rel={p:touch.get(p[0],0)+touch.get(p[1],0) for p in h4}
        order=sorted(h4,key=lambda p:(-rel[p],-pair_probs.get(p,0.0),p[0],p[1]))

        checkpoints={}
        for target in (6,5,4,3):
            tickets=set(h4)
            removed=[]
            for p in order:
                if len(tickets)<=target: break
                tickets.discard(p); removed.append(p)
            checkpoints[target]=(tickets,removed)

        rec={
            "race_id":race.race_id,"race_date":race.race_date,
            "result":list(race.order),"board":[list(x) for x in board],
            "board_unique_riders":len(set(board[0])|set(board[1])|set(board[2])),
            "base_selected_pairs":[list(x) for x in sorted(base)],
            "h1_pair":list(strong_pair),"h2_pair":list(strong_line_pair) if strong_line_pair else None,
            "h3_pair":list(h3_pair) if h3_pair else None,"h4_pair":list(h4_pair) if h4_pair else None,
            "history_touch":touch,
            "history_cut_order":[list(x) for x in order],
        }

        for key,tix in (("base",base),("h1",h1),("h2",h2),("h3",h3),("h4",h4)):
            n,pay,hc,hp=eval_tickets(tix,paid)
            rec[key+"_tickets"]=n; rec[key+"_payout"]=pay; rec[key+"_hit_count"]=hc
            rec[key+"_hits"]=[list(x) for x in hp]

        for target,(tix,removed) in checkpoints.items():
            key=f"h5_{target}"
            n,pay,hc,hp=eval_tickets(tix,paid)
            rec[key+"_tickets"]=n; rec[key+"_payout"]=pay; rec[key+"_hit_count"]=hc
            rec[key+"_hits"]=[list(x) for x in hp]
            rec[key+"_removed"]=[list(x) for x in removed]
            rec[key+"_selected_pairs"]=[list(x) for x in sorted(tix)]
        rows.append(rec)

    report={
        "study":"2026 Kyodo day1 H1-H4 plus iterative history-relation H5 compression",
        "date":DAY,"calibration_rows":cal_n,
        "definition":{
            "board":"v3.2 forward board; no odds/popularity/payout at inference",
            "h_order":"H1 -> H2 -> H3 -> H4 -> iterative H5",
            "h5":"Rank H4 survivors by cumulative H1-H4 rider touch score desc; tie by joint co-top3 probability desc. Remove one at a time.",
            "checkpoints":"Evaluate the same pre-race H5 cut order at 6,5,4,3 tickets. No result-dependent stopping.",
            "stake":"100 yen equal stake per pair",
        },
        "base":summarize(rows,"base"),"h1":summarize(rows,"h1"),"h2":summarize(rows,"h2"),
        "h3":summarize(rows,"h3"),"h4":summarize(rows,"h4"),
        "h5_6":summarize(rows,"h5_6"),"h5_5":summarize(rows,"h5_5"),
        "h5_4":summarize(rows,"h5_4"),"h5_3":summarize(rows,"h5_3"),
        "races":rows,
    }
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=="__main__":
    main()
