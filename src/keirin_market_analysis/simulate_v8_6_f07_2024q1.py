from __future__ import annotations

import json
from pathlib import Path

from simulate_v8_1_f02_2024q1 import load, pi, pl, streak, STAKE
from v8_4_f05_incremental_growth import build_v8_4_f05
from v8_6_f07_breakeven_adjusted_growth import build_v8_6_f07

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts/v8_6_f07_2024q1"


def summarize(rows):
    b = len(rows)
    h = sum(r["hit"] for r in rows)
    tickets = sum(r["ticket_count"] for r in rows)
    stake = tickets * STAKE
    payout = sum(r["payout_yen"] for r in rows)
    losing_hits = sum(1 for r in rows if r["hit"] and r["payout_yen"] < r["ticket_count"] * STAKE)
    return {
        "races": b,
        "hits": h,
        "hit_rate_pct": 100*h/b if b else None,
        "tickets": tickets,
        "avg_tickets": tickets/b if b else None,
        "min_tickets": min((r["ticket_count"] for r in rows), default=None),
        "max_tickets": max((r["ticket_count"] for r in rows), default=None),
        "stake_yen": stake,
        "payout_yen": payout,
        "profit_yen": payout-stake,
        "roi_pct": 100*payout/stake if stake else None,
        "max_losing_streak": streak(rows) if rows else 0,
        "losing_hit_races": losing_hits,
        "losing_hit_share_pct": 100*losing_hits/h if h else None,
    }


def main():
    races, trio, tf, pay = load()
    base=[]; new=[]
    population=0
    for rid,r in sorted(races.items(), key=lambda kv:(kv[1].get("race_date",""),kv[0])):
        if r.get("meeting_grade")!="F1" or not (r.get("race_type") or "").startswith("Ｓ級"):
            continue
        if pi(r.get("entry_count"))!=7: continue
        if len(trio.get(rid,{}))!=35 or len(tf.get(rid,{}))!=210: continue
        cars=sorted({v for c in trio[rid] for v in c})
        lines=pl(r.get("predicted_line_formation"))
        if len(cars)!=7 or lines is None or set(v for line in lines for v in line)!=set(cars): continue
        if rid not in pay or not pay[rid]: continue
        population += 1

        b=build_v8_4_f05(trio[rid],tf[rid],r.get("predicted_line_formation") or "")
        n=build_v8_6_f07(trio[rid],tf[rid],r.get("predicted_line_formation") or "")
        if not b.get("buy"):
            continue
        for d,rows in ((b,base),(n,new)):
            ts=tuple(d["tickets"])
            wins=[t for t in ts if t in pay[rid]]
            rows.append({
                "race_id":rid,"race_date":r.get("race_date"),"track":r.get("track"),
                "ticket_count":len(ts),"hit":int(bool(wins)),
                "payout_yen":sum(pay[rid][t] for t in wins),
                "formation":d.get("formation"),"q_mass":d.get("q_mass"),
            })

    res={
        "scheme":"v8.6-F07","dataset":"2024Q1","population":population,
        "selector":"maximize BAHS=sum q(t)*min(1,odds(t)/N) over unchanged v8.4 growth path",
        "thresholds":"no fitted threshold; only mathematical break-even 1.0",
        "v8_4":summarize(base),"v8_6":summarize(new),
        "delta":{
            "avg_tickets":summarize(new)["avg_tickets"]-summarize(base)["avg_tickets"],
            "hit_rate_pp":summarize(new)["hit_rate_pct"]-summarize(base)["hit_rate_pct"],
            "roi_pp":summarize(new)["roi_pct"]-summarize(base)["roi_pct"],
            "profit_yen":summarize(new)["profit_yen"]-summarize(base)["profit_yen"],
        }
    }
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/"summary.json").write_text(json.dumps(res,ensure_ascii=False,indent=2),encoding="utf-8")
    print("V8_6_F07_Q1_RESULT_BEGIN")
    print(json.dumps(res,ensure_ascii=False,indent=2))
    print("V8_6_F07_Q1_RESULT_END")

if __name__=="__main__": main()
