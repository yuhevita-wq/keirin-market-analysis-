#!/usr/bin/env python3
"""Inspect actual race facts behind W01's locked formation audit; no model changes."""
from __future__ import annotations
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"research/independent_trifecta_2023"))
sys.path.insert(0,str(ROOT/"research/deletion_ai"))
sys.path.insert(0,str(ROOT))
import observe as obs
from research.v6_c1_weekly_2023.week01 import load_settled_payments

WINS=[
"2820230104020012",
"4520230107010006",
"2320230101010009",
]
LOSSES_TO_CHECK=[
"2320230101020012",
"2720230101010012",
"7120230103010008",
"7120230103030002",
"8620221231020007",
"5420230101020003",
"4520230107010004",
"4220230102020008",
"2720230101030012",
"5620221231030005",
"7120230103030003",
"8120230105010006",
"6220221230030004",
"8320230104010009",
]
def main():
    cards,audit=obs.ingest()
    ids=set(WINS+LOSSES_TO_CHECK)
    official,conflict,counts=load_settled_payments(ids)
    payload=[]
    for rid in WINS+LOSSES_TO_CHECK:
        if rid not in cards:raise AssertionError(("race card not found",rid))
        c=cards[rid]
        positions={}
        for car in sorted(c["scores"],key=int):
            line,pos,n=c["positions"][car]
            positions[car]={"score":round(c["scores"][car],3),
                            "score_rank_cap4":c["ranks"][car],
                            "line_number":line,"position_in_line":pos,"line_size":n}
        payout=official.get(rid)
        result="-".join(c["order"][:3])
        if rid in WINS and (not payout or result!="-".join(payout[0])):
            raise AssertionError(("expected winning race has no official paid result",rid))
        payload.append({"race_id":rid,"date":c["date"],
                        "actual_first_three":result,
                        "line_formation":"/".join("-".join(x) for x in c["groups"]),
                        "positions":positions,
                        "official_payout_yen_100":payout[1] if payout and rid not in conflict and "-".join(payout[0])==result else None,
                        "official_paid_status":"verified" if payout and rid not in conflict and "-".join(payout[0])==result else "missing_or_conflicting"})
    print("W01_SIX_RACES_CANDIDATE_ACTUALS",json.dumps(payload,ensure_ascii=False),flush=True)
if __name__=="__main__":main()
