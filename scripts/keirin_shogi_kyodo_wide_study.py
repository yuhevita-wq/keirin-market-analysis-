#!/usr/bin/env python3
from __future__ import annotations

"""Targeted forward study for Kyodo News Cup early-day wide conversion.

Hypothesis:
The Kyodo News Cup's automatic race assignment on days 1-2 may preserve
"top-3 set" information while degrading exact finishing-order information,
making wide conversion more useful than in ordinary G2 races.

No odds/popularity are used for ticket selection. Wide payouts are post-race
rewards only.
"""

import importlib.util
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/"scripts/keirin_shogi_ninecar_wide_study.py"
OUT=ROOT/"results/keirin_shogi/ninecar_wide_kyodo_study"
OUT.mkdir(parents=True,exist_ok=True)

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name]=mod
    spec.loader.exec_module(mod)
    return mod

w=load("ninecar_wide_kyodo_base",BASE)

EVENTS={
  2023:("2023-09-15","2023-09-18"),
  2024:("2024-09-13","2024-09-16"),
  2025:("2025-09-12","2025-09-15"),
}

CORE=[
 "board_cross_all","board_adjacent","board_row12","board_row13","board_row23",
 "joint_top1","joint_top3","joint_top5","joint_top8",
 "cross_joint_top1","cross_joint_top3","cross_joint_top5","cross_joint_top8",
 "joint_rank3to5","joint_rank4to8","cross_joint_rank3to5","cross_joint_rank4to8",
]

def in_range(date,start,end):
    return start<=date<=end

def subset(rows, pred):
    return [r for r in rows if pred(r)]

def metrics(rows_by_strategy,pred):
    return {s:w.summarize(subset(rows_by_strategy[s],pred)) for s in CORE}

def main():
    races=w.v2.load_races()
    payouts,_=w.load_wide_payouts()
    years=[]
    all_records={s:[] for s in w.STRATEGIES}

    for year in (2023,2024,2025):
        scored,rules,cal_n=w.build_forward_scored(races,year)
        ev=w.evaluate_year(year,scored,rules,payouts)
        for s in w.STRATEGIES:
            all_records[s].extend(ev["records"][s])
        start,end=EVENTS[year]
        days12_end=start[:8]+f"{int(start[-2:])+1:02d}"
        # Event dates are four consecutive days; day1-2 are start and start+1.
        from datetime import date,timedelta
        d0=date.fromisoformat(start)
        d1=(d0+timedelta(days=1)).isoformat()
        d2=(d0+timedelta(days=2)).isoformat()
        d3=(d0+timedelta(days=3)).isoformat()

        recs=ev["records"]
        event_pred=lambda r,s=start,e=end: in_range(r["race_date"],s,e) and r["grade"]=="G2"
        early_pred=lambda r,a=start,b=d1: in_range(r["race_date"],a,b) and r["grade"]=="G2"
        late_pred=lambda r,a=d2,b=d3: in_range(r["race_date"],a,b) and r["grade"]=="G2"
        other_g2=lambda r,s=start,e=end: r["grade"]=="G2" and not in_range(r["race_date"],s,e)

        years.append({
          "year":year,
          "event_dates":[start,end],
          "day12_dates":[start,d1],
          "day34_dates":[d2,d3],
          "kyodo_all":metrics(recs,event_pred),
          "kyodo_day1_2":metrics(recs,early_pred),
          "kyodo_day3_4":metrics(recs,late_pred),
          "other_g2_same_year":metrics(recs,other_g2),
        })

    # pool historical Kyodo across 2023-2025
    pooled={}
    for label in ("kyodo_all","kyodo_day1_2","kyodo_day3_4","other_g2"):
        pooled[label]={}
        for s in CORE:
            rows=[]
            for year,(start,end) in EVENTS.items():
                d0=__import__("datetime").date.fromisoformat(start)
                d1=(d0+__import__("datetime").timedelta(days=1)).isoformat()
                d2=(d0+__import__("datetime").timedelta(days=2)).isoformat()
                d3=(d0+__import__("datetime").timedelta(days=3)).isoformat()
                yr=[r for r in all_records[s] if r["race_date"].startswith(str(year))]
                if label=="kyodo_all":
                    rows += [r for r in yr if r["grade"]=="G2" and in_range(r["race_date"],start,end)]
                elif label=="kyodo_day1_2":
                    rows += [r for r in yr if r["grade"]=="G2" and in_range(r["race_date"],start,d1)]
                elif label=="kyodo_day3_4":
                    rows += [r for r in yr if r["grade"]=="G2" and in_range(r["race_date"],d2,d3)]
                else:
                    rows += [r for r in yr if r["grade"]=="G2" and not in_range(r["race_date"],start,end)]
            pooled[label][s]=w.summarize(rows)

    comparison=[]
    for s in CORE:
        e=pooled["kyodo_day1_2"][s]
        l=pooled["kyodo_day3_4"][s]
        o=pooled["other_g2"][s]
        comparison.append({
          "strategy":s,
          "early_roi":e["roi_pct"],
          "early_hit":e["race_hit_rate_pct"],
          "early_n":e["bet_races"],
          "late_roi":l["roi_pct"],
          "other_g2_roi":o["roi_pct"],
          "early_minus_other":(e["roi_pct"]-o["roi_pct"]) if e["roi_pct"] is not None and o["roi_pct"] is not None else None,
          "early_minus_late":(e["roi_pct"]-l["roi_pct"]) if e["roi_pct"] is not None and l["roi_pct"] is not None else None,
        })
    comparison.sort(key=lambda x:(x["early_minus_other"] if x["early_minus_other"] is not None else -999),reverse=True)

    report={
      "study":"kyodo_news_cup_wide_special_structure",
      "hypothesis":"Kyodo days 1-2 automatic assignment may favor top3-pair/wide conversion relative to normal G2.",
      "event_windows":EVENTS,
      "guards":{
        "prediction_odds_used":False,
        "prediction_popularity_used":False,
        "payout_post_race_only":True,
        "2026_event_excluded_from validation":True,
      },
      "years":years,
      "pooled":pooled,
      "comparison":comparison,
    }
    (OUT/"summary.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    lines=["# Kyodo News Cup wide-special study","",
           "Historical validation only: 2023-2025. 2026 is excluded because it generated the hypothesis.","",
           "|strategy|early 1-2 ROI|late 3-4 ROI|other G2 ROI|early-other|early hit|n|",
           "|---|---:|---:|---:|---:|---:|---:|"]
    for x in comparison:
        lines.append(f"|{x['strategy']}|{x['early_roi']:.1f}%|{x['late_roi']:.1f}%|{x['other_g2_roi']:.1f}%|{x['early_minus_other']:+.1f}pt|{x['early_hit']:.1f}%|{x['early_n']}|")
    (OUT/"summary.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(json.dumps({"top":comparison[:10],"out":str(OUT/"summary.json")},ensure_ascii=False,indent=2))

if __name__=="__main__":
    main()
