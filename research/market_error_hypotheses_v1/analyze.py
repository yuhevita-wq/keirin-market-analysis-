"""Exploratory, complete-7-rider cross-market mispricing diagnostics.
Uses 2023–2026 S F1 archived final odds, matches outcomes; no OOS claim.
One row per race per selector. No results are used in the selections.
"""
from __future__ import annotations
import csv
import itertools
import json
import math
from collections import defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/"data/audits/market_mispricing_hypotheses_2023_2026.json"

def rows(path):
    with path.open(encoding="utf-8-sig",newline="") as f:
        yield from csv.DictReader(f)

def folder_list():
    for year in (2023,2024,2025):
        for f in sorted((ROOT/f"data/{year}/s_class_f1_all_parts").glob("*")):
            yield f
    yield ROOT/"data/2026_h1/s_class_f1_all"
    for year in (2023,2024,2025):
        yield ROOT/f"data/{year}/s_class_yosen"
    yield ROOT/"data/2026_h1/s_class_yosen"

def oddsfiles(f):
    if (f/"trifecta_final_odds.csv").exists():
        return [f/"trifecta_final_odds.csv"]
    return sorted(f.glob("trifecta_final_odds_part*.csv"))

def results(f):
    result=defaultdict(list)
    for r in rows(f/"results.csv"):
        try: p,n=int(r["finish_position"]),int(r["car_no"])
        except (KeyError, ValueError):continue
        if p in (1,2,3):result[r["race_id"]].append((p,n))
    return {rid:tuple(n for _,n in sorted(v)) for rid,v in result.items()
            if len(v)==3 and sorted(x[0] for x in v)==[1,2,3]}

def linemap(race, cars):
    text=race.get("predicted_line_formation","")
    out={}
    groups=[]
    for g in text.split("/"):
        try: line=[int(v.strip()) for v in g.split("-")]
        except ValueError:return {}, "unknown"
        if not set(line).issubset(cars) or len(line)!=len(set(line)):return {}, "unknown"
        groups.append(line)
        for c in line:out[c]=len(groups)-1
    if len(out)!=len(cars) or not all(out.get(c) is not None for c in cars):
        return {}, "unknown"
    shape="-".join(map(str,sorted((len(g) for g in groups),reverse=True)))
    return out,shape

def shape3(v,line):
    if not line or any(c not in line for c in v):return "unknown"
    hist={}
    for car in v:hist[line[car]]=hist.get(line[car],0)+1
    return {1:"three_lines",2:"two_lines",3:"same_line"}[len(hist)]

def oddsval(r):
    if r.get("odds_status")!="available":return None
    try: v=float(r["odds"])
    except (ValueError,KeyError):return None
    if not math.isfinite(v) or v<1:return None
    return v

def init():
    return {"races":0,"tickets":0,"wins":0,"expected_trio":0.0,"expected_tri3":0.0,
            "return_units":0.0,"sum_logloss_trio":0.0,"sum_logloss_tri3":0.0,
            "sum_logloss_gain_sq":0.0}
def put(agg,win,odds,qtrio,qtri3,entire=False):
    agg["tickets"]+=1
    agg["wins"]+=int(win)
    agg["expected_trio"]+=qtrio
    agg["expected_tri3"]+=qtri3
    if win:agg["return_units"]+=odds
    if entire:
        gain=math.log(qtri3/qtrio)
        agg["races"]+=1
        agg["sum_logloss_trio"]-=math.log(qtrio)
        agg["sum_logloss_tri3"]-=math.log(qtri3)
        agg["sum_logloss_gain_sq"]+=gain*gain

def summary(a):
    count=a["tickets"]
    r=a["races"]
    obs=a["wins"]
    return {"races":r,"tickets":count,"wins":obs,
            "expected_trio":round(a["expected_trio"],3),
            "expected_tri3":round(a["expected_tri3"],3),
            "observed_over_trio_expected":round(obs/a["expected_trio"],4) if a["expected_trio"] else None,
            "observed_over_tri3_expected":round(obs/a["expected_tri3"],4) if a["expected_tri3"] else None,
            "roi_pct_final_trio":round(a["return_units"]/count*100,2) if count else None,
            "logloss_trio":round(a["sum_logloss_trio"]/r,6) if r else None,
            "logloss_tri3":round(a["sum_logloss_tri3"]/r,6) if r else None,
            "logloss_gain":round((a["sum_logloss_trio"]-a["sum_logloss_tri3"])/r,6) if r else None,
            "gain_se":round(math.sqrt(max(0, (a["sum_logloss_gain_sq"]/r-((a["sum_logloss_trio"]-a["sum_logloss_tri3"])/r)**2)/(r-1))),6) if r>1 else None}

def ratio_bin(r):
    if r is None:return "no_votes"
    if r<2:return "tri3_votes_under_2x_trio"
    if r<5:return "tri3_votes_2x_to_5x"
    if r<10:return "tri3_votes_5x_to_10x"
    return "tri3_votes_10x_plus"

def ev_bin(r):
    if r<.70:return "lt_0.70"
    if r<.85:return "0.70_0.85"
    if r<1:return "0.85_1.00"
    if r<1.05:return "1.00_1.05"
    if r<1.10:return "1.05_1.10"
    if r<1.25:return "1.10_1.25"
    return "ge_1.25"

def runner():
    data=defaultdict(init)
    rows_counter=defaultdict(int)
    counted=set()
    skipped=defaultdict(int)
    for folder in folder_list():
        odds_paths=oddsfiles(folder)
        if not odds_paths or not all((folder/name).exists() for name in ("races.csv","entries.csv","results.csv","trio_final_odds.csv")):
            continue
        print("processing",folder.relative_to(ROOT),flush=True)
        races={r["race_id"]:r for r in rows(folder/"races.csv")}
        entries=defaultdict(set)
        for r in rows(folder/"entries.csv"):
            try:entries[r["race_id"]].add(int(r["car_no"]))
            except (ValueError,KeyError):pass
        win=results(folder)
        trio=defaultdict(dict)
        votes3={}
        for r in rows(folder/"trio_final_odds.csv"):
            v=oddsval(r)
            if v is not None:
                try:
                    key=tuple(sorted(int(x) for x in r["combination"].split("=")))
                    if len(set(key))==3:trio[r["race_id"]][key]=v
                except (ValueError,KeyError):pass
            if r["race_id"] not in votes3:
                try:votes3[r["race_id"]]=float(r["total_votes"])
                except (ValueError,KeyError):votes3[r["race_id"]]=None

        def finish(rid, odds, tv3):
            if rid is None:return
            if rid in counted:
                skipped["duplicates"]+=1
                return
            if rid not in races or rid not in win or len(entries[rid])!=7:
                skipped["missing_result_or_not7"]+=1
                return
            cars=entries[rid]
            if len(odds)!=210 or len(trio.get(rid,{}))!=35:
                skipped["incomplete_market"]+=1
                return
            winner=tuple(sorted(win[rid]))
            if winner not in trio[rid] or tuple(win[rid]) not in odds:
                skipped["invalid_winner"]+=1
                return
            s3=sum(1/o for o in odds.values())
            st=sum(1/o for o in trio[rid].values())
            if not s3 or not st:return
            q3={k:(1/o)/s3 for k,o in odds.items()}
            qt={k:(1/o)/st for k,o in trio[rid].items()}
            q3set=defaultdict(float)
            for k,q in q3.items():q3set[tuple(sorted(k))]+=q
            if not all(q3set[s]>0 for s in qt):return
            counted.add(rid)
            race=races[rid]
            year=race["race_date"][:4]
            line,shape=linemap(race,cars)
            type_=race.get("race_type","unknown")
            try:ratio=tv3/votes3[rid] if tv3 and votes3[rid] else None
            except (TypeError,ZeroDivisionError):ratio=None
            rbin=ratio_bin(ratio)
            lltr=qt[winner]
            ll3=q3set[winner]
            for k in (("year",year),("line",year,shape),("type",year,type_),
                      ("liquidity",year,rbin),("overall","all"),
                      ("shape",shape)):
                a=data[k];put(a,True,trio[rid][winner],lltr,ll3,entire=True)
            rows_counter["complete_races"]+=1
            popularity={s:i+1 for i,s in enumerate(sorted(qt,key=lambda s:(-qt[s],s)))}
            for s,pr in qt.items():
                odd=trio[rid][s]
                ex=q3set[s]*odd
                wins=(s==winner)
                kind=shape3(s,line)
                for k in (("evbin",year,ev_bin(ex)),
                          ("evbin_all",ev_bin(ex)),
                          ("ticket_shape",year,kind),
                          ("shape_ev",year,shape,ev_bin(ex)) if shape in ("3-2-2","2-2-2-1") else ("ignore",),
                          ("popular_ev",year,"top10" if popularity[s]<=10 else "other",ev_bin(ex))):
                    if k[0]=="ignore":continue
                    put(data[k],wins,odd,qt[s],q3set[s])
            # One candidate per race with no outcome input.
            for only_pop in ("all","top10"):
                ranked=[s for s in qt if only_pop=="all" or popularity[s]<=10]
                pick=max(ranked,key=lambda s:(q3set[s]*trio[rid][s],-popularity[s]))
                ex=q3set[pick]*trio[rid][pick]
                for threshold in (1.0,1.05,1.10,1.20):
                    if ex<threshold:continue
                    for k in (("selector",year,only_pop,str(threshold)),
                              ("selector_by_shape",year,shape,only_pop,str(threshold))):
                        put(data[k],winner==pick,trio[rid][pick],qt[pick],q3set[pick])
            # Trifecta market probabilities may be good predictors but not value at extreme odds.
        for path in odds_paths:
            rid=None;odds={};tv3=None
            for r in rows(path):
                nextid=r["race_id"]
                if rid is not None and nextid!=rid:
                    finish(rid,odds,tv3)
                    odds={};tv3=None
                rid=nextid
                if tv3 is None:
                    try:tv3=float(r["total_votes"])
                    except (ValueError,KeyError):pass
                value=oddsval(r)
                if value is None:continue
                try:
                    combo=tuple(int(v) for v in r["combination"].split("-"))
                    if len(combo)==3 and len(set(combo))==3:odds[combo]=value
                except (ValueError,KeyError):pass
            finish(rid,odds,tv3)
    report={
        "status":"EXPLORATORY_LOOK_ELSEWHERE_BEFORE_TRADING",
        "scope":"seven-rider S F1 and additional nonduplicated S prelim, 2023-2026H1",
        "price_phase":"final",
        "decision_features":"market odds + source predicted lines + market volume only; results used for grading only",
        "counted_races":len(counted),"counts":dict(rows_counter),
        "skipped":dict(skipped),
        "metric_definition":{
            "cross_market_xEV":"normalized trifecta-set implied probability multiplied by final trio odds",
            "line_shape":"sorted length of published predicted lines; no outcome-dependent allocation",
            "ticket_shape":"number of distinct published predicted line IDs in unordered 3-person set",
            "single_race_rule":"choose max cross_market_xEV, optional rank<=10 filter, before reading winner",
            "observation":"Every selector includes at most one ticket per race, 100 yen equally weighted",
            "caveat":"No previous 15min T-15 observations here, no untouched prospective test, xEV bins explored ex post"},
        "year":{y:summary(data[("year",y)]) for y in ("2023","2024","2025","2026")},
        "line_by_year":{y:{s:summary(data[("line",y,s)]) for s in
            ("3-2-2","2-2-2-1","3-3-1","3-1-3","4-2-1","4-3","other","unknown")
            if data[("line",y,s)]["races"]} for y in ("2023","2024","2025","2026")},
        "type_by_year":{y:{key[2]:summary(v) for key,v in data.items() if key[0]=="type" and key[1]==y}
                        for y in ("2023","2024","2025","2026")},
        "liquidity_by_year":{y:{key[2]:summary(v) for key,v in data.items() if key[0]=="liquidity" and key[1]==y}
                        for y in ("2023","2024","2025","2026")},
        "ev_buckets_all":{key[1]:summary(v) for key,v in data.items() if key[0]=="evbin_all"},
        "ev_buckets_by_year":{y:{key[2]:summary(v) for key,v in data.items() if key[0]=="evbin" and key[1]==y}
                        for y in ("2023","2024","2025","2026")},
        "ticket_shape_by_year":{y:{key[2]:summary(v) for key,v in data.items() if key[0]=="ticket_shape" and key[1]==y}
                        for y in ("2023","2024","2025","2026")},
        "single_pick_thresholds":{y:{p:{t:summary(data[("selector",y,p,t)]) for t in ("1.0","1.05","1.1","1.2") }
                         for p in ("all","top10")} for y in ("2023","2024","2025","2026")},
        "single_pick_by_line":{y:{key[2]+"_"+key[3]+"_"+key[4]:summary(v)
                         for key,v in data.items() if key[0]=="selector_by_shape" and key[1]==y
                         and key[2] in ("3-2-2","2-2-2-1") and key[4] in ("1.05","1.1")}
                         for y in ("2023","2024","2025","2026")},
        "popular_ev_by_year":{y:{key[2]+"_"+key[3]:summary(v)
                          for key,v in data.items() if key[0]=="popular_ev" and key[1]==y}
                          for y in ("2023","2024","2025","2026")},
    }
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("AUDIT",json.dumps({"counted_races":report["counted_races"],
          "year":report["year"],"selectors":report["single_pick_thresholds"]},
           ensure_ascii=False),flush=True)

if __name__=="__main__":
    runner()
