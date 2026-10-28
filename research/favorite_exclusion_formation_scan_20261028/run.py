"""Exploratory research, NOT a live purchase system.

Questions fixed before running:
 Q1 Does favorite fail more than full-trifecta market predicts within price bands?
 Q2 Which actual positional-rank triplet sets have repeated non-favorite excess?
 Q3 Do those entire formation sets pay back their entire ticket cost?
 Q4 Does a union of TWO disjoint formation cells improve stability?
 Q5 Does a frozen 2021-23 pattern survive 2024 and 2025/2026H1?
 No winners or validation periods used to select a buying rule.
"""
from __future__ import annotations
import os,sys,math,json,itertools
from collections import defaultdict,Counter
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from research.snowberg_observation_corrected_20261027 import observe as corrected

OUT=ROOT/"research/favorite_exclusion_formation_scan_20261028"
BANDS=(2.,3.,5.,8.,10.,12.,15.,20.,25.,30.)
LABELS=("lt2","2to3","3to5","5to8","8to10","10to12","12to15","15to20","20to25","25to30","ge30")
YEARS=("2021","2022","2023","2024","2025","2026")
DEV=YEARS[:3]
VAL=("2024",)
FORWARD=YEARS[4:]
CAPS=("all",8,16,32)
CARS=("all",7)
CELLS=tuple(itertools.product(range(3),repeat=3))
STATS=defaultdict(lambda:[0.,0.,0.,0.,0.,0.,0.])
FAV=defaultdict(lambda:[0.,0.,0.])
AUDIT=Counter()
ILLUSTRATIONS=defaultdict(list) # First-encountered future examples, no best-result selection

def odds_band(v):
    for i,t in enumerate(BANDS):
        if v<t:return LABELS[i]
    return LABELS[-1]

def takeout_market(q, o):
    return (1.0/o)/q

def observe(rid,day,books,finish,rtype,source):
    AUDIT["callbacks"]+=1
    if source!="sealed_four_markets_orientation_verified":
        raise AssertionError("Trifecta key orientation was not verified")
    book=books.get("3rentan",{})
    cars=tuple(sorted({c for tri in book for c in tri}))
    n=len(cars)
    if n<5 or n>9 or len(book)!=n*(n-1)*(n-2):
        AUDIT["invalid_books"]+=1;return
    finish=tuple(finish)
    if finish not in book:
        AUDIT["unmatched_result"]+=1;return
    try:
        odds={tri:float(x) for tri,x in book.items()}
        if any(not math.isfinite(x) or x<1. for x in odds.values()):raise ValueError()
    except (ValueError,TypeError):
        AUDIT["invalid_odds"]+=1;return
    favorite=min(odds,key=lambda tri:(odds[tri],tri))
    band=odds_band(odds[favorite])
    inv={tri:1./o for tri,o in odds.items()}
    Z=sum(inv.values())
    marg=[defaultdict(float) for _ in range(3)]
    for tri,v in inv.items():
        for i in range(3):marg[i][tri[i]]+=v
    ranks=[{car:j for j,car in enumerate(sorted(cars,key=lambda c:(-marg[i][c],c)))} for i in range(3)]
    def cid(tri):
        k=0
        for i in range(3):
            k=k*3+(0 if ranks[i][tri[i]]<2 else 1 if ranks[i][tri[i]]<4 else 2)
        return k
    favorite_cell=cid(favorite)
    mass=[0.]*27
    points=[0]*27
    for tri,v in inv.items():
        i=cid(tri)
        mass[i]+=v/Z
        points[i]+=1
    year=day[:4]
    if year not in YEARS:raise RuntimeError("Out-of-period records")
    for group in CARS:
        if group==7 and n!=7:continue
        if year in FORWARD and len(ILLUSTRATIONS[(year,band,group)])<8:
            ILLUSTRATIONS[(year,band,group)].append({
                "race_id":rid,"date":day,"favorite":favorite,
                "favorite_odds":odds[favorite],"finish":finish,
                "odds":odds,"ranks":ranks,"favorite_cell":favorite_cell})
        f=FAV[(year,band,group)]
        f[0]+=1
        f[1]+=int(finish!=favorite)
        f[2]+=1-inv[favorite]/Z
        for cap in CAPS:
            for k in range(27):
                # Skip the WHOLE rank-cartesian formation if it contains favorite.
                # Never subtract an individual ticket and mislabel it a formation.
                if k==favorite_cell or not points[k]:continue
                if cap!="all" and points[k]>cap:continue
                st=STATS[(year,band,group,cap,k)]
                st[0]+=1
                st[1]+=points[k]
                st[2]+=mass[k]
                if cid(finish)==k: # Winning triple is never the favorite
                    st[3]+=1
                    st[4]+=100*odds[finish]
                st[5]+=100*points[k]
                st[6]+=100*points[k]/Z
    AUDIT["usable"]+=1
    AUDIT["year_"+year]+=1

def add(a,b):
    return [x+y for x,y in zip(a,b)]
def sum_stats(years,band,group,cap,cells):
    out=[0.]*7
    for y in years:
        for c in cells:
            out=add(out,STATS[(y,band,group,cap,c)])
    return out
def metrics(s):
    exposures,points,expected,hits,pay,stake,baseline=s
    return {"cell_race_exposures":int(exposures),"tickets":int(points),
            "avg_points_per_cell_race":round(points/exposures,3) if exposures else None,
            "expected_hits":round(expected,4),"actual_hits":int(hits),
            "hit_over_market":round(hits/expected,4) if expected else None,
            "stake_yen":round(stake),"payout_yen":round(pay),
            "profit_yen":round(pay-stake),
            "roi_pct":round(100*pay/stake,3) if stake else None,
            "normalized_market_reference_roi_pct":round(100*baseline/stake,3) if stake else None}
def actual_car_formations(band,group,cap,cells):
    """Display full genuine 3連単 formations. Never print rank indexes as actual car numbers."""
    for year in FORWARD:
        for race in ILLUSTRATIONS[(year,band,group)]:
            favorite=race["favorite"]
            book=race["odds"]
            ranks=race["ranks"]
            chosen=[]
            components=[]
            for cell in cells:
                if cell==race["favorite_cell"]:continue
                groups=[]
                for slot,bin_index in enumerate(CELLS[cell]):
                    ordered=sorted(ranks[slot],key=lambda c:ranks[slot][c])
                    groups.append([c for c in ordered if
                        (0 if ranks[slot][c]<2 else 1 if ranks[slot][c]<4 else 2)==bin_index])
                valid=[t for t in itertools.product(*groups) if len(set(t))==3]
                if not valid or (cap!="all" and len(valid)>cap):continue
                if favorite in valid:raise AssertionError("Favorite not excluded")
                components.append({"1着":groups[0],"2着":groups[1],
                    "3着":groups[2],"有効点数":len(valid)})
                chosen.extend(valid)
            chosen=sorted(set(chosen))
            if not chosen:continue
            if favorite in chosen:raise AssertionError("Forbidden favorite")
            actual=tuple(race["finish"])
            payout=100*book[actual] if actual in chosen else 0
            return {"race_id":race["race_id"],"date":race["date"],
                "example_selection":"first eligible amongst first 8 future races, not by result",
                "favorite_car_numbers":"-".join(map(str,favorite)),
                "favorite_odds":race["favorite_odds"],
                "actual_car_formations":components,
                "tickets":[{"car_numbers":"-".join(map(str,t)),
                    "odds":book[t],"stake_yen":100} for t in chosen],
                "distinct_points":len(chosen),
                "stake_yen":100*len(chosen),
                "actual_finish":"-".join(map(str,actual)),
                "payout_yen":round(payout,1),
                "profit_yen":round(payout-100*len(chosen),1)}
    return None

def item(band,group,cap,cells):
    tr=metrics(sum_stats(DEV,band,group,cap,cells))
    yr={y:metrics(sum_stats([y],band,group,cap,cells)) for y in YEARS}
    worst=min((yr[y]["roi_pct"] or 0) for y in DEV)
    allthree=all(yr[y]["roi_pct"] is not None and yr[y]["roi_pct"]>100 for y in DEV)
    return {"cells":list(cells),"rank_triplets":[list(CELLS[x]) for x in cells],
            "train":tr,"worst_dev_year_roi_pct":worst,"positive_roi_all_3_dev_years":allthree,
            "validation_2024":yr["2024"],
            "concrete_example":actual_car_formations(band,group,cap,cells),
            "forward_2025":yr["2025"],"forward_2026H1":yr["2026"],
            "forward_combined":metrics(sum_stats(FORWARD,band,group,cap,cells)),
            "dev_year_results":{y:yr[y] for y in DEV},
            "validation_pass": yr["2024"]["roi_pct"] is not None
              and yr["2024"]["roi_pct"]>100 and yr["2024"]["actual_hits"]>yr["2024"]["expected_hits"]}

def main():
    if not os.environ.get(corrected.old.bridge.SECRET_ENV):
        raise RuntimeError("Private key unavailable; refuse fallback")
    corrected.base_observe=observe
    corrected.old.save=lambda status:print("CHECKPOINT",status,dict(AUDIT),flush=True)
    corrected.old.sealed()
    if AUDIT["usable"]<100000:
        raise RuntimeError("Full universe has not been evaluated; refusing partial conclusion")
    candidates=[(c,) for c in range(27)]+list(itertools.combinations(range(27),2))
    reports=[]
    all_rows=[]
    for band in LABELS:
        for group in CARS:
            for cap in CAPS:
                # Rankings based on predeclared TRAIN years only.
                scored=[]
                for cells in candidates:
                    tr=metrics(sum_stats(DEV,band,group,cap,cells))
                    if tr["cell_race_exposures"]<300 or tr["expected_hits"]<15:
                        continue
                    yrs=[metrics(sum_stats([y],band,group,cap,cells)) for y in DEV]
                    worst=min(y["roi_pct"] or 0 for y in yrs)
                    # Conservative training ranking; requires profitability ALL three years,
                    # not maximization of the biggest in-sample cash windfall.
                    stable=all(y["roi_pct"] is not None and y["roi_pct"]>100 for y in yrs)
                    scored.append((int(stable),worst,tr["hit_over_market"] or 0,cells))
                scored.sort(reverse=True)
                rows=[item(band,group,cap,cells) for stable,worst,ratio,cells in scored[:5]]
                stable_count=sum(1 for x in scored if x[0])
                eligible=len(scored)
                chosen=next((x for x in rows if x["positive_roi_all_3_dev_years"]),None)
                reports.append({"favorite_band":band,"cars":group,"component_point_cap":cap,
                    "candidates_screened":eligible,"train_stable_candidate_count":stable_count,
                    "top5_training_only":rows,"frozen_best_stable":chosen,
                    "decision":"RESEARCH_ONLY_NO_BET",
                    "note":"2-cell unions may skip one component if it contains the favorite; the remaining component is still a whole formation."})
                # Publish *all* single-cell results (not just winners) for audit.
                for c in range(27):
                    all_rows.append({"band":band,"cars":group,"cap":cap,
                         **item(band,group,cap,(c,))})
    miss=[]
    for band in LABELS:
        for group in CARS:
            for ys,name in [(DEV,"development_2021_23"),(VAL,"validation_2024"),(FORWARD,"forward_2025_2026H1")]:
                count=hits=expected=0.
                for y in ys:
                    a=FAV[(y,band,group)]
                    count+=a[0];hits+=a[1];expected+=a[2]
                miss.append({"band":band,"cars":group,"period":name,
                    "races":int(count),
                    "observed_favorite_miss_pct":round(100*hits/count,4) if count else None,
                    "market_expected_favorite_miss_pct":round(100*expected/count,4) if count else None,
                    "excess_miss_pp":round(100*(hits-expected)/count,4) if count else None})
    d={"status":"FAVORITE_EXCLUSION_FORMATION_SCAN_COMPLETE",
       "generated_utc":datetime.now(timezone.utc).isoformat(),
       "source_sha":os.getenv("GITHUB_SHA",""),
       "audit":dict(AUDIT),
       "research_questions":["Q1 favorite miss vs normalized market within odds bins",
          "Q2 non-favorite positional rank rectangle 27-cell residual frequency",
          "Q3 100yen-per-ticket whole-rectangle ROI with all losses counted",
          "Q4 unions of two disjoint rectangle formations, not ticket-picking",
          "Q5 frozen 2021-23 discovery vs 2024 and 2025-26H1",
          "Q6 actual-number formation and full ticket-price audit"],
       "definition":{
         "rank_by_position":"Marginal inverse-trifecta-odds market support, separately at 1st/2nd/3rd",
         "rank_bins":["1-2","3-4","5+"],
         "cell":"Position-specific market groups are INPUT LABELS ONLY; final formations are numeric car sets 1着 x 2着 x 3着, repeated car tickets excluded",
         "required_result":"Actual 1着/2着/3着 numbered cars, every exact ticket and odds, full stake, result and payment; not symbolic rank strings",
         "mandatory_favorite_exclusion":"If a cell includes 1番人気 ticket it is not bought in that race AT ALL",
         "unions":"Up to 2 whole cells; no candidate individual winning-ticket selection",
         "point_caps":"Each component separately has no limit/8/16/32; union can have up to twice the cap",
         "selection":"Only 2021-23 years rank candidates; fixed 2024/2025/2026 records, no reselection",
         "odds":"Historical finalized 3連単 odds; not confirmed available at T-15"
       },
       "favorite_miss":miss,
       "candidate_reports":reports,
       "all_single_cells":all_rows,
       "warnings":["Many competing cells/unions and odds bins; training best is subject to multiple-testing bias.",
          "All years have been observed by other research projects, so tests are time-separated not untouched blind.",
          "Coarse marginal rank cells cannot express individual line-head/follower or arbitrary seat subsets.",
          "A positive market residual is not a positive expected ROI after takeout.",
          "This read-only research does not change existing models, rules, or live websites."]}
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/"summary.json").write_text(json.dumps(d,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    compact={k:v for k,v in d.items() if k!="all_single_cells"}
    (OUT/"compact.json").write_text(json.dumps(compact,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("COMPLETE",json.dumps({"audit":dict(AUDIT),
        "favorite_miss_20to25":[x for x in miss if x["band"]=="20to25"],
        "promising":[{"band":r["favorite_band"],"cars":r["cars"],"cap":r["component_point_cap"],
          "candidate":r["frozen_best_stable"]["rank_triplets"] if r["frozen_best_stable"] else None,
          "dev":r["frozen_best_stable"]["train"]["roi_pct"] if r["frozen_best_stable"] else None,
          "validation":r["frozen_best_stable"]["validation_2024"]["roi_pct"] if r["frozen_best_stable"] else None,
          "forward":r["frozen_best_stable"]["forward_combined"]["roi_pct"] if r["frozen_best_stable"] else None
         } for r in reports if r["cars"]==7 and r["component_point_cap"]==16]},
        ensure_ascii=False),flush=True)
if __name__=="__main__":main()
