"""Observe 2021 ONLY: realized joint rank-position SETS vs whole-market probabilities.

NO selection, fitting, investment, payouts, ROI screening, or thresholds by outcome.
2022-2026H1 MUST NOT BE READ BY THIS PROGRAM. This experiment does not
claim those years were never inspected in prior projects.

27 mutually disjoint Cartesian outcome-formation cells:
 For each possible ordered finishing triple (a,b,c), group by pre-race rank
 of a in FIRST-position market marginal, b in SECOND, c in THIRD,
 with ex ante rank bins 1-2, 3-4, 5+.
 One cell is a 3-slot formation (with identical riders excluded).
 All possible outcomes partition into exactly 27 disjoint cells.

Expected frequency = sum of market implied normalized 1/3rentan odds,
not the model's estimate of true probabilities. NO PROFITS IN THIS REPORT.
"""
from __future__ import annotations
import os,sys,json,math,itertools,tempfile
from pathlib import Path
from collections import Counter,defaultdict
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from research.snowberg_observation_corrected_20261027 import observe as corrected
bridge=corrected.old.bridge
OUT=ROOT/"research/formation_subset_observation_20261028"
READ_FIRST="2021-01-01"
READ_LAST="2021-12-31"
RESERVED_START="2022-01-01"
CAR_BANDS=("rank_1_2","rank_3_4","rank_5_plus")
ALL_CELLS=tuple(itertools.product(range(3),repeat=3))
AUDIT=Counter()
STATS=defaultdict(lambda:defaultdict(lambda: [0.,0.,0.,0.,0.,0.]))
EXAMPLES={}
RACE_SUM=Counter()

def binof(rank):
    return 0 if rank<2 else 1 if rank<4 else 2

def calc(rid,day,books,finish,rtype,source):
    if source!="sealed_four_markets_orientation_verified":
        raise RuntimeError("Required validated (a,b,c) price orientation")
    if not READ_FIRST<=day<=READ_LAST:
        raise RuntimeError("2022+ reserved: source read was attempted")
    t=books.get("3rentan",{})
    cars=tuple(sorted({car for w in t for car in w}))
    n=len(cars)
    expected_n=n*(n-1)*(n-2)
    if len(t)!=expected_n or any(not math.isfinite(float(v)) or float(v)<1 for v in t.values()):
        AUDIT["invalid_complete_trifecta_book"]+=1;return
    if not set(t)==set(itertools.permutations(cars,3)):
        AUDIT["noncanonical_or_missing_trifecta_labels"]+=1;return
    actual=tuple(finish)
    if actual not in t: AUDIT["result_outside_trifecta_market"]+=1;return
    inv_sum=sum(1/float(v) for v in t.values())
    if not math.isfinite(inv_sum) or inv_sum<=0:raise AssertionError("Cannot normalize market odds")
    pos_marg=[defaultdict(float) for _ in range(3)]
    for w,odd in t.items():
        pr=(1/float(odd))/inv_sum
        for j in range(3):pos_marg[j][w[j]]+=pr
    position_ranks=[]
    formation_slots=[]
    for j in range(3):
        order=sorted(cars,key=lambda c:(-pos_marg[j][c],c))
        position_ranks.append({car:binof(rank) for rank,car in enumerate(order)})
        formation_slots.append([[c for idx,c in enumerate(order) if binof(idx)==b] for b in range(3)])
    masses=[0.0]*27
    legal_counts=[0]*27
    for w,odd in t.items():
        b0,b1,b2=(position_ranks[j][w[j]] for j in range(3))
        idx=9*b0+3*b1+b2
        masses[idx]+=(1/float(odd))/inv_sum
        legal_counts[idx]+=1
    if sum(legal_counts)!=expected_n or not math.isclose(sum(masses),1,abs_tol=1e-9):
        raise AssertionError("All 3rentan possible outcomes must be partitioned exactly once")
    winning_group=tuple(position_ranks[j][actual[j]] for j in range(3))
    group_keys=(("2021","all"),("2021","half_1" if day<="2021-06-30" else "half_2"),
                ("2021","cars_"+str(n)),("2021","type_"+str(rtype)))
    for part in group_keys:
        for b0,b1,b2 in ALL_CELLS:
            idx=9*b0+3*b1+b2
            row=STATS[part][idx]
            q=masses[idx]
            row[0]+=1 # race count
            row[1]+=q # expected hits in the set
            row[2]+=q*(1-q) # Bernoulli variance proxy
            row[3]+=int(winning_group==(b0,b1,b2)) # actual hits
            row[4]+=legal_counts[idx] # legal points in this set
            row[5]+=int(legal_counts[idx]>0)
    if n not in EXAMPLES:
        # strictly first observed completed race for each car count, never best-result case
        groups=[{"rank_bands":[CAR_BANDS[i] for i in v],
                 "first":formation_slots[0][v[0]],"second":formation_slots[1][v[1]],
                 "third":formation_slots[2][v[2]],
                 "legal_points":legal_counts[9*v[0]+3*v[1]+v[2]],
                 "market_mass":round(masses[9*v[0]+3*v[1]+v[2]],6)}
                for v in (winning_group,(0,0,0),(1,1,1),(2,2,2))]
        EXAMPLES[n]={"example_policy":"FIRST_ENCOUNTERED_RACE_FOR_THIS_CAR_COUNT",
                     "race_id":rid,"date":day,"actual_result":list(actual),
                     "winning_cell":[CAR_BANDS[i] for i in winning_group],
                     "illustrative_cells":groups}
    AUDIT["valid_races"]+=1
    AUDIT["cars_"+str(n)]+=1
    AUDIT["half_1" if day<="2021-06-30" else "half_2"]+=1

def run():
    AUDIT["read_period_2021_only"]=1
    labels,ls=bridge.verified_results(2021)
    AUDIT["verified_result_count"]=ls["verified_distinct_results"]
    catalog,info=bridge.catalog(READ_FIRST,READ_LAST)
    AUDIT["encrypted_file_count"]=info["sealed_files"]
    if any(d<READ_FIRST or d>READ_LAST for d in catalog):
        raise AssertionError("Archive catalog leaked reserved period")
    corrected.base_observe=calc
    old=corrected.old
    clean_tmp,key=bridge.secure_key_temp()
    try:
        for day,files in catalog.items():
            union={};conflicts=set()
            with tempfile.TemporaryDirectory(prefix="formation_obs_2021_") as tmproot:
                for path,sha in files:
                    for rid,rows in bridge.decrypt_sealed(path,sha,key,tmproot).items():
                        if rid in conflicts:continue
                        canonical=sorted(rows)
                        if rid in union and union[rid]!=canonical:
                            union.pop(rid);conflicts.add(rid)
                        else:union[rid]=canonical
            AUDIT["conflicting_books"]+=len(conflicts)
            for rid,rows in union.items():
                norm,reason=bridge.normalize_book(rid,day,rows)
                if reason:
                    AUDIT["source_filtered_"+reason]+=1;continue
                if rid not in labels:
                    AUDIT["missing_results"]+=1;continue
                books=defaultdict(dict)
                for row in rows:
                    kind=row[2]
                    if kind not in ("2shatan","2shafuku","3rentan","3renpuku"):continue
                    lo,hi=float(row[4]),float(row[5])
                    if lo!=hi: raise AssertionError("Source market has price intervals")
                    t=tuple(map(int,row[3].split("-")))
                    if kind in ("2shafuku","3renpuku"):t=tuple(sorted(t))
                    books[kind][t]=lo
                # requires independent public orientation verification by n, year;
                # when source matches, it calls calc() using corrected all 3rentan keys
                corrected.corrected_observe(rid,day,books,labels[rid],rows[0][1],"sealed_four_markets")
            AUDIT["days_loaded"]+=1
            if AUDIT["days_loaded"]%45==0:
                print("PROGRESS",day,"days",AUDIT["days_loaded"],"valid",AUDIT["valid_races"],flush=True)
    finally:
        clean_tmp.cleanup()
    if AUDIT["valid_races"]<20000:
        raise RuntimeError("2021 observation fewer than 20000 qualified races, not accepted")
    summary=[]
    for part,cellmap in sorted(STATS.items()):
        for idx in range(27):
            row=cellmap[idx]
            b=(idx//9,(idx//3)%3,idx%3)
            exp=float(row[1]);obs=int(row[3]);v=float(row[2]);races=int(row[0])
            summary.append({
                "period":part[0],"segment":part[1],
                "first_rank_band":CAR_BANDS[b[0]],"second_rank_band":CAR_BANDS[b[1]],"third_rank_band":CAR_BANDS[b[2]],
                "races":races,"expected_market_hits":exp,"observed_hits":obs,
                "observed_rate":obs/races if races else None,
                "market_expected_rate":exp/races if races else None,
                "observed_minus_market":obs-exp,
                "observed_over_market":obs/exp if exp else None,
                "null_standard_deviations":(obs-exp)/math.sqrt(v) if v>0 else None,
                "mean_legal_ticket_count":float(row[4])/races if races else None,
                "possible_races":int(row[5])
            })
    for segment in set(x["segment"] for x in summary):
        ss=[x for x in summary if x["segment"]==segment]
        if len(ss)!=27:raise AssertionError("Full 27-cell partition not reported")
        if sum(x["observed_hits"] for x in ss)!=ss[0]["races"]:
            raise AssertionError("Each race must hit EXACTLY ONE set")
        if not math.isclose(sum(x["expected_market_hits"] for x in ss),ss[0]["races"],rel_tol=1e-8,abs_tol=1e-5):
            raise AssertionError("All sets' market prices must add to one")
    report={
        "status":"2021_ONLY_DESCRIPTIVE_FORMATION_SET_OBSERVATION_COMPLETE",
        "generated_utc":datetime.now(timezone.utc).isoformat(),
        "source_commit":os.environ.get("GITHUB_SHA","local"),
        "audit":dict(AUDIT),
        "sample_control":{
            "THIS_experiment_2021":"OBSERVATION_ONLY. Only 2021 labels and ciphertext catalog decrypted.",
            "2022":"RESERVED_FOR_LATER_REPLICATION",
            "2023":"RESERVED_FOR_LATER_HYPOTHESIS",
            "2024":"RESERVED_FOR_LATER_VALIDATION",
            "2025":"RESERVED_FOR_LATER_VALIDATION",
            "2026_H1":"RESERVED_FOR_LATER_VALIDATION",
            "proviso":"These later periods have been analyzed in previous, separate projects; they are not virgin/never-seen data overall."
        },
        "definitions":{
            "market_proxy":"q(ordered triple) = (1/archived final trifecta odds)/sum_all(1/odds)",
            "partition":"Each rider ranked by the marginal market support for that exact finishing position; rank bins [1,2],[3,4],[5,n]. 27 Cartesian products of those position sets, remove repeated riders.",
            "unit_of_observation":"Each race contributes ONE actual set hit; for EACH of 27 sets adds its pre-event market mass. No selected bets.",
            "null_variance":"sum_r q_r(F)*(1-q_r(F)); z is descriptive and not corrected for 27 comparisons or race/day dependence.",
            "no_purchase":"No ROI, investment, synthetic odds threshold, best subset search, train/test fitting, or production change performed.",
            "market_vs_truth":"Odds-normalized support is market IMPLIED distribution, NOT known true probability.",
            "preexisting_orientation":"2021 each n independently verified at public labeled odds before canonical (a,b,c) use."
        },
        "cells":summary,"examples":EXAMPLES,
        "limitations":["27 fixed disjoint band-signature formations do NOT exhaust all possible unions or overlapping formations.",
                       "Market rank bands are ex-ante descriptive partitions, not a hypothesis fitted to 2021 results.",
                       "All odds are final historical not T-15; observations cannot certify live profitability.",
                       "Multiple comparisons and race-level dependence make large z scores exploratory only.",
                       "No market underpricing or positive expected betting returns are inferred until independent validation and exact stakes.",
                       "No rider-line roles present in this observation; no claim about line-specific subsets."]}
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/"summary.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    (OUT/"README.md").write_text("# Decision-SET observation / 2021 ONLY\n\n"
          "No training, no candidate selection, no bet selection. 2022-2026 archived data not opened in this experiment.\n"
          "See summary.json for the whole 27-cell partition and H1/H2 replication evidence.\n",encoding="utf-8")
    print("FINISHED",json.dumps({"audit":dict(AUDIT),
          "all_cells":[x for x in summary if x["segment"]=="all"]},ensure_ascii=False),flush=True)
if __name__=="__main__":run()
