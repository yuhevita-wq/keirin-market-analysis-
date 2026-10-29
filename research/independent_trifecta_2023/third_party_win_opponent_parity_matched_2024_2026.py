#!/usr/bin/env python3
"""Exploratory observational test of third-party-win configuration, not causation.

A 'weak' (rank 4+) rider in a third line may win *despite* weakness when two
strong opposing LINE LEADERS are competitive. Unlike a result-only anecdote,
every eligible rider/race (wins and non-wins) is enumerated. Compare rival
leader scores within <=3 vs >8 while matching riders on own score deficit,
role, group size, race grade, field size and full score rank. Do not use
odds, market, payout, post-race track dynamics or data from prior predictions.

Crucial: this does NOT measure whether a lead battle occurred, and observed
wins do not prove any 'necessity'. This checks whether the association still
exists after exact/binned common support. No selection on result, no favored
3-2-2 shape. Previously observed 2024/25/26 are exploratory, NOT a new
unseen confirmation. 2026 race-grade is missing in existing reader; it is
included as a separately labeled exploratory sensitivity series, never
misrepresented as exact race-grade-matched.

A race can supply multiple candidate riders; their wins are correlated.
Per-stratum null z below treats candidate exposures as independent and is
diagnostic ONLY; not a cluster-corrected confirmatory significance result.
"""
from __future__ import annotations
import json,math,sys
from pathlib import Path
from collections import defaultdict,Counter

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"research/independent_trifecta_2023"))
sys.path.insert(0,str(ROOT/"research/independent_trifecta_2024"))
import line_dependence_v2 as v2
import calibration_audit as r24
import validate_2025 as r25
import v4_pattern_floor_2026 as r26

OUT=ROOT/"data/audits/third_party_weak_rider_opponent_parity_matched_2024_2026.json"
GRADES=("A3","A1_A2","S_F1","S_grade")
LEAD_GAP_TREAT=3.0
LEAD_GAP_CONTROL=8.0

def binned_gap(x):
    if x<=3:return "0to3"
    if x<=6:return "3to6"
    if x<=10:return "6to10"
    if x<=15:return "10to15"
    if x<=22:return "15to22"
    return "gt22"

def category(x):
    if x<=3:return "0to3"
    if x<=8:return "3to8"
    return "gt8"

def key(e,c,grade,year,mode):
    ranks=sorted(e["scores"],key=lambda x:(-e["scores"][x],int(x)))
    rank=ranks.index(c)+1
    g,pos,size=e["positions"][c]
    own_group=e["groups"][g-1]
    gap_top=max(e["scores"].values())-e["scores"][c]
    gap_line=max(e["scores"][x] for x in own_group)-e["scores"][c]
    role=v2.rlabel(e,c)
    if mode=="basic":
        return (year,grade,role,size,len(e["scores"]),rank,binned_gap(gap_top))
    if mode=="plus":
        return (year,grade,role,size,len(e["scores"]),rank,
                binned_gap(gap_top),binned_gap(gap_line))
    if mode=="strict":
        return (year,grade,role,size,len(e["scores"]),rank,
                binned_gap(gap_top),binned_gap(gap_line),
                category(max(e["scores"].values())-
                         max(e["scores"][x] for x in own_group)))
    raise ValueError(mode)

def qualify_rider(e,c):
    """Third PARTY line is weaker than both strongest opposing line leaders.

    The 'weakness' condition is rider full-field score rank 4+, not an
    inferred result. Does not assume which rider will actually win.
    """
    scores=e["scores"]
    rank=sorted(scores,key=lambda x:(-scores[x],int(x))).index(c)+1
    if rank<4:return None
    g,_,_=e["positions"][c]
    if len(e["groups"])<3:return None
    leaders=[(idx+1,group[0],scores[group[0]]) for idx,group in enumerate(e["groups"])]
    others=sorted([(h,k,s) for h,k,s in leaders if h!=g],
                  key=lambda x:(-x[2],int(x[1])))
    if len(others)<2:return None
    best,second=others[:2]
    my_leader=scores[e["groups"][g-1][0]]
    # Third line must be weaker by its initial leader score. This tests the
    # user's third-party-floats-up narrative, not random weak riders.
    if my_leader>second[2]:return None
    gap=best[2]-second[2]
    if gap<=LEAD_GAP_TREAT:
        exposure="near"
    elif gap>LEAD_GAP_CONTROL:
        exposure="far"
    else:
        exposure="middle"
    return exposure,{
        "rank":rank,
        "own_leader_score":my_leader,
        "enemy_leader_best":best[2],
        "enemy_leader_second":second[2],
        "rival_leader_score_gap":gap,
    }

def group_view(cell):
    n,w=cell
    return {"eligible_rider_exposures":n,"actual_first_places":w,
            "first_rate_pct":round(100*w/n,6) if n else None}

def strata_compare(strata,min_per_arm=1):
    """Direct standardization on *matched* common-support strata.

    Weight each stratum equally proportional to min(treatment,control)
    exposures, and report coverage/exclusions. This is not equivalent
    to effect of actual track tactics. Approximate z ignores clustering.
    """
    eligible=[]
    win_t=win_c=n_t=n_c=0
    sumweight=weighted_risk_diff=weighted_risk_t=weighted_risk_c=0.
    var=0.
    num=den=0.
    for k,ab in strata.items():
        t,c=ab["near"],ab["far"]
        nt,yt=t;nc,yc=c
        if nt<min_per_arm or nc<min_per_arm:continue
        w=min(nt,nc)
        pt=yt/nt;pc=yc/nc
        eligible.append((k,nt,yt,nc,yc,w,pt,pc))
        n_t+=nt;win_t+=yt;n_c+=nc;win_c+=yc
        sumweight+=w
        weighted_risk_diff+=w*(pt-pc)
        weighted_risk_t+=w*pt
        weighted_risk_c+=w*pc
        pooled=(yt+yc)/(nt+nc)
        var+=w*w*pooled*(1-pooled)*(1/nt+1/nc)
        num+=yt*(nc-yc)/(nt+nc)
        den+=(nt-yt)*yc/(nt+nc)
    if sumweight==0:return {"matched_strata":0,"matched_n_t":0,"matched_n_c":0}
    z=weighted_risk_diff/math.sqrt(var) if var>0 else None
    return {
       "matched_strata":len(eligible),
       "matched_n_t":n_t,"matched_n_c":n_c,
       "matched_first_wins_near":win_t,"matched_first_wins_far":win_c,
       "matched_naive_near_rate_pct":round(100*win_t/n_t,5),
       "matched_naive_far_rate_pct":round(100*win_c/n_c,5),
       "standardized_near_rate_pct":round(100*weighted_risk_t/sumweight,6),
       "standardized_far_rate_pct":round(100*weighted_risk_c/sumweight,6),
       "standardized_difference_percentage_points":round(
           100*weighted_risk_diff/sumweight,6),
       "standardized_ratio":round(weighted_risk_t/weighted_risk_c,5)
          if weighted_risk_c else None,
       "CMH_common_odds_ratio":round(num/den,5) if den else None,
       "naive_independent_candidate_z_NOT_CLUSTER_CORRECTED":round(z,4) if z is not None else None,
       "weight":"min(treated candidates,control candidates) within each common-support stratum",
       "matched_stratum_samples_top_12":[
           {"stratum":list(k),"near_n":nt,"near_w":yt,
            "far_n":nc,"far_w":yc,
            "near_pct":round(100*pt,4),"far_pct":round(100*pc,4)}
            for k,nt,yt,nc,yc,w,pt,pc in sorted(eligible,key=lambda t:-t[5])[:12]
       ],
    }

def run():
    rows=[];by_year={};meta={}
    for year,fn in (("2024",r24.ingest_2024),("2025",r25.read_2025),
                    ("2026",r26.ingest_2026)):
        events,audit=fn()
        expect={"2024":25348,"2025":24635,"2026":11797}[year]
        if len(events)!=expect:raise AssertionError((year,len(events),expect))
        reasons=Counter()
        selected=0
        for rid,e in events.items():
            grade=e.get("race_class")
            if year!="2026" and grade not in GRADES:
                reasons["unusable_race_class"]+=1
                continue
            if year=="2026":grade="UNKNOWN_2026_NOT_GRADE_MATCHED"
            for c in e["scores"]:
                v=qualify_rider(e,c)
                if v is None:continue
                exposure,info=v
                first=int(e["order"][0]==c)
                rows.append((year,e,c,grade,exposure,first,rid,info))
                selected+=1
                reasons["eligible_"+exposure]+=1
        by_year[year]={"races":len(events),"eligible_rider_exposures":selected,
                       "eligibility":dict(reasons)}
        meta[year]=audit
        print("THIRD_PARTY_CANDIDATES_EXTRACTED",year,len(events),selected,
              dict(reasons),flush=True)
    # We use all members of the observed card, including actual losers;
    # NO result-dependent conditioning in exposure eligibility.
    axes=("all_roles","leader_only","non_leader",
          "first_role:leader","first_role:second","first_role:third",
          "grade:A3","grade:A1_A2","grade:S_F1","grade:S_grade")
    doc={}
    for axis in axes:
        subset=[]
        for row in rows:
            year,e,c,grade,exp,win,rid,info=row
            role=v2.rlabel(e,c)
            selected=(
                axis=="all_roles" or
                axis=="leader_only" and role=="leader" or
                axis=="non_leader" and role!="leader" or
                axis.startswith("first_role:") and role==axis.split(":")[1] or
                axis.startswith("grade:") and grade==axis.split(":")[1]
            )
            if selected:subset.append(row)
        if not subset:continue
        data={}
        for period in ("2024","2025","2026","2024_2025","2024_2025_2026"):
            drows=[row for row in subset if (
                row[0]==period or
                period=="2024_2025" and row[0] in ("2024","2025") or
                period=="2024_2025_2026"
            )]
            if not drows:continue
            raw=defaultdict(lambda:[0,0])
            for yr,e,c,grade,exp,win,rid,info in drows:
                z=raw[exp];z[0]+=1;z[1]+=win
            m={}
            for mode in ("basic","plus","strict"):
                strata=defaultdict(lambda:{"near":[0,0],"far":[0,0]})
                for yr,e,c,grade,exp,win,rid,info in drows:
                    if exp not in ("near","far"):continue
                    forkey=key(e,c,grade,yr,mode)
                    z=strata[forkey][exp]
                    z[0]+=1;z[1]+=win
                m[mode]=strata_compare(strata,min_per_arm=1)
            data[period]={
                "raw":{k:group_view(raw[k]) for k in ("near","middle","far")},
                "match_models":m,
                "eligible_rider_exposures":len(drows)}
        doc[axis]=data
    result={
      "state":"THIRD_PARTY_WIN_OPPONENT_PARITY_MATCHED_OBSERVATION_COMPLETE",
      "question":"Do third-party lower-ranked riders win more frequently when two stronger rival line leaders have close pre-race scores, after matching on own ability/role/group size/race class?",
      "data_years_2024_2026_all_previously_observed":True,
      "all_race_line_shapes_evaluated":True,
      "score_rank_minimum":4,
      "exposure":"Difference between two top rival line leaders' PRE-RACE scores, excluding candidate's own line",
      "near_leaders_score_gap_le_3":True,
      "far_leaders_score_gap_gt_8":True,
      "third_party_condition":"own line leader has score <= second of two highest rival line leader scores, at least 3 total lines",
      "weak_condition":"rider full-field score rank >=4",
      "matched_dimensions":"year, source race grade, rider role, original group size, field size, exact rider score rank, gap to full-field highest score (bins <=3,6,10,15,22,over22); plus also gap to best own-line score",
      "matched_2026_grade_covered":False,
      "year2026_grade_handling":"Separate UNKNOWN grade stratum; cannot be pooled as matched with known 2024/2025 race grades",
      "race_grade_classes":list(GRADES),
      "do_not_interpret_as_causal_effect":True,
      "does_not_observe_realized_race_tactics":True,
      "no_odds_used":True,
      "normalization":"Matched common-support strata only, min(n_near,n_far) weighting; rare strata may not overlap",
      "statistical_uncertainty":"Naive z is illustrative only, candidate wins within a race are correlated, no cluster-robust significance claimed",
      "years":by_year,"source_coverage":meta,
      "comparisons":doc,
      "conclusions_require_new_and_independent_observations":True,
      "limitations":[
         "2024-2026 results were already inspected: this is an exploratory observational retrospective comparison.",
         "Third-party leader competitiveness does not prove that a lead battle, traction exhaustion, hesitation, or tactical conflict occurred.",
         "Matching on score gaps/ranks/role/grade does not guarantee causal exchangeability; residual race-level confounding remains.",
         "Some fully matched cells lack near/far overlap; coverage is reported for each axis and year.",
         "2026 existing parser lacks race_class; its UNKNOWN grade comparisons are separate exploratory sensitivity results.",
         "No conclusion of inevitability or advantage versus betting market can be made.",
         "Race grades A1_A2/A3/S_grade/S_F1 are source-directory classifications, not certified grades for all raw source files.",
         "Published formation might be provisional rather than final on-track line; no observation of race action.",
      ]
    }
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("THIRD_PARTY_OPPONENT_PARITY_MATCHED_RESEARCH_COMPLETE",
          json.dumps({
              "cohorts":by_year,
              "leader_2024_2025":doc["leader_only"]["2024_2025"],
              "all_roles_2024_2025":doc["all_roles"]["2024_2025"],
              "leader_2026":doc["leader_only"]["2026"]
            },ensure_ascii=False),flush=True)

if __name__=="__main__":run()
