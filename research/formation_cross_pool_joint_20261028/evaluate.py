"""Research: price disagreement ACROSS POOLS predicts returns of entire 3rentan formations?

New external idea:
Given final 3rentan joint market q_t and separately priced 3renpuku
and 2shatan pools, build two internally coherent joint distributions:
  P3(w) = q3p(unordered(w)) * q_t(w) / sum_{perms(u)}q_t
  P2(w) = q2t(w[0:2]) * q_t(w) / sum_k q_t(a,b,k)
P3 and P2 both sum to 1, retaining only within-group ordering from q_t.
Use P3/P2 for pricing *whole Cartesian formations*, never selecting
a single ticket. Evaluate true equal-100-yen-ticket payout by exact result.
No model weights or purchase rules are updated.
"""
from __future__ import annotations
import sys,os,math,json,itertools
import numpy as np
from pathlib import Path
from collections import Counter,defaultdict
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from research.snowberg_observation_corrected_20261027 import observe as corrected
from research.picking_winners_true_formations_20261028.observe import create_base,tickets
OUT=ROOT/"research/formation_cross_pool_joint_20261028"
YEAR_RANGE=tuple(str(i) for i in range(2021,2027))
MODES=("3p","2t","half")
POINT_CAPS=(8,12,16,24,32)
SYN_MIN=(3.0,3.5,5.0,8.0)
EXPECTED_MIN=(0.8,0.9,1.0,1.05,1.1,1.2)
METHODS=("base_only","change_1","change_2","change_3","change_12","change_13","change_23","change_123")
AUDIT=Counter()
AGGS=defaultdict(lambda:np.zeros(10,dtype=np.float64))
DIAGNOSTICS=defaultdict(lambda:np.zeros(9,dtype=np.float64))
EXAMPLES=[]
SMALL=1e-30

def log(row,pts,syn,win,odds,pr3,pr2,mark,qbase):
    z=AGGS[row]
    z[0]+=1
    z[1]+=pts
    z[2]+=100*pts
    z[3]+=100*odds*win
    z[4]+=win
    z[5]+=syn
    z[6]+=pr3
    z[7]+=pr2
    z[8]+=mark
    z[9]+=qbase

def observe(rid,day,books,finish,rtype,source):
    AUDIT["callbacks"]+=1
    if source!="sealed_four_markets_orientation_verified":
        raise RuntimeError("Must use independently verified ordered-key correction")
    t=books.get("3rentan",{})
    p=books.get("3renpuku",{})
    d=books.get("2shatan",{})
    cars=tuple(sorted({x for w in t for x in w}))
    n=len(cars)
    if len(t)!=n*(n-1)*(n-2) or len(p)!=math.comb(n,3) or len(d)!=n*(n-1):
        AUDIT["bad_complete_market"]+=1;return
    finish=tuple(finish)
    if finish not in t: AUDIT["missing_actual_finish"]+=1;return
    if any(float(o)<=1 or not math.isfinite(float(o)) for o in itertools.chain(t.values(),p.values(),d.values())):
        AUDIT["bad_odds"]+=1;return
    zt=sum(1/float(o) for o in t.values())
    zp=sum(1/float(o) for o in p.values())
    zd=sum(1/float(o) for o in d.values())
    qt={w:1/float(o)/zt for w,o in t.items()}
    qp={w:1/float(o)/zp for w,o in p.items()}
    qd={w:1/float(o)/zd for w,o in d.items()}
    t_unordered=defaultdict(float);t_ordered_pair=defaultdict(float)
    for w,q in qt.items():
        t_unordered[tuple(sorted(w))]+=q
        t_ordered_pair[w[:2]]+=q
    # These are three coherent probability distributions on the same 3-way state space.
    q3p={w:qt[w]*qp[tuple(sorted(w))]/t_unordered[tuple(sorted(w))] for w in t}
    q2t={w:qt[w]*qd[w[:2]]/t_ordered_pair[w[:2]] for w in t}
    if abs(sum(q3p.values())-1)>1e-8 or abs(sum(q2t.values())-1)>1e-8:
        raise AssertionError("Cross-pool projection must be normalized")
    base,ranks,ks=create_base(cars,t)
    base_set=tickets(base)
    forms=[("base_only",base_set,base,None)]
    for mask in ("1","2","3","12","13","23","123"):
        alt=list(base);valid=True
        for ch in mask:
            pos=int(ch)-1
            shift=tuple(cars[int(i)] for i in np.roll(ranks[pos],-1)[:ks[pos]])
            if set(shift)==set(base[pos]):valid=False;break
            alt[pos]=shift
        if not valid:continue
        alt=tuple(alt)
        second=tickets(alt)
        if len(second)<4 or second==base_set:continue
        forms.append(("change_"+mask,base_set|second,base,alt))
    candidates=[]
    for label,F,first,second in forms:
        npoints=len(F)
        if npoints>max(POINT_CAPS) or npoints<4:continue
        inv=sum(1/float(t[w]) for w in F)
        syn=1/inv
        mean_odds=float(sum(q3p[w]*float(t[w]) for w in F)/npoints)
        ex_odds=float(sum(q2t[w]*float(t[w]) for w in F)/npoints)
        # Prediction of real equal-stake ROI, not probability of hitting *one* ticket.
        scores={"3p":mean_odds,"2t":ex_odds,"half":.5*(mean_odds+ex_odds)}
        candidates.append((label,F,npoints,syn,scores,first,second))
    if not candidates:
        AUDIT["no_form_32_points"]+=1;return
    year=day[:4]
    won=float(t[finish])
    all_key=(year,"all_candidates")
    DIAGNOSTICS[all_key][0]+=1
    DIAGNOSTICS[all_key][1]+=len(candidates)
    for cap in POINT_CAPS:
        for min_syn in SYN_MIN:
            eligible=[x for x in candidates if x[2]<=cap and x[3]>=min_syn]
            if not eligible:continue
            for mode in MODES:
                # deterministic tie break; all values determined before observed finish.
                best=max(eligible,key=lambda x:(x[4][mode],-x[2],-METHODS.index(x[0])))
                lab,F,pts,syn,est,first,second=best
                hit=int(finish in F)
                ratio=est[mode]/(1/zt)
                for e_min in EXPECTED_MIN:
                    if est[mode]<e_min:continue
                    log((year,mode,cap,min_syn,e_min),pts,syn,hit,won,est["3p"],est["2t"],ratio,1/zt)
                for e_min in (0.8,1.0,1.1):
                    if est[mode]<e_min:continue
                    b=DIAGNOSTICS[(year,mode,cap,min_syn,e_min)]
                    b[0]+=1;b[1]+=hit;b[2]+=pts;b[3]+=100*pts
                    b[4]+=100*won*hit;b[5]+=est[mode];b[6]+=syn
                    b[7]+=int(lab!="base_only");b[8]+=ratio
                if year in ("2025","2026") and cap==16 and min_syn==3.5 and est[mode]>=1 and len(EXAMPLES)<90:
                    EXAMPLES.append({
                        "race_id":rid,"year":year,"method":mode,"candidate":lab,
                        "first":[list(a) for a in first],
                        "second":[list(a) for a in second] if second is not None else None,
                        "points":pts,"synthetic_odds":round(syn,3),
                        "cross_pool_pred_equal_stake_roi":round(est[mode],3),
                        "baseline_pred_equal_stake_roi":round(1/zt,3),
                        "actual_finish":list(finish),"hit":bool(hit),
                        "stake":100*pts,"payout":100*won*hit
                    })
    AUDIT["usable_races"]+=1
    AUDIT["year_"+year]+=1

def clean(v):
    k,v=tuple(v[0]),v[1]
    return {"race_count":int(v[0]),"avg_points":v[1]/v[0] if v[0] else None,
            "stake_yen":round(float(v[2])),"payout_yen":round(float(v[3])),
            "profit_yen":round(float(v[3]-v[2])),
            "ROI":v[3]/v[2] if v[2] else None,
            "hit_count":int(v[4]),"hit_rate":v[4]/v[0] if v[0] else None,
            "mean_synthetic_odds":v[5]/v[0] if v[0] else None,
            "mean_trio_model_predicted_roi":v[6]/v[0] if v[0] else None,
            "mean_exacta_model_predicted_roi":v[7]/v[0] if v[0] else None,
            "mean_relative_model_disagreement":v[8]/v[0] if v[0] else None,
            "mean_trifecta_only_predicted_roi":v[9]/v[0] if v[0] else None}

def results(v):
    x=np.zeros(10)
    for year in v:
        x+=AGGS.get(year,np.zeros(10))
    return clean((("combined",),x))

def rank(train,test):
    # minimum number of purchase opportunities per year; multiple testing is acknowledged.
    m=[]
    for mode in MODES:
      for cap in POINT_CAPS:
       for syn in SYN_MIN:
        for ep in EXPECTED_MIN:
            rows=[AGGS.get((y,mode,cap,syn,ep),np.zeros(10)) for y in train]
            if sum(r[0] for r in rows)<200 or any(r[0]<25 for r in rows):
                continue
            stat=results([(y,mode,cap,syn,ep) for y in train])
            m.append((stat["ROI"],mode,cap,syn,ep,stat))
    m.sort(key=lambda x:(-x[0],x[2],x[3],x[4],MODES.index(x[1])))
    if not m:return {"train":train,"test_year":test,"status":"INSUFFICIENT_EXPOSURE"}
    b=m[0]
    b_test=results([(test,b[1],b[2],b[3],b[4])])
    return {
      "training_years":train,"test_year":test,"candidate_count":len(m),
      "best_rule":{"mode":b[1],"point_cap":b[2],"min_synthetic_odds":b[3],"min_predicted_equal_stake_roi":b[4]},
      "train":b[5],"blind_year_test_even_if_not_bought":b_test,
      "buy_decision":"BUY" if b[0]>1 else "SKIP",
      "actual_blind_year_result":b_test if b[0]>1 else {"stake_yen":0,"payout_yen":0,"profit_yen":0},
      "top5":[{"roi":round(x[0],5),"mode":x[1],"point_cap":x[2],"min_syn":x[3],"min_pred":x[4],"races":x[5]["race_count"]} for x in m[:5]]
    }

def main():
    if not os.environ.get(corrected.old.bridge.SECRET_ENV):
        raise RuntimeError("The verified sealed archive key is required")
    corrected.base_observe=observe
    corrected.old.save=lambda status:print("CHECKPOINT",status,dict(AUDIT),flush=True)
    corrected.old.sealed()
    if AUDIT["usable_races"]<120000:raise RuntimeError("Unacceptable full-universe truncation")
    rules=[]
    for (year,mode,cap,syn,ep),a in sorted(AGGS.items()):
        rules.append({"year":year,"model":mode,"point_cap":cap,"min_synthetic_odds":syn,"min_predicted_roi":ep,**clean(((year,mode,cap,syn,ep),a))})
    chronological=[rank(YEAR_RANGE[:4],"2025"),rank(YEAR_RANGE[:5],"2026")]
    report={
       "status":"CROSS_POOL_FORMATION_PRICE_DISAGREEMENT_EXPERIMENT_COMPLETE",
       "source_commit":os.environ.get("GITHUB_SHA","unknown"),
       "generated_utc":datetime.now(timezone.utc).isoformat(),
       "audit":dict(AUDIT),
       "method":{
         "external_idea":"Coherent joint probabilities, cross-market marginals, and anomalous collection prices; inspired by external racing price-model research and combinatorial market constraints.",
         "p3p_joint":"q3p[sorted(a,b,c)] * q3t[a,b,c]/sum_{permutations of sorted(a,b,c)}q3t[perm]",
         "p2t_joint":"q2t[a,b] * q3t[a,b,c]/sum_{k!=a,b}q3t[a,b,k]",
         "average":"50:50 mixture of coherent p3p_joint and p2t_joint",
         "selection":"Given formation candidates, maximize expected equal-stake 3連単 basket ROI from indicated external pool projection; thresholds applied before outcome observation.",
         "ticket_space":"Actual A×B×C Cartesian formation per candidate; union of two proper formations, remove duplicate exact bets.",
         "comparison_family":"One existing base and seven hold/shift modifications; does NOT explore all possible formations or prove global optimum.",
         "selection_parameters":"Grid min estimated equal-stake ROI, min synthetic odds, max points. 2021-24 training->2025 test, 2021-25 training->2026H1 test.",
         "risk":"No-bet if no training grid surpasses break-even after minimum exposure (200 pooled, >=25 each year).",
         "financial_returns":"Exactly 100 yen per unique point; actual result must exactly be among paid bets, no credit for partial wins."},
       "chronological":chronological,
       "all_rule_year_cells":rules,
       "sample_formations":EXAMPLES,
       "limitations":[
         "Historical FINAL odds only. No T-15 snapshots; cannot claim pre-close executability.",
         "No race-level alignment of quotes from the independently pooled markets beyond archived final data.",
         "The conditional within-triple and within-top-two uses trifecta market, so external projections are not independent forecasts.",
         "Market priors are NOT calibrated true probabilities; price discordance is a screening indicator, not proof of market error.",
         "Uses only eight existing candidate shapes (usually large), even with 8–32 point caps. Arbitrary shape optimization remains open.",
         "Selection over many rules risks multiple comparisons; held-out years previously used in other projects, therefore not virgin data.",
         "Any found positive ROI needs independent future period and proper wager-sized settlement.",
         "Original research code, models, running purchase rules and websites unchanged."
       ]}
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/"compact.json").write_text(json.dumps({k:v for k,v in report.items() if k not in ("all_rule_year_cells","sample_formations")},ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    (OUT/"summary.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("COMPLETE",json.dumps({"audit":dict(AUDIT),"chronological":chronological},ensure_ascii=False),flush=True)
if __name__=="__main__":main()
