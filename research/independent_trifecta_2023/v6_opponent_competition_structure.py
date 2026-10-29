#!/usr/bin/env python3
"""V6 competition-relations research (no odds, no score-only uplift claims).

Question: Which relations among *all opponents* allow low-ranked riders to win?
The actual race dynamics are not observed. Use PRE-RACE competition proxies:
number of strong rivals on opposing lines, inter-line strength parity, role of
each rider, own/other line top strength, score differences and pairwise
matchup interactions. Distinguish learned score proxy from real causality.

Frozen predecessor: original V2 line+rider (2023 Jan-Oct), V3 pattern gate
lambda=.15, V5 rider interaction head (2023 Jan-Oct). Both *frozen in
algorithm*, retrained on the same original period. NO new data leakage.
V6 learns two residual softmax heads: (a) choice of a LINE in opposition to
others; (b) choice of a RIDER inside that line in context of rival lines.
This allows opponents to influence both stages, not just weaken scores.

Train V6 only 2023 Jan-Oct; select residual scales 0,.25,.50,1 using
PREVIOUSLY INSPECTED 2023 Nov-Dec development. 2024,25,26 historical
replays are not genuinely unknown holdouts. Cannot promote from them.
All data without market odds, and no special 3-2-2 priorities.
Every 3ren-tan ticket summed and checked for normalization.
"""
from __future__ import annotations

import json,math,random,sys
from pathlib import Path
from collections import defaultdict,Counter

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"research/independent_trifecta_2023"))
sys.path.insert(0,str(ROOT/"research/independent_trifecta_2024"))
import observe as original
import line_dependence_v2 as v2
import aab_guarded_v3 as v3
import v5_rider_score_role_size_interactions as v5
import calibration_audit as src24
import validate_2025 as src25
import v4_pattern_floor_2026 as src26

OUT=ROOT/"data/audits/trifecta_v6_opponent_competition_structure_2023_2026.json"
SCALES=(0.,.25,.5,1.)
SEED=20231031;EPOCHS=4;LR=.009;SHRINK=.977
YEARS={"2024":25348,"2025":24635,"2026":11797}
PATTERNS=("AAA","AAB","ABA","ABB","ABC")
TOL=1e-8

def cap(x,a=-2.5,b=2.5):
    return max(a,min(b,x))

def opponent_environment(e,prefix):
    rem=set(e["scores"])-set(prefix)
    groups={g:members for g,members in v2.line_members(e,prefix).items()}
    leader_evidence={
        g: next((m for m in members if e["positions"][m][1]==1),None)
        for g,members in groups.items()
    }
    # The whole line's best available cyclist is a measurable strategic
    # proxy, not a known attack/future race move.
    tops={g:max(e["scores"][x] for x in members)
          for g,members in groups.items()}
    linescores=sorted(tops.values(),reverse=True)
    parity=(linescores[0]-linescores[1]) if len(linescores)>1 else 100.
    conflict=sum(abs(linescores[0]-s)<=5 for s in linescores[1:])
    return rem,groups,tops,leader_evidence,parity,conflict

def linefeatures(e,prefix,g,env):
    _,groups,tops,leaders,parity,conflict=env
    members=groups[g]
    rivals=[(h,tops[h]) for h in tops if h!=g]
    own=tops[g]
    if not rivals:
        return {}  # a sole remaining line cannot win/lose at line-choice
    otherbest=max(z for h,z in rivals)
    otherstrongest_leader=max(
        (e["scores"][leaders[h]] for h,z in rivals if leaders[h] is not None),
        default=otherbest
    )
    close=sum(z>=own-5 for h,z in rivals)
    stronger=sum(z>own+3 for h,z in rivals)
    relative=cap((own-otherbest)/12.)
    dist=abs(own-otherbest)
    size=min(4,len(e["groups"][g-1]))
    prefix_stage=len(prefix)+1
    lf={
        f"s{prefix_stage}|relative_best":relative,
        f"s{prefix_stage}|relative_best_squared":relative*abs(relative),
        f"s{prefix_stage}|rival_count_close_5":cap(close/3.,0,1),
        f"s{prefix_stage}|rival_count_stronger_3":cap(stronger/3.,0,1),
        f"s{prefix_stage}|parity_x_relative":(1. if parity<=5 else 0.)*relative,
        f"s{prefix_stage}|high_conflict_x_relative":(1. if conflict>=2 else 0.)*relative,
        f"s{prefix_stage}|leader_vs_enemy_leader":
            cap((e["scores"][leaders[g]]-otherstrongest_leader)/12.)
            if leaders[g] is not None else 0.,
        f"s{prefix_stage}|one_active_line_leader":
            int(leaders[g] is not None),
        f"s{prefix_stage}|relative_x_line_size_{size}":relative,
    }
    return lf

def riderfeatures(e,prefix,c,env):
    _,groups,tops,leaders,parity,conflict=env
    g,pos,size=e["positions"][c]
    rem=groups[g]
    if len(rem)<2:return {}
    rivals=[e["scores"][x] for h,mem in groups.items() if h!=g for x in mem]
    if not rivals:
        # Remaining within-line selection still has intra-line interaction.
        rivals=[e["scores"][x] for x in rem if x!=c]
    score=e["scores"][c]
    enemybest=max(rivals)
    enemymean=sum(rivals)/len(rivals)
    insidebest=max(e["scores"][x] for x in rem)
    gap=cap((insidebest-score)/12.,0,2.5)
    role=v2.rlabel(e,c)
    s=len(prefix)+1
    key=f"s{s}|role={role}"
    pressure=cap((enemybest-score)/12.)
    parityweight=1. if parity<=5 else .0
    closecount=sum(abs(x-score)<=5 for x in rivals)
    stronger_count=sum(x>=score+3 for x in rivals)
    # Feature groups discriminate real "who is competing against whom"
    # rather than assigning unstructured probability to any weak rider.
    return {
        key+"|relative_to_enemy_best":pressure,
        key+"|relative_to_enemy_mean":cap((enemymean-score)/12.),
        key+"|inside_gap":gap,
        key+"|enemy_pressure_x_inside_gap":pressure*gap,
        key+"|line_parity_x_inside_gap":parityweight*gap,
        key+"|high_conflict_x_gap":int(conflict>=2)*gap,
        key+"|enemy_near":cap(closecount/4.,0,1),
        key+"|enemy_stronger":cap(stronger_count/4.,0,1),
        key+"|opponent_leader_alive":int(
            any(leaders[h] in groups[h] for h in groups if h!=g and leaders[h] is not None)
        ),
        key+"|opponent_pressure_x_role_size_"+str(min(size,4)):pressure,
    }

def softmax_over_prior(prior,feats,weights,alpha):
    if alpha==0 or len(prior)==1:
        return dict(prior)
    vals={c:math.log(max(1e-300,p))+alpha*sum(
        weights.get(f,0.)*v for f,v in feats[c].items())
        for c,p in prior.items()}
    return v2.softmax(vals)

def choice(e,prefix,wg,wr,delta5,wline,wrider,alpha,baseline="V6"):
    """Two-stage pre-race competitive softmax: same formation, different relations."""
    gp,cp=v2.struct_components(e,prefix,wg,wr)
    groups=v2.line_members(e,prefix)
    env=opponent_environment(e,prefix)
    if baseline=="V3":
        gp2=gp
        rider={c:cp[c] for g,members in groups.items() for c in members}
    else:
        gp2=gp if baseline=="V5" else softmax_over_prior(
            gp,{g:linefeatures(e,prefix,g,env) for g in groups},wline,alpha
        )
        rider={}
        for g,members in groups.items():
            v5r=v5.rider_conditional(e,prefix,g,members,cp,delta5,1.)
            rp=v5r if baseline=="V5" else softmax_over_prior(
                v5r,{c:riderfeatures(e,prefix,c,env) for c in members},wrider,alpha
            )
            rider.update(rp)
    out={c:gp2[g]*rider[c] for g,members in groups.items() for c in members}
    if abs(sum(out.values())-1)>TOL:raise AssertionError("Choice mass not one")
    return out

def train(e_train,wg,wr,delta5):
    wl={};wc={}
    cases=list(e_train.values())
    rng=random.Random(SEED)
    for epoch in range(EPOCHS):
        rng.shuffle(cases)
        lr=LR/math.sqrt(epoch+1)
        total_line=0.;total_rider=0.;nl=0;nc=0
        for e in cases:
            for st in (1,2,3):
                prefix=e["order"][:st-1]
                winner=e["order"][st-1]
                targetg=e["positions"][winner][0]
                env=opponent_environment(e,prefix)
                gp,cp=v2.struct_components(e,prefix,wg,wr)
                groups=v2.line_members(e,prefix)
                if len(groups)>1:
                    fl={g:linefeatures(e,prefix,g,env) for g in groups}
                    pred=softmax_over_prior(gp,fl,wl,1.)
                    grad=defaultdict(float)
                    for g in groups:
                        z=(int(g==targetg)-pred[g])
                        for k,v in fl[g].items():grad[k]+=z*v
                    for k,val in grad.items():wl[k]=wl.get(k,0.)+lr*val
                    total_line-=math.log(max(1e-15,pred[targetg]));nl+=1
                members=groups[targetg]
                if len(members)>1:
                    base=v5.rider_conditional(e,prefix,targetg,members,cp,delta5,1.)
                    fc={c:riderfeatures(e,prefix,c,env) for c in members}
                    pred=softmax_over_prior(base,fc,wc,1.)
                    grad=defaultdict(float)
                    for c in members:
                        z=int(c==winner)-pred[c]
                        for k,v in fc[c].items():grad[k]+=z*v
                    for k,val in grad.items():wc[k]=wc.get(k,0.)+lr*val
                    total_rider-=math.log(max(1e-15,pred[winner]));nc+=1
        for w in (wl,wc):
            for key in w:w[key]*=SHRINK
        print("V6_RIVALRY_TRAIN_EPOCH",epoch+1,"n_line",nl,"n_rider",nc,
              "NLL_line",round(total_line/nl,7),"NLL_rider",round(total_rider/nc,7),
              "coeffs",(len(wl),len(wc)),flush=True)
    return wl,wc

def dists(e,wg,wr,delta5,wl,wc,alpha):
    """Three full ordered-ticket probability distributions. No actual results accessed."""
    models=("V3","V5","V6")
    cars=sorted(e["scores"],key=int)
    candidates={}
    prefixes=[()]
    prefixes.extend((a,) for a in cars)
    prefixes.extend((a,b) for a in cars for b in cars if a!=b)
    for pre in prefixes:
        candidates[pre]={name:choice(e,pre,wg,wr,delta5,wl,wc,alpha,name)
                         for name in models}
    raw={name:{} for name in models}
    for a in cars:
        for b in cars:
            if a==b:continue
            for c in cars:
                if c==a or c==b:continue
                for name in models:
                    ch=candidates
                    raw[name][a,b,c]=ch[()][name][a]*ch[(a,)][name][b]*ch[(a,b)][name][c]
    gate=v3.predict_non_aaa(e,dists.pw)
    res={}
    for name in models:
        dd=raw[name]
        if abs(sum(dd.values())-1)>TOL:
            raise AssertionError(("Incomplete raw competition mass",name,sum(dd.values())))
        variant,pat=v3.variants(e,dd,gate)
        result=variant[.15][0]
        if abs(sum(result.values())-1)>TOL:raise AssertionError("Bad gated mass")
        res[name]=result
    return res,pat,candidates

def bucket(p):
    if p<.0001:return "<0.01%"
    if p<.0005:return "0.01-0.05%"
    if p<.001:return "0.05-0.1%"
    if p<.005:return "0.1-0.5%"
    if p<.01:return "0.5-1%"
    if p<.02:return "1-2%"
    if p<.05:return "2-5%"
    return "5%+"

def geometry(e,car,cs):
    """Rival-strength parity and role are PROXIES, not observed racing strategy."""
    leaders=[(g, next((x for x in line if e["positions"][x][1]==1),line[0]))
             for g,line in enumerate(e["groups"],1)]
    leader_scores=sorted((e["scores"][c] for _,c in leaders),reverse=True)
    clash=(leader_scores[0]-leader_scores[1]) if len(leader_scores)>1 else 100.
    bucket_clash=("rival_leaders_within_3" if clash<=3 else
                  "rival_leaders_within_8" if clash<=8 else "rival_leaders_far")
    allranks=sorted(e["scores"],key=lambda c:(-e["scores"][c],int(c)))
    rank=allranks.index(car)+1
    role=v2.rlabel(e,car)
    return rank,role,bucket_clash

class Tracker:
    def __init__(self):
        self.n=0;self.nll=0.;self.brier=0.
        self.daily=defaultdict(lambda:[0.,0.,0])
        self.tail=defaultdict(lambda:[0.,0,0])
        self.own=defaultdict(lambda:[0.,0,0])
        self.pattern=defaultdict(lambda:[0.,0])
        self.winner_weak=defaultdict(lambda:[0.,0,0])
        self.winner_by_role=defaultdict(lambda:[0.,0,0])
        self.shapes=Counter()
    def add(self,e,dist,pat,base,candidates,name):
        self.n+=1;self.shapes[v2.line_shape(e)]+=1
        p=dist[e["order"]]
        loss=-math.log(p)
        brier=1-2*p+sum(x*x for x in dist.values())
        self.nll+=loss;self.brier+=brier
        day=self.daily[e["date"]];day[0]+=loss;day[1]+=brier;day[2]+=1
        for t,z in dist.items():
            hit=int(t==e["order"])
            fixed=bucket(base[t]);own=bucket(z)
            self.tail[fixed][0]+=z;self.tail[fixed][1]+=hit;self.tail[fixed][2]+=1
            self.own[own][0]+=z;self.own[own][1]+=hit;self.own[own][2]+=1
            zz=self.pattern[pat[t]];zz[0]+=z;zz[1]+=hit
        # One pre-race choice distribution over all possible 1st-place riders.
        # This is NOT conditioned on a known winning line.
        for c,q in candidates[()][name].items():
            rank,role,clash=geometry(e,c,candidates)
            actual=int(c==e["order"][0])
            group=("rank4plus" if rank>=4 else "rank1to3")
            key=group+"|"+role+"|"+clash
            u=self.winner_weak[key];u[0]+=q;u[1]+=actual;u[2]+=1
            ur=self.winner_by_role[role+"|"+clash];ur[0]+=q;ur[1]+=actual;ur[2]+=1
    def pack(self):
        return {
           "races":self.n,"NLL":round(self.nll/self.n,9),
           "Brier":round(self.brier/self.n,9),
           "shapes_tested":len(self.shapes),
           "shape_race_counts":dict(sorted(self.shapes.items())),
           "V3_fixed_ticket_cohort":{
               k:{"predicted":round(v[0],6),"actual":v[1],"tickets":v[2]}
               for k,v in sorted(self.tail.items())},
           "own_probability_band":{
               k:{"predicted":round(v[0],6),"actual":v[1],"tickets":v[2]}
               for k,v in sorted(self.own.items())},
           "line_pattern":{
               k:{"predicted":round(v[0],5),"actual":v[1]}
               for k,v in sorted(self.pattern.items())},
           "weak_1st_choice_competition_context":{
               k:{"predicted":round(v[0],5),"actual":v[1],"rider_exposures":v[2]}
               for k,v in sorted(self.winner_weak.items())},
           "role_1st_choice_competition_context":{
               k:{"predicted":round(v[0],5),"actual":v[1],"rider_exposures":v[2]}
               for k,v in sorted(self.winner_by_role.items())},
        }

def ci(a,b,idx):
    day=sorted(set(a.daily)|set(b.daily))
    diff=(a.nll-b.nll if idx==0 else a.brier-b.brier)/a.n
    se=math.sqrt(sum((a.daily[d][idx]-b.daily[d][idx]-diff*a.daily[d][2])**2 for d in day))/a.n
    return {"V6_gain_over_comparator":round(diff,9),
            "approx_day_95pct_CI":[round(diff-1.96*se,9),round(diff+1.96*se,9)],
            "days":len(day)}

def evaluate(events,wg,wr,delta5,wl,wc,alpha,tag):
    trackers={n:Tracker() for n in ("V3","V5","V6")}
    for i,(rid,e) in enumerate(events.items(),1):
        pred,pat,cand=dists(e,wg,wr,delta5,wl,wc,alpha)
        for name,t in trackers.items():
            t.add(e,pred[name],pat,pred["V3"],cand,name)
        if i%2500==0:print("V6_COMPETITOR_ALL_TICKETS",tag,i,flush=True)
    return trackers

def select_dev(results):
    # Precommitted gate: full trifecta NLL improves over V5 and Brier no worse.
    opt=[a for a,r in results.items() if a>0 and
         r["V6"].nll/r["V6"].n<r["V5"].nll/r["V5"].n-1e-8 and
         r["V6"].brier/r["V6"].n<=r["V5"].brier/r["V5"].n+1e-8]
    return min(opt,key=lambda x:(results[x]["V6"].nll/x,x)) if opt else 0.

def main():
    data,coverage=original.ingest()
    train={k:v for k,v in data.items() if v["date"]<="2023-10-31"}
    dev={k:v for k,v in data.items() if v["date"]>"2023-10-31"}
    assert (len(train),len(dev))==(19248,4192)
    wg,wr=v2.fit_structural(train)
    pw,n=v3.train_pattern_gate(train)
    dists.pw=pw
    delta5=v5.fit_residual(train,wg,wr)
    wl,wc=train and train_opponent(train,wg,wr,delta5)
    del train,data
    development={}
    for alpha in SCALES:
        development[alpha]=evaluate(dev,wg,wr,delta5,wl,wc,alpha,"2023_DEV_"+str(alpha))
    selected=select_dev(development)
    print("V6_COMPETITION_ALPHA_DEVELOPMENT_CHOSEN",selected,flush=True)
    tests={}
    for year,fn in (("2024",src24.ingest_2024),
                    ("2025",src25.read_2025),
                    ("2026",src26.ingest_2026)):
        events,source_audit=fn()
        assert len(events)==YEARS[year],(year,len(events))
        results=evaluate(events,wg,wr,delta5,wl,wc,selected,year)
        expected={ "2024":(4.31982650,4.31060107),
                   "2025":(4.36613863,4.35552835),
                   "2026":(4.40591444,4.39199251)}[year]
        for name,ref in zip(("V3","V5"),expected):
            if abs(results[name].nll/results[name].n-ref)>2e-5:
                raise AssertionError(("Baseline V3/V5 drifted",year,name,results[name].nll/results[name].n,ref))
        baseline=results["V3"].tail["<0.01%"]
        rival=results["V6"].tail["<0.01%"]
        better_tail=abs(rival[1]-rival[0])<abs(baseline[1]-baseline[0])
        c={
          "NLL":ci(results["V5"],results["V6"],0),
          "Brier":ci(results["V5"],results["V6"],1),
        }
        passed=bool(selected>0 and
                    c["NLL"]["approx_day_95pct_CI"][0]>0 and
                    c["Brier"]["V6_gain_over_comparator"]>=-1e-8 and
                    better_tail)
        tests[year]={
            "source_audit":source_audit,
            "all_models":{name:t.pack() for name,t in results.items()},
            "V6_vs_V5":c,
            "fixed_V3_rare_absolute_error_improved":better_tail,
            "year_guard_passed":passed,
        }
        print("V6_COMPETITOR_YEAR_COMPLETE",json.dumps({
            "year":year,"alpha":selected,
            "scores":{k:{"NLL":round(v.nll/v.n,8),"Brier":round(v.brier/v.n,8)}
                      for k,v in results.items()},
            "V6_vs_V5":c,
            "fixed_V3_rare_predicted":{k:round(v.tail["<0.01%"][0],4)
                                        for k,v in results.items()},
            "fixed_V3_rare_actual":baseline[1],
            "year_gate_pass":passed},ensure_ascii=False),flush=True)
        del events
    output={
      "state":"COMPETITION_RELATION_TRIFECTA_RESEARCH_V6_COMPLETE",
      "research_question":"What measurable opponent-line strength/role relations are associated with underpredicted wins by low-ranked riders?",
      "training":"2023 Jan-Oct only",
      "development":"2023 Nov-Dec, previously inspected",
      "historical_replay":["2024","2025","2026"],
      "all_line_shaped_sampled":True,
      "favored_line_shape":None,
      "baseline_V3_V5_frozen":True,
      "opponent_relationship_model":"Two residual conditional-softmax heads: per-line and per-rider interactions with other lines/remaining competitors",
      "cannot_infer_actual_race_tactics_from_scores":True,
      "no_odds_payout_market":True,
      "development_candidate_scales":SCALES,
      "development_metrics":{str(a):{n:{"NLL":round(t.nll/t.n,9),
                    "Brier":round(t.brier/t.n,9)} for n,t in r.items()}
                    for a,r in development.items()},
      "selected_scale_2023_only":selected,
      "coefficient_line":dict(sorted(wl.items())),
      "coefficient_rider":dict(sorted(wc.items())),
      "test_years":tests,
      "historical_all_years_strict_guard":bool(selected>0 and all(
         tests[year]["year_guard_passed"] for year in tests)),
      "production_promoted":False,
      "valid_blind_confirmatory_period_remaining":"NONE: 2024-26 already inspected, future period needed",
      "limitations":[
        "Line leaders' score gaps measure available pre-race competition proxies, not observed attacks, tactics or motives.",
        "True mechanical causality requires additional in-race moves, on-track footage or a designed natural experiment.",
        "Candidate model is evaluated against V3, V5 on all tickets and fixed V3 ultra-rare cohort, but not against odds or ROI.",
        "2024-26 previously inspected in earlier research; any use to choose a future hypothesis invalidates fresh holdout.",
        "Pre-race availability/timestamps of historical line cards not audited.",
        "No specific line formation was privileged; all line shapes equally weighted by races."
      ],
    }
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(output,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("V6_COMPETITOR_STRUCTURE_RESEARCH_COMPLETE",
          json.dumps({
            "selected_alpha":selected,"all_years_gate":output["historical_all_years_strict_guard"],
            "dev":output["development_metrics"],
            "years":{y:{"scores":{k:{"NLL":v["NLL"],"Brier":v["Brier"]}
                          for k,v in x["all_models"].items()},
                         "rare_actual":x["all_models"]["V3"]["V3_fixed_ticket_cohort"]["<0.01%"]["actual"],
                         "rare_predicted":{k:v["V3_fixed_ticket_cohort"]["<0.01%"]["predicted"]
                                           for k,v in x["all_models"].items()},
                         "year_pass":x["year_guard_passed"]} for y,x in tests.items()}
          },ensure_ascii=False),flush=True)

def train_opponent(train,wg,wr,delta5):
    return train(train,wg,wr,delta5)

if __name__=="__main__":
    main()
