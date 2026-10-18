#!/usr/bin/env python3
"""G3-only third finisher RESULT-FIRST census. Never fit coefficients here.

P(third=k | observed first=i, observed second=j) baseline:
    q_k / sum_{h not in (i,j)} q_h
where q is G3-only independent first probability, fitted using prior G3 years.
2022 q is in-sample for empirical census only; it MUST NOT train third law.
G3 first and second original coefficients remain immutable.
"""
from __future__ import annotations
import collections,hashlib,json,math
from pathlib import Path
import numpy as np
from scipy.special import logsumexp
from train_g3_first import load,OUT
from train_g3_second import first_by_calendar,STAGE

HERE=Path(__file__).resolve().parent

MEMBERSHIP=("both_line","first_line_only","second_line_only","neither_line")
DIRECTIONS=(
    "ahead_of_both","between_first_second","behind_both",
    "ahead_of_first","directly_behind_first","further_behind_first",
    "ahead_of_second","directly_behind_second","further_behind_second",
    "neither_line"
)
ROLES=("leader","second","third_or_later","solo")
FEATURES=("log_q",)+tuple("owner_"+t for t in MEMBERSHIP[:3])+tuple(
    "direction_"+t for t in DIRECTIONS[:-1])+tuple("role_"+t for t in ROLES[1:])

def characterize(car,first,second,position):
    group,k=position[car]
    g1,i=position[first]
    g2,j=position[second]
    same1=group==g1;same2=group==g2
    if same1 and same2:
        owner="both_line"
        direction=("ahead_of_both" if k<min(i,j) else
                   "behind_both" if k>max(i,j) else "between_first_second")
    elif same1:
        owner="first_line_only"
        direction=("ahead_of_first" if k<i else
                   "directly_behind_first" if k==i+1 else "further_behind_first")
    elif same2:
        owner="second_line_only"
        direction=("ahead_of_second" if k<j else
                   "directly_behind_second" if k==j+1 else "further_behind_second")
    else:
        owner="neither_line";direction="neither_line"
    role=("solo" if len(group)==1 else
          "leader" if k==0 else
          "second" if k==1 else "third_or_later")
    return owner,direction,role

def baseline_and_features(r,qs):
    n=len(r["cars"])
    first=r["cars"][r["winner"]]
    second=r["actual_second_car"];third=r["actual_third_car"]
    if len({first,second,third})!=3:
        raise ValueError("Podium includes duplicate entrants")
    if not {first,second,third}.issubset(r["cars"]):
        raise ValueError("Podium contains nonstarter")
    pos={}
    for line in r["line"]:
        for k,car in enumerate(line):
            if car in pos:raise ValueError("Duplicate lineup entry")
            pos[car]=(tuple(line),k)
    if set(pos)!=set(r["cars"]):raise ValueError("Predicted line does not cover race")
    P=np.zeros((9,len(FEATURES)),dtype=float)
    M=np.zeros(9,dtype=bool)
    actual=np.zeros(9,dtype=bool)
    expected=np.zeros(9,dtype=float)
    total=sum(qs[k] for k,car in enumerate(r["cars"]) if car not in (first,second))
    if total<=0:raise ValueError("Invalid Harville baseline")
    categories={}
    for k,car in enumerate(r["cars"]):
        owner,direction,role=characterize(car,first,second,pos)
        categories[car]=(owner,direction,role)
        P[k]=[math.log(max(qs[k],1e-15))]+[
            float(owner==v) for v in MEMBERSHIP[:3]
        ]+[
            float(direction==v) for v in DIRECTIONS[:-1]
        ]+[
            float(role==v) for v in ROLES[1:]
        ]
        M[k]=car not in (first,second)
        actual[k]=car==third
        if M[k]:expected[k]=qs[k]/total
    if actual.sum()!=1 or not M[actual][0] or abs(expected.sum()-1)>1e-8:
        raise RuntimeError("Invalid conditional third distribution")
    return P,M,int(r["cars"].index(third)),expected,categories

def compute_census():
    races,source=load()
    q,prior_year_fits=first_by_calendar(races)
    eligible=[];missing=collections.Counter()
    for r in races:
        a=r["cars"][r["winner"]]
        b=r.get("actual_second_car")
        c=r.get("actual_third_car")
        if b not in r["cars"] or b==a:
            missing["missing_invalid_or_nonunique_second"]+=1;continue
        if c not in r["cars"] or c==a or c==b:
            missing["missing_invalid_or_nonunique_third"]+=1;continue
        eligible.append(r)
    n=len(eligible)
    X=np.zeros((n,9,len(FEATURES)),dtype=float)
    M=np.zeros((n,9),dtype=bool)
    Y=np.zeros(n,dtype=np.int64)
    baseline=np.zeros((n,9),dtype=float)
    obs_by=collections.defaultdict(lambda:collections.Counter())
    candidate_by=collections.defaultdict(lambda:collections.Counter())
    pred_by=collections.defaultdict(lambda:collections.Counter())
    candidate_count=0
    ids=[]
    for ix,r in enumerate(eligible):
        P,mask,target,pr,categories=baseline_and_features(r,q[r["id"]])
        X[ix]=P;M[ix]=mask;Y[ix]=target;baseline[ix]=pr
        first=r["cars"][r["winner"]]
        second=r["actual_second_car"]
        position={car:(tuple(line),k) for line in r["line"] for k,car in enumerate(line)}
        config=("same_first_second_line" if position[first][0]==position[second][0]
                else "different_first_second_lines")
        date=r["date"]
        tag=("year_"+date[:4],config,("year_"+date[:4]+":"+config))
        for car,k in zip(r["cars"],range(len(r["cars"]))):
            if not M[ix,k]:continue
            candidate_count+=1
            owner,direction,role=categories[car]
            groups={
                "line_ownership":owner,
                "directed_position":direction,
                "candidate_role":role,
                "first_second_configuration":config,
                "configuration_x_ownership":config+"|"+owner,
                "configuration_x_position":config+"|"+direction
            }
            for group,value in groups.items():
                keys=(group+"|"+value,)
                for extra in ("all", "year_"+date[:4]):
                    for key in keys:
                        candidate_by[extra][key]+=1
                        pred_by[extra][key]+=float(pr[k])
                        if car==r["actual_third_car"]:obs_by[extra][key]+=1
        ids.append({"race_id":r["id"],"date":date,"field":len(r["cars"]),
                    "first":first,"second":second,"third":r["actual_third_car"],
                    "first_q_in_sample":date.startswith("2022"),
                    "line_configuration":config})
    if len(races)!=8995 or len(eligible)<8800:
        raise RuntimeError("Unexpected shift in frozen G3 population")
    if candidate_count!=int(M.sum()) or sum(Y>=0)!=n:
        raise RuntimeError("Invalid third candidate count")
    if len(set(z["race_id"] for z in ids))!=n:
        raise RuntimeError("Duplicate G3 third cases")
    STAGE.mkdir(parents=True,exist_ok=True);OUT.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(STAGE/"g3_third_matrix.npz",X=X,mask=M,y=Y,baseline=baseline)
    (STAGE/"g3_third_index.json").write_text(json.dumps(ids,ensure_ascii=False)+"\n",encoding="utf-8")
    table={}
    for section in sorted(candidate_by):
        table[section]={}
        for key in sorted(candidate_by[section]):
            count=candidate_by[section][key]
            exp=float(pred_by[section][key]);obs=int(obs_by[section][key])
            table[section][key]={
                "candidate_rows":count,
                "harville_expected_third":exp,
                "actual_third":obs,
                "observed_minus_expected":obs-exp,
                "actual_candidate_frequency":obs/count if count else None,
                "baseline_candidate_frequency":exp/count if count else None
            }
    cohort={
        "status":"G3_THIRD_RESULT_FIRST_EMPIRICAL_BEFORE_FITTING",
        "research_grade":"G3開催 S級 only, excludes G1 G2 F1 F2 A級 girls",
        "all_first_eligible_races":len(races),
        "eligible_unique_first_second_third_races":n,
        "candidate_third_rows":candidate_count,
        "missing_podium_breakdown":dict(missing),
        "baseline_equation":"P(third=k | actual first=i, actual second=j)=q_k / (1-q_i-q_j), where q from prior-year-only G3 first model",
        "conditions_on_real_first_second_for_empirical_measurement":True,
        "not_an_unconditional_trifecta_accuracy":True,
        "G3_only_prior_year_first_model_fits":prior_year_fits,
        "features_to_be_considered_after_census":list(FEATURES),
        "first_fit_2022_in_sample_observational_only":True,
        "all_discovery_data_included":True,
        "data_collected_retrospectively_asof_not_certified":True,
        "previous_G3_first_and_second_parameters_untouched":True,
        "original_source_race_id_sha256":source["source_race_ids_sha256"],
        "eligible_third_race_ids_sha256":hashlib.sha256(("\n".join(sorted(z["race_id"] for z in ids))+"\n").encode()).hexdigest(),
        "observed_vs_expected":table,
        "no_third_coefficient_fitted":True
    }
    path=OUT/"g3_third_empirical.json"
    path.write_text(json.dumps(cohort,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("G3_THIRD_CENSUS",json.dumps({
        "races":n,"candidate_rows":candidate_count,
        "missing":dict(missing),
        "line_ownership":{k:v for k,v in table["all"].items() if k.startswith("line_ownership|")},
        "directed_position":{k:v for k,v in table["all"].items() if k.startswith("directed_position|")}
    },ensure_ascii=False),flush=True)

if __name__=="__main__":
    compute_census()
