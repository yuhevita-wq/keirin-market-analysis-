"""Odds-only probability model: no cards, scores, riders' histories or lineups.

Outcome CSV is read ONLY for the winning-order target; every feature is built
exclusively from complete trifecta and trio final-odds distributions.
"""
from __future__ import annotations
import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "data/audits/odds_only_model_2023_2026.json"
WEIGHTS = ROOT / "data/models/odds_only_model_2025.json"
LAMS = (0.002, 0.01, 0.05, 0.2)
FEATURES = (
    "market_logprob_temperature",
    "trifecta_set_minus_trio_set_logprob",
    "cross_market_gap_x_set_popularity",
    "cross_market_gap_x_order_surprise",
    "order_conditional_popularity",
    "pair12_dependence",
    "pair23_dependence",
    "pair13_dependence",
    "trio_set_logpop",
)
GROUPS = {"temp": (0,), "cross": (0,1,2,3), "full": tuple(range(len(FEATURES)))}
D = len(FEATURES)
MAX_K = 504  # up to 9 runners, 9*8*7 exact orders


def csv_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as file:
        yield from csv.DictReader(file)


def odds_dirs():
    dirs=[]
    for y in (2023,2024,2025):
        dirs += sorted((ROOT/f"data/{y}/s_class_f1_all_parts").glob("*"))
    dirs.append(ROOT/"data/2026_h1/s_class_f1_all")
    for y in (2023,2024,2025):
        dirs.append(ROOT/f"data/{y}/s_class_yosen")
    dirs.append(ROOT/"data/2026_h1/s_class_yosen")
    return [d for d in dirs if d.is_dir()]


def odds_files(folder):
    f=folder/"trifecta_final_odds.csv"
    return [f] if f.exists() else sorted(folder.glob("trifecta_final_odds_part*.csv"))


def winners(folder):
    d=defaultdict(list)
    for r in csv_rows(folder/"results.csv"):
        try: pos,car=int(r["finish_position"]),int(r["car_no"])
        except (KeyError,ValueError):continue
        if pos in (1,2,3):d[r["race_id"]].append((pos,car))
    return {rid:tuple(c for _,c in sorted(a)) for rid,a in d.items()
            if len(a)==3 and [p for p,_ in sorted(a)]==[1,2,3]}


def load_odds_value(row):
    if row.get("odds_status")!="available":return None
    try:
        o=float(row["odds"])
        return o if math.isfinite(o) and o>=1 else None
    except (KeyError,ValueError):return None


@dataclass
class Race:
    date:str
    rid:str
    x:np.ndarray
    logq:np.ndarray
    y:int
    odds:np.ndarray
    trio_y:int
    trio_p:np.ndarray
    trio_q:np.ndarray
    trio_odds:np.ndarray


def make_race(rid, date, ordered_odds, trio_odds, actual):
    if not actual or len(set(actual))!=3:return None
    cars=sorted(set(c for key in ordered_odds for c in key))
    n=len(cars)
    if n<5 or n>9 or len(ordered_odds)!=n*(n-1)*(n-2):return None
    if len(trio_odds)!=math.comb(n,3):return None
    if any(set(key)!=set(cars) for key in ()) :return None
    if actual not in ordered_odds or tuple(sorted(actual)) not in trio_odds:return None
    keys=sorted(ordered_odds)
    sets=sorted(trio_odds)
    if any(k[0]==k[1] or k[1]==k[2] or k[0]==k[2] or any(c not in cars for c in k) for k in keys):return None
    if any(len(set(k))!=3 or any(c not in cars for c in k) for k in sets):return None
    a=np.array([1/ordered_odds[k] for k in keys],dtype=np.float64)
    b=np.array([1/trio_odds[k] for k in sets],dtype=np.float64)
    if a.min()<=0 or b.min()<=0:return None
    p3=a/a.sum();pt=b/b.sum()
    ix_set={s:i for i,s in enumerate(sets)}
    sid=np.array([ix_set[tuple(sorted(k))] for k in keys],dtype=np.int16)
    ps=np.bincount(sid,weights=p3,minlength=len(sets))
    d=np.clip(np.log(ps/pt),-2,2)
    set_pop=(np.log(ps)+math.log(len(sets)))/3
    order_pop=(np.log(p3)-np.log(ps[sid])+math.log(6))/2
    # Market-only pair interactions. No rider identifiers are numeric features.
    pos=[defaultdict(float) for _ in range(3)]
    pair=[defaultdict(float) for _ in range(3)]
    for k,v in zip(keys,p3):
        for j in range(3):pos[j][k[j]]+=float(v)
        for idx,(i,j) in enumerate(((0,1),(1,2),(0,2))):
            pair[idx][(k[i],k[j])]+=float(v)
    deps=np.zeros((len(keys),3),dtype=np.float32)
    for i,k in enumerate(keys):
        for t,(j,h) in enumerate(((0,1),(1,2),(0,2))):
            z=pair[t][(k[j],k[h])]/(pos[j][k[j]]*pos[h][k[h]])
            deps[i,t]=np.clip(math.log(max(z,1e-9)),-2,2)
    x=np.zeros((len(keys),D),dtype=np.float32)
    x[:,0]=(np.log(p3)+math.log(len(keys)))/5
    x[:,1]=d[sid]
    x[:,2]=d[sid]*set_pop[sid]
    x[:,3]=d[sid]*order_pop
    x[:,4]=order_pop
    x[:,5:8]=deps
    x[:,8]=set_pop[sid]
    return Race(date,rid,x,np.log(p3).astype(np.float32),keys.index(actual),
                np.array([ordered_odds[k] for k in keys],dtype=np.float32),
                ix_set[tuple(sorted(actual))],
                ps.astype(np.float32),pt.astype(np.float32),
                np.array([trio_odds[k] for k in sets],dtype=np.float32))


def gather():
    year=defaultdict(list);seen=set();c=defaultdict(int)
    for folder in odds_dirs():
        files=odds_files(folder)
        if not files or not (folder/"results.csv").exists() or not (folder/"trio_final_odds.csv").exists():
            continue
        print("SCAN",folder.relative_to(ROOT),flush=True)
        truth=winners(folder)
        trio=defaultdict(dict)
        for row in csv_rows(folder/"trio_final_odds.csv"):
            odds=load_odds_value(row)
            if odds is None:continue
            try:s=tuple(sorted(int(v) for v in row["combination"].split("=")))
            except (KeyError,ValueError):continue
            if len(s)==3 and len(set(s))==3:trio[row["race_id"]][s]=odds
        for path in files:
            prev=None;orders={};date=""
            def finish():
                if prev is None:return
                if prev in seen:
                    c["duplicates"]+=1;return
                race=make_race(prev,date,orders,trio.get(prev,{}),truth.get(prev))
                if race is None:
                    c["invalid_or_incomplete"]+=1;return
                seen.add(prev)
                year[int(race.date[:4])].append(race)
            for row in csv_rows(path):
                rid=row["race_id"]
                if prev is not None and rid!=prev:
                    finish()
                    orders={};date=""
                prev=rid
                if not date:date=row["race_date"]
                odds=load_odds_value(row)
                if odds is None:continue
                try:k=tuple(int(v) for v in row["combination"].split("-"))
                except (KeyError,ValueError):continue
                if len(k)==3 and len(set(k))==3:orders[k]=odds
            finish()
    return {y:sorted(a,key=lambda r:(r.date,r.rid)) for y,a in year.items()},dict(c)


def pack(rows):
    n=len(rows)
    if not n:raise ValueError("No complete races")
    X=np.zeros((n,MAX_K,D),np.float32)
    logq=np.full((n,MAX_K),-10000,np.float32)
    winners=np.zeros(n,np.int16)
    odds=np.zeros((n,MAX_K),np.float32)
    lens=np.zeros(n,np.int16)
    for i,r in enumerate(rows):
        k=len(r.logq)
        X[i,:k]=r.x
        logq[i,:k]=r.logq
        odds[i,:k]=r.odds
        winners[i]=r.y
        lens[i]=k
    return X,logq,winners,odds,lens


def probabilities(packed,w):
    X,L,_,_,_=packed
    z=L+np.einsum("nkd,d->nk",X,np.asarray(w,np.float32),optimize=True)
    z-=z.max(axis=1,keepdims=True)
    p=np.exp(z);p/=p.sum(axis=1,keepdims=True)
    return p


def fit(rows,active,lam):
    packer=pack(rows)
    X,L,Y,_,_=packer
    idx=np.arange(len(Y))
    def fun(v):
        w=np.zeros(D,np.float32);w[list(active)]=v
        q=probabilities(packer,w)
        yq=np.maximum(q[idx,Y],1e-30)
        res=q.copy();res[idx,Y]-=1
        grad=np.einsum("nk,nkd->d",res,X,optimize=True)/len(Y)
        return float(-np.log(yq).mean()+lam*np.dot(v,v)/2),grad[list(active)].astype(np.float64)+lam*v
    ans=minimize(fun,np.zeros(len(active)),method="L-BFGS-B",jac=True,
                 bounds=[(-3.0,3.0) for _ in active],
                 options={"maxiter":85,"ftol":3e-9})
    if not math.isfinite(ans.fun):raise ValueError("Unstable optimizer")
    w=np.zeros(D,np.float32);w[list(active)]=ans.x
    return w


def evaluate(rows,w=None):
    pk=pack(rows)
    X,L,Y,odds,k=pk
    idx=np.arange(len(Y))
    p=np.exp(L) if w is None else probabilities(pk,w)
    loss=-np.log(np.maximum(p[idx,Y],1e-30))
    brier=(p*p).sum(axis=1)-2*p[idx,Y]+1
    return {"n":len(Y),"logloss":float(loss.mean()),"brier":float(brier.mean()),
            "top_choice_pct":float(100*(p.argmax(axis=1)==Y).mean())},loss


def group_ci(a,b,records):
    diff=a-b
    mean=float(diff.mean())
    day=defaultdict(list)
    for z,r in zip(diff,records):day[r.date].append(float(z-mean))
    # Sandwich SE, cluster on the race date (races from same date may correlate).
    g=len(day);n=len(diff)
    se=math.sqrt((g/(g-1))*sum(sum(v)**2 for v in day.values()))/n if g>1 else None
    return {"gain_positive_is_better":mean,"race_date_clusters":g,
            "cluster_standard_error":se,
            "approx_95pct_ci":[mean-1.96*se,mean+1.96*se] if se is not None else None}


def summary_eval(records,coeffs):
    ref,ref_l=evaluate(records,None)
    out={"market":ref}
    ls={"market":ref_l}
    for name,w in coeffs.items():
        out[name],ls[name]=evaluate(records,w)
    for name in coeffs:
        out[name+"_minus_market"]=group_ci(ls["market"],ls[name],records)
    out["full_minus_cross"]=group_ci(ls["cross"],ls["full"],records)
    return out


def trio_metrics(records,alpha):
    losses=[];bres=[]
    for r in records:
        prob=r.trio_q**(1-alpha)*r.trio_p**alpha
        prob/=prob.sum()
        losses.append(-math.log(max(1e-30,prob[r.trio_y])))
        bres.append(float((prob**2).sum()-2*prob[r.trio_y]+1))
    return {"n":len(records),"logloss":sum(losses)/len(losses),
            "brier":sum(bres)/len(bres)}


def main():
    years,skipped=gather()
    ys=(2023,2024,2025,2026)
    if not all(years.get(y) for y in ys):raise RuntimeError("Incomplete historical year coverage")
    print("COUNTS",[(y,len(years[y])) for y in ys],"skipped",skipped,flush=True)
    tuning={}
    chosen={}
    for g,active in GROUPS.items():
        trials=[]
        for lam in LAMS:
            w=fit(years[2023],active,lam)
            score=evaluate(years[2024],w)[0]["logloss"]
            trials.append({"lambda":lam,"logloss_2024":score})
            print("TUNE",g,lam,score,flush=True)
        good=min(trials,key=lambda x:x["logloss_2024"])
        chosen[g]=good["lambda"];tuning[g]=trials
    alpha_trials=[]
    for alpha in (0,.25,.5,.75,1,1.25):
        alpha_trials.append({"alpha":alpha,"validation_2024":trio_metrics(years[2024],alpha)["logloss"]})
    alpha=min(alpha_trials,key=lambda x:x["validation_2024"])["alpha"]
    results={}
    for test,training in ((2024,(2023,)),(2025,(2023,2024)),(2026,(2023,2024,2025))):
        full_train=[r for y in training for r in years[y]]
        coeffs={g:fit(full_train,active,chosen[g]) for g,active in GROUPS.items()}
        rec=years[test]
        table=summary_eval(rec,coeffs)
        table["trio_market"]=trio_metrics(rec,0)
        table["trio_tri3_aggregate"]=trio_metrics(rec,1)
        table["trio_alpha_blend"]=trio_metrics(rec,alpha)
        table["trio_blend_alpha_tuned_on_2024"]=alpha
        table["coefficient_vectors"]={g:{k:round(float(v),6) for k,v in zip(FEATURES,w) if abs(v)>1e-8} for g,w in coeffs.items()}
        results[str(test)]=table
        print("EVAL",test,json.dumps({k:table[k] for k in ("market","temp","cross","full","full_minus_market","full_minus_cross","trio_market","trio_tri3_aggregate","trio_alpha_blend")}),flush=True)
    report={
        "status":"ODDS_ONLY_EXPLORATORY_FINAL_ODDS_RESEARCH",
        "features_only":["trifecta_final_odds.csv","trio_final_odds.csv"],
        "label_only":"results.csv winning order; never used for feature construction",
        "no_race_cards_or_lineup_or_score":True,
        "results_by_year":{str(y):len(years[y]) for y in ys},
        "skipped":skipped,
        "training":"2023 train, 2024 tune, 2025 OOT with 2023-24 training, 2026 OOT with 2023-25 training",
        "features":FEATURES,
        "lambda_trials":tuning,
        "lambda_selected":chosen,
        "trio_alpha_trials":alpha_trials,
        "trio_alpha_selected":alpha,
        "models":results,
        "limitations":[
            "Archived final odds, not fixed pre-start prices",
            "2024 is used for hyperparameter selection, not independent",
            "2025/2026 have appeared in project analyses, so are not wholly untouched",
            "No profits claimed; exact-order probability score and live return are different targets",
            "Odds-only research does not read class labels, rider histories, lineups, or race cards",
        ],
    }
    OUTPUT.parent.mkdir(parents=True,exist_ok=True)
    WEIGHTS.parent.mkdir(parents=True,exist_ok=True)
    OUTPUT.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    WEIGHTS.write_text(json.dumps({"status":"RESEARCH_ONLY_FINAL_ODDS","features":FEATURES,
        "training_through":"2025-12-31",
        "coefficients":results["2026"]["coefficient_vectors"],
        "selected_lambda":chosen,"trio_alpha":alpha},ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("FINISHED",OUTPUT,flush=True)


if __name__=="__main__":
    main()
