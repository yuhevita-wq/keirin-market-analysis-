#!/usr/bin/env python3
"""Independent G3 S-class first-place research. ZERO F1 weights transferred.

Read original G3 2022-2026H1 ZIP data. Validate entrant, actual winner and
published line. Fit G3 parameters from G3 results only, choose among competing
feature families on 2025, evaluate chronologically on 2026H1.
All pre-race fields are retrospective captures, not verified real-time as-of.
"""
from __future__ import annotations
import collections,csv,hashlib,io,json,math,sys,unicodedata,zipfile
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"research/g3_independent_v1/vendor"))
import discover_v2 as struct
BASE=ROOT/"data/grade_races/g3"
OUT=ROOT/"research/g3_independent_v1/output"
FIELDS=tuple(struct.BASE_FEATURES)
STRUCT=tuple(struct.STRUCT_FEATURES)
FEATURE_NAMES=FIELDS+STRUCT+("term999_foreign_category",)
FAMILIES={
    "score_only":("score",),
    "full_no_line":FIELDS+("term999_foreign_category",),
    "score_and_line":("score",)+STRUCT,
    "full_plus_line":FEATURE_NAMES
}
ALPHAS=(0.0001,0.003,0.03)

def norm(s):
    return unicodedata.normalize("NFKC",str(s or "")).strip().upper()
def numeric(s):
    try:
        v=float(str(s or "").replace(",","").replace("%","").strip())
        return v if math.isfinite(v) else np.nan
    except (TypeError,ValueError):return np.nan
def zrows(z,name):
    path=next((s for s in z.namelist() if s.endswith("/"+name) or s==name),None)
    if not path:raise ValueError(f"Missing {name} in {z.filename}")
    with z.open(path) as f:
        yield from csv.DictReader(io.TextIOWrapper(f,encoding="utf-8-sig",newline=""))
def g3(card):
    return norm(card.get("meeting_grade")) in ("G3","GIII") and norm(card.get("race_type")).startswith("S級")

def load():
    records=[];seen=set();counts=collections.Counter();sources=[]
    for path in sorted(BASE.rglob("*.zip")):
        with zipfile.ZipFile(path) as z:
            cards={e["race_id"]:e for e in zrows(z,"races.csv") if e.get("race_id")}
            riders=collections.defaultdict(list)
            for e in zrows(z,"entries.csv"):riders[e["race_id"]].append(e)
            results=collections.defaultdict(lambda:collections.defaultdict(list))
            for e in zrows(z,"results.csv"):
                p=str(e.get("finish_position","")).strip()
                if p in ("1","2","3"):
                    results[e["race_id"]][p].append(e.get("car_no"))
            for rid,entries in riders.items():
                counts["all_source_races"]+=1
                if not rid or rid in seen:
                    counts["duplicate_id"]+=1;continue
                seen.add(rid)
                card=cards.get(rid)
                if card is None or not g3(card):
                    counts["non_g3_s_or_missing_card"]+=1;continue
                entries.sort(key=lambda e:int(e["car_no"]) if e.get("car_no","").isdigit() else 99)
                try:
                    cars=[int(e["car_no"]) for e in entries]
                    winners=[int(v) for v in results[rid]["1"]]
                except (TypeError,ValueError,KeyError):
                    counts["bad_cars"]+=1;continue
                if not 5<=len(cars)<=9 or len(set(cars))!=len(cars):
                    counts["irregular_field"]+=1;continue
                if len(winners)!=1 or winners[0] not in cars:
                    counts["no_unique_winner"]+=1;continue
                date=card.get("race_date","")
                if not len(date)==10 or any(e.get("race_date")!=date for e in entries):
                    counts["invalid_date"]+=1;continue
                line=struct.parse_line(card.get("predicted_line_formation",""),cars)
                if line is None:
                    counts["missing_or_invalid_line"]+=1;continue
                x=np.array([[numeric(e.get(k)) for k in FIELDS] for e in entries],dtype=float)
                if not np.isfinite(x).any():
                    counts["all_base_features_missing"]+=1;continue
                foreign=(x[:,FIELDS.index("term")]==999).astype(float)
                x[foreign>0,FIELDS.index("term")]=np.nan
                r={"id":rid,"date":date,"cars":cars,"winner":cars.index(winners[0]),
                   "x":x,"line":line,"rtype":card.get("race_type"),"track":card.get("track")}
                line_x=struct.build_interaction(r,line)
                r["x"]=np.concatenate((x,line_x,foreign[:,None]),axis=1)
                r["retro"]=any((not e.get("captured_at_utc") or e.get("captured_at_utc")[:10]>=date)
                               for e in entries)
                r["podium_complete"]=all(len(results[rid][str(p)])==1 for p in (1,2,3))
                # Stored for subsequent G3-only conditional finishing research.
                # NOT included in the pre-race feature matrix or first-place fit.
                r["actual_second_car"]=(
                    int(results[rid]["2"][0]) if len(results[rid]["2"])==1
                    and str(results[rid]["2"][0]).isdigit() else None
                )
                r["actual_third_car"]=(
                    int(results[rid]["3"][0]) if len(results[rid]["3"])==1
                    and str(results[rid]["3"][0]).isdigit() else None
                )
                records.append(r)
                counts["eligible"]+=1
                counts[f"field_{len(cars)}"]+=1
                counts["retrospective_capture"]+=int(r["retro"])
                counts["foreign_races"]+=int(foreign.sum()>0)
                counts["podium_complete"]+=int(r["podium_complete"])
        sources.append({"path":str(path.relative_to(ROOT)),
                        "sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
                        "bytes":path.stat().st_size})
    records.sort(key=lambda x:(x["date"],x["id"]))
    for y in (2022,2023,2024,2025,2026):
        counts[f"eligible_{y}"]=sum(r["date"].startswith(str(y)) for r in records)
    cohort={"status":"G3_INDEPENDENT_DATA_AUDITED",
            "scope":"G3 S級 only (not F1, not A級, not girls)",
            "counts":dict(counts),
            "source_archives":sources,
            "source_race_ids_sha256":hashlib.sha256(("\n".join(sorted(r["id"] for r in records))+"\n").encode()).hexdigest(),
            "feature_names":FEATURE_NAMES,
            "caution":"All captures are retrospective, and the original time fidelity is not certified. Candidate winner/features are independently extracted from results/entries with result labels not used as predictors.",
            "F1_coefficients_reused":False}
    return records,cohort

def matrix(races,transform):
    feature_len=len(transform["mean"])*2
    X=np.zeros((len(races),9,feature_len),dtype=float)
    M=np.zeros((len(races),9),dtype=bool)
    y=np.zeros(len(races),dtype=int)
    for n,r in enumerate(races):
        raw=r["x"];missing=~np.isfinite(raw)
        filled=np.where(missing,transform["median"],raw)
        zz=(filled-transform["mean"])/transform["std"]
        f=np.column_stack((zz,missing.astype(float)))
        k=len(r["cars"]);X[n,:k]=f;M[n,:k]=True;y[n]=r["winner"]
    return X,M,y

def transform_for(races,names):
    raw=np.concatenate([r["x"] for r in races])
    median=np.array([np.median(col[np.isfinite(col)]) if np.isfinite(col).any() else 0.
                     for col in raw.T])
    filled=np.where(np.isfinite(raw),raw,median)
    mean=filled.mean(axis=0);std=filled.std(axis=0)
    std[std<1.e-6]=1.
    if "term999_foreign_category" in names:
        ix=names.index("term999_foreign_category")
        median[ix]=0;mean[ix]=0;std[ix]=1
    return {"median":median,"mean":mean,"std":std}

def select(races,names):
    ids=[FEATURE_NAMES.index(c) for c in names]
    return [{**r,"x":r["x"][:,ids]} for r in races]

def fit(races,tr,alpha):
    X,M,y=matrix(races,tr);n=len(races)
    H=np.zeros_like(M,dtype=float);H[np.arange(n),y]=1
    def fun(beta):
        logit=np.einsum("rnf,f->rn",X,beta,optimize=True)
        logit=np.where(M,logit,-1.e7)
        lp=logit-logsumexp(logit,axis=1,keepdims=True)
        p=np.exp(lp)
        ll=-np.mean(lp[np.arange(n),y])+alpha*(beta@beta)*0.5
        grad=np.einsum("rnf,rn->f",X,p-H,optimize=True)/n+alpha*beta
        return ll,grad
    res=minimize(fun,np.zeros(X.shape[-1]),jac=True,method="L-BFGS-B",
                 options={"maxiter":1700,"ftol":1e-12,"gtol":1e-8})
    if not np.isfinite(res.fun) or not np.isfinite(res.x).all():
        raise RuntimeError("Unable to fit numerical independent G3 model")
    return res.x,{"success":bool(res.success),"iterations":int(res.nit),"message":str(res.message)}

def evaluate(races,tr,beta):
    X,M,y=matrix(races,tr)
    logit=np.einsum("rnf,f->rn",X,beta,optimize=True)
    logit=np.where(M,logit,-1.e7)
    p=np.exp(logit-logsumexp(logit,axis=1,keepdims=True))
    ll=-np.log(np.maximum(p[np.arange(len(y)),y],1e-300))
    brier=np.square(p-(np.arange(9)[None,:]==y[:,None])).sum(axis=1)
    uniform=np.array([math.log(len(r["cars"])) for r in races])
    peryear={}
    for year in sorted({r["date"][:4] for r in races}):
        ix=[i for i,r in enumerate(races) if r["date"].startswith(year)]
        peryear[year]={"races":len(ix),"log_loss":float(ll[ix].mean())}
    bins=[]
    for lo,hi in zip([0,.01,.025,.05,.1,.15,.2,.3,.4,.5,.6,.7,.8],
                     [.01,.025,.05,.1,.15,.2,.3,.4,.5,.6,.7,.8,1.01]):
        obs=np.zeros_like(p,dtype=bool);obs[np.arange(len(y)),y]=True
        sel=M&(p>=lo)&(p<hi)
        if sel.any():
            bins.append({"range":[lo,min(hi,1)],"candidate_rows":int(sel.sum()),
                         "expected_winners":float(p[sel].sum()),
                         "actual_winners":int(obs[sel].sum())})
    return {"races":len(races),"mean_log_loss":float(ll.mean()),
            "mean_uniform_log_loss":float(uniform.mean()),
            "benter_pseudo_r2":float(1-ll.mean()/uniform.mean()),
            "mean_brier":float(brier.mean()),
            "top1_accuracy":float(np.mean(np.argmax(p,axis=1)==y)),
            "top2_contains_winner":float(np.mean(np.any(np.argsort(-p,axis=1)[:,:2]==y[:,None],axis=1))),
            "by_year":peryear,"calibration":bins}

def main():
    races,cohort=load()
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/"g3_cohort_audit.json").write_text(json.dumps(cohort,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    train=[r for r in races if r["date"]<"2025-01-01"]
    val=[r for r in races if "2025-01-01"<=r["date"]<"2026-01-01"]
    hold=[r for r in races if "2026-01-01"<=r["date"]<"2026-07-01"]
    if min(map(len,(train,val,hold)))<200:
        raise RuntimeError(f"Insufficient G3 sample: {len(train),len(val),len(hold)}")
    candidates=[]
    for family,names in FAMILIES.items():
        trset=select(train,names);vs=select(val,names)
        scal=transform_for(trset,names)
        for alpha in ALPHAS:
            beta,optim=fit(trset,scal,alpha)
            vm=evaluate(vs,scal,beta)
            candidates.append({"family":family,"alpha":alpha,"names":names,
                               "train_races":len(train),"validation":vm,"optim":optim})
            print("G3_MODEL_CANDIDATE",family,alpha,
                  "valid_loss",round(vm["mean_log_loss"],7),flush=True)
    candidates.sort(key=lambda c:c["validation"]["mean_log_loss"])
    selected=candidates[0]
    # Select exclusively by 2025. Refit on all old races through 2025.
    old=select(train+val,selected["names"])
    test=select(hold,selected["names"])
    scal=transform_for(old,selected["names"])
    beta,optim=fit(old,scal,selected["alpha"])
    test_metrics=evaluate(test,scal,beta)
    full_train_metrics=evaluate(old,scal,beta)
    report={
        "status":"G3_INDEPENDENT_FIRST_WIN_TRAINED",
        "scope":"G3 S級 only, no F1 training rows or parameters",
        "total_eligible_races":len(races),"year_counts":cohort["counts"],
        "training_2022_2024":len(train),"validation_2025":len(val),
        "development_fit_2022_2025":len(old),"test_2026h1":len(hold),
        "selected_features":selected["names"],"selected_ridge":selected["alpha"],
        "best_validation":selected["validation"],
        "model_comparisons":[{k:v for k,v in c.items() if k!="names"} for c in candidates],
        "test":test_metrics,"development_fit":full_train_metrics,
        "selection_rule":"Best 2025 validation log-loss, no tuning on 2026H1 labels",
        "g3_model_does_not_inherit_f1_coefficients":True,
        "source_snapshot_asof_validity_verified":False,
        "today_2026_10_18_excluded_from_training":True,
        "model_coefficients":beta.tolist(),
        "transform":{k:v.tolist() for k,v in scal.items()},
        "fitting_diagnostic":optim,
        "source_race_ids_sha256":cohort["source_race_ids_sha256"],
        "caution":"Retrospective card data may not match true historical pre-race values; out-of-time evaluation is not a prospective proof."
    }
    (OUT/"g3_first_win_report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("G3_INDEPENDENT_FIRST_MODEL_REPORT",json.dumps({
        k:v for k,v in report.items() if k in ("total_eligible_races","training_2022_2024","validation_2025","test_2026h1","selected_features","selected_ridge","test","best_validation")
    },ensure_ascii=False),flush=True)
if __name__=="__main__":
    main()
