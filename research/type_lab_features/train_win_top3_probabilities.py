#!/usr/bin/env python3
"""Fit audited pre-race rider abilities to coherent win and top-three probabilities.

Purely historical prior-day features, with racing labels held in separate CMS files.
Single Plackett-Luce ranked-top-three likelihood. Probability mass by construction:
sum P(first)=1, sum P(top-three)=3, P(top-three)>=P(first). All events 7 or 9 cars.
Fit 2021..2024-06, temperature tune 2024-07..12, untouched test 2025..2026H1.
No final odds as inputs, no rider ID/name predictors, no future feature leakage.
"""
from __future__ import annotations
import argparse
import contextlib
import gzip
import hashlib
import json
import math
import os
import subprocess
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.optimize import minimize, minimize_scalar

ROOT=Path(__file__).resolve().parents[2]
CERT=ROOT/"odds_market/vault_public_cert.pem"
YEARS=tuple(range(2021,2027))
EXPECTED=80882
EPS=1e-14
FEATURE_NAMES=(
    "elo_relative_200","log_starts","cold_start","history_5plus",
    "career_win_smooth","career_top2_smooth","career_top3_smooth",
    "r30_win_smooth","r30_top3_smooth","r30_log_starts",
    "r90_win_smooth","r90_top3_smooth","r90_log_starts",
    "r365_win_smooth","r365_top3_smooth","r365_log_starts",
    "log_days_since_last","days_missing","field_size_9_indicator",
)
D=len(FEATURE_NAMES)

@contextlib.contextmanager
def ephemeral_private_key():
    secret=os.environ.pop("KEIRIN_SEVEN_BET_PRIVATE_KEY_PEM","")
    if not secret:raise RuntimeError("OWNER_KEY_MISSING")
    with tempfile.TemporaryDirectory(prefix="type_lab_probability_key_") as td:
        pem=Path(td)/"key.pem"
        pem.write_bytes(secret.encode("utf8").replace(b"\r\n",b"\n"))
        pem.chmod(0o600)
        secret=""
        yield pem

def decrypt_rows(file,key):
    command=["openssl","cms","-decrypt","-inform","DER","-in",str(file),
             "-recip",str(CERT),"-inkey",str(key)]
    p=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL)
    try:
        with gzip.GzipFile(fileobj=p.stdout,mode="rb") as g:
            for ln in g:
                if ln.strip():yield json.loads(ln)
        if p.wait(timeout=90)!=0:raise RuntimeError("CMS_INPUT_SIGNATURE_FAILURE")
    finally:
        if p.poll() is None:p.kill()
        p.wait()

class Sealed:
    def __init__(self,target):self.target=Path(target)
    def __enter__(self):
        self.target.parent.mkdir(parents=True,exist_ok=True)
        self.proc=subprocess.Popen(["openssl","cms","-encrypt","-binary","-aes-256-gcm",
            "-stream","-outform","DER","-out",str(self.target),str(CERT)],
            stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        self.gz=gzip.GzipFile(fileobj=self.proc.stdin,mode="wb",mtime=0)
        self.count=0
        return self
    def write(self,obj):
        self.gz.write((json.dumps(obj,ensure_ascii=False,separators=(",",":"),allow_nan=False)+"\n").encode("utf8"))
        self.count+=1
    def __exit__(self,exc,_,__):
        self.gz.close();self.proc.stdin.close()
        rc=self.proc.wait(timeout=120)
        if exc is not None or rc:
            self.target.unlink(missing_ok=True)
            if exc is None:raise RuntimeError("CMS_OUTPUT_ENCRYPTION_FAILED")
        return False

def sha256(file):
    h=hashlib.sha256()
    with Path(file).open("rb") as f:
        for block in iter(lambda:f.read(1048576),b""):h.update(block)
    return h.hexdigest()

def bounded_rate(count,rate,prior,alpha):
    k=float(count)
    if k<=0 or rate is None:return float(prior)
    return float((k*min(1,max(0,float(rate)))+alpha*prior)/(k+alpha))

def rider_features(r):
    c=int(r["history_observations"])
    if c<0:raise RuntimeError("NEGATIVE_HISTORY")
    kday=r.get("days_since_last_observed_race")
    if kday is not None and int(kday)<1:raise RuntimeError("NON_POSITIVE_PRIOR_GAP")
    e=float(r["asof_elo_pairwise_observed"])
    if not math.isfinite(e):raise RuntimeError("ELO_NAN")
    cw=bounded_rate(c,r.get("lifetime_observed_win_rate"),0.13,10)
    c2=bounded_rate(c,r.get("lifetime_observed_top2_rate"),0.27,10)
    c3=bounded_rate(c,r.get("lifetime_observed_top3_rate"),0.4,10)
    w={}
    for window in (30,90,365):
        d=r.get("recent_%dd"%window) or {}
        n=int(d.get("starts",0))
        if n<0:raise RuntimeError("NEGATIVE_WINDOW_STARTS")
        w[window]=(
            bounded_rate(n,d.get("win_rate"),cw,5),
            bounded_rate(n,d.get("top3_rate"),c3,5),
            math.log1p(n)/3
        )
    x=[
        (e-1500)/200,math.log1p(c)/5,float(c==0),float(c>=5),
        cw,c2,c3,
        *w[30],*w[90],*w[365],
        math.log1p(min(365,int(kday)))/6 if kday is not None else 0,
        float(kday is None),
        0, # race-wide 9-car flag set later
    ]
    if len(x)!=D:raise RuntimeError("INTERNAL_FEATURE_DIMENSION")
    return x

def split_day(date):
    if date<="2024-06-30":return "train"
    if date<="2024-12-31":return "calibration"
    return "test"

def load_all(folder,key):
    samples=[]
    counter=Counter()
    global_ids=set()
    year_counts=Counter()
    for year in YEARS:
        x=folder/f"prior_ability_{year}.jsonl.gz.cms"
        y=folder/f"supervised_targets_{year}.jsonl.gz.cms"
        if not x.exists() or not y.exists():raise RuntimeError("SIX_YEAR_ARCHIVE_INCOMPLETE")
        ymap={}
        for target in decrypt_rows(y,key):
            rid=str(target["race_id"])
            if rid in ymap:raise RuntimeError("DUPLICATE_TARGET")
            ymap[rid]=target
        this_year=0
        for ob in decrypt_rows(x,key):
            rid=str(ob["race_id"])
            day=str(ob["race_date"])
            if rid in global_ids:raise RuntimeError("DUPLICATED_RACE_ACROSS_YEARS")
            global_ids.add(rid)
            if day[:4]!=str(year) or (year==2026 and day>"2026-06-30"):
                raise RuntimeError("INVALID_YEAR_SCOPE")
            if ob.get("prediction_features_only_prior_calendar_days") is not True:
                raise RuntimeError("PRE_RACE_FLAG_NOT_PROVEN")
            target=ymap.pop(rid,None)
            if target is None or target["race_date"]!=day:
                raise RuntimeError("PRE_RACE_AND_LABEL_JOIN_FAILED")
            riders=ob["riders"]
            n=int(ob["n_cars"])
            if n not in (7,9) or len(riders)!=n:raise RuntimeError("BAD_FIELD_SIZE")
            cars=[int(s["car"]) for s in riders]
            if cars!=list(range(1,n+1)):raise RuntimeError("NOT_SORTED_DISTINCT_CARS")
            for r in riders:
                if r.get("event_date")!=day or r.get("no_lookahead") is not True:
                    raise RuntimeError("FUTURE_DATA_OR_BAD_ASOF")
                if r.get("history_left_censored_before_2021") is not True:
                    raise RuntimeError("UNEXPECTED_HISTORICAL_STATUS")
            finish=tuple(int(t) for t in target["finish_order"])
            if len(finish)!=3 or len(set(finish))!=3 or not set(finish)<=set(cars):
                raise RuntimeError("INVALID_PODIUM_TARGET")
            rows=np.asarray([rider_features(r) for r in riders],dtype=np.float64)
            rows[:,0]-=rows[:,0].mean()
            rows[:,-1]=float(n==9)
            if not np.isfinite(rows).all():raise RuntimeError("NON_FINITE_PREDICTOR")
            samples.append((rid,day,rows,np.array([v-1 for v in finish],dtype=np.int64)))
            year_counts[str(year)]+=1
            this_year+=1
        if ymap:raise RuntimeError("ORPHANED_LABEL")
        if this_year==0:raise RuntimeError("ZERO_YEAR_TRAINING_ROWS")
        print("MODEL_INPUT_YEAR",json.dumps({"year":year,"examples":this_year}),flush=True)
    if len(samples)!=EXPECTED:raise RuntimeError("SAMPLE_SIZE_UNEXPECTED_%d"%len(samples))
    samples.sort(key=lambda x:(x[1],x[0]))
    counter.update({"samples":len(samples),"rider_examples":sum(x[2].shape[0] for x in samples)})
    return samples,dict(year_counts),dict(counter)

def tensors(rows):
    k=len(rows)
    X=np.zeros((k,9,D),dtype=np.float64)
    mask=np.zeros((k,9),dtype=bool)
    podium=np.zeros((k,3),dtype=np.int64)
    for i,(_,_,r,y) in enumerate(rows):
        n=len(r)
        X[i,:n]=r;mask[i,:n]=True;podium[i]=y
    return X,mask,podium

def train_normalize(X,mask):
    stats=X[mask]
    means=stats.mean(axis=0);stds=stats.std(axis=0)
    stds=np.maximum(stds,0.1)
    return means,stds

def apply_scaling(X,mask,mean,std):
    Z=(X-mean)/std
    Z[~mask]=0
    return Z

def order_loglik_grad(logits,mask,podium):
    """Top-three conditional PL negative log likelihood; gradient dL/dscore."""
    n=logits.shape[0]
    scores=np.where(mask,logits,-1e10)
    remain=mask.copy()
    grad=np.zeros_like(scores)
    losses=np.zeros(n,dtype=np.float64)
    ii=np.arange(n)
    for stage in range(3):
        if not np.all(remain[ii,podium[:,stage]]):raise RuntimeError("LABEL_NOT_IN_REMAINING")
        available=np.where(remain,scores,-1e10)
        maxs=available.max(axis=1,keepdims=True)
        unnorm=np.exp(available-maxs)*remain
        denom=unnorm.sum(axis=1,keepdims=True)
        probs=unnorm/denom
        chosen=podium[:,stage]
        losses += -scores[ii,chosen]+maxs[:,0]+np.log(denom[:,0])
        grad += probs
        grad[ii,chosen]-=1.
        remain[ii,chosen]=False
    return float(losses.mean()),grad

def fit_pl(X,M,Y,l2=0.02):
    def obj(w):
        score=np.einsum("nkd,d->nk",X,w,optimize=True)
        loss,delta=order_loglik_grad(score,M,Y)
        dw=np.einsum("nkd,nk->d",X,delta,optimize=True)/len(X)
        return loss+0.5*l2*np.dot(w,w), dw+l2*w
    result=minimize(obj,np.zeros(D),method="L-BFGS-B",jac=True,
                    options={"maxiter":120,"ftol":1e-10})
    if not result.success and np.max(np.abs(result.jac))>1e-3:
        raise RuntimeError("PL_OPTIMIZATION_FAILED_"+str(result.message))
    return result.x,{"iterations":int(result.nit),"objective":float(result.fun),
                     "gradient_absmax":float(np.max(np.abs(result.jac)))}

def sequential_marginals(logits,mask):
    """Exact P(1st) and P(top3), not independent logistic outputs."""
    B,K=mask.shape
    m=np.where(mask,logits,-1e9)
    e=np.exp(m-m.max(axis=1,keepdims=True))*mask
    e/=e.sum(axis=1,keepdims=True)
    first=e.copy()
    top3=first.copy()
    # enumerate who places first j and second k; distribute third i.
    for j in range(K):
        pj=first[:,j]
        if np.all(pj==0):continue
        z1=np.maximum(1.-pj,EPS)
        second=np.where(mask,e/z1[:,None],0.)
        second[:,j]=0.
        top3+=pj[:,None]*second
        for k in range(K):
            if k==j:continue
            pk=second[:,k]
            z2=np.maximum(1.-pj-e[:,k],EPS)
            third=np.where(mask,e/z2[:,None],0.)
            third[:,j]=0.;third[:,k]=0.
            top3+=(pj*pk)[:,None]*third
    top3=np.where(mask,top3,0.0)
    if not (np.allclose(first.sum(axis=1),1.,atol=2e-10) and
            np.allclose(top3.sum(axis=1),3.,atol=2e-9) and
            np.all(top3>=first-1e-11) and np.all(top3<=1+1e-10)):
        raise RuntimeError("PROBABILITY_CONSISTENCY_VIOLATION")
    return first,top3

def report_metrics(p1,p3,mask,podium):
    ii=np.arange(len(p1))
    # Winner class log loss and full-field multiclass Brier score.
    win=p1[ii,podium[:,0]]
    correct=np.zeros_like(p1);correct[ii,podium[:,0]]=1.
    top=np.zeros_like(p3)
    for rank in range(3):top[ii,podium[:,rank]]=1.
    p=np.clip(p3[mask],EPS,1-EPS);q=top[mask]
    sorted_bins=[(0,.1),(.1,.2),(.2,.3),(.3,.4),(.4,.5),(.5,.6),(.6,.7),(.7,.8),(.8,.9),(.9,1.00001)]
    bins=[]
    for a,b in sorted_bins:
        hit=(p>=a)&(p<b)
        if np.any(hit):bins.append({"lo":a,"hi":min(b,1.),"n":int(hit.sum()),
             "pred_mean":round(float(p[hit].mean()),6),
             "actual_rate":round(float(q[hit].mean()),6)})
    return {"races":int(len(p1)),
            "win_multiclass_logloss":float(-np.log(np.maximum(win,EPS)).mean()),
            "win_multiclass_brier":float(((p1-correct)**2).sum(axis=1).mean()),
            "top3_binary_logloss":float(-(q*np.log(p)+(1-q)*np.log(1-p)).mean()),
            "top3_binary_brier":float(((p-q)**2).mean()),
            "top3_reliability_bins":bins}

def assess(logits,mask,podium,temperature):
    p1,p3=sequential_marginals(logits/temperature,mask)
    m=report_metrics(p1,p3,mask,podium)
    m["pl_top3_conditional_logloss"]=order_loglik_grad(logits/temperature,mask,podium)[0]
    return m,p1,p3

def selftest():
    from itertools import permutations
    logits=np.array([[0.3,-0.7,0.2,0.5,-0.9,-1.1,0,0,0]],dtype=float)
    mask=np.array([[True]*7+[False]*2],dtype=bool)
    a,b=sequential_marginals(logits,mask)
    e=np.exp(logits[0,:7]);e/=e.sum()
    exact=np.zeros(7)
    for j,k,i in permutations(range(7),3):
        exact[i]+=e[j]*e[k]/(1-e[j])*e[i]/(1-e[j]-e[k])
        exact[k]+=e[j]*e[k]/(1-e[j])*e[i]/(1-e[j]-e[k])
        exact[j]+=e[j]*e[k]/(1-e[j])*e[i]/(1-e[j]-e[k])
    assert np.allclose(b[0,:7],exact,atol=1e-10)
    assert abs(b.sum()-3)<1e-10 and abs(a.sum()-1)<1e-10
    assert split_day("2024-06-30")=="train" and split_day("2024-07-01")=="calibration"
    assert split_day("2024-12-31")=="calibration" and split_day("2025-01-01")=="test"
    # Numerical gradient check for the ordered top-three model.
    scores=np.array([[.3,.1,-.2,.2,.9,-.7,-.8,-100,-100]],float)
    mm=np.array([[True]*7+[False]*2])
    target=np.array([[4,0,3]],dtype=np.int64)
    loss,grad=order_loglik_grad(scores,mm,target)
    eps=1e-6
    for i in range(7):
        v=scores.copy();v[0,i]+=eps
        l2,_=order_loglik_grad(v,mm,target)
        assert abs((l2-loss)/eps-grad[0,i])<2e-6
    print("PL_COHERENT_P1_P3_SELFTEST_OK",flush=True)

def run(args):
    output=Path(args.output)
    output.mkdir(parents=True,exist_ok=True)
    with ephemeral_private_key() as key:
        rows,per_year,counts=load_all(Path(args.input),key)
        pieces={"train":[],"calibration":[],"test":[]}
        for row in rows:pieces[split_day(row[1])].append(row)
        for k in pieces:
            if not pieces[k]:raise RuntimeError("EMPTY_CHRONOLOGICAL_PARTITION_"+k)
        features={}
        for name,part in pieces.items():features[name]=tensors(part)
        X,M,Y=features["train"]
        mu,sigma=train_normalize(X,M)
        xtrain=apply_scaling(X,M,mu,sigma)
        weights,training=fit_pl(xtrain,M,Y)
        Xv,Mv,Yv=features["calibration"]
        logitsv=np.einsum("nkd,d->nk",apply_scaling(Xv,Mv,mu,sigma),weights,optimize=True)
        def val_loss(logT):
            T=np.exp(logT)
            return order_loglik_grad(logitsv/T,Mv,Yv)[0]
        opt=minimize_scalar(val_loss,method="bounded",
                            bounds=(math.log(0.4),math.log(3.0)),
                            options={"xatol":1e-5})
        if not opt.success:raise RuntimeError("TEMPERATURE_FIT_FAILED")
        T=float(math.exp(opt.x))
        report={"schema":"type_lab_pl_coherent_win_top3_v1",
            "all_historical_market_books_underlying":134336,
            "clean_asof_race_feature_rows":len(rows),"per_year":per_year,
            "partition_rule":{"train":"race_date <= 2024-06-30",
                              "calibration":"2024-07-01 to 2024-12-31",
                              "heldout_test":"2025-01-01 to 2026-06-30"},
            "counts":{name:len(vals) for name,vals in pieces.items()},
            "model":"Plackett-Luce linear rider score, top-three sequential ranked likelihood",
            "markets_used_to_fit_model":False,
            "features_strict_prior_calendar_days":True,
            "coherent":{"p1_sum":1,"p3_sum":3,"p3_ge_p1":True},
            "train_optimizer":training,
            "calibration_temperature":T,
            "rider_id_used_only_for_rebuilding_prior_history":True,
            "rider_identity_still_surrogate":True,
            "no_true_15min_market_odds":True,
            "performance":{},
            "benchmarks":{},
            "not_betting_ready":True}
        encrypted_preds=output/"heldout_test_predictions.jsonl.gz.cms"
        for name,(X1,M1,Y1) in features.items():
            logits=np.einsum("nkd,d->nk",apply_scaling(X1,M1,mu,sigma),weights,optimize=True)
            metrics,p1,p3=assess(logits,M1,Y1,T)
            report["performance"][name]=metrics
            baseline_logits=np.zeros_like(logits)
            uniform,_u1,_u3=assess(baseline_logits,M1,Y1,1)
            report["benchmarks"][name]={"uniform_selection":uniform}
            print("PL_PROBABILITY_METRICS",json.dumps({
                "split":name,"n":len(X1),"win_logloss":round(metrics["win_multiclass_logloss"],7),
                "win_brier":round(metrics["win_multiclass_brier"],7),
                "top3_logloss":round(metrics["top3_binary_logloss"],7),
                "top3_brier":round(metrics["top3_binary_brier"],7),
                "win_logloss_uniform":round(uniform["win_multiclass_logloss"],7),
                "top3_brier_uniform":round(uniform["top3_binary_brier"],7)
            }),flush=True)
            if name=="test":
                with Sealed(encrypted_preds) as f:
                    for row,a,b in zip(pieces[name],p1,p3):
                        rid,day,original,finish=row
                        n=len(original)
                        f.write({"race_id":rid,"race_date":day,
                                 "n_cars":n,"price_phase":"not_used",
                                 "p1":{str(c):float(a[c-1]) for c in range(1,n+1)},
                                 "p3":{str(c):float(b[c-1]) for c in range(1,n+1)},
                                 "target_finish_order_kept_separate_from_features":[int(k)+1 for k in finish],
                                 "source":"heldout_true_probability_estimate_no_odds",
                                 "training_included_this_race":False})
                report["encrypted_heldout_predictions"]={"name":encrypted_preds.name,
                    "records":f.count,"sha256":sha256(encrypted_preds)}
        # Store fitted model coefficients only in owner-encrypted file, not public repo.
        model=output/"pl_probability_model.jsonl.gz.cms"
        with Sealed(model) as s:
            s.write({"feature_names":FEATURE_NAMES,
                "mean":mu.tolist(),"std":sigma.tolist(),"weights":weights.tolist(),
                "temperature":T,"training_end":"2024-06-30",
                "calibration_end":"2024-12-31","n_train":len(pieces["train"])})
        report["encrypted_model"]={"name":model.name,"sha256":sha256(model)}
        # Production-only refit AFTER untouched historical evaluation. This model
        # has no independent 2025/2026 test; never report historical heldout scores
        # as its own out-of-sample performance.
        allX,allM,allY=tensors(rows)
        allmean,allstd=train_normalize(allX,allM)
        full_weights,full_fit=fit_pl(apply_scaling(allX,allM,allmean,allstd),allM,allY)
        production=output/"pl_full_80882_production_model.jsonl.gz.cms"
        with Sealed(production) as f:
            f.write({"schema":"type_lab_pl_coherent_win_top3_production_refit_v1",
                "feature_names":FEATURE_NAMES,
                "mean":allmean.tolist(),"std":allstd.tolist(),
                "weights":full_weights.tolist(),"temperature":T,
                "temperature_source":"historical_2024H2_calibration_frozen",
                "full_history_races_used":len(rows),
                "last_known_race_date":rows[-1][1],
                "independent_holdout_assessment_of_this_refit":False,
                "asof_only_features":True})
        report["latest_production_refit"]={
            "n_races_used":len(rows),
            "last_observed_date":rows[-1][1],
            "frozen_temperature_from_2024H2":T,
            "optimizer":full_fit,
            "encrypted_model_name":production.name,
            "encrypted_sha256":sha256(production),
            "separate_unseen_test_for_this_refit":False,
            "previous_2025_2026_test_scores_belong_to_pre_2025_frozen_model":True
        }
        print("PL_FULL_HISTORY_REFIT_DONE",json.dumps({
           "all_races_trained":len(rows),"last_day":rows[-1][1],
           "iterations":full_fit["iterations"]}),flush=True)
        # All 80,882 are *in-sample* for the separate full-history refit.
        # NEVER market these probabilities as out-of-time validation.
        production_probs=output/"production_in_sample_all_80882_p1p3.jsonl.gz.cms"
        prod_features=apply_scaling(allX,allM,allmean,allstd)
        prod_logits=np.einsum("nkd,d->nk",prod_features,full_weights,optimize=True)
        prod_p1,prod_p3=sequential_marginals(prod_logits/T,allM)
        with Sealed(production_probs) as writer:
            for original,a,b in zip(rows,prod_p1,prod_p3):
                rid,day,rider_feature_data,_finish=original
                n=len(rider_feature_data)
                writer.write({
                    "race_id":rid,"race_date":day,"n_cars":n,
                    "p1":{str(c):float(a[c-1]) for c in range(1,n+1)},
                    "p3":{str(c):float(b[c-1]) for c in range(1,n+1)},
                    "source":"production_refit_80882_in_sample",
                    "in_sample":True,"independent_test":False,
                    "includes_future_relative_to_earlier_historical_races":True
                })
        if writer.count!=len(rows):raise RuntimeError("PRODUCTION_PREDICTION_COUNT_ERROR")
        report["production_in_sample_probabilities"]={
            "name":production_probs.name,"records":writer.count,
            "sha256":sha256(production_probs),
            "in_sample":True,"valid_for_historical_accuracy_evaluation":False
        }
        print("PRODUCTION_80882_PROBABILITIES_WRITTEN",json.dumps({
            "records":writer.count,"in_sample":True}),flush=True)
        (output/"probability_training_report.json").write_text(json.dumps(report,ensure_ascii=False,
                     indent=2,allow_nan=False)+"\n",encoding="utf8")
        if not (report["counts"]["train"]>20000 and report["counts"]["test"]>20000):
            raise RuntimeError("TRAIN_OR_TEST_TOO_SMALL")
        print("PL_PROBABILITY_TRAINING_COMPLETE",json.dumps({
            "examples":len(rows),"train":len(pieces["train"]),
            "calibration":len(pieces["calibration"]),"test":len(pieces["test"]),
            "temperature":round(T,6),"heldout_win_ll":
            round(report["performance"]["test"]["win_multiclass_logloss"],7),
            "heldout_top3_brier":
            round(report["performance"]["test"]["top3_binary_brier"],7)
        }),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--selftest",action="store_true")
    p.add_argument("--input")
    p.add_argument("--output")
    args=p.parse_args()
    if args.selftest:selftest()
    else:
        if not args.input or not args.output:p.error("--input and --output required")
        run(args)
