"""Anchor-first learning: arrival and conditional payout, full-flow evaluation, abstention."""
import csv, json, re, hashlib, unicodedata
from itertools import combinations
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
import math
import gc

ROOT=Path("data")
OUT=Path("axis_exacta_output"); OUT.mkdir(exist_ok=True)
NUM="age period score s_count b_count nige_count makuri_count sashi_count mark_count first_count second_count third_count outside_count win_rate top2_rate top3_rate gear".split()
CAT="style class grade prefecture".split()
def rows(p):
    with p.open(encoding="utf-8-sig",newline="") as f: yield from csv.DictReader(f)
def num(x):
    try: return float(str(x).replace("%","").replace(",",""))
    except: return np.nan
def integer(x):
    try: return int(x)
    except: return None
def norm(x): return unicodedata.normalize("NFKC",str(x or ""))
def load():
    audit=Counter(); records={}
    dirs=sorted({p.parent for p in ROOT.rglob("entries.csv") if all((p.parent/f).exists() for f in ("races.csv","results.csv","payouts.csv"))})
    for d in dirs:
        cards={x["race_id"]:x for x in rows(d/"races.csv") if "2025-01-01"<=x.get("race_date","")<="2026-10-31"}
        if not cards: continue
        ent=defaultdict(list); fin=defaultdict(list); pay=defaultdict(list)
        for e in rows(d/"entries.csv"):
            if e.get("race_id") in cards: ent[e["race_id"]].append(e)
        for e in rows(d/"results.csv"):
            if e.get("race_id") in cards: fin[e["race_id"]].append(e)
        for p in rows(d/"payouts.csv"):
            if p.get("race_id") not in cards: continue
            kind=norm(p.get("bet_code") or p.get("ticket_type") or p.get("bet_type")).lower()
            kind={"exacta":"exacta","2車単":"exacta","二車単":"exacta","trio":"trio","3連複":"trio","三連複":"trio","trifecta_box":"trio"}.get(kind)
            if kind is None: continue
            if p.get("status","paid") not in ("paid",""): continue
            c=tuple(map(int,re.findall(r"\d+",norm(p.get("combination")))))
            v=num(p.get("payout_yen"))
            if len(c)==(2 if kind=="exacta" else 3) and len(set(c))==len(c) and np.isfinite(v) and v>0: pay[p["race_id"]].append((kind,c if kind=="exacta" else tuple(sorted(c)),float(v)))
        for rid,c in cards.items():
            es=ent[rid]; ns=[integer(e.get("car_no")) for e in es]; n=integer(c.get("entry_count")) or len(es)
            if n<5 or len(es)!=n or None in ns or len(set(ns))!=n: audit["bad_entries"]+=1;continue
            if not all(any(np.isfinite(num(e.get(k))) for k in NUM) for e in es): audit["missing_features"]+=1;continue
            f=[(integer(z.get("finish_position")),integer(z.get("car_no"))) for z in fin[rid]]
            one=[a for p,a in f if p==1]; two=[a for p,a in f if p==2]
            three=[a for p,a in f if p==3]
            pp=list(set(pay[rid])); ep=[(c,v) for k,c,v in pp if k=="exacta"]; tp=[(c,v) for k,c,v in pp if k=="trio"]
            if len(one)!=1 or len(two)!=1 or len(three)!=1 or len(ep)!=1 or len(tp)!=1 or ep[0][0]!=(one[0],two[0]) or tp[0][0]!=tuple(sorted(one+two+three)) or not set(one+two+three)<=set(ns): audit["missing_or_nonstandard_result_payout"]+=1;continue
            if any(p is None for p,a in f): audit["incident_or_missing_finish"]+=1;continue
            if rid in records: audit["duplicate"]+=1;continue
            lines={}
            for li,l in enumerate(norm(c.get("predicted_line_formation")).split("/")):
                cars=[int(v) for v in re.findall(r"\d+",l)]
                for j,a in enumerate(cars): lines[a]=(li,j+1,len(cars))
            if set(lines)!=set(ns): lines={}
            base={}
            for e,a in zip(es,ns):
                x={k:num(e.get(k)) for k in NUM}
                x.update({k:norm(e.get(k)) or "?" for k in CAT})
                x.update(car_no=a,n=n,track=norm(c.get("track")),race_type=norm(c.get("race_type")),meeting_grade=norm(c.get("meeting_grade")),race_no=num(c.get("race_no")),line_position=lines.get(a,(0,np.nan,np.nan))[1],line_size=lines.get(a,(0,np.nan,np.nan))[2])
                base[a]=x
            for k in NUM:
                vals=np.array([base[a][k] for a in ns]); good=vals[np.isfinite(vals)]; mean=good.mean() if len(good) else np.nan
                for a,v in zip(ns,vals):
                    base[a][k+"_relative"]=v-mean
                    base[a][k+"_rank"]=(np.sum(good<v)+(np.sum(good==v)+1)/2)/len(good) if np.isfinite(v) else np.nan
            records[rid]=(c["race_date"],base,lines,tuple(one+two+three),ep[0][1],tp[0][1])
    return records,dict(audit)


def split(date):
    if date < "2025-10-01": return 0
    if date < "2026-01-01": return 1
    if date < "2026-04-01": return 2
    if date < "2026-07-01": return 3
    return 4

FIELDS=NUM+[k+"_relative" for k in NUM]+[k+"_rank" for k in NUM]+["car_no","line_position","line_size"]
def extrema(x,axis=0):
    ok=np.isfinite(x)
    lo=np.min(np.where(ok,x,np.inf),axis=axis)
    hi=np.max(np.where(ok,x,-np.inf),axis=axis)
    lo[~np.isfinite(lo)]=np.nan;hi[~np.isfinite(hi)]=np.nan
    return lo,hi
def prepare(rec,kind):
    # Every possible anchor set; all other riders are covered.
    total=sum(math.comb(len(v[1]),kind) for v in rec.values())
    width=4*len(FIELDS)+10
    X=np.empty((total,width),dtype=np.float32)
    y=np.zeros(total,dtype=np.int8);payout=np.zeros(total,dtype=np.float32)
    part=np.empty(total,dtype=np.int8)
    meta=[]; off=0
    for rid,(date,base,lines,podium,ep,tp) in rec.items():
        cars=sorted(base);n=len(cars);keys=list(combinations(cars,kind))
        arr=np.array([[base[a].get(k,np.nan) for k in FIELDS] for a in cars],dtype=np.float32)
        ix=np.array([[cars.index(a) for a in key] for key in keys])
        rest=np.array([[i for i in range(n) if i not in row] for row in ix])
        lo,hi=extrema(arr[ix],axis=1);ol,oh=extrema(arr[rest],axis=1)
        pts=2*(n-1) if kind==1 else n-2 if kind==2 else 1+3*(n-3)
        context=[]
        for axes in keys:
            pairs=list(combinations(axes,2))
            same=[int(lines[a][0]==lines[b][0]) for a,b in pairs if a in lines and b in lines]
            context.append([n,pts,base[cars[0]]["race_no"],len(set(v[0] for v in lines.values())) if lines else np.nan,
                 sum(same) if len(same)==len(pairs) else np.nan,
                 len(set(base[a]["style"] for a in axes)),len(set(base[a]["class"] for a in axes)),
                 len(set(base[a]["prefecture"] for a in axes)),min(axes),max(axes)])
        end=off+len(keys)
        X[off:end]=np.column_stack((lo,hi,ol,oh,np.asarray(context)))
        part[off:end]=split(date)
        for j,axes in enumerate(keys,off):
            if kind==1:
                hit=1 if axes[0]==podium[0] else 2 if axes[0]==podium[1] else 0
            else:hit=int(len(set(axes)&set(podium))>=2)
            y[j]=hit
            payout[j]=(ep if kind==1 else tp) if hit else 0
            if split(date)>=2:
                meta.append((rid,date,kind,",".join(map(str,axes)),",".join(str(a) for a in cars if a not in axes),100*pts,j))
        off=end
    return X,y,payout,part,meta

def model_params():
    return dict(iterations=350,depth=6,learning_rate=.06,verbose=100,thread_count=4,random_seed=42,allow_writing_files=False)

def learn(rec,kind):
    X,y,pay,part,meta=prepare(rec,kind)
    tr=part==0;es=part==1;ev=part>=2
    clf=CatBoostClassifier(**model_params())
    clf.fit(X[tr],y[tr],eval_set=(X[es],y[es]),early_stopping_rounds=40)
    clf.save_model(str(OUT/f"anchor_{kind}_arrival.cbm"))
    pp=clf.predict_proba(X[ev]);classes=list(clf.classes_)
    pred=np.zeros(len(pp));prob=1-pp[:,classes.index(0)]
    detail={}
    for cls in ([1,2] if kind==1 else [1]):
        a=tr&(y==cls);b=es&(y==cls)
        # Fit conditional mean actual payout, not log payout or clipped winners.
        reg=CatBoostRegressor(**model_params())
        reg.fit(X[a],pay[a]/100,eval_set=(X[b],pay[b]/100),early_stopping_rounds=40)
        reg.save_model(str(OUT/f"anchor_{kind}_payout_{cls}.cbm"))
        conditional=np.maximum(0,reg.predict(X[ev]))*100
        pred+=pp[:,classes.index(cls)]*conditional
        detail[f"payout_{cls}_train_count"]=int(a.sum())
        detail[f"payout_{cls}_pred_mean"]=float(conditional.mean())
        detail[f"payout_{cls}_best_iteration"]=int(reg.get_best_iteration())
        del reg
    # Save feature definitions for reproducibility. No outcome columns in X.
    (OUT/f"anchor_{kind}_features.json").write_text(json.dumps(
        [f"{g}_{k}" for g in ("anchor_min","anchor_max","other_min","other_max") for k in FIELDS]+
        ["n","points","race_no","line_count","same_line_pairs","style_count","class_count","prefecture_count","min_car","max_car"]))
    d=pd.DataFrame(meta,columns=["rid","date","kind","axes","opponents","stake","idx"])
    assert np.array_equal(d.idx.to_numpy(),np.flatnonzero(ev))
    d["payout"]=pay[ev];d["hit"]=y[ev]>0
    d["direction"]=np.where((kind==1)&(y[ev]==1),"front",np.where((kind==1)&(y[ev]==2),"reverse",np.where(y[ev]>0,"trio","miss")))
    d["p"]=prob;d["expected_payout"]=pred;d["expected_roi"]=100*pred/d.stake;d["expected_profit"]=pred-d.stake
    d["split"]=d.date.map(split)
    diagnostics=dict(rows=len(X),arrival_best_iteration=int(clf.get_best_iteration()),**detail)
    for period in (2,3,4):
        z=d[d.split==period]
        diagnostics[str(period)]={"arrival_brier":float(np.mean((z.p-z.hit.astype(float))**2)),"mean_predicted_payout":float(z.expected_payout.mean()),"mean_actual_payout":float(z.payout.mean())}
    print("ANCHOR MODEL",kind,diagnostics,flush=True)
    del X,y,pay,part,clf,pp;gc.collect()
    return d.drop(columns="idx"),diagnostics

def summarize(d,total):
    stake=float(d.stake.sum());pay=float(d.payout.sum());hits=d[d.hit]
    return dict(races=int(len(d)),coverage=len(d)/total if total else None,stake=stake,payout=pay,profit=pay-stake,
        roi=100*pay/stake if stake else None,hit_rate=float(d.hit.mean()) if len(d) else None,mean_hit_payout=float(hits.payout.mean()) if len(hits) else None,
        mean_points=float(d.stake.mean()/100) if len(d) else None,mean_predicted_roi=float(d.expected_roi.mean()) if len(d) else None,
        front_hits=int((d.direction=="front").sum()),reverse_hits=int((d.direction=="reverse").sum()),
        front_payout=float(d.loc[d.direction=="front","payout"].sum()),reverse_payout=float(d.loc[d.direction=="reverse","payout"].sum()),
        kinds={str(k):int(v) for k,v in d.kind.value_counts().items()})

def main():
    rec,audit=load()
    counts=Counter(split(v[0]) for v in rec.values())
    print("DATA",len(rec),dict(counts),audit,flush=True)
    if min(counts.get(i,0) for i in range(5))<100: raise RuntimeError("Insufficient chronological data")
    detail=[];diagnostics={}
    for k in (1,2,3):
        d,m=learn(rec,k);detail.append(d);diagnostics[str(k)]=m
    d=pd.concat(detail,ignore_index=True);del detail,rec
    report=dict(counts=dict(counts),audit=audit,diagnostics=diagnostics,policies={},limitations=[
        "Each anchor is evaluated with every other rider covered: exacta both directions, trio at least two chosen axes.",
        "Anchor arrival classifier and conditional arithmetic-mean payout regressors trained separately; all candidate anchors considered.",
        "No all-combination pre-race odds. Historical card timestamp provenance unverified.",
        "2026 July-October was already inspected in earlier experiments; not pristine holdout.",
        "Excluded incident and missing-result races can bias estimates.",
        "Candidate policies require predicted positive net profit; calibration chooses minimum predicted ROI from fixed grid, with 50 calibration races; independent confirmation in Apr-Jun requires positive realized profit and at least 50 races.",
        "Full flow, grid and support floor are engineering choices; no actual wagers or guarantee."])
    selected=[]
    # First choose the most likely anchor among candidates with strictly positive predicted net.
    # No automatic substitution of loss-making high-probability anchors.
    for name,kinds in {"exacta":[1],"trio_2axis":[2],"trio_3axis":[3],"trio_adaptive":[2,3],"joint":[1,2,3]}.items():
        sub=d[d.kind.isin(kinds)]
        base=sub.loc[sub.groupby("rid").p.idxmax()].copy()
        eligible=sub[sub.expected_profit>0]
        # One candidate per race. If none has positive expectation, no action.
        if len(eligible):
            q=eligible.loc[eligible.groupby("rid").p.idxmax()].copy()
        else:q=eligible.copy()
        cal=q[q.split==2];conf=q[q.split==3];test=q[q.split==4]
        trials=[]
        for threshold in (100,105,110,120,130,150,175,200,250,300):
            a=cal[cal.expected_roi>threshold]
            trials.append(dict(threshold=threshold,**summarize(a,counts[2])))
        good=[x for x in trials if x["races"]>=50 and x["profit"]>0]
        choice=max(good,key=lambda x:x["profit"])["threshold"] if good else None
        c=cal[cal.expected_roi>choice] if choice is not None else cal.iloc[:0]
        v=conf[conf.expected_roi>choice] if choice is not None else conf.iloc[:0]
        confirmed=choice is not None and len(v)>=50 and float(v.payout.sum()-v.stake.sum())>0
        t=test[test.expected_roi>choice] if confirmed else test.iloc[:0]
        q["policy"]=name;q["selected"]=False
        if confirmed:q.loc[q.expected_roi>choice,"selected"]=True
        selected.append(q)
        report["policies"][name]=dict(threshold=choice,confirmed=bool(confirmed),calibration_trials=trials,
            baseline={str(i):summarize(base[base.split==i],counts[i]) for i in (2,3,4)},
            positive_prediction={str(i):summarize(q[q.split==i],counts[i]) for i in (2,3,4)},
            selected_calibration=summarize(c,counts[2]),confirmation=summarize(v,counts[3]),selected_test=summarize(t,counts[4]))
    pd.concat(selected,ignore_index=True).to_csv(OUT/"anchor_selected_detail.csv",index=False)
    # Candidate-level audit enables inspecting every axis, including rejected ones.
    d.to_csv(OUT/"anchor_all_candidates.csv.gz",index=False,compression="gzip")
    (OUT/"report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2))
    compact={k:v for k,v in report.items() if k!="policies"}
    compact["policies"]={k:{a:b for a,b in v.items() if a!="calibration_trials"} for k,v in report["policies"].items()}
    print("RESULT",json.dumps(compact,ensure_ascii=False),flush=True)
if __name__=="__main__":main()
