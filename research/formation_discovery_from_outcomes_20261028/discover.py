"""Discovery, NOT preselected betting patterns.
Use actual finish -> whole odds -> learn conditional payout-weighted rider/line-role
membership maps. Build irregular union of cartesian rectangles via minimum
description length on train; evaluate unmodified on later dates.
No fixed 34 forms, no favorite formation chosen by ROI leaderboard.
"""
from __future__ import annotations
import itertools, json, math, os, sys, csv, tempfile
from collections import defaultdict, Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from research.snowberg_observation_corrected_20261027 import observe as corrected
bridge=corrected.old.bridge
OUT=Path(__file__).resolve().parent
START="2023-01-01";END="2023-04-30"
SOURCES=("data/a_class_12/2023_01_02/races.csv","data/a_class_12/2023_03_04/races.csv",
"data/2023/s_class_f1_all_parts/2023_q1/races.csv","data/2023/s_class_f1_all_parts/2023_q2/races.csv")
def binprice(v):return "<3" if v<3 else "3-5" if v<5 else "5-8" if v<8 else "8-12" if v<12 else "12+"
def load_lines():
    out={}
    for source in SOURCES:
        with (ROOT/source).open(encoding="utf-8-sig",newline="") as f:
            for row in csv.DictReader(f):
                day=row.get("race_date","");raw=row.get("predicted_line_formation","")
                if not START<=day<=END or not raw:continue
                try:lines=tuple(tuple(int(z) for z in t.split("-")) for t in raw.split("/"))
                except ValueError:continue
                members=[z for line in lines for z in line]
                if len(members)<5 or len(members)>9 or len(set(members))!=len(members):continue
                klass="S" if "s_class" in source else "A"
                stage=row.get("race_type","").replace("Ａ級","").replace("Ｓ級","")
                shape="-".join(map(str,sorted((len(x) for x in lines),reverse=True)))
                key=(day,lines,klass,stage,shape)
                rid=row.get("race_id","")
                if rid in out and out[rid]!=key:out[rid]=None
                elif rid not in out:out[rid]=key
    return {k:v for k,v in out.items() if v is not None}
def role_features(w,lines):
    m={car:(i,j,len(line)) for i,line in enumerate(lines) for j,car in enumerate(line)}
    a,b,c=(m[z] for z in w)
    def r(x):return "H" if x[1]==0 else "B" if x[1]==1 else "T"
    return (r(a[0] if False else a),r(b),r(c),
            int(a[0]==b[0]),int(a[0]==c[0]),int(b[0]==c[0]),
            min(a[2],3),min(b[2],3),min(c[2],3))
def rectangles(cars,score,limit=25):
    """Induce whole Cartesian boxes from scored outcomes; no registered box types.
    The rectangle generator starts from high residual tickets, but expands each
    rectangle across ALL 3 axes, scoring its entire legal ticket set. Each
    resulting box is a genuine complete formation, not a list of singleton picks.
    """
    available=set(itertools.permutations(cars,3))
    remaining=set(available)
    forms=[]
    for _ in range(limit):
        if not remaining:break
        seed=max(remaining,key=lambda w:(score.get(w,-1e10),w))
        if score.get(seed,-1e10)<=0:break
        axes=[{x} for x in seed]
        def members(ax):
            return {w for w in itertools.product(*map(sorted,ax)) if len(set(w))==3}
        def objective(ax):
            ws=members(ax)
            return sum(score.get(w,-1e10) for w in ws if w in remaining)
        while True:
            opts=[]
            for j in range(3):
                for car in cars:
                    if car in axes[j]:continue
                    test=[set(x) for x in axes];test[j].add(car)
                    ws=members(test)
                    if not ws or len(ws & remaining)<2:continue
                    val=objective(test)
                    if val>objective(axes)+1e-9:opts.append((val,-len(ws),j,car,test))
            if not opts:break
            axes=max(opts,key=lambda x:x[:4])[4]
        ws=members(axes)&remaining
        if len(ws)<4:break
        forms.append(tuple(tuple(sorted(a)) for a in axes))
        remaining-=ws
    return forms
def union(forms):
    return set().union(*(set(w for w in itertools.product(*f) if len(set(w))==3) for f in forms)) if forms else set()
def main():
    lines=load_lines();audit=Counter();records=[]
    labels,_=bridge.verified_results(2023)
    catalog,_=bridge.catalog(START,END)
    temp,key=bridge.secure_key_temp()
    try:
        for day,files in catalog.items():
            merged={};bad=set()
            with tempfile.TemporaryDirectory() as tmp:
                for path,sha in files:
                    for rid,rows in bridge.decrypt_sealed(path,sha,key,tmp).items():
                        if rid in bad:continue
                        if rid in merged and sorted(merged[rid])!=sorted(rows):
                            bad.add(rid);merged.pop(rid,None);continue
                        merged[rid]=rows
            for rid,rows in merged.items():
                if rid not in lines or rid not in labels:continue
                line=lines[rid]
                if line[0]!=day:continue
                odds={}
                for row in rows:
                    if row[2]!="3rentan":continue
                    lo,hi=float(row[4]),float(row[5])
                    if lo!=hi:continue
                    w=tuple(map(int,row[3].split("-")));odds[w]=lo
                n=len(set(z for w in odds for z in w))
                if len(odds)!=n*(n-1)*(n-2):continue
                finish=tuple(labels[rid])
                if finish not in odds:continue
                linecars=set(z for l in line[1] for z in l)
                if linecars!=set(z for w in odds for z in w):continue
                records.append((day,rid,line,odds,finish))
            audit["days"]+=1
            if audit["days"]%20==0:print("LOADED",day,len(records),flush=True)
    finally:temp.cleanup()
    # Discovery from outcome-first empirical conditional distributions.
    # Weighted price ratio of whole outcome role-patterns, learned on Jan-Feb only.
    train=[r for r in records if r[0]<="2023-02-28"]
    test=[r for r in records if r[0]>="2023-03-01"]
    hist=defaultdict(lambda:defaultdict(lambda:[0.,0.,0]))
    for day,rid,line,odds,finish in train:
        _,ls,klass,stage,shape=line
        cell=(klass,stage,shape,binprice(min(odds.values())))
        invsum=sum(1/x for x in odds.values())
        for w,o in odds.items():
            f=role_features(w,ls)
            hist[cell][f][1]+=(1/o)/invsum
        hist[cell][role_features(finish,ls)][0]+=1
        hist[cell][role_features(finish,ls)][2]+=1
    # The empirical ratio is regularized to prevent small-cell overfit;
    # price-aware objective is applied to all legal ordered outcomes.
    def predict(record):
        day,rid,line,odds,finish=record
        _,ls,klass,stage,shape=line
        cell=(klass,stage,shape,binprice(min(odds.values())))
        stats=hist.get(cell,{})
        if sum(v[0] for v in stats.values())<30:return [],set()
        cars=tuple(sorted({z for w in odds for z in w}))
        invsum=sum(1/x for x in odds.values())
        # Relative frequency corrected by prior mass, with 30 race pseudo-counts.
        q={}
        for w,o in odds.items():
            f=role_features(w,ls)
            obs,market,_=stats.get(f,(0,0,0))
            total=sum(v[0] for v in stats.values())
            multiplier=(obs+30*market/max(total,1))/(market+30*market/max(total,1)) if market>0 else 1.
            # A ticket's expected value must exceed break-even, not merely exceed market frequency.
            q[w]=((1/o)/invsum)*multiplier*o-1.
        forms=rectangles(cars,q)
        return forms,union(forms)
    def summarize(part):
        sums=Counter();examples=[]
        for r in part:
            forms,ws=predict(r)
            if not ws:continue
            _,rid,line,odds,finish=r
            sums["races"]+=1;sums["points"]+=len(ws);sums["stake"]+=100*len(ws)
            hit=int(finish in ws);sums["hits"]+=hit
            sums["payout"]+=100*odds[finish]*hit
            if len(examples)<12:examples.append({"race_id":rid,"date":r[0],"line":line[1],"forms":forms,"points":len(ws),"hit":hit,"payout":100*odds[finish]*hit})
        return {"races":sums["races"],"tickets":sums["points"],"stake_yen":sums["stake"],"payout_yen":round(sums["payout"]),"profit_yen":round(sums["payout"]-sums["stake"]),"roi_pct":round(100*sums["payout"]/sums["stake"],2) if sums["stake"] else None,"examples":examples}
    OUT.mkdir(parents=True,exist_ok=True)
    result={"status":"EXPERIMENTAL_OUTCOME_FIRST_UNION_DISCOVERY","audit":dict(audit),"records":len(records),"training_records":len(train),"test_records":len(test),"method":"Learn empirical role co-occurrence vs market in Jan-Feb, then generate novel Cartesian union boxes from expected payout scores; fixed holdout Mar-Apr.","train":summarize(train),"holdout":summarize(test),"caveats":["Exploratory, no fixed 34 forms.","Outcome role buckets still an inductive bias; not a proof of global optimal formation.","Uses final odds; not live betting.","No fixed selection from holdout outcomes."]}
    (OUT/"result.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    print("DISCOVERY_RESULT",json.dumps({k:v for k,v in result.items() if k not in ("train","holdout")}),flush=True)
    print("TRAIN",json.dumps({k:v for k,v in result["train"].items() if k!="examples"}),flush=True)
    print("HOLDOUT",json.dumps({k:v for k,v in result["holdout"].items() if k!="examples"}),flush=True)
if __name__=="__main__":main()
