"""Stage 1: learn joint finish probabilities with a race-normalized proper loss. No betting policies."""
import csv,gzip,json,math,re,sys
from pathlib import Path
from collections import defaultdict,Counter
from itertools import combinations,permutations
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

OUT=Path("final_decision_output"); OUT.mkdir(exist_ok=True)
ROOT=Path("data"); SOURCE=ROOT/"rider_condition_2025_202610_full"
SEED=20261127
KINDS={"exacta":"2車単","trio":"3連複"}
def save(name,obj):
    p=OUT/name; tmp=p.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
    tmp.replace(p)
def log(obj): print(json.dumps(obj,ensure_ascii=False,allow_nan=False),flush=True)
def combo(s,k):
    t=tuple(map(int,re.findall(r"\d+",str(s))))
    if len(t)!=k or len(set(t))!=k or any(x<1 or x>9 for x in t): return None
    return t
def read_payments():
    found={k:defaultdict(set) for k in KINDS}; counts=Counter()
    files=[]
    for family in ("2025","2026_h1","a_class_12","a3_challenge"):
        files.extend(p for p in (ROOT/family).rglob("payouts.csv") if family in ("2025","2026_h1") or p.parent.name.startswith(("2025","2026_01","2026_03","2026_05")))
    for p in sorted(files):
        with p.open(encoding="utf-8-sig",newline="") as f:
            reader=csv.DictReader(f)
            need={"race_id","ticket_type","combination","payout_yen","status"}
            if not need.issubset(reader.fieldnames or []): counts["schema_invalid"]+=1;continue
            counts["files"]+=1
            for row in reader:
                if row["status"]!="paid":continue
                for kind,label in KINDS.items():
                    if row["ticket_type"]!=label:continue
                    c=combo(row["combination"],2 if kind=="exacta" else 3)
                    try: yen=int(row["payout_yen"].replace(",",""))
                    except (ValueError,AttributeError):continue
                    if c is None or yen<100:continue
                    if kind=="trio":c=tuple(sorted(c))
                    found[kind][row["race_id"]].add((c,yen))
    return found,dict(counts)
def read_labels():
    r=pd.concat([pd.read_csv(SOURCE/f"races_{y}.csv.gz",dtype=str) for y in (2025,2026)])
    d=pd.concat([pd.read_csv(SOURCE/f"rider_races_{y}.csv.gz",dtype=str) for y in (2025,2026)])
    valid=set(r.loc[r.verified_podium.str.lower().isin(["true","1","yes"]),"race_id"])
    dates={}; labels={}
    for rid,g in d[d.race_id.isin(valid)].groupby("race_id",sort=False):
        if not g.result_status.eq("full_result").all() or g.race_date.nunique()!=1:continue
        date=g.race_date.iloc[0]
        if not "2025-01-01"<=date<"2026-11-01":continue
        car=pd.to_numeric(g.car_no,errors="coerce")
        pos=pd.to_numeric(g.finish_position,errors="coerce")
        if car.isna().any() or car.duplicated().any() or not all((pos==i).sum()==1 for i in (1,2,3)):continue
        cars=tuple(sorted(map(int,car)))
        if not 5<=len(cars)<=9:continue
        podium=tuple(int(car[pos==i].iloc[0]) for i in (1,2,3))
        dates[rid]=date;labels[rid]=(cars,podium)
    return dates,labels
def book(raw,cars,k,ordered):
    expected=set(permutations(cars,k) if ordered else combinations(cars,k))
    b={}
    for key,v in raw.items():
        c=combo(key,k)
        if c is None:return None
        if not ordered:c=tuple(sorted(c))
        try:v=float(v)
        except (ValueError,TypeError):return None
        if not math.isfinite(v) or v<1 or c in b:return None
        b[c]=1/v
    if set(b)!=expected:return None
    total=sum(b.values());return {c:v/total for c,v in b.items()}
def market_books(raw,cars):
    three=book(raw.get("3連単",{}),cars,3,True)
    trio=book(raw.get("3連複",{}),cars,3,False)
    if three is None or trio is None:return None
    pair=defaultdict(float); podium=defaultdict(float); first=defaultdict(float)
    for (a,b,c),v in three.items():
        pair[a,b]+=v; first[a]+=v
        for n in (a,b,c):podium[n]+=v
    order=sorted(cars,key=lambda c:(-podium[c],c))
    return dict(pair),trio,order,first,podium

def rider_history():
    d=pd.concat([pd.read_csv(SOURCE/f"rider_races_{y}.csv.gz",dtype=str) for y in (2025,2026)],ignore_index=True)
    d["date"]=pd.to_datetime(d.race_date,errors="raise")
    d["car"]=pd.to_numeric(d.car_no,errors="coerce")
    d["pos"]=pd.to_numeric(d.finish_position,errors="coerce")
    d["key"]=d.provisional_player_key.fillna("")
    d["cls"]=d["class"].fillna("").str.strip().str.upper()
    # Only verified outcome records; no current day or cross-class history.
    hist=d[(d.key!="")&(d.cls!="")&d.race_id.isin(labels)&d.result_status.eq("full_result")&d.pos.notna()].copy()
    hist["starts"]=1
    for k in (1,2,3):hist[f"p{k}"]=(hist.pos==k).astype(int)
    cols=["starts","p1","p2","p3"]
    keys=["key","cls","date"]
    daily=hist.groupby(keys)[cols].sum().reset_index()
    calendar=d[keys].drop_duplicates()
    daily=calendar.merge(daily,how="left",on=keys).fillna(0).sort_values(keys)
    daily[cols]=daily.groupby(["key","cls"])[cols].cumsum()-daily[cols]
    d=d.merge(daily,on=keys,how="left",validate="many_to_one")
    # Weak symmetric smoothing for sparse histories; counts remain features.
    for k in (1,2,3):d[f"rate{k}"]=(d[f"p{k}"]+1)/(d.starts+7)
    d["logstarts"]=np.log1p(d.starts)
    for k in ("term","line_position","line_size"):d[k]=pd.to_numeric(d[k],errors="coerce")
    d["front"]=d["style"].fillna("").str.contains("逃",regex=False).astype(float)
    d["both"]=d["style"].fillna("").str.contains("両",regex=False).astype(float)
    d["chase"]=d["style"].fillna("").str.contains("追",regex=False).astype(float)
    for c in ("SS","S1","S2","A1","A2","A3"):d["cls_"+c]=d.cls.eq(c).astype(float)
    fs=["rate1","rate2","rate3","logstarts","term","line_position","line_size","front","both","chase"]+["cls_"+c for c in ("SS","S1","S2","A1","A2","A3")]
    # Include missingness only for numeric source columns, avoid inf.
    for k in ("term","line_position","line_size"):
        d["missing_"+k]=d[k].isna().astype(float);fs.append("missing_"+k)
    d[fs]=d[fs].fillna(0)
    out={}
    for rid,g in d[d.race_id.isin(labels)].groupby("race_id",sort=False):
        out[rid]={int(car):vec for car,vec in zip(g.car,g[fs].to_numpy(dtype=np.float32)) if pd.notna(car)}
    return out,fs

import lightgbm as lgb
from scipy.optimize import minimize_scalar
from sklearn.metrics import mean_squared_error
WINDOWS=[(3,"2025-04-01","2025-07-01"),(6,"2025-07-01","2025-10-01"),(9,"2025-10-01","2026-01-01"),(12,"2026-01-01","2026-04-01"),(15,"2026-04-01","2026-07-01")]
def softmax(z,counts):
    z=np.asarray(z,dtype=np.float64)
    counts=np.asarray(counts,dtype=np.int64)
    starts=np.r_[0,np.cumsum(counts)[:-1]]
    assert len(z)==counts.sum() and np.all(counts>0)
    q=np.exp(z-np.repeat(np.maximum.reduceat(z,starts),counts))
    return q/np.repeat(np.add.reduceat(q,starts),counts)

from functools import lru_cache
import hashlib
def build(kind):
    xp=[];xm=[];ys=[];ms=[];cs=[];ds=[];ids=[];tickets=[]
    rfeature_names=None;mfeature_names=None
    for date,rid,cars,(pair,trio,order,first,podium),_ in races:
        rr=riders[rid];assert set(rr)==set(cars)
        c=list(permutations(cars,2) if kind=="exacta" else combinations(cars,3))
        c=np.asarray(c,dtype=np.int8);n=len(cars)
        vec=np.stack([rr[car] for car in cars])
        f=vec[:,:7]
        ctx=np.r_[n,f.mean(0),f.std(0)]
        ctx_names=["field_size"]+["field_mean_"+k for k in rnames[:7]]+["field_sd_"+k for k in rnames[:7]]
        m=pair if kind=="exacta" else trio
        prob=np.array([m[tuple(t)] for t in c],dtype=float)
        assert np.isclose(prob.sum(),1) and np.all(prob>0)
        pm=np.array([[first[int(car)] for car in t] for t in c])
        pp=np.array([[podium[int(car)] for car in t] for t in c])
        selected=np.array([[rr[int(car)] for car in t] for t in c],dtype=np.float32)
        # Car numbers are available pre-race and used only as participant features.
        if kind=="exacta":
            x=np.column_stack([selected[:,0],selected[:,1],selected[:,0,:7]-selected[:,1,:7],c,np.tile(ctx,(len(c),1))])
            rn=["head_"+k for k in rnames]+["tail_"+k for k in rnames]+["difference_"+k for k in rnames[:7]]+["head_car","tail_car"]+ctx_names
            cross=np.array([sum(trio[t] for t in trio if int(a) in t and int(b) in t) for a,b in c])
            mx=np.column_stack([np.log(prob),pm,pp,cross])
            mn=["market_log_pair","market_head_first","market_tail_first","market_head_podium","market_tail_podium","market_trio_joint_two"]
        else:
            x=np.column_stack([selected.mean(1),selected.min(1),selected.max(1),selected.std(1),c.min(1),c.max(1),np.tile(ctx,(len(c),1))])
            rn=[prefix+k for prefix in ("mean_","min_","max_","sd_") for k in rnames]+["min_car","max_car"]+ctx_names
            cross=np.array([sum(pair[int(a),int(b)] for a,b in permutations(t,2)) for t in c])
            mx=np.column_stack([np.log(prob),pm.min(1),pm.max(1),pm.sum(1),pp.min(1),pp.max(1),pp.sum(1),cross])
            mn=["market_log_trio","market_min_first","market_max_first","market_sum_first","market_min_podium","market_max_podium","market_sum_podium","market_pair_within_trio"]
        market_context=np.r_[-np.sum(prob*np.log(prob))/np.log(len(prob)),np.sort(prob)[::-1][:5],np.sort(list(first.values()))[::-1][:3],np.sort(list(podium.values()))[::-1][:3]]
        mx=np.column_stack([mx,np.tile(market_context,(len(c),1))])
        mn+=["market_entropy"]+["market_combo_"+str(k) for k in range(1,6)]+["market_first_"+str(k) for k in range(1,4)]+["market_podium_"+str(k) for k in range(1,4)]
        correct=labels[rid][1][:2] if kind=="exacta" else tuple(sorted(labels[rid][1]))
        y=np.all(c==np.array(correct),axis=1).astype(np.float32);assert y.sum()==1
        xp.append(x.astype(np.float32));xm.append(mx.astype(np.float32));ys.append(y);ms.append(prob);cs.append(len(c));ds.append(date);ids.append(rid)
        tickets.extend("-".join(map(str,t)) if kind=="exacta" else "=".join(map(str,t)) for t in c)
        if rfeature_names is None:rfeature_names=rn;mfeature_names=mn
        assert rn==rfeature_names and mn==mfeature_names
    x=np.concatenate(xp);m=np.concatenate(xm);y=np.concatenate(ys);p=np.concatenate(ms);counts=np.asarray(cs,dtype=np.int64)
    assert x.shape[1]==len(rfeature_names) and m.shape[1]==len(mfeature_names) and np.isfinite(x).all() and np.isfinite(m).all()
    return x,m,y,p,counts,np.asarray(ds,dtype="datetime64[D]"),ids,np.asarray(tickets),rfeature_names,mfeature_names

@lru_cache(maxsize=32)
def grammar(cars,kind):
    """Enumerate exact feasible structural ticket sets in the declared grammar.
    No candidate is chosen by outcome or by any learned formation score.
    Overlapping rectangular axes are excluded by design, and reported as such.
    A duplicate ticket set may belong to different structural descriptions."""
    ts=list(permutations(cars,2) if kind=="exacta" else combinations(cars,3))
    ix={t:i for i,t in enumerate(ts)};groups=defaultdict(set)
    maxk=8 if kind=="exacta" else 10;mink=2 if kind=="exacta" else 1
    def add(f,seq):
        ids=tuple(sorted(set(ix[t] for t in seq)))
        if mink<=len(ids)<=maxk:groups[f,len(ids)].add(ids)
    if kind=="exacta":
        for a in cars:
            rest=[b for b in cars if b!=a]
            for k in range(2,min(maxk,len(rest))+1):
                for p in combinations(rest,k):
                    add("head",[(a,b) for b in p]);add("tail",[(b,a) for b in p])
            for k in range(1,min(4,len(rest))+1):
                for p in combinations(rest,k):
                    add("fold",[(x,y) for b in p for x,y in ((a,b),(b,a))])
        for axes in combinations(cars,2):
            rest=[c for c in cars if c not in axes]
            for k in (2,3,4):
                for p in combinations(rest,k):
                    add("rect_head",[(a,b) for a in axes for b in p])
                    add("rect_tail",[(b,a) for a in axes for b in p])
        for p in combinations(cars,3):add("box",permutations(p,2))
    else:
        for a in cars:
            rest=[c for c in cars if c!=a]
            for k in range(2,min(5,len(rest))+1):
                for p in combinations(rest,k):
                    add("one_axis",[tuple(sorted((a,b,c))) for b,c in combinations(p,2)])
        for axes in combinations(cars,2):
            rest=[c for c in cars if c not in axes]
            for k in range(1,min(maxk,len(rest))+1):
                for p in combinations(rest,k):
                    add("two_axis",[tuple(sorted((*axes,c))) for c in p])
        for k in (3,4,5):
            for p in combinations(cars,k):add("box",combinations(p,3))
        for pool in combinations(cars,5):
            for axes in combinations(pool,2):
                add("multi_axis",[t for t in combinations(pool,3) if set(t)&set(axes)])
    arrays={key:np.asarray(sorted(g),dtype=np.int16) for key,g in sorted(groups.items())}
    return ts,arrays
def select(weights,groups,min_k,max_k):
    # Additivity: for disjoint mutually exclusive outcomes and equal stake,
    # maximum total weight at fixed k is the k highest individual weights.
    # Each structural family is solved exactly within the declared grammar.
    order=np.argsort(-weights,axis=0,kind="stable")
    chosen={}
    for k in range(min_k,min(max_k,len(weights))+1):
        for col in range(weights.shape[1]):chosen["unrestricted",k,col]=np.sort(order[:k,col])
    best={}
    for (family,k),idx in groups.items():
        s=weights[idx].sum(axis=1)
        winner=np.argmax(s,axis=0)
        for col,j in enumerate(winner):
            ids=idx[j];chosen[family,k,col]=ids
            key=(k,col);value=float(s[j,col])
            if key not in best or value>best[key][0]:
                best[key]=(value,ids)
    for (k,col),(_,ids) in best.items():chosen["best_structured",k,col]=ids
    return chosen

# Fixed retrospective external-date diagnostic. No fitting, no threshold sweep.
# This is not an independent holdout or a pre-close executable strategy.
def save(name,obj):
    (OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False,default=lambda x:x.item() if isinstance(x,np.generic) else x.tolist() if isinstance(x,np.ndarray) else str(x)),encoding="utf-8")
def selftest():
    ts,gr=grammar(tuple(range(1,8)),"trio")
    w=np.arange(len(ts),dtype=float)[:,None]
    s=select(w,gr,1,10)
    assert len(s["best_structured",3,0])==3
    assert np.allclose(softmax(np.array([1.,2.,3.]),[3]).sum(),1)
    print("Frozen diagnostic self-test passed",flush=True)
selftest()
if "--self-test" in sys.argv:sys.exit(0)
kind=sys.argv[1];assert kind in KINDS
dates,labels=read_labels()
# History is reconstructed with the original definition, strictly earlier calendar days.
# The model parameters are immutable. Known past July results may enter later
# July rider history exactly as in the original model, never current-day results.
riders,rnames=rider_history()
source=ROOT/"2026_future_block1"/"s_class_f1_20260701_20260830"
pd0=pd.read_csv(source/"payouts.csv",dtype=str)
pay=defaultdict(set)
for row in pd0.itertuples(index=False):
    if row.status!="paid" or row.ticket_type!=KINDS[kind]:continue
    c=combo(row.combination,2 if kind=="exacta" else 3)
    if c is None:continue
    if kind=="trio":c=tuple(sorted(c))
    try:yen=int(row.payout_yen.replace(",",""))
    except (ValueError,AttributeError):continue
    if yen>=100:pay[row.race_id].add((c,yen))
root=ROOT/"odds_universe_v1"/"data"/"2026_future_block1"/"s_class_f1_20260701_20260830"
raw=defaultdict(lambda:{"3連単":{},"3連複":{}})
for folder in sorted(root.glob("batch_*")):
    for key,filename,k in (("3連単","trifecta_final_odds.csv.gz",3),("3連複","trio_final_odds.csv.gz",3)):
        with gzip.open(folder/filename,"rt",encoding="utf-8-sig",newline="") as h:
            for row in csv.DictReader(h):
                if row.get("odds_phase") not in ("final",None,"") or row.get("odds_status") not in ("available",None,""):continue
                c=combo(row["combination"],k)
                if c is None:continue
                if key=="3連複":c=tuple(sorted(c))
                try:v=float(row["odds"])
                except (ValueError,TypeError):continue
                if not math.isfinite(v) or v<1:continue
                tag=("-" if key=="3連単" else "=").join(map(str,c))
                b=raw[row["race_id"]][key]
                assert tag not in b or np.isclose(b[tag],v)
                b[tag]=v
audit=Counter();races=[];actual_prices={}
for rid,books in raw.items():
    if rid not in labels:audit["missing_verified_label"]+=1;continue
    date=dates[rid]
    if not "2026-07-01"<=date<"2026-09-01":audit["outside_fixed_dates"]+=1;continue
    cars,podium=labels[rid];mb=market_books(books,cars)
    if mb is None:audit["incomplete_books"]+=1;continue
    paid=pay[rid]
    target=podium[:2] if kind=="exacta" else tuple(sorted(podium))
    if len(paid)!=1 or next(iter(paid))[0]!=target:audit["missing_or_mismatched_payout"]+=1;continue
    races.append((date,rid,cars,mb,{kind:next(iter(paid))}))
    actual_prices[rid]=books
races.sort(key=lambda x:(x[0],x[1]))
audit["eligible"]=len(races);audit["raw_book_races"]=len(raw)
print(json.dumps({"coverage":dict(audit),"bet":kind}),flush=True)
# Eligibility criteria locked before reading diagnostic outcomes.
# Final odds alone cannot support prospective approval, regardless of observed ROI.
# Existing past-only statistical gate yielded no qualifying cases.
predeclared={"period":"2026-07-01 through 2026-08-31","model_source_run":46005962698,
  "model_file":f"{kind}_market_plus_rider_15m.txt","model_training_end_exclusive":"2026-04-01",
  "policy":"fixed 2-point exacta proxy diagnostic or 3-point trio direct expected return diagnostic",
  "trio_threshold":1.0,"exacta_threshold":None,
  "min_eligible_races":1000,"min_diagnostic_selected":200,"min_dates":30,
  "required_independent_holdout":True,"required_preclose_prices":True,
  "approval":"all evidence requirements and one-sided 95% day-cluster lower ROI bound above 100%; otherwise not approved",
  "no_retuning":True}
save("locked_protocol.json",predeclared)
if not races:
    save("final_decision.json",{"bet":kind,"decision":"NOT_APPROVED","reason":"no matched diagnostic races","coverage":dict(audit)})
    sys.exit(0)
X,M,Y,market,C,D,R,T,RN,MN=build(kind)
paths=list(Path("frozen_model").rglob(predeclared["model_file"]));assert len(paths)==1,paths
model=lgb.Booster(model_file=str(paths[0]))
assert model.num_feature()==M.shape[1]+X.shape[1]
p=softmax(model.predict(np.column_stack([M,X]),raw_score=True)+np.log(market),C)
O=np.r_[0,np.cumsum(C)]
rows=[]
for j,(date,rid,cars,mb,payout) in enumerate(races):
    a,b=int(O[j]),int(O[j+1]);q=market[a:b];pr=p[a:b];y=Y[a:b]
    ts,groups=grammar(cars,kind)
    assert len(ts)==b-a and list(T[a:b])==[("-" if kind=="exacta" else "=").join(map(str,z)) for z in ts]
    k=2 if kind=="exacta" else 3
    price=1/q if kind=="exacta" else np.array([actual_prices[rid]["3連複"]["=".join(map(str,z))] for z in ts])
    weights=np.column_stack([pr,pr*price])
    chosen=select(weights,groups,2 if kind=="exacta" else 1,8 if kind=="exacta" else 10)
    ids=chosen.get(("best_structured",k,1))
    if ids is None:continue
    score=float(np.mean(pr[ids]*price[ids]))
    gross=float(y[ids].sum()*payout[kind][1]/100/k)
    rows.append({"race_id":rid,"date":date,"points":k,"score":score,"predicted_hit":float(pr[ids].sum()),"actual_hit":int(y[ids].sum()),"gross":gross,"diagnostic_selected":bool(kind=="trio" and score>=1)})
d=pd.DataFrame(rows);d.to_csv(OUT/"fixed_diagnostic.csv",index=False)
def assess(x):
    if not len(x):return {"n":0,"roi":None,"lower95":None}
    day=x.groupby("date").gross.agg(["sum","count"])
    rng=np.random.default_rng(20261128);idx=rng.integers(0,len(day),size=(2000,len(day)))
    boot=day["sum"].to_numpy()[idx].sum(1)/day["count"].to_numpy()[idx].sum(1)
    return {"n":len(x),"days":len(day),"roi":float(x.gross.mean()),"lower95":float(np.quantile(boot,.05)),
            "hit":float(x.actual_hit.mean()),"predicted_hit":float(x.predicted_hit.mean()),"score":float(x.score.mean())}
allr=assess(d);selected=assess(d[d.diagnostic_selected])
blockers=["period previously examined in wider research; not a certified independent holdout",
          "only final odds; no verified pre-close decision-time prices"]
if len(d)<1000:blockers.append("fewer than 1000 eligible diagnostic races")
if kind=="exacta":blockers.append("no direct exacta odds")
if selected["n"]<200:blockers.append("fewer than 200 direct-EV diagnostic cases")
if selected.get("days",0)<30:blockers.append("fewer than 30 selected dates")
if selected["lower95"] is None or selected["lower95"]<=1:blockers.append("no demonstrated lower confidence bound above break-even")
report={"bet":kind,"decision":"NOT_APPROVED" if blockers else "APPROVED","blockers":blockers,"coverage":dict(audit),
        "all_diagnostic":allr,"selected_diagnostic":selected,"model_sha256":hashlib.sha256(paths[0].read_bytes()).hexdigest(),
        "independent_validation":False,"preclose_prices_verified":False,"new_fitting":False}
save("final_decision.json",report);print(json.dumps({"final_decision":report},ensure_ascii=False),flush=True)
