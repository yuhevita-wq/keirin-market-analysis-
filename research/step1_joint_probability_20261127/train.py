"""Stage 1: learn joint finish probabilities with a race-normalized proper loss. No betting policies."""
import csv,gzip,json,math,re,sys
from pathlib import Path
from collections import defaultdict,Counter
from itertools import combinations,permutations
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

OUT=Path("joint_probability_output"); OUT.mkdir(exist_ok=True)
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
        if not "2025-01-01"<=date<"2026-07-01":continue
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
def fit_model(x,y,counts,offset):
    counts=np.asarray(counts,dtype=np.int64); starts=np.r_[0,np.cumsum(counts)[:-1]]
    assert np.all(np.add.reduceat(y,starts)==1)
    offset=np.asarray(offset,dtype=np.float64)
    def objective(pred,dataset):
        p=softmax(pred+offset,counts)
        return p-y,np.maximum(p*(1-p),1e-6)
    params={"objective":objective,"metric":"None","learning_rate":.05,"num_leaves":23,
        "min_data_in_leaf":300,"min_sum_hessian_in_leaf":.05,"lambda_l2":10,
        "seed":SEED,"deterministic":True,"force_col_wise":True,"num_threads":4,"verbosity":-1}
    ds=lgb.Dataset(x,label=y,free_raw_data=True)
    return lgb.train(params,ds,num_boost_round=120)
def loss(z,y,counts,t=1.):
    p=softmax(z/t,counts)
    return float(-np.log(np.maximum(p[y==1],1e-15)).mean())
def test():
    counts=np.array([3,4]);z=np.array([0.,1.,2.,1.,3.,2.,0.]);p=softmax(z,counts)
    assert np.allclose(np.add.reduceat(p,[0,3]),1)
    assert np.allclose(p,softmax(z+np.repeat([10.,-5.],counts),counts))
    rng=np.random.default_rng(SEED)
    x=rng.normal(size=(1200,5)).astype(np.float32);cc=np.full(400,3)
    y=np.zeros(1200);y[np.arange(400)*3+np.argmax(x[:,0].reshape(-1,3),axis=1)]=1
    m=fit_model(x,y,cc,np.zeros(len(y)))
    pr=softmax(m.predict(x,raw_score=True),cc)
    assert np.isfinite(pr).all() and np.allclose(np.add.reduceat(pr,np.arange(0,1200,3)),1)
    print("race conditional probability self-test passed",flush=True)
test()
if "--self-test" in sys.argv:sys.exit(0)
pay,paudit=read_payments(); dates,labels=read_labels()
log({"payout_sources":paudit,"payout_races":{k:len(v) for k,v in pay.items()},"verified_entries":len(labels)})
# Store compact per-race books. Inputs are final historical prices, not verified pre-close snapshots.
races=[];audit=Counter();monthly={k:Counter() for k in KINDS}
for y in (2025,2026):
    with gzip.open(ROOT/"odds_only"/f"odds_{y}.jsonl.gz","rt",encoding="utf-8") as f:
        for line in f:
            obj=json.loads(line);rid=obj["race_id"]
            if rid not in labels:continue
            cars,podium=labels[rid]; date=dates[rid]
            mb=market_books(obj["odds"],cars)
            if mb is None:audit["book_mismatch"]+=1;continue
            p={}
            for k in KINDS:
                values=pay[k].get(rid,set())
                expected=podium[:2] if k=="exacta" else tuple(sorted(podium))
                # Require a single official paid combination matching independent finish labels.
                if len(values)==1:
                    t,yen=next(iter(values))
                    if t==expected:
                        p[k]=(t,yen);monthly[k][date[:7]]+=1
                    else:audit[k+"_payout_mismatch"]+=1
                else:audit[k+"_missing_or_ambiguous_payout"]+=1
            if len(p)==len(KINDS):races.append((date,rid,cars,mb,p))
races.sort(key=lambda x:(x[0],x[1]))
assert len({x[1] for x in races})==len(races)
save("data_audit.json",{"payout_sources":paudit,"verified_entries":len(labels),"matched_races":{k:sum(v.values()) for k,v in monthly.items()},"monthly":{k:dict(v) for k,v in monthly.items()},"excluded":dict(audit),"odds":"final trifecta and trio; exacta market unavailable; exacta pair probabilities are trifecta marginals, not exacta odds","label":"verified finishing order; official payout combination used only for cohort verification; payout amounts not used"})
log({"monthly":{k:dict(v) for k,v in monthly.items()},"excluded":dict(audit)})


assert len(races)>20000,("common cohort too small",len(races))
log({"common_verified_cohort":len(races),"previous_reference_races":24666,"cohort_definition":"both paid combinations match verified podium"})
riders,rnames=rider_history()
# No final odds, realized payouts, or outcome-derived same-day information enter rider-only features.
# Exacta uses ordered participant roles. Trio uses permutation-invariant statistics.
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
def evaluate(p,y,counts):
    starts=np.r_[0,np.cumsum(counts)[:-1]];n=len(counts)
    assert np.allclose(np.add.reduceat(p,starts),1,atol=1e-9)
    true=p[y==1]
    ranks=[]
    for a,b in zip(starts,np.cumsum(counts)):
        ix=np.flatnonzero(y[a:b])[0];q=p[a:b]
        # Tie breaks use candidate order only, equally applied to all models.
        ranks.append(int(np.sum(q>q[ix])+np.sum(q[:ix]==q[ix])+1))
    ranks=np.asarray(ranks)
    bins=np.array([0,.01,.02,.05,.10,.20,.40,.60,.80,1.0000001])
    binid=np.clip(np.searchsorted(bins,p,side="right")-1,0,len(bins)-2)
    rel=[]
    for j in range(len(bins)-1):
        mask=binid==j
        if mask.any():rel.append({"lo":float(bins[j]),"hi":float(bins[j+1]),"count":int(mask.sum()),"pred_mean":float(p[mask].mean()),"observed":float(y[mask].mean())})
    return {"races":n,"log_loss":float(-np.log(np.maximum(true,1e-15)).mean()),
            "brier":float((np.add.reduceat(p*p,starts)-2*true+1).mean()),
            "top1":float((ranks<=1).mean()),"top3":float((ranks<=3).mean()),"top5":float((ranks<=5).mean()),
            "mean_true_rank":float(ranks.mean()),"candidate_ece":float(sum(x["count"]*abs(x["pred_mean"]-x["observed"]) for x in rel)/len(p)),"reliability":rel}
results=[]
targets=[sys.argv[1]] if len(sys.argv)>1 and sys.argv[1] in KINDS else list(KINDS)
for kind in targets:
    X,M,Y,market,C,D,R,T,RN,MN=build(kind)
    O=np.r_[0,np.cumsum(C)]
    save(kind+"_features.json",{"rider":RN,"market":MN,"history":"same class, strictly earlier days","line":"position and size only; line identity not available"})
    log({"bet":kind,"races":len(D),"combinations":len(Y),"rider_features":len(RN),"market_features":len(MN)})
    assert len(D)==len(races)
    past=defaultdict(list)
    for step,(months,end,test_end) in enumerate(WINDOWS):
        v=int(np.searchsorted(D,np.datetime64(end)));z=int(np.searchsorted(D,np.datetime64(test_end)))
        assert v>=1000 and z-v>=1000
        parts=[]
        a,b=int(O[v]),int(O[z]);cc=C[v:z];y=Y[a:b]
        market_train=np.log(market[:a]);market_test=np.log(market[a:b])
        scores={"uniform":np.zeros(b-a),"market_raw":market_test}
        for name,train,test,offtr,offte,names in (
            ("rider_only",X[:a],X[a:b],np.zeros(a),np.zeros(b-a),RN),
            ("market_adjusted",M[:a],M[a:b],market_train,market_test,MN),
            ("market_plus_rider",np.column_stack([M[:a],X[:a]]),np.column_stack([M[a:b],X[a:b]]),market_train,market_test,MN+RN)):
            model=fit_model(train,Y[:a],C[:v],offtr)
            scores[name]=model.predict(test,raw_score=True)+offte
            model.save_model(str(OUT/f"{kind}_{name}_{months}m.txt"))
            gain=model.feature_importance(importance_type="gain")
            save(f"{kind}_{name}_{months}m_importance.json",dict(zip(names,map(float,gain))))
            del model,train,test
        # Temperature is fitted on prior out-of-fold forecasts only, never current outcomes.
        for name,score in list(scores.items()):
            p=softmax(score,cc)
            row={"bet":kind,"months":months,"test_start":end,"test_end_exclusive":test_end,"model":name,"calibration":"raw","temperature":1.,"train_races":v,**evaluate(p,y,cc)}
            results.append(row);log({k:v for k,v in row.items() if k!="reliability"})
            if name!="uniform" and past[name]:
                old=past[name]
                assert max(t[3] for t in old)<np.datetime64(end)
                def obj(log_t):
                    t=float(np.exp(log_t))
                    return sum(loss(s,yy,c,t)*len(c) for s,yy,c,_ in old)/sum(len(c) for _,_,c,_ in old)
                opt=minimize_scalar(obj,bounds=(math.log(.4),math.log(3.)),method="bounded")
                temp=float(np.exp(opt.x));cal=softmax(score/temp,cc)
                row={"bet":kind,"months":months,"test_start":end,"test_end_exclusive":test_end,"model":name,"calibration":"past_oof_temperature","temperature":temp,"train_races":v,"calibration_races":sum(len(c) for _,_,c,_ in old),**evaluate(cal,y,cc)}
                results.append(row);log({k:v for k,v in row.items() if k!="reliability"})
            else:cal=p
            if name!="uniform":past[name].append((score.copy(),y.copy(),cc.copy(),D[z-1]))
            if name!="uniform":
                parts.append(pd.DataFrame({"race_id":np.repeat(R[v:z],cc),"date":np.repeat(D[v:z].astype(str),cc),"combination":T[a:b],"model":name,"months":months,"probability_raw":p,"probability_calibrated":cal,"actual":y.astype(np.int8)}))
        save("learning_results.json",results)
        # Save incrementally; no payout data in probability files.
        pd.concat(parts,ignore_index=True).to_csv(OUT/f"{kind}_{months}m_oof_probabilities.csv.gz",index=False,compression="gzip")
    del X,M,Y,market,C,D,R,T,past,parts
summary=[]
for kind in targets:
    for name in ("uniform","market_raw","rider_only","market_adjusted","market_plus_rider"):
        for cal in ("raw","past_oof_temperature"):
            rr=[r for r in results if r["bet"]==kind and r["model"]==name and r["calibration"]==cal and r["months"]>=6]
            if not rr:continue
            n=sum(r["races"] for r in rr)
            row={"bet":kind,"model":name,"calibration":cal,"races":n}
            for k in ("log_loss","brier","top1","top3","top5","mean_true_rank"):
                row[k]=sum(r["races"]*r[k] for r in rr)/n
            summary.append(row);log({"pooled":row})
save("pooled_results.json",summary)
save("status.json",{"status":"completed","stage":"1 probability learning only","limitations":["same reused 2025-2026H1 period, no independent new holdout","market inputs are final odds, not verified pre-close","exacta market proxy from trifecta","line identity absent","separate exacta and trio distributions, cross-ticket consistency not imposed","no formation, purchase thresholds or ROI analysis"]})
