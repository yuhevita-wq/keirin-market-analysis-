"""New experiment: learn expected realized return of formation actions, not winning-ticket ranks."""
import csv,gzip,json,math,re,sys
from pathlib import Path
from collections import defaultdict,Counter
from itertools import combinations,permutations
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

OUT=Path("buy_skip_ml_output"); OUT.mkdir(exist_ok=True)
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
def formations(kind,cars,market,order):
    """Finite action grammar; all templates evaluated, no outcome-dependent construction."""
    actions={}
    def add(name,family,axis,partner_count,seq):
        ts=tuple(sorted(set(seq)))
        if ts: actions[name]=(family,axis,partner_count,ts)
    n=len(cars)
    for k in (1,2,3,4,5,6,8,10):
        if k<=len(market):add(f"unstructured:{k}",0,0,k,sorted(market,key=lambda x:(-market[x],x))[:k])
    if kind=="exacta":
        for i,a in enumerate(order[:4]):
            for direction in ("head","tail","fold"):
                def weight(b):
                    if direction=="head":return market[a,b]
                    if direction=="tail":return market[b,a]
                    return market[a,b]+market[b,a]
                p=sorted((b for b in cars if b!=a),key=lambda b:(-weight(b),b))
                for k in range(1,min(5,len(p))+1):
                    if direction=="fold" and k>4:continue
                    seq=[t for b in p[:k] for t in ([(a,b)] if direction=="head" else [(b,a)] if direction=="tail" else [(a,b),(b,a)])]
                    add(f"{direction}:{i}:{k}",{"head":1,"tail":2,"fold":3}[direction],i+1,k,seq)
        for k in (2,3,4):
            if k<=n:add(f"box:{k}",4,0,k,permutations(order[:k],2))
        for i,j in combinations(range(min(4,n)),2):
            axes={order[i],order[j]}
            for k in (2,3):
                for direction in ("head","tail"):
                    others=sorted(cars,key=lambda b:(-sum(market.get((a,b) if direction=="head" else (b,a),0) for a in axes),b))[:k]
                    seq=[(a,b) if direction=="head" else (b,a) for a in axes for b in others if a!=b]
                    add(f"rect_{direction}:{i}:{j}:{k}",5 if direction=="head" else 6,1+i+j,k,seq)
    else:
        for i,a in enumerate(order[:4]):
            rest=[c for c in cars if c!=a]
            score=lambda b:sum(v for t,v in market.items() if a in t and b in t)
            p=sorted(rest,key=lambda b:(-score(b),b))
            for k in range(2,min(5,len(p))+1):
                add(f"one_axis:{i}:{k}",1,i+1,k,(tuple(sorted((a,b,c))) for b,c in combinations(p[:k],2)))
        for i,j in combinations(range(min(5,n)),2):
            a,b=order[i],order[j]
            p=sorted((c for c in cars if c not in (a,b)),key=lambda c:(-market[tuple(sorted((a,b,c)))],c))
            for k in range(1,min(5,len(p))+1):
                add(f"two_axis:{i}:{j}:{k}",2,i+j+1,k,(tuple(sorted((a,b,c))) for c in p[:k]))
        for k in (3,4,5):
            if k<=n:add(f"box:{k}",3,0,k,combinations(sorted(order[:k]),3))
        for i,j in combinations(range(min(4,n)),2):
            axes={order[i],order[j]}
            for k in (3,4):
                p=order[:k+1]
                seq=[tuple(sorted(t)) for t in combinations(p,3) if set(t)&axes]
                add(f"multi_axis:{i}:{j}:{k}",4,i+j+1,k,seq)
    # Deduplicate exact same ticket sets, preferring the actual structural template over an unstructured list.
    seen=set();out=[]
    for name,(f,a,k,ts) in sorted(actions.items(),key=lambda item:(item[1][0]==0,item[0])):
        if name.startswith("unstructured:"):
            out.append((name,f,a,k,ts));continue
        if ts not in seen:out.append((name,f,a,k,ts));seen.add(ts)
    return out
def selftest():
    cars=tuple(range(1,8))
    a={t:1/42 for t in permutations(cars,2)}
    b={t:1/35 for t in combinations(cars,3)}
    for kind,market in (("exacta",a),("trio",b)):
        acts=formations(kind,cars,market,list(cars))
        assert len(acts)>10
        for name,f,ax,k,ts in acts:
            assert len(ts)==len(set(ts)) and set(ts)<=set(market)
            assert all(len(set(t))==len(t) for t in ts)
        assert any(x[0]=="unstructured:3" for x in acts)
    assert combo("1=2=3",3)==(1,2,3)
    assert book({"1-2":3},cars,2,True) is None
selftest()
if "--self-test" in sys.argv: print("formation self-test passed");sys.exit(0)
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
            if p:races.append((date,rid,cars,mb,p))
races.sort(key=lambda x:(x[0],x[1]))
assert len({x[1] for x in races})==len(races)
save("data_audit.json",{"payout_sources":paudit,"verified_entries":len(labels),"matched_races":{k:sum(v.values()) for k,v in monthly.items()},"monthly":{k:dict(v) for k,v in monthly.items()},"excluded":dict(audit),"odds":"final trifecta and trio; exacta market unavailable; exacta pair probabilities are trifecta marginals, not exacta odds","reward":"actual paid yen per 100 yen divided by action points; equal notional stake per race; skip net zero"})
log({"monthly":{k:dict(v) for k,v in monthly.items()},"excluded":dict(audit)})
records=[];choices=[]
def report(kind,mode,months,model,indices,y,hits,points,names,race_count):
    # -1 is skip, otherwise action row index.
    sel=indices[indices>=0]
    bought=len(sel); gross=float(y[sel].sum()) if bought else 0.
    return {"bet":kind,"mode":mode,"months":months,"policy":model,"races":race_count,"selected":bought,
      "coverage":bought/race_count,"hit_rate_selected":float(hits[sel].mean()) if bought else None,
      "mean_points":float(points[sel].mean()) if bought else None,
      "return_pct":100*gross/bought if bought else None,"net_per_eligible_race":(gross-bought)/race_count,
      "formation_counts":dict(Counter(names[i].split(":")[0] for i in sel))}
def build(kind):
    X=[];Y=[];H=[];P=[];N=[];offset=[0];ds=[];rids=[]
    for date,rid,cars,(pair,trio,order,first,podium),p in races:
        if kind not in p:continue
        market=pair if kind=="exacta" else trio
        vals=np.array(sorted(market.values(),reverse=True));n=len(cars)
        entropy=-float(np.sum(vals*np.log(vals)))/math.log(len(vals))
        context=[n,entropy,*vals[:5],*sorted(first.values(),reverse=True)[:3],*sorted(podium.values(),reverse=True)[:3]]
        correct,yen=p[kind]
        for name,f,a,k,ts in formations(kind,cars,market,order):
            probs=np.array([market[t] for t in ts])
            involved=set(c for t in ts for c in t)
            rank=[order.index(c)+1 for c in involved]
            feat=[f,a,k,len(ts),len(involved),min(rank),max(rank),float(np.mean(rank)),
                  float(probs.sum()),float(probs.max()),float(probs.min()),float(probs.mean()),float(probs.std()),
                  float(sum(podium[c] for c in involved)),*context]
            hit=correct in ts
            X.append(feat);Y.append(yen/100/len(ts) if hit else 0.);H.append(hit);P.append(len(ts));N.append(name)
        ds.append(date);rids.append(rid);offset.append(len(Y))
    return np.array(X,dtype=np.float32),np.array(Y,dtype=np.float64),np.array(H,dtype=bool),np.array(P,dtype=np.int16),N,np.array(offset),np.array(ds),rids

# The original data pipeline, candidate grammar, features, reward, training
# hyperparameters and argmax above are fixed. Only a second-stage decision
# model is learned. The first stage is refitted on chronological prefixes
# exactly as the source experiment, never on a race it is predicting.
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error

WINDOWS=[(3,"2025-04-01","2025-07-01"),(6,"2025-07-01","2025-10-01"),(9,"2025-10-01","2026-01-01"),(12,"2026-01-01","2026-04-01"),(15,"2026-04-01","2026-07-01")]
EXPECTED={
 "exacta":[76.19672066995382,77.3838963246229,79.48396137223122,85.88850845681785,78.63685973207251],
 "trio":[79.28617176697713,83.6724736208484,79.42348730415992,74.44520530858654,84.01949697924876]}
EXPECTED_COUNTS=[4259,4707,2468,4454,4230]
def interval_ci(indices,y,day):
    # Descriptive date-cluster bootstrap, conditional on fixed historical decisions.
    unique,inv=np.unique(day,return_inverse=True)
    sel=indices>=0
    gross=np.zeros(len(indices));gross[sel]=y[indices[sel]]
    daily_gross=np.bincount(inv,weights=gross,minlength=len(unique))
    daily_stake=np.bincount(inv,weights=sel.astype(float),minlength=len(unique))
    rng=np.random.default_rng(SEED)
    ix=rng.integers(0,len(unique),size=(1000,len(unique)))
    g=daily_gross[ix].sum(1);s=daily_stake[ix].sum(1)
    net=(g-s)/np.bincount(inv,minlength=len(unique))[ix].sum(1)
    valid=s>0
    return {"net_per_race_ci95":np.quantile(net,[.025,.975]).tolist(),
            "return_pct_ci95":(100*np.quantile(g[valid]/s[valid],[.025,.975])).tolist() if valid.any() else None}
def emit(kind,months,policy,indices,Y,H,P,N,D,v,z,train_count,selector_n):
    row=report(kind,"final_reused_holdout" if months==15 else "expanding",months,policy,indices,Y,H,P,N,z-v)
    row.update({"formation_train_races":train_count,"selector_train_races":selector_n,
                "test_start":str(D[v]),"test_end":str(D[z-1]),
                **interval_ci(indices,Y,D[v:z])})
    records.append(row);log(row)
    return row
for kind in KINDS:
    X,Y,H,P,N,O,D,R=build(kind)
    log({"bet":kind,"races":len(D),"candidate_actions":len(Y)})
    assert len(D)==24666,("fixed source coverage changed",kind,len(D))
    past_x=[];past_y=[];past_dates=[];stage_choices=[]
    for step,(months,end,test_end) in enumerate(WINDOWS):
        v=int(np.searchsorted(D,end));z=int(np.searchsorted(D,test_end))
        assert z-v==EXPECTED_COUNTS[step],("fixed fold coverage changed",kind,months,z-v)
        a,b=int(O[v]),int(O[z])
        weights=np.repeat(1/np.diff(O[:v+1]),np.diff(O[:v+1]))
        # Exactly the original formation model and deterministic seed.
        model=LGBMRegressor(objective="regression",n_estimators=120,learning_rate=.05,num_leaves=23,min_child_samples=200,reg_lambda=10,n_jobs=4,verbosity=-1,random_state=SEED,deterministic=True,force_col_wise=True)
        model.fit(X[:a],Y[:a],sample_weight=weights)
        pred=model.predict(X[a:b])
        chosen=[];features=[];base_scores=[]
        for r in range(v,z):
            s,t=int(O[r]),int(O[r+1]);p=pred[s-a:t-a]
            i=s+int(np.argmax(p))
            best=float(p.max());ordered=np.sort(p)[::-1]
            # All are available from the frozen formation engine BEFORE the result.
            # No realized result, payout, race identifier or calendar date is an input.
            extra=[best,float(ordered[0]-ordered[1]) if len(ordered)>1 else 0.,
                   float(np.mean(p)),float(np.std(p)),float(np.quantile(p,.9)),float(len(p))]
            features.append(np.r_[X[i],extra]);chosen.append(i);base_scores.append(best)
        chosen=np.asarray(chosen,dtype=int);F=np.asarray(features,dtype=np.float64)
        base_scores=np.asarray(base_scores,dtype=np.float64);reward=Y[chosen]
        assert np.isfinite(F).all()
        # Reproduce every published fixed-formation all-race benchmark exactly.
        old_roi=float(100*reward.mean())
        assert abs(old_roi-EXPECTED[kind][step])<1e-7,("formation mismatch",kind,months,old_roi)
        log({"bet":kind,"months":months,"fixed_formation_verified":True,"all_race_return":old_roi})
        n_hist=sum(len(x) for x in past_y)
        emit(kind,months,"fixed_all_races",chosen,Y,H,P,N,D,v,z,v,n_hist)
        emit(kind,months,"original_score_gt_1",np.where(base_scores>1,chosen,-1),Y,H,P,N,D,v,z,v,n_hist)
        emit(kind,months,"skip_all",np.full(len(chosen),-1,dtype=int),Y,H,P,N,D,v,z,v,n_hist)
        gate_scores={}
        if n_hist:
            FX=np.concatenate(past_x);RY=np.concatenate(past_y);FD=np.concatenate(past_dates)
            assert FD.max()<D[v],("selector leakage",FD.max(),D[v])
            # Compare two preregistered regressors. Neither can alter a ticket.
            # No policy selection or threshold fitting on current/future folds.
            models={
              "gate_ridge":make_pipeline(StandardScaler(),Ridge(alpha=100.)),
              "gate_lightgbm":LGBMRegressor(objective="regression",n_estimators=80,learning_rate=.04,num_leaves=15,min_child_samples=150,reg_lambda=10,n_jobs=4,verbosity=-1,random_state=SEED,deterministic=True,force_col_wise=True)
            }
            for name,gate in models.items():
                gate.fit(FX,RY)
                g=np.asarray(gate.predict(F),dtype=float)
                assert np.isfinite(g).all()
                gate_scores[name]=g
                row=emit(kind,months,name,np.where(g>1,chosen,-1),Y,H,P,N,D,v,z,v,n_hist)
                log({"bet":kind,"months":months,"gate":name,"oos_mse":float(mean_squared_error(reward,g)),"mean_prediction":float(g.mean()),"mean_realized":float(reward.mean())})
        else:
            log({"bet":kind,"months":months,"gate_status":"collect first OOF training quarter only"})
        for q,r in enumerate(range(v,z)):
            i=chosen[q]
            entry={"bet":kind,"formation_train_months":months,"date":str(D[r]),"race_id":R[r],
                   "fixed_action":N[i],"points":int(P[i]),"actual_gross":float(Y[i]),
                   "base_score":float(base_scores[q]),"original_buy":bool(base_scores[q]>1)}
            for name,g in gate_scores.items():
                entry[name+"_score"]=float(g[q]);entry[name+"_buy"]=bool(g[q]>1)
            stage_choices.append(entry)
        # Only after the evaluation is recorded may its labels train the next gate.
        past_x.append(F);past_y.append(reward.copy());past_dates.append(D[v:z].copy())
        save("learning_stages.json",records)
        pd.DataFrame(stage_choices).to_csv(OUT/f"{kind}_fixed_actions_and_decisions.csv",index=False)
    del X,Y,H,P,N,O,D,R
save("status.json",{"status":"completed","source_commit":"85eac08304a059b7444eebe366a8b5c34fffd133","source_blob":"893381e45f1b83e495a4da1eb07b754ac0ca9bec","formation_all_race_benchmarks_verified":True,"rows":len(records),"limitations":["formation engine algorithm and hyperparameters frozen, fold weights refitted only on past data","gate trained only on prior out-of-fold selected actions","2026 April-June was already inspected in the source experiment, not a new untouched holdout","final odds, no verified pre-close snapshot","exacta probabilities derived from trifecta, not direct exacta odds","equal notional stake per selected race; no live wagering","no tuning on the evaluated quarters"]})
