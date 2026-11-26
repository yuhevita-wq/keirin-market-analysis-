"""New experiment: learn expected realized return of formation actions, not winning-ticket ranks."""
import csv,gzip,json,math,re,sys
from pathlib import Path
from collections import defaultdict,Counter
from itertools import combinations,permutations
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

OUT=Path("formation_ml_output"); OUT.mkdir(exist_ok=True)
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
for kind in KINDS:
    X,Y,H,P,N,O,D,R=build(kind)
    log({"bet":kind,"races":len(D),"candidate_actions":len(Y)})
    if not len(D):continue
    # Expanding chronological folds. Same final quarter never used for tuning.
    for months,end,test_end in [(3,"2025-04-01","2025-07-01"),(6,"2025-07-01","2025-10-01"),(9,"2025-10-01","2026-01-01"),(12,"2026-01-01","2026-04-01"),(15,"2026-04-01","2026-07-01")]:
        v=int(np.searchsorted(D,end)); z=int(np.searchsorted(D,test_end))
        if v<100 or z-v<100:
            log({"bet":kind,"months":months,"skipped":"insufficient labeled races","train_races":v,"test_races":z-v});continue
        a,b=int(O[v]),int(O[z])
        # Equal weight per historical race; regression learns actual returns of entire formations.
        weights=np.repeat(1/np.diff(O[:v+1]),np.diff(O[:v+1]))
        model=LGBMRegressor(objective="regression",n_estimators=120,learning_rate=.05,num_leaves=23,min_child_samples=200,reg_lambda=10,n_jobs=4,verbosity=-1,random_state=SEED,deterministic=True,force_col_wise=True)
        model.fit(X[:a],Y[:a],sample_weight=weights)
        pred=model.predict(X[a:b])
        # Fixed comparator chosen solely on training rewards, accounting for skipped unavailable actions.
        sums=defaultdict(float); ct=Counter()
        for name,reward in zip(N[:a],Y[:a]):
            sums[name]+=float(reward)-1;ct[name]+=1
        best=max(sums,key=sums.get);best=best if sums[best]>0 else None
        learned=[];always=[];fixed=[];top3=[]
        for r in range(v,z):
            s,t=int(O[r]),int(O[r+1]);p=pred[s-a:t-a]
            i=s+int(np.argmax(p))
            always.append(i);learned.append(i if p.max()>1 else -1)
            f=next((j for j in range(s,t) if N[j]==best),-1);fixed.append(f)
            top3.append(next((j for j in range(s,t) if N[j]=="unstructured:3"),-1))
            choices.append({"bet":kind,"months":months,"date":str(D[r]),"race_id":R[r],"action":N[i] if p.max()>1 else "skip","predicted_gross":float(p.max()),"realized_gross":float(Y[i]) if p.max()>1 else 0.,"points":int(P[i]) if p.max()>1 else 0})
        for name,ix in [("learned_with_skip",learned),("learned_all_races",always),("train_best_fixed",fixed),("market_top3_reference",top3)]:
            row=report(kind,"final" if months==15 else "expanding",months,name,np.asarray(ix,dtype=int),Y,H,P,N,z-v)
            row.update({"train_races":v,"test_start":end,"test_end_exclusive":test_end,"train_best_fixed_action":best})
            records.append(row);log(row)
        save("learning_stages.json",records)
        pd.DataFrame(choices).to_csv(OUT/"out_of_sample_actions.csv",index=False)
    del X,Y,H,P,N,O,D,R
save("status.json",{"status":"completed" if records else "no_eligible_folds","results":len(records),"limitations":["historical final odds, not pre-close snapshots","finite candidate grammar, no guarantee of exhaustive optimal formations","no exacta odds; exacta uses cross-market signals and actual payout labels","no rider or lineup features in this first market-only formation experiment","not live validated"]})
