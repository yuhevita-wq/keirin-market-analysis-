"""New experiment: learn expected realized return of formation actions, not winning-ticket ranks."""
import csv,gzip,json,math,re,sys
from pathlib import Path
from collections import defaultdict,Counter
from itertools import combinations,permutations
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

OUT=Path("exacta_iterative_output"); OUT.mkdir(exist_ok=True)
ROOT=Path("data"); SOURCE=ROOT/"rider_condition_2025_202610_full"
SEED=20261127
KINDS={"exacta":"2車単"}
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

from lightgbm import LGBMClassifier
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from itertools import product
# New exacta-only experiment. No old formation or gate is imported.
WINDOWS=[(3,"2025-04-01","2025-07-01"),(6,"2025-07-01","2025-10-01"),(9,"2025-10-01","2026-01-01"),(12,"2026-01-01","2026-04-01"),(15,"2026-04-01","2026-07-01")]
OLD_ALL=[76.19672066995382,77.3838963246229,79.48396137223122,85.88850845681785,78.63685973207251]
OLD_RIDGE=[None,94.82701652089406,89.51712328767124,76.01686507936509,88.88888888888887]
# Candidate templates are a search space, not fixed purchase rules. All use 2-8 distinct tickets.
def candidate_actions(cars,pair,rider):
    n=len(cars);pairs=list(permutations(cars,2));idx={p:i for i,p in enumerate(pairs)}
    mp=np.array([pair[t] for t in pairs],dtype=float)
    # class-specific history (smoothed marginal place rates); not a calibrated joint probability.
    hp=np.array([rider[a][0]*rider[b][1] for a,b in pairs],dtype=float)
    hp/=hp.sum()
    scores=[mp,hp,np.sqrt(mp*hp)]
    for i in range(len(scores)):scores[i]=scores[i]/scores[i].sum()
    actions={}
    def add(f,src,ts):
        inds=tuple(sorted({idx[t] for t in ts if t in idx}))
        if 2<=len(inds)<=8 and inds not in actions:actions[inds]=(f,src)
    for si,score in enumerate(scores):
        order=sorted(cars,key=lambda a:(-sum(score[idx[a,b]] for b in cars if b!=a),a))
        # market explores every axis; alternative rankings explore four axes
        for a in (cars if si==0 else order[:4]):
            for f in range(3):
                def val(b):
                    return score[idx[a,b]] if f==0 else score[idx[b,a]] if f==1 else score[idx[a,b]]+score[idx[b,a]]
                partners=sorted((b for b in cars if b!=a),key=lambda b:(-val(b),b))
                for k in ((2,3,4) if f<2 else (1,2,3)):
                    if k>len(partners):continue
                    add(f,si,[t for b in partners[:k] for t in ([(a,b)] if f==0 else [(b,a)] if f==1 else [(a,b),(b,a)])])
        for k in (2,3,4,6,8):
            add(3,si,[pairs[i] for i in sorted(range(len(pairs)),key=lambda i:(-score[i],pairs[i]))[:k]])
        for axes in combinations(order[:3],2):
            for f in (4,5):
                others=sorted(cars,key=lambda b:(-sum(score[idx[t]] for a in axes for t in [(a,b) if f==4 else (b,a)] if t in idx),b))
                for k in (2,3):
                    add(f,si,[(a,b) if f==4 else (b,a) for a in axes for b in others[:k]])
        for group in combinations(order[:4],3):
            add(6,si,list(permutations(group,2)))
    return pairs,mp,hp,actions
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
def action_features(cars,mb,rider):
    pair,trio,_,first,podium=mb
    pairs,mp,hp,actions=candidate_actions(cars,pair,rider)
    assert actions
    ix=list(actions);n=len(cars)
    A=np.zeros((len(ix),len(pairs)),dtype=np.float32)
    for j,inds in enumerate(ix):A[j,list(inds)]=1
    points=A.sum(1)
    head=np.stack([rider[a] for a,b in pairs]);tail=np.stack([rider[b] for a,b in pairs])
    pm=np.column_stack([head,tail,head[:,:7]-tail[:,:7],
                        [first[a] for a,b in pairs],[first[b] for a,b in pairs],
                        [podium[a] for a,b in pairs],[podium[b] for a,b in pairs],
                        np.log(np.maximum(mp,1e-10)),np.log(np.maximum(hp,1e-10))]).astype(np.float32)
    avg=(A@pm)/points[:,None]
    support=A@mp; hist_support=A@hp
    maxp=np.where(A>0,mp[None,:],0).max(1)
    minp=np.where(A>0,mp[None,:],np.inf).min(1)
    family=np.array([actions[i][0] for i in ix])
    src=np.array([actions[i][1] for i in ix])
    nh=np.array([len({pairs[i][0] for i in inds}) for inds in ix])
    nt=np.array([len({pairs[i][1] for i in inds}) for inds in ix])
    overlap=np.array([len({pairs[i][0] for i in inds}&{pairs[i][1] for i in inds}) for inds in ix])
    rv=np.stack([rider[c] for c in cars]);m_sorted=np.sort(mp)[::-1]
    context=np.r_[n,-np.sum(mp*np.log(mp))/np.log(len(mp)),m_sorted[:5],
                  np.sort(first.values() if False else list(first.values()))[::-1][:3],
                  np.sort(list(podium.values()))[::-1][:3],rv[:,:7].mean(0),rv[:,:7].std(0)]
    X=np.column_stack([np.eye(7)[family],np.eye(3)[src],points,nh,nt,overlap,support,hist_support,maxp,minp,
                       support/points,hist_support/points,avg,np.tile(context,(len(ix),1))]).astype(np.float32)
    return X,ix,pairs,family,points
def smoke():
    cars=tuple(range(1,8));pair={t:1/42 for t in permutations(cars,2)}
    trio={t:1/35 for t in combinations(cars,3)}
    first={a:1/7 for a in cars};podium={a:3/7 for a in cars}
    r={a:np.r_[np.full(3,1/7),0.,100.,1.,3.,np.zeros(12)].astype(np.float32) for a in cars}
    X,ix,ps,fs,pts=action_features(cars,(pair,trio,list(cars),first,podium),r)
    assert len(ix)>20 and len(ix)==len(set(ix)) and np.isfinite(X).all()
    assert np.all((pts>=2)&(pts<=8))
    assert np.asarray(["2025-04-01","2025-06-30"],dtype="datetime64[D]").max()<np.datetime64("2025-07-01","D")
    print("exacta redesign self-test passed",flush=True)
smoke()
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
            if p:races.append((date,rid,cars,mb,p))
races.sort(key=lambda x:(x[0],x[1]))
assert len({x[1] for x in races})==len(races)
save("data_audit.json",{"payout_sources":paudit,"verified_entries":len(labels),"matched_races":{k:sum(v.values()) for k,v in monthly.items()},"monthly":{k:dict(v) for k,v in monthly.items()},"excluded":dict(audit),"odds":"final trifecta and trio; exacta market unavailable; exacta pair probabilities are trifecta marginals, not exacta odds","reward":"actual paid yen per 100 yen divided by action points; equal notional stake per race; skip net zero"})
log({"monthly":{k:dict(v) for k,v in monthly.items()},"excluded":dict(audit)})

assert len(races)==24666,("source coverage changed",len(races))
rider,feature_names=rider_history()
save("feature_audit.json",{"rider_features":feature_names,"history":"same class, verified prior calendar days only","line_data":"position, size; no line identity, no invented same-line indicator","min_points":2,"max_points":8,"exacta_odds":"unavailable; trifecta marginal market input","payout":"actual exacta payout labels"})
# Build compact feature blocks with no Python float list per action.
blocks=[];ys=[];hs=[];ps=[];fs=[];offset=[0];D=[];R=[]
for date,rid,cars,mb,p in races:
    if "exacta" not in p:continue
    rr=rider[rid]
    assert set(rr)==set(cars),(rid,set(rr),set(cars))
    x,ix,pairs,family,points=action_features(cars,mb,rr)
    target,yen=p["exacta"]; j=pairs.index(target)
    hit=np.array([j in inds for inds in ix],dtype=bool)
    y=(hit.astype(np.float32)*(yen/100)/points).astype(np.float32)
    blocks.append(x);ys.append(y);hs.append(hit);ps.append(points.astype(np.int8));fs.append(family.astype(np.int8))
    offset.append(offset[-1]+len(y));D.append(date);R.append(rid)
X=np.concatenate(blocks);del blocks
Y=np.concatenate(ys);H=np.concatenate(hs);P=np.concatenate(ps);FAM=np.concatenate(fs)
del ys,hs,ps,fs,rider,races
O=np.asarray(offset,dtype=np.int64);D=np.asarray(D,dtype="datetime64[D]")
log({"eligible_races":len(D),"candidate_actions":len(Y),"features":X.shape[1]})
assert len(D)==24666 and len(X)==len(Y) and np.isfinite(X).all()
def reg(n_estimators=100,leaves=23,min_child=200):
    return LGBMRegressor(objective="regression",n_estimators=n_estimators,learning_rate=.05,num_leaves=leaves,min_child_samples=min_child,reg_lambda=10,n_jobs=4,verbosity=-1,random_state=SEED,deterministic=True,force_col_wise=True)
def cls():
    return LGBMClassifier(objective="binary",n_estimators=100,learning_rate=.05,num_leaves=23,min_child_samples=200,reg_lambda=10,n_jobs=4,verbosity=-1,random_state=SEED,deterministic=True,force_col_wise=True)
def metrics(pick,day):
    valid=pick>=0;sel=pick[valid];n=len(day);k=len(sel)
    ret=Y[sel].astype(float);gross=float(ret.sum())
    _,inv=np.unique(day,return_inverse=True)
    gr=np.zeros(n);gr[valid]=ret
    dg=np.bincount(inv,weights=gr);ds=np.bincount(inv,weights=valid.astype(float))
    count=np.bincount(inv)
    rng=np.random.default_rng(SEED)
    samp=rng.integers(0,len(dg),size=(1000,len(dg)))
    g=dg[samp].sum(1);s=ds[samp].sum(1);den=count[samp].sum(1)
    ci=(100*np.quantile(g[s>0]/s[s>0],[.025,.975])).tolist() if (s>0).any() else None
    return {"test_races":n,"bought":k,"coverage":k/n,"return_pct":100*gross/k if k else None,
            "hit_rate":float(H[sel].mean()) if k else None,"mean_points":float(P[sel].mean()) if k else None,
            "net_per_eligible_race":(gross-k)/n,"net_ci95":np.quantile((g-s)/den,[.025,.975]).tolist(),
            "roi_ci95":ci,"family_counts":{str(i):int(c) for i,c in Counter(FAM[sel]).items()}}

# Five successive generations: continuation of both the formation booster
# and purchase decision booster. The exact same chronological folds, features,
# candidate sets and actual payout labels are preserved.
# No test result is used as a training label until it is chronologically past.
GENS=5
BASE_ALL=[73.36620508357935,84.17190719725905,77.93256991710423,76.6887067493886,81.69580351622956]
BASE_GATE=[None,66.52216235317672,79.42726846346864,75.28768265777015,102.72479135651913]
def continuation(stage):
    return LGBMRegressor(objective="regression",n_estimators=40,
        learning_rate=.015 if stage=="formation" else .02,
        num_leaves=15 if stage=="formation" else 11,
        min_child_samples=400 if stage=="formation" else 250,
        reg_lambda=20,n_jobs=4,verbosity=-1,random_state=SEED,
        deterministic=True,force_col_wise=True)
# Small execution-path test exercises init_model, feature shapes and date safety.
def continuation_test():
    rng=np.random.default_rng(SEED)
    a=rng.normal(size=(256,5)); y=np.maximum(0,a[:,0]+.2)
    m=LGBMRegressor(n_estimators=3,min_child_samples=10,verbosity=-1,n_jobs=2,random_state=SEED)
    m.fit(a,y)
    n=LGBMRegressor(n_estimators=2,min_child_samples=10,verbosity=-1,n_jobs=2,random_state=SEED)
    n.fit(a,y,init_model=m.booster_)
    assert n.booster_.num_trees()==5
    assert np.isfinite(n.predict(a)).all()
continuation_test()
records=[];decisions=[]
history={g:[] for g in range(1,GENS+1)}
past_net={g:0. for g in history}
past_gate_quarters=0
previous_reference_ok=True
for step,(months,end,test_end) in enumerate(WINDOWS):
    v=int(np.searchsorted(D,np.datetime64(end,"D")));z=int(np.searchsorted(D,np.datetime64(test_end,"D")))
    assert z-v==[4259,4707,2468,4454,4230][step]
    a,b=int(O[v]),int(O[z]);weights=np.repeat(1/np.diff(O[:v+1]),np.diff(O[:v+1]))
    # Auxiliary features are exactly the prior direct-model pipeline.
    hit_model=cls();hit_model.fit(X[:a],H[:a],sample_weight=weights)
    mask=H[:a];payout=reg();payout.fit(X[:a][mask],Y[:a][mask],sample_weight=weights[mask])
    ph=hit_model.predict_proba(X[a:b])[:,1]
    pg=np.maximum(payout.predict(X[a:b]),0)
    del hit_model,payout
    prior_formation=None;prior_gate=None;fold_gate={}
    # Generation selector is decided before observing the quarter. A no-buy
    # comparator is eligible; this is separate from the reported fixed variants.
    selected_gen=max(past_net,key=past_net.get) if past_gate_quarters and max(past_net.values())>0 else 0
    for gen in range(1,GENS+1):
        m=reg() if gen==1 else continuation("formation")
        m.fit(X[:a],Y[:a],sample_weight=weights,init_model=prior_formation)
        prior_formation=m.booster_
        score=np.asarray(m.predict(X[a:b]),dtype=float);assert np.isfinite(score).all()
        chosen=[];features=[];base_scores=[]
        for r in range(v,z):
            s,t=int(O[r]),int(O[r+1]);q=score[s-a:t-a]
            i=s+int(np.argmax(q));chosen.append(i);best=float(q.max());base_scores.append(best)
            ordered=np.sort(q)
            features.append(np.r_[X[i],best,float(ordered[-1]-ordered[-2]),float(q.mean()),float(q.std()),float(ph[i-a]),float(pg[i-a]),float(score[i-a])])
        pick=np.asarray(chosen,dtype=int);G=np.asarray(features,dtype=np.float64);base=np.asarray(base_scores)
        assert np.isfinite(G).all()
        past=history[gen];train_n=sum(len(t[1]) for t in past)
        def emit(policy,selection,gate_trees=None):
            row={"generation":gen,"policy":policy,"formation_months":months,
                "formation_train_races":v,"gate_train_races":train_n,"test_start":end,"test_end_exclusive":test_end,
                "formation_trees":int(m.booster_.num_trees()),"gate_trees":gate_trees,
                **metrics(selection,D[v:z])}
            records.append(row);log(row);return row
        all_row=emit("all",pick)
        if gen==1:
            # Prior result uses float32 labels; equality up to numerical roundoff.
            assert abs(all_row["return_pct"]-BASE_ALL[step])<1e-6,("generation1 all mismatch",months,all_row["return_pct"])
        emit("raw_gt_1",np.where(base>1,pick,-1))
        gate_scores=None
        if past:
            GX=np.concatenate([t[0] for t in past]);GY=np.concatenate([t[1] for t in past]);GD=np.concatenate([t[2] for t in past])
            assert GD.max()<D[v]
            gate=reg(80,15,150) if gen==1 else continuation("gate")
            gate.fit(GX,GY,init_model=prior_gate)
            prior_gate=gate.booster_
            gate_scores=np.asarray(gate.predict(G),dtype=float);assert np.isfinite(gate_scores).all()
            selected=np.where(gate_scores>1,pick,-1)
            gate_row=emit("learned_gate",selected,int(gate.booster_.num_trees()))
            if gen==1:
                assert abs(gate_row["return_pct"]-BASE_GATE[step])<1e-6,("generation1 gate mismatch",months,gate_row["return_pct"])
            fold_gate[gen]=(selected,gate_row)
        for j,r in enumerate(range(v,z)):
            i=pick[j]
            entry={"generation":gen,"formation_months":months,"race_id":R[r],"date":str(D[r]),
                "family":int(FAM[i]),"points":int(P[i]),"actual_gross":float(Y[i]),"base_score":float(base[j])}
            if gate_scores is not None:
                entry["gate_score"]=float(gate_scores[j]);entry["gate_buy"]=bool(gate_scores[j]>1)
            decisions.append(entry)
        history[gen].append((G,Y[pick].copy(),D[v:z].copy()))
        save("learning_stages.json",records)
        pd.DataFrame(decisions).to_csv(OUT/"oos_decisions.csv",index=False)
    if fold_gate:
        selected=fold_gate[selected_gen][0] if selected_gen else np.full(z-v,-1,dtype=int)
        row={"generation":selected_gen,"policy":"past_only_generation_selection","formation_months":months,
             "past_gate_quarters":past_gate_quarters,"test_start":end,"test_end_exclusive":test_end,
             **metrics(selected,D[v:z])}
        records.append(row);log(row)
        for gen,(_,r) in fold_gate.items():past_net[gen]+=r["net_per_eligible_race"]*(z-v)
        past_gate_quarters+=1
    save("learning_stages.json",records)
    del ph,pg,score,prior_formation,prior_gate
# Pool nonoverlapping OOF quarters with same notional stake per purchased race.
summary=[]
for gen in range(1,GENS+1):
    for policy in ("all","raw_gt_1","learned_gate"):
        rr=[x for x in records if x["generation"]==gen and x["policy"]==policy and x["formation_months"]>=6]
        bought=sum(x["bought"] for x in rr);eligible=sum(x["test_races"] for x in rr)
        gross=sum(x["bought"]*(x["return_pct"] or 0)/100 for x in rr)
        hits=sum(x["bought"]*(x["hit_rate"] or 0) for x in rr)
        row={"generation":gen,"policy":policy,"quarters":len(rr),"eligible":eligible,"bought":bought,
            "return_pct":100*gross/bought if bought else None,"hit_rate":hits/bought if bought else None,
            "net_per_eligible_race":(gross-bought)/eligible if eligible else None}
        summary.append(row);log({"pooled":row})
save("pooled_results.json",summary)
save("status.json",{"status":"completed","generations":GENS,"prior_source_commit":"e2c4c1a2c9969fbcdd827e1c0275f2b6e1ece2db","generation1_reproduced":True,
    "limitations":["same data reused, not a new independent holdout","no direct exacta odds or verified preclose snapshot","candidate search space and data unchanged","extra boosting can overfit; no claim of improvement before all quarters are evaluated","3連複 and production untouched"]})
