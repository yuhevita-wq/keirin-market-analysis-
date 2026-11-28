"""Stage 1: learn joint finish probabilities with a race-normalized proper loss. No betting policies."""
import csv,gzip,json,math,re,sys
from pathlib import Path
from collections import defaultdict,Counter
from itertools import combinations,permutations
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

OUT=Path("handoff_output"); OUT.mkdir(exist_ok=True)
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


assert len(races)>20000,("common cohort too small",len(races)) # minimum size guard, not fixed equality
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

# New features are restricted to race context, pair interactions and strictly earlier-day
# histories. Market residual learning uses log(q) as offset and proper conditional log loss.
# No payout or realized return is a training target.
from itertools import groupby
def pair_identity():
    d=pd.concat([pd.read_csv(SOURCE/f"rider_races_{y}.csv.gz",dtype=str,usecols=["race_id","car_no","provisional_player_key","class"]) for y in (2025,2026)],ignore_index=True)
    d=d[d.race_id.isin({r[1] for r in races})].copy()
    d["car"]=pd.to_numeric(d.car_no,errors="coerce")
    d=d[d.car.notna()]
    out={}
    for rid,g in d.groupby("race_id",sort=False):
        a={}
        for car,key,cls in zip(g.car,g.provisional_player_key,g["class"]):
            a[int(car)]=(str(key),str(cls)) if pd.notna(key) and str(key).strip() and pd.notna(cls) and str(cls).strip() else None
        out[rid]=a
    return out
def histkey(ids,a,b,ordered):
    ka,kb=ids.get(a),ids.get(b)
    if ka is None or kb is None or ka[0]==kb[0]:return None
    return (ka,kb) if ordered else tuple(sorted((ka,kb)))
def interaction_features(kind):
    identities=pair_identity()
    hp=defaultdict(lambda:np.zeros(3,dtype=np.float64))
    he=defaultdict(lambda:np.zeros(3,dtype=np.float64))
    context=[];history=[];hist_counts=Counter()
    names=["term_gap","style_same","front_pair","chase_pair","front_chase_pair","rate1_product","rate2_product","rate3_product","podium_history_product","rate1_gap","rate2_gap","rate3_gap","line_pos_gap","line_size_equal","market_pair_lift","market_podium_pair_lift","history_podium_gap","market_history_podium_disagreement"]
    hn=["prior_joint_log_count","prior_joint_observed_shrunk","prior_joint_market_shrunk","prior_joint_residual_shrunk"]
    if kind=="exacta":
        names+=["head_front_tail_chase","head_chase_tail_front","ordered_rate_product","head_line_pos_minus_tail","ordered_market_lift"]
        hn+=["prior_ordered_log_count","prior_ordered_observed_shrunk","prior_ordered_market_shrunk","prior_ordered_residual_shrunk"]
    idx={k:i for i,k in enumerate(rnames)}
    def val(rr,k):return float(rr[idx[k]])
    def read(h,k):
        if k is None or k not in h:return [0.,0.,0.,0.]
        n,y,q=h[k];return [math.log1p(n),y/(n+10),q/(n+10),(y-q)/(n+10)]
    for date,day in groupby(races,key=lambda r:r[0]):
        pending_p=defaultdict(lambda:np.zeros(3));pending_e=defaultdict(lambda:np.zeros(3))
        for _,rid,cars,(pair,trio,order,first,podium),_ in day:
            ids=identities.get(rid,{})
            rr=riders[rid]
            actual=labels[rid][1]
            tripod=defaultdict(float);joint=defaultdict(float)
            for t,q in trio.items():
                for a in t:tripod[a]+=q
                for a,b in combinations(t,2):joint[tuple(sorted((a,b)))]+=q
            ps={a:sum(val(rr[a],k) for k in ("rate1","rate2","rate3")) for a in cars}
            def rankmap(vals):return {a:i+1 for i,a in enumerate(sorted(cars,key=lambda a:(-vals[a],a)))}
            rh=rankmap(ps);rm=rankmap(podium)
            hist_first=rankmap({a:val(rr[a],"rate1") for a in cars});mf=rankmap(first)
            ctx=[];hh=[];pairs=list(permutations(cars,2) if kind=="exacta" else combinations(cars,2))
            pairfeatures={};pairhist={}
            for a,b in pairs:
                u,v=rr[a],rr[b];k=tuple(sorted((a,b)))
                ft,bt,ct=[val(u,x) for x in ("front","both","chase")]
                fs,bs,cs=[val(v,x) for x in ("front","both","chase")]
                f=[abs(val(u,"term")-val(v,"term")),ft*fs+bt*bs+ct*cs,ft*fs,ct*cs,ft*cs+ct*fs,
                   *[val(u,x)*val(v,x) for x in ("rate1","rate2","rate3")],ps[a]*ps[b],
                   *[abs(val(u,x)-val(v,x)) for x in ("rate1","rate2","rate3")],
                   abs(val(u,"line_position")-val(v,"line_position")),float(val(u,"line_size")==val(v,"line_size")),
                   (pair[a,b]+pair[b,a])/max(first[a]*first[b],1e-9),
                   joint[k]/max(tripod[a]*tripod[b],1e-9),abs(ps[a]-ps[b]),(rm[a]-rh[a])+(rm[b]-rh[b])]
                h=read(hp,histkey(ids,a,b,False))
                if kind=="exacta":
                    f += [ft*cs,ct*fs,val(u,"rate1")*val(v,"rate2"),val(u,"line_position")-val(v,"line_position"),pair[a,b]/max(first[a]*podium[b],1e-9)]
                    h += read(he,histkey(ids,a,b,True))
                pairfeatures[a,b]=np.array(f,dtype=np.float32);pairhist[a,b]=np.array(h,dtype=np.float32)
            tickets=list(permutations(cars,2) if kind=="exacta" else combinations(cars,3))
            for t in tickets:
                if kind=="exacta":
                    f=pairfeatures[t];h=pairhist[t]
                else:
                    p=np.stack([pairfeatures[x] for x in combinations(t,2)]);h0=np.stack([pairhist[x] for x in combinations(t,2)])
                    f=np.r_[p.mean(0),p.min(0),p.max(0)];h=np.r_[h0.mean(0),h0.min(0),h0.max(0)]
                rank=np.array([rm[a]-rh[a] for a in t]);frank=np.array([mf[a]-hist_first[a] for a in t])
                extra=[rank.mean(),rank.min(),rank.max(),frank.mean(),frank.min(),frank.max(),
                       np.mean([ps[a] for a in t]),np.mean([podium[a] for a in t]),np.std([rm[a]-rh[a] for a in cars]),
                       sum(val(rr[a],"front") for a in cars),sum(val(rr[a],"chase") for a in cars)]
                ctx.append(np.r_[f,extra]);hh.append(h)
            context.append(np.array(ctx,dtype=np.float32));history.append(np.array(hh,dtype=np.float32))
            # Stage all observations in pending maps, so even earlier races on the same day
            # cannot affect current features. No cross-class pair history is combined.
            for a,b in combinations(cars,2):
                k=histkey(ids,a,b,False)
                if k is not None:
                    hist_counts["joint_observations"]+=1
                    hist_counts["joint_prior_available"]+=int(k in hp)
                    pending_p[k]+=np.array([1.,float(a in actual and b in actual),joint[a,b]])
            if kind=="exacta":
                for a,b in permutations(cars,2):
                    k=histkey(ids,a,b,True)
                    if k is not None:
                        hist_counts["ordered_observations"]+=1
                        hist_counts["ordered_prior_available"]+=int(k in he)
                        pending_e[k]+=np.array([1.,float((a,b)==actual[:2]),pair[a,b]])
        for k,v in pending_p.items():hp[k]+=v
        for k,v in pending_e.items():he[k]+=v
    if kind=="trio":
        names=[prefix+n for prefix in ("mean_","min_","max_") for n in names]
        hn=[prefix+n for prefix in ("mean_","min_","max_") for n in hn]
    names+=["podium_rank_disagreement_mean","podium_rank_disagreement_min","podium_rank_disagreement_max",
            "first_rank_disagreement_mean","first_rank_disagreement_min","first_rank_disagreement_max",
            "mean_history_podium","mean_market_podium","field_rank_disagreement_sd","field_front_count","field_chase_count"]
    A=np.concatenate(context);H=np.concatenate(history)
    assert A.shape[1]==len(names) and H.shape[1]==len(hn) and np.isfinite(A).all() and np.isfinite(H).all()
    save("interaction_audit.json",{"current_features":names,"historical_features":hn,"coverage":dict(hist_counts),"history_prior_days_only":True,"prior_shrinkage_denominator":10,"line_identity_available":False})
    log({"interaction_coverage":dict(hist_counts),"current_features":len(names),"historical_features":len(hn)})
    return A,H,names,hn

from functools import lru_cache
from scipy.optimize import minimize
from scipy.stats import norm
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


# Freeze, verify and export only the already-selected final-fold models.
# No learning, no purchase recommendations, no deployment.
import hashlib,shutil
kind=sys.argv[1];assert kind in KINDS
chosen={"exacta":"tree_interaction_regularized","trio":"tree_history_regularized"}[kind]
base=Path("frozen_source")
def locate(name):
    paths=list(base.rglob(name));assert len(paths)==1,(name,paths);return paths[0]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
report=json.loads(locate("final_report.json").read_text())
assert report["next_model_configuration_from_all_known_folds"]==chosen
assert report["decision"]=="NOT_APPROVED_FOR_LIVE_USE"
X,M,Y,market,C,D,R,T,RN,MN=build(kind)
A,H,AN,HN=interaction_features(kind)
O=np.r_[0,np.cumsum(C)]
cohort_hash=hashlib.sha256("\n".join(f"{date}\t{rid}\t{count}" for date,rid,count in zip(D,R,C)).encode()).hexdigest()
assert report["cohort_sha256"]==cohort_hash
feature_names=MN+RN+AN+(HN if kind=="trio" else [])
model_path=locate(f"{kind}_{chosen}_15m.txt")
model=lgb.Booster(model_file=str(model_path))
assert model.num_feature()==len(feature_names)
assert model.current_iteration()==80
v=int(np.searchsorted(D,np.datetime64("2026-04-01")));z=int(np.searchsorted(D,np.datetime64("2026-07-01")))
a,b=int(O[v]),int(O[z]);counts=C[v:z]
matrix=np.column_stack([M[a:b],X[a:b],A[a:b]]+([H[a:b]] if kind=="trio" else []))
assert matrix.dtype==np.float32 and matrix.shape[1]==len(feature_names)
raw=model.predict(matrix,raw_score=True);q=market[a:b]
p=softmax(raw+np.log(q),counts)
ref=pd.read_csv(locate("selected_oof_15m.csv.gz"),dtype={"race_id":str,"date":str,"combination":str})
ref=ref[ref.model==chosen].reset_index(drop=True)
assert len(ref)==len(p)
assert ref.race_id.tolist()==np.repeat(R[v:z],counts).tolist()
assert ref.date.tolist()==np.repeat(D[v:z].astype(str),counts).tolist()
assert ref.combination.tolist()==T[a:b].tolist()
assert np.array_equal(ref.actual.to_numpy(),Y[a:b])
diff=float(np.max(np.abs(ref.probability.to_numpy()-p)))
assert diff<=1e-10,diff
# Reference full odds inputs for fixed, deterministic fixture subset.
# Every field size occurring in the last fold gets up to four examples.
chosen_races=[]
for size in sorted({len(races[j][2]) for j in range(v,z)}):
    ii=[j for j in range(v,z) if len(races[j][2])==size]
    chosen_races.extend(ii[k] for k in sorted(set(np.linspace(0,len(ii)-1,min(4,len(ii)),dtype=int))))
chosen_races=sorted(chosen_races)
ids=set(R[j] for j in chosen_races);books={}
with gzip.open(ROOT/"odds_only"/"odds_2026.jsonl.gz","rt",encoding="utf-8") as h:
    for line in h:
        o=json.loads(line)
        if o["race_id"] in ids:books[o["race_id"]]=o["odds"]
assert set(books)==ids
identity=pair_identity()
fixtures=[]
for j in chosen_races:
    aa,bb=int(O[j]-a),int(O[j+1]-a);rid=R[j];cars=races[j][2]
    ts,gg=grammar(cars,kind)
    if kind=="trio":
        odds_map={"=".join(map(str,sorted(combo(k,3)))):float(val) for k,val in books[rid]["3連複"].items()}
        price=np.array([odds_map[t] for t in T[a+aa:a+bb]])
        mode="direct_final_odds"
    else:
        price=1/q[aa:bb];mode="market_relative_proxy_not_monetary_odds"
    weights=np.column_stack([p[aa:bb],p[aa:bb]*price])
    fm=select(weights,gg,2 if kind=="exacta" else 1,8 if kind=="exacta" else 10)
    formations=[]
    for (family,k,col),idx in sorted(fm.items()):
        if family not in ("best_structured","unrestricted"):continue
        formations.append({"family":family,"points":int(k),"objective":"hit_max" if col==0 else "value_max",
          "tickets":[str(T[a+aa+int(i)]) for i in idx],"score":float(weights[idx,col].sum() if col==0 else weights[idx,col].mean())})
    fixtures.append({"race_id":rid,"date":str(D[j]),"cars":list(cars),"odds":books[rid],
         "rider_feature_names":rnames,"rider_features":{str(k):val.tolist() for k,val in riders[rid].items()},
         "rider_identity":{str(k):val for k,val in identity.get(rid,{}).items()},
         "combination":T[a+aa:a+bb].tolist(),"market_probability":q[aa:bb].tolist(),
         "interaction_features":A[a+aa:a+bb].tolist(),"pair_history_features":H[a+aa:a+bb].tolist() if kind=="trio" else None,
         "feature_matrix":matrix[aa:bb].tolist(),"raw_score":raw[aa:bb].tolist(),"probability":p[aa:bb].tolist(),
         "formation_score_mode":mode,"formations":formations})
with gzip.open(OUT/"golden_fixtures.json.gz","wt",encoding="utf-8") as h:json.dump(fixtures,h,ensure_ascii=False,allow_nan=False)
shutil.copy2(model_path,OUT/"model.txt")
shutil.copy2(locate("protocol.json"),OUT/"research_protocol.json")
shutil.copy2(locate("selection_log.json"),OUT/"selection_log.json")
shutil.copy2(locate("final_report.json"),OUT/"research_final_report.json")
shutil.copy2("research/ml_centered_20261128/train.py",OUT/"reference_train.py")
shutil.copy2(__file__,OUT/"replay.py")
schema={"kind":kind,"model_configuration":chosen,"feature_names_in_order":feature_names,"feature_count":len(feature_names),
 "dtype":"float32","candidate_order":"ascending sorted cars, itertools.permutations(cars,2)" if kind=="exacta" else "ascending sorted cars, itertools.combinations(cars,3)",
 "feature_blocks":["M","X","A"]+(["H"] if kind=="trio" else []),"formula":"race_softmax(raw_score + log(market_probability))",
 "training_end_exclusive":"2026-04-01","trained_races":v,"verified_replay_races":z-v,
 "probability_abs_tolerance":1e-10,"replay_max_absolute_difference":diff,"fixture_races":len(fixtures),
 "source_run":46093918439,"source_commit":"aec47b7b45f06b133a0cda11a543a5903d1da954",
 "odds_phase":"historical final only","historical_independent_holdout":False,
 "purchase_policy":"no qualified policy; do not use shadow diagnostics as recommendations",
 "known_limitations":["exacta q is trifecta marginal not exacta quote","no direct exacta purchase gate","same-line identity not available","history requires verified past results and same class; pair history requires complete historical matched cohort","no pre-close validation","the last fold model is not refit through June"]}
save("schema.json",schema)
manifest={"status":"verified_frozen_reproduction" if diff<=1e-10 else "failed","schema":schema,"files":{p.name:sha(p) for p in sorted(OUT.iterdir()) if p.is_file()}}
save("manifest.json",manifest);log({"handoff_manifest":manifest})
