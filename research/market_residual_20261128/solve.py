"""Stage 2: fixed probability artifact; deterministic formation optimization, no model fitting or purchase selection."""
import csv,gzip,json,math,re,sys
from pathlib import Path
from collections import defaultdict,Counter
from itertools import combinations,permutations
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

OUT=Path("market_residual_math_output"); OUT.mkdir(exist_ok=True)
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


from functools import lru_cache
from itertools import product
import hashlib
INPUT=Path("market_residual_output")
MODELS=[("market_raw","probability_raw"),("rider_only","probability_raw"),("market_adjusted","probability_raw"),("market_plus_rider","probability_raw"),("market_plus_rider","probability_calibrated"),("current_interactions","probability_raw"),("historical_interactions","probability_raw")]
MODEL_NAMES=["market_raw","rider_only","market_adjusted","market_plus_rider","market_plus_rider_calibrated","current_interactions","historical_interactions"]
MONTHS=(3,6,9,12,15)
def quote_book(raw,cars,kind):
    ordered=kind=="exacta";k=2 if ordered else 3
    expected=set(permutations(cars,k) if ordered else combinations(cars,k))
    b={}
    if not isinstance(raw,dict):return None
    for key,v in raw.items():
        c=combo(key,k)
        if c is None:return None
        if not ordered:c=tuple(sorted(c))
        try:v=float(v)
        except (ValueError,TypeError):return None
        if not math.isfinite(v) or v<1 or c in b:return None
        b[c]=v
    return b if set(b)==expected else None
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
def selftest():
    cars=tuple(range(1,6));ts,g=grammar(cars,"exacta")
    rng=np.random.default_rng(7);w=rng.random((len(ts),2))
    got=select(w,g,2,8)
    for c in range(2):
        assert np.isclose(w[got["unrestricted",2,c],c].sum(),max(w[list(t),c].sum() for t in combinations(range(len(ts)),2)))
        assert np.isclose(w[got["head",2,c],c].sum(),max(w[t,c].sum() for t in g["head",2]))
    ts,g=grammar(cars,"trio")
    assert len(ts)==10 and ("two_axis",3) in g and ("box",10) in g
    q=np.arange(1,11)/55
    assert np.isclose(q.sum(),1)
    print("mathematical optimizer self-test passed",flush=True)
selftest()
if "--self-test" in sys.argv:sys.exit(0)
kind=sys.argv[1]
assert kind in KINDS
files={}
for m in MONTHS:
    matches=list(INPUT.rglob(f"{kind}_{m}m_oof_probabilities.csv.gz"))
    assert len(matches)==1,(m,matches)
    files[m]=matches[0]
save("input_manifest.json",{"baseline_run":46005962698,"files":{str(m):{"name":p.name,"sha256":hashlib.sha256(p.read_bytes()).hexdigest()} for m,p in files.items()},"models":MODEL_NAMES,"note":"fixed OOF forecasts, no fit"})
pay,paudit=read_payments()
# Odds are looked up by race_id, never reconstructed from the winning payout.
# The exacta fallback uses the market-relative index p/q, NOT a monetary EV.
odds={}
audit=Counter()
for y in (2025,2026):
    with gzip.open(ROOT/"odds_only"/f"odds_{y}.jsonl.gz","rt",encoding="utf-8") as f:
        for line in f:
            obj=json.loads(line);rid=obj["race_id"];raw=obj["odds"]
            if rid not in pay[kind]:continue
            if rid in odds:raise ValueError("duplicate odds race "+rid)
            odds[rid]=raw.get(KINDS[kind])
            audit["raw_present" if odds[rid] else "raw_missing"]+=1
save("odds_audit_initial.json",dict(audit))
# For speed use vector accumulators indexed by deterministic policy key.
stats=defaultdict(lambda:np.zeros(7,dtype=np.float64))
def accumulate(key,hit,p_hit,expected,actual,regret):
    v=stats[key]
    v[0]+=1;v[1]+=hit;v[2]+=p_hit;v[3]+=expected;v[4]+=actual;v[5]+=regret;v[6]+=regret>1e-10
def output():
    rows=[]
    for key,a in sorted(stats.items()):
        months,src,model,obj,fam,k=key;n=int(a[0])
        rows.append({"bet":kind,"months":months,"odds_source":src,"model":model,"objective":obj,"family":fam,"points":k,"races":n,
                     "hit_rate":float(a[1]/n),"mean_predicted_hit":float(a[2]/n),
                     "mean_expected_return_pct":float(100*a[3]/n) if src=="direct" else None,
                     "mean_market_relative_index":float(a[3]/n) if src=="proxy" else None,
                     "actual_return_pct":float(100*a[4]/n),"mean_objective_regret":float(a[5]/n),"fraction_nonoptimal":float(a[6]/n)})
    pd.DataFrame(rows).to_csv(OUT/"formation_metrics.csv",index=False)
    return rows
for months,path in files.items():
    d=pd.read_csv(path,dtype={"race_id":str,"combination":str,"date":str})
    assert set(d.model)==set(m for m,_ in MODELS)
    base=d[d.model=="market_raw"].copy()
    assert not base.duplicated(["race_id","combination"]).any()
    key=["race_id","combination"]
    base=base.sort_values(key).reset_index(drop=True)
    aligned=[]
    for model,col in MODELS:
        a=d.loc[d.model==model,key+[col,"actual","date"]].sort_values(key).reset_index(drop=True)
        assert a[key].equals(base[key]) and np.array_equal(a.actual,base.actual) and np.array_equal(a.date,base.date)
        aligned.append(a[col].to_numpy(dtype=float))
    probs=np.column_stack(aligned);assert np.isfinite(probs).all() and (probs>0).all()
    out_rows=[];n_races=0;fold_audit=Counter()
    for rid,index in base.groupby("race_id",sort=False).indices.items():
        j=np.asarray(index);g=base.iloc[j];p=probs[j];n_races+=1
        assert np.allclose(p.sum(0),1,atol=1e-8) and g.actual.sum()==1
        t=[combo(s,2 if kind=="exacta" else 3) for s in g.combination]
        assert all(x is not None for x in t)
        cars=tuple(sorted({c for x in t for c in x}))
        canon,groups=grammar(cars,kind)
        lookup={x:i for i,x in enumerate(t)};assert len(lookup)==len(canon) and set(lookup)==set(canon)
        order=np.array([lookup[x] for x in canon]);p=p[order]
        actual=g.actual.to_numpy()[order].astype(bool)
        payment=pay[kind].get(rid,set());assert len(payment)==1
        winning,yen=next(iter(payment));assert canon[int(np.argmax(actual))]==winning
        raw=odds.get(rid)
        q=quote_book(raw,cars,kind)
        if q is not None:
            price=np.array([q[x] for x in canon]);src="direct"
            if kind=="trio":assert np.allclose((1/price)/(1/price).sum(),p[:,0],atol=1e-8)
        else:
            if kind=="trio":raise ValueError("trio direct odds missing/malformed "+rid)
            price=1/p[:,0];src="proxy"
        fold_audit[src]+=1
        # Weight columns: hit probability, then probability times observed odds.
        # For a proxy book these are relative value ratios, not expected yen.
        ev=p*price[:,None];w=np.column_stack([p,ev])
        sel=select(w,groups,2 if kind=="exacta" else 1,8 if kind=="exacta" else 10)
        for (fam,k,col),ids in sel.items():
            m=col%len(MODELS);obj="hit_max" if col<len(MODELS) else "value_max"
            hit=int(actual[ids].any());ph=float(p[ids,m].sum());ep=float(ev[ids,m].mean())
            gross=yen/100/k if hit else 0.
            ref=sel["unrestricted",k,col];regret=max(0.,float(w[ref,col].sum()-w[ids,col].sum()))/(1 if obj=="hit_max" else k)
            accumulate((months,src,MODEL_NAMES[m],obj,fam,k),hit,ph,ep,gross,regret)
            # Preserve two fixed model variants for detailed race-level audit,
            # all other variants remain in the full aggregate table.
            if fam in ("unrestricted","best_structured") and m in (0,3,5,6):
                out_rows.append((rid,g.date.iloc[0],months,src,MODEL_NAMES[m],obj,fam,k,
                                 " ".join(("-" if kind=="exacta" else "=").join(map(str,canon[i])) for i in ids),ph,ep,hit,gross))
    assert n_races=={3:4259,6:4707,9:2468,12:4454,15:4230}[months]
    pd.DataFrame(out_rows,columns=["race_id","date","months","odds_source","model","objective","family","points","tickets","predicted_hit","expected_return_or_proxy","actual_hit","actual_gross"]).to_csv(OUT/f"race_formations_{months}m.csv.gz",index=False,compression="gzip")
    rows=output()
    log({"fold":{"bet":kind,"months":months,"races":n_races,"odds_sources":dict(fold_audit)}})
    # Log compact fixed-point comparison; all sizes and families in CSV.
    for r in rows:
        if r["months"]==months and r["model"] in ("market_raw","market_plus_rider","current_interactions","historical_interactions") and r["family"] in ("unrestricted","best_structured") and r["points"] in (2,3,4,6):
            log({"formation":r})
    del d,base,aligned,probs,out_rows
# Pooled rows use sums, never select a winning historical point count.
pooled=defaultdict(lambda:np.zeros(7))
for key,a in stats.items():
    months,*rest=key
    if months>=6:pooled[tuple(rest)]+=a
summ=[]
for key,a in sorted(pooled.items()):
    src,model,obj,fam,k=key;n=int(a[0])
    r={"bet":kind,"odds_source":src,"model":model,"objective":obj,"family":fam,"points":k,"races":n,
       "hit_rate":float(a[1]/n),"predicted_hit":float(a[2]/n),"actual_return_pct":float(100*a[4]/n),
       "expected_return_pct":float(100*a[3]/n) if src=="direct" else None,"market_relative_index":float(a[3]/n) if src=="proxy" else None,
       "mean_objective_regret":float(a[5]/n),"fraction_nonoptimal":float(a[6]/n)}
    summ.append(r)
    if model in ("market_raw","market_plus_rider","current_interactions","historical_interactions") and fam in ("unrestricted","best_structured"):log({"pooled":r})
save("pooled_metrics.json",summ)
save("status.json",{"status":"completed","stage":"2 mathematical formation only","baseline_run":46005962698,"purchase_gate":False,"model_training":False,
"limitations":["reused historical time period","final odds only","exacta direct odds may be unavailable; proxy index not expected monetary return","equal notional stake per race; no integer stake rounding","family grammar specified by researcher","point count treated as fixed comparison dimension, not selected from test outcomes","no statistical testing or profit guarantee"]})
