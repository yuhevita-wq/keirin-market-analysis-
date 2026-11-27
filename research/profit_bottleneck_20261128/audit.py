"""Stage 2: fixed probability artifact; deterministic formation optimization, no model fitting or purchase selection."""
import csv,gzip,json,math,re,sys
from pathlib import Path
from collections import defaultdict,Counter
from itertools import combinations,permutations
import numpy as np
import pandas as pd


OUT=Path("bottleneck_output"); OUT.mkdir(exist_ok=True)
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



"""Retrospective accounting audit; no fitting, optimization, or new purchase rules."""
import hashlib
from scipy.stats import norm
from functools import lru_cache
BASE=Path("frozen_artifact")
MONTHS=(3,6,9,12,15);MODELS=("market_plus_rider","current_interactions","historical_interactions")
ALL_MODELS=("market_raw",)+MODELS
SEED=20261128;BOOT=2000
def json_default(o):
    if isinstance(o,np.generic):return o.item()
    if isinstance(o,np.ndarray):return o.tolist()
    raise TypeError(type(o).__name__)
def save(name,obj):
    p=OUT/name;tmp=p.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False,default=json_default),encoding="utf-8");tmp.replace(p)
def log(obj):print(json.dumps(obj,ensure_ascii=False,allow_nan=False,default=json_default),flush=True)
def file(name):
    p=list(BASE.rglob(name));assert len(p)==1,(name,p);return p[0]
def interval(y,dates,den=None,week=False):
    y=np.asarray(y,dtype=float);dates=pd.to_datetime(np.asarray(dates,dtype=str))
    if week:dates=dates.to_period("W").astype(str)
    den=np.ones(len(y)) if den is None else np.asarray(den,dtype=float)
    assert len(y)==len(dates)==len(den) and np.isfinite(y).all() and np.isfinite(den).all()
    if not len(y) or den.sum()<=0:return {"mean":None,"ci95":None,"clusters":0}
    _,inv=np.unique(dates,return_inverse=True)
    a=np.bincount(inv,weights=y);b=np.bincount(inv,weights=den);n=len(a)
    rng=np.random.default_rng(SEED);i=rng.integers(0,n,size=(BOOT,n));d=b[i].sum(1);valid=d>0
    ci=np.quantile(a[i].sum(1)[valid]/d[valid],[.025,.975]).tolist() if valid.any() else None
    return {"mean":float(a.sum()/b.sum()),"ci95":ci,"clusters":n}
def quote(raw,cars,kind):
    k=2 if kind=="exacta" else 3
    expected=set(permutations(cars,2) if k==2 else combinations(cars,3))
    if not isinstance(raw,dict):return None
    b={}
    for key,v in raw.items():
        c=combo(key,k)
        if c is None:return None
        if k==3:c=tuple(sorted(c))
        try:v=float(v)
        except (ValueError,TypeError):return None
        if not math.isfinite(v) or v<1 or c in b:return None
        b[c]=v
    return b if set(b)==expected else None
def attribution(p,q,price,actual,paid):
    # mutually exclusive outcomes, equal notional stake per race
    n=len(p);assert n>0
    market=float(np.dot(q,price)/n);pred=float(np.dot(p,price)/n)
    observed_quote=float(np.dot(actual,price)/n)
    actual_paid=float(np.sum(actual)*paid/n)
    terms=np.array([market-1,pred-market,observed_quote-pred,actual_paid-observed_quote])
    assert np.isclose(terms.sum(),actual_paid-1,atol=1e-9)
    return terms,pred,actual_paid
def switch_decomposition(a,b,ga,gb):
    # net per eligible race; symmetric algebraic decomposition of fixed
    # historical actions and fixed diagnostic masks, not causal effects.
    ga=np.asarray(ga,dtype=float);gb=np.asarray(gb,dtype=float)
    a=np.asarray(a,dtype=float)-1;b=np.asarray(b,dtype=float)-1
    construction=(ga+gb)/2*(b-a);selection=(gb-ga)*(a+b)/2
    total=gb*b-ga*a
    assert np.allclose(construction+selection,total)
    return construction,selection,total
def selftest():
    t,p,r=attribution(np.array([.2,.1]),np.array([.15,.1]),np.array([5.,7.]),np.array([1.,0.]),6.)
    assert np.isclose(p,.85) and r==3 and np.isclose(t.sum(),2)
    a,b,c=switch_decomposition([2,0,3],[0,4,2],[1,0,1],[0,1,1])
    assert np.allclose(a+b,c)
    assert interval([1,1],["2025-01-01","2025-01-02"])["ci95"]==[1.,1.]
    print("accounting and decomposition self-test passed",flush=True)
selftest()
if "--self-test" in sys.argv:sys.exit(0)
kind=sys.argv[1];assert kind in ("exacta","trio")
pay,paudit=read_payments()
odds={};odds_raw=Counter()
for y in (2025,2026):
    with gzip.open(ROOT/"odds_only"/f"odds_{y}.jsonl.gz","rt",encoding="utf-8") as f:
        for line in f:
            o=json.loads(line);rid=o["race_id"]
            if rid not in pay[kind]:continue
            assert rid not in odds
            raw=o["odds"].get(KINDS[kind]);odds[rid]=raw
            odds_raw["present" if raw else "missing"]+=1
# Fixed score bands selected before this audit; do not optimize or rank thresholds.
RATIO_BINS=[0,.8,.95,1.05,1.2,1.5,2.,float("inf")]
PRICE_BINS=[0,5,10,20,50,100,300,float("inf")]
EV_BINS=[0,.8,.9,1.,1.1,1.2,1.5,float("inf")]
def binname(v,bins):
    i=max(0,min(len(bins)-2,int(np.searchsorted(bins,v,side="right")-1)))
    return f"{bins[i]:g}–{bins[i+1]:g}"
def band(v,bins):
    ix=np.clip(np.searchsorted(bins,v,side="right")-1,0,len(bins)-2)
    return np.array([f"{bins[i]:g}–{bins[i+1]:g}" for i in ix])
manifest={};fparts=[];cparts=[];audit=Counter();race_quality=[]
for months in MONTHS:
    pp=file(f"{kind}_{months}m_oof_probabilities.csv.gz");ff=file(f"race_formations_{months}m.csv.gz")
    manifest[str(months)]={"probabilities":hashlib.sha256(pp.read_bytes()).hexdigest(),"formations":hashlib.sha256(ff.read_bytes()).hexdigest()}
    d=pd.read_csv(pp,dtype={"race_id":str,"date":str,"combination":str})
    f=pd.read_csv(ff,dtype={"race_id":str,"date":str,"tickets":str})
    f=f[(f.family=="best_structured") & f.model.isin(ALL_MODELS)].copy()
    # Full-coverage fixed point sizes only; avoid rare structures.
    sizes=(2,3,4,6,8) if kind=="exacta" else (1,2,3,4,6,9,10)
    f=f[f.points.isin(sizes)]
    base=d[d.model=="market_raw"].sort_values(["race_id","combination"]).reset_index(drop=True)
    assert not base.duplicated(["race_id","combination"]).any()
    probs=[]
    for m in ALL_MODELS:
        a=d[d.model==m].sort_values(["race_id","combination"]).reset_index(drop=True)
        assert a[["race_id","combination","actual","date"]].equals(base[["race_id","combination","actual","date"]])
        probs.append(a.probability_raw.to_numpy(float))
    P=np.column_stack(probs);del d,a,probs
    assert np.isfinite(P).all() and (P>0).all()
    fg=f.groupby("race_id",sort=False).indices
    result=[];candidate=[];n=0
    for rid,ii in base.groupby("race_id",sort=False).indices.items():
        n+=1;g=base.iloc[ii];p=P[ii];q=p[:,0];actual=g.actual.to_numpy(float);date=g.date.iloc[0]
        assert actual.sum()==1 and np.allclose(p.sum(0),1,atol=1e-8)
        t=g.combination.tolist();ix={x:j for j,x in enumerate(t)}
        tuples=[combo(x,2 if kind=="exacta" else 3) for x in t];cars=tuple(sorted({c for x in tuples for c in x}))
        payments=pay[kind][rid];assert len(payments)==1
        win,yen=next(iter(payments));assert tuples[np.argmax(actual)]==win
        paid=yen/100
        direct=quote(odds.get(rid),cars,kind)
        if direct is not None:
            price=np.array([direct[x] for x in tuples]);src="direct"
            assert np.allclose(q,(1/price)/(1/price).sum(),atol=1e-8) if kind=="trio" else True
            rq=float(q@price/len(q))
            audit["direct"]+=1
            race_quality.append((rid,date,months,src,rq,float(price[np.argmax(actual)]),paid))
        else:
            assert kind=="exacta"
            price=None;src="proxy";audit["proxy"]+=1
        gf=f.iloc[fg[rid]]
        assert len(gf)==len(sizes)*2*len(ALL_MODELS)
        for r in gf.itertuples(index=False):
            mi=ALL_MODELS.index(r.model);ids=np.array([ix[x] for x in r.tickets.split()])
            assert len(ids)==r.points and len(set(ids))==r.points
            pp0=p[ids,mi];qq=q[ids];yy=actual[ids]
            assert np.isclose(pp0.sum(),r.predicted_hit,atol=1e-9)
            assert np.isclose(yy.sum(),r.actual_hit) and np.isclose(yy.sum()*paid/r.points,r.actual_gross)
            ratio=pp0/qq
            record={"race_id":rid,"date":date,"months":months,"source":src,"model":r.model,"objective":r.objective,"points":int(r.points),
                    "predicted_hit":float(pp0.sum()),"market_hit":float(qq.sum()),"actual_hit":int(yy.sum()),"gross":float(r.actual_gross),
                    "score":float(r.expected_return_or_proxy),"mean_ratio":float(ratio.mean()),"mean_market_probability":float(qq.mean())}
            if price is not None:
                terms,ev,ar=attribution(pp0,qq,price[ids],yy,paid)
                assert np.isclose(ev,r.expected_return_or_proxy,atol=1e-8)
                record.update(dict(zip(("market_drag","model_increment","probability_error","settlement_difference"),terms)))
            else:
                assert np.isclose(ratio.mean(),r.expected_return_or_proxy,atol=1e-8)
            result.append(record)
        # Candidate diagnostics on the same race, not using payout to pick tickets.
        # All candidates, plus selected candidates from previously saved two-point
        # value-max formation. Compare with all remaining candidates at same bins.
        for mi,model in enumerate(ALL_MODELS):
            picked=gf[(gf.model==model)&(gf.objective=="value_max")&(gf.points==2)]
            assert len(picked)==1
            sel=np.zeros(len(q),dtype=int)
            for x in picked.tickets.iloc[0].split():sel[ix[x]]=1
            ratio=p[:,mi]/q
            cp=pd.DataFrame({"date":date,"months":months,"model":model,"selected":sel,
                             "ratio_band":band(ratio,RATIO_BINS),"price_band":band(price,PRICE_BINS) if price is not None else "unavailable",
                             "n":1,"p_sum":p[:,mi],"q_sum":q,"y_sum":actual})
            if price is not None:
                cp["p_price"]=p[:,mi]*price;cp["q_price"]=q*price;cp["y_price"]=actual*price
            candidate.append(cp)
    assert n=={3:4259,6:4707,9:2468,12:4454,15:4230}[months]
    fparts.append(pd.DataFrame(result))
    cb=pd.concat(candidate,ignore_index=True)
    cparts.append(cb.groupby(["date","months","model","selected","ratio_band","price_band"],as_index=False).sum(numeric_only=True))
    log({"fold":{"bet":kind,"months":months,"races":n}})
    del base,P,f,fg,result,candidate,cb
save("manifest.json",{"source_run":46024184584,"files":manifest,"payout_files":paudit,"odds_raw":dict(odds_raw),"matched_odds":dict(audit)})
F=pd.concat(fparts,ignore_index=True);C=pd.concat(cparts,ignore_index=True)
F.to_csv(OUT/"race_accounting.csv.gz",index=False,compression="gzip")
C.to_csv(OUT/"candidate_calibration_daily.csv.gz",index=False,compression="gzip")
pool=F[F.months>=6].copy();assert pool.race_id.nunique()==15859
rows=[];components=[];fixed=[];switches=[];bands=[]
def describe(g):
    n=len(g)
    if not n:return {"n":0}
    a={"n":n,"roi":float(g.gross.mean()),"score":float(g.score.mean()),"hit_pred":float(g.predicted_hit.mean()),
       "hit_obs":float(g.actual_hit.mean()),"hit_gap":interval(g.actual_hit-g.predicted_hit,g.date),
       "roi_interval":interval(g.gross,g.date),"roi_interval_week":interval(g.gross,g.date,week=True),
       "positive_quarters":int((g.groupby("months").gross.mean()>1).sum()),
       "eligible_net":float((g.gross-1).sum()/15859)}
    if kind=="trio":
        a["accounting"]={k:interval(g[k],g.date) for k in ("market_drag","model_increment","probability_error","settlement_difference")}
        assert np.isclose(sum(v["mean"] for v in a["accounting"].values()),a["roi"]-1)
    return a
for (model,objective,k),g in pool.groupby(["model","objective","points"]):
    row={"bet":kind,"model":model,"objective":objective,"points":int(k),"subset":"all",**describe(g)}
    rows.append(row)
    if model in MODELS and objective=="value_max" and k in (2,3,4):log({"accounting":row})
    if model in MODELS and objective=="value_max":
        for t in (1.,1.1,1.2):
            a=g[g.score>=t]
            row={"bet":kind,"model":model,"objective":objective,"points":int(k),"subset":"fixed_diagnostic","threshold":t,"score_type":"direct_expected_gross" if kind=="trio" else "market_relative_proxy",**describe(a)}
            rows.append(row)
            if k in (2,3,4) and t==1.:log({"accounting":row})
        g=g.copy();g["score_band"]=band(g.score.to_numpy(),EV_BINS)
        for sb,b in g.groupby("score_band"):
            bands.append({"bet":kind,"model":model,"points":int(k),"score_band":sb,**describe(b)})
        for months,b in g.groupby("months"):
            fixed.append({"bet":kind,"model":model,"points":int(k),"months":int(months),"subset":"all",**describe(b)})
            for t in (1.,1.1,1.2):
                fixed.append({"bet":kind,"model":model,"points":int(k),"months":int(months),"subset":"fixed_diagnostic","threshold":t,**describe(b[b.score>=t])})
# Counterfactual accounting on the very same historical races; does not infer cause.
for k,g in pool[pool.objective=="value_max"].groupby("points"):
    a=g[g.model=="market_plus_rider"].sort_values("race_id")
    for model in MODELS[1:]:
        b=g[g.model==model].sort_values("race_id")
        assert a.race_id.tolist()==b.race_id.tolist()
        for t in (1.,1.1,1.2):
            ga=a.score.to_numpy()>=t;gb=b.score.to_numpy()>=t
            construction,selection,total=switch_decomposition(a.gross,b.gross,ga,gb)
            row={"bet":kind,"model":model,"points":int(k),"threshold":t,"baseline_n":int(ga.sum()),"challenger_n":int(gb.sum()),
                 "both_n":int((ga&gb).sum()),"baseline_only_n":int((ga&~gb).sum()),"challenger_only_n":int((gb&~ga).sum()),
                 "construction":interval(construction,a.date),"selection":interval(selection,a.date),"total":interval(total,a.date),
                 "total_week":interval(total,a.date,week=True)}
            switches.append(row)
            if k in (2,3,4) and t==1.:log({"switch_decomposition":row})
# Price and forecast-correction calibration, conditioned on chosen vs unchosen.
cal=[];c=C[C.months>=6]
for group,g in c.groupby(["model","selected","ratio_band","price_band"]):
    model,selected,rb,pb=group;n=int(g.n.sum())
    row={"bet":kind,"model":model,"selected":int(selected),"ratio_band":rb,"price_band":pb,"n":n,
        "pred":float(g.p_sum.sum()/n),"market":float(g.q_sum.sum()/n),"actual":float(g.y_sum.sum()/n),
        "pred_error":interval(g.y_sum-g.p_sum,g.date,g.n),"market_error":interval(g.y_sum-g.q_sum,g.date,g.n)}
    if kind=="trio":
        row["price_weighted_pred_error"]=interval(g.y_price-g.p_price,g.date,g.n)
        row["pred_gross"]=float(g.p_price.sum()/n);row["observed_quote_gross"]=float(g.y_price.sum()/n)
    cal.append(row)
# Aggregate selected candidates by correction band (price bands combined).
for group,g in c[c.selected==1].groupby(["model","ratio_band"]):
    model,rb=group;n=int(g.n.sum())
    row={"bet":kind,"model":model,"ratio_band":rb,"n":n,"pred":float(g.p_sum.sum()/n),"market":float(g.q_sum.sum()/n),"actual":float(g.y_sum.sum()/n),
         "pred_error":interval(g.y_sum-g.p_sum,g.date,g.n)}
    if kind=="trio":row.update({"pred_gross":float(g.p_price.sum()/n),"observed_quote_gross":float(g.y_price.sum()/n),"price_weighted_error":interval(g.y_price-g.p_price,g.date,g.n)})
    log({"selected_calibration":row})
save("formation_accounting.json",rows);save("score_band_diagnostics.json",bands);save("quarterly_accounting.json",fixed)
save("switch_decomposition.json",switches);save("candidate_calibration.json",cal)
if race_quality:
    pd.DataFrame(race_quality,columns=["race_id","date","months","source","market_gross","winning_quote","paid_multiple"]).to_csv(OUT/"quote_settlement_quality.csv",index=False)
save("status.json",{"status":"completed","scope":"retrospective accounting, not causal proof","source_run":46024184584,
    "limits":["same historical periods reused","no direct exacta odds: monetary decomposition impossible","final prices only","no independent holdout","fixed diagnostic thresholds inherited; no new purchase rules","day and week bootstrap intervals marginal, not multiplicity adjusted","selected vs unselected conditioning not causal"]})
