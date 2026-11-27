"""Stage 3: statistical audit of immutable stage 1 and 2 artifacts.
No model fit, formation changes or real-world bets. Reused retrospective periods."""
import sys,json,hashlib,math
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
from scipy.stats import norm
OUT=Path("market_residual_statistics_output");OUT.mkdir(exist_ok=True)
SEED=20261127;BOOT=2000
MONTHS=(3,6,9,12,15)
def json_default(o):
    if isinstance(o,np.generic):return o.item()
    if isinstance(o,np.ndarray):return o.tolist()
    raise TypeError(f"Unsupported JSON type: {type(o).__name__}")
def save(n,o):
    p=OUT/n;tmp=p.with_suffix(".tmp")
    tmp.write_text(json.dumps(o,ensure_ascii=False,indent=2,allow_nan=False,default=json_default),encoding="utf-8")
    tmp.replace(p)
def log(o):print(json.dumps(o,ensure_ascii=False,allow_nan=False,default=json_default),flush=True)
def finite(v):return float(v) if np.isfinite(v) else None
def holm(ps):
    p=np.asarray(ps,dtype=float);order=np.argsort(p);v=np.maximum.accumulate(p[order]*(len(p)-np.arange(len(p))))
    out=np.empty(len(p));out[order]=np.minimum(v,1);return out
def day_stat(y,day,den=None):
    y=np.asarray(y,dtype=float);day=np.asarray(day,dtype=str)
    den=np.ones(len(y)) if den is None else np.asarray(den,dtype=float)
    assert len(day)==len(y)==len(den) and np.isfinite(y).all() and np.isfinite(den).all()
    if not len(y) or den.sum()<=0:return {"mean":None,"ci95":None,"se":None,"p_positive":1.,"days":0}
    days,inv=np.unique(day,return_inverse=True);n=len(days)
    a=np.bincount(inv,weights=y);b=np.bincount(inv,weights=den);mean=float(a.sum()/b.sum())
    e=a-mean*b
    se=float(np.sqrt(n/(n-1)*np.sum(e*e))/b.sum()) if n>1 else None
    rng=np.random.default_rng(SEED);ix=rng.integers(0,n,size=(BOOT,n))
    ga=a[ix].sum(1);gb=b[ix].sum(1);valid=gb>0
    ci=np.quantile(ga[valid]/gb[valid],[.025,.975]).tolist() if valid.any() else None
    p=float(norm.sf(mean/se)) if se and se>0 else (0. if mean>0 and n>1 else 1.)
    return {"mean":mean,"ci95":ci,"se":se,"p_positive":p,"days":n}
def selftest():
    obj={"points":np.int64(2),"nested":[np.float32(.5),np.bool_(True),np.array([1,2],dtype=np.int64)]}
    assert json.loads(json.dumps(obj,allow_nan=False,default=json_default))=={"points":2,"nested":[.5,True,[1,2]]}
    assert np.allclose(holm([.01,.04,.03]),[.03,.06,.06])
    d=np.repeat(["2025-01-01","2025-01-02","2025-01-03"],2)
    q=day_stat(np.ones(6),d);assert q["mean"]==1 and q["ci95"]==[1.,1.]
    q=day_stat(np.array([1,2,3,4,5,6]),d,np.full(6,2));assert q["mean"]==1.75
    print("statistical audit self-test passed",flush=True)
selftest()
if "--self-test" in sys.argv:sys.exit(0)
kind=sys.argv[1];assert kind in ("exacta","trio")
def file(root,name):
    p=list(Path(root).rglob(name));assert len(p)==1,(name,p);return p[0]
probfiles={m:file("market_residual_output",f"{kind}_{m}m_oof_probabilities.csv.gz") for m in MONTHS}
formfiles={m:file("market_residual_math_output",f"race_formations_{m}m.csv.gz") for m in MONTHS}
manifest={"baseline_stage1_run":46005962698,"baseline_stage2_run":46007518529,
          "probability_files":{str(m):hashlib.sha256(p.read_bytes()).hexdigest() for m,p in probfiles.items()},
          "formation_files":{str(m):hashlib.sha256(p.read_bytes()).hexdigest() for m,p in formfiles.items()}}
save("manifest.json",manifest)
# Probability calibration and paired proper scoring-rule comparison by race.
# Positive improvement = lower loss than comparator; bootstrap clusters calendar days.
prob_rows=[];cal_rows=[];winner_rows=[];race_parts=[];cal_parts=[]
comparisons=[("market_plus_rider","current_interactions"),("market_plus_rider","historical_interactions"),("current_interactions","historical_interactions"),("market_raw","historical_interactions")]
models={"market_raw":("market_raw","probability_raw"),"rider_only":("rider_only","probability_raw"),
        "market_adjusted":("market_adjusted","probability_raw"),"market_plus_rider":("market_plus_rider","probability_raw"),
        "market_plus_rider_calibrated":("market_plus_rider","probability_calibrated"),"current_interactions":("current_interactions","probability_raw"),"historical_interactions":("historical_interactions","probability_raw")}
bins=[0,.01,.02,.05,.1,.2,.4,.6,.8,1.00000001]
for months,p in probfiles.items():
    d=pd.read_csv(p,dtype={"race_id":str,"date":str,"combination":str})
    key=["race_id","combination"]
    base=d[d.model=="market_raw"].sort_values(key).reset_index(drop=True)
    assert not base.duplicated(key).any()
    actual=base.actual.to_numpy(dtype=int);rid=base.race_id.to_numpy();date=base.date.to_numpy()
    starts=np.r_[0,np.flatnonzero(rid[1:]!=rid[:-1])+1];counts=np.diff(np.r_[starts,len(base)])
    assert np.all(np.add.reduceat(actual,starts)==1)
    race=pd.DataFrame({"race_id":rid[starts],"date":date[starts],"months":months})
    for label,(src,col) in models.items():
        a=d[d.model==src].sort_values(key).reset_index(drop=True)
        assert a[key].equals(base[key]) and np.array_equal(a.actual,actual)
        q=a[col].to_numpy(dtype=float)
        assert (q>0).all() and np.allclose(np.add.reduceat(q,starts),1,atol=1e-8)
        true=q[actual==1]
        race[label+"_log_loss"]=-np.log(true)
        race[label+"_brier"]=np.add.reduceat(q*q,starts)-2*true+1
        r=np.argmax(np.zeros(1)) # no outcome-dependent selection
        ix=np.clip(np.searchsorted(bins,q,side="right")-1,0,len(bins)-2)
        # Candidate count and summed predicted/observed successes, clustered by day.
        temp=pd.DataFrame({"date":date,"months":months,"model":label,"bin":ix,
            "predicted":q,"actual":actual,"count":1})
        cal_parts.append(temp.groupby(["date","months","model","bin"],as_index=False).sum())
    race_parts.append(race)
    del d,base,a,temp
pr=pd.concat(race_parts,ignore_index=True);assert not pr.race_id.duplicated().any()
cal=pd.concat(cal_parts,ignore_index=True)
for months in (3,6,9,12,15,"pooled"):
    r=pr[pr.months>=6] if months=="pooled" else pr[pr.months==months]
    for baseline,challenger in comparisons:
        for metric in ("log_loss","brier"):
            a=r[baseline+"_"+metric].to_numpy();b=r[challenger+"_"+metric].to_numpy()
            st=day_stat(a-b,r.date)
            row={"bet":kind,"months":months,"baseline":baseline,"challenger":challenger,"metric":metric,
                 "races":len(r),"baseline_mean":float(a.mean()),"challenger_mean":float(b.mean()),"improvement":st}
            prob_rows.append(row)
    c=cal[cal.months>=6] if months=="pooled" else cal[cal.months==months]
    for (model,bi),g in c.groupby(["model","bin"]):
        n=int(g["count"].sum());pred=float(g.predicted.sum()/n);obs=float(g.actual.sum()/n)
        # signed calibration gap, positive = observed exceeds predicted
        st=day_stat(g.actual-g.predicted,g.date,g["count"])
        cal_rows.append({"bet":kind,"months":months,"model":model,"lo":bins[bi],"hi":bins[bi+1],
                         "candidates":n,"predicted":pred,"observed":obs,"gap":st})
pooled=[r for r in prob_rows if r["months"]=="pooled"]
# Correct for all eight predeclared paired score tests per bet; use two-sided p for diagnostic differences.
pv=[]
for r in pooled:
    st=r["improvement"];se=st["se"]
    pv.append(float(2*norm.sf(abs(st["mean"]/se))) if se and se>0 else 1.)
for r,p in zip(pooled,holm(pv)):
    r["holm_two_sided_p"]=float(p);log({"probability_comparison":r})
save("probability_comparison.json",prob_rows);save("calibration_by_bin.json",cal_rows)
pr.to_csv(OUT/"probability_race_scores.csv.gz",index=False,compression="gzip")
# Formation comparisons: exact immutable rows from stage 2. No new ticket construction.
f=pd.concat([pd.read_csv(p,dtype={"race_id":str,"date":str,"tickets":str}) for p in formfiles.values()],ignore_index=True)
keys=["model","objective","family","points"]
assert not f.duplicated(["race_id",*keys]).any()
assert f.groupby("months").race_id.nunique().to_dict()=={3:4259,6:4707,9:2468,12:4454,15:4230}
assert set(f.odds_source)==({"proxy"} if kind=="exacta" else {"direct"})
# Full-coverage sizes only; rare structural sizes are explicitly excluded from inference.
sizes=(2,3,4,6,8) if kind=="exacta" else (1,2,3,4,6,9,10)
f=f[f.points.isin(sizes)].copy()
assert f.groupby(keys).race_id.nunique().eq(20118).all()
rows=[];pairs=[];threshold_rows=[];qualification=[];concentration=[]
def assess(g):
    n=len(g);y=g.actual_gross.to_numpy(dtype=float);day=g.date.to_numpy()
    st=day_stat(y-1,day);h=day_stat(g.actual_hit-g.predicted_hit,day)
    gross=float(y.sum());top=np.sort(y)[::-1];daily=g.groupby("date").actual_gross.sum().sort_values(ascending=False)
    return {"n":n,"roi_pct":100*float(y.mean()),"roi_ci95_pct":[100*(x+1) for x in st["ci95"]],
        "net_se":st["se"],"p_positive":st["p_positive"],"days":st["days"],
        "actual_hit_rate":float(g.actual_hit.mean()),"predicted_hit_rate":float(g.predicted_hit.mean()),
        "hit_gap_ci95":h["ci95"],"score_mean":float(g.expected_return_or_proxy.mean()),
        "top1_payout_share":float(top[:1].sum()/gross) if gross else None,
        "top5_payout_share":float(top[:5].sum()/gross) if gross else None,
        "top_day_payout_share":float(daily.iloc[0]/gross) if gross else None,
        "roi_without_top5_hits_pct":100*float((gross-top[:5].sum())/n),
        "positive_quarters":int((g.groupby("months").actual_gross.mean()>1).sum())}
for k,g in f.groupby(keys,sort=True):
    common={"bet":kind,**dict(zip(keys,k)),"odds_source":g.odds_source.iloc[0]}
    for months in (3,6,9,12,15,"pooled"):
        a=g[g.months>=6] if months=="pooled" else g[g.months==months]
        row={**common,"months":months,**assess(a)}
        rows.append(row)
# Compare rider addition with market reference using same race dates/tickets policy; no causal claim.
for (objective,family,points),g in f.groupby(["objective","family","points"]):
    m=g[g.model=="market_plus_rider"].sort_values("race_id")
    for model in ("current_interactions","historical_interactions"):
        h=g[g.model==model].sort_values("race_id")
        assert m.race_id.tolist()==h.race_id.tolist()
        use=m.months.to_numpy()>=6
        st=day_stat((h.actual_gross.to_numpy()-m.actual_gross.to_numpy())[use],m.date.to_numpy()[use])
        pairs.append({"bet":kind,"model":model,"baseline":"market_plus_rider","objective":objective,"family":family,"points":int(points),"n":int(use.sum()),"paired_return_difference":st})
# Predeclared historical score bands, not learned cutoffs. Direct EV and proxy kept separate.
# The reference threshold for a monetary expected return is 1, while the
# proxy is only a relative ratio and cannot certify expected profit.
thresholds=[1.0,1.1,1.2]
scope=f[f.model.isin(["market_plus_rider","current_interactions","historical_interactions"])&(f.objective=="value_max")].copy()
for (model,family,points),g in scope.groupby(["model","family","points"]):
    for t in thresholds:
        a=g[g.expected_return_or_proxy>=t]
        for months in (3,6,9,12,15,"pooled"):
            b=a[a.months>=6] if months=="pooled" else a[a.months==months]
            row={"bet":kind,"model":model,"family":family,"points":int(points),"threshold":t,
                 "score_type":"expected_gross" if kind=="trio" else "market_relative_index",
                 "months":months,"eligible":int((g.months>=6).sum() if months=="pooled" else (g.months==months).sum())}
            if len(b):row.update(assess(b))
            else:row.update({"n":0,"roi_pct":None,"roi_ci95_pct":None,"p_positive":1.,"days":0})
            threshold_rows.append(row)
# All pooled historical comparisons belong to one family of tests per bet;
# adjusted p is descriptive, not a confirmatory result on this reused dataset.
all_pooled=[r for r in rows+threshold_rows if r["months"]=="pooled"]
for r,p in zip(all_pooled,holm([r["p_positive"] for r in all_pooled])):
    r["holm_positive_p"]=float(p)
for r,p in zip(pairs,holm([r["paired_return_difference"]["p_positive"] for r in pairs])):
    r["holm_positive_p"]=float(p)
# Strict past-only statistical qualification of fixed diagnostics.
# 3連複: prior observed mean net lower bound (Bonferroni within current
# family of thresholds/points) >0; >=200 cases and >=30 distinct dates.
# No thresholds learned from current quarter, no deployment, and exacta
# has no monetary qualification because direct prices are missing.
# These minima and alpha are research design, not inferred optimal values.
policy_count=len(sizes)*2*len(thresholds)*3
zcrit=float(norm.isf(.05/policy_count))
for months in (6,9,12,15):
    for (model,family,points),g in scope.groupby(["model","family","points"]):
        for t in thresholds:
            past=g[(g.months<months)&(g.expected_return_or_proxy>=t)]
            test=g[(g.months==months)&(g.expected_return_or_proxy>=t)]
            assert not len(past) or not len(test) or past.date.max()<test.date.min()
            st=day_stat(past.actual_gross-1,past.date)
            lb=st["mean"]-zcrit*st["se"] if st["mean"] is not None and st["se"] is not None else None
            passed=bool(kind=="trio" and len(past)>=200 and st["days"]>=30 and lb is not None and lb>0)
            row={"bet":kind,"model":model,"months":months,"family":family,"points":int(points),"threshold":t,
                 "prior_n":len(past),"prior_days":st["days"],"prior_lower_bound_net":lb,"qualified":passed,
                 "next_period_candidate_n":len(test),"next_period_qualified_n":len(test) if passed else 0,
                 "next_period_roi_pct":100*float(test.actual_gross.mean()) if passed and len(test) else None,
                 "reason":"direct exacta odds absent" if kind=="exacta" else "passed" if passed else "insufficient prior evidence"}
            qualification.append(row)
save("formation_statistics.json",rows);save("paired_formation_comparison.json",pairs)
for r in pairs:
    if r["family"]=="best_structured":log({"paired_formation_comparison":r})
save("score_threshold_diagnostics.json",threshold_rows);save("past_only_qualification.json",qualification)
for r in rows:
    if r["months"]=="pooled" and r["family"]=="best_structured" and r["model"] in ("market_plus_rider","current_interactions","historical_interactions"):log({"formation_statistics":r})
for r in threshold_rows:
    if r["months"]=="pooled" and r["family"]=="best_structured":log({"threshold_diagnostic":r})
log({"qualification_summary":{"bet":kind,"checks":len(qualification),"qualified":sum(r["qualified"] for r in qualification),
    "qualified_by_month":{str(m):sum(r["qualified"] for r in qualification if r["months"]==m) for m in (6,9,12,15)},
    "monetary_gate_possible":kind=="trio","alpha":.05,"bonferroni_policies":policy_count}})
save("status.json",{"status":"completed","stage":3,"baseline_stage1_run":46005962698,"baseline_stage2_run":46007518529,
    "design":{"bootstrap":"calendar-day cluster, 2000 draws, marginal intervals","multiplicity":"Holm within bet; Bonferroni for prior-only qualification","thresholds":thresholds,"min_prior_races":200,"min_prior_days":30},
    "limitations":["all periods previously examined; no independent holdout","final odds not verified pre-close","exacta odds entirely proxy; no monetary EV qualification","day clusters may not remove dependence across days","normal approximation p-values unreliable in sparse heavy-tailed groups","equal notional race stake","no real bets or deployment"]})
