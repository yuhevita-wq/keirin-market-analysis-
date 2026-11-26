"""Independent probability-learning experiment; no old model/threshold imports."""
from pathlib import Path
from itertools import permutations, combinations
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier

ROOT=Path("data/rider_condition_2025_202610_full")
OUT=Path("fresh_ml_output")
OUT.mkdir(exist_ok=True)
SEED=20261126
d=pd.concat([pd.read_csv(ROOT/f"rider_races_{y}.csv.gz",dtype=str) for y in (2025,2026)],ignore_index=True)
r=pd.concat([pd.read_csv(ROOT/f"races_{y}.csv.gz",dtype=str) for y in (2025,2026)],ignore_index=True)
assert not d.duplicated(["race_id","car_no"]).any()
assert not r.duplicated("race_id").any()
d["date"]=pd.to_datetime(d.race_date,errors="raise")
d["pos"]=pd.to_numeric(d.finish_position,errors="coerce")
d["car"]=pd.to_numeric(d.car_no,errors="coerce")
verified=set(r.loc[r.verified_podium.str.lower().isin(["true","1","yes"]),"race_id"])
assert verified, "No verified podium flags"
# History uses only fully recorded results of prior calendar dates. No same-day updates.
h=d.loc[d.result_status.eq("full_result") & d.pos.notna() & d.provisional_player_key.notna(),["provisional_player_key","date","pos"]].copy()
h["wins"]=(h.pos==1).astype(int)
h["top3"]=(h.pos<=3).astype(int)
h["starts"]=1
daily=h.groupby(["provisional_player_key","date"])[["starts","wins","top3"]].sum().reset_index()
# Include all entry dates even when no history outcome is present.
calendar=d.loc[d.provisional_player_key.notna(),["provisional_player_key","date"]].drop_duplicates()
daily=calendar.merge(daily,how="left",on=["provisional_player_key","date"]).fillna(0).sort_values(["provisional_player_key","date"])
cols=["starts","wins","top3"]
daily[cols]=daily.groupby("provisional_player_key")[cols].cumsum()-daily[cols]
d=d.merge(daily,how="left",on=["provisional_player_key","date"],validate="many_to_one")
d[cols]=d[cols].fillna(0)
d["history_win"]=d.wins/d.starts.replace(0,np.nan)
d["history_top3"]=d.top3/d.starts.replace(0,np.nan)
d["history_count"]=np.log1p(d.starts)
numeric=["car","term","line_position","line_size","history_win","history_top3","history_count"]
for col in numeric:
    d[col]=pd.to_numeric(d[col],errors="coerce")
# Fixed, interpretable category indicators; no learned mapping from evaluation data.
for c in ["SS","S1","S2","A1","A2","A3"]:
    name="class_"+c
    d[name]=d["class"].fillna("").str.upper().str.replace(" ","").eq(c).astype(float)
    numeric.append(name)
for c,name in [("逃","style_front"),("両","style_both"),("追","style_chase")]:
    d[name]=d["style"].fillna("").str.contains(c,regex=False).astype(float)
    numeric.append(name)
# Missingness is represented separately; no future-fitted imputation.
a=d[numeric].to_numpy(dtype=np.float32)
a=np.concatenate([np.nan_to_num(a,nan=0),np.isnan(a).astype(np.float32)],axis=1)
feature_names=numeric+["missing_"+x for x in numeric]
d=d.assign(_row=np.arange(len(d)))
valid=[]
excluded={}
for rid,g in d.groupby("race_id",sort=False):
    reason=None
    if rid not in verified: reason="unverified_podium"
    elif not g.result_status.eq("full_result").all(): reason="incomplete_or_nonstarter"
    elif len(g)<3 or g.car.isna().any() or g.car.duplicated().any(): reason="invalid_field"
    elif any((g.pos==k).sum()!=1 for k in (1,2,3)): reason="nonunique_podium"
    elif g.date.nunique()!=1: reason="inconsistent_date"
    if reason:
        excluded[reason]=excluded.get(reason,0)+1
        continue
    g=g.sort_values("car")
    valid.append((rid,g.date.iloc[0],a[g._row.to_numpy()],g.pos.to_numpy(),g.car.to_numpy()))
valid.sort(key=lambda x:(x[1],x[0]))
assert valid
audit={"source_races":int(d.race_id.nunique()),"verified_podium_races":len(verified),"eligible_races":len(valid),"excluded":excluded,"source_features":feature_names,"monthly_eligible":{str(k):int(v) for k,v in pd.Series([x[1] for x in valid]).dt.to_period("M").value_counts().sort_index().items()},"odds_used":False,"return_evaluated":False,"history_cutoff":"strictly before race calendar day; history may predate a training window"}
(OUT/"data_audit.json").write_text(json.dumps(audit,ensure_ascii=False,indent=2))
print(json.dumps(audit,ensure_ascii=False),flush=True)
dates=np.array([x[1].to_datetime64() for x in valid],dtype="datetime64[D]")
def make_data(kind):
    xs=[]; ys=[]; lens=[]; keys=[]
    for rid,day,f,p,cars in valid:
        n=len(p)
        ix=np.asarray(list(permutations(range(n),2) if kind=="exacta" else combinations(range(n),3)))
        z=f[ix]
        if kind=="exacta":
            x=np.concatenate([z[:,0],z[:,1],z[:,0]-z[:,1]],axis=1)
            y=(p[ix[:,0]]==1)&(p[ix[:,1]]==2)
        else:
            x=np.concatenate([z.mean(axis=1),z.min(axis=1),z.max(axis=1)],axis=1)
            y=np.all((p[ix]>=1)&(p[ix]<=3),axis=1)
        x=np.column_stack([x,np.full(len(ix),n,np.float32)])
        assert y.sum()==1
        xs.append(x.astype(np.float32));ys.append(y.astype(np.int8));lens.append(len(ix))
    lengths=np.array(lens)
    return np.concatenate(xs),np.concatenate(ys),lengths,np.r_[0,lengths.cumsum()]
def interval(a,b):
    left=np.searchsorted(dates,np.datetime64(a))
    right=np.searchsorted(dates,np.datetime64(b))
    assert right>left,(a,b)
    return left,right
def probs(scores,lengths):
    starts=np.r_[0,np.cumsum(lengths)[:-1]]
    z=np.exp(np.clip(scores-np.repeat(np.maximum.reduceat(scores,starts),lengths),-50,0))
    return z/np.repeat(np.add.reduceat(z,starts),lengths)
def evaluate(scores,y,lengths):
    starts=np.r_[0,np.cumsum(lengths)[:-1]]
    p=probs(scores,lengths)
    true=np.flatnonzero(y)
    # A deterministic tie break; uniform top-k is evaluated analytically below.
    ranks=np.array([1+np.sum(p[s:s+l]>p[t])+np.sum(p[s:t]==p[t]) for s,l,t in zip(starts,lengths,true)])
    return {"races":len(lengths),"log_loss":float(-np.log(np.clip(p[true],1e-15,1)).mean()),"brier":float(np.add.reduceat((p-y)**2,starts).mean()),"top1":float((ranks<=1).mean()),"top3":float((ranks<=3).mean()),"top5":float((ranks<=5).mean()),"mean_true_rank":float(ranks.mean())}
def uniform(lengths):
    return {"races":len(lengths),"log_loss":float(np.log(lengths).mean()),"brier":float((1-1/lengths).mean()),"top1":float((1/lengths).mean()),"top3":float((np.minimum(3,lengths)/lengths).mean()),"top5":float((np.minimum(5,lengths)/lengths).mean()),"mean_true_rank":float(((lengths+1)/2).mean())}
def addmonths(s,n):
    return (pd.Timestamp(s)+pd.DateOffset(months=n)).strftime("%Y-%m-%d")
windows=[]
for m in range(3,19,3):
    end=addmonths("2025-01-01",m)
    windows.append(("expanding",m,"2025-01-01",end,end,addmonths(end,3)))
    windows.append(("fixed_test",m,addmonths("2026-07-01",-m),"2026-07-01","2026-07-01","2026-10-01"))
# October reserved; no tuning or model selection from the reported folds.
windows.append(("final",21,"2025-01-01","2026-10-01","2026-10-01","2026-11-01"))
records=[]
for kind in ("exacta","trio"):
    X,Y,L,O=make_data(kind)
    for mode,m,ta,tb,va,vb in windows:
        u,v=interval(ta,tb); w,z=interval(va,vb)
        assert v<=w
        tr=slice(O[u],O[v]); te=slice(O[w],O[z])
        meta=dict(bet=kind,mode=mode,months=m,train_start=ta,train_end_exclusive=tb,test_start=va,test_end_exclusive=vb,train_races=v-u,train_candidates=int(O[v]-O[u]),test_candidates=int(O[z]-O[w]))
        # Fixed hyperparameters; no evaluation-based manual changes or threshold searches.
        scaler=StandardScaler().fit(X[tr])
        linear=SGDClassifier(loss="log_loss",alpha=0.0001,max_iter=25,tol=0.0001,random_state=SEED,average=True)
        linear.fit(scaler.transform(X[tr]),Y[tr])
        tree=LGBMClassifier(objective="binary",n_estimators=120,learning_rate=.05,num_leaves=31,min_child_samples=100,reg_lambda=1,verbosity=-1,n_jobs=4,random_state=SEED,deterministic=True,force_col_wise=True)
        tree.fit(X[tr],Y[tr])
        for name,result in [("uniform",uniform(L[w:z])),("linear",evaluate(linear.decision_function(scaler.transform(X[te])),Y[te],L[w:z])),("gbdt",evaluate(tree.predict(X[te],raw_score=True),Y[te],L[w:z]))]:
            row={**meta,"model":name,**result}
            records.append(row)
            print(json.dumps(row,ensure_ascii=False),flush=True)
        pd.DataFrame(records).to_csv(OUT/"learning_curve.csv",index=False)
    del X,Y,L,O
(OUT/"status.json").write_text(json.dumps({"status":"completed","rows":len(records),"note":"No odds or returns; race-normalized scores not separately calibrated; October reserved for this experiment only."},indent=2))
