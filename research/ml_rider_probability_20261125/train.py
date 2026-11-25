"""Chronological, read-only rider probability experiment. No bet rules."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import log_loss, brier_score_loss
from sklearn.linear_model import LogisticRegression

root=Path("data/rider_condition_2025_202610_full")
out=Path("ml-rider-results");out.mkdir(exist_ok=True)
d=pd.concat([pd.read_csv(root/f"rider_races_{y}.csv.gz",dtype={"race_id":str}) for y in (2025,2026)],ignore_index=True)
assert not d.duplicated(["race_id","car_no"]).any()
d["race_date"]=pd.to_datetime(d.race_date)
d["pos"]=pd.to_numeric(d.finish_position,errors="coerce")
# Strict evaluation subset: complete, unique 1/2/3 labels. Non-finishers
# with a verified start count as non-podium; unknown races are excluded.
ok=d.result_status.eq("full_result") & (d.pos.gt(0) | d.finish_label.fillna("").str.contains("失格|落車|棄権"))
d["valid"]=ok
d["y"]=d.pos.le(3).astype(int)
d["win"]=d.pos.eq(1).astype(int)
d["second"]=d.pos.eq(2).astype(int)
d["third"]=d.pos.eq(3).astype(int)
g=d.groupby("race_id")
complete=g.valid.all() & g.win.sum().eq(1) & g.second.sum().eq(1) & g.third.sum().eq(1) & g.size().ge(4)
d=d[d.race_id.isin(complete[complete].index)].copy()
d["field"]=d.groupby("race_id").car_no.transform("size")
cats=["track","meeting_grade","race_type","class","style","line_role"]
for c in cats: d[c]=d[c].fillna("unknown").astype(str)
d["rider"]=d.provisional_player_key.fillna("").astype(str)
d["known_id"]=d.rider.ne("")
# Isolate unidentified riders rather than creating a shared synthetic career.
d.loc[~d.known_id,"rider"]="unknown:"+d.loc[~d.known_id,"race_id"]+":"+d.loc[~d.known_id,"car_no"].astype(str)
# All historical aggregates stop at the preceding calendar day, even within a meeting.
features=[]
for name, extra in [("overall",[]),("track",["track"]),("type",["race_type"]),("day",["meeting_day_no"]),("line",["line_role"])]:
    keys=["rider","class"]+extra
    for k in keys: d[k]=d[k].fillna("unknown") if d[k].dtype=="object" else d[k].fillna(-1)
    daily=d.groupby(keys+["race_date"],dropna=False,as_index=False).agg(n=("y","size"),hit=("y","sum"),w=("win","sum"),s=("second","sum"),t=("third","sum"))
    daily=daily.sort_values("race_date")
    for col in ["n","hit","w","s","t"]:
        daily[col]=daily.groupby(keys,dropna=False)[col].cumsum()-daily[col]
    n=name+"_n"; daily= daily.rename(columns={"n":n,"hit":name+"_hit","w":name+"_w","s":name+"_s","t":name+"_t"})
    d=d.merge(daily,on=keys+["race_date"],how="left",validate="many_to_one")
    features.append(n)
    for target, prior in [("hit",3/d.field),("w",1/d.field),("s",1/d.field),("t",1/d.field)]:
        col=name+"_"+target+"_rate"
        d[col]=(d[name+"_"+target]+10*prior)/(d[n]+10)
        features.append(col)
num=["car_no","term","meeting_day_no","race_no","line_position","line_size","field"]
for c in num: d[c]=pd.to_numeric(d[c],errors="coerce")
# Within-race relative strength, without accessing outcomes.
for c in ["overall_hit_rate","overall_w_rate","type_hit_rate"]:
    f=c+"_relative";d[f]=d[c]-d.groupby("race_id")[c].transform("mean");features.append(f)
X=d[cats+num+features].copy()
train=d.race_date.between("2025-04-01","2026-04-30")
val=d.race_date.between("2026-05-01","2026-07-31")
test=d.race_date.between("2026-08-01","2026-10-31")
assert train.sum()>0 and val.sum()>0 and test.sum()>0
assert set(d.loc[train,"race_id"]).isdisjoint(d.loc[test,"race_id"])
def logit(p):
    p=np.clip(p,1e-6,1-1e-6);return np.log(p/(1-p)).reshape(-1,1)
def score(y,p):
    return {"brier":float(brier_score_loss(y,p)),"logloss":float(log_loss(y,p,labels=[0,1]))}
report={"source_rows":int(len(pd.read_csv(root/"rider_races_2025.csv.gz",usecols=["race_id"]))+len(pd.read_csv(root/"rider_races_2026.csv.gz",usecols=["race_id"]))),"eligible_races":int(d.race_id.nunique()),"splits":{k:{"rows":int(m.sum()),"races":int(d.loc[m,"race_id"].nunique())} for k,m in [("train",train),("validation",val),("test",test)]},"features":list(X.columns),"targets":{}}
pred=d.loc[test,["race_id","race_date","car_no","meeting_grade","class","y","win","second","third"]].copy()
for target,base in [("y","overall_hit_rate"),("win","overall_w_rate"),("second","overall_s_rate"),("third","overall_t_rate")]:
    model=CatBoostClassifier(iterations=500,depth=6,learning_rate=.06,loss_function="Logloss",random_seed=42,thread_count=4,verbose=False,allow_writing_files=False)
    model.fit(X.loc[train],d.loc[train,target],cat_features=cats)
    rawval=model.predict_proba(X.loc[val])[:,1];raw=model.predict_proba(X.loc[test])[:,1]
    cal=LogisticRegression().fit(logit(rawval),d.loc[val,target])
    p=cal.predict_proba(logit(raw))[:,1]
    # Convert independent probabilities to a race-consistent total by a
    # monotone logit shift; preserves order without arbitrary rescaling.
    tmp=pd.DataFrame({"race":d.loc[test,"race_id"].to_numpy(),"p":p})
    z=logit(p).ravel();idx=tmp.groupby("race",sort=False).ngroup().to_numpy()
    size=np.bincount(idx);total=3 if target=="y" else 1
    lo=np.full(len(size),-30.);hi=np.full(len(size),30.)
    for _ in range(40):
        mid=(lo+hi)/2
        pp=1/(1+np.exp(-np.clip(z+mid[idx],-50,50)))
        s=np.bincount(idx,weights=pp)
        lo=np.where(s<total,mid,lo);hi=np.where(s>=total,mid,hi)
    adj=1/(1+np.exp(-np.clip(z+((lo+hi)/2)[idx],-50,50)))
    actual=d.loc[test,target];uniform=total/d.loc[test,"field"]
    metrics={k:score(actual,v) for k,v in [("uniform",uniform),("same_class_history",d.loc[test,base]),("raw_ml",raw),("calibrated_ml",p),("race_consistent_ml",adj)]}
    per_month={}
    for month in sorted(d.loc[test,"race_date"].dt.strftime("%Y-%m").unique()):
        mask=d.loc[test,"race_date"].dt.strftime("%Y-%m").eq(month).to_numpy()
        per_month[month]={"history":score(actual.to_numpy()[mask],d.loc[test,base].to_numpy()[mask]),"ml":score(actual.to_numpy()[mask],adj[mask])}
    report["targets"][target]={"metrics":metrics,"months":per_month,"calibration":{"coefficient":float(cal.coef_[0,0]),"intercept":float(cal.intercept_[0])},"importance":dict(sorted(zip(X.columns,map(float,model.feature_importances_)),key=lambda kv:-kv[1]))}
    pred[target+"_prob"]=adj
    model.save_model(str(out/(target+".cbm")))
    print(target,metrics,flush=True)
pred.to_csv(out/"test_predictions.csv.gz",index=False)
(out/"report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2))
print(json.dumps(report["splits"]),flush=True)
