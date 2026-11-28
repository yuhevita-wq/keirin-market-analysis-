"""Chronological supervised learning: individual vs actual published line membership.
No odds, betting rule, previous model or live site changes."""
import io, json, subprocess, zipfile, re
from collections import defaultdict, Counter
from pathlib import Path
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

BASE=Path("data/rider_condition_2025_202610_full")
OUT=Path("research/line_rider_ml_20261128/output")
OUT.mkdir(parents=True,exist_ok=True)
def save(name,obj):
    (OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+"\n")
def git_bytes(path):
    return subprocess.check_output(["git","show","HEAD:"+path],stderr=subprocess.DEVNULL)
def csv_bytes(b):
    return pd.read_csv(io.BytesIO(b),dtype=str,keep_default_na=False,low_memory=False)
def norm(s): return str(s).strip()
def num(s): return pd.to_numeric(s,errors="coerce")
def entry_sources():
    summary=json.loads((BASE/"summary.json").read_text())
    for source in dict.fromkeys(x["source"] for x in summary["source_groups"]):
        try:
            if "!" in source:
                p,prefix=source.split("!",1)
                with zipfile.ZipFile(io.BytesIO(git_bytes(p))) as z:
                    prefix=prefix.rstrip("/")
                    pfx=prefix+"/" if prefix else ""
                    yield source,csv_bytes(z.read(pfx+"entries.csv")),csv_bytes(z.read(pfx+"races.csv"))
            else:
                def read(kind):
                    for ext in (".csv",".csv.gz"):
                        try:
                            b=git_bytes(source+"/"+kind+ext)
                            if ext.endswith(".gz"):
                                import gzip
                                b=gzip.decompress(b)
                            return csv_bytes(b)
                        except subprocess.CalledProcessError: pass
                    raise FileNotFoundError(source+"/"+kind)
                yield source,read("entries"),read("races")
        except Exception as e:
            errors[source]=str(e)
def valid_partition(groups,cars):
    g=tuple(sorted(tuple(x) for x in groups))
    flat=[x for line in g for x in line]
    if len(flat)!=len(set(flat)) or set(flat)!=set(cars) or not all(g):
        return None
    return g
def parse_formation(s,cars):
    s=norm(s).replace("－","-").replace("／","/").replace(" ","")
    if not re.fullmatch(r"[1-9]+(?:-[1-9]+)*(?:/[1-9]+(?:-[1-9]+)*)*",s): return None
    return valid_partition([re.findall(r"[1-9]",line) for line in s.split("/")],cars)
def partition(e,r):
    cars=e["car_no"].map(norm).tolist()
    candidates=[]
    for f in ("predicted_line_formation","line_formation","line_prediction"):
        if f in r:
            p=parse_formation(r[f],cars)
            if p: candidates.append(p)
    for f in ("line_id","line_group","line_no"):
        if f not in e or "line_position" not in e: continue
        if e[f].map(norm).eq("").any(): continue
        groups=[]
        for _,g in e.groupby(f,sort=False):
            pos=num(g["line_position"])
            if pos.isna().any() or set(pos)!=set(range(1,len(g)+1)): break
            if "line_size" in g and not num(g["line_size"]).eq(len(g)).all(): break
            groups.append(g.assign(_pos=pos).sort_values("_pos")["car_no"].map(norm).tolist())
        p=valid_partition(groups,cars)
        if p: candidates.append(p)
    return candidates[0] if candidates and len(set(candidates))==1 else None
errors={}
raw_groups=defaultdict(set)
raw_schemas={}
for src,e,r in entry_sources():
    raw_schemas[src]=list(e.columns)
    if not {"race_id","car_no"}<=set(e.columns) or "race_id" not in r: continue
    rmap=r.drop_duplicates("race_id").set_index("race_id").to_dict("index")
    for rid,g in e.groupby("race_id",sort=False):
        if g["car_no"].duplicated().any(): continue
        p=partition(g,rmap.get(rid,{}))
        if p: raw_groups[rid].add(p)
print("Raw line partitions",len(raw_groups),flush=True)
d=pd.concat([pd.read_csv(BASE/f"rider_races_{y}.csv.gz",dtype=str,keep_default_na=False) for y in (2025,2026)],ignore_index=True)
required={"race_id","race_date","car_no","provisional_player_key","class","style","finish_position","result_status","line_position","line_size"}
assert required<=set(d.columns), sorted(required-set(d.columns))
assert not d.duplicated(["race_id","car_no"]).any()
d["date"]=pd.to_datetime(d.race_date,errors="raise")
d["rank"]=num(d.finish_position)
d["field"]=d.groupby("race_id").car_no.transform("size")
d["label"]=d["rank"].where(d["rank"].isin([1,2,3]),0).astype(int)
# Full valid result races only. Exclude nonstarters, ambiguous ranks, unknown identities.
valid_ids=[]
for rid,g in d.groupby("race_id",sort=False):
    if len(g) not in (7,9): continue
    if not g.result_status.eq("full_result").all(): continue
    if any((g.label==k).sum()!=1 for k in (1,2,3)): continue
    if g.provisional_player_key.eq("").any() or g["class"].eq("").any(): continue
    # Non-numeric result must be explicitly identified as an actual non-finishing starter.
    if ((g["rank"].isna()) & ~g.finish_label.str.contains("失格|落車|棄権",regex=True)).any(): continue
    valid_ids.append(rid)
h=d[d.race_id.isin(valid_ids)].copy()
# History from the identical class, strictly previous dates (no same-day outcomes).
keys=["provisional_player_key","class","date"]
for k in (1,2,3): h[f"y{k}"]=(h.label==k).astype(int)
h["ytop"]=(h.label>0).astype(int)
h["n"]=1
hist=h.groupby(keys,sort=True)[["n","y1","y2","y3","ytop"]].sum().reset_index()
pkeys=keys[:2]
hist=hist.sort_values(keys)
for f in ["n","y1","y2","y3","ytop"]:
    hist["prev_"+f]=hist.groupby(pkeys,sort=False)[f].cumsum()-hist[f]
for f in ("y1","y2","y3","ytop"):
    hist["rate_"+f]=hist["prev_"+f]/hist.prev_n.replace(0,np.nan)
# Previous 90 calendar days, excluding current date.
rolling=[]
for _,g in hist.groupby(pkeys,sort=False):
    g=g.sort_values("date").set_index("date")
    s=g[["n","y1","y2","y3","ytop"]].rolling("90D",closed="left").sum().reset_index(drop=True)
    s=s.add_prefix("recent_")
    rolling.append(pd.concat([g.reset_index()[keys],s],axis=1))
rec=pd.concat(rolling,ignore_index=True)
for f in ("y1","y2","y3","ytop"):
    rec["recent_rate_"+f]=rec["recent_"+f]/rec.recent_n.replace(0,np.nan)
individual=["prev_n","rate_y1","rate_y2","rate_y3","rate_ytop","recent_n","recent_rate_y1","recent_rate_y2","recent_rate_y3","recent_rate_ytop"]
h=h.merge(hist[keys+individual[:5]],on=keys,validate="many_to_one").merge(rec[keys+individual[5:]],on=keys,validate="many_to_one")
# Restrict every comparison to exactly the same complete, unambiguous lines.
rows=[]
rejections=Counter()
for rid,g in h.groupby("race_id",sort=False):
    ps=raw_groups.get(rid,set())
    if len(ps)!=1:
        rejections["missing_line" if not ps else "conflicting_line"]+=1
        continue
    p=next(iter(ps))
    if {x for line in p for x in line}!=set(g.car_no): 
        rejections["car_mismatch"]+=1;continue
    bycar=g.set_index("car_no")
    # Validate against consolidated position/size wherever known.
    if any((norm(bycar.loc[car,"line_position"]) not in ("",str(i+1))) or
           (norm(bycar.loc[car,"line_size"]) not in ("",str(len(line))))
           for line in p for i,car in enumerate(line)):
        rejections["consolidated_line_mismatch"]+=1;continue
    strengths={i:bycar.loc[list(line),"rate_ytop"].mean() for i,line in enumerate(p)}
    for i,line in enumerate(p):
        other=[len(x) for j,x in enumerate(p) if i!=j]
        for j,car in enumerate(line):
            mate=bycar.loc[[x for x in line if x!=car]]
            rv=[v for k,v in strengths.items() if k!=i and pd.notna(v)]
            rows.append(dict(race_id=rid,car_no=car,line_group=i,line_pos=j+1,line_len=len(line),
                line_count=len(p),max_other_len=max(other,default=0),single_lines=sum(len(x)==1 for x in p),
                lead_rate=bycar.loc[line[0],"rate_ytop"],mate_rate=mate.rate_ytop.mean(),
                mate_win=mate.rate_y1.mean(),mate_recent=mate.recent_rate_ytop.mean(),
                rival_strength=max(rv,default=np.nan),own_strength=strengths[i]))
assert rows, "No verified line partitions; inspect source schemas"
a=h.merge(pd.DataFrame(rows),on=["race_id","car_no"],validate="one_to_one")
a=a.sort_values(["date","race_id","car_no"]).reset_index(drop=True)
audit=dict(source_rows=len(d),source_races=int(d.race_id.nunique()),valid_result_races=len(valid_ids),line_races=int(a.race_id.nunique()),
           raw_line_races=len(raw_groups),rejections=dict(rejections),source_errors=errors,source_schemas=raw_schemas,
           monthly_races=a.groupby(a.date.dt.strftime("%Y-%m")).race_id.nunique().astype(int).to_dict(),
           identity="provisional name/prefecture/term; histories within exact class",line="published predicted formation, not actual tactics",
           cohort="7 and 9 starters, unique podium, valid full results, same cohort for all models",no_odds=True)
save("audit.json",audit)
print(json.dumps({k:v for k,v in audit.items() if k not in ("source_schemas","source_errors")},ensure_ascii=False),flush=True)
common=["field","meeting_grade","race_type","meeting_day_no"]
personal=individual+["class","style","term","prefecture"]
line=["line_pos","line_len","line_count","max_other_len","single_lines"]
interaction=["lead_rate","mate_rate","mate_win","mate_recent","rival_strength","own_strength"]
variants={"context":common,"individual":common+personal,"line":common+line,"joint":common+personal+line+interaction}
categorical={"meeting_grade","race_type","class","style","prefecture"}
for col in set(sum(variants.values(),[])):
    if col in categorical: a[col]=a[col].fillna("").astype(str)
    else: a[col]=num(a[col]).astype(float)
# Keep the final 3 months identical for all learning-curve fits.
test=a[a.date.between("2026-08-01","2026-10-31")].copy()
assert test.race_id.nunique()>=100, "Insufficient untouched test races"
def evaluate(frame,prob):
    z=frame[["race_id","date","car_no","line_group","label"]].copy()
    for k in range(4): z[f"p{k}"]=prob[:,k]
    z["ptop"]=1-z.p0
    z["loss"]=-np.log(np.clip(prob[np.arange(len(z)),z.label.to_numpy()],1e-12,1))
    z["brier_top"]=(z.ptop-(z.label>0))**2
    z["win_pred"]=z.p1/z.groupby("race_id").p1.transform("sum").clip(lower=1e-12)
    z["win_brier"]=(z.win_pred-(z.label==1))**2
    def stats(f):
        races=[]
        for rid,g in f.groupby("race_id",sort=False):
            winner=g.loc[g.p1.idxmax()]
            top=g.nlargest(3,"ptop")
            order2=g.drop(index=winner.name).loc[lambda x:x.p2==x.p2.max()].iloc[0]
            line_pred=g.groupby("line_group").p1.sum().idxmax()
            actual_line=g.loc[g.label.eq(1),"line_group"].iloc[0]
            races.append((winner.label==1,(top.label>0).sum()/3,
                          winner.label==1 and order2.label==2,top.label.gt(0).all(),line_pred==actual_line))
        m=np.asarray(races,float).mean(axis=0)
        return dict(races=len(races),multiclass_logloss=float(f.loss.mean()),top3_brier=float(f.brier_top.mean()),
                    winner_brier=float(f.win_brier.mean()),winner_top1=float(m[0]),top3_recall=float(m[1]),
                    exact_top2_order_diagnostic=float(m[2]),exact_top3_set_diagnostic=float(m[3]),winner_line_accuracy=float(m[4]))
    return {"overall":stats(z),"months":{str(k):stats(g) for k,g in z.groupby(z.date.dt.strftime("%Y-%m"))}}
report={"test_period":["2026-08-01","2026-10-31"],"models":{},"notes":"Fixed hyperparameters; no test-period model selection. Exact top2/top3 diagnostics, not calibrated combination probabilities. Histories use only earlier dates including completed test-period days."}
for cut in ("2025-06-30","2025-12-31","2026-07-31"):
    train=a[a.date<=cut]
    assert train.race_id.nunique()>=100
    for name,features in variants.items():
        model=CatBoostClassifier(iterations=300,depth=6,learning_rate=0.06,loss_function="MultiClass",
                                 random_seed=42,thread_count=4,verbose=False,allow_writing_files=False)
        model.fit(train[features],train.label,cat_features=[f for f in features if f in categorical])
        assert list(model.classes_)==[0,1,2,3]
        pred=model.predict_proba(test[features])
        result=evaluate(test,pred)
        result["train_races"]=int(train.race_id.nunique())
        result["feature_importance"]={f:float(v) for f,v in zip(features,model.get_feature_importance())}
        report["models"][cut+"/"+name]=result
        save("metrics.json",report)
        print(cut,name,result["overall"],flush=True)
        if cut=="2026-07-31": model.save_model(str(OUT/(name+".cbm")))
save("metrics.json",report)
