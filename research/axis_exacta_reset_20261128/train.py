"""Independent exacta anchor learning. No existing predictions/models used."""
import csv, json, re, hashlib, unicodedata
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.metrics import log_loss, brier_score_loss

ROOT=Path("data")
OUT=Path("axis_exacta_output"); OUT.mkdir(exist_ok=True)
NUM="age period score s_count b_count nige_count makuri_count sashi_count mark_count first_count second_count third_count outside_count win_rate top2_rate top3_rate gear".split()
CAT="style class grade prefecture".split()
def rows(p):
    with p.open(encoding="utf-8-sig",newline="") as f: yield from csv.DictReader(f)
def num(x):
    try: return float(str(x).replace("%","").replace(",",""))
    except: return np.nan
def integer(x):
    try: return int(x)
    except: return None
def norm(x): return unicodedata.normalize("NFKC",str(x or ""))
def load():
    audit=Counter(); records={}
    dirs=sorted({p.parent for p in ROOT.rglob("entries.csv") if all((p.parent/f).exists() for f in ("races.csv","results.csv","payouts.csv"))})
    for d in dirs:
        cards={x["race_id"]:x for x in rows(d/"races.csv") if "2025-01-01"<=x.get("race_date","")<="2026-10-31"}
        if not cards: continue
        ent=defaultdict(list); fin=defaultdict(list); pay=defaultdict(list)
        for e in rows(d/"entries.csv"):
            if e.get("race_id") in cards: ent[e["race_id"]].append(e)
        for e in rows(d/"results.csv"):
            if e.get("race_id") in cards: fin[e["race_id"]].append(e)
        for p in rows(d/"payouts.csv"):
            if p.get("race_id") not in cards: continue
            if norm(p.get("bet_code") or p.get("ticket_type") or p.get("bet_type")).lower() not in {"exacta","2車単","二車単"}: continue
            if p.get("status","paid") not in ("paid",""): continue
            c=tuple(map(int,re.findall(r"\d+",norm(p.get("combination")))))
            v=num(p.get("payout_yen"))
            if len(c)==2 and c[0]!=c[1] and np.isfinite(v) and v>0: pay[p["race_id"]].append((c,float(v)))
        for rid,c in cards.items():
            es=ent[rid]; ns=[integer(e.get("car_no")) for e in es]; n=integer(c.get("entry_count")) or len(es)
            if n<5 or len(es)!=n or None in ns or len(set(ns))!=n: audit["bad_entries"]+=1;continue
            if not all(any(np.isfinite(num(e.get(k))) for k in NUM) for e in es): audit["missing_features"]+=1;continue
            f=[(integer(z.get("finish_position")),integer(z.get("car_no"))) for z in fin[rid]]
            one=[a for p,a in f if p==1]; two=[a for p,a in f if p==2]
            pp=list(set(pay[rid]))
            if len(one)!=1 or len(two)!=1 or len(pp)!=1 or pp[0][0]!=(one[0],two[0]): audit["missing_or_nonstandard_result_payout"]+=1;continue
            if any(p is None for p,a in f): audit["incident_or_missing_finish"]+=1;continue
            if rid in records: audit["duplicate"]+=1;continue
            lines={}
            for li,l in enumerate(norm(c.get("predicted_line_formation")).split("/")):
                cars=[int(v) for v in re.findall(r"\d+",l)]
                for j,a in enumerate(cars): lines[a]=(li,j+1,len(cars))
            if set(lines)!=set(ns): lines={}
            base={}
            for e,a in zip(es,ns):
                x={k:num(e.get(k)) for k in NUM}
                x.update({k:norm(e.get(k)) or "?" for k in CAT})
                x.update(car_no=a,n=n,track=norm(c.get("track")),race_type=norm(c.get("race_type")),meeting_grade=norm(c.get("meeting_grade")),race_no=num(c.get("race_no")),line_position=lines.get(a,(0,np.nan,np.nan))[1],line_size=lines.get(a,(0,np.nan,np.nan))[2])
                base[a]=x
            for k in NUM:
                vs=pd.Series({a:x[k] for a,x in base.items()})
                for a,x in base.items():
                    x[k+"_relative"]=x[k]-vs.mean()
                    x[k+"_rank"]=vs.rank(pct=True).loc[a]
            records[rid]=(c["race_date"],base,lines,pp[0][0],pp[0][1])
    return records,dict(audit)

def frame(records):
    axes=[];pairs=[]
    for rid,(date,base,lines,winner,pay) in records.items():
        for a,x in base.items():
            axes.append(dict(x,rid=rid,date=date,a=a,y=1 if a==winner[0] else 2 if a==winner[1] else 0,ret=pay if a in winner else 0))
            for b,z in base.items():
                if a==b:continue
                # Ordered pair labels; both directions share the same candidate set.
                p={**{"a_"+k:v for k,v in x.items()},**{"b_"+k:v for k,v in z.items()}, "same_line":float(lines[a][0]==lines[b][0]) if a in lines and b in lines else np.nan}
                pairs.append(dict(p,rid=rid,date=date,a=a,b=b,y=int((a,b)==winner),ret=pay if (a,b)==winner else 0))
    return pd.DataFrame(axes),pd.DataFrame(pairs)
def features(d):
    cols=[c for c in d if c not in {"rid","date","a","b","y","ret"} and d[c].nunique(dropna=True)>1]
    cats=[c for c in cols if d[c].dtype=="object"]
    return cols,cats
def period(d): return np.where(d.date<"2026-01-01","train",np.where(d.date<"2026-07-01","validation","test"))
def model(d,kind):
    cols,cats=features(d[d.split=="train"].drop(columns="split"))
    tr=d[d.split=="train"];ev=d[d.split=="validation"]
    if kind=="cls":
        m=CatBoostClassifier(iterations=500,depth=6,learning_rate=.05,verbose=False,thread_count=4,random_seed=42,allow_writing_files=False)
        target="y"
    else:
        m=CatBoostRegressor(iterations=500,depth=6,learning_rate=.05,verbose=False,thread_count=4,random_seed=42,allow_writing_files=False)
        target="ret"
    m.fit(tr[cols],tr[target],cat_features=cats,eval_set=(ev[cols],ev[target]),early_stopping_rounds=50)
    m.save_model(str(OUT/(("axis" if "b" not in d else "pair")+"_"+kind+".cbm")))
    return m,cols
def metrics(rows):
    st=sum(z["stake"] for z in rows); ret=sum(z["payout"] for z in rows)
    return dict(races=len(rows),stake=st,payout=ret,profit=ret-st,roi=100*ret/st if st else None,hit_rate=float(np.mean([z["payout"]>0 for z in rows])) if rows else None,axis_top2=float(np.mean([z["axis_hit"] for z in rows])) if rows else None,largest_payout=max((z["payout"] for z in rows),default=0))
def main():
    rec,audit=load()
    print("DATA",len(rec),audit,flush=True)
    a,p=frame(rec)
    for d in (a,p):d["split"]=period(d)
    counts=a.groupby("split").rid.nunique().to_dict()
    print("SPLITS",counts,flush=True)
    if min(counts.get(s,0) for s in ["train","validation","test"])<100: raise RuntimeError("Insufficient verified races in time split")
    ac,ax=model(a,"cls"); ar,rx=model(a,"reg"); pc,px=model(p,"cls")
    a["p1"]=ac.predict_proba(a[ax])[:,list(ac.classes_).index(1)]
    a["p2"]=ac.predict_proba(a[ax])[:,list(ac.classes_).index(2)]
    a["top2"]=a.p1+a.p2
    a["return_score"]=ar.predict(a[rx])
    a["random"]=a.apply(lambda z:int(hashlib.sha256(f"{z.rid}:{z.a}".encode()).hexdigest()[:12],16),axis=1)
    p["prob"]=pc.predict_proba(p[px])[:,list(pc.classes_).index(1)]
    p["prob"]=p.prob/p.groupby("rid").prob.transform("sum")
    detail=[]; diagnostics={}
    for split in ("validation","test"):
        aa=a[a.split==split];pp=p[p.split==split]
        diagnostics[split]={"axis_top2_brier":brier_score_loss((aa.y>0).astype(int),aa.top2),"pair_log_loss":float(-np.log(pp.loc[pp.y==1,"prob"].clip(1e-12)).mean())}
        pg={rid:g for rid,g in pp.groupby("rid")}
        for rid,g in aa.groupby("rid"):
            _,base,_,winner,payout=rec[rid]; pairs=pg[rid]
            lookup={(int(z.a),int(z.b)):z.prob for z in pairs.itertuples()}
            for strategy in ("p1","top2","return_score","random"):
                anchor=int(g.sort_values([strategy,"a"],ascending=[False,True]).iloc[0].a)
                opponents=sorted((b for b in base if b!=anchor),key=lambda b:(-(lookup[anchor,b]+lookup[b,anchor]),b))
                for k in (2,3,4,5,6,8):
                    if k>len(opponents):continue
                    opp=opponents[:k]; hit=anchor in winner and any(b in winner for b in opp)
                    detail.append(dict(split=split,rid=rid,date=rec[rid][0],strategy=strategy,k=k,anchor=anchor,opponents=",".join(map(str,opp)),axis_hit=anchor in winner,stake=200*k,payout=payout if hit else 0,direction=("front" if winner[0]==anchor else "reverse") if hit else "miss"))
    d=pd.DataFrame(detail);d.to_csv(OUT/"race_detail.csv",index=False)
    summary={}
    for (s,st,k),g in d.groupby(["split","strategy","k"]):
        v=metrics(g.to_dict("records"))
        for direction in ("front","reverse"):
            sub=g[g.direction==direction];v[direction+"_hits"]=len(sub);v[direction+"_payout"]=float(sub.payout.sum())
        summary[f"{s}/{st}/{k}"]=v
    # Compare prespecified configurations on validation. Test is never used to choose.
    candidates=[(key,v) for key,v in summary.items() if key.startswith("validation/") and "/random/" not in key]
    best=max(candidates,key=lambda kv:kv[1]["roi"])[0]
    report={"audit":audit,"counts":counts,"diagnostics":diagnostics,"validation_selected":best,"selected_test":summary.get(best.replace("validation/","test/")),"all":summary,"limitations":["No all-combination odds are used; historical payouts are labels/evaluation only.","No source timestamp audit; retrospective racecard fields must be checked before deployment.","Nonstandard/missing results excluded; missingness may bias sample.","Each ticket assumed 100 yen. No race skipping or stakes optimization."]}
    (OUT/"report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2,default=lambda v:v.item()))
    print(json.dumps(report,ensure_ascii=False,default=lambda v:v.item()),flush=True)
if __name__=="__main__":main()
