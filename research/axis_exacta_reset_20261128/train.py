"""Direct payout learning; no finishing-position probability model or rank prefilter."""
import csv, json, re, hashlib, unicodedata
from itertools import combinations
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
import math
import gc

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
            kind=norm(p.get("bet_code") or p.get("ticket_type") or p.get("bet_type")).lower()
            kind={"exacta":"exacta","2車単":"exacta","二車単":"exacta","trio":"trio","3連複":"trio","三連複":"trio","trifecta_box":"trio"}.get(kind)
            if kind is None: continue
            if p.get("status","paid") not in ("paid",""): continue
            c=tuple(map(int,re.findall(r"\d+",norm(p.get("combination")))))
            v=num(p.get("payout_yen"))
            if len(c)==(2 if kind=="exacta" else 3) and len(set(c))==len(c) and np.isfinite(v) and v>0: pay[p["race_id"]].append((kind,c if kind=="exacta" else tuple(sorted(c)),float(v)))
        for rid,c in cards.items():
            es=ent[rid]; ns=[integer(e.get("car_no")) for e in es]; n=integer(c.get("entry_count")) or len(es)
            if n<5 or len(es)!=n or None in ns or len(set(ns))!=n: audit["bad_entries"]+=1;continue
            if not all(any(np.isfinite(num(e.get(k))) for k in NUM) for e in es): audit["missing_features"]+=1;continue
            f=[(integer(z.get("finish_position")),integer(z.get("car_no"))) for z in fin[rid]]
            one=[a for p,a in f if p==1]; two=[a for p,a in f if p==2]
            three=[a for p,a in f if p==3]
            pp=list(set(pay[rid])); ep=[(c,v) for k,c,v in pp if k=="exacta"]; tp=[(c,v) for k,c,v in pp if k=="trio"]
            if len(one)!=1 or len(two)!=1 or len(three)!=1 or len(ep)!=1 or len(tp)!=1 or ep[0][0]!=(one[0],two[0]) or tp[0][0]!=tuple(sorted(one+two+three)) or not set(one+two+three)<=set(ns): audit["missing_or_nonstandard_result_payout"]+=1;continue
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
                vals=np.array([base[a][k] for a in ns]); good=vals[np.isfinite(vals)]; mean=good.mean() if len(good) else np.nan
                for a,v in zip(ns,vals):
                    base[a][k+"_relative"]=v-mean
                    base[a][k+"_rank"]=(np.sum(good<v)+(np.sum(good==v)+1)/2)/len(good) if np.isfinite(v) else np.nan
            records[rid]=(c["race_date"],base,lines,tuple(one+two+three),ep[0][1],tp[0][1])
    return records,dict(audit)


# Amounts below are in units of 100 yen. Only actual winning payouts are labels.
# A symmetric exacta pair costs two units; an unordered trio costs one.
def split(date):
    if date < "2025-10-01": return "train"
    if date < "2026-01-01": return "early_stop"
    if date < "2026-07-01": return "validation"
    return "test"

FIELDS=NUM+["score_relative","score_rank","top2_rate_relative","top3_rate_relative","line_position","line_size"]
def prepare(rec,arity):
    sizes=[math.comb(len(v[1]),arity) for v in rec.values()]
    # Two group summaries plus group ranges for selected raw card variables.
    width=2*len(FIELDS)+8+7
    X=np.empty((sum(sizes),width),dtype=np.float32)
    y=np.zeros(sum(sizes),dtype=np.float32)
    bounds={}; parts=np.empty(sum(sizes),dtype=np.int8)
    labels={"train":0,"early_stop":1,"validation":2,"test":3}
    off=0
    for rid,(date,base,lines,podium,ep,tp) in rec.items():
        cars=sorted(base)
        keys=list(combinations(cars,arity))
        arr=np.array([[base[a].get(k,np.nan) for k in FIELDS] for a in cars],dtype=np.float32)
        ii=np.array([[cars.index(a) for a in key] for key in keys])
        A=arr[ii]
        # Preserve all combinations, not just high ranking riders.
        valid=np.isfinite(A)
        lo=np.min(np.where(valid,A,np.inf),axis=1)
        hi=np.max(np.where(valid,A,-np.inf),axis=1)
        lo[~np.isfinite(lo)]=np.nan;hi[~np.isfinite(hi)]=np.nan
        field_score=arr[:,FIELDS.index("score")]
        finite=field_score[np.isfinite(field_score)]
        context=[len(cars),base[cars[0]]["race_no"],float(len(set(v[0] for v in lines.values()))) if lines else np.nan,
                 float(np.mean(finite)) if len(finite) else np.nan,float(np.std(finite)) if len(finite) else np.nan,
                 float(np.max(finite)-np.min(finite)) if len(finite) else np.nan,
                 float(sum(1 for a in cars if a in lines and lines[a][1]==1)) if lines else np.nan]
        extra=[]
        for key in keys:
            pairs=list(combinations(key,2))
            same=[int(lines[a][0]==lines[b][0]) for a,b in pairs if a in lines and b in lines]
            adjacent=[int(lines[a][0]==lines[b][0] and abs(lines[a][1]-lines[b][1])==1) for a,b in pairs if a in lines and b in lines]
            extra.append([min(key),max(key),sum(key),sum(same) if len(same)==len(pairs) else np.nan,
                          sum(adjacent) if len(adjacent)==len(pairs) else np.nan,
                          float(len(set(base[a]["style"] for a in key))),float(len(set(base[a]["class"] for a in key))),
                          float(len(set(base[a]["prefecture"] for a in key)))])
        end=off+len(keys)
        X[off:end]=np.column_stack((lo,hi,np.asarray(extra),np.tile(context,(len(keys),1))))
        winner=tuple(sorted(podium[:arity]))
        y[off+keys.index(winner)]=(ep if arity==2 else tp)/100
        parts[off:end]=labels[split(date)]
        if parts[off]>=2: bounds[rid]=(off,end,keys)
        off=end
    return X,y,parts,bounds

def fit_direct(rec,arity):
    X,y,parts,bounds=prepare(rec,arity)
    tr=parts==0;es=parts==1
    # Direct expected payout regression, with real large payouts untrimmed.
    # Every candidate ticket group receives equal weight; fixed model specification.
    reg=CatBoostRegressor(iterations=450,depth=6,learning_rate=.05,loss_function="RMSE",verbose=100,thread_count=4,random_seed=42,allow_writing_files=False)
    reg.fit(X[tr],y[tr],eval_set=(X[es],y[es]),early_stopping_rounds=40)
    reg.save_model(str(OUT/f"direct_{arity}.cbm"))
    ev=parts>=2
    pred=np.zeros(len(y),dtype=np.float32)
    pred[ev]=np.maximum(0,reg.predict(X[ev]))
    diagnostics={"rows":int(len(y)),"train_rows":int(tr.sum()),"early_stop_rows":int(es.sum()),"best_iteration":int(reg.get_best_iteration()),"validation_actual_mean":float(y[parts==2].mean()),"validation_pred_mean":float(pred[parts==2].mean()),"test_actual_mean":float(y[parts==3].mean()),"test_pred_mean":float(pred[parts==3].mean())}
    result={rid:dict(zip(keys,map(float,pred[start:end]))) for rid,(start,end,keys) in bounds.items()}
    del X,y,parts,pred,reg;gc.collect()
    print("DIRECT PAYOUT MODEL",arity,diagnostics,flush=True)
    return result,diagnostics

def candidate(rid,date,kind,axes,opp,tickets,est,podium,ep,tp):
    pts=len(tickets)*(2 if kind==1 else 1)
    result=tuple(sorted(podium[:2 if kind==1 else 3]))
    hit=result in tickets
    payout=(ep if kind==1 else tp) if hit else 0
    return dict(rid=rid,date=date,kind=kind,axes=",".join(map(str,axes)),opponents=",".join(map(str,opp)),stake=100*pts,payout=payout,
                score=est-pts,estimated_roi=100*est/pts,axis_hit=(axes[0] in podium[:2]) if kind==1 else len(set(axes)&set(podium))>=2,
                direction=("front" if axes[0]==podium[0] else "reverse") if hit and kind==1 else "trio" if hit else "miss")

def best_formations(rec,exacta,trio):
    for rid,(date,base,lines,podium,ep,tp) in rec.items():
        if split(date) not in ("validation","test"): continue
        cars=sorted(base)
        for kind,pred in ((1,exacta[rid]),(2,trio[rid]),(3,trio[rid])):
            best=None
            for axes in combinations(cars,kind):
                rest=[a for a in cars if a not in axes]
                if len(rest)<2:continue
                # For a fixed anchor set every extra opponent contributes disjoint tickets.
                # Sorting by predicted marginal payout therefore searches the best subset for every width.
                if kind==1:
                    block=lambda a:{tuple(sorted((axes[0],a)))}
                    core=set()
                else:
                    block=lambda a:{tuple(sorted((*ab,a))) for ab in combinations(axes,2)}
                    core={tuple(axes)} if kind==3 else set()
                blocks={a:block(a) for a in rest}
                scores={a:sum(pred[t] for t in blocks[a]) for a in rest}
                order=sorted(rest,key=lambda a:(-scores[a],a))
                tickets=set(core);est=sum(pred[t] for t in core)
                for k,a in enumerate(order,1):
                    tickets.update(blocks[a]);est+=scores[a]
                    if k<2: continue
                    pts=len(tickets)*(2 if kind==1 else 1)
                    score=est-pts
                    if best is None or score>best["score"]:
                        best=candidate(rid,date,kind,axes,order[:k],tickets,est,podium,ep,tp)
            if best is not None:yield best

def summary(d,total):
    stake=float(d.stake.sum());pay=float(d.payout.sum());hit=d[d.payout>0]
    return dict(races=int(len(d)),coverage=float(len(d)/total) if total else None,stake=stake,payout=pay,profit=pay-stake,
                roi=100*pay/stake if stake else None,hit_rate=float(len(hit)/len(d)) if len(d) else None,
                mean_points=float(d.stake.mean()/100) if len(d) else None,mean_hit_payout=float(hit.payout.mean()) if len(hit) else None,
                front_hits=int((d.direction=="front").sum()),reverse_hits=int((d.direction=="reverse").sum()),
                front_payout=float(d.loc[d.direction=="front","payout"].sum()),reverse_payout=float(d.loc[d.direction=="reverse","payout"].sum()),
                formation_counts={str(k):int(v) for k,v in d.kind.value_counts().items()})

def main():
    rec,audit=load()
    counts=Counter(split(v[0]) for v in rec.values())
    print("DATA",len(rec),audit,"SPLITS",dict(counts),flush=True)
    if min(counts.get(k,0) for k in ("train","early_stop","validation","test"))<100:raise RuntimeError("Insufficient data")
    p2,d2=fit_direct(rec,2);p3,d3=fit_direct(rec,3)
    d=pd.DataFrame(best_formations(rec,p2,p3));d["split"]=d.date.map(split)
    d.to_csv(OUT/"direct_best_candidates.csv",index=False)
    report=dict(counts=dict(counts),audit=audit,diagnostics={"exacta":d2,"trio":d3},policies={},limitations=[
      "All eligible anchor sets evaluated; opponent subsets optimized for each width from marginal direct payout predictions.",
      "Only raw historical racecard fields, no all-combination pre-race odds; no finish probability prefilter.",
      "Historical card timestamp provenance unverified; incident and missing races excluded.",
      "One formation per race per policy, 100 yen per ticket; no money actually wagered.",
      "Threshold chosen on 2026 H1 validation only; 2026 Jul-Oct previously inspected in earlier research, so not a pristine research holdout.",
      "Candidate minimum two opponents and model parameters are design choices. No evidence of future profitability implied."])
    detail=[]
    for name,kinds in {"exacta":[1],"trio_2axis":[2],"trio_3axis":[3],"trio_adaptive":[2,3],"joint":[1,2,3]}.items():
        sub=d[d.kind.isin(kinds)]
        sub=sub.loc[sub.groupby("rid").score.idxmax()].copy()
        va=sub[sub.split=="validation"];te=sub[sub.split=="test"]
        # Threshold grid fixed before viewing test. No-action is an explicit alternative.
        thresholds=[0,.05,.1,.2,.4,.8,1.5,3,5,10,20,50,100]
        trials=[]
        for t in thresholds:
            s=summary(va[va.score>t],len(va));trials.append(dict(threshold=t,**s))
        eligible=[x for x in trials if x["profit"]>0 and x["races"]>=100]
        chosen=max(eligible,key=lambda x:(x["profit"],x["races"]))["threshold"] if eligible else None
        v=va[va.score>chosen] if chosen is not None else va.iloc[:0]
        z=te[te.score>chosen] if chosen is not None else te.iloc[:0]
        sub["selected"]=(sub.score>chosen) if chosen is not None else False
        sub["policy"]=name;detail.append(sub)
        report["policies"][name]={"threshold":chosen,"validation_trials":trials,"all_validation":summary(va,len(va)),
          "all_test":summary(te,len(te)),"selected_validation":summary(v,len(va)),"selected_test":summary(z,len(te)),
          "monthly_test":{m:summary(z[z.date.str[:7]==m],len(g)) for m,g in te.groupby(te.date.str[:7])}}
    pd.concat(detail,ignore_index=True).to_csv(OUT/"direct_policy_detail.csv",index=False)
    (OUT/"report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2))
    # Compact results for logs; full threshold trials remain in artifact.
    compact={k:v for k,v in report.items() if k!="policies"}
    compact["policies"]={k:{a:b for a,b in v.items() if a!="validation_trials"} for k,v in report["policies"].items()}
    print("RESULT",json.dumps(compact,ensure_ascii=False),flush=True)
if __name__=="__main__":main()
