"""Two-pillar learning from raw racecards, results and payouts only."""
import csv, json, re, hashlib, unicodedata
from itertools import combinations
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


def split(date):
    if date < "2025-07-01": return "rider_train"
    if date < "2026-01-01": return "formation_train"
    if date < "2026-07-01": return "validation"
    return "test"

def summarize(d):
    if not len(d): return {}
    stake=float(d.stake.sum()); pay=float(d.payout.sum())
    return dict(races=len(d),stake=stake,payout=pay,profit=pay-stake,roi=100*pay/stake,hit_rate=float((d.payout>0).mean()),mean_points=float((d.stake/100).mean()),axis_hit_rate=float(d.axis_hit.mean()),front_hits=int((d.direction=="front").sum()),reverse_hits=int((d.direction=="reverse").sum()),front_payout=float(d.loc[d.direction=="front","payout"].sum()),reverse_payout=float(d.loc[d.direction=="reverse","payout"].sum()))

def formation_rows(rec, probs):
    # Each candidate retains exacta symmetry, or at least two of the chosen trio axes.
    for rid,(date,base,lines,podium,ep,tp) in rec.items():
        if split(date)=="rider_train": continue
        pr=probs[rid]; cars=sorted(base); n=len(cars)
        top2={a:pr[a][0]+pr[a][1] for a in cars}
        top3={a:top2[a]+pr[a][2] for a in cars}
        rank2=sorted(cars,key=lambda a:(-top2[a],a))
        rank3=sorted(cars,key=lambda a:(-top3[a],a))
        for kind in (1,2,3):
            rank=rank2 if kind==1 else rank3
            q=top2 if kind==1 else top3
            # All single anchors, all pairs of the top three, all triples of the top four.
            axis_sets=[(a,) for a in cars] if kind==1 else combinations(rank[:kind+1],kind)
            for axes in axis_sets:
                rest=[a for a in rank if a not in axes]
                for k in range(2,len(rest)+1):
                    opp=rest[:k]
                    if kind==1:
                        a=axes[0]
                        tickets={(a,b) for b in opp}|{(b,a) for b in opp}
                        hit=podium[:2] in tickets
                        payout=ep if hit else 0
                        axis_hit=a in podium[:2]
                        direction=("front" if a==podium[0] else "reverse") if hit else "miss"
                    else:
                        tickets={tuple(sorted((*ab,c))) for ab in combinations(axes,2) for c in (*axes,*opp) if c not in ab}
                        hit=tuple(sorted(podium)) in tickets
                        payout=tp if hit else 0
                        axis_hit=len(set(axes)&set(podium))>=2
                        direction="trio" if hit else "miss"
                    pts=len(tickets)
                    aq=sorted([q[a] for a in axes],reverse=True)+[0]*3
                    oq=[q[a] for a in opp]
                    pairline=[float(lines[a][0]==lines[b][0]) for a,b in combinations(axes,2) if a in lines and b in lines]
                    x=dict(kind=kind,n=n,k=k,points=pts,axis_p1=aq[0],axis_p2=aq[1],axis_p3=aq[2],axis_sum=sum(q[a] for a in axes),
                           axis_win=max(pr[a][0] for a in axes),axis_second=max(pr[a][1] for a in axes),
                           opponent_sum=sum(oq),opponent_min=min(oq),opponent_max=max(oq),
                           gap12=q[rank[0]]-q[rank[1]],gap23=q[rank[1]]-q[rank[2]],gap34=q[rank[2]]-q[rank[3]],
                           axis_rank_sum=sum(rank.index(a)+1 for a in axes),same_line=np.mean(pairline) if pairline else np.nan)
                    yield dict(x,rid=rid,date=date,split=split(date),axes=",".join(map(str,axes)),opponents=",".join(map(str,opp)),stake=100*pts,payout=payout,axis_hit=axis_hit,direction=direction,target=payout/(100*pts))

def choose(d, kinds, score):
    sub=d[d.kind.isin(kinds)]
    return sub.loc[sub.groupby("rid")[score].idxmax()]

def main():
    rec,audit=load()
    counts=Counter(split(v[0]) for v in rec.values())
    print("DATA",len(rec),audit,"SPLITS",dict(counts),flush=True)
    if min(counts.get(k,0) for k in ("rider_train","formation_train","validation","test"))<100: raise RuntimeError("Insufficient common exacta/trio data")
    riders=[]
    for rid,(date,base,lines,podium,ep,tp) in rec.items():
        for a,x in base.items():
            riders.append(dict(x,rid=rid,date=date,a=a,y=podium.index(a)+1 if a in podium else 0))
    d=pd.DataFrame(riders);del riders
    train=d.date<"2025-07-01"
    cols=[c for c in d if c not in {"rid","date","a","y"} and d.loc[train,c].nunique(dropna=True)>1]
    cats=[c for c in cols if not pd.api.types.is_numeric_dtype(d[c])]
    m=CatBoostClassifier(iterations=350,depth=6,learning_rate=.07,verbose=100,thread_count=4,random_seed=42,allow_writing_files=False)
    m.fit(d.loc[train,cols],d.loc[train,"y"],cat_features=cats)
    m.save_model(str(OUT/"rider.cbm"))
    (OUT/"rider_features.json").write_text(json.dumps(dict(columns=cols,categorical=cats)))
    ev=d.loc[~train,["rid","a","y","date"]].copy()
    pred=m.predict_proba(d.loc[~train,cols])
    for i in (1,2,3): ev["p"+str(i)]=pred[:,list(m.classes_).index(i)]
    diagnostics={}
    for name,g in ev.groupby(ev.date.map(split)):
        diagnostics[name]=dict(top2_brier=float(brier_score_loss(g.y.isin([1,2]),g.p1+g.p2)),top3_brier=float(brier_score_loss(g.y.isin([1,2,3]),g.p1+g.p2+g.p3)))
    probs=defaultdict(dict)
    for z in ev.itertuples(index=False): probs[z.rid][z.a]=(z.p1,z.p2,z.p3)
    del d,ev,pred
    print("RIDER TRAIN COMPLETE",diagnostics,flush=True)
    # Keep only small numeric arrays during fitting; no per-combination object feature matrices.
    batches=[]; buf=[]
    for row in formation_rows(rec,probs):
        buf.append(row)
        if len(buf)==20000:
            batches.append(pd.DataFrame(buf));buf=[]
    if buf:batches.append(pd.DataFrame(buf))
    c=pd.concat(batches,ignore_index=True);del batches,buf,rec,probs
    fcols=[k for k in c if k not in {"rid","date","split","axes","opponents","stake","payout","axis_hit","direction","target"}]
    X=c[fcols].astype("float32")
    tr=c.split=="formation_train"; va=c.split=="validation"
    reg=CatBoostRegressor(iterations=400,depth=6,learning_rate=.05,verbose=100,thread_count=4,random_seed=42,allow_writing_files=False)
    # Each race has equal total training weight regardless of how many formations exist.
    wt=1/c.groupby("rid").rid.transform("size")
    reg.fit(X.loc[tr],c.loc[tr,"target"],sample_weight=wt[tr],eval_set=(X.loc[va],c.loc[va,"target"]),early_stopping_rounds=40)
    reg.save_model(str(OUT/"formation_return.cbm"))
    (OUT/"formation_features.json").write_text(json.dumps(fcols))
    c["prediction"]=reg.predict(X);del X
    c["prob_baseline"]=c.axis_sum
    report=dict(counts=dict(counts),audit=audit,diagnostics=diagnostics,policies={},monthly={},limitations=["No complete odds used; historical payout is outcome only.","Retrospective racecard timestamp provenance unverified.","Only common valid exacta and trio races; excluded incidents may bias sample.","Candidate ranges and model hyperparameters are engineering choices, not discovered facts.","No betting advice or guarantee of future profitability."])
    selected=[]
    for name,kinds in {"exacta":[1],"trio_2axis":[2],"trio_3axis":[3],"trio_adaptive":[2,3],"joint":[1,2,3]}.items():
        for part in ("validation","test"):
            v=c[c.split==part]; g=choose(v,kinds,"prediction").copy()
            g["policy"]=name;selected.append(g)
            summary=summarize(g);summary["formation_counts"]={str(k):int(x) for k,x in g.kind.value_counts().items()}
            report["policies"][name+"/"+part]=summary
            report["monthly"][name+"/"+part]={m:summarize(h) for m,h in g.groupby(g.date.str[:7])}
    detail=pd.concat(selected,ignore_index=True)
    detail.to_csv(OUT/"race_detail.csv",index=False)
    c[c.split.isin(["validation","test"])].groupby(["split","kind","k"]).agg(candidates=("rid","size")).to_csv(OUT/"candidate_counts.csv")
    (OUT/"report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2,default=lambda v:v.item()))
    print(json.dumps(report,ensure_ascii=False,default=lambda v:v.item()),flush=True)
if __name__=="__main__":main()
