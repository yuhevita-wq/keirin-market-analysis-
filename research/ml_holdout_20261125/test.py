import sys, gzip, csv, json, math
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np
import pandas as pd
import lightgbm as lgb
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"ml_integrated_20261125"))
import importlib.util
spec=importlib.util.spec_from_file_location("integrated_frozen",Path(__file__).resolve().parents[1]/"ml_integrated_20261125/train.py")
integrated=importlib.util.module_from_spec(spec);spec.loader.exec_module(integrated)
# train imports odds module under a conflicting name; explicitly load correct
odds=integrated.odds
D=Path("data")
F=D/"2026_future_block1/s_class_f1_20260701_20260830"
B=D/"odds_universe_v1/data/2026_future_block1/s_class_f1_20260701_20260830"
OUT=Path("ml-holdout-results");OUT.mkdir(exist_ok=True)
M=Path("frozen-models")
def load_market():
    books=defaultdict(lambda:{"3連単":{},"3連複":{}})
    bad=set()
    for batch in sorted(B.glob("batch_*")):
        for kind,name in [("3連単","trifecta"),("3連複","trio")]:
            with gzip.open(batch/(name+"_final_odds.csv.gz"),"rt",encoding="utf-8-sig",newline="") as f:
                for row in csv.DictReader(f):
                    rid=row["race_id"]
                    try:
                        val=float(row["odds"]);key=row["combination"]
                        if not math.isfinite(val) or val<1 or row.get("odds_status","available")!="available":raise ValueError()
                        if kind=="3連複":key="=".join(map(str,sorted(map(int,key.split("=")))))
                        if key in books[rid][kind] and books[rid][kind][key]!=val:raise ValueError()
                        books[rid][kind][key]=val
                    except (ValueError,KeyError):bad.add(rid)
    return {r:b for r,b in books.items() if r not in bad}
def main():
    d=integrated.load_history()
    d=d[d.race_date.between("2026-07-01","2026-08-30")].copy()
    eligible={rid:set(map(int,g.car_no)) for rid,g in d.groupby("race_id")}
    old=integrated.line_sources
    def extra():
        yield from old()
        with (F/"entries.csv").open(encoding="utf-8-sig",newline="") as f:yield str(F/"entries.csv"),csv.DictReader(f)
    integrated.line_sources=extra
    lines,lc=integrated.load_lines(eligible)
    records={}
    for rid,g in d[d.race_id.isin(lines)].groupby("race_id"):
        g=g.sort_values("car_no")
        records[rid]=(g.car_no.to_numpy(dtype=int),g[["base_rate","base_n","track_rate","type_rate","day_rate"]].to_numpy(dtype=float),g.race_date.iloc[0].strftime("%Y-%m-%d"),tuple(g.sort_values("pos").car_no.iloc[:3].astype(int)))
    payout=defaultdict(list)
    with (F/"payouts.csv").open(encoding="utf-8-sig",newline="") as f:
        for row in csv.DictReader(f):
            if row["ticket_type"]!="3連複" or row["status"]!="paid":continue
            try:
                k=tuple(sorted(map(int,row["combination"].replace("-","=").split("="))))
                v=int(row["payout_yen"].replace(",",""))
                if len(k)==3 and v>0:payout[row["race_id"]].append((k,v))
            except ValueError:continue
    report=json.loads((M/"report.json").read_text())
    model=lgb.Booster(model_file=str(M/"full.txt"))
    om=lgb.Booster(model_file=str(M/"odds_only.txt"))
    shrink=report["models"]["full"]["shrink"];oshrink=report["models"]["odds_only"]["shrink"]
    names=odds.SET_NAMES+integrated.H_NAMES+integrated.L_NAMES+integrated.I_NAMES
    assert model.feature_name()==names
    books=load_market();counts=Counter(source=len(books)); rows=[]
    for rid,b in sorted(books.items()):
        if rid not in records:counts["missing_history_or_line"]+=1;continue
        cars,h,date,actual=records[rid]
        if date<"2026-07-01":raise ValueError("not holdout")
        pr=payout.get(rid,[])
        if len(pr)!=1 or pr[0][0]!=tuple(sorted(actual)):counts["payout_missing_or_conflict"]+=1;continue
        rec={"race_id":rid,"odds":b}
        try:f=odds.features(rec,actual)
        except (ValueError,KeyError,TypeError):f=None
        rawcars=sorted({int(v) for k in b["3連単"] for v in k.split("-")})
        if f is None or list(cars)!=rawcars:counts["invalid_market"]+=1;continue
        x,_,t,a,_,y,_,n,_=f
        S,_=odds.layout(cars);r=h[:,0];nr=h[:,1]
        index={int(c):i for i,c in enumerate(cars)}
        lid=np.zeros(n,dtype=int);size=np.zeros(n,dtype=int);pos=np.zeros(n,dtype=int);strength=np.zeros(n);partner=np.zeros(n)
        for j,ln in enumerate(lines[rid]):
            ix=np.array([index[c] for c in ln]);lid[ix]=j;size[ix]=len(ix);pos[ix]=np.arange(1,len(ix)+1)
            strength[ix]=r[ix].mean();partner[ix]=(r[ix].sum()-r[ix])/(len(ix)-1) if len(ix)>1 else r[ix]
        marginal=np.bincount(S.ravel(),weights=np.repeat(a,3),minlength=n)
        rs=r[S];ns=nr[S];ls=lid[S];sz=size[S];ps=pos[S]
        same=np.stack([ls[:,i]==ls[:,j] for i,j in [(0,1),(0,2),(1,2)]],axis=1)
        sc=same.sum(1); maxc=np.where(sc==3,3,np.where(sc==1,2,1))
        m=len(S);rep=lambda z:np.full(m,z)
        rh=np.column_stack([rs.min(1),rs.mean(1),rs.max(1),np.ptp(rs,axis=1),ns.min(1),ns.mean(1),h[:,2][S].mean(1),h[:,3][S].mean(1),h[:,4][S].mean(1),odds.ranks(r)[S].mean(1),(r-marginal)[S].mean(1)])
        ll=np.column_stack([sc,maxc,3-(sc>0)-(sc==3),sz.min(1),sz.mean(1),sz.max(1),ps.min(1),ps.mean(1),ps.max(1),(ps==1).sum(1),(ps==2).sum(1),(sz==1).sum(1),rep(len(lines[rid])),rep(size.max())])
        pair_r=np.stack([(rs[:,i]+rs[:,j])/2 for i,j in [(0,1),(0,2),(1,2)]],axis=1)
        ii=np.column_stack([strength[S].mean(1),strength[S].max(1),(pair_r*same).sum(1)/np.maximum(sc,1),partner[S].mean(1)])
        context=np.array([n,len(lines[rid]),size.max(),a.max(),odds.entropy(a),x[0,22],r.max(),np.ptp(r),float(r.argmax()==marginal.argmax()),np.mean(np.abs(r-marginal)),strength.max()])

        X=np.column_stack([x,rh,ll,ii]).astype(np.float32)
        pp=odds.softmax(np.log(a)+shrink*model.predict(X,raw_score=True),np.array([len(a)]))
        op=odds.softmax(np.log(a)+oshrink*om.predict(x,raw_score=True),np.array([len(a)]))
        entropy=context[4];js=context[5];top=context[6];line=context[10]
        cat="C1" if entropy<=2.93393 and line>0.503298 and top>0.620582 else ("C2" if 2.93393<entropy<=3.03585 else ("C3" if entropy>3.03585 and js>0.00229799 else "outside"))
        prices=np.array([b["3連複"]["=".join(map(str,cars[s]))] for s in S])
        # Fixed illustrative selection rules, never tune them on the holdout.
        choices={"full_top3":np.argsort(-pp,kind="stable")[:3],"odds_top3":np.argsort(-op,kind="stable")[:3],"market_top3":np.argsort(-t,kind="stable")[:3],"full_positive_ev":np.flatnonzero(pp*prices>1),"odds_positive_ev":np.flatnonzero(op*prices>1)}
        item={"race_id":rid,"date":date,"condition":cat,"entropy":float(entropy),"js":float(js),"top_rate":float(top),"line_rate":float(line),"actual":"=".join(map(str,sorted(actual))),"payout":pr[0][1],"gain":float(np.log(pp[y]/op[y]))}
        for policy,ix in choices.items():
            hit=int(y in ix);item[policy]={"tickets":len(ix),"hit":hit,"stake":100*len(ix),"return":pr[0][1]*hit,"selections":["=".join(map(str,cars[S[i]])) for i in ix]}
        rows.append(item)
    counts["accepted"]=len(rows)
    assert rows,counts
    def summ(sub,pol):
        a=[r[pol] for r in sub];stake=sum(x["stake"] for x in a);ret=sum(x["return"] for x in a);bought=sum(x["tickets"]>0 for x in a);hit=sum(x["hit"] for x in a)
        return {"eligible_races":len(sub),"bought_races":bought,"tickets":sum(x["tickets"] for x in a),"hits":hit,"stake":stake,"return":ret,"profit":ret-stake,"roi_pct":100*ret/stake if stake else None,"hit_pct":100*hit/bought if bought else None}
    out={"counts":dict(counts),"line_coverage":lc,"dates":[min(r["date"] for r in rows),max(r["date"] for r in rows)],"groups":{},"limitations":["Final odds are not executable pre-close prices","Sample restricted to available S-class F1 July odds, not project-wide untouched","Fixed prior conditions, no re-fitting","Top 3 by probability and all positive model-implied EV are illustrative fixed choices; 100 yen per ticket; no selection after results"]}
    for name in ["C1","C2","C3","combined","outside","all"]:
        subset=[r for r in rows if name=="all" or (name=="combined" and r["condition"]!="outside") or r["condition"]==name]
        out["groups"][name]={p:summ(subset,p) for p in choices}
    (OUT/"report.json").write_text(json.dumps(out,ensure_ascii=False,indent=2))
    (OUT/"race_details.json").write_text(json.dumps(rows,ensure_ascii=False,indent=2))
    print("RESULT",json.dumps(out,ensure_ascii=False),flush=True)
if __name__=="__main__":main()
