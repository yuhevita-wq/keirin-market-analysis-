"""Learn interpretable anchor-level market residuals, never monetary decisions."""
import gzip,json,math,re
from pathlib import Path
from itertools import combinations,permutations
from collections import Counter
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression
from scipy.special import logit,expit
from scipy.stats import norm
OUT=Path("market_edge_output");OUT.mkdir(exist_ok=True)
SRC=Path("data/rider_condition_2025_202610_full")
AUD=Counter()
def load():
    rr=pd.concat([pd.read_csv(SRC/f"rider_races_{y}.csv.gz",dtype=str) for y in (2025,2026)],ignore_index=True)
    rc=pd.concat([pd.read_csv(SRC/f"races_{y}.csv.gz",dtype=str) for y in (2025,2026)],ignore_index=True)
    valid=set(rc.loc[rc.verified_podium.str.lower().isin(["true","1","yes"]),"race_id"])
    rr=rr[rr.race_id.isin(valid)&rr.result_status.eq("full_result")].copy()
    rr["car"]=pd.to_numeric(rr.car_no,errors="coerce")
    rr["pos"]=pd.to_numeric(rr.finish_position,errors="coerce")
    rr["key"]=rr.provisional_player_key.fillna("")
    rr["cls"]=rr["class"].fillna("")
    rr=rr[rr.race_date.between("2025-01-01","2026-06-30")]
    rr["starts"]=1;rr["top2"]=rr.pos.isin([1,2]).astype(int);rr["top3"]=rr.pos.isin([1,2,3]).astype(int)
    keys=["key","cls","race_date"];cols=["starts","top2","top3"]
    history=rr[(rr.key!="")&(rr.cls!="")&rr.pos.notna()].groupby(keys)[cols].sum().reset_index().sort_values(keys)
    history[cols]=history.groupby(keys[:2])[cols].cumsum()-history[cols]
    history=history.rename(columns={k:"prior_"+k for k in cols})
    rr=rr.merge(history,on=keys,how="left",validate="many_to_one")
    cards={}
    for rid,g in rr.groupby("race_id",sort=False):
        if g.car.isna().any() or g.car.duplicated().any() or g.race_date.nunique()!=1 or not 5<=len(g)<=9 or not all((g.pos==i).sum()==1 for i in (1,2,3)):continue
        cards[rid]=g
    return cards
def book(raw,cars,k,ordered):
    target=set(permutations(cars,k) if ordered else combinations(cars,k));out={}
    for key,v in raw.items():
        t=tuple(map(int,re.findall(r"\d+",str(key))))
        if not ordered:t=tuple(sorted(t))
        try:v=float(v)
        except (ValueError,TypeError):return None
        if t not in target or t in out or not np.isfinite(v) or v<1:return None
        out[t]=1/v
    if set(out)!=target:return None
    z=sum(out.values())
    return {k:v/z for k,v in out.items()}
def numeric(v):
    try:return float(v)
    except (ValueError,TypeError):return np.nan
def feats(g,cars):
    f={}
    for row in g.to_dict("records"):
        a=int(row["car"]);n=numeric(row["prior_starts"]);t2=numeric(row["prior_top2"]);t3=numeric(row["prior_top3"])
        f[a]=dict(history_n=n,history_top2=(t2+1)/(n+7) if np.isfinite(n) else np.nan,
                  history_top3=(t3+1)/(n+7) if np.isfinite(n) else np.nan,
                  term=numeric(row.get("term")),line_position=numeric(row.get("line_position")),line_size=numeric(row.get("line_size")),
                  front=float("逃" in str(row.get("style"))),chase=float("追" in str(row.get("style"))),car=a)
    return f
def build(cards):
    rows={1:[],2:[],3:[]};seen=set()
    for year in (2025,2026):
        with gzip.open(Path("data/odds_only")/f"odds_{year}.jsonl.gz","rt",encoding="utf-8") as file:
            for line in file:
                obj=json.loads(line);rid=obj["race_id"]
                if rid not in cards or rid in seen:continue
                g=cards[rid];cars=tuple(sorted(map(int,g.car)));podium=tuple(int(g.loc[g.pos==i,"car"].iloc[0]) for i in (1,2,3))
                odds=obj["odds"];three=book(odds.get("3連単",{}),cars,3,True);trio=book(odds.get("3連複",{}),cars,3,False)
                if three is None or trio is None:AUD["incomplete_book"]+=1;continue
                seen.add(rid);date=g.race_date.iloc[0];f=feats(g,cars)
                p2={a:sum(p for t,p in three.items() if a in t[:2]) for a in cars}
                p3={a:sum(p for t,p in trio.items() if a in t) for a in cars}
                pair3={ab:sum(p for t,p in trio.items() if set(ab)<=set(t)) for ab in combinations(cars,2)}
                ent=-sum(p*math.log(p) for p in trio.values())/math.log(len(trio))
                for kind in (1,2,3):
                    for axes in combinations(cars,kind):
                        q=p2[axes[0]] if kind==1 else pair3[axes] if kind==2 else sum(pair3[ab] for ab in combinations(axes,2))-2*trio[axes]
                        hit=int(axes[0] in podium[:2]) if kind==1 else int(len(set(axes)&set(podium))>=2)
                        x=dict(rid=rid,date=date,kind=kind,q=q,y=hit,n=len(cars),market_entropy=ent,
                               market_top2_max=max(p2.values()),market_top3_max=max(p3.values()),
                               market_anchor_min=min((p2 if kind==1 else p3)[a] for a in axes),
                               market_anchor_max=max((p2 if kind==1 else p3)[a] for a in axes))
                        for k in f[cars[0]]:
                            z=[f[a][k] for a in axes if np.isfinite(f[a][k])]
                            x[k+"_min"]=min(z) if z else -1
                            x[k+"_max"]=max(z) if z else -1
                        rows[kind].append(x)
    AUD["matched_races"]=len(seen)
    return {k:pd.DataFrame(v) for k,v in rows.items()}
def baseline_matrix(d):
    q=np.clip(d.q.to_numpy(),1e-6,1-1e-6);l=logit(q);n=d.n.to_numpy()
    return np.column_stack([l,l*l,l*l*l,n,l*n])
def stats(g):
    if not len(g):return None
    e=g.y-g.cal;daily=pd.DataFrame({"date":g.date,"e":e}).groupby("date").e.agg(["sum","size"])
    # Cluster robust SE for ratio mean. Candidate outcomes within race/day not independent.
    n=len(g);mu=float(e.mean());D=len(daily)
    se=float(np.sqrt(D/(D-1)*np.sum((daily["sum"]-mu*daily["size"])**2))/n) if D>1 else None
    return dict(candidates=n,races=int(g.rid.nunique()),days=D,raw_market=float(g.q.mean()),calibrated_market=float(g.cal.mean()),actual=float(g.y.mean()),
                raw_gap=float((g.y-g.q).mean()),gap=mu,se=se,ci95=[mu-1.96*se,mu+1.96*se] if se is not None else None,
                p_two=float(2*norm.sf(abs(mu/se))) if se and se>0 else 1.0)
def loss(y,p):
    p=np.clip(p,1e-8,1-1e-8);return float(np.mean(-y*np.log(p)-(1-y)*np.log(1-p)))
def paths(tree,cols):
    out={}
    def visit(i,path):
        t=tree.tree_;f=t.feature[i]
        if f<0:out[i]=path;return
        name=cols[f];z=float(t.threshold[i])
        visit(t.children_left[i],path+[dict(feature=name,op="<=",value=z)])
        visit(t.children_right[i],path+[dict(feature=name,op=">",value=z)])
    visit(0,[]);return out
def study(d,kind):
    a=d.date<"2025-07-01";b=(d.date>="2025-07-01")&(d.date<"2026-01-01")
    v=(d.date>="2026-01-01")&(d.date<"2026-04-01");t=d.date>="2026-04-01"
    if min(d.loc[m,"rid"].nunique() for m in (a,b,v,t))<300:raise RuntimeError("insufficient dates")
    # Calibrate market using only earlier races. Tree learns residuals, not winners or payout.
    cal=LogisticRegression(max_iter=500,C=1).fit(baseline_matrix(d[a]),d.loc[a,"y"])
    d["cal"]=cal.predict_proba(baseline_matrix(d))[:,1]
    # Market shape and independent rider data are eligible explanatory factors.
    cols=[c for c in d if c not in {"rid","date","kind","y","cal"}]
    d[cols]=d[cols].replace([np.inf,-np.inf],np.nan).fillna(-1)
    # Approximate Newton step for offset logistic loss; deliberately shallow interpretable model.
    q=d.loc[b,"cal"].to_numpy();w=np.maximum(q*(1-q),.01)
    tree=DecisionTreeRegressor(max_depth=3,min_samples_leaf=max(500,int(b.sum()*.025)),random_state=42)
    tree.fit(d.loc[b,cols],(d.loc[b,"y"].to_numpy()-q)/w,sample_weight=w)
    d["leaf"]=tree.apply(d[cols]);d["correction"]=tree.predict(d[cols])
    d["model"]=expit(logit(np.clip(d.cal,1e-6,1-1e-6))+d.correction)
    out=[]
    for leaf,rule in paths(tree,cols).items():
        z=d[d.leaf==leaf]
        out.append(dict(kind=kind,leaf=int(leaf),rule=rule,discovery=stats(z[b.loc[z.index]]),
                        confirmation=stats(z[v.loc[z.index]]),test=stats(z[t.loc[z.index]])))
    diag={}
    for name,m in [("calibration",a),("discovery",b),("confirmation",v),("test",t)]:
        z=d[m];diag[name]=dict(races=int(z.rid.nunique()),raw_logloss=loss(z.y,z.q),market_logloss=loss(z.y,z.cal),residual_logloss=loss(z.y,z.model))
    return out,diag
def main():
    data=build(load());print("AUDIT",dict(AUD),flush=True)
    results=[];diag={}
    for k,d in data.items():
        print("ANCHOR",k,len(d),flush=True);rules,dd=study(d,k);results+=rules;diag[k]=dd
    # Fixed family correction across all discovered leaves, separately in each later period.
    n=len(results)
    for z in results:
        a=z["confirmation"];b=z["test"];c=z["discovery"]
        def stable(s):
            return s is not None and s["races"]>=100 and s["days"]>=30 and s["p_two"]*n<.05
        z["confirmed"]=bool(stable(a) and stable(b) and c and np.sign(a["gap"])==np.sign(b["gap"])==np.sign(c["gap"]))
        z["direction"]="underestimated" if z["confirmed"] and b["gap"]>0 else "overestimated" if z["confirmed"] else "unconfirmed"
    report=dict(audit=dict(AUD),diagnostics=diag,multiple_tests=n,confirmed=[z for z in results if z["confirmed"]],all_rules=results,limitations=[
        "Final odds only, not actionable pre-close odds. Normalized inverse odds approximate market beliefs.",
        "Exacta anchor market is a trifecta marginal, not an actual exacta market; trio uses direct trio market.",
        "2025 H1 market calibration, 2025 H2 discovery, 2026 Q1 confirmation and Q2 test; all dates previously available to researchers, not a pristine future test.",
        "Confirmed residuals are probability advantages over calibrated market, not proof of profit or causation.",
        "Day cluster uncertainty and Bonferroni across leaves; historical rider fields and archived predicted lines have incomplete as-of provenance.",
        "No ticket creation, monetary selection, payout optimization or site changes."])
    (OUT/"report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False))
    print("RESULT",json.dumps(report,ensure_ascii=False,allow_nan=False),flush=True)
if __name__=="__main__":main()
