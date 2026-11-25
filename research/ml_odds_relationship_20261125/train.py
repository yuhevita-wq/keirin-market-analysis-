"""Historical odds calibration and formation pricing, not live execution."""
import sys, json, csv, gc, math
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
import importlib.util
spec=importlib.util.spec_from_file_location("line_prior",ROOT/"ml_line_arrival_structure_20261125"/"train.py")
line=importlib.util.module_from_spec(spec);spec.loader.exec_module(line)
odds=line.odds
OUT=Path("ml-odds-relationship-results");OUT.mkdir(exist_ok=True)
MARKET=["log_q","log_other_q","log_cross_ratio","log_price","price_rank","pool_entropy","other_entropy","inverse_sum","field_size","log_structure_q","log_structure_other_q","log_structure_count","log_structure_combined_odds"]
STRUCT=["rank_1","rank_2","rank_3","role_1","role_2","role_3","size_1","size_2","size_3","same_12","same_13","same_23","line_count","largest_line","front_count","single_count"]
HIST=["rate_min","rate_mean","rate_max","rate_spread","partner_mean"]
NAMES=MARKET+STRUCT+HIST
EDGES=np.array([1.,2.,3.,5.,10.,20.,30.,50.,100.,200.,500.,1000.,np.inf])
def candidates(rec,pool):
    n=len(rec["cars"]);S,O=odds.layout(rec["cars"])
    ordered=pool=="trifecta";C=O if ordered else S
    qt=odds.norm(1/rec["oa"]);qu=odds.norm(1/rec["ob"]);ag=qt.reshape(-1,6).sum(1)
    if ordered:
        price=rec["oa"];q=qt;other=(qu[:,None]*(qt.reshape(-1,6)/ag[:,None])).ravel();hit=rec["hit"]
    else:
        price=rec["ob"];q=qu;other=ag;hit=rec["hit"]//6
    ls=rec["lid"][C];ps=rec["pos"][C];sz=rec["size"][C];rs=rec["rank"][C]
    rr=rec["r"][C];pr=rec["partner"][C]
    if not ordered:
        ix=np.lexsort((ps,rs),axis=1)
        ls=np.take_along_axis(ls,ix,axis=1);ps=np.take_along_axis(ps,ix,axis=1)
        sz=np.take_along_axis(sz,ix,axis=1);rs=np.take_along_axis(rs,ix,axis=1)
    keys=rs[:,0]*100+rs[:,1]*10+rs[:,2]
    ids,gi=np.unique(keys,return_inverse=True)
    mass=np.bincount(gi,weights=q);omass=np.bincount(gi,weights=other)
    cnt=np.bincount(gi);inv=np.bincount(gi,weights=1/price)
    rep=lambda x:np.full(len(q),x)
    market=np.column_stack([np.log(q),np.log(other),np.log(other/q),np.log(price),odds.ranks(q),rep(odds.entropy(q)),rep(odds.entropy(other)),rep((1/price).sum()),rep(n),np.log(mass[gi]),np.log(omass[gi]),np.log(cnt[gi]),-np.log(inv[gi])])
    struct=np.column_stack([rs,ps,sz,ls[:,0]==ls[:,1],ls[:,0]==ls[:,2],ls[:,1]==ls[:,2],rep(len(set(rec["lid"]))),rep(max(rec["size"])),(ps==1).sum(1),(sz==1).sum(1)])
    history=np.column_stack([rr.min(1),rr.mean(1),rr.max(1),np.ptp(rr,axis=1),pr.mean(1)])
    assert len(price)==len(q) and np.isclose(q.sum(),1) and np.isclose(other.sum(),1)
    return dict(x=np.column_stack([market,struct,history]).astype(np.float32),q=q,other=other,price=price,hit=hit,gi=gi,keys=ids,shape=rec["shape"],rid=rec["rid"],date=rec["date"])
def pack(rr):
    s=np.array([len(r["q"]) for r in rr]);start=odds.groups(s)
    return dict(s=s,q=np.concatenate([r["q"] for r in rr]),other=np.concatenate([r["other"] for r in rr]),x=np.concatenate([r["x"] for r in rr]),yi=start+np.array([r["hit"] for r in rr]),ids=[r["rid"] for r in rr])
def model_pred(p,score,scale):
    return odds.softmax(np.log(p["other"])+scale*score,p["s"])
def binprice(o):
    i=int(np.searchsorted(EDGES,o,side="right")-1)
    i=max(0,min(i,len(EDGES)-2));return f"{EDGES[i]:g}-{EDGES[i+1]:g}"
def add(agg,key,q,p,price,y,date):
    # One nominal unit per candidate or per complete structural group, no stake optimization.
    a=agg.setdefault(key,[0,0,0.,0.,0.,0.,set(),defaultdict(lambda:[0,0.])])
    a[0]+=1;a[1]+=y;a[2]+=q;a[3]+=p;a[4]+=1/price;a[5]+=y*price;a[6].add(date)
    a[7][date][0]+=1;a[7][date][1]+=y*price
def summarize(agg,path):
    with path.open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.writer(f);w.writerow(["split","unit","shape","structure","odds_band","observations","hits","market_expected_hits","model_expected_hits","break_even_expected_hits","actual_frequency","market_probability","model_probability","break_even_probability","actual_over_market","actual_over_model","historical_return_pct","return_ci_low_pct","return_ci_high_pct","days"])
        for k,a in sorted(agg.items()):
            n,h,q,p,be,ret,days,cl=a
            mean=ret/n;g=len(days);se=math.sqrt(g/max(g-1,1)*sum((v[1]-mean*v[0])**2 for v in cl.values()))/n
            w.writerow([*k,n,h,q,p,be,h/n,q/n,p/n,be/n,h/q if q else "",h/p if p else "",100*mean,100*(mean-1.96*se),100*(mean+1.96*se),g])
def audit(rr,p,pred,split,agg,pool):
    for start,r in zip(odds.groups(p["s"]),rr):
        q=r["q"];o=r["price"];v=pred[start:start+len(q)];y=np.zeros(len(q),dtype=int);y[r["hit"]]=1
        gi=r["gi"];keys=r["keys"]
        for j in range(len(q)):
            st=str(keys[gi[j]]);b=binprice(o[j])
            for shape,st2 in [("all","all"),(r["shape"],st)]:
                add(agg,(split,"ticket",shape,st2,b),float(q[j]),float(v[j]),float(o[j]),int(y[j]),r["date"])
        # Full market-ranked line sequence / unordered line multiset.
        # Equal-return allocation within each group: one nominal unit total.
        mq=np.bincount(gi,weights=q);mp=np.bincount(gi,weights=v);inv=np.bincount(gi,weights=1/o);hy=np.bincount(gi,weights=y).astype(int)
        for j,k in enumerate(keys):
            co=1/inv[j];b=binprice(co)
            for shape,st in [("all","all"),(r["shape"],str(k))]:
                add(agg,(split,"complete_line_structure",shape,st,b),float(mq[j]),float(mp[j]),float(co),int(hy[j]),r["date"])
def experiment(raw,pool):
    rr={k:[] for k in ("train","validation","test")}
    for r in raw:rr[r["split"]].append(candidates(r,pool))
    pp={k:pack(v) for k,v in rr.items()};tr,va,te=[pp[k] for k in ("train","validation","test")]
    for p in pp.values():assert p["x"].shape[1]==len(NAMES)
    m=len(MARKET);s=m+len(STRUCT)
    # Baseline includes both pools, while structure adds roles/rank interactions.
    # Group price features are reserved for structure models; market-only uses no grouping.
    cols={"market_only":list(range(9)),"market_line":list(range(s)),"market_line_history":list(range(len(NAMES)))}
    vp={"raw":va["q"],"cross_pool":va["other"]};tp={"raw":te["q"],"cross_pool":te["other"]};info={}
    for name,ix in cols.items():
        print("FIT",pool,name,flush=True)
        model=odds.fit(tr["x"][:,ix],tr["other"],tr["s"],tr["yi"],[NAMES[j] for j in ix])
        a=model.predict(va["x"][:,ix],raw_score=True);b=model.predict(te["x"][:,ix],raw_score=True)
        shrink=odds.tune_residual(va["other"],a,va["s"],va["yi"])
        vp[name]=model_pred(va,a,shrink);tp[name]=model_pred(te,b,shrink)
        info[name]={"shrink":shrink,"importance":dict(zip([NAMES[j] for j in ix],map(float,model.feature_importance(importance_type="gain"))))}
        model.save_model(str(OUT/f"{pool}_{name}.txt"))
        del model;gc.collect()
    validation={k:odds.report(v,va["yi"],va["s"]) for k,v in vp.items()}
    best=min(cols,key=lambda k:validation[k]["logloss"])
    test={k:odds.report(v,te["yi"],te["s"]) for k,v in tp.items()}
    paired={k:odds.ci(odds.loss(tp["market_only"],te["yi"])-odds.loss(v,te["yi"]),te["ids"]) for k,v in tp.items() if k!="market_only"}
    out={"races":{k:len(v["ids"]) for k,v in pp.items()},"validation":validation,"test":test,"chosen":best,"paired_vs_market_only":paired,"models":info}
    agg={}
    # No price threshold is selected on test. Summaries show the complete universe.
    for split,p,pred in [("validation",va,vp[best]),("test",te,tp[best])]:
        audit(rr[split],p,pred,split,agg,pool)
    summarize(agg,OUT/f"{pool}_price_structure_calibration.csv")
    print("ODDS_RESULT",pool,json.dumps(out,ensure_ascii=False),flush=True)
    # Print aggregate price-band diagnostics to run logs for direct inspection.
    for k,a in sorted(agg.items()):
        if k[2]!="all":continue
        n,h,q,p,be,ret,days,cl=a
        print("PRICE_BAND",pool,*k,n,h,round(h/q,5) if q else None,round(h/p,5) if p else None,round(ret/n*100,3),flush=True)
    return out
def main():
    raw,cov,lc=line.records()
    out={"coverage":cov,"line_coverage":lc,"results":{},"definitions":{"train":"2025-04 to 2026-03","validation":"2026-04","test":"2026-05 to 06, previously researched","price":"complete final odds, not executable pre-close snapshots","line_rank":"sum of trifecta implied top-three marginals","structure":"three digits of market-ranked lines; sorted for trio, ordered for trifecta","formation":"all tickets for the same ranked line structure, synthetic equal-return odds=1/sum reciprocal price","historical_return":"nominal historical return, not actual live return or exact rounded stake allocation","limits":"No profit-maximizing selection on test; intervals approximate day-clustered and unadjusted; independent rate research proxy; later complete odds not available."}}
    for pool in ("trio","trifecta"):
        out["results"][pool]=experiment(raw,pool);gc.collect()
        (OUT/"report.json").write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding="utf-8")
    print("COMPLETE",flush=True)
if __name__=="__main__":main()
