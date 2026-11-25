"""Line-arrival-structure research: complete candidate universe, chronological group likelihood.
No live betting recommendations; final odds describe historical prices only.
"""
import sys, json, math, csv, gc
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"ml_integrated_20261125"))
import importlib.util
spec=importlib.util.spec_from_file_location("integrated_prior",ROOT/"ml_integrated_20261125"/"train.py")
integ=importlib.util.module_from_spec(spec);spec.loader.exec_module(integ)
odds=integ.odds;D=Path("data")
OUT=Path("ml-line-structure-results");OUT.mkdir(exist_ok=True)
SPLITS=("train","validation","test")
MARKET=["log_group_mass","log_group_candidate_count","log_price_min","log_price_mean","log_price_max","price_log_sd","pool_entropy","field_size"]
STRUCT=["line_count","max_line_size","line_size_1","line_size_2","line_size_3","same_line_pairs","same_line_12","same_line_13","same_line_23","occupied_lines","role_1","role_2","role_3","rank_1","rank_2","rank_3","rank_min","rank_max","rank_mean","group_single_count","front_count","second_count","third_or_later_count"]
HIST=["rate_min","rate_mean","rate_max","rate_range","partner_rate_mean"]
def records():
    d=integ.load_history()
    d=d[d.race_date.between("2025-04-01","2026-06-30")].copy()
    eligible={rid:set(map(int,g.car_no)) for rid,g in d.groupby("race_id")}
    lines,lc=integ.load_lines(eligible)
    rows={}
    for rid,g in d[d.race_id.isin(lines)].groupby("race_id"):
        g=g.sort_values("car_no")
        rows[rid]=(g.car_no.to_numpy(dtype=int),g.base_rate.to_numpy(dtype=float),g.race_date.iloc[0].strftime("%Y-%m-%d"),tuple(g.sort_values("pos").car_no.iloc[:3].astype(int)))
    coverage=Counter();out=[]
    for year in (2025,2026):
        truth={}
        for r in odds.read(D/"result_universe_v2/combined"/f"results_{year}.jsonl.gz"):
            try:
                v=tuple(map(int,r["finish_order"]))
                if len(v)==3 and len(set(v))==3:truth[str(r["race_id"])]=v
            except (KeyError,ValueError,TypeError):continue
        seen=set()
        for rec in odds.read(D/"odds_only"/f"odds_{year}.jsonl.gz"):
            rid=str(rec["race_id"]);coverage["odds_"+str(year)]+=1
            if rid in seen:raise ValueError("duplicate odds race")
            seen.add(rid)
            if rid not in rows or rid not in truth:continue
            cars,r,date,actual=rows[rid]
            if actual!=truth[rid]:coverage["result_conflict"]+=1;continue
            try:
                a=rec["odds"]["3連単"];b=rec["odds"]["3連複"];n=len(cars)
                if not 5<=n<=9 or len(a)!=n*(n-1)*(n-2) or len(b)!=math.comb(n,3):raise ValueError()
                S,O=odds.layout(cars)
                oa=np.array([a["-".join(map(str,cars[x]))] for x in O],dtype=float)
                ob=np.array([b["=".join(map(str,cars[x]))] for x in S],dtype=float)
                if not(np.isfinite(oa).all() and np.isfinite(ob).all() and (oa>=1).all() and (ob>=1).all()):raise ValueError()
                hits=np.flatnonzero((cars[O]==np.array(actual)).all(1))
                if len(hits)!=1:raise ValueError()
            except (KeyError,ValueError,TypeError):coverage["invalid_odds"]+=1;continue
            index={int(c):i for i,c in enumerate(cars)}
            lid=np.zeros(n,dtype=int);pos=np.zeros(n,dtype=int);size=np.zeros(n,dtype=int);partner=np.zeros(n)
            for j,ln in enumerate(lines[rid]):
                ix=np.array([index[c] for c in ln]);lid[ix]=j;pos[ix]=np.arange(1,len(ix)+1);size[ix]=len(ix)
                partner[ix]=(r[ix].sum()-r[ix])/(len(ix)-1) if len(ix)>1 else r[ix]
            # Common pre-race line rank from normalized trifecta top-three marginal
            # sum per line; market ties deterministic by original line index.
            q=odds.norm(1/oa);mr=np.bincount(O.ravel(),weights=np.repeat(q,3),minlength=n)
            line_mass=np.bincount(lid,weights=mr)
            rank=np.argsort(np.argsort(-line_mass,kind="stable"),kind="stable")[lid]+1
            shape="+".join(map(str,sorted([len(x) for x in lines[rid]],reverse=True)))
            split="train" if date<="2026-03-31" else ("validation" if date<="2026-04-30" else "test")
            out.append(dict(rid=rid,date=date,split=split,cars=cars,r=r,lid=lid,pos=pos,size=size,partner=partner,rank=rank,shape=shape,oa=oa,ob=ob,hit=int(hits[0])))
            coverage[split]+=1
        print("JOIN",year,dict(coverage),flush=True)
    return out,dict(coverage),lc
def key_of(ls,ps,sz,rs,ordered,ranked):
    # Anonymous line identities are canonicalized; rank variant preserves
    # market-ranked identities. Role+size distinguishes front/second/single.
    if ordered:
        seen={}
        ids=[int(r) if ranked else seen.setdefault(int(l),len(seen)+1) for l,r in zip(ls,rs)]
        return "/".join(f"{l}:{int(p)}:{int(s)}" for l,p,s in zip(ids,ps,sz))
    by=defaultdict(list)
    for l,p,s,r in zip(ls,ps,sz,rs):by[int(r) if ranked else int(l)].append((int(p),int(s)))
    blocks=[(l,tuple(sorted(v))) for l,v in by.items()]
    if ranked:blocks.sort()
    else:blocks=sorted((0,v) for _,v in blocks)
    return "/".join((str(l)+":" if ranked else "")+"+".join(f"{p}:{s}" for p,s in v) for l,v in blocks)
def simple_pattern(ls,ordered):
    if not ordered:return "+".join(map(str,sorted(Counter(map(int,ls)).values(),reverse=True)))
    seen={};return "".join("ABC"[seen.setdefault(int(l),len(seen))] for l in ls)
def one_race(rec,pool,ranked):
    ordered=pool=="trifecta";n=len(rec["cars"]);S,O=odds.layout(rec["cars"])
    C=O if ordered else S;price=rec["oa"] if ordered else rec["ob"];hit=rec["hit"] if ordered else rec["hit"]//6
    q=odds.norm(1/price);logs=np.log(price)
    ls=rec["lid"][C];ps=rec["pos"][C];sz=rec["size"][C];rs=rec["rank"][C]
    rr=rec["r"][C];pr=rec["partner"][C]
    same=(ls[:,0]==ls[:,1]).astype(int)+(ls[:,0]==ls[:,2]).astype(int)+(ls[:,1]==ls[:,2]).astype(int)
    ixsort=np.tile(np.arange(3),(len(C),1)) if ordered else np.lexsort((ps,rs),axis=1)
    pp=np.take_along_axis(ps,ixsort,axis=1)
    rk=np.take_along_axis(rs,ixsort,axis=1)
    ss=np.take_along_axis(sz,ixsort,axis=1)
    ll=np.take_along_axis(ls,ixsort,axis=1)
    flags=np.column_stack([ll[:,0]==ll[:,1],ll[:,0]==ll[:,2],ll[:,1]==ll[:,2]])
    sf=np.column_stack([np.full(len(C),len(set(rec["lid"]))),np.full(len(C),max(rec["size"])),ss,same,flags,3-(same==1).astype(int)-2*(same==3).astype(int),pp,rk,rs.min(1),rs.max(1),rs.mean(1),(sz==1).sum(1),(ps==1).sum(1),(ps==2).sum(1),(ps>=3).sum(1)])
    hf=np.column_stack([rr.min(1),rr.mean(1),rr.max(1),np.ptp(rr,axis=1),pr.mean(1)])
    groups=defaultdict(list)
    for j in range(len(C)):groups[key_of(ls[j],ps[j],sz[j],rs[j],ordered,ranked)].append(j)
    probs=[];features=[];keys=[];actual_group=-1;counts=[];pattern=[];price_summary=[];hit_cond=0.
    ent=odds.entropy(q)
    for i,(key,idx) in enumerate(groups.items()):
        ix=np.array(idx);w=q[ix];mass=w.sum();w=w/mass
        a=logs[ix]
        mf=[math.log(mass),math.log(len(ix)),a.min(),a.mean(),a.max(),a.std(),ent,n]
        # market-weighted summaries characterize a structural group before results.
        features.append(np.r_[mf,np.average(sf[ix],axis=0,weights=w),np.average(hf[ix],axis=0,weights=w)])
        probs.append(mass);keys.append(key);counts.append(len(ix));pattern.append(simple_pattern(ls[ix[0]],ordered))
        price_summary.append((float(price[ix].min()),float(price[ix].max()),float(np.median(price[ix])),float(price[ix].sum())))
        if hit in idx:actual_group=i;hit_cond=float(q[hit]/mass)
    assert actual_group>=0 and np.isclose(sum(probs),1.) and sum(counts)==len(C)
    return dict(q=np.array(probs),x=np.array(features,dtype=np.float32),y=actual_group,keys=keys,pattern=pattern,counts=counts,price_summary=price_summary,winning_price=float(price[hit]),hit_cond=hit_cond,rid=rec["rid"],date=rec["date"],shape=rec["shape"])
def pack(rows):
    sizes=np.array([len(r["q"]) for r in rows]);start=odds.groups(sizes)
    return dict(s=sizes,q=np.concatenate([r["q"] for r in rows]),x=np.concatenate([r["x"] for r in rows]),yi=start+np.array([r["y"] for r in rows]),ids=[r["rid"] for r in rows],dates=[r["date"] for r in rows])
def descriptive(rows,pool,kind):
    # Universe includes all possible groups in each race, not just winners.
    agg={}
    for split,rr in rows.items():
        for r in rr:
            for i,k in enumerate(r["keys"]):
                for level,code in [("pattern",r["pattern"][i]),("role_structure",k)]:
                    a=agg.setdefault((split,r["shape"],level,code),[set(),0,0,0.,0,0.,[],[]])
                    a[0].add(r["rid"]);a[1]+=1;a[2]+=r["counts"][i];a[3]+=r["q"][i];a[5]+=r["price_summary"][i][3];a[6].append(r["price_summary"][i][2])
                    if i==r["y"]:a[4]+=1;a[7].append(r["winning_price"])
    path=OUT/f"{pool}_{kind}_structure_stats.csv"
    with path.open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.writer(f);w.writerow(["split","line_shape","level","structure","eligible_races","group_occurrences","candidate_count","market_expected_hits","actual_hits","market_probability_per_eligible_race","actual_frequency_per_eligible_race","mean_candidate_odds","median_group_median_odds","winning_odds_median","winning_odds_p25","winning_odds_p75"])
        for (split,shape,level,code),a in sorted(agg.items()):
            ids,ng,nc,exp,h,price,med,win=a;n=len(ids)
            w.writerow([split,shape,level,code,n,ng,nc,exp,h,exp/n,h/n,price/nc,np.median(med),*(np.quantile(win,[.5,.25,.75]) if win else ["","",""])])
def experiment(raw,pool,kind):
    rows={k:[] for k in SPLITS}
    for rec in raw:rows[rec["split"]].append(one_race(rec,pool,kind=="ranked"))
    descriptive(rows,pool,kind)
    pp={k:pack(v) for k,v in rows.items()};tr,va,te=[pp[k] for k in SPLITS]
    assert len(tr["ids"])>=500 and len(va["ids"])>=200 and len(te["ids"])>=500
    assert tr["x"].shape[1]==len(MARKET)+len(STRUCT)+len(HIST)
    names=MARKET+STRUCT+HIST;m=len(MARKET);s=m+len(STRUCT)
    cols={"market_control":list(range(m)),"structure_only":list(range(m,s)),"structure_market":list(range(s)),"structure_market_history":list(range(len(names)))}
    vp={"raw_market":va["q"]};tp={"raw_market":te["q"]};model_info={}
    for tag,ix in cols.items():
        print("FIT",pool,kind,tag,flush=True)
        model=odds.fit(tr["x"][:,ix],tr["q"],tr["s"],tr["yi"],[names[j] for j in ix])
        v=model.predict(va["x"][:,ix],raw_score=True);t=model.predict(te["x"][:,ix],raw_score=True)
        shrink=odds.tune_residual(va["q"],v,va["s"],va["yi"])
        vp[tag]=odds.softmax(np.log(va["q"])+shrink*v,va["s"])
        tp[tag]=odds.softmax(np.log(te["q"])+shrink*t,te["s"])
        model_info[tag]={"shrink":shrink,"importance":dict(zip([names[j] for j in ix],map(float,model.feature_importance(importance_type="gain"))))}
        model.save_model(str(OUT/f"{pool}_{kind}_{tag}.txt"))
        del model;gc.collect()
    val={k:odds.report(v,va["yi"],va["s"]) for k,v in vp.items()}
    test={k:odds.report(v,te["yi"],te["s"]) for k,v in tp.items()}
    best=min(("structure_only","structure_market","structure_market_history"),key=lambda k:val[k]["logloss"])
    control=odds.loss(tp["market_control"],te["yi"])
    paired={k:odds.ci(control-odds.loss(v,te["yi"]),te["ids"]) for k,v in tp.items() if k!="market_control"}
    # A common exact-ticket metric across grouping resolutions: redistribute
    # group probability inside each group according to original market weights.
    cond=-np.log(np.array([r["hit_cond"] for r in rows["test"]]))
    joint={k:float((odds.loss(v,te["yi"])+cond).mean()) for k,v in tp.items()}
    monthly={}
    for mo in sorted({d[:7] for d in te["dates"]}):
        mask=np.array([d.startswith(mo) for d in te["dates"]])
        monthly[mo]={"races":int(mask.sum()),"group_logloss":{k:float(odds.loss(v,te["yi"])[mask].mean()) for k,v in tp.items()}}
    # Fixed structure taxonomy, observed/expected and predictions for every
    # held-out group, so one can audit calibration without selected winners.
    for split,p,preds in [("validation",va,vp),("test",te,tp)]:
        with (OUT/f"{pool}_{kind}_{split}_groups.csv").open("w",encoding="utf-8",newline="") as f:
            w=csv.writer(f);w.writerow(["race_id","date","shape","pattern","role_structure","candidate_count","actual","observed_winning_odds","median_candidate_odds"]+list(preds))
            for j,(start,r) in enumerate(zip(odds.groups(p["s"]),rows[split])):
                for i,k in enumerate(r["keys"]):
                    hit=int(i==r["y"])
                    w.writerow([r["rid"],r["date"],r["shape"],r["pattern"][i],k,r["counts"][i],hit,r["winning_price"] if hit else "",r["price_summary"][i][2]]+[float(v[start+i]) for v in preds.values()])
    out={"races":{k:len(v["ids"]) for k,v in pp.items()},"mean_groups":{k:float(v["s"].mean()) for k,v in pp.items()},"selected_by_validation":best,"validation":val,"test":test,"paired_vs_market_control":paired,"exact_ticket_logloss_market_within_group":joint,"months":monthly,"models":model_info}
    print("STRUCTURE_RESULT",pool,kind,json.dumps(out,ensure_ascii=False),flush=True)
    return out
def main():
    raw,cov,lc=records()
    assert len({r["rid"] for r in raw})==len(raw)
    out={"coverage":cov,"line_coverage":lc,"results":{},"definitions":{"anonymous":"canonical same-line pattern + role and line size, without market-ranked line identity","ranked":"same role/size but line identity replaced by pre-race market rank (sum of trifecta-implied top-three marginal probabilities)","trio":"unordered; canonical line role blocks","trifecta":"ordered; role and line size for each finishing place","market":"normalized reciprocal complete final odds for the same pool, aggregated over all tickets of the group","history":"prior-day same-class research proxy, not verified live formula","dates":"train 2025-04 to 2026-03; validation 2026-04; evaluation 2026-05 to 06, previously researched","limits":"No live odds, no financial-return simulation, no actual formation construction; missing or conflicting lines excluded. Reported subgroup frequencies descriptive, not causal."}}
    for pool in ("trio","trifecta"):
        for kind in ("anonymous","ranked"):
            out["results"][pool+"_"+kind]=experiment(raw,pool,kind);gc.collect()
            (OUT/"report.json").write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding="utf-8")
    print("COMPLETE",json.dumps({"coverage":cov,"experiments":list(out["results"])},ensure_ascii=False),flush=True)
if __name__=="__main__":main()
