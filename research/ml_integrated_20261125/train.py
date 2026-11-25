"""Chronological research: independent rider history x observed line groups x odds.
Exact trio probabilities; no betting rules. All conditions discovered on validation.
"""
import sys, csv, io, zipfile, itertools, json, math
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"ml_odds_only_20261125"))
import train as odds
OUT=Path("ml-integrated-results"); OUT.mkdir(exist_ok=True)
D=Path("data")
H_NAMES=["independent_min","independent_mean","independent_max","independent_range","history_n_min","history_n_mean","track_rate_mean","type_rate_mean","day_rate_mean","independent_rank_mean","independent_minus_market_mean"]
# Last history column is a cross term, only enabled in the full model.
L_NAMES=["same_line_pairs","max_same_line_count","selected_line_count","line_size_min","line_size_mean","line_size_max","position_min","position_mean","position_max","front_count","second_count","single_count","race_line_count","race_max_line_size"]
I_NAMES=["line_strength_mean","line_strength_max","same_line_independent_pair_mean","line_partner_rate_mean"]
R_NAMES=["rider_count","line_count","max_line_size","market_top_trio","market_entropy","market_disagreement_js","independent_top_rate","independent_spread","independent_market_top_agree","independent_market_abs_gap","strongest_line_rate"]
def load_history():
    d=pd.concat([pd.read_csv(D/"rider_condition_2025_202610_full"/f"rider_races_{y}.csv.gz",dtype={"race_id":str}) for y in [2025,2026]],ignore_index=True)
    assert not d.duplicated(["race_id","car_no"]).any()
    d["race_date"]=pd.to_datetime(d.race_date)
    d["pos"]=pd.to_numeric(d.finish_position,errors="coerce")
    d["y"]=d.pos.between(1,3).astype(int)
    d["valid"]=d.result_status.eq("full_result") & (d.pos.gt(0)|d.finish_label.fillna("").str.contains("失格|落車|棄権"))
    d["win"]=d.pos.eq(1).astype(int);d["second"]=d.pos.eq(2).astype(int);d["third"]=d.pos.eq(3).astype(int)
    g=d.groupby("race_id")
    valid=g.valid.all() & g.win.sum().eq(1)&g.second.sum().eq(1)&g.third.sum().eq(1)
    d=d[d.race_id.isin(valid[valid].index)].copy()
    d["field"]=d.groupby("race_id").car_no.transform("size")
    d["rider"]=d.provisional_player_key.fillna("").astype(str)
    missing=d.rider.eq("")|d["class"].fillna("").eq("")
    d.loc[missing,"rider"]="unknown:"+d.loc[missing,"race_id"]+":"+d.loc[missing,"car_no"].astype(str)
    for c in ["class","track","race_type","meeting_day_no"]:d[c]=d[c].fillna("unknown").astype(str)
    for name,extra in [("base",[]),("track",["track"]),("type",["race_type"]),("day",["meeting_day_no"])]:
        keys=["rider","class"]+extra
        daily=d.groupby(keys+["race_date"],as_index=False).agg(n=("y","size"),hits=("y","sum")).sort_values("race_date")
        for c in ["n","hits"]:daily[c]=daily.groupby(keys)[c].cumsum()-daily[c]
        daily=daily.rename(columns={"n":name+"_n","hits":name+"_hits"})
        d=d.merge(daily,on=keys+["race_date"],validate="many_to_one")
        d[name+"_rate"]=(d[name+"_hits"]+10*3/d.field)/(d[name+"_n"]+10)
    return d
def line_sources():
    paths=[]
    for pat in ["2025/s_class_f1_all_parts/*/entries.csv","2025/s_class_yosen/entries.csv","2026_h1/s_class_f1_all/entries.csv","2026_h1/s_class_yosen/entries.csv","rider_condition_2026_backfill/*/entries.csv","rider_condition_gap_2026/batch_*/entries.csv","a_class_12/*/entries.csv","a3_challenge/*/entries.csv"]:
        paths.extend(D.glob(pat))
    for p in sorted(set(paths)):
        with p.open(encoding="utf-8-sig",newline="") as f:yield str(p),csv.DictReader(f)
    zips=list(D.glob("grade_races/g1g2/*2025*.zip"))+list(D.glob("grade_races/g3/2025/*.zip"))+list(D.glob("grade_races/g3/2026_h1/*.zip"))
    for p in sorted(set(zips)):
        with zipfile.ZipFile(p) as z:
            for member in z.namelist():
                if member.rsplit("/",1)[-1]=="entries.csv":
                    with z.open(member) as b,io.TextIOWrapper(b,encoding="utf-8-sig",newline="") as f:yield str(p)+"!"+member,csv.DictReader(f)
def load_lines(eligible):
    found={};conflict=set();counts=Counter()
    for name,rows in line_sources():
        per=defaultdict(list)
        for r in rows:
            rid=str(r.get("race_id",""))
            if rid not in eligible:continue
            try:
                car=int(r["car_no"]);lid=str(r["line_id"]).strip();pos=int(r["line_position"]);size=int(r["line_size"])
                if not lid or lid in {"0","nan","None"} or pos<1 or size<1:continue
            except (KeyError,ValueError,TypeError):continue
            per[rid].append((car,lid,pos,size))
        for rid,rr in per.items():
            if len(rr)!=len(eligible[rid]) or {r[0] for r in rr}!=eligible[rid]:continue
            by=defaultdict(list)
            for car,lid,pos,size in rr:by[lid].append((car,pos,size))
            if any(sorted(v[1] for v in line)!=list(range(1,len(line)+1)) or any(v[2]!=len(line) for v in line) for line in by.values()):continue
            canonical=tuple(sorted(tuple(v[0] for v in sorted(line,key=lambda v:v[1])) for line in by.values()))
            if rid in found and found[rid]!=canonical:conflict.add(rid)
            else:found[rid]=canonical
            counts["valid_source_races"]+=1
        counts["sources"]+=1
    for rid in conflict:found.pop(rid,None)
    counts["conflicting_races"]=len(conflict);counts["unique_valid_races"]=len(found)
    print("LINE_COVERAGE",dict(counts),flush=True)
    return found,dict(counts)
def pack():
    d=load_history()
    d=d[d.race_date.between("2025-04-01","2026-06-30")].copy()
    # Only raw line sources with actual group IDs are accepted.
    eligible={rid:set(map(int,g.car_no)) for rid,g in d.groupby("race_id")}
    lines,lc=load_lines(eligible)
    records={}
    for rid,g in d[d.race_id.isin(lines)].groupby("race_id"):
        g=g.sort_values("car_no")
        records[rid]=(g.car_no.to_numpy(dtype=int),g[["base_rate","base_n","track_rate","type_rate","day_rate"]].to_numpy(dtype=float),g.race_date.iloc[0].strftime("%Y-%m-%d"),tuple(g.sort_values("pos").car_no.iloc[:3].astype(int)))
    counts=Counter(); packs={k:[] for k in ["train","validation","test"]}
    for year in [2025,2026]:
        truth={}
        for r in odds.read(D/"result_universe_v2/combined"/f"results_{year}.jsonl.gz"):
            try:
                x=tuple(map(int,r["finish_order"]))
                if len(x)==3 and len(set(x))==3:truth[str(r["race_id"])]=x
            except (KeyError,TypeError,ValueError):continue
        seen=set()
        for rec in odds.read(D/"odds_only"/f"odds_{year}.jsonl.gz"):
            rid=str(rec["race_id"]);counts[str(year)+"_odds"]+=1
            if rid in seen:raise ValueError("duplicate race")
            seen.add(rid)
            if rid not in records or rid not in truth:continue
            cars,h,date,actual=records[rid]
            if actual!=truth[rid]:counts["result_conflict"]+=1;continue
            try:
                f=odds.features(rec,truth[rid])
                rawcars=sorted({int(v) for k in rec["odds"]["3連単"] for v in k.split("-")})
            except (KeyError,ValueError,TypeError):continue
            if f is None or list(cars)!=rawcars:counts["invalid_or_car_mismatch"]+=1;continue
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
            split="train" if date<="2025-09-30" else ("validation" if date<="2025-12-31" else "test")
            packs[split].append((rid,date,y,t,a,np.column_stack([x,rh,ll,ii]).astype(np.float32),context,S,n))
            counts[split]+=1
        print("JOIN",year,dict(counts),flush=True)
    def cat(rows):
        assert rows
        rows.sort(key=lambda r:(r[1],r[0]));sizes=np.array([len(r[3]) for r in rows])
        return dict(ids=[r[0] for r in rows],dates=[r[1] for r in rows],s=sizes,yi=np.r_[0,np.cumsum(sizes)[:-1]]+np.array([r[2] for r in rows]),t=np.concatenate([r[3] for r in rows]),a=np.concatenate([r[4] for r in rows]),x=np.concatenate([r[5] for r in rows]),context=np.stack([r[6] for r in rows]),sets=[r[7] for r in rows],n=[r[8] for r in rows])
    return {k:cat(v) for k,v in packs.items()},dict(counts),lc
def model_pred(model,p,cols,shrink):
    return odds.softmax(np.log(p["a"])+shrink*model.predict(p["x"][:,cols],raw_score=True),p["s"])
def conditions(tree,X,names):
    leaf=tree.apply(X);nodes=tree.tree_;rules={}
    def walk(i,rule):
        if nodes.children_left[i]==nodes.children_right[i]:rules[i]=" AND ".join(rule) or "all";return
        f=names[nodes.feature[i]];v=nodes.threshold[i]
        walk(nodes.children_left[i],rule+[f"{f} <= {v:.6g}"]);walk(nodes.children_right[i],rule+[f"{f} > {v:.6g}"])
    walk(0,[])
    return leaf,rules
def main():
    pp,counts,lc=pack();tr,va,te=[pp[k] for k in ["train","validation","test"]]
    assert len(tr["ids"])>=500 and len(va["ids"])>=200 and len(te["ids"])>=500,(counts,lc)
    names=odds.SET_NAMES+H_NAMES+L_NAMES+I_NAMES; o=len(odds.SET_NAMES);h=len(H_NAMES);l=len(L_NAMES)
    models={"odds_only":list(range(o)),"odds_history":list(range(o+h-1)),"odds_line":list(range(o))+list(range(o+h,o+h+l)),"full":list(range(len(names)))}
    vp={};tp={};details={}
    for key,cols in models.items():
        print("FIT",key,flush=True)
        m=odds.fit(tr["x"][:,cols],tr["a"],tr["s"],tr["yi"],[names[i] for i in cols])
        shrink=odds.tune_residual(va["a"],m.predict(va["x"][:,cols],raw_score=True),va["s"],va["yi"])
        vp[key]=model_pred(m,va,cols,shrink);tp[key]=model_pred(m,te,cols,shrink)
        details[key]={"shrink":shrink,"importance":dict(zip([names[i] for i in cols],map(float,m.feature_importance(importance_type="gain"))))}
        m.save_model(str(OUT/(key+".txt")))
    pred={"trio_market":te["t"],"trifecta_aggregate":te["a"],**tp}
    results={k:odds.report(p,te["yi"],te["s"]) for k,p in pred.items()}
    def marginal_metrics(p):
        starts=odds.groups(te["s"]);losses=[];briers=[]
        for i,(start,n,S) in enumerate(zip(starts,te["n"],te["sets"])):
            local=p[start:start+len(S)];r=np.bincount(S.ravel(),weights=np.repeat(local,3),minlength=n)
            y=np.zeros(n);y[S[te["yi"][i]-start]]=1;r=np.clip(r,1e-12,1-1e-12)
            losses.extend(-y*np.log(r)-(1-y)*np.log(1-r));briers.extend((r-y)**2)
        return {"logloss":float(np.mean(losses)),"brier":float(np.mean(briers))}
    result_marginal={k:marginal_metrics(p) for k,p in pred.items()}
    gain_v=odds.loss(vp["odds_only"],va["yi"])-odds.loss(vp["full"],va["yi"])
    gain_t=odds.loss(tp["odds_only"],te["yi"])-odds.loss(tp["full"],te["yi"])
    tree=DecisionTreeRegressor(max_depth=3,min_samples_leaf=max(100,len(gain_v)//10),random_state=42).fit(va["context"],gain_v)
    lv,rules=conditions(tree,va["context"],R_NAMES);lt,_=conditions(tree,te["context"],R_NAMES)
    race_conditions=[]
    for k,rule in rules.items():
        v=lv==k;t=lt==k
        if t.any():race_conditions.append({"rule":rule,"validation_n":int(v.sum()),"validation_gain":float(gain_v[v].mean()),"test_n":int(t.sum()),"test_gain":float(gain_t[t].mean()),"test_ci":odds.ci(gain_t[t],[x for x,b in zip(te["ids"],t) if b])})
    # Candidate rules: shallow surrogate of learned correction, fitted only
    # to validation predictions. Report actual observed frequencies on test.
    cn=["cross_log_ratio","log_trifecta_set_q","independent_min","independent_mean","independent_minus_market_mean","same_line_pairs","line_size_mean","position_mean","line_partner_rate_mean"]
    ix=[names.index(k) for k in cn]
    delta=np.log(vp["full"]/vp["odds_only"])
    ct=DecisionTreeRegressor(max_depth=3,min_samples_leaf=max(500,len(delta)//12),random_state=42).fit(va["x"][:,ix],delta)
    cv,cr=conditions(ct,va["x"][:,ix],cn);ctest,_=conditions(ct,te["x"][:,ix],cn)
    y=np.zeros(len(te["a"]));y[te["yi"]]=1
    cand=[]
    for k,rule in cr.items():
        v=cv==k;t=ctest==k
        if t.any():cand.append({"rule":rule,"validation_n":int(v.sum()),"validation_mean_log_adjustment":float(delta[v].mean()),"test_n":int(t.sum()),"actual_hit_rate":float(y[t].mean()),"odds_ml_mean_probability":float(tp["odds_only"][t].mean()),"full_ml_mean_probability":float(tp["full"][t].mean()),"actual_hits":int(y[t].sum())})
    months={}
    for month in sorted(set(x[:7] for x in te["dates"])):
        mask=np.array([x.startswith(month) for x in te["dates"]])
        months[month]={k:float(odds.loss(p,te["yi"])[mask].mean()) for k,p in pred.items()}
    out={"counts":counts,"line_coverage":lc,"split_counts":{k:len(v["ids"]) for k,v in pp.items()},"trio":results,"rider_top3_marginals":result_marginal,"ci_full_vs":{k:odds.ci(odds.loss(p,te["yi"])-odds.loss(tp["full"],te["yi"]),te["ids"]) for k,p in pred.items() if k!="full"},"months":months,"race_conditions":race_conditions,"candidate_conditions":cand,"models":details,"limitations":["Research independent rate = same exact class prior-day cumulative top3 smoothed by 10 field-average pseudo-observations; not verified identical to live app","Only complete verified line group sources and exact race/car/podium matches","2025 Apr-Sep train, Oct-Dec validation, 2026 Jan-Jun test; Jan-Mar 2025 history warm-up","Final odds only; no live execution or return evaluated","Conditions are exploratory validation-tree rules, test subgroup CIs not multiplicity adjusted","Trio joint model and derived rider top3 probabilities only; exact-order model not part of this run"]}
    (OUT/"report.json").write_text(json.dumps(out,ensure_ascii=False,indent=2))
    # Human-readable summary and all learned condition groups.
    with (OUT/"conditions.md").open("w",encoding="utf-8") as f:
        f.write("# Integrated ML: condition report\n\n")
        for k in ["split_counts","trio","rider_top3_marginals","ci_full_vs","months","race_conditions","candidate_conditions","limitations"]:
            f.write("## "+k+"\n\n```json\n"+json.dumps(out[k],ensure_ascii=False,indent=2)+"\n```\n\n")
    print("RESULT",json.dumps(out,ensure_ascii=False),flush=True)
if __name__=="__main__":main()
