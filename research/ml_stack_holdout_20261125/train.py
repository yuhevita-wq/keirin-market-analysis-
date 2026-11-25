"""Chronological integration and first post-June partial holdout. Historical simulation only."""
import importlib.util,json,csv,math,gc
from pathlib import Path
from collections import defaultdict,Counter
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.special import expit,logit
from scipy.optimize import minimize_scalar
ROOT=Path(__file__).resolve().parents[1]
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,ROOT/path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
form=module("formation","ml_formation_20261125/train.py")
rel=module("reliability","ml_race_reliability_20261125/train.py")
prior=form.prior;line=prior.line;odds=prior.odds
D=Path("data");OUT=Path("ml-stack-holdout-results");OUT.mkdir(exist_ok=True)
def clip(p):return np.clip(np.asarray(p,dtype=np.float64),1e-9,1-1e-9)
form.clip=clip
def optimize(f,bounds):
    r=minimize_scalar(f,bounds=bounds,method="bounded");return min([bounds[0],bounds[1],float(r.x)],key=f)
def future():
    d=line.integ.load_history()
    d=d[d.race_date.between("2026-07-01","2026-10-31")]
    path=D/"2026_future_block1/s_class_f1_20260701_20260830/entries.csv"
    entries=pd.read_csv(path,dtype={"race_id":str})
    groups={rid:g for rid,g in entries.groupby("race_id")}
    hist={rid:g.sort_values("car_no") for rid,g in d.groupby("race_id")}
    truth={}
    for r in odds.read(D/"result_universe_v2/combined/results_2026.jsonl.gz"):
        try:
            v=tuple(map(int,r["finish_order"]))
            if len(v)==3 and len(set(v))==3:truth[str(r["race_id"])]=v
        except (KeyError,ValueError,TypeError):pass
    books=defaultdict(lambda: [dict(),dict()]);bad=set()
    dirs=sorted((D/"all_classes/odds_s_class/2026_future_block1").rglob("summary.json"))
    for p in dirs:
        for k,name in enumerate(("trifecta_final_odds.csv.gz","trio_final_odds.csv.gz")):
            file=p.parent/name
            if not file.exists():continue
            for r in pd.read_csv(file,dtype={"race_id":str}).to_dict("records"):
                rid=r["race_id"]
                try:
                    sep="-" if k==0 else "="
                    c=tuple(map(int,str(r["combination"]).split(sep)))
                    if len(c)!=3 or len(set(c))!=3:raise ValueError()
                    if k:c=tuple(sorted(c))
                    v=float(r["odds"])
                    if not math.isfinite(v) or v<1:raise ValueError()
                    if c in books[rid][k] and books[rid][k][c]!=v:raise ValueError()
                    books[rid][k][c]=v
                except (ValueError,TypeError,KeyError):bad.add(rid)
    counts=Counter(odds_races=len(books));out=[]
    for rid,(a,b) in sorted(books.items()):
        if rid in bad:counts["invalid_price"]+=1;continue
        if rid not in hist or rid not in groups or rid not in truth:counts["missing_match"]+=1;continue
        g=hist[rid];e=groups[rid];cars=g.car_no.to_numpy(dtype=int);n=len(cars)
        if n<5 or n>9 or len(e)!=n or set(e.car_no)!=set(cars):counts["car_mismatch"]+=1;continue
        actual=tuple(g.sort_values("pos").car_no.iloc[:3].astype(int))
        if actual!=truth[rid]:counts["result_conflict"]+=1;continue
        try:
            assert e[["line_id","line_position","line_size"]].notna().all().all()
            ln=[]
            for _,gg in e.groupby("line_id"):
                gg=gg.sort_values("line_position")
                assert list(map(int,gg.line_position))==list(range(1,len(gg)+1))
                assert all(gg.line_size==len(gg))
                ln.append(tuple(map(int,gg.car_no)))
            assert sum(map(len,ln))==n
            S,O=odds.layout(cars)
            assert len(a)==len(O) and len(b)==len(S)
            oa=np.array([a[tuple(cars[x])] for x in O]);ob=np.array([b[tuple(cars[x])] for x in S])
            hit=int(np.flatnonzero((cars[O]==actual).all(1))[0])
        except (AssertionError,KeyError,ValueError,IndexError,TypeError):counts["bad_line_or_book"]+=1;continue
        index={c:i for i,c in enumerate(cars)};r=g.base_rate.to_numpy(float)
        lid=np.zeros(n,int);pos=np.zeros(n,int);size=np.zeros(n,int);partner=np.zeros(n)
        for j,l in enumerate(sorted(ln)):
            ix=np.array([index[c] for c in l]);lid[ix]=j;pos[ix]=np.arange(1,len(ix)+1);size[ix]=len(ix)
            partner[ix]=(r[ix].sum()-r[ix])/(len(ix)-1) if len(ix)>1 else r[ix]
        q=odds.norm(1/oa);mr=np.bincount(O.ravel(),weights=np.repeat(q,3),minlength=n)
        rank=np.argsort(np.argsort(-np.bincount(lid,weights=mr),kind="stable"),kind="stable")[lid]+1
        date=g.race_date.iloc[0].strftime("%Y-%m-%d")
        out.append(dict(rid=rid,date=date,split="holdout",cars=cars,r=r,lid=lid,pos=pos,size=size,partner=partner,rank=rank,shape="+".join(map(str,sorted(map(len,ln),reverse=True))),oa=oa,ob=ob,hit=hit))
        counts["accepted"]+=1
    print("HOLDOUT_COVERAGE",json.dumps(dict(counts)),flush=True)
    return out,dict(counts)
# Refit the previously researched feature families on strictly earlier data.
BASES={"market":list(range(9)),"line":list(range(len(prior.MARKET)+len(prior.STRUCT))),"history":list(range(len(prior.NAMES)))}
def base_predict(train,pred,tag):
    tr=prior.pack(train);pp=prior.pack(pred);v={}
    for name,ix in BASES.items():
        model=odds.fit(tr["x"][:,ix],tr["other"],tr["s"],tr["yi"],[prior.NAMES[j] for j in ix])
        p=odds.softmax(np.log(pp["other"])+model.predict(pp["x"][:,ix],raw_score=True),pp["s"])
        v[name]=[p[i:i+n] for i,n in zip(odds.groups(pp["s"]),pp["s"])]
        model.save_model(str(OUT/f"{tag}_{name}.txt"))
        del model;gc.collect()
    return v
def race_features(raw):
    X=[];Y=defaultdict(list);B=defaultdict(list);names=None
    for r in raw:
        f,mc,t=rel.race(r)
        if names is None:names=list(f)
        X.append(list(f.values()))
        for k,(y,b) in t.items():Y[k].append(y);B[k].append(b)
    return np.array(X,dtype=np.float32),{k:np.array(v,float) for k,v in Y.items()},{k:np.array(v,float) for k,v in B.items()},names
def reliability_predict(X,Y,B,names,ti,vi,tag):
    ret=[]
    for key in Y:
        y=Y[key];b=B[key];binary=not key.endswith("surprise")
        if binary:
            offset=logit(clip(b))
            bias=optimize(lambda a:float(rel.binaryloss(y[ti],expit(offset[ti]+a)).mean()),(-4.,4.))
            init=offset+bias;label=y[ti]
        else:
            bias=float(y[ti].mean());init=np.full(len(y),bias);label=y[ti]-bias
        ds=lgb.Dataset(X[ti],label=label,init_score=init[ti] if binary else None,feature_name=names)
        m=lgb.train(dict(objective="binary" if binary else "regression",metric="None",learning_rate=.03,num_leaves=12,min_data_in_leaf=200,lambda_l2=10,num_threads=4,verbosity=-1,seed=42),ds,num_boost_round=100)
        raw=m.predict(X[vi],raw_score=True)
        p=expit(init[vi]+raw) if binary else init[vi]+raw
        ret.append(p-b[vi])
        m.save_model(str(OUT/f"{tag}_{key}.txt"))
    return np.column_stack(ret)
def member_ids(rec,pool,label):
    S,O=odds.layout(rec["cars"]);C=rec["cars"][O];z=np.zeros(len(O),bool)
    for part in label.split(" + "):
        a=part.split("-");assert len(a)==3
        mask=np.ones(len(O),bool)
        for j,ax in enumerate(a):mask&=np.isin(C[:,j],list(map(int,ax)))
        z|=mask
    ix=np.flatnonzero(z)
    return ix if pool=="trifecta" else np.unique(ix//6)
EXTRA=["market_model_mass","history_model_mass","market_model_return","history_model_return"]
def collect(raw,rows,indices,preds,rp,pool):
    out=[]
    for j,i in enumerate(indices):
        r=raw[i];X,y,m,ng=form.forms(r,rows[i],preds["line"][j],pool)
        a=[]
        for meta in m:
            ix=member_ids(r,pool,meta[6]);assert len(ix)==meta[0]
            pm=float(preds["market"][j][ix].sum());ph=float(preds["history"][j][ix].sum())
            a.append([pm,ph,pm*meta[1],ph*meta[1],*rp[j]])
        out.append(dict(x=np.column_stack([X,np.array(a)]).astype(np.float32),y=y,m=m,date=r["date"],rid=r["rid"],shape=r["shape"],generated=ng))
    return out
def fit_meta(train,val,names,ix):
    X,y,w=form.concat(train);V,z,vw=form.concat(val)
    ds=lgb.Dataset(X[:,ix],label=y,weight=w,init_score=logit(clip(X[:,1])),feature_name=[names[i] for i in ix])
    m=lgb.train(dict(objective="binary",metric="None",learning_rate=.03,num_leaves=15,min_data_in_leaf=300,lambda_l2=10,num_threads=4,verbosity=-1,seed=42),ds,num_boost_round=120)
    v=m.predict(V[:,ix],raw_score=True)
    def loss(a):
        p=expit(logit(clip(V[:,1]))+a*v)
        return float(np.average(rel.binaryloss(z,p),weights=vw))
    s=optimize(loss,(0.,1.));return m,s,loss(s)
def pred(m,s,ix,r):
    return expit(logit(clip(r["x"][:,1]))+s*m.predict(r["x"][:,ix],raw_score=True,num_threads=1))
def evaluate(rows,models,chosen,pool,split):
    # One candidate per race, all 2-48 generated points permitted. No thresholds
    # selected against the holdout. Break-even is a mathematical threshold, not
    # a claim of real edge or a live betting instruction.
    bets=defaultdict(list)
    for r in rows:
        co=np.array([m[1] for m in r["m"]]);X=r["x"]
        probs={"raw_market":X[:,2],"line_base":X[:,1]}
        for key,(m,s,ix) in models.items():probs[key]=pred(m,s,ix,r)
        probs["selected"]=probs[chosen]
        for key,p in probs.items():
            j=int(np.argmax(p*co));a=r["m"][j];ev=float(p[j]*co[j])
            bets[key+"_all"].append((r,a,float(p[j]),ev,True))
            bets[key+"_break_even"].append((r,a,float(p[j]),ev,ev>1))
    out={}
    with (OUT/f"{pool}_{split}_race_audit.csv").open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.writer(f);w.writerow(["strategy","race_id","date","shape","included","formation","points","combined_odds","probability","expected_return","hit","equal_return","uniform_return"])
        for key,bb in bets.items():
            for r,m,p,ev,t in bb:w.writerow([key,r["rid"],r["date"],r["shape"],int(t),m[6],m[0],m[1],p,ev,m[4],m[2],m[3]])
            v=[a for r,a,p,ev,t in bb if t];n=len(v)
            ret=sum(a[2] for a in v);u=sum(a[3] for a in v)
            out[key]={"eligible":len(bb),"included":n,"hits":sum(a[4] for a in v),"hit_rate":sum(a[4] for a in v)/n if n else None,"equal_return_pct":100*ret/n if n else None,"uniform_return_pct":100*u/n if n else None,"net_units":ret-n,"mean_points":float(np.mean([a[0] for a in v])) if n else None,"median_combined":float(np.median([a[1] for a in v])) if n else None,"families":dict(Counter(a[5] for a in v))}
    print("STACK_EVAL",pool,split,json.dumps(out,ensure_ascii=False,allow_nan=False),flush=True)
    return out
def run(raw,pool,rf):
    X,Y,B,rnames=rf;rows=[prior.candidates(r,pool) for r in raw];dates=np.array([r["date"] for r in raw])
    def idx(a,b):return np.flatnonzero((dates>=a)&(dates<=b))
    train=[]
    for a,b,c in [("2025-07-01","2025-09-30","2025-06-30"),("2025-10-01","2025-12-31","2025-09-30"),("2026-01-01","2026-03-31","2025-12-31")]:
        ti=idx("2025-04-01",c);vi=idx(a,b)
        print("STACK_FOLD",pool,a,len(ti),len(vi),flush=True)
        v=base_predict([rows[i] for i in ti],[rows[i] for i in vi],pool+"_"+a)
        rp=reliability_predict(X,Y,B,rnames,ti,vi,pool+"_"+a)
        train.extend(collect(raw,rows,vi,v,rp,pool));del v,rp;gc.collect()
    ti=idx("2025-04-01","2026-03-31");vi=idx("2026-04-01","2026-04-30")
    # Previously inspected May-June is diagnostic only; post-June is holdout.
    di=idx("2026-05-01","2026-06-30");hi=idx("2026-07-01","2026-10-31")
    pi=np.r_[vi,di,hi]
    v=base_predict([rows[i] for i in ti],[rows[i] for i in pi],pool+"_frozen")
    rp=reliability_predict(X,Y,B,rnames,ti,pi,pool+"_frozen")
    def block(ix,start):
        end=start+len(ix)
        return collect(raw,rows,ix,{k:p[start:end] for k,p in v.items()},rp[start:end],pool)
    val=block(vi,0);diag=block(di,len(vi));hold=block(hi,len(vi)+len(di))
    names=form.NAMES+EXTRA+["race_"+k for k in Y];n0=len(form.NAMES);n1=n0+len(EXTRA)
    nohist=list(range(form.HSTART));market_extra=[n0,n0+2]
    cols={"formation_only":nohist,"market_line_stack":nohist+market_extra,"history_stack":list(range(n1)),"all_stack":list(range(len(names)))}
    V,z,vw=form.concat(val);scores={"line_base":float(np.average(rel.binaryloss(z,V[:,1]),weights=vw))};models={};details={}
    for k,ix in cols.items():
        m,s,loss=fit_meta(train,val,names,ix);models[k]=(m,s,ix);scores[k]=loss
        m.save_model(str(OUT/f"{pool}_{k}.txt"));details[k]={"shrink":s,"features":[names[i] for i in ix]}
    chosen=min(scores,key=scores.get)
    frozen={"chosen":chosen,"validation_loss":scores,"models":details,"selection_rule":"argmax estimated probability times synthetic combined odds; diagnostic inclusion if estimated return > 1; no holdout tuning"}
    (OUT/f"{pool}_frozen.json").write_text(json.dumps(frozen,ensure_ascii=False,indent=2))
    print("STACK_FROZEN",pool,json.dumps(frozen,ensure_ascii=False),flush=True)
    return {"frozen":frozen,"counts":{"formation_train":len(train),"validation":len(val),"diagnostic":len(diag),"holdout":len(hold)},"validation":evaluate(val,models,chosen,pool,"validation"),"diagnostic":evaluate(diag,models,chosen,pool,"diagnostic"),"holdout":evaluate(hold,models,chosen,pool,"holdout") if hold else {}}
def main():
    raw,cov,lc=line.records();future_rows,fc=future()
    assert not set(r["rid"] for r in raw)&set(r["rid"] for r in future_rows)
    raw+=future_rows
    rf=race_features(raw)
    out={"coverage":cov,"line_coverage":lc,"holdout_coverage":fc,"results":{},"limits":["Partial post-June archive only, nonrandom S F1 subset. Previously studied May-June is diagnostic, not fresh holdout.","Refitted feature-family approximations rather than exact archived individual model weights; structure-and-odds features combined in line base.","Final odds, hypothetical fractional allocation and uniform stake comparisons; no live executions. No stakes optimized.","No threshold or architecture changes using holdout. Candidate search bounded by existing formation generator. The previously observed numerical clipping error is corrected in this run.","Historical archives and prior-day rate matching can exclude races; no imputation of missing line or odds."]}
    for p in ("trio","trifecta"):
        out["results"][p]=run(raw,p,rf);gc.collect()
        (OUT/"report.json").write_text(json.dumps(out,ensure_ascii=False,indent=2,allow_nan=False))
    print("COMPLETE",flush=True)
if __name__=="__main__":main()
