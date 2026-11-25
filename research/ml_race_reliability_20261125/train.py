"""Independent race-level market reliability. No tickets, past model scores or betting rules."""
import importlib.util,json,csv
from pathlib import Path
import numpy as np
import lightgbm as lgb
from scipy.special import expit,logit
from scipy.optimize import minimize_scalar
from sklearn.metrics import roc_auc_score
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("line_data",ROOT/"ml_line_arrival_structure_20261125"/"train.py")
loader=importlib.util.module_from_spec(spec);spec.loader.exec_module(loader)
odds=loader.odds
OUT=Path("ml-race-reliability-results");OUT.mkdir(exist_ok=True)
def clip(p):return np.clip(np.asarray(p,dtype=float),1e-9,1-1e-9)
def book(q,o):
    s=np.sort(q)[::-1];c=np.cumsum(s);n=len(q)
    out={"entropy":float(odds.entropy(q)),"entropy_norm":float(odds.entropy(q)/np.log(n)),"top1":s[0],"top2":s[:2].sum(),"top3":s[:3].sum(),"top5":s[:5].sum(),"top10":s[:10].sum(),"top_gap":s[0]-s[1],"top_ratio":s[0]/s[1],"hhi":np.dot(s,s),"inverse_sum":(1/o).sum(),"log_odds_sd":np.log(o).std()}
    for t in (.5,.8,.9):
        k=min(n,int(np.searchsorted(c,t)+1))
        out[f"count_mass_{t}"]=k/n
        out[f"mass_{t}"]=c[k-1]
    return out
def js(a,b):
    return float(odds.entropy((a+b)/2)-(odds.entropy(a)+odds.entropy(b))/2)
def race(r):
    S,O=odds.layout(r["cars"]);n=len(r["cars"])
    t=odds.norm(1/r["oa"]);u=odds.norm(1/r["ob"]);a=t.reshape(-1,6).sum(1)
    cond=t.reshape(-1,6)/a[:,None]
    f={"field_size":n}
    for prefix,q,o in [("trio",u,r["ob"]),("trifecta",t,r["oa"])]:
        f.update({prefix+"_"+k:float(v) for k,v in book(q,o).items()})
    f.update(cross_js=js(u,a),cross_l1=float(np.abs(u-a).sum()),cross_max_gap=float(np.max(np.abs(u-a))),cross_top_agree=float(np.argmax(u)==np.argmax(a)),conditional_entropy=float(np.dot(a,odds.entropy(cond,1))),conditional_max_mean=float(np.dot(a,cond.max(1))))
    # Raw line information and its relationship with raw market probabilities.
    # Neither prior model scores nor independent rider history are used.
    lid=r["lid"];size=np.bincount(lid);line_probs=[]
    f["market_feature_count"]=0 # marker removed before features are returned
    for q,C,prefix in [(u,S,"trio"),(t,O,"trifecta")]:
        ls=lid[C]
        same=(ls[:,0]==ls[:,1]).astype(int)+(ls[:,0]==ls[:,2]).astype(int)+(ls[:,1]==ls[:,2]).astype(int)
        marginal=np.bincount(ls.ravel(),weights=np.repeat(q,3),minlength=len(size))/3
        line_probs.append(marginal)
    market_count=len(f)-1;del f["market_feature_count"]
    f.update(line_count=len(size),largest_line=int(size.max()),singles=int((size==1).sum()),two_lines=int((size==2).sum()),three_lines=int((size==3).sum()),four_plus_lines=int((size>=4).sum()),line_size_hhi=float(np.sum((size/n)**2)),line_cross_js=js(*line_probs))
    for q,C,prefix,mp in [(u,S,"trio",line_probs[0]),(t,O,"trifecta",line_probs[1])]:
        ls=lid[C];same=(ls[:,0]==ls[:,1]).astype(int)+(ls[:,0]==ls[:,2]).astype(int)+(ls[:,1]==ls[:,2]).astype(int)
        f.update({prefix+"_line_top":float(mp.max()),prefix+"_line_entropy":float(odds.entropy(mp)),prefix+"_same_three":float(q[same==3].sum()),prefix+"_two_one":float(q[same==1].sum()),prefix+"_all_distinct":float(q[same==0].sum()),prefix+"_line_strength_spread":float(np.ptp(mp))})
    targets={}
    for pool,q,hit in [("trio",u,r["hit"]//6),("trifecta",t,r["hit"])]:
        ix=np.argsort(-q,kind="stable");rank=int(np.flatnonzero(ix==hit)[0]);c=np.cumsum(q[ix])
        for label,k in [("top1_miss",1),("outside_mass50",int(np.searchsorted(c,.5)+1)),("outside_mass80",int(np.searchsorted(c,.8)+1))]:
            targets[pool+"_"+label]=(int(rank>=k),float(1-c[k-1]))
        # Excess realized surprise over what the market itself expects.
        log=-np.log(q);ent=float(np.dot(q,log));sd=float(np.sqrt(np.dot(q,(log-ent)**2)))
        targets[pool+"_surprise"]=(float((log[hit]-ent)/max(sd,1e-6)),0.)
    return f,market_count,targets
def optimize(fun,bounds):
    r=minimize_scalar(fun,bounds=bounds,method="bounded")
    return min([bounds[0],bounds[1],float(r.x)],key=fun)
def binaryloss(y,p):
    p=clip(p);return -y*np.log(p)-(1-y)*np.log1p(-p)
def stats(y,p,binary):
    if binary:
        return {"logloss":float(binaryloss(y,p).mean()),"brier":float(np.mean((y-p)**2)),"auc":float(roc_auc_score(y,p)) if len(np.unique(y))==2 else None}
    return {"mse":float(np.mean((y-p)**2)),"mae":float(np.mean(np.abs(y-p)))}
def run(raw):
    rows=[];Y={};B={};names=None
    for r in raw:
        f,mc,t=race(r)
        if names is None:names=list(f)
        assert list(f)==names
        rows.append(list(f.values()))
        for k,(y,b) in t.items():Y.setdefault(k,[]).append(y);B.setdefault(k,[]).append(b)
    X=np.asarray(rows,dtype=np.float32);assert np.isfinite(X).all()
    masks={k:np.array([r["split"]==k for r in raw]) for k in ("train","validation","test")}
    tr,va,te=[masks[k] for k in ("train","validation","test")]
    cols={"market_only":list(range(mc)),"market_line":list(range(len(names)))}
    report={};predcols={}
    for name in Y:
        y=np.array(Y[name],dtype=float);b=np.array(B[name],dtype=float);binary=not name.endswith("surprise")
        if binary:
            offset=logit(clip(b))
            intercept=optimize(lambda a:float(binaryloss(y[tr],expit(offset[tr]+a)).mean()),(-4.,4.))
            base=expit(offset+intercept);init=offset+intercept
            loss=lambda yy,pp:binaryloss(yy,pp)
        else:
            intercept=float(y[tr].mean());base=np.full(len(y),intercept);init=base
            loss=lambda yy,pp:(yy-pp)**2
        pp={"raw_market":b,"global_calibration":base};info={}
        for tag,ix in cols.items():
            ds=lgb.Dataset(X[tr][:,ix],label=y[tr] if binary else y[tr]-base[tr],init_score=init[tr] if binary else None,feature_name=[names[j] for j in ix])
            model=lgb.train({"objective":"binary" if binary else "regression","metric":"None","learning_rate":.03,"num_leaves":12,"min_data_in_leaf":200,"lambda_l2":10,"num_threads":4,"verbosity":-1,"seed":42},ds,num_boost_round=150)
            rawp=model.predict(X[:,ix],raw_score=True)
            predict=lambda a:expit(init+a*rawp) if binary else base+a*rawp
            scale=optimize(lambda a:float(loss(y[va],predict(a)[va]).mean()),(0.,1.))
            pp[tag]=predict(scale)
            info[tag]={"shrink":scale,"importance":dict(zip([names[j] for j in ix],map(float,model.feature_importance(importance_type="gain"))))}
            model.save_model(str(OUT/f"{name}_{tag}.txt"))
        val={k:stats(y[va],p[va],binary) for k,p in pp.items()};metric="logloss" if binary else "mse"
        chosen=min(("global_calibration","market_only","market_line"),key=lambda k:val[k][metric])
        p=pp[chosen];test={k:stats(y[te],v[te],binary) for k,v in pp.items()}
        ids=[r["rid"] for r,m in zip(raw,te) if m]
        paired={k:odds.ci(loss(y[te],base[te])-loss(y[te],v[te]),ids) for k,v in pp.items() if k!="global_calibration"}
        # Rank conditional excess error rather than raw predicted race difficulty.
        score=p-b;cuts=np.quantile(score[va],[.2,.4,.6,.8]);bins=np.searchsorted(cuts,score,side="right")
        strata=[]
        for j in range(5):
            for month in ("2026-04","2026-05","2026-06","test"):
                scope=te if month=="test" else np.array([r["date"].startswith(month) for r in raw])
                m=scope&(bins==j)
                if not m.any():continue
                strata.append({"bin":j+1,"period":month,"races":int(m.sum()),"observed":float(y[m].mean()),"market_expected":float(b[m].mean()),"predicted":float(p[m].mean()),"observed_minus_market":float((y[m]-b[m]).mean())})
        predcols[name]=[y,b,p,score]
        z={"target":name,"global_intercept":intercept,"chosen_by_validation":chosen,"validation":val,"test":test,"paired_vs_global":paired,"validation_excess_cutpoints":cuts.tolist(),"strata":strata,"models":info}
        report[name]=z
        print("RELIABILITY_RESULT",json.dumps(z,ensure_ascii=False,allow_nan=False),flush=True)
    with (OUT/"race_reliability_predictions.csv").open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.writer(f);w.writerow(["race_id","date","split","shape"]+[k+"_"+s for k in predcols for s in ("observed","market_expected","predicted","excess")])
        for i,r in enumerate(raw):
            if r["split"]=="train":continue
            w.writerow([r["rid"],r["date"],r["split"],r["shape"]]+[float(a[i]) for cols in predcols.values() for a in cols])
    return report, names
def main():
    raw,cov,lc=loader.records()
    assert len(set(r["rid"] for r in raw))==len(raw)
    report,names=run(raw)
    out={"coverage":cov,"line_coverage":lc,"features":names,"targets":report,"limitations":["No prior trained model predictions, no formation/ticket outputs, no rider rate as features. Matching uses prior raw-data loader and excludes incomplete races.","Final odds only. No live prices or proof of economic advantage.","May-June previously examined. No pristine holdout.","Target is race-level market forecast error, not buy/skip labels or realized profit. No purchase rule or stake optimization.","Eight target comparisons and subgroup intervals exploratory; not multiplicity-adjusted."]}
    (OUT/"report.json").write_text(json.dumps(out,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
    print("COMPLETE",flush=True)
if __name__=="__main__":main()
