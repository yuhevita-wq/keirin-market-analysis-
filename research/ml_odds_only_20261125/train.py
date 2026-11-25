"""Odds-only distribution learning; no rider data, no betting simulation."""
from pathlib import Path
import gzip,json,itertools,math
import numpy as np
import lightgbm as lgb
from collections import Counter
from scipy.optimize import minimize_scalar

OUT=Path("ml-odds-results");OUT.mkdir(exist_ok=True)
SET_NAMES=["log_trio_q","log_trifecta_set_q","cross_log_ratio","trio_rank_frac","trifecta_set_rank_frac","order_entropy","order_max_share","order_min_share","rider_trio_min","rider_trio_mean","rider_trio_max","rider_trifecta_min","rider_trifecta_mean","rider_trifecta_max","trio_pair_min","trio_pair_mean","trio_pair_max","n_riders","market_trio_entropy","market_trifecta_entropy","market_top_trio","market_top_order","market_disagreement_js","trio_inverse_sum","trifecta_inverse_sum","race_type_code"]
ORDER_NAMES=["log_cond_q","conditional_rank","log_order_q","log_set_trio_q","log_set_trifecta_q","set_cross_ratio","first_marginal","second_marginal","third_marginal","first_second_pair","first_third_pair","second_third_pair","first_trio_marginal","second_trio_marginal","third_trio_marginal","order_entropy","market_trio_entropy","market_trifecta_entropy","n_riders","race_type_code"]
EPS=1e-15
def entropy(p,axis=None):
    return -(p*np.log(np.maximum(p,EPS))).sum(axis=axis)
def ranks(p):
    return np.argsort(np.argsort(-p,kind="stable"),kind="stable")/(max(len(p)-1,1))
def read(p):
    with gzip.open(p,"rt",encoding="utf-8") as f:
        for s in f: yield json.loads(s)
def norm(x):return x/x.sum()
def classify(t):
    s=str(t)
    for i,keys in enumerate([("初日特選","初特"),("準決",),("決勝",),("予選",),("選抜",),("一般",),("特選",),("特秀",),("優秀",),("特別",)],1):
        if any(k in s for k in keys):return i
    return 0
cache={}
def layout(cars):
    n=len(cars)
    if n not in cache:
        sets=list(itertools.combinations(range(n),3))
        orders=np.array([o for s in sets for o in itertools.permutations(s)],dtype=int)
        S=np.array(sets)
        cache[n]=(S,orders)
    return cache[n]
def features(rec,actual):
    a=rec["odds"]["3連単"];b=rec["odds"]["3連複"]
    cars=sorted({int(v) for key in a for v in key.split("-")})
    n=len(cars)
    if not 5<=n<=9 or len(a)!=n*(n-1)*(n-2) or len(b)!=math.comb(n,3):return
    S,O=layout(cars);C=np.array(cars)
    oa=np.array([a["-".join(map(str,C[o]))] for o in O],dtype=float)
    ob=np.array([b["=".join(map(str,C[s]))] for s in S],dtype=float)
    if not(np.isfinite(oa).all() and np.isfinite(ob).all() and (oa>=1).all() and (ob>=1).all()):return
    q=norm(1/oa);t=norm(1/ob);agg=q.reshape(-1,6).sum(1);cond=q.reshape(-1,6)/agg[:,None]
    matches=np.flatnonzero((C[O]==np.array(actual)).all(axis=1))
    if len(matches)!=1:return
    iy=int(matches[0]);sy=iy//6
    rtri=np.bincount(S.ravel(),weights=np.repeat(t,3),minlength=n)
    ragg=np.bincount(S.ravel(),weights=np.repeat(agg,3),minlength=n)
    pairs=np.zeros((n,n))
    for j,k in [(0,1),(0,2),(1,2)]:np.add.at(pairs,(S[:,j],S[:,k]),t)
    pairv=np.stack([pairs[S[:,j],S[:,k]] for j,k in [(0,1),(0,2),(1,2)]],axis=1)
    ent1=entropy(t);ent2=entropy(q);entc=entropy(cond,1)
    mix=(t+agg)/2
    js=entropy(mix)-(entropy(t)+entropy(agg))/2
    typ=classify(rec.get("race_type"));repeat=lambda v:np.full(len(S),v)
    summarize=lambda x:[x.min(axis=1),x.mean(axis=1),x.max(axis=1)]
    fs=np.column_stack([np.log(t),np.log(agg),np.log(agg/t),ranks(t),ranks(agg),entc,cond.max(1),cond.min(1),*summarize(rtri[S]),*summarize(ragg[S]),*summarize(pairv),repeat(n),repeat(ent1),repeat(ent2),repeat(t.max()),repeat(q.max()),repeat(js),repeat((1/ob).sum()),repeat((1/oa).sum()),repeat(typ)]).astype(np.float32)
    # Conditional order model sees the six orders of the observed set as a
    # conditional likelihood training/evaluation group, not as a predictor.
    # Its joint score equals set likelihood times conditional likelihood.
    o=O[6*sy:6*sy+6];c=cond[sy]
    marg=[np.bincount(O[:,i],weights=q,minlength=n) for i in range(3)]
    pmat=[]
    for j,k in [(0,1),(0,2),(1,2)]:
        m=np.zeros((n,n));np.add.at(m,(O[:,j],O[:,k]),q);pmat.append(m[o[:,j],o[:,k]])
    rep=lambda v:np.full(6,v)
    fo=np.column_stack([np.log(c),ranks(c),np.log(q[6*sy:6*sy+6]),rep(np.log(t[sy])),rep(np.log(agg[sy])),rep(np.log(agg[sy]/t[sy])),*[marg[i][o[:,i]] for i in range(3)],*pmat,*[rtri[o[:,i]] for i in range(3)],rep(entc[sy]),rep(ent1),rep(ent2),rep(n),rep(typ)]).astype(np.float32)
    return fs,fo,t,agg,c,sy,iy%6,n,typ
def pack(years):
    xs=[];xo=[];ts=[];ags=[];cs=[];sy=[];oy=[];sizes=[];ids=[];types=[];ns=[];counts=Counter()
    for year in years:
        truths={}
        for r in read(Path(f"data/result_universe_v2/combined/results_{year}.jsonl.gz")):
            try:
                k=str(r["race_id"]);v=tuple(map(int,r["finish_order"]))
                if len(v)==3 and len(set(v))==3:truths[k]=v
            except (ValueError,TypeError,KeyError):continue
        seen=set()
        for r in read(Path(f"data/odds_only/odds_{year}.jsonl.gz")):
            rid=str(r["race_id"]);counts[str(year)+"_source"]+=1
            if rid in seen:counts["duplicate"]+=1;continue
            seen.add(rid)
            if rid not in truths:counts["unmatched"]+=1;continue
            try:f=features(r,truths[rid])
            except (KeyError,ValueError,TypeError):f=None
            if f is None:counts["invalid"]+=1;continue
            s,o,t,a,c,i,j,n,typ=f
            xs.append(s);xo.append(o);ts.append(t);ags.append(a);cs.append(c)
            sizes.append(len(t));sy.append(i);oy.append(j);ids.append(rid);types.append(typ);ns.append(n)
            counts[str(year)+"_accepted"]+=1
        print("READ",year,dict(counts),flush=True)
    assert ids,"No matched races"
    sizes=np.array(sizes);starts=np.r_[0,np.cumsum(sizes)[:-1]]
    return dict(x=np.concatenate(xs),o=np.concatenate(xo),t=np.concatenate(ts),a=np.concatenate(ags),c=np.concatenate(cs),s=sizes,os=np.full(len(ids),6),yi=starts+np.array(sy),oi=np.arange(len(ids))*6+np.array(oy),ids=ids,n=ns,types=types,counts=dict(counts))
def groups(s):return np.r_[0,np.cumsum(s)[:-1]]
def softmax(z,s):
    ix=groups(s)
    p=np.exp(np.clip(z-np.repeat(np.maximum.reduceat(z,ix),s),-80,0))
    return p/np.repeat(np.add.reduceat(p,ix),s)
def fit(X,q,s,yi,names):
    label=np.zeros(len(X),dtype=np.float32);label[yi]=1
    ds=lgb.Dataset(X,label=label,init_score=np.log(q),feature_name=names,free_raw_data=False)
    def objective(pred,ds):
        p=softmax(pred,s)
        return p-label,np.maximum(2*p*(1-p),1e-5)
    model=lgb.train({"objective":objective,"metric":"None","learning_rate":.05,"num_leaves":15,"min_data_in_leaf":300,"lambda_l2":5,"num_threads":4,"verbosity":-1,"seed":42,"force_col_wise":True},ds,num_boost_round=180)
    return model
def loss(p,yi):return -np.log(np.maximum(p[yi],EPS))
def report(p,yi,s):
    l=loss(p,yi)
    return {"logloss":float(l.mean()),"brier":float(np.mean(1-2*p[yi]+np.add.reduceat(p*p,groups(s))))}
def pick_alpha(q,other,s,yi):
    fun=lambda a:loss(softmax((1-a)*np.log(q)+a*np.log(other),s),yi).mean()
    return float(minimize_scalar(fun,bounds=(0,1),method="bounded").x)
def tune_temp(q,s,yi):
    return float(minimize_scalar(lambda a:loss(softmax(a*np.log(q),s),yi).mean(),bounds=(.5,1.5),method="bounded").x)
def tune_residual(q,pred,s,yi):
    return float(minimize_scalar(lambda a:loss(softmax(np.log(q)+a*pred,s),yi).mean(),bounds=(0,1),method="bounded").x)
def ci(diff,ids):
    # Race ID embeds meeting start date and day number.
    from datetime import datetime,timedelta
    clusters={}
    for rid,x in zip(ids,diff):
        day=(datetime.strptime(rid[2:10],"%Y%m%d")+timedelta(days=int(rid[10:12])-1)).date()
        a=clusters.setdefault(day,[0.,0]);a[0]+=float(x);a[1]+=1
    m=float(np.mean(diff));g=len(clusters)
    se=math.sqrt(g/max(g-1,1)*sum((x-m*n)**2 for x,n in clusters.values()))/len(diff)
    return {"mean_improvement":m,"95pct_ci":[m-1.96*se,m+1.96*se],"days":g}
def main():
    train=pack([2021,2022,2023,2024]);val=pack([2025]);test=pack([2026])
    assert not set(train["ids"])&set(val["ids"]) and not set(train["ids"])&set(test["ids"]) and not set(val["ids"])&set(test["ids"])
    print("FIT set",flush=True)
    sm=fit(train["x"],train["t"],train["s"],train["yi"],SET_NAMES)
    print("FIT conditional order",flush=True)
    om=fit(train["o"],train["c"],train["os"],train["oi"],ORDER_NAMES)
    sv=sm.predict(val["x"],raw_score=True);ov=om.predict(val["o"],raw_score=True)
    alpha=pick_alpha(val["t"],val["a"],val["s"],val["yi"])
    st=tune_temp(val["t"],val["s"],val["yi"]);ot=tune_temp(val["c"],val["os"],val["oi"])
    sr=tune_residual(val["t"],sv,val["s"],val["yi"]);orr=tune_residual(val["c"],ov,val["os"],val["oi"])
    S=test["s"];O=test["os"];Y=test["yi"];I=test["oi"]
    pset={"trio_market":test["t"],"trifecta_aggregate":test["a"],"temperature":softmax(st*np.log(test["t"]),S),"simple_blend":softmax((1-alpha)*np.log(test["t"])+alpha*np.log(test["a"]),S),"ml":softmax(np.log(test["t"])+sr*sm.predict(test["x"],raw_score=True),S)}
    porder={"trifecta_conditional":test["c"],"temperature":softmax(ot*np.log(test["c"]),O),"ml":softmax(np.log(test["c"])+orr*om.predict(test["o"],raw_score=True),O)}
    joint={"trifecta_market":loss(pset["trifecta_aggregate"],Y)+loss(porder["trifecta_conditional"],I),"simple_blend":loss(pset["simple_blend"],Y)+loss(porder["trifecta_conditional"],I),"ml":loss(pset["ml"],Y)+loss(porder["ml"],I)}
    import csv
    out={"counts":{"train":train["counts"],"validation":val["counts"],"test":test["counts"]},"races":{"train":len(train["ids"]),"validation":len(val["ids"]),"test":len(test["ids"])},"tuned":{"blend_alpha":alpha,"set_temperature":st,"order_temperature":ot,"set_residual_shrink":sr,"order_residual_shrink":orr},"trio":{k:report(p,Y,S) for k,p in pset.items()},"conditional_order":{k:report(p,I,O) for k,p in porder.items()},"joint_trifecta_logloss":{k:float(v.mean()) for k,v in joint.items()},"ci":{"trio_ml_vs_market":ci(loss(pset["trio_market"],Y)-loss(pset["ml"],Y),test["ids"]),"trio_ml_vs_blend":ci(loss(pset["simple_blend"],Y)-loss(pset["ml"],Y),test["ids"]),"trifecta_ml_vs_market":ci(joint["trifecta_market"]-joint["ml"],test["ids"]),"trifecta_ml_vs_blend":ci(joint["simple_blend"]-joint["ml"],test["ids"])},"importance":{"set":dict(zip(SET_NAMES,map(float,sm.feature_importance(importance_type="gain")))),"conditional_order":dict(zip(ORDER_NAMES,map(float,om.feature_importance(importance_type="gain"))))},"limitations":["Final odds only, not executable pre-close prices","Conditional order evaluated given realized set for factorized likelihood only, not a known set at prediction time","Previously researched years, not pristine holdout","No return or purchase rule evaluated"]}
    sm.save_model(str(OUT/"trio_model.txt"));om.save_model(str(OUT/"conditional_order_model.txt"))
    (OUT/"report.json").write_text(json.dumps(out,ensure_ascii=False,indent=2))
    print("RESULT",json.dumps(out,ensure_ascii=False),flush=True)
if __name__=="__main__":main()
