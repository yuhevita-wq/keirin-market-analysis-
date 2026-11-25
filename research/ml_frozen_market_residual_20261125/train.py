"""Two-stage frozen-market residual learning. Research only; no deployment."""
import sys, json, gc
from pathlib import Path
import numpy as np
from sklearn.tree import DecisionTreeRegressor
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"ml_integrated_20261125"))
import importlib.util
spec=importlib.util.spec_from_file_location("integrated_previous",ROOT/"ml_integrated_20261125"/"train.py")
integ=importlib.util.module_from_spec(spec);spec.loader.exec_module(integ)
odds=integ.odds
OUT=Path("ml-frozen-residual-results");OUT.mkdir(exist_ok=True)
NAMES=odds.SET_NAMES+integ.H_NAMES+integ.L_NAMES+integ.I_NAMES
O=len(odds.SET_NAMES);H=len(integ.H_NAMES);L=len(integ.L_NAMES)
def market_predict(p,m,mode,k):
    a=p["a"];t=p["t"];s=p["s"]
    if mode=="blend":return odds.softmax((1-k)*np.log(a)+k*np.log(t),s)
    if mode=="temperature":return odds.softmax(k*np.log(a),s)
    if mode=="ml":return odds.softmax(np.log(a)+k*m.predict(p["x"][:,:O],raw_score=True),s)
    return a.copy()
def fit_market():
    tr=odds.pack([2021,2022,2023]);va=odds.pack([2024])
    print("FIT frozen market",len(tr["ids"]),len(va["ids"]),flush=True)
    m=odds.fit(tr["x"],tr["a"],tr["s"],tr["yi"],odds.SET_NAMES)
    a=va["a"];s=va["s"];y=va["yi"]
    params={"raw":0.0,"blend":odds.pick_alpha(a,va["t"],s,y),"temperature":odds.tune_temp(a,s,y),"ml":odds.tune_residual(a,m.predict(va["x"],raw_score=True),s,y)}
    scores={k:odds.report(market_predict(va,m,k,v),y,s) for k,v in params.items()}
    mode=min(scores,key=lambda k:scores[k]["logloss"])
    m.save_model(str(OUT/"historical_market.txt"))
    info={"train_years":[2021,2022,2023],"validation_year":2024,"train_races":len(tr["ids"]),"validation_races":len(va["ids"]),"chosen":mode,"params":params,"validation_scores":scores}
    del tr,va;gc.collect()
    return m,mode,params[mode],info
def feature_block(p,kind):
    x=p["x"]
    if kind=="market_control":return x[:,:O],odds.SET_NAMES
    # No raw market price, rank or probability in independent correction.
    hist=list(range(O,O+H-1));line=list(range(O+H,O+H+L));cross=list(range(O+H+L,len(NAMES)))
    if kind=="history":cols=hist
    elif kind=="line":cols=line
    else:cols=hist+line+cross
    X=x[:,cols];names=[NAMES[i] for i in cols]
    if kind=="disagreement":
        # Market discrepancy relative to the frozen market, not raw price alone.
        # Candidate's three rider marginals derived from the baseline distribution.
        marg=[];gaps=[]
        for i,(start,S,n) in enumerate(zip(odds.groups(p["s"]),p["sets"],p["n"])):
            q=p["frozen"][start:start+len(S)]
            r=np.bincount(S.ravel(),weights=np.repeat(q,3),minlength=n)
            marg.append(r[S].mean(1))
        marginal=np.concatenate(marg)
        gap=x[:,NAMES.index("independent_mean")]-marginal
        sc=x[:,NAMES.index("same_line_pairs")]
        strength=x[:,NAMES.index("line_partner_rate_mean")]
        pos=x[:,NAMES.index("position_mean")]
        X=np.column_stack([X,gap,gap*sc,gap*strength,gap*pos])
        names+=["independent_minus_frozen_marginal","gap_x_same_line","gap_x_partner","gap_x_position"]
    return X.astype(np.float32,copy=False),names
def evaluate(p,preds):
    return {k:odds.report(v,p["yi"],p["s"]) for k,v in preds.items()}
def main():
    market,mode,k,market_info=fit_market()
    pp,counts,lc=integ.pack()
    tr,va,te=[pp[k] for k in ("train","validation","test")]
    assert not set(tr["ids"])&set(va["ids"]) and not set(tr["ids"])&set(te["ids"]) and not set(va["ids"])&set(te["ids"])
    assert len(tr["ids"])>=500 and len(va["ids"])>=200 and len(te["ids"])>=500
    for p in pp.values():p["frozen"]=market_predict(p,market,mode,k)
    vp={"frozen":va["frozen"]};tp={"frozen":te["frozen"]};info={}
    for kind in ("market_control","history","line","joint","disagreement"):
        print("FIT residual",kind,flush=True)
        X,names=feature_block(tr,kind);V,_=feature_block(va,kind);T,_=feature_block(te,kind)
        model=odds.fit(X,tr["frozen"],tr["s"],tr["yi"],names)
        v=model.predict(V,raw_score=True);t=model.predict(T,raw_score=True)
        shrink=odds.tune_residual(va["frozen"],v,va["s"],va["yi"])
        vp[kind]=odds.softmax(np.log(va["frozen"])+shrink*v,va["s"])
        tp[kind]=odds.softmax(np.log(te["frozen"])+shrink*t,te["s"])
        info[kind]={"shrink":shrink,"importance":dict(zip(names,map(float,model.feature_importance(importance_type="gain"))))}
        model.save_model(str(OUT/(kind+".txt")))
        del X,V,T,model;gc.collect()
    # Choose using April validation only. May-June never determines the choice.
    vs=evaluate(va,vp);best=min(("history","line","joint","disagreement"),key=lambda c:vs[c]["logloss"])
    gainv=odds.loss(vp["market_control"],va["yi"])-odds.loss(vp[best],va["yi"])
    gaint=odds.loss(tp["market_control"],te["yi"])-odds.loss(tp[best],te["yi"])
    # Conditions based only on independent race-level structure and disagreement,
    # not odds/entropy thresholds alone. Descriptive, not causal.
    ctxix=[0,1,2,6,7,8,9,10];names=[integ.R_NAMES[i] for i in ctxix]
    tree=DecisionTreeRegressor(max_depth=2,min_samples_leaf=max(100,len(gainv)//8),random_state=42).fit(va["context"][:,ctxix],gainv)
    lv,rules=integ.conditions(tree,va["context"][:,ctxix],names);lt,_=integ.conditions(tree,te["context"][:,ctxix],names)
    rules_out=[]
    for leaf,rule in rules.items():
        a=lv==leaf;b=lt==leaf
        if b.any():rules_out.append({"condition":rule,"validation_n":int(a.sum()),"validation_gain_vs_control":float(gainv[a].mean()),"test_n":int(b.sum()),"test_gain_vs_control":odds.ci(gaint[b],[rid for rid,ok in zip(te["ids"],b) if ok])})
    pred={"raw_trio":te["t"],"raw_trifecta_aggregate":te["a"],**tp}
    baseline_loss=odds.loss(te["frozen"],te["yi"]);control_loss=odds.loss(tp["market_control"],te["yi"])
    out={"frozen_market":market_info,"join_counts":counts,"line_coverage":lc,"split_counts":{k:len(p["ids"]) for k,p in pp.items()},"chosen_independent_model":best,"validation":vs,"test":evaluate(te,pred),"paired_vs_frozen":{k:odds.ci(baseline_loss-odds.loss(v,te["yi"]),te["ids"]) for k,v in tp.items() if k!="frozen"},"paired_vs_market_control":{k:odds.ci(control_loss-odds.loss(v,te["yi"]),te["ids"]) for k,v in tp.items() if k not in ("frozen","market_control")},"conditions":rules_out,"models":info,"months":{},"limitations":["Independent rate is prior-day same-class research proxy, not verified live formula.","Historical market model fixed after 2024; same-period market-only correction is separate control.","Independent residual training 2025-04 to 2026-03; April validation; May-June test previously researched, not pristine.","Only matched complete line and final-odds records; missing markets not imputed.","Three-rider unordered event; not evidence of live execution, financial return, or causal relation.","Condition subgroup confidence intervals exploratory and unadjusted."]}
    for month in sorted({d[:7] for d in te["dates"]}):
        mask=np.array([d.startswith(month) for d in te["dates"]])
        out["months"][month]={"races":int(mask.sum()),"logloss":{k:float(odds.loss(v,te["yi"])[mask].mean()) for k,v in pred.items()}}
    (OUT/"report.json").write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding="utf-8")
    print("RESULT",json.dumps(out,ensure_ascii=False),flush=True)
if __name__=="__main__":main()
