"""Refinement study: chronological output ablation, coherent probabilities and coverage/price tradeoffs.
Historical diagnostic research only. No live deployment or new holdout tuning.
"""
import os
os.environ["OPENBLAS_NUM_THREADS"]="4"
os.environ["OMP_NUM_THREADS"]="4"
import gc,inspect,importlib.util,json,csv,math
from pathlib import Path
from collections import defaultdict,Counter
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit,logit
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("stack",ROOT/"ml_stack_holdout_20261125"/"train.py")
stack=importlib.util.module_from_spec(spec);spec.loader.exec_module(stack)
form=stack.form;prior=stack.prior;odds=stack.odds;rel=stack.rel
OUT=Path("ml-refinement-results");OUT.mkdir(exist_ok=True)
stack.OUT=OUT;form.clip=stack.clip
# Keep all generated valid rectangles and unions instead of pruning to a few
# candidates per family/point bucket. The original family generator/cap remains.
src=inspect.getsource(form.forms)
start=src.index("    chosen={}");end=src.index("    if not chosen:",start)
src=src[:start]+"    chosen={item[0]:item for items in strata.values() for item in items}\n"+src[end:]
src=src.replace("def forms(","def expanded_forms(",1)
exec(compile(src,"expanded_forms","exec"),form.__dict__)
CHANNELS=["cross_pool","market_correction","line_correction","history_correction"]
def feat(row,p,rp):
    q=np.asarray(row["q"],float);other=row["other"];a=p["market"];b=p["line"];c=p["history"]
    z=np.column_stack([np.log(other/q),np.log(a/other),np.log(b/a),np.log(c/b)])
    # Race reliability is a race-constant signal: interactions are necessary
    # to alter a normalized distribution over tickets.
    return np.column_stack([z,*[z*float(v) for v in rp]]).astype(np.float32)
def data_for(rows,ii,p,rp):
    return [dict(q=rows[i]["q"],x=feat(rows[i],{k:v[j] for k,v in p.items()},rp[j]),hit=rows[i]["hit"],rid=rows[i]["rid"],date=rows[i]["date"]) for j,i in enumerate(ii)]
def pack(rr,ix):
    s=np.array([len(r["q"]) for r in rr],dtype=int);starts=odds.groups(s)
    return (np.concatenate([r["x"][:,ix] for r in rr]),np.concatenate([r["q"] for r in rr]),s,starts+np.array([r["hit"] for r in rr]))
def fit_linear(train,val,ix):
    X,q,s,y=pack(train,ix);V,vq,vs,vy=pack(val,ix)
    n=len(s);start=np.log(q)
    def objective(w):
        p=odds.softmax(start+X@w.astype(np.float32),s)
        loss=float(odds.loss(p,y).mean()+0.0001*np.dot(w,w))
        p[y]-=1.
        grad=(X.T@p.astype(np.float32)).astype(float)/n+0.0002*w
        return loss,grad
    opt=minimize(objective,np.zeros(len(ix)),method="L-BFGS-B",jac=True,options={"maxiter":100,"ftol":1e-9,"gtol":1e-6})
    assert np.isfinite(opt.x).all()
    v=odds.softmax(np.log(vq)+V@opt.x.astype(np.float32),vs)
    # Separate validation calibration includes exact zero and one.
    scale=stack.optimize(lambda a:float(odds.loss(odds.softmax(np.log(vq)+a*(V@opt.x.astype(np.float32)),vs),vy).mean()),(0.,1.))
    w=opt.x*scale
    v=odds.softmax(np.log(vq)+V@w.astype(np.float32),vs)
    loss=float(odds.loss(v,vy).mean())
    return w,loss,dict(converged=bool(opt.success),message=str(opt.message),scale=scale,iterations=int(opt.nit))
def predict(row,model):
    w,ix=model
    p=odds.softmax(np.log(row["q"])+row["x"][:,ix]@w.astype(np.float32),np.array([len(row["q"])]))
    assert np.isclose(p.sum(),1.) and np.all(p>=0)
    return p
def logloss(p,h):return float(-np.log(max(float(p[h]),1e-12)))
def formation_indices(rec,pool,meta):
    return [stack.member_ids(rec,pool,m[6]) for m in meta]
def summaries(items):
    n=len(items);v=np.array([m[2] for _,m,_,_ in items],float);u=np.array([m[3] for _,m,_,_ in items],float)
    if not n:return {"races":0,"hits":0,"hit_rate":None,"equal_return_pct":None,"uniform_return_pct":None}
    mean=v.mean();daily=defaultdict(list)
    for (r,_,_,_),t in zip(items,v):daily[r["date"]].append(float(t))
    g=len(daily);se=math.sqrt(g/max(1,g-1)*sum((sum(a)-mean*len(a))**2 for a in daily.values()))/n
    return dict(races=n,hits=int(sum(m[4] for _,m,_,_ in items)),hit_rate=float(np.mean([m[4] for _,m,_,_ in items])),mean_points=float(np.mean([m[0] for _,m,_,_ in items])),median_combined=float(np.median([m[1] for _,m,_,_ in items])),mean_predicted=float(np.mean([p for _,_,p,_ in items])),equal_return_pct=float(100*mean),uniform_return_pct=float(100*u.mean()),ci_pct=[float(100*(mean-1.96*se)),float(100*(mean+1.96*se))],net_units=float(v.sum()-n))
def edge_update(acc,meta,ix,p,direct):
    # For every small candidate, choose its nearest strict superset by added
    # ticket count; ties are deterministic label order, independent of outcome.
    bits=[sum(1<<int(i) for i in a) for a in ix]
    order=sorted(range(len(ix)),key=lambda j:(meta[j][0],meta[j][6]))
    for i in order:
        j=next((j for j in order if meta[j][0]>meta[i][0] and bits[i]&bits[j]==bits[i]),None)
        if j is None:continue
        k=meta[j][0]-meta[i][0];y=meta[j][4]-meta[i][4];d=float(p[j]-p[i]);assert d>=-1e-8 and y in (0,1)
        a=acc.setdefault(str(k) if k<=6 else "7+",[0,0,0.,0.,0.,0,0])
        a[0]+=1;a[1]+=y;a[2]+=d;a[3]+=(y-d)**2
        if direct is not None:
            dd=float(direct[j]-direct[i]);a[4]+=(y-dd)**2;a[5]+=int(dd<0);a[6]+=1
def edge_summary(acc):
    return {k:dict(pairs=a[0],observed_increment=a[1]/a[0],predicted_increment=a[2]/a[0],coherent_brier=a[3]/a[0],direct_brier=a[4]/a[6] if a[6] else None,direct_negative_increment=a[5],direct_pairs=a[6]) for k,a in acc.items()}
def evaluate(raw,rows,ii,base,rp,dd,models,chosen,heads,pool,split):
    metrics=defaultdict(list);loss=defaultdict(list);formloss=defaultdict(list);edge={};monthly=defaultdict(list)
    audit=OUT/f"{pool}_{split}_frontier.csv"
    with audit.open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.writer(f);w.writerow(["model","universe","objective","race_id","date","points","formation","combined_odds","probability","hit","equal_return","uniform_return"])
        for j,i in enumerate(ii):
            rec=raw[i];r=rows[i];d=dd[j];b=base["line"][j]
            probs={"raw_market":r["q"],"line_base":b}
            probs.update({k:predict(d,m) for k,m in models.items()})
            probs["selected_coherent"]=probs[chosen]
            for name,p in probs.items():loss[name].append(logloss(p,r["hit"]))
            X,y,m,ng=form.forms(rec,r,b,pool)
            members=formation_indices(rec,pool,m)
            extra=[]
            for a,z in zip(m,members):
                pm=float(base["market"][j][z].sum());ph=float(base["history"][j][z].sum())
                extra.append([pm,ph,pm*a[1],ph*a[1],*rp[j]])
            rr=dict(x=np.column_stack([X,np.array(extra)]).astype(np.float32),y=y,m=m,rid=r["rid"],date=r["date"])
            ps={k:np.array([p[z].sum() for z in members]) for k,p in probs.items()}
            for k,(head,s,ix) in heads.items():ps[k]=stack.pred(head,s,ix,rr)
            for k,p in ps.items():formloss[k].append(float(rel.binaryloss(y,p).mean()))
            edge_update(edge,m,members,ps["selected_coherent"],ps["direct_full"])
            def compare(universe,meta,ps):
                co=np.array([a[1] for a in meta]);k=np.array([a[0] for a in meta])
                for name,p in ps.items():
                    ev=p*co
                    # Compare objectives instead of treating the EV maximum as
                    # the uniquely correct formation. No diagnostic tuning.
                    best=int(np.argmax(ev))
                    choices={"all_ev_max":best}
                    eligible=np.flatnonzero(ev>1.)
                    if len(eligible):
                        choices["ev_max_above_break_even"]=best
                        choices["coverage_max_above_break_even"]=int(eligible[np.argmax(p[eligible])])
                    for budget in (2,4,6,8,12,18,24,36,48):
                        allowed=np.flatnonzero(k<=budget)
                        if len(allowed):
                            choices[f"coverage_budget_{budget}"]=int(allowed[np.argmax(p[allowed])])
                    for obj,z in choices.items():
                        # Every comparison has one nominal race unit.
                        a=meta[z];key=(name,universe,obj)
                        metrics[key].append((rec,a,float(p[z]),float(ev[z])))
                        w.writerow([name,universe,obj,rec["rid"],rec["date"],a[0],a[6],a[1],float(p[z]),a[4],a[2],a[3]])
            compare("original_retained",m,ps)
            # Expanded generator has the same family grammar but no final
            # per-family/per-budget pruning. Use coherent predictions only.
            _,__,mm,ng=form.expanded_forms(rec,r,probs["selected_coherent"],pool)
            zz=formation_indices(rec,pool,mm)
            pp={k:np.array([v[z].sum() for z in zz]) for k,v in probs.items() if k in ("full","selected_coherent","line_base")}
            compare("expanded",mm,pp)
    result={ "|".join(k):summaries(v) for k,v in metrics.items()}
    for k,v in result.items():v["eligible_races"]=len(ii)
    ll={k:float(np.mean(v)) for k,v in loss.items()};fl={k:float(np.mean(v)) for k,v in formloss.items()}
    paired={k:odds.ci(np.array(loss["full"])-np.array(v),[raw[i]["rid"] for i in ii]) for k,v in loss.items() if k!="full"}
    monthly={}
    for month in sorted(set(raw[i]["date"][:7] for i in ii)):
        mask=np.array([raw[i]["date"].startswith(month) for i in ii])
        monthly[month]={k:float(np.mean(np.array(v)[mask])) for k,v in loss.items()}
    report={"exact_logloss":ll,"formation_logloss":fl,"paired_improvement_vs_full":paired,"monthly_exact_logloss":monthly,"nested_increment":edge_summary(edge),"strategies":result}
    print("REFINE_METRICS",pool,split,json.dumps({k:v for k,v in report.items() if k!="strategies"},ensure_ascii=False,allow_nan=False),flush=True)
    for k,v in result.items():
        if k.endswith("above_break_even") or k.endswith("all_ev_max"):print("REFINE_POLICY",pool,split,k,json.dumps(v,allow_nan=False),flush=True)
    return report
def run(raw,pool,rf):
    X,Y,B,rnames=rf;rows=[prior.candidates(r,pool) for r in raw];dates=np.array([r["date"] for r in raw])
    def idx(a,b):return np.flatnonzero((dates>=a)&(dates<=b))
    train=[];direct_train=[]
    for a,b,c in [("2025-07-01","2025-09-30","2025-06-30"),("2025-10-01","2025-12-31","2025-09-30"),("2026-01-01","2026-03-31","2025-12-31")]:
        ti=idx("2025-04-01",c);vi=idx(a,b)
        print("REFINE_FOLD",pool,a,len(ti),len(vi),flush=True)
        p=stack.base_predict([rows[i] for i in ti],[rows[i] for i in vi],pool+"_"+a)
        rp=stack.reliability_predict(X,Y,B,rnames,ti,vi,pool+"_"+a)
        train.extend(data_for(rows,vi,p,rp))
        direct_train.extend(stack.collect(raw,rows,vi,p,rp,pool))
        # Training meta is not required for direct-head fitting.
        for r in direct_train:r["m"]=None
        del p,rp;gc.collect()
    ti=idx("2025-04-01","2026-03-31");vi=idx("2026-04-01","2026-04-30");di=idx("2026-05-01","2026-06-30");pi=np.r_[vi,di]
    base=stack.base_predict([rows[i] for i in ti],[rows[i] for i in pi],pool+"_frozen")
    rp=stack.reliability_predict(X,Y,B,rnames,ti,pi,pool+"_frozen")
    bv={k:p[:len(vi)] for k,p in base.items()};bd={k:p[len(vi):] for k,p in base.items()}
    dv=data_for(rows,vi,bv,rp[:len(vi)]);dd=data_for(rows,di,bd,rp[len(vi):])
    names=CHANNELS+[f"{ch}__race_{k}" for k in Y for ch in CHANNELS]
    n=len(names)
    # Output-level ablations: dependent later specialists may retain earlier
    # information. In particular remove line + its history increment together.
    groups={"full":list(range(n)),"no_cross_output":[j for j in range(n) if j%4!=0],"no_market_output":[j for j in range(n) if j%4!=1],"no_line_history_outputs":[j for j in range(n) if j%4 in (0,1)],"no_history_output":[j for j in range(n) if j%4!=3],"no_reliability":list(range(4))}
    models={};val={};details={}
    for k,ix in groups.items():
        print("REFINE_STACK_FIT",pool,k,flush=True)
        w,loss,info=fit_linear(train,dv,ix);models[k]=(w,ix);val[k]=loss;details[k]={"weights":dict(zip([names[i] for i in ix],w.tolist())),**info}
    chosen=min(val,key=val.get)
    # Independent formation evaluation head from the previous architecture.
    # Same OOF base data and April calibration; no May-June fitting.
    direct_val=stack.collect(raw,rows,vi,bv,rp[:len(vi)],pool)
    fn=form.NAMES+stack.EXTRA+["race_"+k for k in Y]
    n0=len(form.NAMES);n1=n0+len(stack.EXTRA);ixfull=list(range(len(fn)))
    head_groups={"direct_full":ixfull,"direct_no_reliability":list(range(n1)),"direct_no_history":[i for i in ixfull if i not in list(range(form.HSTART,n0))+[n0+1,n0+3]]}
    heads={};hinfo={}
    for k,ix in head_groups.items():
        m,s,l=stack.fit_meta(direct_train,direct_val,fn,ix);heads[k]=(m,s,ix);hinfo[k]={"shrink":s,"validation_loss":l}
        m.save_model(str(OUT/f"{pool}_{k}.txt"))
    fixed={"coherent_choice_by_April_exact_logloss":chosen,"validation_exact_loss":val,"coherent_models":details,"direct_models":hinfo}
    (OUT/f"{pool}_frozen.json").write_text(json.dumps(fixed,ensure_ascii=False,indent=2,allow_nan=False))
    print("REFINE_FROZEN",pool,json.dumps(fixed,ensure_ascii=False,allow_nan=False),flush=True)
    del train,direct_train,direct_val;gc.collect()
    return {"frozen":fixed,"validation":evaluate(raw,rows,vi,bv,rp[:len(vi)],dv,models,chosen,heads,pool,"validation"),"diagnostic":evaluate(raw,rows,di,bd,rp[len(vi):],dd,models,chosen,heads,pool,"diagnostic")}
def main():
    raw,cov,lc=stack.line.records();rf=stack.race_features(raw)
    out={"coverage":cov,"line_coverage":lc,"results":{},"limitations":["Existing model preserved on prior branch. May-June repeatedly inspected, diagnostic only. No fresh holdout until complete matching data are available.","Specialist output ablation, not raw-feature end-to-end ablation; cross/market information is shared in specialist models. Removing a correction does not remove all underlying information.","Coherent ticket distribution learns weights via chronological OOF exact-outcome logloss; candidate hit probabilities are sums, hence monotonic for nested candidates.","Formation head compares prior method. Expanded candidates remove final pruning, not all generator limits (2-48 tickets, existing line/axis/foldback/union grammar).","Coverage/return tradeoff grid is fixed in advance, no post-hoc optimal point count or objective chosen on May-June. Break-even constraint uses historical theoretical prices; no live execution.","Nested pairs overlap and are descriptive, not independent observations. Multiple comparisons exploratory. Final odds and theoretical fractional stake allocation only."]}
    for p in ("trio","trifecta"):
        out["results"][p]=run(raw,p,rf);gc.collect()
        (OUT/"report.json").write_text(json.dumps(out,ensure_ascii=False,indent=2,allow_nan=False))
    print("COMPLETE",flush=True)
if __name__=="__main__":main()
