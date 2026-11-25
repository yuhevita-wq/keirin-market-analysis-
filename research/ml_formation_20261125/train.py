"""Formation-level ML: chronological cross-fitting; historical final odds only."""
import csv,gc,json,math,importlib.util,itertools
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np
import lightgbm as lgb
from scipy.special import expit,logit
from scipy.optimize import minimize_scalar
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("prior",ROOT/"ml_odds_relationship_20261125"/"train.py")
prior=importlib.util.module_from_spec(spec);spec.loader.exec_module(prior)
odds=prior.odds
OUT=Path("ml-formation-results");OUT.mkdir(exist_ok=True)
CAP=48
BUDGETS=(2,4,6,8,12,18,24,36,48)
NAMES=["log_points","prob","market_prob","other_prob","log_combined","log_equal_stake_mean_odds","uniform_expected_return","equal_return_expected","market_equal_return_expected","prob_entropy","max_ticket_share","min_ticket_share","line_structure_count","dominant_structure_share","same_line_share","distinct_line_share","axis_min","axis_max","axis_mean","fixed_riders","role_front_mean","role_second_mean","market_rank_mean","market_rank_max","field_size","line_count","largest_line","parts","family_code","race_entropy","top_ticket_probability","rate_min","rate_mean","rate_max","partner_mean","rate_spread"]
HSTART=NAMES.index("rate_min")
FAMILIES=["line","line_roles","axis","role_axis","rider_axis","foldback","union"]
def clip(p):return np.clip(p,1e-8,1-1e-8)
def fit_base(train, pred):
    tr=prior.pack(train);pp=prior.pack(pred)
    ix=list(range(len(prior.MARKET)+len(prior.STRUCT)))
    m=odds.fit(tr["x"][:,ix],tr["other"],tr["s"],tr["yi"],[prior.NAMES[j] for j in ix])
    # Fixed full residual strength (not tuned on future fold).
    v=odds.softmax(np.log(pp["other"])+m.predict(pp["x"][:,ix],raw_score=True),pp["s"])
    ans=[v[i:i+n] for i,n in zip(odds.groups(pp["s"]),pp["s"])]
    del tr,pp,m;gc.collect()
    return ans
def forms(rec,r,p,pool):
    ordered=pool=="trifecta";n=len(rec["cars"]);S,O=odds.layout(rec["cars"]);C=O if ordered else S
    # A rectangle is a true Cartesian slot formation, excluding duplicate riders.
    # Trio rectangles are expanded over all six orders then deduplicated.
    q=r["q"];o=r["price"];gi=r["gi"];keys=r["keys"]
    posmask=np.array([[O[:,j]==a for a in range(n)] for j in range(3)])
    found={}
    def rect(axes):
        a,b,c=(np.any(posmask[j,list(ax)],axis=0) for j,ax in enumerate(axes))
        z=np.flatnonzero(a&b&c)
        if not ordered:z=np.unique(z//6)
        return z
    def add(z,parts,fam):
        z=np.unique(z)
        if len(z)<2 or len(z)>CAP:return
        key=tuple(map(int,z))
        if key not in found or len(parts)<len(found[key][0]):
            found[key]=(parts,fam)
    def addrect(axes,fam):
        axes=tuple(tuple(sorted(set(map(int,a)))) for a in axes)
        if any(not a for a in axes):return
        add(rect(axes),(axes,),fam)
    rank=rec["rank"];role=rec["pos"];lineids=rec["lid"]
    line_sets={int(k):tuple(np.flatnonzero(rank==k)) for k in np.unique(rank)}
    # All line-rank structures, and role refinements of the better-covered ones.
    groups=[]
    for g,k in enumerate(keys):
        z=np.flatnonzero(gi==g)
        rr=tuple(map(int,str(k)))
        if len(rr)!=3:continue
        axes=tuple(line_sets[x] for x in rr)
        addrect(axes,"line")
        groups.append((float(p[z].sum()),float(p[z].sum()/np.sum(1/o[z])),g,axes))
    selected={x[2]:x for x in sorted(groups,key=lambda x:-x[0])[:8]+sorted(groups,key=lambda x:-x[1])[:5]}
    for _,__,g,axes in selected.values():
        for j in range(3):
            for a in axes[j]:
                aa=list(axes);aa[j]=(a,);addrect(aa,"line_roles")
        # Retain fixed role pairs rather than assume which finishing slot is fixed.
        for j,k in ((0,1),(0,2),(1,2)):
            for a in axes[j]:
                for b in axes[k]:
                    aa=list(axes);aa[j]=(a,);aa[k]=(b,);addrect(aa,"line_roles")
    # Rider marginals by slot; trio uses membership marginal.
    if ordered:
        marg=np.array([np.bincount(C[:,j],weights=p,minlength=n) for j in range(3)])
    else:
        z=np.bincount(C.ravel(),weights=np.repeat(p,3),minlength=n)
        marg=np.tile(z,(3,1))
    marketm=np.array([np.bincount(C[:,j],weights=q,minlength=n) for j in range(3)]) if ordered else np.tile(np.bincount(C.ravel(),weights=np.repeat(q,3),minlength=n),(3,1))
    for mm in (marg,marketm):
        order=np.argsort(-mm,axis=1,kind="stable")
        # Axis widths vary independently: single and double axes, asymmetric
        # and symmetric rectangles. No fixed ticket count assumption.
        for widths in itertools.product(range(1,min(n,5)+1),repeat=3):
            if math.prod(widths)>CAP*2:continue
            if not ordered and tuple(sorted(widths))!=widths:continue
            axes=tuple(tuple(order[j,:w]) for j,w in enumerate(widths));addrect(axes,"axis")
    # Rider-specific and role-specific shapes from high-probability seeds.
    for i in np.argsort(-p,kind="stable")[:6]:
        a,b,c=map(int,C[i]);seed=(a,b,c)
        for j in range(3):
            rest=[k for k in range(3) if k!=j]
            order=np.argsort(-marg[j],kind="stable")
            for width in (2,3,5,n):
                axes=[(a,),(b,),(c,)]
                axes[j]=tuple(dict.fromkeys([seed[j],*order[:width]]))
                addrect(axes,"rider_axis")
                # Foldback: exchange the two fixed positions, preserve third.
                if ordered:
                    ax2=axes.copy();ax2[rest[0]],ax2[rest[1]]=ax2[rest[1]],ax2[rest[0]]
                    add(np.union1d(rect(axes),rect(ax2)),(tuple(axes),tuple(ax2)),"foldback")
        for j in range(3):
            axes=[(a,),(b,),(c,)]
            axes[j]=tuple(np.flatnonzero(role==role[seed[j]]))
            addrect(axes,"role_axis")
    # Union of two valid rectangular structures; deduplicate exact tickets.
    # Limited beam selected without actual finish or realized return.
    items=list(found.items())
    def mass(z):return float(p[list(z)].sum())
    def value(z):return float(p[list(z)].sum()/np.sum(1/o[list(z)]))
    beam=dict(sorted(items,key=lambda t:-mass(t[0]))[:8]+sorted(items,key=lambda t:-value(t[0]))[:8])
    for (z1,(a,f1)),(z2,(b,f2)) in itertools.combinations(beam.items(),2):
        if len(a)==len(b)==1 and len(set(z1)&set(z2))<min(len(z1),len(z2)):
            add(np.union1d(z1,z2),a+b,"union")
    # Keep broad candidate families and point-count strata. The full generated
    # count is retained; bounded deterministic pruning limits runtime.
    strata=defaultdict(list)
    for z,(parts,fam) in found.items():
        b=next((v for v in BUDGETS if len(z)<=v),CAP)
        ix=np.array(z);pr=float(p[ix].sum());co=float(1/np.sum(1/o[ix]))
        strata[(fam,b)].append((z,parts,fam,pr,pr*co,float(q[ix].sum())))
    chosen={}
    for items in strata.values():
        for item in sorted(items,key=lambda x:-x[3])[:2]+sorted(items,key=lambda x:-x[4])[:2]+sorted(items,key=lambda x:-x[5])[:1]:
            chosen[item[0]]=item
    if not chosen:raise ValueError("no formation")
    X=[];Y=[];meta=[]
    for z,parts,fam,P,_,__ in chosen.values():
        ix=np.array(z);k=len(ix);v=p[ix];w=v/P;prices=o[ix];Q=float(q[ix].sum());other=float(r["other"][ix].sum())
        co=float(1/np.sum(1/prices));masses=np.bincount(gi[ix],weights=w);ls=lineids[C[ix]];rr=rank[C[ix]];ps=role[C[ix]]
        same=(ls[:,0]==ls[:,1])|(ls[:,0]==ls[:,2])|(ls[:,1]==ls[:,2])
        distinct=~same
        widths=[len(a) for part in parts for a in part]
        fixed=len(set.intersection(*[set(C[j]) for j in ix]))
        rates=rec["r"][C[ix]];partners=rec["partner"][C[ix]]
        x=[math.log(k),P,Q,other,math.log(co),math.log(prices.mean()),float(np.dot(v,prices)/k),P*co,Q*co,float(odds.entropy(w)),float(w.max()),float(w.min()),int(np.count_nonzero(masses)),float(masses.max()),float(np.dot(w,same)),float(np.dot(w,distinct)),min(widths),max(widths),float(np.mean(widths)),fixed,float(np.dot(w,(ps==1).sum(1))),float(np.dot(w,(ps==2).sum(1))),float(np.dot(w,rr.mean(1))),float(np.dot(w,rr.max(1))),n,len(set(lineids)),int(max(rec["size"])),len(parts),FAMILIES.index(fam),float(odds.entropy(p)),float(p.max()),float(np.dot(w,rates.min(1))),float(np.dot(w,rates.mean(1))),float(np.dot(w,rates.max(1))),float(np.dot(w,partners.mean(1))),float(np.dot(w,np.ptp(rates,axis=1)))]
        hit=r["hit"] in z
        equal_return=co if hit else 0.
        uniform_return=float(o[r["hit"]]/k) if hit else 0.
        # Formation strings are verified by expanding each part again.
        check=np.unique(np.concatenate([rect(a) for a in parts]))
        assert np.array_equal(check,ix)
        label=" + ".join("-".join("".join(map(str,rec["cars"][list(ax)])) for ax in part) for part in parts)
        meta.append((k,co,equal_return,uniform_return,int(hit),fam,label,P,Q))
        X.append(x);Y.append(int(hit))
    return np.array(X,dtype=np.float32),np.array(Y,dtype=np.int8),meta,len(found)
def collect(raw,rows,idx,preds,pool,keep_meta):
    out=[]
    for i,p in zip(idx,preds):
        X,y,m,n= forms(raw[i],rows[i],p,pool)
        out.append(dict(x=X,y=y,m=m if keep_meta else None,date=raw[i]["date"],rid=raw[i]["rid"],shape=raw[i]["shape"],generated=n))
    return out
def concat(rows):
    return np.concatenate([r["x"] for r in rows]),np.concatenate([r["y"] for r in rows]),np.concatenate([np.full(len(r["y"]),1/len(r["y"])) for r in rows])
def fit_formation(train,val,ix):
    X,y,w=concat(train);V,z,vw=concat(val);base=logit(clip(X[:,1]))
    ds=lgb.Dataset(X[:,ix],label=y,weight=w,init_score=base,feature_name=[NAMES[j] for j in ix])
    model=lgb.train(dict(objective="binary",metric="None",learning_rate=.03,num_leaves=15,min_data_in_leaf=300,lambda_l2=10,num_threads=4,verbosity=-1,seed=42),ds,num_boost_round=120)
    s=model.predict(V[:,ix],raw_score=True)
    def loss(a):
        p=clip(expit(logit(clip(V[:,1]))+a*s))
        return float(np.average(-z*np.log(p)-(1-z)*np.log(1-p),weights=vw))
    shrink=float(minimize_scalar(loss,bounds=(0,1),method="bounded").x)
    return model,shrink,loss(0),loss(shrink)
def predict(model,shrink,ix,r):
    return expit(logit(clip(r["x"][:,1]))+shrink*model.predict(r["x"][:,ix],raw_score=True,num_threads=1))
def evaluate(rows,preds,pool,split):
    # Fixed comparison grid, no threshold optimized on final evaluation.
    # One formation per race. Point budget is a comparison axis, not a prescribed cap.
    metrics=defaultdict(list);examples=[]
    for r,pm in zip(rows,preds):
        X=r["x"];meta=r["m"];k=np.array([m[0] for m in meta]);co=np.array([m[1] for m in meta])
        options={"market_coverage":X[:,2],"base_coverage":X[:,1],"learned_coverage":pm,"base_price":X[:,1]*co,"learned_price":pm*co}
        for b in BUDGETS:
            allowed=np.flatnonzero(k<=b)
            if not len(allowed):continue
            for mode,score in options.items():
                j=allowed[np.argmax(score[allowed])];m=meta[j]
                metrics[(b,mode)].append((r["date"],*m,float(pm[j]),r["rid"],r["shape"]))
                if mode in ("learned_coverage","learned_price") and b in (6,12,24) and len(examples)<150:
                    examples.append([split,pool,r["rid"],r["date"],r["shape"],b,mode,m[5],m[6],m[0],m[1],m[7],float(pm[j]),m[4]])
    with (OUT/f"{pool}_{split}_examples.csv").open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.writer(f);w.writerow(["split","pool","race_id","date","shape","budget","objective","family","formation","points","combined_odds","base_probability","learned_probability","hit"]);w.writerows(examples)
    result=[]
    for (b,mode),v in sorted(metrics.items()):
        n=len(v);hit=sum(m[5] for m in v);ret=np.array([m[3] for m in v]);uniform=np.array([m[4] for m in v]);days=defaultdict(list)
        for m,t in zip(v,ret):days[m[0]].append(t)
        mean=ret.mean();g=len(days);se=math.sqrt(g/max(g-1,1)*sum((sum(a)-mean*len(a))**2 for a in days.values()))/n
        result.append(dict(budget=b,objective=mode,races=n,hits=int(hit),hit_rate=hit/n,mean_points=float(np.mean([m[1] for m in v])),median_combined=float(np.median([m[2] for m in v])),equal_return_pct=float(100*mean),uniform_return_pct=float(100*uniform.mean()),return_ci_pct=[float(100*(mean-1.96*se)),float(100*(mean+1.96*se))],families=dict(Counter(m[6] for m in v))))
    # Full race-level audit for every budget and objective.
    with (OUT/f"{pool}_{split}_selected.csv").open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.writer(f);w.writerow(["budget","objective","date","points","combined_odds","equal_return","uniform_return","hit","family","formation","base_probability","market_probability","learned_probability","race_id","shape"])
        for (b,mode),v in sorted(metrics.items()):
            for m in v:w.writerow([b,mode,*m])
    return result
def run(raw,pool):
    rows=[prior.candidates(r,pool) for r in raw]
    dates=np.array([r["date"] for r in raw])
    def idx(a,b):return np.flatnonzero((dates>=a)&(dates<=b))
    train=[]
    # Strict expanding-window out-of-fold probabilities for formation learning.
    for a,b,c in [("2025-07-01","2025-09-30","2025-06-30"),("2025-10-01","2025-12-31","2025-09-30"),("2026-01-01","2026-03-31","2025-12-31")]:
        ti=idx("2025-04-01",c);vi=idx(a,b)
        print("BASE_FOLD",pool,a,len(ti),len(vi),flush=True)
        pp=fit_base([rows[i] for i in ti],[rows[i] for i in vi])
        train.extend(collect(raw,rows,vi,pp,pool,False));del pp;gc.collect()
    ti=idx("2025-04-01","2026-03-31");vi=idx("2026-04-01","2026-04-30");te=idx("2026-05-01","2026-06-30")
    pp=fit_base([rows[i] for i in ti],[rows[i] for i in np.r_[vi,te]])
    val=collect(raw,rows,vi,pp[:len(vi)],pool,True);test=collect(raw,rows,te,pp[len(vi):],pool,True)
    del rows,pp;gc.collect()
    models={};scores={}
    for name,ix in [("no_history",list(range(HSTART))),("history",list(range(len(NAMES))))]:
        model,shrink,baseline,loss=fit_formation(train,val,ix)
        models[name]=(model,shrink,ix)
        scores[name]={"shrink":shrink,"base_validation_loss":baseline,"validation_loss":loss,"importance":dict(zip([NAMES[i] for i in ix],map(float,model.feature_importance(importance_type="gain"))))}
        model.save_model(str(OUT/f"{pool}_{name}.txt"))
    chosen=min(scores,key=lambda k:scores[k]["validation_loss"])
    model,shrink,ix=models[chosen]
    vp=[predict(model,shrink,ix,r) for r in val];tp=[predict(model,shrink,ix,r) for r in test]
    def logloss(rows,preds):
        return float(np.mean([np.mean(-r["y"]*np.log(clip(p))-(1-r["y"])*np.log(1-clip(p))) for r,p in zip(rows,preds)]))
    out={"races":{"formation_train":len(train),"validation":len(val),"test":len(test)},"candidate_counts":{k:{"mean_generated":float(np.mean([r["generated"] for r in rr])),"mean_retained":float(np.mean([len(r["y"]) for r in rr]))} for k,rr in [("train",train),("validation",val),("test",test)]},"chosen":chosen,"models":scores,"test_formation_logloss":{"base":logloss(test,[r["x"][:,1] for r in test]),"learned":logloss(test,tp)},"validation":evaluate(val,vp,pool,"validation"),"test":evaluate(test,tp,pool,"test")}
    print("FORMATION_RESULT",pool,json.dumps(out,ensure_ascii=False),flush=True)
    return out
def main():
    raw,cov,lc=prior.line.records()
    out={"coverage":cov,"line_coverage":lc,"results":{},"limits":["Final odds, not live purchase prices; historical hypothetical returns only.","Candidate rectangles, foldbacks and unions are bounded search, not globally exhaustive; 2 to 48 distinct tickets, no single-ticket formations.","Formation training uses expanding-window cross-fitted base probabilities; April model choice; May-June previously researched, not pristine.","Budgets and objectives fixed for comparison, no final-test optimization; every race included where a candidate exists, no purchase/skip decision learned.","Equal-return allocation is theoretical fractional stake; uniform ticket allocation reported separately.","Independent rate is prior-day research proxy; no production implementation."]}
    for pool in ("trio","trifecta"):
        out["results"][pool]=run(raw,pool);gc.collect()
        (OUT/"report.json").write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding="utf-8")
    print("COMPLETE",flush=True)
if __name__=="__main__":main()
