"""Learn conditional formation/point switching from prior out-of-fold alternatives.
Research only: equal notional stake, reused historical data, no causal claims."""
import json
from collections import defaultdict,Counter
from pathlib import Path
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error
SEED=20261127
def log(x):print(json.dumps(x,ensure_ascii=False,allow_nan=False),flush=True)
def reg():
    return LGBMRegressor(objective="regression",n_estimators=100,learning_rate=.04,
        num_leaves=15,min_child_samples=200,reg_lambda=15,n_jobs=4,verbosity=-1,
        deterministic=True,force_col_wise=True,random_state=SEED)
def rank_summary(s):
    a=[int(x) for x in s.split(",") if x.isdigit()]
    return [len(a),min(a) if a else 0,max(a) if a else 0,float(np.mean(a)) if a else 0.]
def selftest():
    assert rank_summary("1,5")==[2,1,5,3.]
    assert rank_summary("-")==[0,0,0,0.]
    rng=np.random.default_rng(SEED);x=rng.normal(size=(400,8));y=x[:,0]+rng.normal(size=400)
    m=reg();m.fit(x,y);assert np.isfinite(m.predict(x)).all()
class Study:
    def __init__(self,bet,out,names,families,context_n):
        self.bet=bet;self.out=Path(out);self.names=list(names);self.families=list(families)
        self.context_n=context_n;self.history=[];self.rows=[];self.behavior=[];self.decisions=[];self.importance=[];self.cross=[]
        self.columns=None
    def save(self):
        for k,v in (("learning_results",self.rows),("switch_behavior",self.behavior),("feature_contributions",self.importance),("cross_rank_summary",self.cross)):
            p=self.out/(k+".json");p.write_text(json.dumps(v,ensure_ascii=False,allow_nan=False,indent=2))
        pd.DataFrame(self.decisions).to_csv(self.out/"race_decisions.csv",index=False)
    def metrics(self,sel,reward,day,pts,hit=None):
        buy=sel>=0; n=len(sel);k=int(buy.sum());gross=np.zeros(n)
        gross[buy]=reward[sel[buy]]
        _,inv=np.unique(day,return_inverse=True)
        a=np.bincount(inv,weights=gross);b=np.bincount(inv,weights=buy.astype(float));c=np.bincount(inv)
        rng=np.random.default_rng(SEED);ix=rng.integers(0,len(a),size=(1000,len(a)))
        ga=a[ix].sum(1);st=b[ix].sum(1);ct=c[ix].sum(1)
        return {"eligible":n,"bought":k,"return_pct":100*float(gross.sum())/k if k else None,
            "hit_rate":float((gross[buy]>0).mean()) if k else None,
            "mean_points":float(pts[sel[buy]].mean()) if k else None,
            "net_per_race":float((gross.sum()-k)/n),
            "net_ci95":np.quantile((ga-st)/ct,[.025,.975]).tolist(),
            "roi_ci95":(100*np.quantile(ga[st>0]/st[st>0],[.025,.975])).tolist() if (st>0).any() else None}
    def inspect(self,months,X,Y,P,O,D,v,z,pick,score,gate_scores,meta_get,fam_get):
        # A candidate is chosen by the frozen base model within each family/point cell.
        # No winner or payout enters selection. Every cell is compared to the original choice.
        a=int(O[v]);day=np.asarray(D[v:z],dtype="datetime64[D]")
        features=[];base_rows=[];ret=[];points=[];family=[];md=[];offset=[0];base_ix=[];fam_weights=[];race_weights=[]
        rank_keys=["axis_market_rank","partner_market_rank","axis_history_rank","head_market_first_rank","tail_market_podium_rank","head_history_first_rank","tail_history_second_rank"] if self.bet=="exacta" else ["axis_market_rank","partner_market_rank"]
        extra_names=["base_score","candidate_score","candidate_minus_base","family_count","family_share","family_points_count","family_points_share","total_candidate_count","points_delta","family_changed"]+[f"family_is_{k}" for k in self.families]
        extra_names += [f"{k}_{q}" for k in rank_keys for q in ("count","min","max","mean")]
        # Retain candidate features and differences to original formation, no post-race inputs.
        self.columns=self.names+["delta_"+x for x in self.names]+extra_names
        for j,r in enumerate(range(v,z)):
            s,t=int(O[r]),int(O[r+1]);f=np.asarray(fam_get(s,t),dtype=int);q=score[s-a:t-a];pt=P[s:t]
            orig=int(pick[j]);ob=orig-s
            cell={}
            for k in range(len(f)):
                key=(int(f[k]),int(pt[k]))
                if key not in cell or q[k]>q[cell[key]]:cell[key]=k
            assert ob in cell.values()
            counts=Counter(int(k) for k in f);cells=Counter(k[0] for k in cell)
            for (ff,pp),k in sorted(cell.items()):
                i=s+k;m=meta_get(r,k)
                ext=[float(q[ob]),float(q[k]),float(q[k]-q[ob]),counts[ff],counts[ff]/len(f),
                     int(np.sum((f==ff)&(pt==pp))),float(np.mean((f==ff)&(pt==pp))),len(f),pp-int(P[orig]),int(ff!=f[ob])]
                ext += [int(ff==n) for n in range(len(self.families))]
                for key in rank_keys:ext+=rank_summary(m.get(key,"-"))
                features.append(np.r_[X[i],X[i]-X[orig],ext])
                ret.append(float(Y[i]));points.append(int(P[i]));family.append(ff);md.append(m)
                base_rows.append(float(Y[orig]))
                fam_weights.append(1/(len(cells)*cells[ff]));race_weights.append(1/len(cell))
                if k==ob:base_ix.append(len(ret)-1)
            offset.append(len(ret))
        A=np.asarray(features,dtype=np.float32);Yc=np.asarray(ret);B=np.asarray(base_rows);Pc=np.asarray(points);Fc=np.asarray(family)
        W=np.asarray(fam_weights);U=np.asarray(race_weights);off=np.asarray(offset);orig=np.asarray(base_ix)
        assert A.shape[1]==len(self.columns) and len(orig)==len(day) and np.isfinite(A).all()
        assert np.allclose(np.add.reduceat(W,off[:-1]),1) and np.allclose(np.add.reduceat(U,off[:-1]),1)
        assert np.array_equal(Yc[orig],B[orig])
        def emit(policy,sel,train_n):
            row={"bet":self.bet,"months":months,"policy":policy,"train_races":train_n,
                **self.metrics(sel,Yc,day,Pc)}
            self.rows.append(row);log({"result":row});return row
        n_hist=sum(len(h["day"]) for h in self.history)
        emit("original_all",orig,n_hist)
        if gate_scores is not None:emit("original_gate",np.where(gate_scores>1,orig,-1),n_hist)
        # Actual best family is deliberately hindsight-only and never a selection policy.
        # Report potential for changing family/points separately from model skill.
        best=np.array([max(range(off[j],off[j+1]),key=lambda k:Yc[k]) for j in range(len(day))])
        gain=Yc[best]-Yc[orig]
        log({"opportunity":{"bet":self.bet,"months":months,"races":len(day),
            "alternative_improves_n":int((gain>1e-9).sum()),"mean_hindsight_gain":float(gain.mean()),
            "mean_cells":float(np.diff(off).mean()),"note":"hindsight upper bound only"}})
        if self.history:
            assert max(h["day"].max() for h in self.history)<day.min()
            AX=np.concatenate([h["x"] for h in self.history]);AY=np.concatenate([h["y"] for h in self.history]);AB=np.concatenate([h["base"] for h in self.history])
            AW=np.concatenate([h["w"] for h in self.history]);AU=np.concatenate([h["u"] for h in self.history])
            variants={"unbalanced_cells":(list(range(A.shape[1])),AU,"absolute"),
                      "balanced_families":(list(range(A.shape[1])),AW,"absolute"),
                      "balanced_delta":(list(range(A.shape[1])),AW,"delta"),
                      "no_candidate_counts":([i for i,n in enumerate(self.columns) if n not in ("family_count","family_share","family_points_count","family_points_share","total_candidate_count")],AW,"absolute"),
                      "no_rank_interactions":([i for i,n in enumerate(self.columns) if not any(n.startswith(k+"_") for k in rank_keys)],AW,"absolute")}
            feature_groups={
                "history":[i for i,n in enumerate(self.columns) if any(k in n for k in ("rate1","rate2","rate3","logstarts","history_rank","history_first","history_second","history_support","history_mean","log_history_pair","source_history","source_fusion"))],
                "line":[i for i,n in enumerate(self.columns) if any(k in n for k in ("line_position","line_size"))],
                "market":[i for i,n in enumerate(self.columns) if any(k in n for k in ("market","entropy"))],
                "scores":[i for i,n in enumerate(self.columns) if n in ("base_score","candidate_score","candidate_minus_base")]
            }
            for label,drop in feature_groups.items():
                if drop:variants["no_direct_"+label]=([i for i in range(A.shape[1]) if i not in drop],AW,"absolute")
            # Both direct and delta models use genuine out-of-fold candidate rewards.
            # Use identical candidate set for every variant; balance family weights
            # during model fitting rather than deleting favorable alternatives.
            absolute_models={}
            for policy,(keep,weight,target) in variants.items():
                m=reg();m.fit(AX[:,keep],AY if target=="absolute" else AY-AB,sample_weight=weight)
                pred=np.asarray(m.predict(A[:,keep]),dtype=float)
                assert np.isfinite(pred).all()
                if target=="absolute":absolute_models[policy]=pred
                new=[]
                for j in range(len(day)):
                    s,t=off[j:j+2]
                    i=s+int(np.argmax(pred[s:t]))
                    if target=="delta" and pred[i]<=0:i=orig[j]
                    new.append(i)
                new=np.asarray(new,dtype=int)
                # Absolute models' >1 gate is a diagnostic threshold only.
                # Delta predicts improvement, NOT profitability; use a separately
                # trained absolute reward model for its gate.
                abs_pred=pred if target=="absolute" else absolute_models["balanced_families"]
                selected=np.where(abs_pred[new]>1,new,-1)
                emit(policy+"/all",new,n_hist)
                emit(policy+"/gate",selected,n_hist)
                changed=new!=orig;family_changed=Fc[new]!=Fc[orig];point_changed=Pc[new]!=Pc[orig]
                info={"bet":self.bet,"months":months,"policy":policy,"changed":int(changed.sum()),
                      "family_changed":int(family_changed.sum()),"points_changed":int(point_changed.sum()),
                      "mean_actual_gain":float((Yc[new]-Yc[orig]).mean()),
                      "improved_when_changed":float((Yc[new][changed]>Yc[orig][changed]).mean()) if changed.any() else None,
                      "worsened_when_changed":float((Yc[new][changed]<Yc[orig][changed]).mean()) if changed.any() else None,
                      "candidate_reward_mse":float(mean_squared_error(Yc if target=="absolute" else Yc-B,pred,sample_weight=W))}
                self.behavior.append(info);log({"switch":info})
                if policy in ("balanced_families","balanced_delta"):
                    c=np.asarray(m.booster_.predict(A[new][:,keep],pred_contrib=True))
                    assert np.allclose(c.sum(1),pred[new],rtol=1e-5,atol=1e-6)
                    for sub,mask in (("all",np.ones(len(new),dtype=bool)),("changed",changed),("family_changed",family_changed),("point_changed",point_changed)):
                        if not mask.any():continue
                        vals=np.abs(c[mask,:-1]).mean(0);order=np.argsort(vals)[-12:][::-1]
                        entry={"bet":self.bet,"months":months,"policy":policy,"subset":sub,"n":int(mask.sum()),"top":[{"feature":self.columns[keep[i]],"mean_abs":float(vals[i]),"mean_signed":float(c[mask,i].mean())} for i in order]}
                        self.importance.append(entry);log({"switch_importance":entry})
                for j,i in enumerate(new):
                    old=orig[j]
                    self.decisions.append({"bet":self.bet,"months":months,"policy":policy,"date":str(day[j]),"race_id":md[i]["race_id"],
                       "old_family":self.families[Fc[old]],"new_family":self.families[Fc[i]],"old_points":int(Pc[old]),"new_points":int(Pc[i]),
                       "old_tickets":md[old]["tickets"],"new_tickets":md[i]["tickets"],"old_gross":float(Yc[old]),"new_gross":float(Yc[i]),"buy":bool(selected[j]>=0),"pred":float(pred[i]),
                       "axis_market_rank":md[i].get("axis_market_rank","-"),"partner_market_rank":md[i].get("partner_market_rank","-")})
        # Cross-tab of actual selected axes and partners, not separate marginal counts.
        for j,i in enumerate(orig):
            self.cross.append({"months":months,"race_id":md[i]["race_id"],"family":self.families[Fc[i]],"points":int(Pc[i]),
                "axis":md[i].get("axis_market_rank","-"),"partner":md[i].get("partner_market_rank","-"),
                "gross":float(Yc[i]),"buy":bool(gate_scores is not None and gate_scores[j]>1)})
        self.history.append({"x":A,"y":Yc,"base":B,"w":W,"u":U,"day":day})
        (self.out/"switch_feature_names.json").write_text(json.dumps(self.columns,ensure_ascii=False,indent=2))
        self.save()
    def finalize(self):
        for policy in sorted({r["policy"] for r in self.rows}):
            rr=[r for r in self.rows if r["policy"]==policy and r["months"]>=6]
            if not rr:continue
            n=sum(r["bought"] for r in rr);e=sum(r["eligible"] for r in rr)
            g=sum(r["bought"]*(r["return_pct"] or 0)/100 for r in rr)
            log({"pooled":{"bet":self.bet,"policy":policy,"quarters":len(rr),"eligible":e,"bought":n,
               "return_pct":100*g/n if n else None,"net_per_race":(g-n)/e}})
        d=pd.DataFrame(self.cross);d=d[(d.months>=6)&d.buy]
        if len(d):
            groups=[]
            for key,g in d.groupby(["family","axis","partner"]):
                groups.append({"family":key[0],"axis":key[1],"partner":key[2],"n":len(g),"roi":float(g.gross.mean()*100)})
            log({"joint_axis_partner":{"bet":self.bet,"groups":sorted(groups,key=lambda x:-x["n"])[:30]}})
        self.save()

if __name__=="__main__":
    selftest()
    print("switch study self-test passed",flush=True)
