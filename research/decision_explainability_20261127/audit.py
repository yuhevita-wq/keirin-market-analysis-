"""Out-of-time model explanations, not a new purchase strategy."""
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier,export_text
from sklearn.metrics import accuracy_score,balanced_accuracy_score,mean_squared_error

class Audit:
    def __init__(self,bet,out,feature_names,gate_names,context_n,family_names):
        self.bet=bet;self.out=Path(out);self.names=list(feature_names)
        self.gnames=self.names+list(gate_names);self.context_n=context_n
        self.family_names=family_names;self.history=defaultdict(list)
        self.rows=[];self.rules=[];self.groups=[];self.comparisons=[];self.meta=[];self.ablation=[]
    def save(self):
        for name,rows in (("feature_attribution",self.rows),("surrogate_rules",self.rules),
                          ("behavior_groups",self.groups),("alternative_comparisons",self.comparisons),
                          ("race_explanations",self.meta),("gate_feature_ablation",self.ablation)):
            p=self.out/(name+".json")
            if name=="race_explanations":
                pd.DataFrame(rows).to_csv(self.out/"race_explanations.csv",index=False)
            else:
                p.write_text(json.dumps(rows,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
    def contributions(self,booster,x,names):
        p=np.asarray(booster.predict(x,pred_contrib=True),dtype=float)
        assert p.shape==(len(x),len(names)+1)
        assert np.allclose(p.sum(1),booster.predict(x),rtol=1e-5,atol=1e-6)
        return p[:,:-1]
    def add_stats(self,gen,months,stage,subset,c,names,score=None):
        if len(c)==0:return
        for k,n in enumerate(names):
            z=c[:,k]
            self.rows.append({"generation":gen,"months":months,"stage":stage,"subset":subset,
                "feature":n,"n":len(z),"mean_abs_contribution":float(np.abs(z).mean()),
                "mean_signed_contribution":float(z.mean())})
    def group_stats(self,gen,months,df):
        for col in ("family","points","axis_market_rank","partner_market_rank","axis_history_rank","source","score_band","availability_band"):
            if col not in df:continue
            for val,g in df.groupby(col,dropna=False):
                n=len(g);b=g[g.buy];k=len(b)
                self.groups.append({"generation":gen,"months":months,"dimension":col,"value":str(val),
                    "selected_formation_count":n,"buy_count":k,"buy_rate":k/n,
                    "actual_all_roi":float(g.gross.mean()*100),
                    "actual_buy_roi":float(b.gross.mean()*100) if k else None,
                    "actual_skip_roi":float(g.loc[~g.buy,"gross"].mean()*100) if k<n else None,
                    "buy_hit_rate":float((b.gross>0).mean()) if k else None})
    def surrogate(self,gen,months,stage,x,y,names,day):
        key=(gen,stage);past=self.history[key]
        if past:
            px=np.concatenate([a[0] for a in past]);py=np.concatenate([a[1] for a in past]);pd_=np.concatenate([a[2] for a in past])
            assert pd_.max()<day.min()
            if len(np.unique(py))>1:
                clf=DecisionTreeClassifier(max_depth=4,min_samples_leaf=100,random_state=20261127)
                clf.fit(px,py);pred=clf.predict(x)
                majority=np.unique(py,return_counts=True)
                base=majority[0][np.argmax(majority[1])]
                self.rules.append({"generation":gen,"months":months,"stage":stage,
                    "train_n":len(py),"test_n":len(y),"fidelity":float(accuracy_score(y,pred)),
                    "balanced_fidelity":float(balanced_accuracy_score(y,pred)),
                    "majority_baseline":float(np.mean(y==base)),
                    "rules":export_text(clf,feature_names=list(names),decimals=5)})
        past.append((x.copy(),y.copy(),day.copy()))
    def inspect(self,gen,months,model,gate,X,Y,P,O,D,v,z,pick,G,base,gate_scores,score,meta_get,fam_get,prior_gate_data=None):
        if gen not in (1,5):return
        assert X.shape[1]==len(self.names) and G.shape[1]==len(self.gnames)
        # Preserve model bodies and feature mapping, so analysis can be independently checked.
        model.booster_.save_model(str(self.out/f"g{gen}_m{months}_formation.txt"))
        if gate is not None:gate.booster_.save_model(str(self.out/f"g{gen}_m{months}_gate.txt"))
        day=np.asarray(D[v:z],dtype="datetime64[D]")
        selected_features=X[pick]
        sc=self.contributions(model.booster_,selected_features,self.names)
        self.add_stats(gen,months,"formation","selected",sc,self.names)
        # The best available competitor of a different family, not a randomly chosen losing ticket.
        alts=[];selrows=[];avail=[];ctx=[]
        a=int(O[v])
        for j,r in enumerate(range(v,z)):
            s,t=int(O[r]),int(O[r+1]);fam=np.asarray(fam_get(s,t))
            i=pick[j];sf=fam[i-s];mask=np.flatnonzero(fam!=sf)
            if len(mask):
                k=s+int(mask[np.argmax(score[s-a:t-a][mask])]);alts.append(k);selrows.append(j)
            avail.append(float(np.mean(fam==sf)))
            ctx.append(np.r_[X[s,-self.context_n:],len(fam),*[np.mean(fam==k) for k in range(len(self.family_names))]])
        if alts:
            alt=np.asarray(alts,dtype=int);sr=np.asarray(selrows,dtype=int)
            ac=self.contributions(model.booster_,X[alt],self.names)
            self.add_stats(gen,months,"formation","selected_minus_best_other_family",sc[sr]-ac,self.names)
            for j,k in zip(sr,alt):
                r=v+int(j);s=int(O[r]);t=int(O[r+1]);families=np.asarray(fam_get(s,t))
                self.comparisons.append({"generation":gen,"months":months,"race_index":r,
                    "chosen_family":self.family_names[int(families[pick[j]-s])],
                    "alternative_family":self.family_names[int(families[k-s])],
                    "score_gap":float(base[j]-score[k-a]),"chosen_gross":float(Y[pick[j]]),
                    "alternative_gross":float(Y[k]),"chosen_points":int(P[pick[j]]),"alternative_points":int(P[k])})
        cx=np.asarray(ctx,dtype=float)
        fy=np.array([int(fam_get(int(O[r]),int(O[r+1]))[pick[r-v]-int(O[r])]) for r in range(v,z)])
        cn=self.names[-self.context_n:]+["candidate_count"]+["candidate_share_"+x for x in self.family_names]
        self.surrogate(gen,months,"formation_family",cx,fy,cn,day)
        buy=np.zeros(len(pick),dtype=bool) if gate_scores is None else gate_scores>1
        gc=None
        if gate_scores is not None:
            gc=self.contributions(gate.booster_,G,self.gnames)
            self.add_stats(gen,months,"purchase","all",gc,self.gnames)
            self.add_stats(gen,months,"purchase","buy",gc[buy],self.gnames)
            self.add_stats(gen,months,"purchase","skip",gc[~buy],self.gnames)
            self.surrogate(gen,months,"purchase_decision",G,buy.astype(int),self.gnames,day)
        rows=[]
        for j,r in enumerate(range(v,z)):
            i=pick[j];m=meta_get(r,i-int(O[r]))
            q={"generation":gen,"months":months,"race_index":r,"date":str(day[j]),"family":self.family_names[fy[j]],
                "points":int(P[i]),"gross":float(Y[i]),"formation_score":float(base[j]),
                "candidate_family_share":avail[j],"availability_band":str(int(avail[j]*10)*10)+"%",
                **m}
            top=np.argsort(np.abs(sc[j]))[-5:][::-1]
            q["formation_top_contributions"]=json.dumps({self.names[k]:float(sc[j,k]) for k in top},ensure_ascii=False)
            if gc is not None:
                q["buy"]=bool(buy[j]);q["gate_score"]=float(gate_scores[j])
                q["score_band"]=str(np.floor(gate_scores[j]*10)/10)
                top=np.argsort(np.abs(gc[j]))[-5:][::-1]
                q["purchase_top_contributions"]=json.dumps({self.gnames[k]:float(gc[j,k]) for k in top},ensure_ascii=False)
            else:q["buy"]=False
            rows.append(q)
        self.meta.extend(rows)
        if gc is not None:self.group_stats(gen,months,pd.DataFrame(rows))
        # Refit the purchase model on prior OOF rows with feature groups removed.
        # This measures predictive contribution conditional on the same selected formations,
        # not causality or profit attributable to the underlying physical factor.
        if prior_gate_data is not None and gen==5:
            from lightgbm import LGBMRegressor
            GX,GY=prior_gate_data
            groups={"market":[i for i,n in enumerate(self.gnames) if "market" in n or "entropy" in n],
                    "rider_history":[i for i,n in enumerate(self.gnames) if any(k in n for k in ("rate1","rate2","rate3","logstarts"))],
                    "line":[i for i,n in enumerate(self.gnames) if "line_position" in n or "line_size" in n],
                    "formation":[i for i,n in enumerate(self.gnames) if n.startswith(("family_","source_")) or n in ("points","axis_rank","partner_count","head_count","tail_count","overlap","rider_count")],
                    "first_model_scores":list(range(len(self.names),len(self.gnames)))}
            for label,drop in {"none":[],**groups}.items():
                if label!="none" and not drop:continue
                keep=[i for i in range(G.shape[1]) if i not in drop]
                # Same diagnostic model specification for all ablations (not the production booster).
                reg=LGBMRegressor(objective="regression",n_estimators=80,learning_rate=.04,num_leaves=15,min_child_samples=150,reg_lambda=10,n_jobs=4,verbosity=-1,random_state=20261127,deterministic=True,force_col_wise=True)
                reg.fit(GX[:,keep],GY);p=reg.predict(G[:,keep]);sel=p>1;gross=Y[pick]
                self.ablation.append({"generation":gen,"months":months,"removed":label,
                    "n_features":len(keep),"n":len(p),"bought":int(sel.sum()),
                    "mse":float(mean_squared_error(gross,p)),"return_pct":float(gross[sel].mean()*100) if sel.any() else None,
                    "net_per_race":float(np.mean(np.where(sel,gross-1,0)))})
        self.save()

    def finalize(self):
        # Compact machine-readable log for review even without artifact download.
        for gen in (1,5):
            for stage,subset in (("formation","selected_minus_best_other_family"),("purchase","all"),("purchase","buy"),("purchase","skip")):
                rs=[r for r in self.rows if r["generation"]==gen and r["stage"]==stage and r["subset"]==subset and r["months"]>=6]
                by=defaultdict(list)
                for r in rs:by[r["feature"]].append(r)
                out=[]
                for name,items in by.items():
                    n=sum(r["n"] for r in items)
                    out.append({"feature":name,"mean_abs":sum(r["n"]*r["mean_abs_contribution"] for r in items)/n,
                                "mean_signed":sum(r["n"]*r["mean_signed_contribution"] for r in items)/n})
                print(json.dumps({"audit_top":{"bet":self.bet,"generation":gen,"stage":stage,"subset":subset,
                     "features":sorted(out,key=lambda r:-r["mean_abs"])[:15]}},ensure_ascii=False,allow_nan=False),flush=True)
            df=pd.DataFrame([r for r in self.meta if r["generation"]==gen and r["months"]>=6])
            for col in ("family","points","axis_market_rank","partner_market_rank","axis_history_rank","source","score_band","availability_band","head_market_first_rank","tail_market_podium_rank","head_history_first_rank","tail_history_second_rank"):
                if col not in df:continue
                group=[]
                for val,d in df.groupby(col,dropna=False):
                    b=d[d.buy];s=d[~d.buy];n=len(d)
                    group.append({"value":str(val),"n":n,"buy":len(b),"buy_rate":len(b)/n,
                                  "all_roi":float(d.gross.mean()*100),"buy_roi":float(b.gross.mean()*100) if len(b) else None,
                                  "skip_roi":float(s.gross.mean()*100) if len(s) else None,
                                  "buy_hit_rate":float((b.gross>0).mean()) if len(b) else None,
                                  "mean_points":float(d.points.mean()),"mean_candidate_share":float(d.candidate_family_share.mean())})
                group.sort(key=lambda a:-a["n"])
                print(json.dumps({"audit_behavior":{"bet":self.bet,"generation":gen,"dimension":col,"groups":group[:20],"total_groups":len(group)}},ensure_ascii=False,allow_nan=False),flush=True)
        for r in self.rules:
            print(json.dumps({"audit_surrogate":{"bet":self.bet,**r}},ensure_ascii=False,allow_nan=False),flush=True)
        for r in self.ablation:
            print(json.dumps({"audit_ablation":{"bet":self.bet,**r}},ensure_ascii=False,allow_nan=False),flush=True)
        self.save()
