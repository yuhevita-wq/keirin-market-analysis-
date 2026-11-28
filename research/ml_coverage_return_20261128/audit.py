"""Fixed-model coverage/return frontier. No new model fitting or deployment."""
import hashlib,json,sys
from pathlib import Path
import numpy as np
import pandas as pd
OUT=Path("coverage_output");OUT.mkdir(exist_ok=True)
SRC=Path("frozen_source")
MODEL="tree_history_regularized"
MONTHS=(3,6,9,12,15)
K=(1,2,3,4,6,9,10)
TARGETS=(.01,.03,.05,.10,.15,.20)
MIN_PAST=200
MIN_DAYS=30
SEED=20261128
def dump(name,obj):
    (OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False,default=lambda x:x.item() if isinstance(x,np.generic) else x.tolist() if isinstance(x,np.ndarray) else str(x)),encoding="utf-8")
def log(obj):print(json.dumps(obj,ensure_ascii=False,allow_nan=False,default=lambda x:x.item() if isinstance(x,np.generic) else x.tolist()),flush=True)
def find(name):
    p=list(SRC.rglob(name));assert len(p)==1,(name,p);return p[0]
def bound_stats(g,mask):
    mask=np.asarray(mask,dtype=bool);assert len(mask)==len(g)
    gross=g.gross.to_numpy(float);selected=g.loc[mask].copy()
    n=len(selected);eligible=len(g)
    ans={"eligible":eligible,"selected":n,"coverage":n/eligible,"days":selected.date.nunique(),
        "roi":float(gross[mask].mean()) if n else None,"net_per_eligible":float(np.sum(gross[mask]-1)/eligible),
        "hit_rate":float((gross[mask]>0).mean()) if n else None}
    if not n:
        return {**ans,"day_ci95":None,"week_ci95":None,"top5_share":None,"roi_without_top5":None}
    values=np.sort(gross[mask])
    s=values.sum();top5=values[-5:].sum()
    ans.update(top5_share=float(top5/s) if s>0 else None,roi_without_top5=float((s-top5)/n))
    # Resample eligible calendar clusters, including abstentions.
    dates=pd.to_datetime(g.date); rng=np.random.default_rng(SEED)
    for key,labels in (("day_ci95",g.date),("week_ci95",dates.dt.to_period("W").astype(str))):
        _,inv=np.unique(labels,return_inverse=True);m=inv.max()+1
        returns=np.bincount(inv,weights=np.where(mask,gross,0),minlength=m)
        stakes=np.bincount(inv,weights=mask.astype(float),minlength=m)
        idx=rng.integers(0,m,size=(2000,m))
        den=stakes[idx].sum(1);num=returns[idx].sum(1)
        boot=num[den>0]/den[den>0]
        ans[key]=np.quantile(boot,[.025,.975]).tolist() if len(boot) else None
    return ans
def get_threshold(g,target,floor):
    t=float(g.score.quantile(1-target))
    return max(1.,t) if floor else t
def selftest():
    g=pd.DataFrame({"date":["2025-01-01","2025-01-02","2025-01-03","2025-01-04"],"score":[.5,.9,1.1,1.3],"gross":[0,2,0,3]})
    assert get_threshold(g,.5,False)==1.
    assert get_threshold(g,.75,True)==1.
    a=bound_stats(g,g.score>=1)
    assert a["selected"]==2 and a["roi"]==1.5 and a["coverage"]==.5
    assert bound_stats(g,g.score>2)["roi"] is None
    print("coverage audit self-test passed",flush=True)
selftest()
if "--self-test" in sys.argv:sys.exit(0)
report=json.loads(find("final_report.json").read_text())
assert report["bet"]=="trio" and report["next_model_configuration_from_all_known_folds"]==MODEL
assert report["decision"]=="NOT_APPROVED_FOR_LIVE_USE"
frames=[];manifest={}
for m in MONTHS:
    f=find(f"policy_oof_{m}m.csv.gz")
    manifest[str(m)]=hashlib.sha256(f.read_bytes()).hexdigest()
    d=pd.read_csv(f,dtype={"race_id":str,"date":str,"model":str})
    d=d[d.model==MODEL].copy()
    assert d.months.eq(m).all() and not d.duplicated(["race_id","points"]).any()
    assert set(d.points)==set(K)
    n=d.race_id.nunique();assert len(d)==n*len(K)
    assert d.groupby("points").race_id.nunique().eq(n).all()
    assert np.isfinite(d[["score","gross"]].to_numpy()).all() and (d.gross>=0).all() and (d.score>0).all()
    frames.append(d)
F=pd.concat(frames,ignore_index=True);assert F.race_id.nunique()==20512
assert not F.duplicated(["race_id","points"]).any()
# All rules fixed before reading the held-out next fold. Thresholds are
# quantiles of earlier scores, not current-fold ranks or hindsight cutoffs.
# Two tracks: enforce predicted gross >= 1 (strict); allow lower scores
# solely to measure why coverage increases are difficult (diagnostic only).
protocol={"model":MODEL,"source_run":46093918439,"source_sha":"aec47b7b45f06b133a0cda11a543a5903d1da954",
          "file_hashes":manifest,"targets":TARGETS,"points":K,"minimum_past_selected":MIN_PAST,"minimum_past_days":MIN_DAYS,
          "policy":"past-only quantile threshold, then choose point count on past realized ROI, evaluate next quarter without change",
          "tracks":["strict_ev_floor","coverage_only_diagnostic"],
          "selection":"highest past ROI among sufficiently represented point counts with past coverage >= 80% of target; ties favor smaller points; no qualifying rule -> abstain",
          "groups":["fixed_each_point_count","auto_2plus","auto_all"],"limits":["reused historical folds","final odds","not independent","no guarantee of future coverage or return","marginal cluster intervals not multiplicity adjusted","no direct exacta odds","no changes to model or site"]}
dump("protocol.json",protocol)
out=[];realized=[]
for m in MONTHS[1:]:
    past=F[F.months<m];current=F[F.months==m];assert past.date.max()<current.date.min()
    for floor in (True,False):
        track="strict_ev_floor" if floor else "coverage_only_diagnostic"
        for target in TARGETS:
            options=[]
            for k in K:
                h=past[past.points==k];g=current[current.points==k]
                t=get_threshold(h,target,floor);pm=h.score.to_numpy()>=t
                n=int(pm.sum());pr=float(h.gross.to_numpy()[pm].mean()) if n else None
                pdays=h.loc[pm,"date"].nunique();pcov=n/len(h)
                opt={"points":k,"threshold":t,"past_selected":n,"past_coverage":pcov,"past_roi":pr,
                     "past_days":int(pdays),"eligible_for_selection":bool(n>=MIN_PAST and pdays>=MIN_DAYS and pcov>=.8*target)}
                options.append(opt)
            for name,opts in [("fixed_each_point_count",options),("auto_2plus",[x for x in options if x["points"]>=2]),("auto_all",options)]:
                if name=="fixed_each_point_count":choices=opts
                else:
                    qualified=[x for x in opts if x["eligible_for_selection"]]
                    choices=[sorted(qualified,key=lambda x:(-x["past_roi"],x["points"]))[0]] if qualified else [None]
                for choice in choices:
                    if choice is None:
                        g=current[current.points==K[0]];mask=np.zeros(len(g),dtype=bool)
                        pinfo={"points":None,"threshold":None,"past_roi":None}
                    else:
                        g=current[current.points==choice["points"]];mask=g.score.to_numpy()>=choice["threshold"]
                        pinfo=choice
                    row={"months":m,"track":track,"target":target,"group":name,**pinfo,**bound_stats(g,mask)}
                    out.append(row)
                    key=str(choice["points"]) if name=="fixed_each_point_count" else "auto"
                    z=g[["race_id","date","gross"]].copy()
                    z["selected"]=mask;z["track"]=track;z["target"]=target;z["group"]=name;z["fixed_key"]=key;z["months"]=m
                    realized.append(z)
                    if name!="fixed_each_point_count":log({"fold":row})
R=pd.concat(realized,ignore_index=True)
summary=[]
for (track,target,group,key),g in R.groupby(["track","target","group","fixed_key"],sort=True):
    assert not g.race_id.duplicated().any() and len(g)==16253
    s=bound_stats(g,g.selected)
    q=[x for x in out if x["track"]==track and x["target"]==target and x["group"]==group and (group!="fixed_each_point_count" or str(x["points"])==key)]
    row={"track":track,"target":target,"group":group,"fixed_key":key,**s,
         "positive_quarters":sum(x["roi"] is not None and x["roi"]>1 for x in q),
         "quarters_with_selection":sum(x["selected"]>0 for x in q),"quarter_points":[x["points"] for x in q]}
    summary.append(row);log({"pooled":row})
# Reference previous rolling shadow results, not a hand-picked final quarter.
shadow=[x for x in report["outer_policy"] if x["role"]=="shadow" and x["months"]>=6]
n=sum(x["selected"] for x in shadow);gross=sum(x["selected"]*x["roi"] for x in shadow if x["selected"])
baseline={"selected":n,"eligible":16253,"coverage":n/16253,"roi":gross/n,"last_quarter_selected":shadow[-1]["selected"],"last_quarter_roi":shadow[-1]["roi"]}
dump("fold_results.json",out);dump("pooled_results.json",summary);dump("baseline.json",baseline)
log({"baseline":baseline})
dump("status.json",{"status":"completed","model":MODEL,"source_run":46093918439,"not_approved_for_live_use":True,
    "interpretation":"Coverage/ROI tradeoff audit of frozen historical OOF predictions. Do not adopt rules using these same test results."})
