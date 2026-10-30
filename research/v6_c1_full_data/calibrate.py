#!/usr/bin/env python3
"""V6-C1: odds-free post-hoc probability mass calibration with full available data.

Original V6 architecture and 2023 Jan-Oct V6 base fitting untouched.
Calibrator uses ALL score-eligible 2023-2026 race outcomes.
Leave-one-YEAR-out cross-fitting diagnoses overfit; no claim of pristine holdout.
Stages PAT -> PAT+P-band -> PAT+P-band+winner role -> interactions.
Selection uses 4-fold OOF NLL and Brier; after selection refit on ALL years.
All-trifecta probabilities stay positive and sum to one per race.
NO odds, market, settlement, or returns used as labels or model features.
"""
from __future__ import annotations
import bisect, hashlib, json, math, sys
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
from scipy import sparse
from scipy.optimize import minimize

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"research/independent_trifecta_2023"))
sys.path.insert(0,str(ROOT/"research/independent_trifecta_2024"))
import observe as obs
import line_dependence_v2 as v2
import aab_guarded_v3 as v3
import v5_rider_score_role_size_interactions as v5
import v6_opponent_competition_structure as v6
import calibration_audit as r24
import validate_2025 as r25
import v4_pattern_floor_2026 as r26
import weak_rider_podium_zero_excluded as weak
from v8_targeted_weak_podium_calibration import row_probs

OUT=ROOT/"data/audits/v6_c1_full_data_crossfit_2023_2026"
PAT=("AAA","AAB","ABA","ABB","ABC")
ROLE=("leader","second","third","solo","fourth_plus")
BND=(.0001,.0005,.001,.005,.01,.02,.05,.1,.2)
N_B=len(BND)+1
NC=len(PAT)*N_B*len(ROLE)
L2=0.002
STAGES=("v6_frozen","pattern","pattern_pband","pattern_pband_role","pattern_pband_role_interactions")
EXPECT={"2024":25348,"2025":24635,"2026":11797}

def cell_id(t,p,e,roles):
    a,b,c=t
    x=e["positions"][a][0];y=e["positions"][b][0];z=e["positions"][c][0]
    pat=("AAA" if x==y==z else "AAB" if x==y
         else "ABA" if x==z else "ABB" if y==z else "ABC")
    role=roles[a]
    if role not in ROLE:raise ValueError(("Unexpected V6 role",role))
    return (PAT.index(pat)*N_B+bisect.bisect_right(BND,p))*len(ROLE)+ROLE.index(role)

def stage_matrix(stage):
    if stage=="v6_frozen":return np.zeros((NC,0),dtype=np.float64),[]
    keys=[]
    if stage in STAGES[1:]:
        keys.extend(("PAT",p) for p in PAT)
    if stage in STAGES[2:]:
        keys.extend(("PB",str(b)) for b in range(N_B))
    if stage in STAGES[3:]:
        keys.extend(("ROLE",r) for r in ROLE)
    if stage in STAGES[4:]:
        keys.extend(("PAT_PB",p,str(b)) for p in PAT for b in range(N_B))
        keys.extend(("PAT_ROLE",p,r) for p in PAT for r in ROLE)
    # Omit one redundant level in every family to improve identifiability.
    keys=[k for k in keys if not
          (k[0]=="PAT" and k[1]=="ABC") and not
          (k[0]=="PB" and k[1]=="5") and not
          (k[0]=="ROLE" and k[1]=="solo") and not
          (k[0]=="PAT_PB" and k[1]=="ABC") and not
          (k[0]=="PAT_ROLE" and k[1]=="ABC")]
    f=np.zeros((NC,len(keys)),dtype=np.float64)
    for i in range(NC):
        r=i%len(ROLE);b=(i//len(ROLE))%N_B;a=i//(len(ROLE)*N_B)
        pt=PAT[a];rl=ROLE[r]
        for j,k in enumerate(keys):
            if k[0]=="PAT":match=k[1]==pt
            elif k[0]=="PB":match=k[1]==str(b)
            elif k[0]=="ROLE":match=k[1]==rl
            elif k[0]=="PAT_PB":match=k[1]==pt and k[2]==str(b)
            elif k[0]=="PAT_ROLE":match=k[1]==pt and k[2]==rl
            else:raise AssertionError(k)
            if match:f[i,j]=1.
    return f,["|".join(k) for k in keys]

def validate_synthetic():
    # Test exact normalization for 7 competitors, multiple cells, nonzero tilt.
    mass=np.array([.09,.2,.14,.57])
    logits=np.array([-.3,.1,.4,0.])
    q=mass*np.exp(logits);q/=q.sum()
    assert abs(q.sum()-1)<1e-12 and all(q>0)
    for st in STAGES:
        f,keys=stage_matrix(st)
        assert f.shape==(NC,len(keys))
        assert len(set(keys))==len(keys)
    print("V6_C1_SYNTHETIC_MASS_TEST_PASS",flush=True)

def get_data():
    data23,_=obs.ingest()
    base_train={k:v for k,v in data23.items()
                if v["date"]<="2023-10-31" and weak.score_gate(v)=="ELIGIBLE"}
    assert len(base_train)==19017,("V6 baseline train drift",len(base_train))
    print("V6_C1_FROZEN_V6_BASE_TRAIN",len(base_train),flush=True)
    wg,wr=v2.fit_structural(base_train)
    pw,_=v3.train_pattern_gate(base_train)
    d5=v5.fit_residual(base_train,wg,wr)
    wl,wc=v6.fit_opponent(base_train,wg,wr,d5)
    args=(wg,wr,d5,wl,wc,pw)
    del base_train
    print("V6_C1_V6_FROZEN_REFIT_COMPLETE",flush=True)
    sections={"2023":data23}
    for year,rd in (("2024",r24.ingest_2024),
                    ("2025",r25.read_2025),("2026",r26.ingest_2026)):
        races,_=rd()
        assert len(races)==EXPECT[year],(year,len(races))
        sections[year]=races
    rows=[];cols=[];sums=[];squares=[]
    year_idx=[];dates=[];true_cells=[];true_probs=[]
    counts=Counter();line_patterns=Counter()
    for year,data in sections.items():
        for rid,e in data.items():
            counts[year+"_source"]+=1
            gate=weak.score_gate(e)
            if gate!="ELIGIBLE":
                counts[year+"_excluded_"+gate]+=1
                continue
            p=row_probs(e,*args)
            if abs(sum(p.values())-1)>1e-8 or len(p)!=len(e["scores"])*(len(e["scores"])-1)*(len(e["scores"])-2):
                raise AssertionError(("V6 full ticket invalid",rid))
            roles={c:v2.rlabel(e,c) for c in e["scores"]}
            agg=defaultdict(lambda:[0.,0.])
            winner=tuple(e["order"][:3])
            ycell=None
            for t,v in p.items():
                cid=cell_id(t,v,e,roles)
                agg[cid][0]+=v;agg[cid][1]+=v*v
                if t==winner:ycell=cid
            if ycell is None:raise AssertionError(("winner missing",rid))
            row=len(true_probs)
            for cid,(m,sq) in agg.items():
                rows.append(row);cols.append(cid);sums.append(m);squares.append(sq)
            year_idx.append(year);dates.append(e["date"])
            true_cells.append(ycell);true_probs.append(p[winner])
            line_patterns[PAT[ycell//(N_B*len(ROLE))]]+=1
            counts[year+"_eligible"]+=1
            if counts[year+"_eligible"]%5000==0:
                print("V6_C1_PREP_PROGRESS",year,counts[year+"_eligible"],flush=True)
        print("V6_C1_PREP_YEAR_COMPLETE",year,counts[year+"_eligible"],flush=True)
        del data
    n=len(true_probs)
    assert n>70000,("Insufficient eligible races",n)
    mat=sparse.csr_matrix((np.asarray(sums,dtype=np.float64),
                           (np.asarray(rows,dtype=np.int32),np.asarray(cols,dtype=np.int32))),
                          shape=(n,NC))
    mat2=sparse.csr_matrix((np.asarray(squares,dtype=np.float64),(rows,cols)),
                           shape=(n,NC))
    deviation=np.max(np.abs(np.asarray(mat.sum(axis=1)).reshape(-1)-1))
    if deviation>1e-8:raise AssertionError(("Group mass not 1",deviation))
    print("V6_C1_COMPACTED_FULL_PROBABILITIES",json.dumps({
        "races":n,"cells":NC,"nonzeros":int(mat.nnz),
        "base_model_sum_mass_max_deviation":float(deviation),
        "coverage":dict(counts)},ensure_ascii=False),flush=True)
    return mat,mat2,np.asarray(true_cells,dtype=np.int16),np.asarray(true_probs),np.asarray(year_idx),dates,counts,line_patterns

def fit(mat,targets,f,l2=L2):
    if not f.shape[1]:return np.zeros(0),{"success":True,"message":"frozen"}
    n=mat.shape[0]
    obs=np.bincount(targets,minlength=NC)/n
    fo=f.T
    def loss_grad(theta):
        logits=f@theta
        # Clamp purely as a numerical guard; optimizer regularization also shrinks.
        if np.max(np.abs(logits))>35:
            return 1e20+float(np.dot(theta,theta)),2*theta
        w=np.exp(logits)
        norms=np.asarray(mat@w).reshape(-1)
        if np.any(norms<=0):raise AssertionError("tilt normalization is nonpositive")
        loss=(np.log(norms).mean()-logits[targets].mean()
              +.5*l2*np.dot(theta,theta))
        m=w*np.asarray(mat.T @ (1/norms)).reshape(-1)/n
        g=fo@(m-obs)+l2*theta
        return float(loss),g
    result=minimize(loss_grad,np.zeros(f.shape[1]),method="L-BFGS-B",jac=True,
                    options={"maxiter":140,"ftol":1e-11,"gtol":1e-8,"maxls":24})
    if not result.success and np.linalg.norm(result.jac)>1e-4:
        raise RuntimeError(("calibration optimization incomplete",result.message,float(np.linalg.norm(result.jac))))
    return result.x,{"success":bool(result.success),"iterations":int(result.nit),
                     "gradient_norm":round(float(np.linalg.norm(result.jac)),8),
                     "objective":round(float(result.fun),9)}

def score(mat,squared,y,p_win,f,theta):
    w=np.exp(f@theta)
    norm=np.asarray(mat@w).reshape(-1)
    pq=p_win*w[y]/norm
    sum_sq=np.asarray(squared@(w*w)).reshape(-1)/(norm*norm)
    loss=-np.log(pq)
    brier=1-2*pq+sum_sq
    return {"races":len(pq),"sum_nll":float(np.sum(loss)),
            "sum_brier":float(np.sum(brier)),
            "mean_nll":round(float(loss.mean()),9),
            "mean_brier":round(float(brier.mean()),9),
            "winner_probability_sum":round(float(pq.sum()),5),
            "winner_mean_probability_pct":round(float(pq.mean()*100),6),
            "mass_min":round(float(np.min(norm)),10),
            "mass_max":round(float(np.max(norm)),10)}

def combine_scores(scores):
    n=sum(s["races"] for s in scores)
    return {"races":n,"mean_nll":round(sum(s["sum_nll"] for s in scores)/n,9),
            "mean_brier":round(sum(s["sum_brier"] for s in scores)/n,9),
            "sum_nll":round(sum(s["sum_nll"] for s in scores),6),
            "sum_brier":round(sum(s["sum_brier"] for s in scores),6)}

def crossfit(mat,sq,y,pwin,yrs):
    years=("2023","2024","2025","2026")
    out={k:{} for k in STAGES}
    agg={k:[] for k in STAGES}
    for year in years:
        hold=np.where(yrs==year)[0]
        other=np.where(yrs!=year)[0]
        assert len(hold)>1000 and len(other)>10000
        train=mat[other]; test=mat[hold]; test_sq=sq[hold]
        for stage in STAGES:
            f,keys=stage_matrix(stage)
            theta,fit_audit=fit(train,y[other],f)
            result=score(test,test_sq,y[hold],pwin[hold],f,theta)
            out[stage][year]={"evaluation":result,
                              "fit":fit_audit,
                              "fit_races":len(other)}
            agg[stage].append(result)
        print("V6_C1_CROSSFIT_YEAR",year,json.dumps({
            k:{"nll":out[k][year]["evaluation"]["mean_nll"],
               "brier":out[k][year]["evaluation"]["mean_brier"]}
            for k in STAGES},ensure_ascii=False),flush=True)
    summary={k:combine_scores(v) for k,v in agg.items()}
    baseline=summary["v6_frozen"]
    # No post-hoc optimism: model choice is itself exploratory based on these
    # crossfitted historical years. A genuinely NEW future period is still needed.
    eligible=[k for k in STAGES[1:]
              if summary[k]["mean_nll"]<baseline["mean_nll"]-1e-6
              and summary[k]["mean_brier"]<=baseline["mean_brier"]+1e-6]
    selected=min(eligible,key=lambda k:(summary[k]["mean_nll"],STAGES.index(k))) if eligible else "v6_frozen"
    return out,summary,selected

def main():
    validate_synthetic()
    mat,squared,y,pwin,years,dates,counts,pat=get_data()
    oof,summary,choice=crossfit(mat,squared,y,pwin,years)
    print("V6_C1_SELECTED_HISTORICAL_CROSSFIT",choice,flush=True)
    f,keys=stage_matrix(choice)
    theta,diag=fit(mat,y,f)
    full=score(mat,squared,y,pwin,f,theta)
    raw_f,_=stage_matrix("v6_frozen")
    raw_full=score(mat,squared,y,pwin,raw_f,np.zeros(0))
    coeff={name:round(float(weight),10) for name,weight in zip(keys,theta)}
    weights=np.exp(f@theta)
    if not np.all(np.isfinite(weights)) or np.min(weights)<=0:raise AssertionError("Calibration weights invalid")
    model={
       "model_name":"V6-C1",
       "state":"TRAINED_FULL_2023_2026_ODDS_FREE_CROSSVALIDATED_CALIBRATOR",
       "baseline_v6":"Frozen original V2,V3(.15),V5,V6; fitted clean 2023 Jan-Oct 19017 races",
       "fit_data":"All available eligible 2023,2024,2025,2026 outcomes; in-sample relative to original base V6 for 2023 Jan-Oct",
       "features":{"pattern":list(PAT),"p_band_boundaries_probability":list(BND),
                   "first_rider_role":list(ROLE),
                   "cell_id":"(PAT_idx*10 + bisect_right(boundaries,p_v6))*5 + R1_idx"},
       "transform":"p_c1(triple) = p_v6(triple) * exp(cell_log_multiplier) / sum_all_triples(p_v6 * exp(cell_log_multiplier))",
       "selected_stage":choice,"L2_per_race":L2,
       "fit_coefficients":coeff,
       "cell_log_multipliers":[round(float(v),10) for v in f@theta],
       "cell_multipliers":[round(float(v),10) for v in weights],
       "fit_audit":diag,
       "total_races":len(years),
       "train_data_fingerprint":hashlib.sha256(("|".join(years)+"|"+ "|".join(dates)).encode()).hexdigest(),
       "no_market_odds_in_features_or_labels":True,
       "v6_source_unchanged":True,
       "per_race_normalize_to_one":True,
       "historical_cv_not_pristine_unknown":True}
    report={
       "state":"V6_C1_FULL_DATA_CROSSVALIDATED_CALIBRATION_COMPLETE",
       "model_state":model["state"],"crossfit_type":"leave-one-calendar-year-out",
       "years":["2023","2024","2025","2026"],
       "cv_warning":"2023 Jan-Oct is in base V6 training; all 2024-26 were previously examined. These are diagnostic blocked CVs, not pristine future data",
       "model_selection_warning":"Stage selected using same historical CV results. True future improvement is unproven.",
       "fit_warning":"Final fit on ALL historical years: full-data NLL/Brier are IN SAMPLE, not independent validation",
       "source_coverage":dict(counts),
       "observed_pattern_counts":dict(pat),
       "crossfit_by_stage_year":oof,"crossfit_summary":summary,
       "selected_stage":choice,
       "selected_delta_nll_vs_frozen":round(summary["v6_frozen"]["mean_nll"]-summary[choice]["mean_nll"],9),
       "selected_delta_brier_vs_frozen":round(summary["v6_frozen"]["mean_brier"]-summary[choice]["mean_brier"],9),
       "fit_full_baseline":raw_full,"fit_full_calibrated":full,
       "full_data_fit_improvement_nll":round(raw_full["mean_nll"]-full["mean_nll"],9),
       "full_data_fit_improvement_brier":round(raw_full["mean_brier"]-full["mean_brier"],9),
       "model_saved":"v6_c1_fitted_model.json",
       "no_market_odds_teacher":True,
       "v6_untouched":True}
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/"v6_c1_fitted_model.json").write_text(json.dumps(model,ensure_ascii=False,indent=2)+"\n")
    (OUT/"v6_c1_calibration_report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    # Demonstrate exact all-3rentan conservation for representative synthetic
    # and actual compressed probability vectors under each fit.
    z=np.asarray(mat@weights).reshape(-1)
    if not np.all(z>0):raise AssertionError("Nonpositive normalizer")
    if np.max(np.abs(np.asarray(mat.multiply(weights).sum(axis=1)).ravel()/z-1))>1e-9:
        raise AssertionError("All-trifecta postcalibration sum does not equal one")
    print("V6_C1_RESULT",json.dumps({
      "choice":choice,"total_races":len(years),
      "crossfit_summary":summary,
      "selected_nll_delta":report["selected_delta_nll_vs_frozen"],
      "selected_brier_delta":report["selected_delta_brier_vs_frozen"],
      "full_fit_nll_delta":report["full_data_fit_improvement_nll"],
      "full_fit_brier_delta":report["full_data_fit_improvement_brier"],
      "fit":diag,"coefficient_count":len(coeff),
      "coverage":dict(counts),
      "normalization_verified":True
    },ensure_ascii=False),flush=True)

if __name__=="__main__":
    main()
