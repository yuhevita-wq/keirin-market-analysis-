#!/usr/bin/env python3
"""2023 direct association: finalized independent trifecta probability vs raw final odds.

NO market inputs are used to train, select, change, reweight or calibrate the AI.
Exclusively compare full matched race ID + 3連単 ticket books against archived
2023 historical FINISHED payout odds. No automatic betting decisions, no ROI,
no assumed takeout rate, no pseudo-price filling.

For price level compare p vs 1/odds AS RAW OBSERVED (NOT fair probability).
For distribution shape also compare p vs (1/odds) / sum(1/odds), labelled
conditional relative market share, NOT an assumed takeout correction.
We retain Jan-Oct IN-SAMPLE vs Nov-Dec base-model holdout separately.
WARNING: 2024-fitted calibrator used for descriptive BACKCAST on all 2023;
its 2023 backcast is NOT chronological prediction.
"""
from __future__ import annotations
import gzip
import json
import hashlib
import itertools
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"research/independent_trifecta_2023"))
import observe as fixed

ODDS=ROOT/"data/odds_only/odds_2023.jsonl.gz"
ODDS_AUDIT=ROOT/"data/audits/odds_only_export.json"
CALIBRATOR=ROOT/"data/audits/independent_trifecta_recalibration_2024.json"
OUT=ROOT/"data/audits/trifecta_market_association_2023.json"
SAMPLE=ROOT/"data/audits/trifecta_market_association_sample_2023.json"
ROWS=ROOT/"data/audits/trifecta_market_association_full_matches_2023.jsonl.gz"
EPS=1e-15
PROB_BANDS=[
  ("<0.1%",0,.001),("0.1-0.5%",.001,.005),
  ("0.5-1%",.005,.01),("1-2%",.01,.02),
  ("2-5%",.02,.05),("5-10%",.05,.1),
  ("10-20%",.1,.2),("20%+",.2,1.00001)
]
RATIO_BANDS=[
 ("p/q <0.25",0,.25),
 ("0.25-0.5",.25,.5),
 ("0.5-0.8",.5,.8),
 ("0.8-1.25",.8,1.25),
 ("1.25-2",1.25,2),
 ("2-4",2,4),
 ("4+",4,math.inf)
]
PRICE_BANDS=[
 ("1-10x",1,10),("10-30x",10,30),("30-100x",30,100),
 ("100-300x",100,300),("300-1000x",300,1000),
 ("1000-5000x",1000,5000),("5000x+",5000,math.inf)
]

def open_odds():
    meta=json.loads(ODDS_AUDIT.read_text("utf-8"))
    assert meta["years"]["2023"]["complete_races"]==6301
    assert meta["price_phase"]=="final"
    checksum=hashlib.sha256(ODDS.read_bytes()).hexdigest()
    if checksum!=meta["years"]["2023"]["sha256"]:
        raise RuntimeError("2023 raw final odds archive sha256 changed")
    books={}
    with gzip.open(ODDS,"rt",encoding="utf-8") as stream:
        for line in stream:
            d=json.loads(line)
            if set(d)!={"race_id","race_type","odds"}:raise ValueError("Unknown market schema")
            rid=str(d["race_id"])
            if rid in books:raise ValueError("Duplicate odds book "+rid)
            books[rid]={"race_type":d["race_type"],"odds":d["odds"]["3連単"],
                        "trio_size":len(d["odds"]["3連複"])}
    if len(books)!=6301:raise RuntimeError("Incomplete odds archive")
    return books,checksum

def make_model_and_calibrator():
    model_report=json.loads(CALIBRATOR.read_text("utf-8"))
    assert model_report["accepted"] and model_report["model_adopted"]=="joint"
    assert model_report["2025_or_later_accessed"] is False
    alpha=model_report["fitted_coefficients"]["joint"]
    events,coverage=fixed.ingest()
    train={k:v for k,v in events.items() if v["date"]<=fixed.TRAIN_END}
    if len(train)!=19248 or len(events)!=23440:
        raise RuntimeError("2023 model universe or training sample changed")
    model=fixed.train_model(train,True)
    return events,model,alpha,coverage

def flags(e,t):
    a,b,c=t
    ga=e["positions"][a][0]
    gb=e["positions"][b][0]
    gc=e["positions"][c][0]
    return (int(ga==gb),int(ga==gc),int(gb==gc),int(ga==gb==gc))

def correction(e,dist,coeff):
    alpha=float(coeff["alpha"])
    betas=tuple(float(x) for x in coeff["line_coefs"])
    values={t:alpha*math.log(max(p,EPS))+sum(x*y for x,y in zip(betas,flags(e,t)))
            for t,p in dist.items()}
    top=max(values.values())
    weights={t:math.exp(v-top) for t,v in values.items()}
    total=sum(weights.values())
    answer={t:v/total for t,v in weights.items()}
    if abs(sum(answer.values())-1)>1e-9:raise RuntimeError("Non-unit corrected p")
    return answer

def choose_category(race_type):
    s=str(race_type).upper()
    if re.search(r"G[123]",s):return "G1-G3"
    if "チャレンジ" in s or "A3" in s:return "A3"
    if re.search(r"A[12]|Ａ[１２]|Ａ級",s):return "A1-A2"
    if "S" in s or "Ｓ" in s:return "S"
    return "other"

def unit_stats():
    return {"races":0,"sum_spearman":0.,"sum_pearson_log":0.,
            "corr_spearman":[],"corr_pearson_log":[],
            "sum_pearson_prob":0.,"sum_js":0.,"overround":[],
            "top1_agree":0,"top3_any_overlap":0,"top3_common_count":0,
            "top10_common_count":0,
            "winner_ai_top1":0,"winner_market_top1":0,
            "winner_ai_top3":0,"winner_market_top3":0,
            "sum_log_p_over_implied_raw":0.,
            "tickets":0}
def join_stats(m,rho,pearson,linear,js,overround,ta,tm,truth,ptrue,market_inv):
    m["races"]+=1;m["sum_spearman"]+=rho;m["sum_pearson_log"]+=pearson
    m["sum_pearson_prob"]+=linear;m["sum_js"]+=js
    m["corr_spearman"].append(float(rho))
    m["corr_pearson_log"].append(float(pearson))
    m["overround"].append(float(overround))
    m["top1_agree"]+=int(ta[0]==tm[0])
    m["top3_any_overlap"]+=int(bool(set(ta[:3])&set(tm[:3])))
    m["top3_common_count"]+=len(set(ta[:3])&set(tm[:3]))
    m["top10_common_count"]+=len(set(ta[:10])&set(tm[:10]))
    m["winner_ai_top1"]+=int(truth==ta[0])
    m["winner_market_top1"]+=int(truth==tm[0])
    m["winner_ai_top3"]+=int(truth in ta[:3])
    m["winner_market_top3"]+=int(truth in tm[:3])
    m["sum_log_p_over_implied_raw"]+=math.log(ptrue/market_inv)

def describe_stats(m):
    n=m["races"]
    if not n:return None
    spe=np.array(m["corr_spearman"],dtype=float)
    pe=np.array(m["corr_pearson_log"],dtype=float)
    ov=np.array(m["overround"],dtype=float)
    return {"races":n,
            "spearman_within_race_mean":round(m["sum_spearman"]/n,7),
            "spearman_within_race_median":round(float(np.median(spe)),7),
            "spearman_q10_q90":[round(float(x),7) for x in np.quantile(spe,[.1,.9])],
            "spearman_below_zero_races":int(np.sum(spe<0)),
            "spearman_above_0p5_pct":round(100*float(np.mean(spe>.5)),3),
            "spearman_above_0p7_pct":round(100*float(np.mean(spe>.7)),3),
            "pearson_logp_vs_log_inverse_odds_mean":round(m["sum_pearson_log"]/n,7),
            "pearson_logp_vs_log_inverse_odds_median":round(float(np.median(pe)),7),
            "pearson_prob_p_vs_inverse_odds_mean":round(m["sum_pearson_prob"]/n,7),
            "mean_jensen_shannon_divergence_nats_normalized_market_shape":round(m["sum_js"]/n,7),
            "same_market_and_ai_top1_pct":round(100*m["top1_agree"]/n,5),
            "top3_any_intersection_pct":round(100*m["top3_any_overlap"]/n,5),
            "mean_common_in_top3":round(m["top3_common_count"]/n,6),
            "mean_common_in_top10":round(m["top10_common_count"]/n,6),
            "actual_winner_ai_top1_pct":round(100*m["winner_ai_top1"]/n,5),
            "actual_winner_market_top1_pct":round(100*m["winner_market_top1"]/n,5),
            "actual_winner_ai_top3_pct":round(100*m["winner_ai_top3"]/n,5),
            "actual_winner_market_top3_pct":round(100*m["winner_market_top3"]/n,5),
            "sum_raw_inverse_odds_median":round(float(np.median(ov)),6),
            "sum_raw_inverse_odds_q10_q90":[round(float(x),6) for x in np.quantile(ov,[.1,.9])],
            "sum_raw_inverse_odds_min_max":[round(float(x),6) for x in [ov.min(),ov.max()]],
            "raw_inverse_odds_note":"sum(1/O) is directly measured whole-book reciprocal sum, NOT a fitted deduction rate"}
def share_stats():
    return {"tickets":0,"sum_p":0.,"sum_market_q":0.,"sum_raw_inverse_odds":0.,
            "winner_tickets":0,"sum_odds":0.}
def add_share(a,p,q,raw,odd,win):
    a["tickets"]+=1;a["sum_p"]+=p;a["sum_market_q"]+=q
    a["sum_raw_inverse_odds"]+=raw
    a["sum_odds"]+=odd
    a["winner_tickets"]+=int(win)
def finish_share(x):
    n=x["tickets"]
    if not n:return None
    return {"tickets":n,"mean_ai_probability_pct":round(100*x["sum_p"]/n,7),
            "mean_relative_market_shape_pct":round(100*x["sum_market_q"]/n,7),
            "mean_raw_inverse_odds_pct":round(100*x["sum_raw_inverse_odds"]/n,7),
            "actual_ticket_hit_frequency_pct":round(100*x["winner_tickets"]/n,7),
            "mean_displayed_odds":round(x["sum_odds"]/n,5),
            "ai_probability_mass":round(x["sum_p"],4),
            "relative_market_mass":round(x["sum_market_q"],4),
            "actual_winners":x["winner_tickets"]}
def structural_stats():
    return {"n":0,"sum_ai":0.,"sum_market_q":0.,"actual":0}
def summarize_struct(d):
    n=d["n"]
    return {"races":n,"ai_estimated_event_pct":round(100*d["sum_ai"]/n,5),
            "market_relative_distribution_event_pct":round(100*d["sum_market_q"]/n,5),
            "empirical_event_pct":round(100*d["actual"]/n,5)}
def ccorr(x,y):
    sx=float(np.std(x));sy=float(np.std(y))
    if sx<=1e-14 or sy<=1e-14:return float("nan")
    return float(np.corrcoef(x,y)[0,1])
def numeric(x):
    v=float(x)
    if not math.isfinite(v) or v<1:raise ValueError("Bad price")
    return v

def main():
    books,price_checksum=open_odds()
    events,model,coef,model_coverage=make_model_and_calibrator()
    p_book=set(events);price_book=set(books)
    only_price=sorted(price_book-p_book)
    matched=sorted(p_book&price_book,key=lambda rid:(events[rid]["date"],rid))
    print("EXACT_ID_CANDIDATES",len(matched),"ONLY_PRICE",len(only_price),flush=True)
    stat=defaultdict(unit_stats)
    stat_precal=defaultdict(unit_stats)
    prob_band=defaultdict(share_stats)
    odds_band=defaultdict(share_stats)
    ratio_band=defaultdict(share_stats)
    struct=defaultdict(structural_stats)
    reason=Counter()
    comparable=0;tickets=0;sample=None
    leakage_parts={"in_sample_2023_jan_oct":0,"forward_2023_nov_dec":0}
    # Independent price export and model have exact matching ticket IDs.
    # Suppress all optional filling of prices: only complete books allowed.
    OUT.parent.mkdir(parents=True,exist_ok=True)
    with gzip.open(ROWS,"wt",encoding="utf-8",compresslevel=6) as export:
        for rid in matched:
            e=events[rid];market=books[rid]["odds"]
            try:
                old=fixed.distribution(e,model,True)
                adjusted=correction(e,old,coef)
                keys=sorted(old)
                token_names=["-".join(k) for k in keys]
                if len(market)!=len(keys) or set(market)!=set(token_names):
                    reason["market_ticket_grid_differs_exactly"]+=1
                    continue
                O=np.array([numeric(market[name]) for name in token_names],dtype=np.float64)
                inv=1/O
                s_inv=float(inv.sum())
                Q=inv/s_inv
                p=np.array([adjusted[k] for k in keys],dtype=np.float64)
                p_before=np.array([old[k] for k in keys],dtype=np.float64)
                if len(keys)!=len(e["scores"])*(len(e["scores"])-1)*(len(e["scores"])-2):
                    raise ValueError("Ticket grid invalid")
                if abs(float(p.sum())-1)>1e-8:
                    raise ValueError("Original model no longer normalized")
                truth=e["order"]
                if truth not in adjusted:raise ValueError("Actual not in grid")
                if not (np.all(p>0) and np.all(inv>0) and np.all(np.isfinite(O))):
                    raise ValueError("Bad probabilities")
                lnp=np.log(p)
                lni=np.log(inv)
                rho=ccorr(rankdata(lnp),rankdata(lni))
                pear=ccorr(lnp,lni)
                lin=ccorr(p,inv)
                if not all(math.isfinite(x) for x in (rho,pear,lin)):
                    reason["constant_or_invalid_ranking"]+=1
                    continue
                midpoint=(p+Q)/2.
                js=0.5*(float(np.sum(p*np.log(p/midpoint)))+
                        float(np.sum(Q*np.log(Q/midpoint))))
                p_prelog=np.log(p_before)
                pre_rho=ccorr(rankdata(p_prelog),rankdata(lni))
                pre_pear=ccorr(p_prelog,lni)
                if not all(math.isfinite(x) for x in (pre_rho,pre_pear)):
                    reason["constant_original_model"]+=1
                    continue
            except (KeyError,ValueError,TypeError,OverflowError) as ex:
                reason["exception_"+type(ex).__name__]+=1;continue
            # Valid: exact same possible trifecta tickets and joined race ID.
            category=choose_category(books[rid]["race_type"])
            shape="-".join(map(str,sorted(map(len,e["groups"]),reverse=True)))
            partition=("in_sample_2023_jan_oct" if e["date"]<="2023-10-31"
                       else "forward_2023_nov_dec")
            leakage_parts[partition]+=1
            truth_idx=keys.index(truth)
            market_top_idx=np.argsort(-inv,kind="stable")
            ai_top_idx=np.argsort(-p,kind="stable")
            p_pre_top_idx=np.argsort(-p_before,kind="stable")
            top_ai=[keys[int(z)] for z in ai_top_idx]
            top_market=[keys[int(z)] for z in market_top_idx]
            top_pre=[keys[int(z)] for z in p_pre_top_idx]
            ai_at_true=float(p[truth_idx]); inv_at_true=float(inv[truth_idx])
            for key in ("overall","period_"+partition,
                        "class_"+category,"shape_"+shape,
                        "field_"+str(len(e["scores"]))):
                join_stats(stat[key],rho,pear,lin,js,s_inv,top_ai,top_market,truth,ai_at_true,inv_at_true)
            join_stats(stat_precal["overall_original_2023_model"],pre_rho,pre_pear,ccorr(p_before,inv),
                       0.,s_inv,top_pre,top_market,truth,float(p_before[truth_idx]),inv_at_true)
            comparable+=1;tickets+=len(keys)
            # One event per race, market shape vs model event totals.
            event_ai=[0.]*5;event_market=[0.]*5
            actual_flags=flags(e,truth)
            for idx,t in enumerate(keys):
                f=flags(e,t)
                event_flags=(*f,int(len({e["positions"][z][0] for z in t})==3))
                for z,flag in enumerate(event_flags):
                    if flag:
                        event_ai[z]+=float(p[idx]);event_market[z]+=float(Q[idx])
            for j,event in enumerate(("same_line_12","same_line_13","same_line_23","same_line_123","three_distinct_lines")):
                z=struct[event];z["n"]+=1;z["sum_ai"]+=event_ai[j]
                z["sum_market_q"]+=event_market[j]
                z["actual"]+=int(actual_flags[j] if j<4 else len({e["positions"][z][0] for z in truth})==3)
            # Ticket-level relationship by fixed axes; true winners used only
            # to report descriptive hit frequencies, never for fitting.
            for idx,t in enumerate(keys):
                pi=float(p[idx]);qi=float(Q[idx]);raw=float(inv[idx]);O_i=float(O[idx])
                won=(idx==truth_idx)
                pb=next(label for label,lo,hi in PROB_BANDS if lo<=pi<hi)
                ob=next(label for label,lo,hi in PRICE_BANDS if lo<=O_i<hi)
                rb=next(label for label,lo,hi in RATIO_BANDS if lo<=pi/qi<hi)
                add_share(prob_band[pb],pi,qi,raw,O_i,won)
                add_share(odds_band[ob],pi,qi,raw,O_i,won)
                add_share(ratio_band[rb],pi,qi,raw,O_i,won)
            record={
                "race_id":rid,"race_date":e["date"],
                "subpopulation":partition,"race_type":books[rid]["race_type"],
                "line_shape":shape,"runners":len(e["scores"]),
                "market_final_price_only":True,
                "s_raw_inverse_odds":round(s_inv,9),
                "spearman_model_vs_raw_inverse_odds":round(rho,9),
                "pearson_log_model_vs_raw_inverse_odds":round(pear,9),
                "pearson_probability_vs_raw_inverse_odds":round(lin,9),
                "js_divergence_model_vs_relative_market_shape":round(js,9),
                "actual_trifecta":"-".join(truth),
                "model_at_actual":round(ai_at_true,10),
                "raw_inverse_odds_at_actual":round(inv_at_true,10),
                "actual_final_odds":float(O[truth_idx]),
                "market_top1":"-".join(top_market[0]),
                "model_top1":"-".join(top_ai[0]),
                "all_trifecta":{
                    token_names[idx]:{
                        "p_ai_corrected":round(float(p[idx]),12),
                        "p_ai_original_2023":round(float(p_before[idx]),12),
                        "final_odds":float(O[idx]),
                        "market_raw_inverse_odds":round(float(inv[idx]),12),
                        "market_relative_q":round(float(Q[idx]),12),
                    }
                    for idx in range(len(keys))
                }
            }
            export.write(json.dumps(record,ensure_ascii=False,separators=(",",":"))+"\n")
            if sample is None and len(e["scores"])==7 and partition=="forward_2023_nov_dec":
                sample={"race_id":rid,"date":e["date"],"period":partition,
                        "race_type":books[rid]["race_type"],"line_shape":shape,
                        "trifecta_winning":"-".join(truth),
                        "relative_market_share_formula":"q_i=(1/O_i)/sum_j(1/O_j); purely shape, no takeout rate",
                        "sum_market_inverse_odds":s_inv,
                        "spearman":rho,"pearson_log":pear,
                        "top10_model":[{"ticket":"-".join(keys[int(z)]),"p_ai_pct":round(100*float(p[z]),7),
                                        "odds":float(O[z]),"raw_inverse_odds_pct":round(100*float(inv[z]),7)}
                                       for z in ai_top_idx[:10]],
                        "top10_market":[{"ticket":"-".join(keys[int(z)]),"p_ai_pct":round(100*float(p[z]),7),
                                        "odds":float(O[z]),"raw_inverse_odds_pct":round(100*float(inv[z]),7)}
                                       for z in market_top_idx[:10]],
                        "full_210":record["all_trifecta"]}
            if comparable%1000==0:
                print("MATCHED_PRICES_AND_AI",comparable,len(matched),flush=True)
    if comparable < 2000:
        raise RuntimeError("Exact-ID complete price coverage unexpectedly small")
    report={
       "state":"COMPLETED_2023_TRIFECTA_PRICE_PROBABILITY_ASSOCIATION",
       "purpose":"Descriptive relationship between independent probability and observed raw finalized 3連単 payout odds",
       "market_odds_in_ai_training":False,"profit_roi_analyzed":False,
       "takeout_fraction_estimated_or_applied":False,
       "raw_price_definition":"O = displayed final decimal odds including any payout rules; raw inverse price signal 1/O, NEVER interpreted as true probability",
       "relative_market_shape_definition":"q = (1/O) / sum_for_same_race_and_bettype(1/O); conditional relative rank/distribution only, not estimated deduction rate",
       "final_odds_not_live_purchase_price":True,
       "2023_in_sample_warning":"2023 Jan-Oct outcomes used for original model; only 2023 Nov-Dec unused by original training",
       "2024_backcast_warning":"Corrected AI applied retrospectively to 2023 using coefficients fitted on 2024 H1; comparisons in 2023 are exploratory and NOT independent backtest or 2023 chronological predictions",
       "training_2023_original":["2023-01-01","2023-10-31"],
       "2024_calibrator_fit":["2024-01-01","2024-06-30"],
       "source_checksum_sha256_original_final_2023_odds":price_checksum,
       "source_2023_odds_file":"data/odds_only/odds_2023.jsonl.gz",
       "model_source":"research/independent_trifecta_2023/observe.py",
       "calibrator_source":"data/audits/independent_trifecta_recalibration_2024.json",
       "counts":{
          "2023_price_complete_races":len(books),
          "2023_model_eligible_races":len(events),
          "exact_race_id_intersection_before_ticket_checks":len(matched),
          "market_complete_but_not_model_ready":len(only_price),
          "matched_all_trifecta_and_model_races":comparable,
          "matched_exact_trifecta_ticket_pairs":tickets,
          "split":leakage_parts,
          "join_rejection":dict(reason)
       },
       "by_group":{k:describe_stats(v) for k,v in stat.items()},
       "original_2023_model_control":{k:describe_stats(v) for k,v in stat_precal.items()},
       "ticket_buckets_by_ai_probability":{k:finish_share(prob_band[k]) for k,_,_ in PROB_BANDS},
       "ticket_buckets_by_final_odds":{k:finish_share(odds_band[k]) for k,_,_ in PRICE_BANDS},
       "ticket_buckets_by_ai_vs_relative_market_shape":{k:finish_share(ratio_band[k]) for k,_,_ in RATIO_BANDS},
       "structural_probability_vs_relative_market_share":{
           k:summarize_struct(v) for k,v in struct.items()},
       "all_matches_archive_file":str(ROWS.relative_to(ROOT)),
       "sample_210_ticket_file":str(SAMPLE.relative_to(ROOT)),
       "limitations":[
         "2023 prices are FINAL KDreams odds; not a T-15 minute order placement snapshot.",
         "Only 6301 historical complete market books available, not the full 2023 race universe.",
         "AI 2023 Jan-Oct fit uses outcomes from the same historical period: association descriptive only.",
         "2024 fitted calibrator used for backcast; it cannot represent information available at the time of 2023 races.",
         "Market-relative q strips book-wide scale solely for shape correlation; q is not a claimed unbiased market probability or adjusted takeout.",
         "The raw reciprocal sum of odds does NOT directly identify regulated takeout or betting volume.",
         "Rider score and line snapshots collected retrospectively; no independent verification of availability as-of historical betting cutoff.",
         "Within-race correlation and discrepancies do not establish model edge, market mispricing, positive expectation or tradability.",
         "Only exact full-book joined ID and ticket grids; no guessed or filled price."
       ]
    }
    OUT.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    SAMPLE.write_text(json.dumps(sample,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("2023_MARKET_ASSOCIATION_COMPLETE",json.dumps({
      "races":comparable,"ticket_pairs":tickets,
      "all":report["by_group"].get("overall"),
      "nov_dec":report["by_group"].get("period_forward_2023_nov_dec"),
      "structures":report["structural_probability_vs_relative_market_share"]
    },ensure_ascii=False),flush=True)
if __name__=="__main__":
    main()
