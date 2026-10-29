#!/usr/bin/env python3
"""Outcome-based DIAGNOSTIC of 2023 probability/market discrepancies.

Only re-reads frozen 2023 full same-ID odds/probability join artifact from
2023 study run 40820189970; it does NOT train or adjust probabilities,
decide bets, use a takeout rate, or generate ROI.

2023 Jan-Oct is IN-SAMPLE for AI. 2023 Nov-Dec is only 391 matched races
out-of-sample for the original 2023 AI; the 2024-calibrated model is a
retrospective 2023 BACKCAST throughout.
"""
from __future__ import annotations
import gzip
import json
import math
from collections import defaultdict,Counter
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
JOIN=ROOT/"data/audits/trifecta_market_association_full_matches_2023.jsonl.gz"
PREV=ROOT/"data/audits/trifecta_market_association_2023.json"
OUT=ROOT/"data/audits/trifecta_market_disagreement_diagnostic_2023.json"

RATIO_BANDS=[
 ("p/q<0.25",0,.25),
 ("0.25-0.5",.25,.5),
 ("0.5-0.8",.5,.8),
 ("0.8-1.25",.8,1.25),
 ("1.25-2",1.25,2),
 ("2-4",2,4),
 ("4+",4,math.inf)
]
RAW_RATIOS=[
 ("p*O<0.25",0,.25),
 ("0.25-0.5",.25,.5),
 ("0.5-0.8",.5,.8),
 ("0.8-1.25",.8,1.25),
 ("1.25-2",1.25,2),
 ("2-4",2,4),
 ("4+",4,math.inf)
]

def blank():
    return {"tickets":0,"sum_p":0.,"sum_baseline_p":0.,"sum_q":0.,
            "sum_raw_inverse":0.,"hits":0,"races_touched":0}
def put(x,p,po,q,raw,win):
    x["tickets"]+=1;x["sum_p"]+=p;x["sum_baseline_p"]+=po;x["sum_q"]+=q
    x["sum_raw_inverse"]+=raw;x["hits"]+=int(win)
def score(x):
    if x["tickets"]==0:return None
    n=x["tickets"]
    return {
      "tickets":n,"observed_hits":x["hits"],
      "expected_hits_ai_corrected":round(x["sum_p"],5),
      "expected_hits_ai_original":round(x["sum_baseline_p"],5),
      "expected_hits_relative_market_shape":round(x["sum_q"],5),
      "raw_reciprocal_odds_sum":round(x["sum_raw_inverse"],5),
      "empirical_frequency_pct":round(100*x["hits"]/n,7),
      "mean_corrected_ai_p_pct":round(100*x["sum_p"]/n,7),
      "mean_original_ai_p_pct":round(100*x["sum_baseline_p"]/n,7),
      "mean_market_relative_q_pct":round(100*x["sum_q"]/n,7),
      "mean_raw_inverse_odds_pct":round(100*x["sum_raw_inverse"]/n,7),
      "delta_hits_vs_corrected_ai":round(x["hits"]-x["sum_p"],5),
      "delta_hits_vs_market_relative_shape":round(x["hits"]-x["sum_q"],5)
    }
def sample_stats():
    return {"races":0,"original_logloss":0.,"corrected_logloss":0.,
            "market_relative_logloss":0.,"original_brier":0.,
            "corrected_brier":0.,"market_relative_brier":0.,
            "corrected_beats_market_races":0,
            "original_beats_market_races":0,
            "by_day":defaultdict(lambda:[0.,0.,0.,0])}
def result(st):
    n=st["races"];day=st["by_day"]
    if not n:return None
    mean=(st["market_relative_logloss"]-st["corrected_logloss"])/n
    se=math.sqrt(sum(((vals[2]-vals[1])-mean*vals[3])**2 for vals in day.values()))/n
    # positive says AI corrected beats market q (in log loss)
    original=(st["market_relative_logloss"]-st["original_logloss"])/n
    se_orig=math.sqrt(sum(((vals[2]-vals[0])-original*vals[3])**2 for vals in day.values()))/n
    return {
       "races":n,
       "mean_logloss_original_ai":round(st["original_logloss"]/n,8),
       "mean_logloss_corrected_ai_backcast":round(st["corrected_logloss"]/n,8),
       "mean_logloss_market_relative_shape":round(st["market_relative_logloss"]/n,8),
       "mean_brier_original_ai":round(st["original_brier"]/n,8),
       "mean_brier_corrected_ai_backcast":round(st["corrected_brier"]/n,8),
       "mean_brier_market_relative_shape":round(st["market_relative_brier"]/n,8),
       "ai_corrected_minus_market_logloss_improvement":round(mean,8),
       "ai_corrected_vs_market_day_cluster_95pct_ci":[round(mean-1.96*se,8),round(mean+1.96*se,8)],
       "ai_original_minus_market_logloss_improvement":round(original,8),
       "ai_original_vs_market_day_cluster_95pct_ci":[round(original-1.96*se_orig,8),
                                                      round(original+1.96*se_orig,8)],
       "model_corrected_beats_market_per_race_pct":round(100*st["corrected_beats_market_races"]/n,5),
       "model_original_beats_market_per_race_pct":round(100*st["original_beats_market_races"]/n,5),
       "days":len(day)
    }

def main():
    prior=json.loads(PREV.read_text("utf-8"))
    assert prior["counts"]["matched_all_trifecta_and_model_races"]==5226
    assert prior["takeout_fraction_estimated_or_applied"] is False
    assert JOIN.is_file()
    group=defaultdict(sample_stats)
    ratios=defaultdict(lambda:defaultdict(blank))
    raw=defaultdict(lambda:defaultdict(blank))
    by_shape=defaultdict(sample_stats)
    count=0
    examples={"ai_overestimates_market_4plus":None,"market_overestimates_ai_under_quarter":None}
    with gzip.open(JOIN,"rt",encoding="utf-8") as f:
        for line in f:
            r=json.loads(line)
            rid=r["race_id"];day=r["race_date"]
            partition=r["subpopulation"]
            assert partition in ("in_sample_2023_jan_oct","forward_2023_nov_dec")
            triples=r["all_trifecta"]
            truth=r["actual_trifecta"]
            assert truth in triples
            key_names=sorted(triples)
            p=[float(triples[t]["p_ai_corrected"]) for t in key_names]
            po=[float(triples[t]["p_ai_original_2023"]) for t in key_names]
            q=[float(triples[t]["market_relative_q"]) for t in key_names]
            raw_inv=[float(triples[t]["market_raw_inverse_odds"]) for t in key_names]
            odds=[float(triples[t]["final_odds"]) for t in key_names]
            assert max(abs(sum(v)-1) for v in (p,po,q))<1e-7
            idx=key_names.index(truth)
            losses=[-math.log(max(v[idx],1e-15)) for v in (po,p,q)]
            briers=[1-2*v[idx]+sum(x*x for x in v) for v in (po,p,q)]
            for lab in ("all",partition):
                st=group[lab]
                st["races"]+=1
                for name,v in zip(("original_logloss","corrected_logloss","market_relative_logloss"),losses):
                    st[name]+=v
                for name,v in zip(("original_brier","corrected_brier","market_relative_brier"),briers):
                    st[name]+=v
                st["corrected_beats_market_races"]+=int(losses[1]<losses[2])
                st["original_beats_market_races"]+=int(losses[0]<losses[2])
                by=st["by_day"][day]
                for k in range(3):by[k]+=losses[k]
                by[3]+=1
            ss=by_shape[(partition,r["line_shape"])]
            ss["races"]+=1
            ss["original_logloss"]+=losses[0]
            ss["corrected_logloss"]+=losses[1]
            ss["market_relative_logloss"]+=losses[2]
            for name,v in zip(("original_brier","corrected_brier","market_relative_brier"),briers):
                ss[name]+=v
            ss["corrected_beats_market_races"]+=int(losses[1]<losses[2])
            ss["original_beats_market_races"]+=int(losses[0]<losses[2])
            bys=ss["by_day"][day]
            for k in range(3):bys[k]+=losses[k]
            bys[3]+=1
            for t,pp,p0,qq,rr,oo in zip(key_names,p,po,q,raw_inv,odds):
                win=t==truth
                ratio=pp/qq
                rawratio=pp*oo
                bin1=next(label for label,low,high in RATIO_BANDS if low<=ratio<high)
                bin2=next(label for label,low,high in RAW_RATIOS if low<=rawratio<high)
                for name in ("all",partition):
                    put(ratios[name][bin1],pp,p0,qq,rr,win)
                    put(raw[name][bin2],pp,p0,qq,rr,win)
                if partition=="forward_2023_nov_dec":
                    if ratio>=4 and examples["ai_overestimates_market_4plus"] is None:
                        examples["ai_overestimates_market_4plus"]={
                           "race_id":rid,"date":day,"trifecta":t,"ai_p_pct":100*pp,
                           "market_q_pct":100*qq,"raw_inverse_odds_pct":100*rr,
                           "final_odds":oo,"actual_win":win}
                    if ratio<.25 and examples["market_overestimates_ai_under_quarter"] is None:
                        examples["market_overestimates_ai_under_quarter"]={
                           "race_id":rid,"date":day,"trifecta":t,"ai_p_pct":100*pp,
                           "market_q_pct":100*qq,"raw_inverse_odds_pct":100*rr,
                           "final_odds":oo,"actual_win":win}
            count+=1
            if count%1000==0:print("SCANNED_ALL_MATCHED_BOOKS",count,flush=True)
    assert count==5226
    report={
       "state":"COMPLETE_2023_MARKET_DISAGREEMENT_OUTCOME_DIAGNOSTIC",
       "no_model_retraining":True,"no_takeout_rate_assumed":True,
       "no_betting_or_roi":True,"market_price_phase":"final",
       "sources":["data/audits/trifecta_market_association_full_matches_2023.jsonl.gz",
                  "data/audits/trifecta_market_association_2023.json"],
       "counts":{"exact_joined_races":count,
                 "forward_nov_dec":group["forward_2023_nov_dec"]["races"],
                 "in_sample_jan_oct":group["in_sample_2023_jan_oct"]["races"]},
       "explanation":"q=(1/O)/sum(1/O) is relative market share; no estimated deduction rate, and NOT a claimed true probability",
       "log_loss_vs_market_normalized_shape":{
          k:result(v) for k,v in group.items()},
       "ratio_bands_ai_prob_divided_by_relative_market_q":{
          k:{lab:score(ratios[k][lab]) for lab,lo,hi in RATIO_BANDS}
          for k in ("all","in_sample_2023_jan_oct","forward_2023_nov_dec")
       },
       "ratio_bands_ai_prob_times_raw_odds_without_any_takeout_assumption":{
          k:{lab:score(raw[k][lab]) for lab,lo,hi in RAW_RATIOS}
          for k in ("all","in_sample_2023_jan_oct","forward_2023_nov_dec")
       },
       "line_shape_log_loss":{
         kind:{sh:result(ss) for (per,sh),ss in by_shape.items() if per==kind and ss["races"]>=15}
         for kind in ("in_sample_2023_jan_oct","forward_2023_nov_dec")
       },
       "examples_forward_2023":examples,
       "important_limits":[
          "2023 Jan-Oct comparison includes in-sample targets for original AI.",
          "2023 Nov-Dec original AI is unused in model fitting, but the corrected AI uses coefficients fit on future 2024 data. Corrected AI 2023 results are retrospective backcast, not chronological OOS.",
          "The 2023 final odds are not purchase-time T-15 odds. Never infer tradability or profit.",
          "Q is relative normalized MARKET SHAPE (no rate assumed); raw 1/O data also preserved separately.",
          "Within each race tickets are mutually exclusive. Log loss is aggregated by race.",
          "This observation is NOT proof of a market inefficiency: the market may be more accurate than independent AI."
       ]
    }
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("2023_DISAGREEMENT_DIAGNOSTIC_COMPLETED",json.dumps({
       "counts":report["counts"],"scores":report["log_loss_vs_market_normalized_shape"],
       "forward_ratio":report["ratio_bands_ai_prob_divided_by_relative_market_q"]["forward_2023_nov_dec"]
    },ensure_ascii=False),flush=True)

if __name__=="__main__":main()
