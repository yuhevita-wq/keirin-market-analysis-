#!/usr/bin/env python3
"""Turn the completed 4.45 MB all-ticket 2023 allocation audit into a concise,
readable diagnostic without changing, refitting, or rerunning the independent AI.
Only pre-registered source categories; report positive/negative and replication.
"""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/"data/audits/independent_trifecta_allocation_errors_2023.json"
OUTPUT=ROOT/"data/audits/independent_trifecta_allocation_errors_2023_digest.json"
j=json.loads(SOURCE.read_text(encoding="utf-8"))
assert j["state"]=="COMPLETED_FROZEN_AI_ALL_2023_TRIFECTA_ALLOCATION_ERROR_AUDIT"
assert j["races_2023_all_eligible"]==23440
assert j["races_NovDec_original_AI_unused"]==4192
assert j["market_odds_or_payout_used"] is False
assert j["AI_retrained"] is False
original=j["models"]["original_2023_AI"]
updated=j["models"]["2024H1_calibrated_backcast"]
periods=("JanOct_in_sample","NovDec_original_AI_holdout","all_2023")
axes=("line_pattern","role1","role2","role3","role12","role23","roles123",
      "score_rank1","score_rank2","score_rank3","score_rank12","score_rank23",
      "score_ranks123","line_size1","line_size2","line_size3",
      "field_size","line_shape","line_shape_x_line_pattern",
      "predicted_probability_band","predicted_trifecta_rank_band")
complex_axes=("line_pattern_x_role1","line_pattern_x_role2","line_pattern_x_role3",
      "line_pattern_x_role12","line_pattern_x_role23",
      "line_pattern_x_rank1","line_pattern_x_rank2","line_pattern_x_rank3",
      "line_pattern_x_role123","line_pattern_x_score_rank12",
      "line_pattern_x_probability_band","line_pattern_x_predicted_rank_band")
def compact(v):
 if not v:return None
 return {k:v[k] for k in ("trifecta_tickets","race_exposures",
  "AI_predicted_event_count","actual_event_count","actual_minus_AI_expected",
  "AI_mean_probability_per_ticket_pct","observed_win_frequency_per_ticket_pct",
  "actual_over_predicted_count","race_event_z","day_cluster_z")}
def rep(model,period,axis,label):
 return model[period].get(axis,{}).get(label)
def direction(x):
 if not x or not x.get("actual_minus_AI_expected"):return "not_established"
 return "AI_underallocated_probability" if x["actual_minus_AI_expected"]>0 else "AI_overallocated_probability"
def direct_pattern(model):
 out={}
 for axis in axes:
  out[axis]={}
  for key,v in model["NovDec_original_AI_holdout"][axis].items():
   if axis in ("line_pattern","role1","role2","role3","score_rank1","score_rank2","score_rank3","predicted_probability_band","predicted_trifecta_rank_band","field_size","line_shape"):
    out[axis][key]={"holdout":compact(v),"train":compact(rep(model,"JanOct_in_sample",axis,key)),"direction":direction(v)}
  if axis in ("role12","role23","roles123","score_rank12","score_rank23","score_ranks123","line_size1","line_size2","line_size3","line_shape_x_line_pattern"):
   rows=[]
   for key,v in model["NovDec_original_AI_holdout"][axis].items():
    if v["AI_predicted_event_count"]<10 or v["race_exposures"]<80:continue
    rows.append((abs(v["race_event_z"] or 0),key,v))
   rows.sort(reverse=True)
   out[axis]={key:{"holdout":compact(v),"train":compact(rep(model,"JanOct_in_sample",axis,key)),
      "direction":direction(v)} for _,key,v in rows[:18]}
 return out

def leaders(model,limit=35):
    entries=[]
    for axis in (*axes,*complex_axes):
      for label,x in model["NovDec_original_AI_holdout"][axis].items():
        if x["AI_predicted_event_count"]<20 or x["race_exposures"]<120 or x["trifecta_tickets"]<500:continue
        train=rep(model,"JanOct_in_sample",axis,label)
        z=x["day_cluster_z"]
        rz=x["race_event_z"]
        if z is None or rz is None:continue
        if not train or train["AI_predicted_event_count"]<70:continue
        entries.append({"axis":axis,"structure":label,"holdout":compact(x),"train":compact(train),
          "error_direction_in_holdout":direction(x),
          "error_direction_also_present_in_training":x["actual_minus_AI_expected"]*train["actual_minus_AI_expected"]>0,
          "absolute_holdout_day_cluster_z":round(abs(z),3)})
    entries.sort(key=lambda v:(-v["absolute_holdout_day_cluster_z"],-abs(v["holdout"]["actual_minus_AI_expected"])))
    return {"considered":len(entries),
       "all_holdout_top":entries[:limit],
       "underallocated_to_these": [x for x in entries if x["error_direction_in_holdout"]=="AI_underallocated_probability"][:limit],
       "overallocated_to_these": [x for x in entries if x["error_direction_in_holdout"]=="AI_overallocated_probability"][:limit],
       "same_direction_both_periods": [x for x in entries if x["error_direction_also_present_in_training"]][:limit]}

def initial_errors(model,limit=30):
   z=model["training_anomalies_checked_against_original_unused_NovDec"]
   out={}
   for field in ("probability_taken_too_much_from","probability_allocated_too_much_to"):
     rows=z[field]
     valid=[{
       "axis":x["axis"],"structure":x["winning_trifecta_structure"],
       "train":compact(x["JanOct_training_diagnostic"]),
       "holdout":compact(x["NovDec_frozen_bucket_check"]),
       "same_error_direction_in_holdout":x["same_direction_in_NovDec"],
       "holdout_predicted_events_at_least_20":x["NovDec_has_at_least_20_predicted_events"]
     } for x in rows]
     out[field]={"top_training":valid[:limit],"reproduced_with_at_least_20_expected": [
       x for x in valid if x["same_error_direction_in_holdout"] and x["holdout_predicted_events_at_least_20"]][:limit]}
   return out

data={
 "source":"data/audits/independent_trifecta_allocation_errors_2023.json",
 "parent_state":j["state"],
 "counts":{"all_2023_races":j["races_2023_all_eligible"],
           "original_AI_training_JanOct":j["races_JanOct_original_AI_training"],
           "original_AI_unseen_NovDec":j["races_NovDec_original_AI_unused"],
           "all_ticket_combinations_per_AI":j["all_2023_trifecta_ticket_combinations_per_model"]},
 "basic_group_definitions":j["line_pattern_definitions"],
 "reading":"Observed minus AI expected > 0 means the AI allocated too LITTLE probability to the outcome group; < 0 means it allocated too MUCH. All groups should sum to full race count within each axis.",
 "odds_used":False,"model_changed":False,
 "primary_original_2023_AI":{
   "groups":direct_pattern(original),
   "largest_holdout_errors":leaders(original,30),
   "anomalies_discovered_in_training_then_checked_in_holdout":initial_errors(original,25)
 },
 "secondary_2024_calibration_backcast_not_chronological":{
   "groups":{axis:direct_pattern(updated)[axis] for axis in ("line_pattern","role1","role2","role3","score_rank1","score_rank2","score_rank3","predicted_probability_band","predicted_trifecta_rank_band")},
   "largest_holdout_errors":leaders(updated,16),
 },
 "interpretation_limitations":[
  "2023 original AI was trained on 2023 Jan-Oct, so training differences are not an independent holdout evaluation.",
  "November-December original 2023 AI holdout comprises 4192 races, not just 391 matched to market odds.",
  "A 2024-H1-fitted calibrated model backcast into 2023 is NOT forward holdout; use original model for genuine chronological diagnosis.",
  "The most extreme top errors are found by scanning many overlapping axes; do not treat rank-order or z as multiple-testing-adjusted causal discoveries.",
  "Probability over/under-allocation is an observed systematic error, not a proven mechanistic cause.",
  "The model independently allocates 1st, 2nd, 3rd using empirical smoothing. Role/line/score at each placement are features; omitted interactions are possible hypotheses.",
  "No external market odds enter this audit; no purchase decisions or ROI.",
  "Archive published line and score point-in-time availability has not been confirmed."
 ]
}
OUTPUT.write_text(json.dumps(data,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
TOPLINE=ROOT/"data/audits/independent_trifecta_allocation_errors_2023_topline.json"
def slim(x):
 if not x:return None
 return {k:x[k] for k in ("AI_predicted_event_count","actual_event_count",
      "actual_minus_AI_expected","actual_over_predicted_count",
      "race_event_z","day_cluster_z")}
def simple_axis(model,per,axis):
 d=model[per][axis]
 return {k:slim(v) for k,v in d.items()
         if (axis in ("line_pattern","role1","role2","role3","score_rank1",
                       "score_rank2","score_rank3","predicted_probability_band",
                       "predicted_trifecta_rank_band")
             or v["AI_predicted_event_count"]>=20)}
def smallleaders(model):
 z=leaders(model,10)
 return [
   {"axis":e["axis"],"label":e["structure"],
    "holdout":slim(e["holdout"]),"train":slim(e["train"]),
    "replicated_sign":e["error_direction_also_present_in_training"]}
   for e in z["all_holdout_top"]]
lite={
 "all_races":23440,"original_AI_training_races":19248,"original_AI_holdout_races":4192,
 "format":"actual minus expected: positive = AI too little; negative = AI too much",
 "model_original_2023":{
    "holdout":{
      k:simple_axis(original,"NovDec_original_AI_holdout",k)
      for k in ("line_pattern","role1","role2","role3",
                "score_rank1","score_rank2","score_rank3",
                "predicted_probability_band","predicted_trifecta_rank_band")},
    "train":{"line_pattern":simple_axis(original,"JanOct_in_sample","line_pattern")},
    "top_holdout_groups":smallleaders(original)
 },
 "2024calibrated_retrofit_2023_not_forward":{
    "holdout_line_pattern":simple_axis(updated,"NovDec_original_AI_holdout","line_pattern"),
    "top_holdout_groups":smallleaders(updated)[:6]
 },
 "interpretation":"Correlation of misallocations with structure does not prove causal reason. 2024 calibration 2023 scores are future-fitted retrospection.",
}
TOPLINE.write_text(json.dumps(lite,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
print("ALLOCATION_ERROR_DIAGNOSTIC_DIGEST",json.dumps({
 "source_bytes":SOURCE.stat().st_size,"digest_bytes":OUTPUT.stat().st_size,
 "all_races":data["counts"]["all_2023_races"],
 "original_AI_holdout_line_patterns":{
   key:{"pred":val["holdout"]["AI_predicted_event_count"],
        "actual":val["holdout"]["actual_event_count"],
        "delta":val["holdout"]["actual_minus_AI_expected"],
        "day_z":val["holdout"]["day_cluster_z"]}
   for key,val in data["primary_original_2023_AI"]["groups"]["line_pattern"].items()},
 "top_holdout_error":data["primary_original_2023_AI"]["largest_holdout_errors"]["all_holdout_top"][:5]
},ensure_ascii=False),flush=True)
