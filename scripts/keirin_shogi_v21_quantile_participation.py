#!/usr/bin/env python3
from __future__ import annotations
import json, math
from pathlib import Path
from statistics import mean
import numpy as np
from sklearn.linear_model import LogisticRegression

SRC=Path("results/keirin_shogi/v19_targeted_participation")
OUT=Path("results/keirin_shogi/v21_quantile_participation")
OUT.mkdir(parents=True,exist_ok=True)

TRAIN_START,TRAIN_END="2025-06-30","2025-10-26"
CAL_START,CAL_END="2025-10-27","2025-12-28"
TEST_START,TEST_END="2025-12-29","2026-06-28"
LOOKBACK_WEEKS=4

ACCEPT={
  "participation_min":0.12,
  "participation_max":0.40,
  "participant_top1_min":0.40,
  "participant_candidate_capture_min":0.70,
  "capture_advantage_vs_skipped_min":0.08,
  "consecutive_weeks":2
}

def entropy(pred):
    return -sum(x["prob"]*math.log(max(x["prob"],1e-12)) for x in pred)

def feat(row):
    pred=row["ranking"]; tops=row["model_top1s"]; cands=row["candidates"]
    p=[x["prob"] for x in pred]
    top1=pred[0]["no"]
    agree=sum(1 for x in tops if x==top1)
    mass=sum(x["prob"] for x in pred if x["no"] in set(cands))
    mp=[]
    for x in pred:
        if x["no"]==top1:
            mp=x.get("model_probs",[]);break
    return np.asarray([
        p[0],p[0]-p[1],p[0]+p[1],mass,entropy(pred),agree/3.0,len(cands)/3.0,
        float(np.std(mp)) if mp else 0.0
    ],float)

def load_rows():
    rows=[]
    for d in sorted(SRC.iterdir()):
        if not d.is_dir(): continue
        p=d/"detail.json"
        if not p.exists(): continue
        week=d.name[:10]
        for r in json.loads(p.read_text(encoding="utf-8")):
            rows.append({
                "week":week,"race_id":r["race_id"],"x":feat(r),
                "top1_hit":int(r["top1_hit"]),"candidate_hit":int(r["candidate_hit"]),
                "candidate_count":int(r["candidate_count"])
            })
    return rows

def gm(rows):
    if not rows:return {"races":0,"top1_hit_rate":0.0,"candidate_capture_rate":0.0,"avg_candidates":0.0}
    return {
      "races":len(rows),
      "top1_hit_rate":sum(r["top1_hit"] for r in rows)/len(rows),
      "candidate_capture_rate":sum(r["candidate_hit"] for r in rows)/len(rows),
      "avg_candidates":mean([r["candidate_count"] for r in rows])
    }

def fit_selector(rows):
    X=np.vstack([r["x"] for r in rows]);y=np.asarray([r["candidate_hit"] for r in rows],int)
    clf=LogisticRegression(C=0.35,class_weight="balanced",max_iter=800,random_state=21)
    clf.fit(X,y); return clf

def add_scores(rows,clf):
    scores=clf.predict_proba(np.vstack([r["x"] for r in rows]))[:,1]
    out=[]
    for r,s in zip(rows,scores):
        q=dict(r);q["selector_score"]=float(s);out.append(q)
    return out

def weekly_groups(rows):
    d={}
    for r in rows:d.setdefault(r["week"],[]).append(r)
    return d

def threshold_from_history(history_rows,target_fraction,maxk):
    eligible=[r["selector_score"] for r in history_rows if r["candidate_count"]<=maxk]
    if not eligible:return 1.0
    n=max(1,int(round(len(eligible)*target_fraction)))
    return float(sorted(eligible)[-n])

def evaluate_period(rows,calibrate_targets=False):
    weeks=sorted(set(r["week"] for r in rows))
    by=weekly_groups(rows)
    all_weeks=weekly_groups(ALL_SCORED)
    configs=[]
    targets=[.12,.15,.18,.20,.22,.25,.28,.30]
    maxks=[1,2]
    for maxk in maxks:
      for target in targets:
        recs=[]
        for w in weeks:
            # only prior 4 weeks, score-distribution only. no current/future labels in threshold.
            prior=[pw for pw in sorted(all_weeks) if pw < w][-LOOKBACK_WEEKS:]
            hist=[r for pw in prior for r in all_weeks[pw]]
            thr=threshold_from_history(hist,target,maxk)
            wr=by[w]
            part=[r for r in wr if r["candidate_count"]<=maxk and r["selector_score"]>=thr]
            skip=[r for r in wr if not (r["candidate_count"]<=maxk and r["selector_score"]>=thr)]
            pm=gm(part);sm=gm(skip);pr=len(part)/len(wr)
            recs.append((pr,pm,sm))
        if not recs:continue
        pool_part=[];pool_skip=[]
        for w in weeks:
            prior=[pw for pw in sorted(all_weeks) if pw < w][-LOOKBACK_WEEKS:]
            hist=[r for pw in prior for r in all_weeks[pw]]
            thr=threshold_from_history(hist,target,maxk)
            for r in by[w]:
                (pool_part if (r["candidate_count"]<=maxk and r["selector_score"]>=thr) else pool_skip).append(r)
        ppm=gm(pool_part);ssm=gm(pool_skip);pr=len(pool_part)/(len(pool_part)+len(pool_skip))
        adv=ppm["candidate_capture_rate"]-ssm["candidate_capture_rate"]
        # prefer real selectivity and capture, penalize too tiny/too broad participation
        obj=ppm["candidate_capture_rate"]+0.35*ppm["top1_hit_rate"]+0.35*adv-0.10*abs(pr-.20)
        configs.append({"max_candidate_count":maxk,"target_fraction":target,"participation_rate":pr,
                        "participant":ppm,"skipped":ssm,"capture_advantage":adv,"objective":obj})
    configs.sort(key=lambda x:(x["objective"],x["participant"]["candidate_capture_rate"],-x["participation_rate"]),reverse=True)
    return configs[0],configs

def apply_test(rows,cfg):
    weeks=sorted(set(r["week"] for r in rows));by=weekly_groups(rows);all_by=weekly_groups(ALL_SCORED)
    weekly=[];streak=0;checkpoint=None;part_all=[];skip_all=[]
    for w in weeks:
        prior=[pw for pw in sorted(all_by) if pw < w][-LOOKBACK_WEEKS:]
        hist=[r for pw in prior for r in all_by[pw]]
        thr=threshold_from_history(hist,cfg["target_fraction"],cfg["max_candidate_count"])
        part=[r for r in by[w] if r["candidate_count"]<=cfg["max_candidate_count"] and r["selector_score"]>=thr]
        skip=[r for r in by[w] if not (r["candidate_count"]<=cfg["max_candidate_count"] and r["selector_score"]>=thr)]
        part_all+=part;skip_all+=skip
        pr=len(part)/len(by[w]);pm=gm(part);sm=gm(skip);adv=pm["candidate_capture_rate"]-sm["candidate_capture_rate"]
        ok=(ACCEPT["participation_min"]<=pr<=ACCEPT["participation_max"] and
            pm["top1_hit_rate"]>=ACCEPT["participant_top1_min"] and
            pm["candidate_capture_rate"]>=ACCEPT["participant_candidate_capture_min"] and
            adv>=ACCEPT["capture_advantage_vs_skipped_min"])
        streak=streak+1 if ok else 0
        rec={"week":w,"threshold":thr,"races":len(by[w]),"participation_rate":pr,"participant":pm,"skipped":sm,
             "capture_advantage_vs_skipped":adv,"passes_acceptance":ok,"streak":streak}
        weekly.append(rec)
        if checkpoint is None and streak>=ACCEPT["consecutive_weeks"]:
            checkpoint={"week":w,"week_index":len(weekly)}
    ppm=gm(part_all);ssm=gm(skip_all)
    pooled={"participation_rate":len(part_all)/(len(part_all)+len(skip_all)),"participant":ppm,"skipped":ssm,
            "capture_advantage_vs_skipped":ppm["candidate_capture_rate"]-ssm["candidate_capture_rate"]}
    return weekly,checkpoint,pooled

rows=load_rows()
train=[r for r in rows if TRAIN_START<=r["week"]<=TRAIN_END]
cal=[r for r in rows if CAL_START<=r["week"]<=CAL_END]
test=[r for r in rows if TEST_START<=r["week"]<=TEST_END]
clf=fit_selector(train)
ALL_SCORED=add_scores(rows,clf)
cal_sc=[r for r in ALL_SCORED if CAL_START<=r["week"]<=CAL_END]
test_sc=[r for r in ALL_SCORED if TEST_START<=r["week"]<=TEST_END]
cfg,grid=evaluate_period(cal_sc)
weekly,checkpoint,pooled=apply_test(test_sc,cfg)

final={
 "algorithm":"keirin_shogi_v21_quantile_participation",
 "source":"v19完全OOS予測",
 "selector_train_period":[TRAIN_START,TRAIN_END],
 "calibration_period":[CAL_START,CAL_END],
 "test_period":[TEST_START,TEST_END],
 "lookback_weeks_for_unlabeled_score_quantile":LOOKBACK_WEEKS,
 "frozen_policy":{"max_candidate_count":cfg["max_candidate_count"],"target_fraction":cfg["target_fraction"]},
 "calibration_result":cfg,
 "acceptance_rule":ACCEPT,
 "test_weeks":len(weekly),
 "checkpoint":checkpoint,
 "pooled_test":pooled,
 "weekly":weekly,
 "method":"参加selector係数は固定。絶対thresholdの経時ドリフトだけを、直前4週間のselector score分布（結果ラベル不使用）の分位点で補正。候補数上限と狙う比率は2025年calibrationで固定。",
 "leakage_guard":"2026テスト期間の結果ラベルはselector・target fraction・候補数上限の決定に未使用。週ごとのthreshold更新には過去4週のscore分布だけを使用。"
}
(OUT/"summary.json").write_text(json.dumps(final,ensure_ascii=False,indent=2),encoding="utf-8")
print(json.dumps(final,ensure_ascii=False,indent=2))
