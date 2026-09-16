#!/usr/bin/env python3
from __future__ import annotations
import json, math
from pathlib import Path
from statistics import mean
import numpy as np
from sklearn.linear_model import LogisticRegression

SRC=Path("results/keirin_shogi/v19_targeted_participation")
OUT=Path("results/keirin_shogi/v22_rolling_selector")
OUT.mkdir(parents=True,exist_ok=True)

TEST_START,TEST_END="2025-12-29","2026-06-28"
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
        if not d.is_dir():continue
        p=d/"detail.json"
        if not p.exists():continue
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
    return {"races":len(rows),
      "top1_hit_rate":sum(r["top1_hit"] for r in rows)/len(rows),
      "candidate_capture_rate":sum(r["candidate_hit"] for r in rows)/len(rows),
      "avg_candidates":mean([r["candidate_count"] for r in rows])}

def fit(rows):
    X=np.vstack([r["x"] for r in rows]);y=np.asarray([r["candidate_hit"] for r in rows],int)
    clf=LogisticRegression(C=0.45,class_weight="balanced",max_iter=600,random_state=22)
    clf.fit(X,y);return clf

def score(rows,clf):
    return clf.predict_proba(np.vstack([r["x"] for r in rows]))[:,1]

def choose_threshold(cal,clf):
    sc=score(cal,clf)
    configs=[]
    eligible=[i for i,r in enumerate(cal) if r["candidate_count"]<=2]
    for frac in (.12,.15,.18,.20,.22,.25,.28,.30,.35,.40):
        n=max(1,min(len(eligible),int(round(len(cal)*frac))))
        vals=sorted([float(sc[i]) for i in eligible])
        thr=vals[-n]
        part=[r for i,r in enumerate(cal) if i in eligible and sc[i]>=thr]
        skip=[r for i,r in enumerate(cal) if not(i in eligible and sc[i]>=thr)]
        pm=gm(part);sm=gm(skip);pr=len(part)/len(cal);adv=pm["candidate_capture_rate"]-sm["candidate_capture_rate"]
        obj=pm["candidate_capture_rate"]+0.35*pm["top1_hit_rate"]+0.35*adv-0.08*abs(pr-.22)
        configs.append({"threshold":thr,"cal_participation_rate":pr,"participant":pm,"skipped":sm,"advantage":adv,"objective":obj})
    configs.sort(key=lambda x:(x["objective"],x["participant"]["candidate_capture_rate"],-x["cal_participation_rate"]),reverse=True)
    return configs[0]

def main():
    rows=load_rows(); weeks=sorted(set(r["week"] for r in rows)); by={w:[r for r in rows if r["week"]==w] for w in weeks}
    test_weeks=[w for w in weeks if TEST_START<=w<=TEST_END]
    weekly=[];streak=0;checkpoint=None;part_all=[];skip_all=[]
    for w in test_weeks:
        prior=[x for x in weeks if x<w]
        train_weeks=prior[-12:-4]
        cal_weeks=prior[-4:]
        if len(train_weeks)<8 or len(cal_weeks)<4:continue
        tr=[r for x in train_weeks for r in by[x]]
        cal=[r for x in cal_weeks for r in by[x]]
        te=by[w]
        clf=fit(tr);cfg=choose_threshold(cal,clf);sc=score(te,clf)
        part=[];skip=[]
        for r,s in zip(te,sc):
            enter=(r["candidate_count"]<=2 and s>=cfg["threshold"])
            (part if enter else skip).append(r)
        part_all+=part;skip_all+=skip
        pr=len(part)/len(te);pm=gm(part);sm=gm(skip);adv=pm["candidate_capture_rate"]-sm["candidate_capture_rate"]
        ok=(ACCEPT["participation_min"]<=pr<=ACCEPT["participation_max"] and
            pm["top1_hit_rate"]>=ACCEPT["participant_top1_min"] and
            pm["candidate_capture_rate"]>=ACCEPT["participant_candidate_capture_min"] and
            adv>=ACCEPT["capture_advantage_vs_skipped_min"])
        streak=streak+1 if ok else 0
        rec={"week":w,"train_weeks":train_weeks,"calibration_weeks":cal_weeks,
             "calibration_choice":cfg,"races":len(te),"participation_rate":pr,
             "participant":pm,"skipped":sm,"capture_advantage_vs_skipped":adv,
             "passes_acceptance":ok,"streak":streak}
        weekly.append(rec)
        if checkpoint is None and streak>=2:checkpoint={"week":w,"week_index":len(weekly)}
    ppm=gm(part_all);ssm=gm(skip_all)
    final={
      "algorithm":"keirin_shogi_v22_rolling_selector",
      "source":"v19完全OOS予測",
      "test_period":[TEST_START,TEST_END],
      "acceptance_rule":ACCEPT,
      "test_weeks":len(weekly),
      "checkpoint":checkpoint,
      "pooled_test":{"participation_rate":len(part_all)/(len(part_all)+len(skip_all)),
        "participant":ppm,"skipped":ssm,
        "capture_advantage_vs_skipped":ppm["candidate_capture_rate"]-ssm["candidate_capture_rate"]},
      "weekly":weekly,
      "method":"各本番週の前12週だけを使用。古い8週でselector学習、直近4週で参加thresholdを校正。候補3人レースは参加禁止。毎週完全walk-forwardで再学習。",
      "leakage_guard":"本番週以降の結果はselector/thresholdに未使用。"
    }
    (OUT/"summary.json").write_text(json.dumps(final,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(final,ensure_ascii=False,indent=2))

if __name__=="__main__":main()
