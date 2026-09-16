#!/usr/bin/env python3
from __future__ import annotations
import json, math
from pathlib import Path
from statistics import mean
import numpy as np
from sklearn.linear_model import LogisticRegression

SRC=Path("results/keirin_shogi/v19_targeted_participation")
OUT=Path("results/keirin_shogi/v20_stable_participation")
OUT.mkdir(parents=True,exist_ok=True)

TRAIN_START,TRAIN_END="2025-06-30","2025-10-26"
CAL_START,CAL_END="2025-10-27","2025-12-28"
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
        start=d.name[:10]
        for r in json.loads(p.read_text(encoding="utf-8")):
            rows.append({
                "week":start,"race_id":r["race_id"],"x":feat(r),
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

def main():
    rows=load_rows()
    tr=[r for r in rows if TRAIN_START<=r["week"]<=TRAIN_END]
    cal=[r for r in rows if CAL_START<=r["week"]<=CAL_END]
    test=[r for r in rows if TEST_START<=r["week"]<=TEST_END]

    X=np.vstack([r["x"] for r in tr]);y=np.asarray([r["candidate_hit"] for r in tr],int)
    clf=LogisticRegression(C=0.35,class_weight="balanced",max_iter=800,random_state=20)
    clf.fit(X,y)

    cal_scores=clf.predict_proba(np.vstack([r["x"] for r in cal]))[:,1]
    configs=[]
    for maxk in (1,2):
      for agree_min in (1,2,3):
        eligible=[]
        for i,r in enumerate(cal):
            agree=int(round(r["x"][5]*3))
            if r["candidate_count"]<=maxk and agree>=agree_min:
                eligible.append(i)
        if not eligible:continue
        es=np.asarray([cal_scores[i] for i in eligible])
        for frac in (.12,.16,.20,.24,.28,.32,.36,.40):
            n=max(1,int(round(len(cal)*frac)))
            n=min(n,len(eligible))
            cutoff=float(np.sort(es)[-n])
            part=[r for i,r in enumerate(cal) if i in eligible and cal_scores[i]>=cutoff]
            skip=[r for i,r in enumerate(cal) if not (i in eligible and cal_scores[i]>=cutoff)]
            pr=len(part)/len(cal);pm=gm(part);sm=gm(skip)
            adv=pm["candidate_capture_rate"]-sm["candidate_capture_rate"]
            obj=pm["candidate_capture_rate"]+0.40*pm["top1_hit_rate"]+0.35*adv-0.08*abs(pr-.25)
            configs.append({
              "max_candidate_count":maxk,"agree_min":agree_min,"threshold":cutoff,
              "cal_participation_rate":pr,"participant":pm,"skipped":sm,
              "capture_advantage":adv,"objective":obj
            })
    configs.sort(key=lambda x:(x["objective"],x["participant"]["candidate_capture_rate"],-x["cal_participation_rate"]),reverse=True)
    cfg=configs[0]

    def apply(rows):
        scores=clf.predict_proba(np.vstack([r["x"] for r in rows]))[:,1] if rows else []
        out=[]
        for r,s in zip(rows,scores):
            agree=int(round(r["x"][5]*3))
            enter=(r["candidate_count"]<=cfg["max_candidate_count"] and agree>=cfg["agree_min"] and s>=cfg["threshold"])
            rr=dict(r);rr["selector_score"]=float(s);rr["participate"]=enter;out.append(rr)
        return out

    test2=apply(test)
    weeks=sorted(set(r["week"] for r in test2))
    weekly=[];streak=0;checkpoint=None
    for w in weeks:
        wr=[r for r in test2 if r["week"]==w]
        part=[r for r in wr if r["participate"]];skip=[r for r in wr if not r["participate"]]
        pr=len(part)/len(wr);pm=gm(part);sm=gm(skip);adv=pm["candidate_capture_rate"]-sm["candidate_capture_rate"]
        ok=(ACCEPT["participation_min"]<=pr<=ACCEPT["participation_max"] and
            pm["top1_hit_rate"]>=ACCEPT["participant_top1_min"] and
            pm["candidate_capture_rate"]>=ACCEPT["participant_candidate_capture_min"] and
            adv>=ACCEPT["capture_advantage_vs_skipped_min"])
        streak=streak+1 if ok else 0
        rec={"week":w,"races":len(wr),"participation_rate":pr,"participant":pm,"skipped":sm,
             "capture_advantage_vs_skipped":adv,"passes_acceptance":ok,"streak":streak}
        weekly.append(rec)
        if checkpoint is None and streak>=ACCEPT["consecutive_weeks"]:
            checkpoint={"week":w,"week_index":len(weekly)}

    part_all=[r for r in test2 if r["participate"]];skip_all=[r for r in test2 if not r["participate"]]
    final={
      "algorithm":"keirin_shogi_v20_stable_participation",
      "source":"v19の完全OOS予測だけを使用",
      "train_period":[TRAIN_START,TRAIN_END],
      "calibration_period":[CAL_START,CAL_END],
      "test_period":[TEST_START,TEST_END],
      "frozen_gate":cfg,
      "acceptance_rule":ACCEPT,
      "test_weeks":len(weekly),
      "checkpoint":checkpoint,
      "pooled_test":{
        "participation_rate":len(part_all)/len(test2),
        "participant":gm(part_all),"skipped":gm(skip_all),
        "capture_advantage_vs_skipped":gm(part_all)["candidate_capture_rate"]-gm(skip_all)["candidate_capture_rate"]
      },
      "weekly":weekly,
      "method":"v19の週ごとに揺れるselectorを廃止。2025前半〜秋のOOS結果でselector学習、2025秋〜年末で参加閾値と候補数上限を固定。その後2026年前半は一切再調整せず検証。",
      "leakage_guard":"2026テスト期間の結果はgate学習・閾値・候補数上限の決定に未使用。"
    }
    (OUT/"summary.json").write_text(json.dumps(final,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(final,ensure_ascii=False,indent=2))

if __name__=="__main__":main()
