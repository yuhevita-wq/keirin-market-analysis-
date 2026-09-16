#!/usr/bin/env python3
from __future__ import annotations

import csv, json, math, importlib.util
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path
from statistics import mean
import numpy as np
from sklearn.linear_model import LogisticRegression

spec=importlib.util.spec_from_file_location("v18","scripts/keirin_shogi_v18_ensemble_walkforward.py")
v18=importlib.util.module_from_spec(spec); spec.loader.exec_module(v18)

BASES=[
    Path("data/2025/s_class_f1_all_parts/2025_q1"),
    Path("data/2025/s_class_f1_all_parts/2025_q2"),
    Path("data/2025/s_class_f1_all_parts/2025_q3"),
    Path("data/2025/s_class_f1_all_parts/2025_q4"),
    Path("data/2026_h1/s_class_f1_all"),
]
OUT_DIR=Path("results/keirin_shogi/v19_targeted_participation")
OUT_DIR.mkdir(parents=True,exist_ok=True)

START_TEST=date(2025,6,30)
END_TEST=date(2026,6,28)

# 「全レースでそこそこ」ではなく「入るレースを絞って当てる」ための事前固定条件。
ACCEPT={
    "participation_min":0.15,
    "participation_max":0.45,
    "participant_top1_min":0.40,
    "participant_candidate_capture_min":0.70,
    "participant_avg_candidates_max":2.10,
    "capture_advantage_vs_skipped_min":0.08,
    "consecutive_weeks":2,
}

def load_all(name):
    out=[]
    for base in BASES:
        p=base/name
        if p.exists():
            with p.open("r",encoding="utf-8-sig",newline="") as f:
                out.extend(csv.DictReader(f))
    return out

def ino(v):
    try:return int(float(v))
    except:return 0

def target_races(races,start,end):
    ss=start.isoformat(); ee=end.isoformat()
    return {r["race_id"]:r for r in races
            if ss<=r.get("race_date","")<=ee
            and ino(r.get("entry_count"))==7
            and r.get("meeting_grade")=="F1"
            and "Ｓ級" in r.get("race_type","")}

def make_races(ids,eb,rb):
    return v18.v17.make_races(ids,eb,rb)

def candidate_policy_from_cal(train_ds, cal_ds):
    models=v18.fit_ensemble(train_ds)
    cache=[]
    for _,rr,w in cal_ds:
        p,t=v18.predict(rr,models);cache.append((p,t,w))
    grid=[]
    for agree in (2,3):
      for p1 in (.30,.34,.38,.42):
       for ratio in (1.3,1.5,1.8,2.1):
        for cum2 in (.48,.54,.60,.66):
            pol={"agree_min":agree,"p1":p1,"ratio12":ratio,"cum2":cum2}
            hits=0;counts=[]
            for p,t,w in cache:
                c=v18.choose(p,t,pol);hits+=int(w in c);counts.append(len(c))
            cap=hits/len(cache);avg=mean(counts);three=sum(k==3 for k in counts)/len(counts)
            # 3人並べるだけの救済を抑える
            obj=cap-0.18*(avg-1.0)-0.10*three
            grid.append({**pol,"capture":cap,"avg_candidates":avg,"three_share":three,"objective":obj})
    grid.sort(key=lambda x:(x["objective"],x["capture"],-x["avg_candidates"]),reverse=True)
    return grid[0]

def entropy(pred):
    return -sum(x["prob"]*math.log(max(x["prob"],1e-12)) for x in pred)

def pred_features(pred,tops,cands):
    top_probs=[x["prob"] for x in pred]
    top1_no=pred[0]["no"]
    agree_top1=sum(1 for x in tops if x==top1_no)
    selected=set(cands)
    selected_mass=sum(x["prob"] for x in pred if x["no"] in selected)
    model_p1=[]
    for x in pred:
        if x["no"]==top1_no:
            model_p1=x.get("model_probs",[])
            break
    p1_sd=float(np.std(model_p1)) if model_p1 else 0.0
    return np.asarray([
        top_probs[0],
        top_probs[0]-top_probs[1],
        top_probs[0]+top_probs[1],
        selected_mass,
        entropy(pred),
        agree_top1/3.0,
        len(cands)/3.0,
        p1_sd,
    ],dtype=float)

def build_meta_rows(ds,core_models,candidate_policy):
    rows=[]
    for rid,rr,winner in ds:
        pred,tops=v18.predict(rr,core_models)
        cands=v18.choose(pred,tops,candidate_policy)
        rows.append({
            "race_id":rid,"winner":winner,"pred":pred,"tops":tops,"cands":cands,
            "x":pred_features(pred,tops,cands),
            "candidate_hit":int(winner in cands),
            "top1_hit":int(pred[0]["no"]==winner),
            "candidate_count":len(cands),
        })
    return rows

def fit_selector(meta_rows):
    X=np.vstack([r["x"] for r in meta_rows])
    y=np.asarray([r["candidate_hit"] for r in meta_rows],dtype=int)
    if len(set(y.tolist()))<2:
        return None
    clf=LogisticRegression(C=0.6,class_weight="balanced",max_iter=500,random_state=19)
    clf.fit(X,y)
    return clf

def selector_scores(meta_rows,selector):
    if selector is None:
        return np.asarray([0.5]*len(meta_rows),dtype=float)
    X=np.vstack([r["x"] for r in meta_rows])
    return selector.predict_proba(X)[:,1]

def group_metrics(rows):
    if not rows:
        return {"races":0,"top1_hit_rate":0.0,"candidate_capture_rate":0.0,"avg_candidates":0.0}
    return {
        "races":len(rows),
        "top1_hit_rate":sum(r["top1_hit"] for r in rows)/len(rows),
        "candidate_capture_rate":sum(r["candidate_hit"] for r in rows)/len(rows),
        "avg_candidates":mean([r["candidate_count"] for r in rows]),
    }

def learn_threshold(val_rows,selector):
    scores=selector_scores(val_rows,selector)
    order=np.argsort(scores)
    # 閾値はvalidation上の参加率15〜45%だけで探索。結果を見て全レースへ広げない。
    candidates=[]
    for frac in (.15,.20,.25,.30,.35,.40,.45):
        nsel=max(1,int(round(len(val_rows)*frac)))
        cutoff=float(scores[order[-nsel]])
        part=[r for r,s in zip(val_rows,scores) if s>=cutoff]
        skip=[r for r,s in zip(val_rows,scores) if s<cutoff]
        pm=group_metrics(part); sm=group_metrics(skip)
        actual_frac=len(part)/len(val_rows)
        advantage=pm["candidate_capture_rate"]-sm["candidate_capture_rate"]
        obj=(pm["candidate_capture_rate"]
             +0.35*pm["top1_hit_rate"]
             +0.30*advantage
             -0.12*max(0.0,pm["avg_candidates"]-1.0)
             -0.10*abs(actual_frac-0.30))
        candidates.append({
            "threshold":cutoff,"validation_participation_rate":actual_frac,
            "participant":pm,"skipped":sm,"capture_advantage":advantage,"objective":obj
        })
    candidates.sort(key=lambda x:(x["objective"],x["participant"]["candidate_capture_rate"],-x["validation_participation_rate"]),reverse=True)
    return candidates[0]

def acceptance(part_rate,pm,sm):
    return (
        ACCEPT["participation_min"]<=part_rate<=ACCEPT["participation_max"]
        and pm["top1_hit_rate"]>=ACCEPT["participant_top1_min"]
        and pm["candidate_capture_rate"]>=ACCEPT["participant_candidate_capture_min"]
        and pm["avg_candidates"]<=ACCEPT["participant_avg_candidates_max"]
        and (pm["candidate_capture_rate"]-sm["candidate_capture_rate"])>=ACCEPT["capture_advantage_vs_skipped_min"]
    )

def main():
    races=load_all("races.csv");entries=load_all("entries.csv");results=load_all("results.csv")
    eb=defaultdict(list);rb=defaultdict(list)
    for x in entries: eb[x["race_id"]].append(x)
    for x in results: rb[x["race_id"]].append(x)

    weekly=[];streak=0;checkpoint=None
    cur=START_TEST
    while cur<=END_TEST:
        te=cur+timedelta(days=6)

        # 13週間を完全に過去だけで分割:
        # 8週 core学習 -> 2週 candidate policy校正 -> 2週 selector学習 -> 1週 threshold校正 -> 次週本番
        h0=cur-timedelta(days=91)
        core_a_end=cur-timedelta(days=36)      # -13〜-6週
        policy_start=cur-timedelta(days=35)
        policy_end=cur-timedelta(days=22)      # -5〜-4週
        selector_train_start=cur-timedelta(days=21)
        selector_train_end=cur-timedelta(days=8) # -3〜-2週
        threshold_start=cur-timedelta(days=7)
        threshold_end=cur-timedelta(days=1)    # -1週

        core_a=make_races(target_races(races,h0,core_a_end).keys(),eb,rb)
        policy_cal=make_races(target_races(races,policy_start,policy_end).keys(),eb,rb)
        if min(len(core_a),len(policy_cal))==0:
            cur+=timedelta(days=7);continue

        cand_policy=candidate_policy_from_cal(core_a,policy_cal)

        core_b=make_races(target_races(races,h0,policy_end).keys(),eb,rb)
        sel_train_ds=make_races(target_races(races,selector_train_start,selector_train_end).keys(),eb,rb)
        core_c=make_races(target_races(races,h0,selector_train_end).keys(),eb,rb)
        threshold_ds=make_races(target_races(races,threshold_start,threshold_end).keys(),eb,rb)
        final_train=make_races(target_races(races,h0,threshold_end).keys(),eb,rb)
        test_ds=make_races(target_races(races,cur,te).keys(),eb,rb)
        if min(len(core_b),len(sel_train_ds),len(core_c),len(threshold_ds),len(final_train),len(test_ds))==0:
            cur+=timedelta(days=7);continue

        # selector学習用予測はselector対象週より前だけで作る
        core_b_models=v18.fit_ensemble(core_b)
        sel_train_rows=build_meta_rows(sel_train_ds,core_b_models,cand_policy)
        selector=fit_selector(sel_train_rows)

        # threshold校正も未来混入なし
        core_c_models=v18.fit_ensemble(core_c)
        threshold_rows=build_meta_rows(threshold_ds,core_c_models,cand_policy)
        threshold_info=learn_threshold(threshold_rows,selector)
        threshold=threshold_info["threshold"]

        # 本番週。coreのみ直前まで再学習。selectorの意味は固定。
        final_models=v18.fit_ensemble(final_train)
        test_rows=build_meta_rows(test_ds,final_models,cand_policy)
        scores=selector_scores(test_rows,selector)
        for r,s in zip(test_rows,scores):
            r["selector_score"]=float(s)
            r["participate"]=bool(s>=threshold)

        part=[r for r in test_rows if r["participate"]]
        skip=[r for r in test_rows if not r["participate"]]
        pm=group_metrics(part);sm=group_metrics(skip);allm=group_metrics(test_rows)
        pr=len(part)/len(test_rows)
        adv=pm["candidate_capture_rate"]-sm["candidate_capture_rate"]
        ok=acceptance(pr,pm,sm)
        streak=streak+1 if ok else 0

        rec={
            "test_period":{"start":cur.isoformat(),"end":te.isoformat()},
            "history_period":{"start":h0.isoformat(),"end":threshold_end.isoformat()},
            "candidate_policy":cand_policy,
            "selector_training_races":len(sel_train_rows),
            "threshold_validation":threshold_info,
            "selector_threshold":threshold,
            "test_races":len(test_rows),
            "participation_rate":pr,
            "all_races":allm,
            "participant":pm,
            "skipped":sm,
            "capture_advantage_vs_skipped":adv,
            "passes_acceptance":ok,
            "acceptance_streak":streak,
        }
        weekly.append(rec)

        wk=OUT_DIR/f"{cur.isoformat()}_{te.isoformat()}";wk.mkdir(parents=True,exist_ok=True)
        (wk/"summary.json").write_text(json.dumps(rec,ensure_ascii=False,indent=2),encoding="utf-8")
        detail=[{
            "race_id":r["race_id"],"winner":r["winner"],"participate":r["participate"],
            "selector_score":r["selector_score"],"candidates":r["cands"],
            "candidate_hit":r["candidate_hit"],"top1_hit":r["top1_hit"],
            "candidate_count":r["candidate_count"],"ranking":r["pred"],"model_top1s":r["tops"]
        } for r in test_rows]
        (wk/"detail.json").write_text(json.dumps(detail,ensure_ascii=False,indent=2),encoding="utf-8")

        if streak>=ACCEPT["consecutive_weeks"]:
            checkpoint={"period":rec["test_period"],"week_index":len(weekly)}
            break

        cur+=timedelta(days=7)

    final={
        "algorithm":"keirin_shogi_v19_targeted_participation",
        "goal":"全レース参加をやめ、事前情報だけで狙うレースを絞って1着候補を当てる",
        "acceptance_rule":ACCEPT,
        "weeks_tested":len(weekly),
        "checkpoint":checkpoint,
        "weekly":weekly,
        "method":"3モデル非線形1着コア + 可変1〜3人候補 + 参加selector。毎週13週間の過去を時系列分割し、core学習、候補数校正、selector学習、参加閾値校正を別期間で行う。参加率15〜45%に制約し、参加側が見送り側より明確に当たることまで要求。",
        "leakage_guard":"各予測週より未来の結果は一切使用しない。selector/thresholdも本番週より前だけで学習。オッズ不使用。車番は特徴量不使用。"
    }
    (OUT_DIR/"summary.json").write_text(json.dumps(final,ensure_ascii=False,indent=2),encoding="utf-8")
    (OUT_DIR/"README.md").write_text(
        "# v19 狙って入る参加selector\n\n"
        "全レース予測から卒業し、事前情報だけで参加レースを15〜45%へ絞る。\n"
        "合格は単発週ではなく2週連続。\n",
        encoding="utf-8"
    )
    print(json.dumps(final,ensure_ascii=False,indent=2))

if __name__=="__main__":
    main()
