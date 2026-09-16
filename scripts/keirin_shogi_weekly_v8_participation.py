#!/usr/bin/env python3
from __future__ import annotations

import csv, json, math, importlib.util
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev

BASE = Path("data/2024/s_class_f1_all_parts/2024_q1")
TRAIN_START, TRAIN_END = "2024-02-26", "2024-03-03"
TEST_START, TEST_END = "2024-03-04", "2024-03-10"
PRIOR_DETAIL = Path("results/keirin_shogi/v7_conditional_second/2024-02-26_2024-03-03/detail.json")
OUT_DIR = Path("results/keirin_shogi/v8_participation/2024-03-04_2024-03-10")
OUT_DIR.mkdir(parents=True, exist_ok=True)

spec = importlib.util.spec_from_file_location("v7", "scripts/keirin_shogi_weekly_v7_conditional_second.py")
v7 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v7)

def load_csv(name):
    with (BASE/name).open("r",encoding="utf-8-sig",newline="") as f:
        return list(csv.DictReader(f))

def ino(v):
    try:return int(float(v))
    except:return 0

def score_stats(scored, boundary):
    vals=sorted([float(x["score"]) for x in scored], reverse=True)
    if not vals:return {"sd":0.0,"gap":0.0}
    sd=pstdev(vals) if len(vals)>1 else 0.0
    if boundary=="first":
        gap=vals[0]-vals[1] if len(vals)>=2 else 0.0
    elif boundary=="second":
        gap=vals[1]-vals[2] if len(vals)>=3 else 0.0
    else:
        gap=vals[2]-vals[3] if len(vals)>=4 else 0.0
    return {"sd":sd,"gap":gap}

def line_strength_spread(rows):
    er=v7.enrich_line_context(rows)
    byline={}
    for r in er:
        byline[r["_lid"]]=float(r["_line_strength"])
    vals=list(byline.values())
    return (max(vals)-min(vals)) if len(vals)>=2 else 0.0

def participation_features(first_sc,second_sc,third_sc,board,rows):
    fs=score_stats(first_sc,"first")
    ss=score_stats(second_sc,"second")
    ts=score_stats(third_sc,"third")
    cond_vals=sorted([float(x.get("conditional_score",0.0)) for x in second_sc],reverse=True)
    cond_gap=(cond_vals[0]-cond_vals[1]) if len(cond_vals)>=2 else 0.0
    return {
        "first_gap":fs["gap"],
        "second_gap":ss["gap"],
        "third_gap":ts["gap"],
        "first_sd":fs["sd"],
        "second_sd":ss["sd"],
        "third_sd":ts["sd"],
        "second_cond_gap":cond_gap,
        "line_strength_spread":line_strength_spread(rows),
        "candidate_cells":len(board["first"])+len(board["second"])+len(board["third"]),
        "first_count":len(board["first"]),
        "second_count":len(board["second"]),
        "third_count":len(board["third"]),
    }

HIGHER_BETTER=["first_gap","second_gap","third_gap","first_sd","second_sd","third_sd","second_cond_gap","line_strength_spread"]
LOWER_BETTER=["candidate_cells"]

def quantiles(vals):
    s=sorted(vals)
    if not s:return []
    def q(p):
        i=(len(s)-1)*p
        lo=int(math.floor(i)); hi=int(math.ceil(i))
        if lo==hi:return s[lo]
        return s[lo]*(hi-i)+s[hi]*(i-lo)
    return sorted(set([q(.25),q(.5),q(.75)]))

def rule_mask(rows,conds):
    out=[]
    for r in rows:
        ok=True
        for f,op,t in conds:
            v=r["features"][f]
            if op==">=" and not (v>=t):ok=False;break
            if op=="<=" and not (v<=t):ok=False;break
        out.append(ok)
    return out

def evaluate_rule(rows,conds):
    mask=rule_mask(rows,conds)
    selected=[r for r,m in zip(rows,mask) if m]
    n=len(rows); k=len(selected)
    if n==0 or k==0:return None
    complete=sum(r["complete"] for r in selected)/k
    participation=k/n
    score=complete*math.sqrt(participation)
    return {"conds":conds,"n":k,"participation_rate":participation,"complete_rate":complete,"selection_score":score}

def learn_rule(train_rows):
    candidates=[]
    for f in HIGHER_BETTER:
        vals=[r["features"][f] for r in train_rows]
        for t in quantiles(vals):
            candidates.append((f,">=",t))
    for f in LOWER_BETTER:
        vals=[r["features"][f] for r in train_rows]
        for t in quantiles(vals):
            candidates.append((f,"<=",t))

    rules=[]
    for c in candidates:
        ev=evaluate_rule(train_rows,[c])
        if ev and ev["n"]>=15 and ev["participation_rate"]>=0.25:rules.append(ev)
    for i,a in enumerate(candidates):
        for b in candidates[i+1:]:
            if a[0]==b[0]:continue
            ev=evaluate_rule(train_rows,[a,b])
            if ev and ev["n"]>=15 and ev["participation_rate"]>=0.25:rules.append(ev)

    if not rules:
        return {"conds":[],"n":len(train_rows),"participation_rate":1.0,
                "complete_rate":mean(r["complete"] for r in train_rows) if train_rows else 0.0,
                "selection_score":0.0}
    rules.sort(key=lambda x:(x["selection_score"],x["complete_rate"],x["participation_rate"]),reverse=True)
    return rules[0]

def build_prior_training_rows(entries_by):
    detail=json.loads(PRIOR_DETAIL.read_text(encoding="utf-8"))
    rows=[]
    for d in detail:
        rid=d["race"]["race_id"]
        er=entries_by.get(rid,[])
        if len(er)!=7:continue
        board=d["board"]
        first_sc=d["first_scores"]; second_sc=d["second_scores"]; third_sc=d["third_scores"]
        feats=participation_features(first_sc,second_sc,third_sc,board,er)
        rows.append({"race_id":rid,"features":feats,"complete":1 if all(d["hits"].values()) else 0})
    return rows

def summary_for(rows):
    n=len(rows)
    if not n:return {"races":0,"first_hit_rate":0,"second_hit_rate":0,"third_hit_rate":0,"complete_capture_rate":0,"avg_candidate_cells":0}
    return {
        "races":n,
        "first_hit_rate":sum(r["hit_1st"] for r in rows)/n,
        "second_hit_rate":sum(r["hit_2nd"] for r in rows)/n,
        "third_hit_rate":sum(r["hit_3rd"] for r in rows)/n,
        "complete_capture_rate":sum(r["complete_capture"] for r in rows)/n,
        "avg_candidate_cells":mean(r["candidate_cells"] for r in rows),
    }

def main():
    races=load_csv("races.csv"); entries=load_csv("entries.csv"); results=load_csv("results.csv")
    train=v7.get_target_races(races,TRAIN_START,TRAIN_END)
    test=v7.get_target_races(races,TEST_START,TEST_END)
    needed=set(train)|set(test)
    # prior participation-learning week also needs entries
    prior_ids=set()
    for d in json.loads(PRIOR_DETAIL.read_text(encoding="utf-8")):
        prior_ids.add(d["race"]["race_id"])
    needed |= prior_ids

    eb=defaultdict(list); rb=defaultdict(list)
    for e in entries:
        if e["race_id"] in needed:eb[e["race_id"]].append(e)
    for r in results:
        if r["race_id"] in needed:rb[r["race_id"]].append(r)

    participation_train=build_prior_training_rows(eb)
    rule=learn_rule(participation_train)

    models=v7.build_models(train.keys(),eb,rb)
    cond=v7.build_conditional_second_model(train.keys(),eb,rb)

    out=[]; detail=[]
    for rid,race in sorted(test.items(),key=lambda kv:(kv[1]["race_date"],kv[1]["track"],ino(kv[1]["race_no"]))):
        er=eb.get(rid,[])
        if len(er)!=7:continue
        actual={}; ok=True
        for p in (1,2,3):
            xs=[x for x in rb.get(rid,[]) if ino(x.get("finish_position"))==p]
            if len(xs)!=1:ok=False;break
            actual[p]=ino(xs[0]["car_no"])
        if not ok:continue

        first_sc=v7.score_rows(er,models["first"],"first"); first_c=v7.choose(first_sc,"first")
        second_base=v7.score_rows(er,models["second"],"second")
        second_sc=v7.conditional_second_scores(er,first_sc,first_c,second_base,cond); second_c=v7.choose(second_sc,"second")
        third_sc=v7.score_rows(er,models["third"],"third"); third_c=v7.choose(third_sc,"third")
        board={"first":first_c,"second":second_c,"third":third_c}
        feats=participation_features(first_sc,second_sc,third_sc,board,er)
        participate=rule_mask([{"features":feats}],rule["conds"])[0] if rule["conds"] else True
        hits={"first":actual[1] in first_c,"second":actual[2] in second_c,"third":actual[3] in third_c}
        complete=all(hits.values())
        rec={
            "race_id":rid,"race_date":race["race_date"],"track":race["track"],"race_no":race["race_no"],
            "participate":int(participate),
            "first_candidates":"-".join(map(str,first_c)),
            "second_candidates":"-".join(map(str,second_c)),
            "third_candidates":"-".join(map(str,third_c)),
            "actual_1st":actual[1],"actual_2nd":actual[2],"actual_3rd":actual[3],
            "hit_1st":int(hits["first"]),"hit_2nd":int(hits["second"]),"hit_3rd":int(hits["third"]),
            "complete_capture":int(complete),
            "candidate_cells":len(first_c)+len(second_c)+len(third_c),
        }
        out.append(rec)
        detail.append({"race":race,"board":board,"actual":actual,"hits":hits,"participate":participate,
                       "participation_features":feats,"rule":rule["conds"],
                       "first_scores":first_sc,"second_scores":second_sc,"third_scores":third_sc})

    participated=[r for r in out if r["participate"]]
    skipped=[r for r in out if not r["participate"]]
    summary={
        "algorithm":"keirin_shogi_v8_participation",
        "train_period":{"start":TRAIN_START,"end":TRAIN_END},
        "test_period":{"start":TEST_START,"end":TEST_END},
        "participation_rule_learning_source":"2024-02-26〜2024-03-03のv7アウトオブサンプル予測と結果",
        "participation_rule":{
            "conditions":[{"feature":f,"operator":op,"threshold":t} for f,op,t in rule["conds"]],
            "train_selected_races":rule["n"],
            "train_participation_rate":rule["participation_rate"],
            "train_complete_capture_rate":rule["complete_rate"],
            "selection_score":rule["selection_score"],
        },
        "all_races":summary_for(out),
        "participated_races":summary_for(participated),
        "skipped_races":summary_for(skipped),
        "participation_rate":len(participated)/len(out) if out else 0,
        "method":"v7の予測ロジックは維持。直前のアウトオブサンプル週で、1/2/3着スコア境界差・各段スコア分散・2着条件付きスコア差・ライン強度差・総配置マス数から、最低25%参加かつ15R以上となる透明な1〜2条件ルールを学習。検証週ではルールを固定。",
        "leakage_guard":"2024-03-04以降の結果は予測・参加判定学習に未使用。オッズ不使用。"
    }

    (OUT_DIR/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    (OUT_DIR/"model.json").write_text(json.dumps({"participation_rule":summary["participation_rule"],"conditional_second":cond},ensure_ascii=False,indent=2),encoding="utf-8")
    (OUT_DIR/"detail.json").write_text(json.dumps(detail,ensure_ascii=False,indent=2),encoding="utf-8")
    with (OUT_DIR/"race_log.csv").open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(out[0].keys()) if out else ["race_id"]);w.writeheader();w.writerows(out)
    (OUT_DIR/"README.md").write_text(
        "# 競輪将棋 v8 参加判定器\n\n"
        "- v7予測は維持\n- オッズ不使用\n- 参加判定は予測時情報のみ\n"
        "- 直前アウトオブサンプル週で透明な参加条件を学習し、次週固定検証\n",
        encoding="utf-8"
    )
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=="__main__":
    main()
