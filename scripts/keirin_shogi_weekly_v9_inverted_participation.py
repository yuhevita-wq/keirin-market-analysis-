#!/usr/bin/env python3
from __future__ import annotations

import csv, json, importlib.util
from collections import defaultdict
from pathlib import Path
from statistics import mean

BASE = Path("data/2024/s_class_f1_all_parts/2024_q1")
TRAIN_START, TRAIN_END = "2024-03-04", "2024-03-10"
TEST_START, TEST_END = "2024-03-11", "2024-03-17"
OUT_DIR = Path("results/keirin_shogi/v9_inverted_participation/2024-03-11_2024-03-17")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# v7予測本体
spec7 = importlib.util.spec_from_file_location("v7", "scripts/keirin_shogi_weekly_v7_conditional_second.py")
v7 = importlib.util.module_from_spec(spec7); spec7.loader.exec_module(v7)

# v8の参加特徴量計算だけ再利用
spec8 = importlib.util.spec_from_file_location("v8", "scripts/keirin_shogi_weekly_v8_participation.py")
v8 = importlib.util.module_from_spec(spec8); spec8.loader.exec_module(v8)

# v8で「入る条件」として採用したものを、そのまま反転して固定。
V8_FIRST_GAP = 22.574611055774504
V8_LINE_SPREAD = 10.73500000000001

def load_csv(name):
    with (BASE/name).open("r",encoding="utf-8-sig",newline="") as f:
        return list(csv.DictReader(f))

def ino(v):
    try:return int(float(v))
    except:return 0

def inverted_participate(features):
    # v8: first_gap>=閾値 AND line_strength_spread>=閾値 なら参加
    # v9: その補集合に参加。つまり「明確さ」ではなく「曖昧さ」を選ぶ。
    return not (
        features["first_gap"] >= V8_FIRST_GAP
        and features["line_strength_spread"] >= V8_LINE_SPREAD
    )

def summarize(rows):
    n=len(rows)
    if not n:
        return {"races":0,"first_hit_rate":0,"second_hit_rate":0,"third_hit_rate":0,
                "complete_capture_rate":0,"avg_candidate_cells":0}
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
    eb=defaultdict(list); rb=defaultdict(list)
    for e in entries:
        if e["race_id"] in needed: eb[e["race_id"]].append(e)
    for r in results:
        if r["race_id"] in needed: rb[r["race_id"]].append(r)

    models=v7.build_models(train.keys(),eb,rb)
    cond=v7.build_conditional_second_model(train.keys(),eb,rb)

    out=[]; detail=[]
    for rid,race in sorted(test.items(),key=lambda kv:(kv[1]["race_date"],kv[1]["track"],ino(kv[1]["race_no"]))):
        er=eb.get(rid,[])
        if len(er)!=7: continue

        actual={}; ok=True
        for p in (1,2,3):
            xs=[x for x in rb.get(rid,[]) if ino(x.get("finish_position"))==p]
            if len(xs)!=1:
                ok=False; break
            actual[p]=ino(xs[0]["car_no"])
        if not ok: continue

        first_sc=v7.score_rows(er,models["first"],"first")
        first_c=v7.choose(first_sc,"first")
        second_base=v7.score_rows(er,models["second"],"second")
        second_sc=v7.conditional_second_scores(er,first_sc,first_c,second_base,cond)
        second_c=v7.choose(second_sc,"second")
        third_sc=v7.score_rows(er,models["third"],"third")
        third_c=v7.choose(third_sc,"third")

        board={"first":first_c,"second":second_c,"third":third_c}
        feats=v8.participation_features(first_sc,second_sc,third_sc,board,er)
        participate=inverted_participate(feats)

        hits={
            "first":actual[1] in first_c,
            "second":actual[2] in second_c,
            "third":actual[3] in third_c,
        }
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
            "first_gap":feats["first_gap"],
            "line_strength_spread":feats["line_strength_spread"],
        }
        out.append(rec)
        detail.append({
            "race":race,"board":board,"actual":actual,"hits":hits,
            "participate":participate,"participation_features":feats,
            "inverted_rule":{
                "original_v8":"first_gap >= 22.5746110558 AND line_strength_spread >= 10.735",
                "v9":"NOT(original_v8)"
            },
            "first_scores":first_sc,"second_scores":second_sc,"third_scores":third_sc,
        })

    participated=[r for r in out if r["participate"]]
    skipped=[r for r in out if not r["participate"]]

    summary={
        "algorithm":"keirin_shogi_v9_inverted_participation",
        "train_period":{"start":TRAIN_START,"end":TRAIN_END},
        "test_period":{"start":TEST_START,"end":TEST_END},
        "participation_rule":{
            "type":"fixed_inverse_of_v8",
            "participate_when":"first_gap < 22.5746110558 OR line_strength_spread < 10.735",
            "source":"v8検証で、従来参加群より見送り群の完全捕捉率が高かったため、v8参加条件の補集合を次週へ固定適用"
        },
        "all_races":summarize(out),
        "participated_races":summarize(participated),
        "skipped_races":summarize(skipped),
        "participation_rate":len(participated)/len(out) if out else 0,
        "method":"予測本体はv7を維持。参加思想だけを反転。『1着が明確・ライン差が大きいほど入りやすい』を捨て、v8で見送っていた曖昧側を参加対象とする。次週では閾値を変更しない。",
        "leakage_guard":"2024-03-11以降の結果は学習・参加判定に未使用。オッズ不使用。"
    }

    (OUT_DIR/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    (OUT_DIR/"model.json").write_text(json.dumps({
        "participation_rule":summary["participation_rule"],
        "conditional_second":cond
    },ensure_ascii=False,indent=2),encoding="utf-8")
    (OUT_DIR/"detail.json").write_text(json.dumps(detail,ensure_ascii=False,indent=2),encoding="utf-8")
    with (OUT_DIR/"race_log.csv").open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(out[0].keys()) if out else ["race_id"])
        w.writeheader(); w.writerows(out)
    (OUT_DIR/"README.md").write_text(
        "# 競輪将棋 v9 反転参加判定\n\n"
        "- v7予測本体は維持\n"
        "- v8の参加条件をそのまま反転\n"
        "- 明確なレースではなく、v8が見送っていた曖昧側に参加\n"
        "- オッズ不使用\n"
        "- 2024-03-11〜03-17で固定検証\n",
        encoding="utf-8"
    )
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=="__main__":
    main()
