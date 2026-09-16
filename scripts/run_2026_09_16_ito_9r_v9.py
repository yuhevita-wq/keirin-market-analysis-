#!/usr/bin/env python3
from __future__ import annotations
import csv, json, importlib.util
from collections import defaultdict
from pathlib import Path

BASE=Path("data/2024/s_class_f1_all_parts/2024_q1")
OUT=Path("results/live/2026-09-16_ito_9r_v9.json")
OUT.parent.mkdir(parents=True,exist_ok=True)

spec7=importlib.util.spec_from_file_location("v7","scripts/keirin_shogi_weekly_v7_conditional_second.py")
v7=importlib.util.module_from_spec(spec7); spec7.loader.exec_module(v7)
spec8=importlib.util.spec_from_file_location("v8","scripts/keirin_shogi_weekly_v8_participation.py")
v8=importlib.util.module_from_spec(spec8); spec8.loader.exec_module(v8)

def read_csv(name):
    with (BASE/name).open("r",encoding="utf-8-sig",newline="") as f:
        return list(csv.DictReader(f))

# Current v9 validated state: train prediction model on 2024-03-11..03-17.
TRAIN_START,TRAIN_END="2024-03-11","2024-03-17"
V8_FIRST_GAP=22.574611055774504
V8_LINE_SPREAD=10.73500000000001

races=read_csv("races.csv"); entries=read_csv("entries.csv"); results=read_csv("results.csv")
train=v7.get_target_races(races,TRAIN_START,TRAIN_END)
eb=defaultdict(list); rb=defaultdict(list)
for e in entries:
    if e["race_id"] in train: eb[e["race_id"]].append(e)
for r in results:
    if r["race_id"] in train: rb[r["race_id"]].append(r)
models=v7.build_models(train.keys(),eb,rb)
cond=v7.build_conditional_second_model(train.keys(),eb,rb)

# 2026-09-16 Ito G3 final 9R. No odds.
# Line: 4-1 / 7-2 / 3 solo / 5 solo / 6 solo
current=[
 {"car_no":"1","player_name":"簗田一輝","score":"114.44","win_rate":"20.0","top2_rate":"48.0","top3_rate":"64.0","b_count":"0","nige_count":"0","makuri_count":"1","sashi_count":"7","mark_count":"4","line_id":"A","line_position":"2","line_size":"2"},
 {"car_no":"2","player_name":"阿部力也","score":"112.56","win_rate":"12.0","top2_rate":"36.0","top3_rate":"60.0","b_count":"0","nige_count":"0","makuri_count":"0","sashi_count":"5","mark_count":"4","line_id":"B","line_position":"2","line_size":"2"},
 {"car_no":"3","player_name":"小川真太郎","score":"107.29","win_rate":"47.0","top2_rate":"47.0","top3_rate":"76.5","b_count":"3","nige_count":"0","makuri_count":"3","sashi_count":"5","mark_count":"0","line_id":"C","line_position":"1","line_size":"1"},
 {"car_no":"4","player_name":"齋木翔多","score":"106.48","win_rate":"24.0","top2_rate":"28.0","top3_rate":"48.0","b_count":"7","nige_count":"2","makuri_count":"4","sashi_count":"1","mark_count":"0","line_id":"A","line_position":"1","line_size":"2"},
 {"car_no":"5","player_name":"脇本勇希","score":"106.22","win_rate":"17.8","top2_rate":"25.0","top3_rate":"39.3","b_count":"6","nige_count":"1","makuri_count":"5","sashi_count":"1","mark_count":"0","line_id":"D","line_position":"1","line_size":"1"},
 {"car_no":"6","player_name":"緒方将樹","score":"104.00","win_rate":"21.8","top2_rate":"34.3","top3_rate":"46.9","b_count":"5","nige_count":"1","makuri_count":"6","sashi_count":"2","mark_count":"2","line_id":"E","line_position":"1","line_size":"1"},
 {"car_no":"7","player_name":"伊東翔貴","score":"100.65","win_rate":"8.6","top2_rate":"21.7","top3_rate":"34.8","b_count":"5","nige_count":"2","makuri_count":"2","sashi_count":"1","mark_count":"0","line_id":"B","line_position":"1","line_size":"2"}
]

first_sc=v7.score_rows(current,models["first"],"first")
first_c=v7.choose(first_sc,"first")
second_base=v7.score_rows(current,models["second"],"second")
second_sc=v7.conditional_second_scores(current,first_sc,first_c,second_base,cond)
second_c=v7.choose(second_sc,"second")
third_sc=v7.score_rows(current,models["third"],"third")
third_c=v7.choose(third_sc,"third")
board={"first":first_c,"second":second_c,"third":third_c}
feats=v8.participation_features(first_sc,second_sc,third_sc,board,current)
participate=not(feats["first_gap"]>=V8_FIRST_GAP and feats["line_strength_spread"]>=V8_LINE_SPREAD)

def top(sc):
    return sorted(sc,key=lambda x:(-x["score"],x["no"]))

result={
 "race":"2026-09-16 伊東9R GIII S級決勝",
 "training_scope":"v9 prediction model trained on 2024-03-11..03-17 F1 S級7車",
 "out_of_distribution_note":"GIII決勝は学習対象F1と異なるため参考試走",
 "board":board,
 "participate":participate,
 "participation_features":feats,
 "first_scores":[{"no":x["no"],"score":x["score"]} for x in top(first_sc)],
 "second_scores":[{"no":x["no"],"score":x["score"],"conditional_score":x.get("conditional_score",0)} for x in top(second_sc)],
 "third_scores":[{"no":x["no"],"score":x["score"]} for x in top(third_sc)]
}
OUT.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
print(json.dumps(result,ensure_ascii=False,indent=2))
