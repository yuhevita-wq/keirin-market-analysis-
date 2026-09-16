#!/usr/bin/env python3
from __future__ import annotations

import csv, json, math, importlib.util
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev

BASE = Path("data/2024/s_class_f1_all_parts/2024_q2")
TRAIN_START, TRAIN_END = "2024-04-01", "2024-04-07"
TEST_START, TEST_END = "2024-04-08", "2024-04-14"
PRIOR_DETAIL = Path("results/keirin_shogi/v11_first_pairwise_core/2024-04-01_2024-04-07/detail.json")
OUT_DIR = Path("results/keirin_shogi/v12_top3_final_rerank/2024-04-08_2024-04-14")
OUT_DIR.mkdir(parents=True, exist_ok=True)

spec11 = importlib.util.spec_from_file_location("v11", "scripts/keirin_shogi_weekly_v11_first_pairwise_core.py")
v11 = importlib.util.module_from_spec(spec11); spec11.loader.exec_module(v11)

def load_csv(name):
    with (BASE/name).open("r",encoding="utf-8-sig",newline="") as f:
        return list(csv.DictReader(f))

def ino(v):
    try:return int(float(v))
    except:return 0

def candidate_meta(x):
    margins=[float(p.get("margin",0.0)) for p in x.get("pairs",[])]
    return {
        "score":float(x.get("score",0.0)),
        "raw_score":float(x.get("raw_score",0.0)),
        "avg_pair_margin":float(x.get("avg_pair_margin",0.0)),
        "worst_pair_margin":float(x.get("worst_pair_margin",0.0)),
        "best_pair_margin":max(margins) if margins else 0.0,
        "pair_margin_sd":pstdev(margins) if len(margins)>1 else 0.0,
    }

RERANK_FEATURES=["score","raw_score","avg_pair_margin","worst_pair_margin","best_pair_margin","pair_margin_sd"]

def learn_reranker():
    detail=json.loads(PRIOR_DETAIL.read_text(encoding="utf-8"))
    diffs=defaultdict(list)
    usable=0; correction_cases=0
    rank2=rank3=0
    for d in detail:
        ranking=sorted(d["v11_ranking"],key=lambda x:(-float(x["score"]),ino(x["no"])))
        actual=ino(d["actual_1st"])
        idx=next((i for i,x in enumerate(ranking) if ino(x["no"])==actual),None)
        if idx is None or idx>2:
            continue
        usable+=1
        if idx==0:
            continue
        correction_cases+=1
        if idx==1: rank2+=1
        if idx==2: rank3+=1
        wrong=ranking[0]; winner=ranking[idx]
        wm=candidate_meta(winner); pm=candidate_meta(wrong)
        for f in RERANK_FEATURES:
            diffs[f].append(wm[f]-pm[f])

    model={"usable_top3_train_races":usable,"correction_cases":correction_cases,
           "winner_was_rank2":rank2,"winner_was_rank3":rank3,"features":{}}
    for f in RERANK_FEATURES:
        vals=diffs[f]
        if not vals:
            continue
        mu=mean(vals); sd=pstdev(vals) or 1.0
        effect=mu/sd
        model["features"][f]={
            "winner_minus_wrong_top1_mean":mu,
            "sd":sd,
            "weight":max(-2.0,min(2.0,effect))
        }
    return model

def rerank_top3(base_ranking, model):
    top3=base_ranking[:3]
    current_top1=top3[0]
    top1m=candidate_meta(current_top1)
    rescored=[]
    for x in top3:
        xm=candidate_meta(x)
        correction=0.0; parts={}
        for f,m in model["features"].items():
            diff=xm[f]-top1m[f]
            z=diff/(m["sd"] or 1.0)
            z=max(-3.0,min(3.0,z))
            c=m["weight"]*z
            parts[f]=c; correction+=c
        # 既存順位を完全破壊しない。最終決戦だけ補正。
        base_rank_bonus={ino(top3[0]["no"]):0.60,ino(top3[1]["no"]):0.30,ino(top3[2]["no"]):0.0}[ino(x["no"])]
        final=correction+base_rank_bonus
        rescored.append({"no":ino(x["no"]),"name":x.get("name",""),"final_score":final,
                         "correction_score":correction,"base_rank_bonus":base_rank_bonus,
                         "base_score":float(x.get("score",0.0)),"parts":parts})
    rescored.sort(key=lambda x:(-x["final_score"],-x["base_score"],x["no"]))
    return rescored

def main():
    reranker=learn_reranker()

    races=load_csv("races.csv"); entries=load_csv("entries.csv"); results=load_csv("results.csv")
    train=v11.get_target_races(races,TRAIN_START,TRAIN_END)
    test=v11.get_target_races(races,TEST_START,TEST_END)
    needed=set(train)|set(test)

    eb=defaultdict(list); rb=defaultdict(list)
    for e in entries:
        if e["race_id"] in needed: eb[e["race_id"]].append(e)
    for r in results:
        if r["race_id"] in needed: rb[r["race_id"]].append(r)

    base_model=v11.learn_pairwise(train.keys(),eb,rb)

    rows_out=[]; detail=[]
    for rid,race in sorted(test.items(),key=lambda kv:(kv[1]["race_date"],kv[1]["track"],ino(kv[1]["race_no"]))):
        er=eb.get(rid,[])
        if len(er)!=7:continue
        w=[x for x in rb.get(rid,[]) if ino(x.get("finish_position"))==1]
        if len(w)!=1:continue
        actual=ino(w[0]["car_no"])

        base=v11.first_rank(er,base_model)
        base_top1=ino(base[0]["no"])
        base_actual_rank=next(i+1 for i,x in enumerate(base) if ino(x["no"])==actual)

        final3=rerank_top3(base,reranker)
        v12_top1=ino(final3[0]["no"])
        v12_rank = (next(i+1 for i,x in enumerate(final3) if ino(x["no"])==actual)
                    if actual in [ino(x["no"]) for x in final3] else base_actual_rank)

        rows_out.append({
            "race_id":rid,"race_date":race["race_date"],"track":race["track"],"race_no":race["race_no"],
            "actual_1st":actual,
            "v11_top1":base_top1,"v11_hit":int(base_top1==actual),"v11_actual_rank":base_actual_rank,
            "v12_top1":v12_top1,"v12_hit":int(v12_top1==actual),"v12_actual_rank":v12_rank,
            "winner_in_v11_top3":int(base_actual_rank<=3)
        })
        detail.append({"race":race,"actual_1st":actual,"v11_ranking":base,"v12_top3_rerank":final3})

    n=len(rows_out)
    def rate(key): return sum(r[key] for r in rows_out)/n if n else 0
    paired={"v12_only_hit":0,"v11_only_hit":0,"both_hit":0,"both_miss":0}
    for r in rows_out:
        a=bool(r["v12_hit"]); b=bool(r["v11_hit"])
        if a and b:paired["both_hit"]+=1
        elif a:paired["v12_only_hit"]+=1
        elif b:paired["v11_only_hit"]+=1
        else:paired["both_miss"]+=1

    summary={
        "algorithm":"keirin_shogi_v12_top3_final_rerank",
        "train_period":{"start":TRAIN_START,"end":TRAIN_END},
        "test_period":{"start":TEST_START,"end":TEST_END},
        "reranker_learning_source":"2024-04-01〜2024-04-07 v11完全ホールドアウト結果",
        "reranker_model":reranker,
        "test_races":n,
        "v11_base":{
            "top1_hit_rate":rate("v11_hit"),
            "top2_capture_rate":sum(1 for r in rows_out if r["v11_actual_rank"]<=2)/n if n else 0,
            "top3_capture_rate":sum(1 for r in rows_out if r["v11_actual_rank"]<=3)/n if n else 0
        },
        "v12_final_rerank":{
            "top1_hit_rate":rate("v12_hit"),
            "winner_in_base_top3_rate":rate("winner_in_v11_top3")
        },
        "paired_top1_comparison":paired,
        "method":"v11で上位3人を抽出し、直前ホールドアウト週で『誤1位 vs 実勝者(2位/3位)』の差から最終決戦補正を学習。次週では上位3人だけを再順位付けし、候補数は広げず1位評価だけを変更。",
        "leakage_guard":"2024-04-08以降の結果は学習に未使用。オッズ不使用。"
    }

    (OUT_DIR/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    (OUT_DIR/"model.json").write_text(json.dumps({"base_model":base_model,"reranker":reranker},ensure_ascii=False,indent=2),encoding="utf-8")
    (OUT_DIR/"detail.json").write_text(json.dumps(detail,ensure_ascii=False,indent=2),encoding="utf-8")
    with (OUT_DIR/"race_log.csv").open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows_out[0].keys()) if rows_out else ["race_id"])
        w.writeheader();w.writerows(rows_out)
    (OUT_DIR/"README.md").write_text(
        "# 競輪将棋 v12 上位3人の最終決戦\n\n"
        "v11上位3人のみ再比較。候補を広げず、1着1位評価だけを修正。\n"
        "学習: 2024-04-01〜04-07のv11ホールドアウト誤り / 検証: 2024-04-08〜04-14。\n",
        encoding="utf-8"
    )
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=="__main__":
    main()
