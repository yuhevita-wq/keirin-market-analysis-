#!/usr/bin/env python3
from __future__ import annotations

import csv, json, importlib.util, math
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev

BASE = Path("data/2024/s_class_f1_all_parts/2024_q2")
TRAIN_START, TRAIN_END = "2024-04-15", "2024-04-21"
TEST_START, TEST_END = "2024-04-22", "2024-04-28"
PRIOR_DETAIL = Path("results/keirin_shogi/v13_gated_rerank/2024-04-15_2024-04-21/detail.json")
OUT_DIR = Path("results/keirin_shogi/v14_direct_challenger/2024-04-22_2024-04-28")
OUT_DIR.mkdir(parents=True, exist_ok=True)

spec11 = importlib.util.spec_from_file_location("v11", "scripts/keirin_shogi_weekly_v11_first_pairwise_core.py")
v11 = importlib.util.module_from_spec(spec11); spec11.loader.exec_module(v11)

def load_csv(name):
    with (BASE/name).open("r",encoding="utf-8-sig",newline="") as f:
        return list(csv.DictReader(f))

def ino(v):
    try:return int(float(v))
    except:return 0

def cand_meta(x):
    margins=[float(p.get("margin",0.0)) for p in x.get("pairs",[])]
    return {
        "score":float(x.get("score",0.0)),
        "raw_score":float(x.get("raw_score",0.0)),
        "avg_pair_margin":float(x.get("avg_pair_margin",0.0)),
        "worst_pair_margin":float(x.get("worst_pair_margin",0.0)),
        "best_pair_margin":max(margins) if margins else 0.0,
        "pair_margin_sd":pstdev(margins) if len(margins)>1 else 0.0,
    }

FEATURES=["score","raw_score","avg_pair_margin","worst_pair_margin","best_pair_margin","pair_margin_sd"]

def diff_features(challenger, leader):
    cm=cand_meta(challenger); lm=cand_meta(leader)
    return {f:cm[f]-lm[f] for f in FEATURES}

def fit_binary(samples):
    # label=1 means challenger actually won, 0 means leader held.
    if not samples:
        return {"n":0,"pos":0,"features":{},"bias":0.0}
    pos=[s for s in samples if s["label"]==1]
    neg=[s for s in samples if s["label"]==0]
    model={"n":len(samples),"pos":len(pos),"features":{}}
    p=(len(pos)+1)/(len(samples)+2)
    model["bias"]=math.log(p/(1-p))
    for f in FEATURES:
        allv=[s["x"][f] for s in samples]
        sd=pstdev(allv) or 1.0
        mp=mean([s["x"][f] for s in pos]) if pos else 0.0
        mn=mean([s["x"][f] for s in neg]) if neg else 0.0
        w=(mp-mn)/sd
        model["features"][f]={"sd":sd,"pos_mean":mp,"neg_mean":mn,"weight":max(-2.0,min(2.0,w))}
    return model

def learn_challengers():
    detail=json.loads(PRIOR_DETAIL.read_text(encoding="utf-8"))
    s2=[]; s3=[]
    for d in detail:
        ranking=sorted(d["v11_ranking"],key=lambda x:(-float(x["score"]),ino(x["no"])))
        if len(ranking)<3: continue
        actual=ino(d["actual_1st"])
        leader=ranking[0]
        rank_actual=next((i+1 for i,x in enumerate(ranking) if ino(x["no"])==actual),None)

        # Rank2 challenge learns only cases where winner is leader or rank2.
        if rank_actual in (1,2):
            s2.append({
                "x":diff_features(ranking[1],leader),
                "label":1 if rank_actual==2 else 0
            })

        # Rank3 challenge learns only cases where winner is leader or rank3.
        if rank_actual in (1,3):
            s3.append({
                "x":diff_features(ranking[2],leader),
                "label":1 if rank_actual==3 else 0
            })

    return {"rank2":fit_binary(s2),"rank3":fit_binary(s3)}

def logit_score(x, model):
    s=float(model.get("bias",0.0))
    parts={}
    for f,m in model.get("features",{}).items():
        z=x[f]/(m["sd"] or 1.0)
        z=max(-3.0,min(3.0,z))
        c=m["weight"]*z
        parts[f]=c; s+=c
    return s,parts

def decide(base, models):
    ranking=sorted(base,key=lambda x:(-float(x["score"]),ino(x["no"])))
    leader=ranking[0]; c2=ranking[1]; c3=ranking[2]
    x2=diff_features(c2,leader); x3=diff_features(c3,leader)
    s2,p2=logit_score(x2,models["rank2"])
    s3,p3=logit_score(x3,models["rank3"])

    # Challenger must have positive evidence against the leader.
    best=("leader",0.0,leader)
    if s2>best[1]: best=("rank2",s2,c2)
    if s3>best[1]: best=("rank3",s3,c3)

    return {
        "winner":ino(best[2]["no"]),
        "source":best[0],
        "rank2_score":s2,
        "rank3_score":s3,
        "rank2_parts":p2,
        "rank3_parts":p3,
    }

def main():
    challenger_models=learn_challengers()

    races=load_csv("races.csv"); entries=load_csv("entries.csv"); results=load_csv("results.csv")
    train=v11.get_target_races(races,TRAIN_START,TRAIN_END)
    test=v11.get_target_races(races,TEST_START,TEST_END)
    needed=set(train)|set(test)

    eb=defaultdict(list); rb=defaultdict(list)
    for e in entries:
        if e["race_id"] in needed: eb[e["race_id"]].append(e)
    for r in results:
        if r["race_id"] in needed: rb[r["race_id"]].append(r)

    # Fresh v11 base for the test week, trained only on the immediately prior week.
    base_model=v11.learn_pairwise(train.keys(),eb,rb)

    out=[]; detail=[]
    for rid,race in sorted(test.items(),key=lambda kv:(kv[1]["race_date"],kv[1]["track"],ino(kv[1]["race_no"]))):
        er=eb.get(rid,[])
        if len(er)!=7:continue
        w=[x for x in rb.get(rid,[]) if ino(x.get("finish_position"))==1]
        if len(w)!=1:continue
        actual=ino(w[0]["car_no"])

        base=v11.first_rank(er,base_model)
        base_top1=ino(base[0]["no"])
        actual_rank=next(i+1 for i,x in enumerate(base) if ino(x["no"])==actual)
        dec=decide(base,challenger_models)
        final=dec["winner"]

        out.append({
            "race_id":rid,"race_date":race["race_date"],"track":race["track"],"race_no":race["race_no"],
            "actual_1st":actual,
            "v11_top1":base_top1,"v11_hit":int(base_top1==actual),
            "v11_actual_rank":actual_rank,
            "v14_top1":final,"v14_hit":int(final==actual),
            "v14_source":dec["source"],
            "rank2_challenge_score":dec["rank2_score"],
            "rank3_challenge_score":dec["rank3_score"],
        })
        detail.append({
            "race":race,"actual_1st":actual,"v11_ranking":base,
            "decision":dec,"v14_top1":final
        })

    n=len(out)
    def rate(key):return sum(r[key] for r in out)/n if n else 0
    paired={"v14_only_hit":0,"v11_only_hit":0,"both_hit":0,"both_miss":0}
    switches={"leader":0,"rank2":0,"rank3":0}
    for r in out:
        switches[r["v14_source"]]+=1
        a=bool(r["v14_hit"]); b=bool(r["v11_hit"])
        if a and b:paired["both_hit"]+=1
        elif a:paired["v14_only_hit"]+=1
        elif b:paired["v11_only_hit"]+=1
        else:paired["both_miss"]+=1

    summary={
        "algorithm":"keirin_shogi_v14_direct_challenger",
        "train_period":{"start":TRAIN_START,"end":TRAIN_END},
        "test_period":{"start":TEST_START,"end":TEST_END},
        "challenger_learning_source":"2024-04-15〜2024-04-21 v13ホールドアウト内のv11順位。rank2/3をleaderと直接比較。",
        "challenger_models":challenger_models,
        "test_races":n,
        "v11_base_top1_hit_rate":rate("v11_hit"),
        "v14_direct_challenger_top1_hit_rate":rate("v14_hit"),
        "v11_top2_capture_rate":sum(1 for r in out if r["v11_actual_rank"]<=2)/n if n else 0,
        "v11_top3_capture_rate":sum(1 for r in out if r["v11_actual_rank"]<=3)/n if n else 0,
        "decision_counts":switches,
        "paired_v14_vs_v11":paired,
        "method":"v11を基本解とし、2位候補と1位候補、3位候補と1位候補を別々の二者問題として学習。実勝者がleaderまたは該当challengerだった事例だけで各モデルを訓練し、challenger側に正の直接証拠がある時だけ1位を置換。候補数は広げない。",
        "leakage_guard":"2024-04-22以降の結果はchallenger学習・v11学習に未使用。オッズ不使用。"
    }

    (OUT_DIR/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    (OUT_DIR/"model.json").write_text(json.dumps({"base_model":base_model,"challenger_models":challenger_models},ensure_ascii=False,indent=2),encoding="utf-8")
    (OUT_DIR/"detail.json").write_text(json.dumps(detail,ensure_ascii=False,indent=2),encoding="utf-8")
    with (OUT_DIR/"race_log.csv").open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(out[0].keys()) if out else ["race_id"])
        w.writeheader();w.writerows(out)
    (OUT_DIR/"README.md").write_text(
        "# 競輪将棋 v14 直接challenger学習\n\n"
        "v11 1位を基本とし、2位→1位、3位→1位の逆転条件を別モデルで直接学習。\n"
        "候補拡張なし。1着1位評価のみ検証。\n",
        encoding="utf-8"
    )
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=="__main__":
    main()
