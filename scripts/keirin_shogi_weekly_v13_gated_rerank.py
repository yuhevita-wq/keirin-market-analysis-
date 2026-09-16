#!/usr/bin/env python3
from __future__ import annotations

import csv, json, math, importlib.util
from collections import defaultdict
from pathlib import Path
from statistics import pstdev

BASE = Path("data/2024/s_class_f1_all_parts/2024_q2")
TRAIN_START, TRAIN_END = "2024-04-08", "2024-04-14"
TEST_START, TEST_END = "2024-04-15", "2024-04-21"
PRIOR_DETAIL = Path("results/keirin_shogi/v12_top3_final_rerank/2024-04-08_2024-04-14/detail.json")
OUT_DIR = Path("results/keirin_shogi/v13_gated_rerank/2024-04-15_2024-04-21")
OUT_DIR.mkdir(parents=True, exist_ok=True)

spec11 = importlib.util.spec_from_file_location("v11", "scripts/keirin_shogi_weekly_v11_first_pairwise_core.py")
v11 = importlib.util.module_from_spec(spec11); spec11.loader.exec_module(v11)

spec12 = importlib.util.spec_from_file_location("v12", "scripts/keirin_shogi_weekly_v12_top3_final_rerank.py")
v12 = importlib.util.module_from_spec(spec12); spec12.loader.exec_module(v12)

def load_csv(name):
    with (BASE/name).open("r",encoding="utf-8-sig",newline="") as f:
        return list(csv.DictReader(f))

def ino(v):
    try:return int(float(v))
    except:return 0

def gate_features(base, final3):
    b=sorted(base,key=lambda x:(-float(x["score"]),ino(x["no"])))
    top1,top2,top3=b[0],b[1],b[2]
    margins=[float(p.get("margin",0.0)) for p in top1.get("pairs",[])]
    base1_final=next((float(x["final_score"]) for x in final3 if ino(x["no"])==ino(top1["no"])),0.0)
    best_alt=max([float(x["final_score"]) for x in final3 if ino(x["no"])!=ino(top1["no"])], default=base1_final)
    return {
        "score_gap12":float(top1["score"])-float(top2["score"]),
        "score_gap13":float(top1["score"])-float(top3["score"]),
        "raw_gap12":float(top1["raw_score"])-float(top2["raw_score"]),
        "avg_pair_margin_top1":float(top1.get("avg_pair_margin",0.0)),
        "worst_pair_margin_top1":float(top1.get("worst_pair_margin",0.0)),
        "pair_margin_sd_top1":pstdev(margins) if len(margins)>1 else 0.0,
        "rerank_alt_advantage":best_alt-base1_final,
    }

GATE_FEATURES=[
    "score_gap12","score_gap13","raw_gap12","avg_pair_margin_top1",
    "worst_pair_margin_top1","pair_margin_sd_top1","rerank_alt_advantage"
]

def quantiles(vals):
    s=sorted(vals)
    if not s:return []
    def q(p):
        i=(len(s)-1)*p
        lo=int(math.floor(i)); hi=int(math.ceil(i))
        if lo==hi:return s[lo]
        return s[lo]*(hi-i)+s[hi]*(i-lo)
    return sorted(set([q(.20),q(.35),q(.50),q(.65),q(.80)]))

def cond_ok(feat, cond):
    if cond is None:return False
    f,op,t=cond
    if op==">=":return feat[f]>=t
    return feat[f]<=t

def learn_gate():
    detail=json.loads(PRIOR_DETAIL.read_text(encoding="utf-8"))
    rows=[]
    for d in detail:
        base=d["v11_ranking"]
        final3=d["v12_top3_rerank"]
        actual=ino(d["actual_1st"])
        v11_top1=ino(sorted(base,key=lambda x:(-float(x["score"]),ino(x["no"])))[0]["no"])
        v12_top1=ino(sorted(final3,key=lambda x:(-float(x["final_score"]),-float(x["base_score"]),ino(x["no"])))[0]["no"])
        rows.append({
            "features":gate_features(base,final3),
            "actual":actual,
            "v11_hit":int(v11_top1==actual),
            "v12_hit":int(v12_top1==actual),
            "v11_top1":v11_top1,
            "v12_top1":v12_top1,
        })

    base_acc=sum(r["v11_hit"] for r in rows)/len(rows)
    candidates=[None]
    for f in GATE_FEATURES:
        vals=[r["features"][f] for r in rows]
        for t in quantiles(vals):
            candidates.append((f,">=",t))
            candidates.append((f,"<=",t))

    best={
        "condition":None,"train_accuracy":base_acc,"switch_rate":0.0,
        "switches":0,"train_races":len(rows),"v11_base_accuracy":base_acc
    }
    for cond in candidates[1:]:
        switches=[cond_ok(r["features"],cond) for r in rows]
        k=sum(switches)
        rate=k/len(rows)
        if k<8 or rate>0.60:
            continue
        hits=0
        for r,sw in zip(rows,switches):
            hits += r["v12_hit"] if sw else r["v11_hit"]
        acc=hits/len(rows)
        cand={
            "condition":cond,"train_accuracy":acc,"switch_rate":rate,
            "switches":k,"train_races":len(rows),"v11_base_accuracy":base_acc
        }
        if (cand["train_accuracy"],-cand["switch_rate"]) > (best["train_accuracy"],-best["switch_rate"]):
            best=cand
    return best

def main():
    gate=learn_gate()

    # v12 reranker itself is unchanged. Only the gate is new.
    reranker=v12.learn_reranker()

    races=load_csv("races.csv"); entries=load_csv("entries.csv"); results=load_csv("results.csv")
    train=v11.get_target_races(races,TRAIN_START,TRAIN_END)
    test=v11.get_target_races(races,TEST_START,TEST_END)
    needed=set(train)|set(test)

    eb=defaultdict(list); rb=defaultdict(list)
    for e in entries:
        if e["race_id"] in needed:eb[e["race_id"]].append(e)
    for r in results:
        if r["race_id"] in needed:rb[r["race_id"]].append(r)

    base_model=v11.learn_pairwise(train.keys(),eb,rb)

    out=[]; detail=[]
    cond=gate["condition"]
    for rid,race in sorted(test.items(),key=lambda kv:(kv[1]["race_date"],kv[1]["track"],ino(kv[1]["race_no"]))):
        er=eb.get(rid,[])
        if len(er)!=7:continue
        w=[x for x in rb.get(rid,[]) if ino(x.get("finish_position"))==1]
        if len(w)!=1:continue
        actual=ino(w[0]["car_no"])

        base=v11.first_rank(er,base_model)
        base_top1=ino(base[0]["no"])
        final3=v12.rerank_top3(base,reranker)
        rerank_top1=ino(final3[0]["no"])

        gf=gate_features(base,final3)
        switch=cond_ok(gf,cond) if cond is not None else False
        final_top1=rerank_top1 if switch else base_top1

        out.append({
            "race_id":rid,"race_date":race["race_date"],"track":race["track"],"race_no":race["race_no"],
            "actual_1st":actual,
            "v11_top1":base_top1,"v11_hit":int(base_top1==actual),
            "v12_top1":rerank_top1,"v12_hit":int(rerank_top1==actual),
            "v13_switch":int(switch),
            "v13_top1":final_top1,"v13_hit":int(final_top1==actual),
            **gf
        })
        detail.append({
            "race":race,"actual_1st":actual,"gate_features":gf,"switch":switch,
            "v11_ranking":base,"v12_top3_rerank":final3,"v13_top1":final_top1
        })

    n=len(out)
    def acc(k):return sum(r[k] for r in out)/n if n else 0
    paired={"v13_only_hit":0,"v11_only_hit":0,"both_hit":0,"both_miss":0}
    for r in out:
        a=bool(r["v13_hit"]); b=bool(r["v11_hit"])
        if a and b:paired["both_hit"]+=1
        elif a:paired["v13_only_hit"]+=1
        elif b:paired["v11_only_hit"]+=1
        else:paired["both_miss"]+=1

    summary={
        "algorithm":"keirin_shogi_v13_gated_rerank",
        "train_period":{"start":TRAIN_START,"end":TRAIN_END},
        "test_period":{"start":TEST_START,"end":TEST_END},
        "gate_learning_source":"2024-04-08〜2024-04-14 v12ホールドアウト結果。正解・不正解の両方を使用。",
        "gate":{
            "condition":None if cond is None else {"feature":cond[0],"operator":cond[1],"threshold":cond[2]},
            "train_accuracy":gate["train_accuracy"],
            "train_v11_base_accuracy":gate["v11_base_accuracy"],
            "train_switch_rate":gate["switch_rate"],
            "train_switches":gate["switches"]
        },
        "test_races":n,
        "v11_base_top1_hit_rate":acc("v11_hit"),
        "v12_always_rerank_top1_hit_rate":acc("v12_hit"),
        "v13_gated_top1_hit_rate":acc("v13_hit"),
        "v13_switch_rate":sum(r["v13_switch"] for r in out)/n if n else 0,
        "paired_v13_vs_v11":paired,
        "method":"v11を基本解として保持。v12の再順位付けを全レースに適用せず、直前ホールドアウト週の正解・不正解双方から学習した単一の透明ゲート条件を満たす時だけ上位3人再順位付けへ切り替える。候補数は広げない。",
        "leakage_guard":"2024-04-15以降の結果はゲート・v11学習に未使用。オッズ不使用。"
    }

    (OUT_DIR/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    (OUT_DIR/"model.json").write_text(json.dumps({"gate":summary["gate"],"reranker":reranker},ensure_ascii=False,indent=2),encoding="utf-8")
    (OUT_DIR/"detail.json").write_text(json.dumps(detail,ensure_ascii=False,indent=2),encoding="utf-8")
    with (OUT_DIR/"race_log.csv").open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(out[0].keys()) if out else ["race_id"])
        w.writeheader();w.writerows(out)
    (OUT_DIR/"README.md").write_text(
        "# 競輪将棋 v13 ゲート付き最終再順位付け\n\n"
        "v11の1位を基本維持し、事前特徴から『逆転を許可するレース』だけv12再順位付けを使う。\n"
        "候補数は広げず1着1位評価のみ検証。\n",
        encoding="utf-8"
    )
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=="__main__":
    main()
