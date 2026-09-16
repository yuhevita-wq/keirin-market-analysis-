#!/usr/bin/env python3
from __future__ import annotations

import csv, json, math, importlib.util
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev

BASE_Q1 = Path("data/2024/s_class_f1_all_parts/2024_q1")
BASE_Q2 = Path("data/2024/s_class_f1_all_parts/2024_q2")
TRAIN_START, TRAIN_END = "2024-03-25", "2024-03-31"
TEST_START, TEST_END = "2024-04-01", "2024-04-07"
OUT_DIR = Path("results/keirin_shogi/v11_first_pairwise_core/2024-04-01_2024-04-07")
OUT_DIR.mkdir(parents=True, exist_ok=True)

spec7 = importlib.util.spec_from_file_location("v7", "scripts/keirin_shogi_weekly_v7_conditional_second.py")
v7 = importlib.util.module_from_spec(spec7); spec7.loader.exec_module(v7)

# 1着専用。top2/top3率・従来pairwise加点・従来gap bonusは使わない。
FIRST_FEATURES = [
    "score","win_rate","b_count","nige_count","makuri_count","sashi_count",
    "line_size","line_strength","line_pos","self_gap_line_max_score",
    "line_strength_vs_other"
]

def load_csv(base,name):
    with (base/name).open("r",encoding="utf-8-sig",newline="") as f:
        return list(csv.DictReader(f))

def num(v):
    try:return float(v)
    except:return 0.0

def ino(v):
    try:return int(float(v))
    except:return 0

def get_target_races(races,start,end):
    return {r["race_id"]:r for r in races
            if start<=r.get("race_date","")<=end
            and ino(r.get("entry_count"))==7
            and r.get("meeting_grade")=="F1"
            and "Ｓ級" in r.get("race_type","")}

def enrich(rows):
    groups=defaultdict(list)
    for r in rows:
        lid=(r.get("line_id") or "").strip() or f"solo_{r.get('car_no')}"
        groups[lid].append(r)
    stats={}
    for lid,rs in groups.items():
        scores=[num(x.get("score")) for x in rs]
        bsum=sum(num(x.get("b_count")) for x in rs)
        attack=sum(num(x.get("nige_count"))+num(x.get("makuri_count")) for x in rs)
        strength=(mean(scores) if scores else 0.0)+0.8*bsum+0.8*attack
        stats[lid]={"size":len(rs),"max_score":max(scores) if scores else 0.0,"strength":strength}
    out=[]
    for r in rows:
        lid=(r.get("line_id") or "").strip() or f"solo_{r.get('car_no')}"
        s=stats[lid]
        others=[v for k,v in stats.items() if k!=lid]
        other_str=max([x["strength"] for x in others], default=s["strength"])
        x=dict(r)
        x["_lid"]=lid
        x["line_size"]=float(s["size"])
        x["line_strength"]=float(s["strength"])
        x["line_pos"]=float(ino(r.get("line_position")) or 4)
        x["self_gap_line_max_score"]=float(s["max_score"]-num(r.get("score")))
        x["line_strength_vs_other"]=float(s["strength"]-other_str)
        out.append(x)
    return out

def fvec(r):
    return {f:num(r.get(f)) for f in FIRST_FEATURES}

def pair_features(a,b):
    av=fvec(a); bv=fvec(b)
    d={f:av[f]-bv[f] for f in FIRST_FEATURES}
    same=1.0 if a["_lid"]==b["_lid"] else 0.0
    apos=ino(a.get("line_position")); bpos=ino(b.get("line_position"))
    d["same_line"]=same
    d["same_line_pos_adv"]=float((bpos-apos) if same and apos and bpos else 0)
    d["a_attack"]=num(a.get("nige_count"))+num(a.get("makuri_count"))
    d["b_attack"]=num(b.get("nige_count"))+num(b.get("makuri_count"))
    d["a_finish"]=num(a.get("sashi_count"))
    d["b_finish"]=num(b.get("sashi_count"))
    return d

def learn_pairwise(train_ids,entries_by,results_by):
    samples=[]
    for rid in train_ids:
        rows0=entries_by.get(rid,[])
        if len(rows0)!=7:continue
        rows=enrich(rows0); byno={ino(r["car_no"]):r for r in rows}
        w=[x for x in results_by.get(rid,[]) if ino(x.get("finish_position"))==1]
        if len(w)!=1:continue
        wno=ino(w[0]["car_no"])
        if wno not in byno:continue
        wr=byno[wno]
        for no,lr in byno.items():
            if no==wno:continue
            samples.append(pair_features(wr,lr))
    if not samples:raise RuntimeError("No training pairs")
    keys=list(samples[0].keys())
    model={"train_pairs":len(samples),"features":{}}
    for k in keys:
        vals=[s[k] for s in samples]
        mu=mean(vals); sd=pstdev(vals) or 1.0
        # winner-vs-loserで平均差が一貫する特徴ほど重くする。
        effect=mu/sd
        weight=max(-2.0,min(2.0,effect))
        model["features"][k]={"winner_minus_loser_mean":mu,"sd":sd,"weight":weight}
    return model

def pair_margin(a,b,model):
    pf=pair_features(a,b)
    s=0.0; parts={}
    for k,m in model["features"].items():
        z=pf[k]/(m["sd"] or 1.0)
        z=max(-3.0,min(3.0,z))
        c=m["weight"]*z
        parts[k]=c; s+=c
    return s,parts

def first_rank(rows0,model):
    rows=enrich(rows0)
    out=[]
    for a in rows:
        margins=[]; pair_parts=[]
        for b in rows:
            if ino(a["car_no"])==ino(b["car_no"]):continue
            m,p=pair_margin(a,b,model)
            margins.append(m); pair_parts.append({"vs":ino(b["car_no"]),"margin":m,"parts":p})
        # 1着は「他6人すべてとの比較」。平均だけでなく最低marginも少し効かせる。
        avg=mean(margins)
        worst=min(margins)
        score=avg+0.25*worst
        out.append({"no":ino(a["car_no"]),"name":a.get("player_name",""),"raw_score":score,
                    "avg_pair_margin":avg,"worst_pair_margin":worst,"pairs":pair_parts})
    out.sort(key=lambda x:(-x["raw_score"],x["no"]))
    vals=[x["raw_score"] for x in out]; lo=min(vals); hi=max(vals)
    for x in out:
        x["score"]=50.0 if hi==lo else (x["raw_score"]-lo)/(hi-lo)*100.0
    return out

def main():
    races1=load_csv(BASE_Q1,"races.csv"); entries1=load_csv(BASE_Q1,"entries.csv"); results1=load_csv(BASE_Q1,"results.csv")
    races2=load_csv(BASE_Q2,"races.csv"); entries2=load_csv(BASE_Q2,"entries.csv"); results2=load_csv(BASE_Q2,"results.csv")
    train=get_target_races(races1,TRAIN_START,TRAIN_END)
    test=get_target_races(races2,TEST_START,TEST_END)

    eb=defaultdict(list); rb=defaultdict(list)
    for e in entries1:
        if e["race_id"] in train:eb[e["race_id"]].append(e)
    for r in results1:
        if r["race_id"] in train:rb[r["race_id"]].append(r)
    for e in entries2:
        if e["race_id"] in test:eb[e["race_id"]].append(e)
    for r in results2:
        if r["race_id"] in test:rb[r["race_id"]].append(r)

    model=learn_pairwise(train.keys(),eb,rb)

    # 同じ学習週で旧v7 1着モデルも作り、同じ未来週で比較する。
    old_models=v7.build_models(train.keys(),eb,rb)

    rows_out=[]; detail=[]
    for rid,race in sorted(test.items(),key=lambda kv:(kv[1]["race_date"],kv[1]["track"],ino(kv[1]["race_no"]))):
        er=eb.get(rid,[])
        if len(er)!=7:continue
        w=[x for x in rb.get(rid,[]) if ino(x.get("finish_position"))==1]
        if len(w)!=1:continue
        actual=ino(w[0]["car_no"])

        newrank=first_rank(er,model)
        new_top1=newrank[0]["no"]
        new_pos=next(i+1 for i,x in enumerate(newrank) if x["no"]==actual)

        oldsc=v7.score_rows(er,old_models["first"],"first")
        oldrank=sorted(oldsc,key=lambda x:(-x["score"],x["no"]))
        old_top1=oldrank[0]["no"]
        old_pos=next(i+1 for i,x in enumerate(oldrank) if x["no"]==actual)

        rows_out.append({
            "race_id":rid,"race_date":race["race_date"],"track":race["track"],"race_no":race["race_no"],
            "actual_1st":actual,
            "v11_top1":new_top1,"v11_hit":int(new_top1==actual),"v11_actual_rank":new_pos,
            "v7_top1":old_top1,"v7_hit":int(old_top1==actual),"v7_actual_rank":old_pos,
            "v11_top1_score":newrank[0]["score"],"v7_top1_score":oldrank[0]["score"]
        })
        detail.append({"race":race,"actual_1st":actual,"v11_ranking":newrank,"v7_ranking":oldrank})

    n=len(rows_out)
    v11_hits=sum(r["v11_hit"] for r in rows_out)
    v7_hits=sum(r["v7_hit"] for r in rows_out)
    v11_top2=sum(1 for r in rows_out if r["v11_actual_rank"]<=2)
    v11_top3=sum(1 for r in rows_out if r["v11_actual_rank"]<=3)
    v7_top2=sum(1 for r in rows_out if r["v7_actual_rank"]<=2)
    v7_top3=sum(1 for r in rows_out if r["v7_actual_rank"]<=3)
    paired={"v11_only_hit":0,"v7_only_hit":0,"both_hit":0,"both_miss":0}
    for r in rows_out:
        if r["v11_hit"] and r["v7_hit"]:paired["both_hit"]+=1
        elif r["v11_hit"]:paired["v11_only_hit"]+=1
        elif r["v7_hit"]:paired["v7_only_hit"]+=1
        else:paired["both_miss"]+=1

    summary={
        "algorithm":"keirin_shogi_v11_first_pairwise_core",
        "train_period":{"start":TRAIN_START,"end":TRAIN_END},
        "test_period":{"start":TEST_START,"end":TEST_END},
        "train_races":len(train),"test_races":n,
        "v11_pairwise_core":{
            "top1_hit_rate":v11_hits/n if n else 0,
            "top2_capture_rate":v11_top2/n if n else 0,
            "top3_capture_rate":v11_top3/n if n else 0
        },
        "same_week_v7_first_baseline":{
            "top1_hit_rate":v7_hits/n if n else 0,
            "top2_capture_rate":v7_top2/n if n else 0,
            "top3_capture_rate":v7_top3/n if n else 0
        },
        "paired_top1_comparison":paired,
        "method":"1着専用コア。従来の全特徴加点ランキングを捨て、学習週の実勝者と各非勝者の直接差から特徴方向を学習。未来週では各選手が他6人に対して持つpair marginを統合し1位を決定。top2/top3率、従来pairwise加点、従来gap bonusは1着コアから除外。オッズ不使用。",
        "leakage_guard":"2024-04-01以降の結果は学習に未使用。Q1最終週のみで学習し、Q2最初の週を完全ホールドアウト。"
    }
    (OUT_DIR/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    (OUT_DIR/"model.json").write_text(json.dumps(model,ensure_ascii=False,indent=2),encoding="utf-8")
    (OUT_DIR/"detail.json").write_text(json.dumps(detail,ensure_ascii=False,indent=2),encoding="utf-8")
    with (OUT_DIR/"race_log.csv").open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows_out[0].keys()) if rows_out else ["race_id"])
        w.writeheader();w.writerows(rows_out)
    (OUT_DIR/"README.md").write_text(
        "# 競輪将棋 v11 1着pairwiseコア\n\n"
        "1着候補を広げる実験ではなく、1位評価そのものの的中率を検証。\n"
        "学習: 2024-03-25〜03-31 / 検証: 2024-04-01〜04-07。\n",
        encoding="utf-8"
    )
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=="__main__":
    main()
