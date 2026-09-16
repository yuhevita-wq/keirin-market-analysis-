#!/usr/bin/env python3
from __future__ import annotations

import csv, json, math
from collections import defaultdict, Counter
from pathlib import Path
from statistics import mean

BASE = Path("data/2024/s_class_f1_all_parts/2024_q1")
TRAIN_START, TRAIN_END = "2024-01-01", "2024-01-07"
TEST_START, TEST_END = "2024-01-08", "2024-01-14"
OUT_DIR = Path("results/keirin_shogi/v3_all_rank_shapes/2024-01-08_2024-01-14")
OUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURES = ["score","win_rate","b_count","nige_count","makuri_count","sashi_count","top2_rate","top3_rate"]
PAIR_FEATURES = ["score","win_rate","b_count","nige_count","makuri_count","sashi_count","top2_rate"]
RANK_LABELS = {1:"first",2:"second",3:"third"}

def read_csv(name):
    with (BASE/name).open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))

def num(v):
    try: return float(v)
    except: return 0.0

def ino(v):
    try: return int(float(v))
    except: return 0

def race_rank(rows, feature):
    ordered = sorted(rows, key=lambda r: (-num(r.get(feature)), ino(r.get("car_no"))))
    return {ino(r["car_no"]): i+1 for i,r in enumerate(ordered)}

def get_target_races(races, start, end):
    return {
        r["race_id"]: r for r in races
        if start <= r.get("race_date","") <= end
        and ino(r.get("entry_count")) == 7
        and r.get("meeting_grade") == "F1"
        and "Ｓ級" in r.get("race_type","")
    }

def build_models(train_ids, entries_by, results_by):
    # 着順別に「その着順に来た選手の順位ベクトル」を別学習。
    models={}
    for finish_pos, label in RANK_LABELS.items():
        target_rank = {f: Counter() for f in FEATURES}
        all_rank = {f: Counter() for f in FEATURES}
        pair_target = Counter()
        pair_all = Counter()
        races_used=0

        for rid in train_ids:
            rows=entries_by.get(rid,[])
            if len(rows)!=7: continue
            target=[x for x in results_by.get(rid,[]) if ino(x.get("finish_position"))==finish_pos]
            if len(target)!=1: continue
            target_no=ino(target[0]["car_no"])
            ranks={f:race_rank(rows,f) for f in FEATURES}
            races_used+=1

            for r in rows:
                no=ino(r["car_no"])
                for f in FEATURES:
                    rk=ranks[f][no]
                    all_rank[f][rk]+=1
                    if no==target_no: target_rank[f][rk]+=1

                for i,a in enumerate(PAIR_FEATURES):
                    for b in PAIR_FEATURES[i+1:]:
                        key=(a,b,ranks[a][no],ranks[b][no])
                        pair_all[key]+=1
                        if no==target_no: pair_target[key]+=1

        base=1/7
        uni={}
        for f in FEATURES:
            uni[f]={}
            for rk in range(1,8):
                t=target_rank[f][rk]
                a=all_rank[f][rk]
                p=(t+1.0)/(a+7.0)
                uni[f][rk]=math.log(max(p,1e-9)/base)

        pair={}
        for key,a in pair_all.items():
            if a<4: continue
            t=pair_target[key]
            p=(t+1.0)/(a+7.0)
            pair[key]=math.log(max(p,1e-9)/base)

        models[label]={
            "train_races":races_used,
            "univariate":uni,
            "pairwise":pair,
            "train_rank_counts":{
                f:{"target":dict(target_rank[f]),"all":dict(all_rank[f])} for f in FEATURES
            }
        }
    return models

def score_rows(rows, model):
    ranks={f:race_rank(rows,f) for f in FEATURES}
    scored=[]; raws=[]
    for r in rows:
        no=ino(r["car_no"])
        parts={}
        s=0.0
        for f in FEATURES:
            v=model["univariate"][f][ranks[f][no]]
            parts[f]=v; s+=v
        pair_s=0.0; used=0
        for i,a in enumerate(PAIR_FEATURES):
            for b in PAIR_FEATURES[i+1:]:
                key=(a,b,ranks[a][no],ranks[b][no])
                if key in model["pairwise"]:
                    pair_s+=model["pairwise"][key]; used+=1
        s += 0.35*pair_s
        parts["pairwise_35pct"]=0.35*pair_s
        raws.append(s)
        scored.append({
            "no":no,"name":r.get("player_name",""),
            "rank_vector":{f:ranks[f][no] for f in FEATURES},
            "raw_score":s,"parts":parts,"pair_terms_used":used
        })
    lo=min(raws); hi=max(raws)
    for x in scored:
        x["score"]=50.0 if hi==lo else (x["raw_score"]-lo)/(hi-lo)*100.0
    return scored

def choose(scored, rank_label):
    s=sorted(scored,key=lambda x:(-x["score"],x["no"]))
    if rank_label=="first":
        count=2
        if len(s)>=2 and s[0]["score"]-s[1]["score"]>=15: count=1
    elif rank_label=="second":
        count=3
        if len(s)>=3 and s[1]["score"]-s[2]["score"]>=15: count=2
    else:
        count=4
        if len(s)>=4 and s[2]["score"]-s[3]["score"]>=15: count=3
    return sorted(x["no"] for x in s[:count])

def main():
    races=read_csv("races.csv"); entries=read_csv("entries.csv"); results=read_csv("results.csv")
    train=get_target_races(races,TRAIN_START,TRAIN_END)
    test=get_target_races(races,TEST_START,TEST_END)
    needed=set(train)|set(test)

    entries_by=defaultdict(list); results_by=defaultdict(list)
    for e in entries:
        if e["race_id"] in needed: entries_by[e["race_id"]].append(e)
    for r in results:
        if r["race_id"] in needed: results_by[r["race_id"]].append(r)

    models=build_models(train.keys(),entries_by,results_by)

    rows_out=[]; detail=[]
    for rid,race in sorted(test.items(),key=lambda kv:(kv[1]["race_date"],kv[1]["track"],ino(kv[1]["race_no"]))):
        erows=entries_by.get(rid,[])
        if len(erows)!=7: continue

        actual={}
        valid=True
        for pos in (1,2,3):
            xs=[x for x in results_by.get(rid,[]) if ino(x.get("finish_position"))==pos]
            if len(xs)!=1: valid=False; break
            actual[pos]=ino(xs[0]["car_no"])
        if not valid: continue

        boards={}
        scored_by_rank={}
        hits={}
        for pos,label in RANK_LABELS.items():
            scored=score_rows(erows,models[label])
            cand=choose(scored,label)
            boards[label]=cand
            scored_by_rank[label]=scored
            hits[label]=actual[pos] in cand

        complete=all(hits.values())
        rows_out.append({
            "race_id":rid,"race_date":race["race_date"],"track":race["track"],"race_no":race["race_no"],"race_type":race["race_type"],
            "first_candidates":"-".join(map(str,boards["first"])),
            "second_candidates":"-".join(map(str,boards["second"])),
            "third_candidates":"-".join(map(str,boards["third"])),
            "actual_1st":actual[1],"actual_2nd":actual[2],"actual_3rd":actual[3],
            "hit_1st":int(hits["first"]),"hit_2nd":int(hits["second"]),"hit_3rd":int(hits["third"]),
            "complete_capture":int(complete),
            "candidate_cells":len(boards["first"])+len(boards["second"])+len(boards["third"]),
        })
        detail.append({
            "race":{k:race[k] for k in ["race_id","race_date","track","race_no","race_type"]},
            "board":boards,"actual":actual,"hits":hits,"complete_capture":complete,
            "scores":scored_by_rank
        })

    n=len(rows_out)
    summary={
        "algorithm":"keirin_shogi_v3_all_rank_shapes",
        "train_period":{"start":TRAIN_START,"end":TRAIN_END},
        "test_period":{"start":TEST_START,"end":TEST_END},
        "train_races_by_rank":{k:v["train_races"] for k,v in models.items()},
        "test_races":n,
        "first_hit_rate":sum(r["hit_1st"] for r in rows_out)/n if n else 0,
        "second_hit_rate":sum(r["hit_2nd"] for r in rows_out)/n if n else 0,
        "third_hit_rate":sum(r["hit_3rd"] for r in rows_out)/n if n else 0,
        "complete_capture_rate":sum(r["complete_capture"] for r in rows_out)/n if n else 0,
        "avg_candidate_cells":mean(r["candidate_cells"] for r in rows_out) if n else 0,
        "avg_first_candidates":mean(len(r["first_candidates"].split("-")) for r in rows_out) if n else 0,
        "avg_second_candidates":mean(len(r["second_candidates"].split("-")) for r in rows_out) if n else 0,
        "avg_third_candidates":mean(len(r["third_candidates"].split("-")) for r in rows_out) if n else 0,
        "features":FEATURES,
        "method":"1着・2着・3着を別々に、第1週の該当着順選手のレース内順位ベクトルから学習。役割・ライン位置は不使用。第2週は全モデル固定。",
        "leakage_guard":"2024-01-08以降の結果は学習に未使用。オッズ不使用。",
    }

    model_out={}
    for label,m in models.items():
        model_out[label]={
            "train_races":m["train_races"],
            "univariate":m["univariate"],
            "pairwise_terms":len(m["pairwise"]),
            "train_rank_counts":m["train_rank_counts"],
        }

    (OUT_DIR/"model.json").write_text(json.dumps(model_out,ensure_ascii=False,indent=2),encoding="utf-8")
    (OUT_DIR/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    (OUT_DIR/"detail.json").write_text(json.dumps(detail,ensure_ascii=False,indent=2),encoding="utf-8")
    with (OUT_DIR/"race_log.csv").open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows_out[0].keys()) if rows_out else ["race_id"])
        w.writeheader(); w.writerows(rows_out)

    md=[
        "# 競輪将棋 v3 全着順・勝者形学習",
        "",
        f"- 学習: {TRAIN_START}〜{TRAIN_END}",
        f"- 検証: {TEST_START}〜{TEST_END}",
        f"- 検証レース: {n}",
        f"- 1着段捕捉率: {summary['first_hit_rate']:.1%}",
        f"- 2着段捕捉率: {summary['second_hit_rate']:.1%}",
        f"- 3着段捕捉率: {summary['third_hit_rate']:.1%}",
        f"- 3段完全捕捉率: {summary['complete_capture_rate']:.1%}",
        f"- 平均配置マス数: {summary['avg_candidate_cells']:.2f}",
        "",
        "## 方針",
        "- 1着・2着・3着をそれぞれ別モデルで学習",
        "- 役割・ライン位置を着順評価に使わない",
        "- 各選手をレース内順位ベクトルで表現",
        "- 第1週学習、第2週固定検証",
        "- オッズ不使用",
    ]
    (OUT_DIR/"README.md").write_text("\n".join(md),encoding="utf-8")

    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=="__main__":
    main()
