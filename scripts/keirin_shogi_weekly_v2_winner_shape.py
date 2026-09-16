#!/usr/bin/env python3
from __future__ import annotations

import csv, json, math
from collections import defaultdict, Counter
from pathlib import Path
from statistics import mean

BASE = Path("data/2024/s_class_f1_all_parts/2024_q1")
TRAIN_START, TRAIN_END = "2024-01-01", "2024-01-07"
TEST_START, TEST_END = "2024-01-08", "2024-01-14"
OUT_DIR = Path("results/keirin_shogi/v2_winner_shape/2024-01-08_2024-01-14")
OUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURES = ["score","win_rate","b_count","nige_count","makuri_count","sashi_count","top2_rate"]
PAIR_FEATURES = ["score","win_rate","b_count","nige_count","makuri_count","top2_rate"]

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

def build_model(train_ids, entries_by, results_by):
    # 学習データ: 各レースで勝者1人、非勝者6人。
    # 「役割」「ライン位置」は使わず、レース内順位ベクトルだけを学習。
    win_rank = {f: Counter() for f in FEATURES}
    all_rank = {f: Counter() for f in FEATURES}
    pair_win = Counter()
    pair_all = Counter()
    races_used = 0

    for rid in train_ids:
        rows = entries_by.get(rid, [])
        if len(rows) != 7: continue
        first = [x for x in results_by.get(rid, []) if ino(x.get("finish_position")) == 1]
        if len(first) != 1: continue
        winner = ino(first[0]["car_no"])
        ranks = {f: race_rank(rows, f) for f in FEATURES}
        races_used += 1

        for r in rows:
            no = ino(r["car_no"])
            for f in FEATURES:
                rk = ranks[f][no]
                all_rank[f][rk] += 1
                if no == winner:
                    win_rank[f][rk] += 1

            for i,a in enumerate(PAIR_FEATURES):
                for b in PAIR_FEATURES[i+1:]:
                    key=(a,b,ranks[a][no],ranks[b][no])
                    pair_all[key] += 1
                    if no == winner:
                        pair_win[key] += 1

    # Laplace平滑。勝者基準率1/7に対するlog-likelihood ratio。
    base = 1/7
    uni = {}
    for f in FEATURES:
        uni[f] = {}
        for rk in range(1,8):
            w = win_rank[f][rk]
            a = all_rank[f][rk]
            p = (w + 1.0) / (a + 7.0)
            uni[f][rk] = math.log(max(p,1e-9) / base)

    # 組み合わせはサンプル不足を避けるため、出現4件以上だけ。
    pair = {}
    for key,a in pair_all.items():
        if a < 4: continue
        w = pair_win[key]
        p = (w + 1.0) / (a + 7.0)
        pair[key] = math.log(max(p,1e-9) / base)

    return {
        "train_races": races_used,
        "univariate": uni,
        "pairwise": pair,
        "train_rank_counts": {
            f: {"winner": dict(win_rank[f]), "all": dict(all_rank[f])} for f in FEATURES
        }
    }

def score_rows(rows, model):
    ranks = {f: race_rank(rows, f) for f in FEATURES}
    scored=[]
    raw_scores=[]
    for r in rows:
        no=ino(r["car_no"])
        parts={}
        s=0.0
        for f in FEATURES:
            v=model["univariate"][f][ranks[f][no]]
            parts[f]=v
            s+=v

        pair_s=0.0
        used_pairs=0
        for i,a in enumerate(PAIR_FEATURES):
            for b in PAIR_FEATURES[i+1:]:
                key=(a,b,ranks[a][no],ranks[b][no])
                if key in model["pairwise"]:
                    pair_s += model["pairwise"][key]
                    used_pairs += 1
        # pairwiseは過学習しやすいので35%だけ加える。
        s += 0.35 * pair_s
        parts["pairwise_35pct"] = 0.35 * pair_s
        raw_scores.append(s)
        scored.append({
            "no": no,
            "name": r.get("player_name",""),
            "rank_vector": {f:ranks[f][no] for f in FEATURES},
            "raw_score": s,
            "parts": parts,
            "pair_terms_used": used_pairs,
        })

    lo=min(raw_scores); hi=max(raw_scores)
    for x in scored:
        x["score"] = 50.0 if hi==lo else (x["raw_score"]-lo)/(hi-lo)*100.0
    return scored

def choose_first(scored):
    s=sorted(scored,key=lambda x:(-x["score"],x["no"]))
    count=2
    if len(s)>=2 and s[0]["score"]-s[1]["score"]>=15:
        count=1
    return sorted(x["no"] for x in s[:count])

def main():
    races=read_csv("races.csv"); entries=read_csv("entries.csv"); results=read_csv("results.csv")
    train=get_target_races(races,TRAIN_START,TRAIN_END)
    test=get_target_races(races,TEST_START,TEST_END)

    entries_by=defaultdict(list); results_by=defaultdict(list)
    needed=set(train)|set(test)
    for e in entries:
        if e["race_id"] in needed: entries_by[e["race_id"]].append(e)
    for r in results:
        if r["race_id"] in needed: results_by[r["race_id"]].append(r)

    model=build_model(train.keys(),entries_by,results_by)

    rows_out=[]; detail=[]
    for rid,race in sorted(test.items(),key=lambda kv:(kv[1]["race_date"],kv[1]["track"],ino(kv[1]["race_no"]))):
        erows=entries_by.get(rid,[])
        if len(erows)!=7: continue
        first=[x for x in results_by.get(rid,[]) if ino(x.get("finish_position"))==1]
        second=[x for x in results_by.get(rid,[]) if ino(x.get("finish_position"))==2]
        third=[x for x in results_by.get(rid,[]) if ino(x.get("finish_position"))==3]
        if not(len(first)==len(second)==len(third)==1): continue
        actual={1:ino(first[0]["car_no"]),2:ino(second[0]["car_no"]),3:ino(third[0]["car_no"])}

        scored=score_rows(erows,model)
        first_candidates=choose_first(scored)
        hit=actual[1] in first_candidates

        rows_out.append({
            "race_id":rid,"race_date":race["race_date"],"track":race["track"],"race_no":race["race_no"],
            "race_type":race["race_type"],"first_candidates":"-".join(map(str,first_candidates)),
            "actual_1st":actual[1],"actual_2nd":actual[2],"actual_3rd":actual[3],
            "hit_1st":int(hit),"candidate_count":len(first_candidates)
        })
        detail.append({
            "race":{k:race[k] for k in ["race_id","race_date","track","race_no","race_type"]},
            "first_candidates":first_candidates,"actual":actual,"hit_1st":hit,"scored":scored
        })

    n=len(rows_out)
    summary={
        "algorithm":"keirin_shogi_v2_winner_shape",
        "train_period":{"start":TRAIN_START,"end":TRAIN_END},
        "test_period":{"start":TEST_START,"end":TEST_END},
        "train_races":model["train_races"],
        "test_races":n,
        "first_hit_rate":sum(r["hit_1st"] for r in rows_out)/n if n else 0,
        "avg_first_candidate_count":mean(r["candidate_count"] for r in rows_out) if n else 0,
        "features":FEATURES,
        "method":"各特徴をレース内1〜7位へ変換。第1週の勝者/全選手から順位別勝率のlog-likelihood ratioを学習し、主要特徴の順位ペアを35%加点。役割・ライン位置は不使用。第2週は完全固定。",
        "leakage_guard":"2024-01-08以降の結果はモデル構築に未使用。オッズ不使用。",
    }

    (OUT_DIR/"model.json").write_text(json.dumps({
        "train_races":model["train_races"],
        "univariate":model["univariate"],
        "train_rank_counts":model["train_rank_counts"],
        "pairwise_terms":len(model["pairwise"]),
    },ensure_ascii=False,indent=2),encoding="utf-8")
    (OUT_DIR/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    (OUT_DIR/"detail.json").write_text(json.dumps(detail,ensure_ascii=False,indent=2),encoding="utf-8")
    with (OUT_DIR/"race_log.csv").open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows_out[0].keys()) if rows_out else ["race_id"])
        w.writeheader(); w.writerows(rows_out)

    md=[
        "# 競輪将棋 v2 勝者の形",
        "",
        f"- 学習: {TRAIN_START}〜{TRAIN_END}",
        f"- 検証: {TEST_START}〜{TEST_END}",
        f"- 学習レース: {model['train_races']}",
        f"- 検証レース: {n}",
        f"- 1着捕捉率: {summary['first_hit_rate']:.1%}",
        f"- 平均1着候補数: {summary['avg_first_candidate_count']:.2f}",
        "",
        "## 方針",
        "- 先頭・番手・単騎などの役割ラベルを1着評価に使わない",
        "- 各選手をレース内順位ベクトルで表現",
        "- 第1週で勝者に多い順位形を学習",
        "- 第2週はモデル固定で検証",
        "- オッズ不使用",
    ]
    (OUT_DIR/"README.md").write_text("\n".join(md),encoding="utf-8")

    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=="__main__":
    main()
