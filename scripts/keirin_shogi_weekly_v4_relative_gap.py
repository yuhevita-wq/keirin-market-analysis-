#!/usr/bin/env python3
from __future__ import annotations

import csv, json, math
from collections import defaultdict, Counter
from pathlib import Path
from statistics import mean

BASE = Path("data/2024/s_class_f1_all_parts/2024_q1")
TRAIN_START, TRAIN_END = "2024-01-08", "2024-01-14"
TEST_START, TEST_END = "2024-01-15", "2024-01-21"
OUT_DIR = Path("results/keirin_shogi/v4_relative_gap/2024-01-15_2024-01-21")
OUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURES = ["score","win_rate","b_count","nige_count","makuri_count","sashi_count","top2_rate","top3_rate"]
PAIR_FEATURES = ["score","win_rate","b_count","nige_count","makuri_count","sashi_count","top2_rate"]
GAP_FEATURES = ["score","win_rate","b_count","nige_count","makuri_count","top2_rate"]
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

def minmax_norm(values):
    lo=min(values); hi=max(values)
    if hi==lo: return [50.0]*len(values)
    return [(v-lo)/(hi-lo)*100.0 for v in values]

def relative_gap_features(rows):
    # 各選手について「レース内トップとの差」と「2番手との差」を作る。
    out={}
    for f in GAP_FEATURES:
        vals=[num(r.get(f)) for r in rows]
        nv=minmax_norm(vals)
        ordered=sorted(nv, reverse=True)
        top=ordered[0]
        second=ordered[1] if len(ordered)>1 else ordered[0]
        for i,r in enumerate(rows):
            no=ino(r["car_no"])
            out.setdefault(no,{})
            out[no][f+"_gap_top"] = top - nv[i]
            out[no][f+"_gap_second"] = second - nv[i]
    return out

def build_models(train_ids, entries_by, results_by):
    models={}
    for finish_pos, label in RANK_LABELS.items():
        target_rank={f:Counter() for f in FEATURES}
        all_rank={f:Counter() for f in FEATURES}
        pair_target=Counter(); pair_all=Counter()
        gap_target=defaultdict(list); gap_all=defaultdict(list)
        races_used=0

        for rid in train_ids:
            rows=entries_by.get(rid,[])
            if len(rows)!=7: continue
            target=[x for x in results_by.get(rid,[]) if ino(x.get("finish_position"))==finish_pos]
            if len(target)!=1: continue
            target_no=ino(target[0]["car_no"])
            ranks={f:race_rank(rows,f) for f in FEATURES}
            gaps=relative_gap_features(rows)
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

                if label=="first":
                    for g,v in gaps[no].items():
                        gap_all[g].append(v)
                        if no==target_no: gap_target[g].append(v)

        base=1/7
        uni={}
        for f in FEATURES:
            uni[f]={}
            for rk in range(1,8):
                t=target_rank[f][rk]; a=all_rank[f][rk]
                p=(t+1.0)/(a+7.0)
                uni[f][rk]=math.log(max(p,1e-9)/base)

        pair={}
        for key,a in pair_all.items():
            if a<4: continue
            t=pair_target[key]
            p=(t+1.0)/(a+7.0)
            pair[key]=math.log(max(p,1e-9)/base)

        gap_model={}
        if label=="first":
            # 勝者はトップとの差が小さい特徴ほど加点。
            # 学習週の勝者平均との差と全選手平均との差を比較し、
            # その差が大きい特徴だけ使う。
            for g,vals in gap_all.items():
                if not vals or not gap_target[g]: continue
                all_avg=mean(vals); win_avg=mean(gap_target[g])
                advantage=max(0.0, all_avg-win_avg)
                gap_model[g]={
                    "weight": advantage/100.0,
                    "winner_avg_gap": win_avg,
                    "all_avg_gap": all_avg
                }

        models[label]={
            "train_races":races_used,
            "univariate":uni,
            "pairwise":pair,
            "gap_model":gap_model,
        }
    return models

def score_rows(rows, model, label):
    ranks={f:race_rank(rows,f) for f in FEATURES}
    gaps=relative_gap_features(rows)
    scored=[]; raws=[]

    for r in rows:
        no=ino(r["car_no"])
        parts={}; s=0.0
        for f in FEATURES:
            v=model["univariate"][f][ranks[f][no]]
            parts[f]=v; s+=v

        pair_s=0.0
        for i,a in enumerate(PAIR_FEATURES):
            for b in PAIR_FEATURES[i+1:]:
                key=(a,b,ranks[a][no],ranks[b][no])
                if key in model["pairwise"]:
                    pair_s+=model["pairwise"][key]
        s+=0.35*pair_s
        parts["pairwise_35pct"]=0.35*pair_s

        if label=="first":
            gap_bonus=0.0
            for g,meta in model["gap_model"].items():
                # gap=0が最高。100に近いほどトップから遠い。
                closeness=max(0.0,100.0-gaps[no][g])/100.0
                gap_bonus += closeness * meta["weight"]
            # 相対差は主役にしすぎず、補正として入れる。
            s += 1.5*gap_bonus
            parts["relative_gap_bonus"]=1.5*gap_bonus

        raws.append(s)
        scored.append({
            "no":no,"name":r.get("player_name",""),
            "rank_vector":{f:ranks[f][no] for f in FEATURES},
            "gaps":gaps[no] if label=="first" else {},
            "raw_score":s,"parts":parts
        })

    lo=min(raws); hi=max(raws)
    for x in scored:
        x["score"]=50.0 if hi==lo else (x["raw_score"]-lo)/(hi-lo)*100.0
    return scored

def choose(scored, label):
    s=sorted(scored,key=lambda x:(-x["score"],x["no"]))
    if label=="first":
        # 1着は相対差で候補数を決める。
        # 1位が大きく抜ければ1枚、僅差なら2枚。
        gap=s[0]["score"]-s[1]["score"]
        count=1 if gap>=18 else 2
    elif label=="second":
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

        boards={}; scores={}; hits={}
        for pos,label in RANK_LABELS.items():
            scored=score_rows(erows,models[label],label)
            cand=choose(scored,label)
            boards[label]=cand; scores[label]=scored
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
            "board":boards,"actual":actual,"hits":hits,"complete_capture":complete,"scores":scores
        })

    n=len(rows_out)
    summary={
        "algorithm":"keirin_shogi_v4_relative_gap",
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
        "first_relative_features":list(models["first"]["gap_model"].keys()),
        "method":"v3の順位ベクトル学習に加え、1着だけは各主要特徴のレース内トップ/2番手との差を学習して補正。1着候補数も1位と2位のスコア差で決定。役割・ライン位置・オッズは不使用。1/8-1/14で学習し、1/15-1/21で固定検証。",
        "leakage_guard":"2024-01-15以降の結果は学習に未使用。オッズ不使用。",
    }

    (OUT_DIR/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    (OUT_DIR/"detail.json").write_text(json.dumps(detail,ensure_ascii=False,indent=2),encoding="utf-8")
    (OUT_DIR/"model.json").write_text(json.dumps({
        label:{
            "train_races":m["train_races"],
            "gap_model":m["gap_model"],
            "pairwise_terms":len(m["pairwise"])
        } for label,m in models.items()
    },ensure_ascii=False,indent=2),encoding="utf-8")
    with (OUT_DIR/"race_log.csv").open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows_out[0].keys()) if rows_out else ["race_id"])
        w.writeheader(); w.writerows(rows_out)

    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=="__main__":
    main()
