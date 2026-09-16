#!/usr/bin/env python3
from __future__ import annotations

import csv, json, math
from collections import defaultdict, Counter
from pathlib import Path
from statistics import mean

BASE = Path("data/2024/s_class_f1_all_parts/2024_q1")
TRAIN_START, TRAIN_END = "2024-01-29", "2024-02-04"
TEST_START, TEST_END = "2024-02-05", "2024-02-11"
OUT_DIR = Path("results/keirin_shogi/v6_line_context/2024-02-05_2024-02-11")
OUT_DIR.mkdir(parents=True, exist_ok=True)

BASE_FEATURES = ["score","win_rate","b_count","nige_count","makuri_count","sashi_count","top2_rate","top3_rate"]
LINE_FEATURES = [
    "line_size_ctx","line_avg_score","line_max_score","line_b_sum","line_attack_sum",
    "self_gap_line_max_score","line_max_vs_other_best","line_strength_vs_other","solo_flag"
]
FEATURES = BASE_FEATURES + LINE_FEATURES
PAIR_FEATURES = ["score","win_rate","b_count","makuri_count","top2_rate","line_avg_score","line_max_vs_other_best","line_strength_vs_other"]
GAP_FEATURES = ["score","win_rate","b_count","nige_count","makuri_count","sashi_count","top2_rate","top3_rate",
                "line_avg_score","line_max_score","line_b_sum","line_attack_sum","line_max_vs_other_best","line_strength_vs_other"]
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

def get_target_races(races,start,end):
    return {r["race_id"]:r for r in races
            if start<=r.get("race_date","")<=end
            and ino(r.get("entry_count"))==7
            and r.get("meeting_grade")=="F1"
            and "Ｓ級" in r.get("race_type","")}

def minmax_norm(values):
    lo=min(values); hi=max(values)
    if hi==lo: return [50.0]*len(values)
    return [(v-lo)/(hi-lo)*100.0 for v in values]

def enrich_line_context(rows):
    # ラインは「関係特徴」としてのみ使用。line_position/役割ラベルは不使用。
    groups=defaultdict(list)
    for r in rows:
        lid=(r.get("line_id") or "").strip()
        if not lid:
            lid=f"solo_{r.get('car_no')}"
        groups[lid].append(r)

    stats={}
    for lid,rs in groups.items():
        scores=[num(x.get("score")) for x in rs]
        bsum=sum(num(x.get("b_count")) for x in rs)
        attack=sum(num(x.get("nige_count"))+num(x.get("makuri_count")) for x in rs)
        stats[lid]={
            "size":len(rs),
            "avg_score":mean(scores) if scores else 0.0,
            "max_score":max(scores) if scores else 0.0,
            "b_sum":bsum,
            "attack_sum":attack,
            "strength":(mean(scores) if scores else 0.0) + 0.8*bsum + 0.8*attack,
        }

    best_other={}
    best_strength_other={}
    for lid in stats:
        others=[v for k,v in stats.items() if k!=lid]
        best_other[lid]=max([x["max_score"] for x in others], default=stats[lid]["max_score"])
        best_strength_other[lid]=max([x["strength"] for x in others], default=stats[lid]["strength"])

    out=[]
    for r in rows:
        lid=(r.get("line_id") or "").strip()
        if not lid: lid=f"solo_{r.get('car_no')}"
        s=stats[lid]
        x=dict(r)
        x["line_size_ctx"]=float(s["size"])
        x["line_avg_score"]=s["avg_score"]
        x["line_max_score"]=s["max_score"]
        x["line_b_sum"]=s["b_sum"]
        x["line_attack_sum"]=s["attack_sum"]
        x["self_gap_line_max_score"]=s["max_score"]-num(r.get("score"))
        x["line_max_vs_other_best"]=s["max_score"]-best_other[lid]
        x["line_strength_vs_other"]=s["strength"]-best_strength_other[lid]
        x["solo_flag"]=1.0 if s["size"]==1 else 0.0
        out.append(x)
    return out

def race_rank(rows, feature):
    # gap系の self_gap_line_max_score と solo_flag も「大きい順」で一旦順位化。
    ordered=sorted(rows,key=lambda r:(-num(r.get(feature)),ino(r.get("car_no"))))
    return {ino(r["car_no"]):i+1 for i,r in enumerate(ordered)}

def relative_gap_features(rows):
    out={}
    for f in GAP_FEATURES:
        vals=[num(r.get(f)) for r in rows]
        nv=minmax_norm(vals)
        ordered=sorted(nv,reverse=True)
        top=ordered[0]
        second=ordered[1] if len(ordered)>1 else ordered[0]
        third=ordered[2] if len(ordered)>2 else ordered[-1]
        for i,r in enumerate(rows):
            no=ino(r["car_no"])
            out.setdefault(no,{})
            out[no][f+"_gap_top"]=top-nv[i]
            out[no][f+"_gap_second"]=abs(second-nv[i])
            out[no][f+"_gap_third"]=abs(third-nv[i])
    return out

def build_models(train_ids, entries_by, results_by):
    models={}
    for finish_pos,label in RANK_LABELS.items():
        target_rank={f:Counter() for f in FEATURES}
        all_rank={f:Counter() for f in FEATURES}
        pair_target=Counter(); pair_all=Counter()
        gap_target=defaultdict(list); gap_all=defaultdict(list)
        races_used=0

        for rid in train_ids:
            rows0=entries_by.get(rid,[])
            if len(rows0)!=7: continue
            rows=enrich_line_context(rows0)
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
        for g,vals in gap_all.items():
            if not vals or not gap_target[g]: continue
            all_avg=mean(vals); target_avg=mean(gap_target[g])
            advantage=max(0.0,all_avg-target_avg)
            if advantage>0:
                gap_model[g]={"weight":advantage/100.0,"target_avg_gap":target_avg,"all_avg_gap":all_avg}

        models[label]={"train_races":races_used,"univariate":uni,"pairwise":pair,"gap_model":gap_model}
    return models

def score_rows(rows0, model, label):
    rows=enrich_line_context(rows0)
    ranks={f:race_rank(rows,f) for f in FEATURES}
    gaps=relative_gap_features(rows)
    scored=[]; raws=[]

    for r in rows:
        no=ino(r["car_no"]); parts={}; s=0.0
        for f in FEATURES:
            v=model["univariate"][f][ranks[f][no]]
            parts[f]=v; s+=v

        pair_s=0.0
        for i,a in enumerate(PAIR_FEATURES):
            for b in PAIR_FEATURES[i+1:]:
                key=(a,b,ranks[a][no],ranks[b][no])
                if key in model["pairwise"]: pair_s+=model["pairwise"][key]
        s+=0.30*pair_s
        parts["pairwise_30pct"]=0.30*pair_s

        gap_bonus=0.0
        for g,meta in model["gap_model"].items():
            closeness=max(0.0,100.0-gaps[no][g])/100.0
            gap_bonus+=closeness*meta["weight"]
        mult={"first":1.5,"second":1.35,"third":1.0}[label]
        s+=mult*gap_bonus
        parts["relative_gap_bonus"]=mult*gap_bonus

        raws.append(s)
        scored.append({
            "no":no,"name":r.get("player_name",""),
            "rank_vector":{f:ranks[f][no] for f in FEATURES},
            "line_context":{f:r.get(f) for f in LINE_FEATURES},
            "raw_score":s,"parts":parts
        })

    lo=min(raws); hi=max(raws)
    for x in scored:
        x["score"]=50.0 if hi==lo else (x["raw_score"]-lo)/(hi-lo)*100.0
    return scored

def choose(scored,label):
    s=sorted(scored,key=lambda x:(-x["score"],x["no"]))
    if label=="first":
        count=1 if s[0]["score"]-s[1]["score"]>=18 else 2
    elif label=="second":
        count=2 if len(s)>=3 and s[1]["score"]-s[2]["score"]>=12 else 3
    else:
        count=3 if len(s)>=4 and s[2]["score"]-s[3]["score"]>=15 else 4
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
        actual={}; valid=True
        for pos in (1,2,3):
            xs=[x for x in results_by.get(rid,[]) if ino(x.get("finish_position"))==pos]
            if len(xs)!=1: valid=False; break
            actual[pos]=ino(xs[0]["car_no"])
        if not valid: continue

        boards={}; scores={}; hits={}
        for pos,label in RANK_LABELS.items():
            scored=score_rows(erows,models[label],label)
            cand=choose(scored,label)
            boards[label]=cand; scores[label]=scored; hits[label]=actual[pos] in cand

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
        detail.append({"race":{k:race[k] for k in ["race_id","race_date","track","race_no","race_type"]},
                       "board":boards,"actual":actual,"hits":hits,"complete_capture":complete,"scores":scores})

    n=len(rows_out)
    summary={
        "algorithm":"keirin_shogi_v6_line_context",
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
        "line_features":LINE_FEATURES,
        "method":"v5の順位ベクトル＋相対差を維持し、line_id/line_sizeから同ライン平均・最大得点、B合計、逃げ+捲り合計、本人とライン最大得点差、自ライン最大得点と他ライン最大得点差、自ライン総合力と他ライン総合力差、単騎フラグを関係特徴として追加。line_positionや役割固定ルールは不使用。1/29-2/4学習、2/5-2/11固定検証。",
        "leakage_guard":"2024-02-05以降の結果は学習に未使用。オッズ不使用。"
    }

    (OUT_DIR/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    (OUT_DIR/"detail.json").write_text(json.dumps(detail,ensure_ascii=False,indent=2),encoding="utf-8")
    (OUT_DIR/"model.json").write_text(json.dumps({
        label:{"train_races":m["train_races"],"pairwise_terms":len(m["pairwise"]),"gap_terms":list(m["gap_model"].keys())}
        for label,m in models.items()
    },ensure_ascii=False,indent=2),encoding="utf-8")
    with (OUT_DIR/"race_log.csv").open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows_out[0].keys()) if rows_out else ["race_id"])
        w.writeheader(); w.writerows(rows_out)
    (OUT_DIR/"README.md").write_text(
        "# 競輪将棋 v6 ライン文脈モデル\n\n"
        f"- 学習: {TRAIN_START}〜{TRAIN_END}\n- 検証: {TEST_START}〜{TEST_END}\n"
        "- ラインは候補制約ではなく関係特徴としてのみ使用\n- line_position不使用\n- オッズ不使用\n",
        encoding="utf-8"
    )
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=="__main__":
    main()
