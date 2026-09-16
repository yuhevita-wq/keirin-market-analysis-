#!/usr/bin/env python3
from __future__ import annotations

import csv, json, math, importlib.util
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev

BASES=[
    Path("data/2024/s_class_f1_all_parts/2024_q1"),
    Path("data/2024/s_class_f1_all_parts/2024_q2"),
]
TRAIN_START, TRAIN_END = "2024-03-04", "2024-04-28"
BASELINE_TRAIN_START, BASELINE_TRAIN_END = "2024-04-22", "2024-04-28"
TEST_START, TEST_END = "2024-04-29", "2024-05-05"
OUT_DIR = Path("results/keirin_shogi/v15_race_level_softmax/2024-04-29_2024-05-05")
OUT_DIR.mkdir(parents=True, exist_ok=True)

spec11 = importlib.util.spec_from_file_location("v11", "scripts/keirin_shogi_weekly_v11_first_pairwise_core.py")
v11 = importlib.util.module_from_spec(spec11); spec11.loader.exec_module(v11)

BASE_FEATURES=[
    "score","win_rate","b_count","nige_count","makuri_count","sashi_count",
    "mark_count","first_count","second_count","third_count"
]

def load_all(name):
    out=[]
    for base in BASES:
        p=base/name
        if p.exists():
            with p.open("r",encoding="utf-8-sig",newline="") as f:
                out.extend(csv.DictReader(f))
    return out

def num(v):
    try:return float(v)
    except:return 0.0

def ino(v):
    try:return int(float(v))
    except:return 0

def target_races(races,start,end):
    return {r["race_id"]:r for r in races
            if start<=r.get("race_date","")<=end
            and ino(r.get("entry_count"))==7
            and r.get("meeting_grade")=="F1"
            and "Ｓ級" in r.get("race_type","")}

def zmap(vals):
    mu=mean(vals); sd=pstdev(vals) or 1.0
    return [(v-mu)/sd for v in vals]

def race_features(rows):
    # One race is one learning unit. No car-number rank features.
    groups=defaultdict(list)
    for r in rows:
        lid=(r.get("line_id") or "").strip() or f"solo_{r.get('car_no')}"
        groups[lid].append(r)

    line_stats={}
    for lid,rs in groups.items():
        avg_score=mean([num(x.get("score")) for x in rs])
        bsum=sum(num(x.get("b_count")) for x in rs)
        attack=sum(num(x.get("nige_count"))+num(x.get("makuri_count")) for x in rs)
        line_stats[lid]={
            "size":len(rs),
            "max_score":max(num(x.get("score")) for x in rs),
            "strength":avg_score+0.8*bsum+0.8*attack,
        }

    raw_by_feature={f:[num(r.get(f)) for r in rows] for f in BASE_FEATURES}
    z_by_feature={f:zmap(vs) for f,vs in raw_by_feature.items()}
    strengths=[]
    for r in rows:
        lid=(r.get("line_id") or "").strip() or f"solo_{r.get('car_no')}"
        strengths.append(line_stats[lid]["strength"])
    z_strength=zmap(strengths)

    feats=[]
    for i,r in enumerate(rows):
        lid=(r.get("line_id") or "").strip() or f"solo_{r.get('car_no')}"
        ls=line_stats[lid]
        others=[v["strength"] for k,v in line_stats.items() if k!=lid]
        other_best=max(others) if others else ls["strength"]
        pos=ino(r.get("line_position"))
        solo=1.0 if ls["size"]==1 else 0.0
        attack_z=z_by_feature["nige_count"][i]+z_by_feature["makuri_count"][i]
        x={}
        for f in BASE_FEATURES:
            x[f"z_{f}"]=z_by_feature[f][i]
        x.update({
            "line_size_1":1.0 if ls["size"]==1 else 0.0,
            "line_size_2":1.0 if ls["size"]==2 else 0.0,
            "line_size_3p":1.0 if ls["size"]>=3 else 0.0,
            "line_pos_1":1.0 if pos==1 else 0.0,
            "line_pos_2":1.0 if pos==2 else 0.0,
            "line_pos_3p":1.0 if pos>=3 else 0.0,
            "solo":solo,
            "z_line_strength":z_strength[i],
            "gap_to_line_max":(num(r.get("score"))-ls["max_score"])/5.0,
            "line_strength_vs_other":(ls["strength"]-other_best)/10.0,
            "attack_x_pos1":attack_z*(1.0 if pos==1 else 0.0),
            "sashi_x_pos2":z_by_feature["sashi_count"][i]*(1.0 if pos==2 else 0.0),
            "makuri_x_line_adv":z_by_feature["makuri_count"][i]*((ls["strength"]-other_best)/10.0),
            "score_x_pos2":z_by_feature["score"][i]*(1.0 if pos==2 else 0.0),
        })
        feats.append({"no":ino(r["car_no"]),"name":r.get("player_name",""),"x":x})
    return feats

def build_dataset(race_ids, eb, rb):
    ds=[]
    for rid in race_ids:
        rows=eb.get(rid,[])
        if len(rows)!=7: continue
        w=[x for x in rb.get(rid,[]) if ino(x.get("finish_position"))==1]
        if len(w)!=1: continue
        winner=ino(w[0]["car_no"])
        fs=race_features(rows)
        if winner not in [x["no"] for x in fs]: continue
        ds.append((rid,fs,winner))
    return ds

def softmax(scores):
    m=max(scores)
    ex=[math.exp(max(-30,min(30,s-m))) for s in scores]
    z=sum(ex)
    return [v/z for v in ex]

def train_softmax(ds, epochs=450, lr=0.035, l2=0.0015):
    names=sorted(next(iter(ds))[1][0]["x"].keys())
    w={f:0.0 for f in names}
    loss_hist=[]
    n=len(ds)
    for ep in range(epochs):
        grad={f:0.0 for f in names}
        loss=0.0
        for rid,fs,winner in ds:
            scores=[sum(w[f]*a["x"][f] for f in names) for a in fs]
            ps=softmax(scores)
            yi=next(i for i,a in enumerate(fs) if a["no"]==winner)
            loss-=math.log(max(ps[yi],1e-12))
            for i,a in enumerate(fs):
                err=ps[i]-(1.0 if i==yi else 0.0)
                for f in names:
                    grad[f]+=err*a["x"][f]
        for f in names:
            grad[f]=grad[f]/n + l2*w[f]
            w[f]-=lr*grad[f]
        if ep in (0,49,99,199,299,449):
            loss_hist.append({"epoch":ep+1,"cross_entropy":loss/n})
    return {"weights":w,"loss_history":loss_hist,"train_races":n}

def predict(fs,model):
    w=model["weights"]; names=w.keys()
    scores=[sum(w[f]*a["x"][f] for f in names) for a in fs]
    ps=softmax(scores)
    out=[]
    for a,s,p in zip(fs,scores,ps):
        out.append({"no":a["no"],"name":a["name"],"logit":s,"prob":p})
    out.sort(key=lambda x:(-x["prob"],x["no"]))
    return out

def choose_candidates(pred, threshold):
    out=[]; cum=0.0
    for x in pred[:3]:
        out.append(x["no"]); cum+=x["prob"]
        if cum>=threshold: break
    return out

def learn_candidate_threshold(ds,model):
    choices=[]
    for t in [0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75]:
        hits=0; counts=[]
        for _,fs,winner in ds:
            p=predict(fs,model)
            c=choose_candidates(p,t)
            hits+=int(winner in c); counts.append(len(c))
        capture=hits/len(ds); avgk=mean(counts)
        objective=capture/math.sqrt(avgk)
        choices.append({"threshold":t,"capture":capture,"avg_candidates":avgk,"objective":objective})
    choices.sort(key=lambda x:(x["objective"],x["capture"],-x["avg_candidates"]),reverse=True)
    return choices[0],choices

def main():
    races=load_all("races.csv"); entries=load_all("entries.csv"); results=load_all("results.csv")
    train=target_races(races,TRAIN_START,TRAIN_END)
    baseline_train=target_races(races,BASELINE_TRAIN_START,BASELINE_TRAIN_END)
    test=target_races(races,TEST_START,TEST_END)
    needed=set(train)|set(test)

    eb=defaultdict(list); rb=defaultdict(list)
    for e in entries:
        if e["race_id"] in needed: eb[e["race_id"]].append(e)
    for r in results:
        if r["race_id"] in needed: rb[r["race_id"]].append(r)

    train_ds=build_dataset(train.keys(),eb,rb)
    test_ds=build_dataset(test.keys(),eb,rb)
    model=train_softmax(train_ds)
    cand_policy,cand_grid=learn_candidate_threshold(train_ds,model)

    # Existing v11 rolling one-week baseline on exactly the same future week.
    old_model=v11.learn_pairwise(baseline_train.keys(),eb,rb)

    out=[]; detail=[]
    for rid,fs,winner in test_ds:
        race=test[rid]
        pred=predict(fs,model)
        top1=pred[0]["no"]
        cands=choose_candidates(pred,cand_policy["threshold"])

        oldrank=v11.first_rank(eb[rid],old_model)
        oldtop=ino(oldrank[0]["no"])

        out.append({
            "race_id":rid,"race_date":race["race_date"],"track":race["track"],"race_no":race["race_no"],
            "actual_1st":winner,
            "v15_top1":top1,"v15_top1_hit":int(top1==winner),
            "v15_first_candidates":"-".join(map(str,cands)),
            "v15_candidate_count":len(cands),
            "v15_candidate_hit":int(winner in cands),
            "v15_top1_prob":pred[0]["prob"],
            "v15_top2_prob":pred[1]["prob"],
            "v15_prob_gap12":pred[0]["prob"]-pred[1]["prob"],
            "v11_top1":oldtop,"v11_top1_hit":int(oldtop==winner),
        })
        detail.append({"race":race,"actual_1st":winner,"v15_ranking":pred,"first_candidates":cands,"v11_ranking":oldrank})

    n=len(out)
    def rate(k):return sum(r[k] for r in out)/n if n else 0
    counts=[r["v15_candidate_count"] for r in out]
    dist={str(k):sum(1 for x in counts if x==k) for k in (1,2,3)}
    paired={"v15_only_hit":0,"v11_only_hit":0,"both_hit":0,"both_miss":0}
    for r in out:
        a=bool(r["v15_top1_hit"]); b=bool(r["v11_top1_hit"])
        if a and b:paired["both_hit"]+=1
        elif a:paired["v15_only_hit"]+=1
        elif b:paired["v11_only_hit"]+=1
        else:paired["both_miss"]+=1

    summary={
        "algorithm":"keirin_shogi_v15_race_level_softmax",
        "train_period":{"start":TRAIN_START,"end":TRAIN_END},
        "test_period":{"start":TEST_START,"end":TEST_END},
        "train_races":len(train_ds),"test_races":n,
        "architecture":"7人を同時入力するrace-level softmax。1レース=1学習単位。勝者vs敗者の水増しpair生成をしない。",
        "v15":{
            "top1_hit_rate":rate("v15_top1_hit"),
            "adaptive_first_capture_rate":rate("v15_candidate_hit"),
            "avg_first_candidates":mean(counts) if counts else 0,
            "candidate_count_distribution":dist,
            "candidate_threshold":cand_policy["threshold"],
            "candidate_policy_train_stats":cand_policy,
        },
        "v11_one_week_baseline":{"top1_hit_rate":rate("v11_top1_hit")},
        "paired_top1_comparison":paired,
        "candidate_threshold_grid":cand_grid,
        "training_loss":model["loss_history"],
        "feature_weights":dict(sorted(model["weights"].items(),key=lambda kv:abs(kv[1]),reverse=True)),
        "method":"過去8週間を使い、各レース内で個人能力・相対能力・ライン構造・相互作用を7人同時に比較して勝者確率を学習。1着は必ず1人に固定せず、学習期間だけで選んだ累積確率閾値により上位1〜3人を盤面候補とする。候補を機械的に3人へ広げる方式ではない。",
        "leakage_guard":"2024-04-29以降の結果は学習・候補閾値選択に未使用。オッズ不使用。車番は特徴量に未使用、同確率時の表示順のみ車番で決定。"
    }
    (OUT_DIR/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    (OUT_DIR/"model.json").write_text(json.dumps({"softmax_model":model,"candidate_policy":cand_policy},ensure_ascii=False,indent=2),encoding="utf-8")
    (OUT_DIR/"detail.json").write_text(json.dumps(detail,ensure_ascii=False,indent=2),encoding="utf-8")
    with (OUT_DIR/"race_log.csv").open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(out[0].keys()) if out else ["race_id"])
        w.writeheader();w.writerows(out)
    (OUT_DIR/"README.md").write_text(
        "# 競輪将棋 1着コア2号機 v15\n\n"
        "- 1レース=1学習単位\n"
        "- 7人同時softmax\n"
        "- 8週間ローリング学習\n"
        "- 1着候補は1人固定ではなく確率構造から1〜3人\n"
        "- オッズ不使用・車番特徴不使用\n",
        encoding="utf-8"
    )
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=="__main__":
    main()
