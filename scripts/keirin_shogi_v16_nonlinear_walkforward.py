#!/usr/bin/env python3
from __future__ import annotations

import csv, json, math
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path
from statistics import mean, pstdev

BASES=[
    Path("data/2024/s_class_f1_all_parts/2024_q1"),
    Path("data/2024/s_class_f1_all_parts/2024_q2"),
    Path("data/2024/s_class_f1_all_parts/2024_q3"),
]
OUT_DIR = Path("results/keirin_shogi/v16_nonlinear_walkforward")
OUT_DIR.mkdir(parents=True, exist_ok=True)

START_TEST=date(2024,5,6)
END_TEST=date(2024,7,28)

# 事前に固定する「納得」の最低条件。単発ラッキーを避けるため2週連続を要求。
ACCEPT={
    "top1_min":0.30,
    "candidate_capture_min":0.65,
    "avg_candidates_max":2.20,
    "three_candidate_share_max":0.50,
    "consecutive_weeks":2,
}

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

def d(s): return date.fromisoformat(s)

def target_races(races,start,end):
    ss=start.isoformat(); ee=end.isoformat()
    return {r["race_id"]:r for r in races
            if ss<=r.get("race_date","")<=ee
            and ino(r.get("entry_count"))==7
            and r.get("meeting_grade")=="F1"
            and "Ｓ級" in r.get("race_type","")}

def zmap(vals):
    mu=mean(vals); sd=pstdev(vals) or 1.0
    return [(v-mu)/sd for v in vals]

def relu(x): return max(0.0,x)

def race_features(rows):
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

    raw={f:[num(r.get(f)) for r in rows] for f in BASE_FEATURES}
    z={f:zmap(v) for f,v in raw.items()}

    strengths=[]
    for r in rows:
        lid=(r.get("line_id") or "").strip() or f"solo_{r.get('car_no')}"
        strengths.append(line_stats[lid]["strength"])
    zls=zmap(strengths)

    out=[]
    for i,r in enumerate(rows):
        lid=(r.get("line_id") or "").strip() or f"solo_{r.get('car_no')}"
        ls=line_stats[lid]
        others=[v["strength"] for k,v in line_stats.items() if k!=lid]
        other_best=max(others) if others else ls["strength"]
        pos=ino(r.get("line_position"))
        adv=(ls["strength"]-other_best)/10.0
        gap=(num(r.get("score"))-ls["max_score"])/5.0
        attack=z["nige_count"][i]+z["makuri_count"][i]

        x={}
        # linear + piecewise nonlinear basis
        for f in BASE_FEATURES:
            v=z[f][i]
            x[f"z_{f}"]=v
            x[f"pos_{f}"]=relu(v)
            x[f"neg_{f}"]=relu(-v)
        x.update({
            "line_pos1":1.0 if pos==1 else 0.0,
            "line_pos2":1.0 if pos==2 else 0.0,
            "line_pos3p":1.0 if pos>=3 else 0.0,
            "line_size1":1.0 if ls["size"]==1 else 0.0,
            "line_size2":1.0 if ls["size"]==2 else 0.0,
            "line_size3p":1.0 if ls["size"]>=3 else 0.0,
            "z_line_strength":zls[i],
            "pos_line_strength":relu(zls[i]),
            "neg_line_strength":relu(-zls[i]),
            "gap_to_line_max":gap,
            "line_strength_vs_other":adv,
            # key interactions
            "score_x_pos2":z["score"][i]*(1.0 if pos==2 else 0.0),
            "sashi_x_pos2":z["sashi_count"][i]*(1.0 if pos==2 else 0.0),
            "attack_x_pos1":attack*(1.0 if pos==1 else 0.0),
            "b_x_pos1":z["b_count"][i]*(1.0 if pos==1 else 0.0),
            "makuri_x_line_adv":z["makuri_count"][i]*adv,
            "score_x_line_adv":z["score"][i]*adv,
            "sashi_x_gap":z["sashi_count"][i]*gap,
            "score_x_gap":z["score"][i]*gap,
            "makuri_x_score":z["makuri_count"][i]*z["score"][i],
            "sashi_x_score":z["sashi_count"][i]*z["score"][i],
        })
        out.append({"no":ino(r["car_no"]),"name":r.get("player_name",""),"x":x})
    return out

def build_dataset(rids, eb, rb):
    ds=[]
    for rid in rids:
        rows=eb.get(rid,[])
        if len(rows)!=7: continue
        wins=[x for x in rb.get(rid,[]) if ino(x.get("finish_position"))==1]
        if len(wins)!=1: continue
        winner=ino(wins[0]["car_no"])
        fs=race_features(rows)
        if winner in [a["no"] for a in fs]:
            ds.append((rid,fs,winner))
    return ds

def softmax(scores):
    m=max(scores)
    ex=[math.exp(max(-30,min(30,s-m))) for s in scores]
    z=sum(ex)
    return [v/z for v in ex]

def train(ds,epochs=500,lr=0.025,l2=0.003):
    names=sorted(ds[0][1][0]["x"].keys())
    w={f:0.0 for f in names}
    n=len(ds)
    for ep in range(epochs):
        g={f:0.0 for f in names}
        for _,fs,winner in ds:
            scores=[sum(w[f]*a["x"][f] for f in names) for a in fs]
            ps=softmax(scores)
            yi=next(i for i,a in enumerate(fs) if a["no"]==winner)
            for i,a in enumerate(fs):
                e=ps[i]-(1.0 if i==yi else 0.0)
                for f in names:g[f]+=e*a["x"][f]
        for f in names:
            g[f]=g[f]/n+l2*w[f]
            w[f]-=lr*g[f]
    return {"weights":w,"train_races":n}

def predict(fs,model):
    w=model["weights"]
    scores=[sum(w[f]*a["x"][f] for f in w) for a in fs]
    ps=softmax(scores)
    out=[{"no":a["no"],"name":a["name"],"prob":p,"logit":s} for a,p,s in zip(fs,ps,scores)]
    out.sort(key=lambda x:(-x["prob"],x["no"]))
    return out

def choose(pred,policy):
    p1=pred[0]["prob"]; p2=pred[1]["prob"]; gap=p1-p2
    if p1>=policy["p1"] and gap>=policy["gap"]:
        return [pred[0]["no"]]
    if p1+p2>=policy["p12"]:
        return [pred[0]["no"],pred[1]["no"]]
    return [pred[0]["no"],pred[1]["no"],pred[2]["no"]]

def learn_policy(inner_train, inner_val):
    m=train(inner_train)
    grid=[]
    for p1 in (0.32,0.36,0.40,0.44):
      for gap in (0.06,0.10,0.14,0.18):
       for p12 in (0.48,0.54,0.60,0.66):
        hits=0; counts=[]
        for _,fs,winner in inner_val:
            pred=predict(fs,m); c=choose(pred,{"p1":p1,"gap":gap,"p12":p12})
            hits+=int(winner in c); counts.append(len(c))
        if not counts: continue
        capture=hits/len(counts); avgk=mean(counts); three=sum(1 for k in counts if k==3)/len(counts)
        # penalize brute-force 3-candidate spread strongly
        objective=capture - 0.12*(avgk-1.0) - 0.10*three
        grid.append({"p1":p1,"gap":gap,"p12":p12,"capture":capture,"avg_candidates":avgk,"three_share":three,"objective":objective})
    grid.sort(key=lambda x:(x["objective"],x["capture"],-x["avg_candidates"]),reverse=True)
    return grid[0],grid[:10]

def metrics(rows):
    n=len(rows)
    if not n:return {}
    return {
        "races":n,
        "top1_hit_rate":sum(r["top1_hit"] for r in rows)/n,
        "candidate_capture_rate":sum(r["candidate_hit"] for r in rows)/n,
        "avg_candidates":mean(r["candidate_count"] for r in rows),
        "three_candidate_share":sum(1 for r in rows if r["candidate_count"]==3)/n,
        "candidate_count_distribution":{str(k):sum(1 for r in rows if r["candidate_count"]==k) for k in (1,2,3)}
    }

def passes(m):
    return (
        m["top1_hit_rate"]>=ACCEPT["top1_min"] and
        m["candidate_capture_rate"]>=ACCEPT["candidate_capture_min"] and
        m["avg_candidates"]<=ACCEPT["avg_candidates_max"] and
        m["three_candidate_share"]<=ACCEPT["three_candidate_share_max"]
    )

def main():
    races=load_all("races.csv"); entries=load_all("entries.csv"); results=load_all("results.csv")
    eb=defaultdict(list); rb=defaultdict(list)
    for e in entries: eb[e["race_id"]].append(e)
    for r in results: rb[r["race_id"]].append(r)

    weekly=[]
    streak=0
    first_checkpoint=None

    cur=START_TEST
    while cur<=END_TEST:
        te=cur+timedelta(days=6)
        tr_start=cur-timedelta(days=56)
        tr_end=cur-timedelta(days=1)
        inner_cut=cur-timedelta(days=14)
        inner_train_ids=target_races(races,tr_start,inner_cut-timedelta(days=1))
        inner_val_ids=target_races(races,inner_cut,tr_end)
        full_train_ids=target_races(races,tr_start,tr_end)
        test_ids=target_races(races,cur,te)

        inner_train=build_dataset(inner_train_ids.keys(),eb,rb)
        inner_val=build_dataset(inner_val_ids.keys(),eb,rb)
        full_train=build_dataset(full_train_ids.keys(),eb,rb)
        test_ds=build_dataset(test_ids.keys(),eb,rb)
        if min(len(inner_train),len(inner_val),len(full_train),len(test_ds))==0:
            cur+=timedelta(days=7); continue

        policy,policy_top=learn_policy(inner_train,inner_val)
        model=train(full_train)

        rows=[]; detail=[]
        for rid,fs,winner in test_ds:
            pred=predict(fs,model)
            c=choose(pred,policy)
            rows.append({
                "race_id":rid,
                "top1_hit":int(pred[0]["no"]==winner),
                "candidate_hit":int(winner in c),
                "candidate_count":len(c),
                "top1_prob":pred[0]["prob"],
                "top2_prob":pred[1]["prob"],
            })
            detail.append({"race_id":rid,"actual_1st":winner,"ranking":pred,"candidates":c})

        m=metrics(rows); ok=passes(m)
        streak=streak+1 if ok else 0
        rec={
            "test_period":{"start":cur.isoformat(),"end":te.isoformat()},
            "train_period":{"start":tr_start.isoformat(),"end":tr_end.isoformat()},
            "full_train_races":len(full_train),
            "inner_train_races":len(inner_train),
            "inner_val_races":len(inner_val),
            "policy":policy,
            "metrics":m,
            "passes_acceptance":ok,
            "acceptance_streak":streak,
        }
        weekly.append(rec)

        wkdir=OUT_DIR/f"{cur.isoformat()}_{te.isoformat()}"
        wkdir.mkdir(parents=True,exist_ok=True)
        (wkdir/"summary.json").write_text(json.dumps(rec,ensure_ascii=False,indent=2),encoding="utf-8")
        (wkdir/"detail.json").write_text(json.dumps(detail,ensure_ascii=False,indent=2),encoding="utf-8")

        if streak>=ACCEPT["consecutive_weeks"] and first_checkpoint is None:
            first_checkpoint={"week_index":len(weekly),"period":rec["test_period"]}

        cur+=timedelta(days=7)

    overall={
        "algorithm":"keirin_shogi_v16_nonlinear_walkforward",
        "acceptance_rule":ACCEPT,
        "weeks_tested":len(weekly),
        "first_two_week_checkpoint":first_checkpoint,
        "weekly":weekly,
        "method":"7人同時race-level softmaxを維持しつつ、各能力の正負piecewise basisとライン・脚質相互作用を追加。各週は直前8週間のみで再学習。候補数は内側2週間の未使用検証データで1/2/3人を動的決定し、3人乱用に罰則。単発の良週では止めず、事前条件を2週連続で満たした地点をチェックポイントとする。",
        "leakage_guard":"各週の予測はその週より前のデータだけで学習。候補ポリシーも各週直前のinner validationのみで決定。オッズ不使用。車番は特徴量不使用。"
    }
    (OUT_DIR/"summary.json").write_text(json.dumps(overall,ensure_ascii=False,indent=2),encoding="utf-8")
    (OUT_DIR/"README.md").write_text(
        "# 競輪将棋 v16 nonlinear walk-forward\n\n"
        "単発の当たり週を拾わない。事前に固定した基準を2週連続で満たすまで時系列検証。\n",
        encoding="utf-8"
    )
    print(json.dumps(overall,ensure_ascii=False,indent=2))

if __name__=="__main__":
    main()
