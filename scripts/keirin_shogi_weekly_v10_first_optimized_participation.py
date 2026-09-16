#!/usr/bin/env python3
from __future__ import annotations

import csv, json, math, importlib.util
from collections import defaultdict
from pathlib import Path
from statistics import mean

BASE = Path("data/2024/s_class_f1_all_parts/2024_q1")
TRAIN_START, TRAIN_END = "2024-03-18", "2024-03-24"
TEST_START, TEST_END = "2024-03-25", "2024-03-31"
PRIOR_DETAIL = Path("results/keirin_shogi/v9_inverted_participation/2024-03-18_2024-03-24/detail.json")
OUT_DIR = Path("results/keirin_shogi/v10_first_optimized_participation/2024-03-25_2024-03-31")
OUT_DIR.mkdir(parents=True, exist_ok=True)

spec7 = importlib.util.spec_from_file_location("v7", "scripts/keirin_shogi_weekly_v7_conditional_second.py")
v7 = importlib.util.module_from_spec(spec7); spec7.loader.exec_module(v7)

spec8 = importlib.util.spec_from_file_location("v8", "scripts/keirin_shogi_weekly_v8_participation.py")
v8 = importlib.util.module_from_spec(spec8); spec8.loader.exec_module(v8)

V8_FIRST_GAP = 22.574611055774504
V8_LINE_SPREAD = 10.73500000000001

def load_csv(name):
    with (BASE/name).open("r",encoding="utf-8-sig",newline="") as f:
        return list(csv.DictReader(f))

def ino(v):
    try:return int(float(v))
    except:return 0

def v9_eligible(features):
    return not (
        features["first_gap"] >= V8_FIRST_GAP
        and features["line_strength_spread"] >= V8_LINE_SPREAD
    )

def ranked_nos(scored):
    return [x["no"] for x in sorted(scored,key=lambda x:(-x["score"],x["no"]))]

def first_structure_features(first_sc, participation_features):
    vals=sorted([float(x["score"]) for x in first_sc], reverse=True)
    return {
        "first_gap12": (vals[0]-vals[1]) if len(vals)>=2 else 0.0,
        "first_gap13": (vals[0]-vals[2]) if len(vals)>=3 else 0.0,
        "first_sd": float(participation_features["first_sd"]),
        "line_strength_spread": float(participation_features["line_strength_spread"]),
    }

def quantiles(vals):
    s=sorted(vals)
    if not s:return []
    def q(p):
        i=(len(s)-1)*p
        lo=int(math.floor(i)); hi=int(math.ceil(i))
        if lo==hi:return s[lo]
        return s[lo]*(hi-i)+s[hi]*(i-lo)
    return sorted(set([q(.25),q(.5),q(.75)]))

def cond_ok(feat, cond):
    if cond is None:return True
    f,op,t=cond
    if op==">=":return feat[f]>=t
    return feat[f]<=t

def learn_first_policy():
    detail=json.loads(PRIOR_DETAIL.read_text(encoding="utf-8"))
    rows=[]
    for d in detail:
        pf=d["participation_features"]
        if not v9_eligible(pf):
            continue
        first_sc=d["first_scores"]
        feats=first_structure_features(first_sc,pf)
        actual=int(d["actual"]["1"] if "1" in d["actual"] else d["actual"][1])
        order=ranked_nos(first_sc)
        rows.append({"features":feats,"actual":actual,"order":order})

    if not rows:
        raise RuntimeError("No prior v9-eligible races for policy learning.")

    candidates=[None]
    for f in ["first_gap12","first_gap13","first_sd","line_strength_spread"]:
        vals=[r["features"][f] for r in rows]
        for t in quantiles(vals):
            candidates.append((f,">=",t))
            candidates.append((f,"<=",t))

    scored=[]
    n=len(rows)
    for cond in candidates:
        selected=[r for r in rows if cond_ok(r["features"],cond)]
        if len(selected)<15 or len(selected)/n<0.25:
            continue
        coverage=len(selected)/n
        for k in (1,2,3):
            hit=sum(1 for r in selected if r["actual"] in r["order"][:k])/len(selected)
            efficiency=hit/math.sqrt(k)
            objective=efficiency*math.sqrt(coverage)
            scored.append({
                "condition":cond,
                "k":k,
                "train_selected":len(selected),
                "train_coverage_within_v9":coverage,
                "train_first_hit_rate":hit,
                "train_efficiency":efficiency,
                "objective":objective,
            })
    scored.sort(key=lambda x:(x["objective"],x["train_first_hit_rate"],-x["k"],x["train_coverage_within_v9"]),reverse=True)
    return scored[0], rows

def summarize(rows):
    n=len(rows)
    if not n:
        return {"races":0,"first_hit_rate":0,"second_hit_rate":0,"third_hit_rate":0,
                "complete_capture_rate":0,"avg_candidate_cells":0}
    return {
        "races":n,
        "first_hit_rate":sum(r["hit_1st"] for r in rows)/n,
        "second_hit_rate":sum(r["hit_2nd"] for r in rows)/n,
        "third_hit_rate":sum(r["hit_3rd"] for r in rows)/n,
        "complete_capture_rate":sum(r["complete_capture"] for r in rows)/n,
        "avg_candidate_cells":mean(r["candidate_cells"] for r in rows),
    }

def main():
    policy, policy_rows = learn_first_policy()

    races=load_csv("races.csv"); entries=load_csv("entries.csv"); results=load_csv("results.csv")
    train=v7.get_target_races(races,TRAIN_START,TRAIN_END)
    test=v7.get_target_races(races,TEST_START,TEST_END)
    needed=set(train)|set(test)

    eb=defaultdict(list); rb=defaultdict(list)
    for e in entries:
        if e["race_id"] in needed: eb[e["race_id"]].append(e)
    for r in results:
        if r["race_id"] in needed: rb[r["race_id"]].append(r)

    models=v7.build_models(train.keys(),eb,rb)
    cond2=v7.build_conditional_second_model(train.keys(),eb,rb)

    out=[]; detail=[]; baseline=[]
    learned_cond=policy["condition"]; k=policy["k"]

    for rid,race in sorted(test.items(),key=lambda kv:(kv[1]["race_date"],kv[1]["track"],ino(kv[1]["race_no"]))):
        er=eb.get(rid,[])
        if len(er)!=7:continue

        actual={}; ok=True
        for p in (1,2,3):
            xs=[x for x in rb.get(rid,[]) if ino(x.get("finish_position"))==p]
            if len(xs)!=1:
                ok=False; break
            actual[p]=ino(xs[0]["car_no"])
        if not ok:continue

        first_sc=v7.score_rows(er,models["first"],"first")
        first_order=ranked_nos(first_sc)
        first_c=sorted(first_order[:k])

        # v9 baseline uses original v7 first-candidate count rule
        first_c_baseline=v7.choose(first_sc,"first")

        second_base=v7.score_rows(er,models["second"],"second")
        second_sc=v7.conditional_second_scores(er,first_sc,first_c,second_base,cond2)
        second_c=v7.choose(second_sc,"second")

        second_sc_baseline=v7.conditional_second_scores(er,first_sc,first_c_baseline,second_base,cond2)
        second_c_baseline=v7.choose(second_sc_baseline,"second")

        third_sc=v7.score_rows(er,models["third"],"third")
        third_c=v7.choose(third_sc,"third")

        board={"first":first_c,"second":second_c,"third":third_c}
        board_baseline={"first":first_c_baseline,"second":second_c_baseline,"third":third_c}

        pf=v8.participation_features(first_sc,second_sc,third_sc,board,er)
        ff=first_structure_features(first_sc,pf)

        base_enter=v9_eligible(pf)
        first_ok=cond_ok(ff,learned_cond)
        participate=base_enter and first_ok

        hits={"first":actual[1] in first_c,"second":actual[2] in second_c,"third":actual[3] in third_c}
        complete=all(hits.values())

        rec={
            "race_id":rid,"race_date":race["race_date"],"track":race["track"],"race_no":race["race_no"],
            "participate":int(participate),
            "v9_eligible":int(base_enter),
            "first_structure_ok":int(first_ok),
            "first_candidates":"-".join(map(str,first_c)),
            "second_candidates":"-".join(map(str,second_c)),
            "third_candidates":"-".join(map(str,third_c)),
            "actual_1st":actual[1],"actual_2nd":actual[2],"actual_3rd":actual[3],
            "hit_1st":int(hits["first"]),"hit_2nd":int(hits["second"]),"hit_3rd":int(hits["third"]),
            "complete_capture":int(complete),
            "candidate_cells":len(first_c)+len(second_c)+len(third_c),
            "first_gap12":ff["first_gap12"],"first_gap13":ff["first_gap13"],
            "first_sd":ff["first_sd"],"line_strength_spread":ff["line_strength_spread"],
        }
        out.append(rec)

        bhits={
            "first":actual[1] in first_c_baseline,
            "second":actual[2] in second_c_baseline,
            "third":actual[3] in third_c,
        }
        baseline.append({
            "race_id":rid,
            "participate":int(base_enter),
            "hit_1st":int(bhits["first"]),"hit_2nd":int(bhits["second"]),"hit_3rd":int(bhits["third"]),
            "complete_capture":int(all(bhits.values())),
            "candidate_cells":len(first_c_baseline)+len(second_c_baseline)+len(third_c),
        })

        detail.append({
            "race":race,"board":board,"actual":actual,"hits":hits,
            "participate":participate,"v9_eligible":base_enter,"first_structure_ok":first_ok,
            "first_policy":policy,"first_structure_features":ff,
            "first_scores":first_sc,"second_scores":second_sc,"third_scores":third_sc,
        })

    participated=[r for r in out if r["participate"]]
    skipped=[r for r in out if not r["participate"]]
    baseline_part=[r for r in baseline if r["participate"]]

    summary={
        "algorithm":"keirin_shogi_v10_first_optimized_participation",
        "train_period":{"start":TRAIN_START,"end":TRAIN_END},
        "test_period":{"start":TEST_START,"end":TEST_END},
        "first_policy_learning_source":"2024-03-18〜2024-03-24 v9固定検証のv9参加対象レース",
        "first_policy":{
            "condition":None if learned_cond is None else {"feature":learned_cond[0],"operator":learned_cond[1],"threshold":learned_cond[2]},
            "first_candidate_count":k,
            "train_selected_races":policy["train_selected"],
            "train_coverage_within_v9":policy["train_coverage_within_v9"],
            "train_first_hit_rate":policy["train_first_hit_rate"],
            "train_efficiency":policy["train_efficiency"],
            "objective":policy["objective"],
        },
        "participation_rule":"v9反転参加条件 AND 1着構造条件",
        "all_races":summarize(out),
        "participated_races":summarize(participated),
        "skipped_races":summarize(skipped),
        "participation_rate":len(participated)/len(out) if out else 0,
        "same_week_v9_baseline_participated":summarize(baseline_part),
        "same_week_v9_baseline_participation_rate":len(baseline_part)/len(baseline) if baseline else 0,
        "method":"v9の反転参加思想を維持し、直前v9参加レースから1着構造(first_gap12/13, first_sd, line_strength_spread)と必要1着候補数k=1..3を同時最適化。目的関数は1着捕捉率/sqrt(k)×sqrt(参加カバレッジ)。次週では条件とkを固定し、参加判定にも同じ1着最適化を使用。",
        "leakage_guard":"2024-03-25以降の結果は1着方針・参加判定・予測学習に未使用。オッズ不使用。"
    }

    (OUT_DIR/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    (OUT_DIR/"model.json").write_text(json.dumps({"first_policy":summary["first_policy"],"conditional_second":cond2},ensure_ascii=False,indent=2),encoding="utf-8")
    (OUT_DIR/"detail.json").write_text(json.dumps(detail,ensure_ascii=False,indent=2),encoding="utf-8")
    with (OUT_DIR/"race_log.csv").open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(out[0].keys()) if out else ["race_id"])
        w.writeheader();w.writerows(out)
    (OUT_DIR/"README.md").write_text(
        "# 競輪将棋 v10 1着最適化＋参加判定\n\n"
        "- v9反転参加条件を維持\n"
        "- 1着の候補構造と候補数を同時最適化\n"
        "- 同じ1着最適化を参加判定にも使用\n"
        "- オッズ不使用\n"
        "- 2024-03-25〜03-31で固定検証\n",
        encoding="utf-8"
    )
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=="__main__":
    main()
