#!/usr/bin/env python3
from __future__ import annotations
import json, importlib.util
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

spec=importlib.util.spec_from_file_location("v23","scripts/keirin_shogi_v23_conditional_second_third.py")
v23=importlib.util.module_from_spec(spec); spec.loader.exec_module(v23)

OUT=Path("results/keirin_shogi/v24_second_diagnosis")
OUT.mkdir(parents=True,exist_ok=True)

def rank_of(ranking,no):
    for i,(n,p) in enumerate(ranking,1):
        if n==no:return i
    return None

def choose(ranking,t=0.45,maxk=3):
    return v23.choose_by_cum(ranking,t,maxk)

def rel_bucket(base,f,s):
    a=base[f];b=base[s]
    if a["line_id"]==b["line_id"]:
        if b["line_position"]>a["line_position"]:return "same_line_behind_first"
        if b["line_position"]<a["line_position"]:return "same_line_ahead_first"
        return "same_line_same_pos"
    return "different_line"

def main():
    vrows=v23.load_v19_details()
    v23.fit_v21_selector(vrows)
    v23.mark_v21_participation(vrows)
    race_by,eb,rb=v23.raw_maps()

    train_ids=v23.target_ids(race_by,v23.RANK_TRAIN_START,v23.RANK_TRAIN_END)
    train_ds=v23.make_training_races(train_ids,eb,rb)
    m2=v23.fit_conditional_model(train_ds,2)

    byrid={r["race_id"]:r for r in vrows}
    rows=[]
    for rid,r in byrid.items():
        es=eb.get(rid,[])
        if len(es)!=7:continue
        race_date=es[0].get("race_date","")
        if not(v23.TEST_START<=race_date<=v23.TEST_END):continue
        if not r.get("v21_participate"):continue
        order=v23.result_order(rb.get(rid,[]))
        if not order:continue
        f,s,t=order
        base=v23.race_base_features(es)

        fw=v23.first_weights(r)
        # current mixture over adopted first candidates
        p2=defaultdict(float)
        for af,w in fw.items():
            q=v23.conditional_probs(base,m2,af)
            for n,p in q.items():p2[n]+=w*p
        z=sum(p2.values()) or 1.0
        current=sorted(((n,p/z) for n,p in p2.items()),key=lambda kv:(-kv[1],kv[0]))

        # oracle: actual first known
        oq=v23.conditional_probs(base,m2,f)
        oracle=sorted(oq.items(),key=lambda kv:(-kv[1],kv[0]))

        # top1-first-only: force the highest-weight first candidate
        assumed=max(fw.items(),key=lambda kv:(kv[1],-kv[0]))[0]
        tq=v23.conditional_probs(base,m2,assumed)
        top1only=sorted(tq.items(),key=lambda kv:(-kv[1],kv[0]))

        fcands=list(map(v23.ino,r["candidates"]))
        current_c=choose(current)
        oracle_c=choose(oracle)
        top1_c=choose(top1only)

        rows.append({
            "race_id":rid,"race_date":race_date,
            "actual_first":f,"actual_second":s,
            "first_candidates":fcands,"first_candidate_count":len(fcands),
            "first_top1":assumed,
            "first_captured":int(f in fcands),
            "actual_second_in_first_candidates":int(s in fcands),
            "actual_second_equals_first_top1":int(s==assumed),
            "relation":rel_bucket(base,f,s),
            "second_line_position":base[s]["line_position"],
            "current_rank":rank_of(current,s),
            "oracle_rank":rank_of(oracle,s),
            "top1only_rank":rank_of(top1only,s),
            "current_hit":int(s in current_c),
            "oracle_hit":int(s in oracle_c),
            "top1only_hit":int(s in top1_c),
            "current_candidates":current_c,
            "oracle_candidates":oracle_c,
            "top1only_candidates":top1_c,
        })

    n=len(rows)
    def rate(key,subset=None):
        rr=rows if subset is None else subset
        return sum(x[key] for x in rr)/len(rr) if rr else 0.0

    rankdist=Counter(x["current_rank"] for x in rows)
    orankdist=Counter(x["oracle_rank"] for x in rows)

    topk={}
    for k in (1,2,3):
        topk[str(k)]=sum((x["current_rank"] or 99)<=k for x in rows)/n
    oracle_topk={}
    for k in (1,2,3):
        oracle_topk[str(k)]=sum((x["oracle_rank"] or 99)<=k for x in rows)/n

    by_first_count={}
    for k in (1,2):
        rr=[x for x in rows if x["first_candidate_count"]==k]
        by_first_count[str(k)]={
            "races":len(rr),
            "current_second_capture":rate("current_hit",rr),
            "oracle_second_capture":rate("oracle_hit",rr),
            "actual_second_in_first_candidates_rate":rate("actual_second_in_first_candidates",rr),
        }

    by_relation={}
    for rel in sorted(set(x["relation"] for x in rows)):
        rr=[x for x in rows if x["relation"]==rel]
        by_relation[rel]={
            "races":len(rr),
            "current_second_capture":rate("current_hit",rr),
            "oracle_second_capture":rate("oracle_hit",rr),
        }

    by_pos={}
    for pos in sorted(set(x["second_line_position"] for x in rows)):
        rr=[x for x in rows if x["second_line_position"]==pos]
        by_pos[str(pos)]={
            "races":len(rr),
            "current_second_capture":rate("current_hit",rr),
            "oracle_second_capture":rate("oracle_hit",rr),
        }

    conflict=[x for x in rows if x["actual_second_in_first_candidates"]]
    no_conflict=[x for x in rows if not x["actual_second_in_first_candidates"]]
    wrongfirst=[x for x in rows if not x["first_captured"]]
    rightfirst=[x for x in rows if x["first_captured"]]
    second_is_top1=[x for x in rows if x["actual_second_equals_first_top1"]]

    summary={
        "algorithm":"v24_second_diagnosis",
        "base":"v21 first frozen + v23 conditional second",
        "test_races":n,
        "headline":{
            "current_adaptive_second_capture":rate("current_hit"),
            "oracle_actual_first_second_capture":rate("oracle_hit"),
            "top1_first_only_second_capture":rate("top1only_hit"),
            "current_second_top1_rate":topk["1"],
            "current_second_top2_rate":topk["2"],
            "current_second_top3_rate":topk["3"],
            "oracle_second_top1_rate":oracle_topk["1"],
            "oracle_second_top2_rate":oracle_topk["2"],
            "oracle_second_top3_rate":oracle_topk["3"],
        },
        "error_propagation":{
            "first_captured_races":len(rightfirst),
            "second_capture_when_first_captured":rate("current_hit",rightfirst),
            "first_missed_races":len(wrongfirst),
            "second_capture_when_first_missed":rate("current_hit",wrongfirst),
        },
        "role_conflict":{
            "actual_second_in_first_candidates_races":len(conflict),
            "rate":len(conflict)/n,
            "second_capture_when_conflict":rate("current_hit",conflict),
            "second_capture_when_no_conflict":rate("current_hit",no_conflict),
            "actual_second_is_first_top1_races":len(second_is_top1),
            "second_capture_when_actual_second_is_first_top1":rate("current_hit",second_is_top1),
        },
        "by_first_candidate_count":by_first_count,
        "by_first_second_relation":by_relation,
        "by_actual_second_line_position":by_pos,
        "current_rank_distribution":{str(k):v for k,v in sorted(rankdist.items(),key=lambda kv:(kv[0] or 99))},
        "oracle_rank_distribution":{str(k):v for k,v in sorted(orankdist.items(),key=lambda kv:(kv[0] or 99))},
        "interpretation_guard":"Oracle comparison uses the actual first only for diagnosis, never as a deployable feature."
    }
    (OUT/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    (OUT/"detail.json").write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=="__main__":main()
