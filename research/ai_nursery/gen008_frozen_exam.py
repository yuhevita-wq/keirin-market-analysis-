"""Frozen out-of-sample GEN-008 two-branch exam: Oct 1-14, 2023; never train."""
from __future__ import annotations
import hashlib, json, math
from collections import defaultdict
from datetime import date,timedelta
import numpy as np
from train_week import ROOT,D
from profit_objective import ProfitPolicy
from gen008_experiment import model_digest,inherited_shadow_policy,score_papers
from parallel_gen008 import Branch,pure_pre_result,seal
from run_parallel_week import forward_compatible_probabilities,revealed_metric,summarize
from source_bridge import joined_races

BASE=ROOT/"research/ai_nursery/outputs/GEN-008-PARALLEL"
PARENT=BASE/"month-01-after-week02"
OUT=BASE/"frozen-exam-2023-10-01-to-14"
WEB=ROOT/"docs/ai-lab/gen008-frozen-exam-2023-10.json"
FIRST,LAST="2023-10-01","2023-10-14"
STAGE="GEN-008-PARALLEL-FROZEN-2023-10-01-14"
BRANCHES=("old","new")

def read(p):
    return json.loads(p.read_text(encoding="utf-8"))

def load_frozen():
    src=read(PARENT/"summary.json")
    assert src["status"]=="COMPLETE" and src["training_only"]
    assert src["dates"]==["2023-08-15","2023-09-14"]
    assert src["reserved_frozen_holdout"]==[FIRST,LAST]
    assert src["holdout_results_read_for_training"]==0
    models,snap={},{}
    for name in BRANCHES:
        cp=read(PARENT/name/"checkpoint.json")
        assert cp["branch"]==name and cp["stage"]==src["id"]
        assert cp["dates"]==src["dates"] and cp["cumulative_races_from_GEN007"]==6247
        w=np.asarray(cp["weights"],dtype=np.float64)
        assert w.shape==(D,) and model_digest(w)==cp["weights_sha256"]
        policy_path=PARENT/name/"behavior_policy.json"
        policy=(ProfitPolicy.inherit(policy_path) if name=="old"
                else inherited_shadow_policy(policy_path))
        assert policy.digest()==cp["policy_sha256"]
        assert cp["weights_sha256"]==src["closing_checkpoints"][name]["prediction"]
        assert cp["policy_sha256"]==src["closing_checkpoints"][name]["behavior"]
        temp=float(cp["temperature"])
        assert math.isclose(temp,src["closing_checkpoints"][name]["temperature"])
        models[name]=Branch(name,w,policy,temp)
        snap[name]={"prediction":model_digest(w),"behavior":policy.digest(),"temperature":temp}
    return models,snap

def run():
    if OUT.exists() or WEB.exists():
        raise RuntimeError("Frozen exam archive already exists; refusing overwrite")
    models,snap=load_frozen()
    rows,source=joined_races(2023,FIRST,LAST)
    if not rows:raise RuntimeError("No eligible holdout races with complete odds")
    ids=[r["race_id"] for r in rows]
    assert len(ids)==len(set(ids))
    assert all(FIRST<=r["race"]["race_date"]<=LAST for r in rows)
    assert all(r["race"]["race_date"]>"2023-09-14" for r in rows)
    OUT.mkdir(parents=True,exist_ok=False)
    results=[]
    daily=defaultdict(list)
    with (OUT/"sealed_pre_result_choices.jsonl").open("x",encoding="utf-8") as paper, \
         (OUT/"revealed_forward_results.jsonl").open("x",encoding="utf-8") as ledger:
        for i,row in enumerate(rows,1):
            pre=pure_pre_result(row)
            picks,payloads={},{}
            for name in BRANCHES:
                picks[name],payloads[name]=models[name].precommit(pre)
                assert picks[name]["prediction_before_sha256"]==snap[name]["prediction"]
                assert picks[name]["policy_before_sha256"]==snap[name]["behavior"]
                assert math.isclose(picks[name]["temperature_before"],snap[name]["temperature"])
                assert picks[name]["actual_paper_stake_yen"]==100*picks[name]["actual_paper_marks"]
            assert picks["new"]["shadow_is_not_spent"] and picks["new"]["shadow_page"] is not None
            seal(paper,{"stage":STAGE,"sequence":i,"date":pre["race"]["race_date"],
                        "race_id":pre["race_id"],"old":picks["old"],"new":picks["new"],
                        "before_outcome_read":True})
            outcome=row["outcome"]
            for name in BRANCHES:
                probs=forward_compatible_probabilities(payloads[name][0],payloads[name])
                loss=revealed_metric(outcome,probs)
                actual=score_papers(picks[name]["pages"],pre["odds"],outcome)
                assert actual["stake_yen"]==picks[name]["actual_paper_stake_yen"]
                shadow_net=None
                if name=="new" and not picks[name]["pages"]:
                    shadow_net=score_papers([picks[name]["shadow_page"]],pre["odds"],outcome)["net_yen"]
                record={"stage":STAGE,"sequence":i,"date":pre["race"]["race_date"],
                        "race_id":pre["race_id"],"branch":name,
                        "entered":bool(picks[name]["pages"]),"actual":actual,
                        **loss,"skipped_shadow_net_yen":shadow_net,
                        "shadow_counted_as_actual_cash":False,
                        "prediction_sha256":snap[name]["prediction"],
                        "policy_sha256":snap[name]["behavior"]}
                seal(ledger,record)
                results.append(record)
                daily[record["date"]].append(record)
            if i%100==0:
                print("GEN008_FROZEN_EXAM_PROGRESS",i,"/",len(rows),flush=True)
    ending={name:{"prediction":model_digest(m.weights),
                  "behavior":m.policy.digest(),"temperature":m.temperature}
            for name,m in models.items()}
    if ending!=snap:raise AssertionError("Frozen exam modified weights, policy or temperature")
    comparison=summarize(results)
    days=[]
    for i in range(14):
        day=(date.fromisoformat(FIRST)+timedelta(days=i)).isoformat()
        rr=daily[day]
        days.append({"date":day,"races":len(rr)//2,
                     "old_net_yen":sum(x["actual"]["net_yen"] for x in rr if x["branch"]=="old"),
                     "new_net_yen":sum(x["actual"]["net_yen"] for x in rr if x["branch"]=="new")})
    summary={"id":STAGE,"status":"COMPLETE","kind":"FROZEN_OUT_OF_SAMPLE",
             "source_stage":str(PARENT.relative_to(ROOT)),"exam_dates":[FIRST,LAST],
             "days":14,"races":len(rows),
             "race_ids_sha256":hashlib.sha256("\n".join(ids).encode()).hexdigest(),
             "frozen_start":snap,"frozen_end":ending,
             "weights_or_policy_updated":False,"no_training_on_exam":True,
             "no_real_bets":True,"historical_final_odds_only":True,
             "same_races_for_both":True,"decisions_sealed_before_result":True,
             "shadow_not_in_actual_cash":True,"comparison":comparison,
             "daily":days,"source_report":source,
             "limitation":"Only complete 3連単/3連複 historical FINAL odds; not pre-close quotes or real bets."}
    (OUT/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    WEB.parent.mkdir(parents=True,exist_ok=True)
    WEB.write_text(json.dumps({"id":STAGE,"status":"COMPLETE","exam_dates":[FIRST,LAST],
                               "races":len(rows),"comparison":comparison,"daily":days,
                               "source":str((OUT/"summary.json").relative_to(ROOT)),
                               "historical_final_odds_only":True},
                              ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("GEN008_FROZEN_EXAM_COMPLETE",
          json.dumps({"races":len(rows),"comparison":comparison},ensure_ascii=False),flush=True)

if __name__=="__main__":
    run()
