"""Creation-stage tests for parallel GEN-008; no historical training occurs."""
from __future__ import annotations
import hashlib
import math
from itertools import permutations,combinations

import numpy as np

from train_week import ROOT,D,feature_matrix
from behavior_policy import policy_context,BehaviorPolicy
from gen008_experiment import (
    coherent_probabilities,calibration_metrics,make_shadow_sheet,
    score_papers,update_temperature,inherited_shadow_policy,ShadowProfitPolicy
)
from parallel_gen008 import (
    Branch,load_both_parents,pure_pre_result,TRAIN_START,TRAIN_END,
    TEST_START,TEST_END,OUT
)


def fixture():
    cars=range(1,6)
    odds={
        "3連単":{"-".join(map(str,x)):23.0 for x in permutations(cars,3)},
        "3連複":{"=".join(map(str,x)):6.0 for x in combinations(cars,3)}
    }
    entries={i:{"car_no":str(i),"score":str(89+i),"age":str(25+i),
                "class":"A1"} for i in cars}
    return {"race_id":"9820230812010001","race":{"race_date":"2023-08-12"},
            "entries":entries,"odds":odds}


def test_probabilities():
    row=fixture()
    arms=feature_matrix(row["entries"],row["odds"])
    w=np.zeros(D)
    mix,marg=coherent_probabilities(arms,w)
    assert len(mix)==len(arms) and np.isclose(mix.sum(),1)
    assert np.isclose(sum(marg["3連単"].values()),1)
    assert np.isclose(sum(marg["3連複"].values()),1)
    assert np.isclose(sum(mix[i] for i,a in enumerate(arms) if a["market"]=="3連単"),.5)
    assert np.isclose(sum(mix[i] for i,a in enumerate(arms) if a["market"]=="3連複"),.5)
    for combo,t in marg["3連複"].items():
        vals=combo.split("=")
        exact=sum(p for selection,p in marg["3連単"].items()
                  if sorted(selection.split("-"))==vals)
        assert np.isclose(t,exact)
    assert np.isclose(marg["3連単"]["1-2-3"],1/60)
    assert np.isclose(marg["3連複"]["1=2=3"],1/10)
    q=calibration_metrics(marg,(3,1,2))
    assert np.isclose(q["trifecta_mass"],1)
    assert np.isclose(q["trio_mass"],1)
    assert q["trifecta_logloss"]>0 and q["trio_logloss"]>0


def test_shadow_before_finish():
    pre=fixture()
    assert "outcome" not in pre
    arms=feature_matrix(pre["entries"],pre["odds"])
    mix,_=coherent_probabilities(arms,np.zeros(D))
    a=make_shadow_sheet(arms,mix,pre["odds"],pre["entries"],pre["race_id"])
    b=make_shadow_sheet(arms,mix,pre["odds"],pre["entries"],pre["race_id"])
    assert a==b
    assert a["point_count"]==len(a["picks"])>=1
    assert a["total_yen"]==100*len(a["picks"])
    actual=score_papers([a],pre["odds"],(3,1,2))
    assert actual["net_yen"]==actual["payout_yen"]-actual["stake_yen"]
    assert actual["stake_yen"]==a["total_yen"]
    assert score_papers([],pre["odds"],(3,1,2))["net_yen"]==0
    # If result labels were mandatory for sheet generation, this would fail.
    assert pure_pre_result({**pre,"outcome":(4,5,1),"paid":9})==pre


def test_shadow_not_cash():
    pre=fixture()
    branch=Branch("new",np.zeros(D),ShadowProfitPolicy())
    branch.policy.weights["enter"][0]=-100.0
    before_hash=branch.policy.digest()
    sealed,payload=branch.precommit(pre)
    assert sealed["actual_paper_stake_yen"]==0
    assert sealed["pages"]==[]
    assert sealed["shadow_page"] is not None
    assert sealed["shadow_page"]["total_yen"]>0
    assert sealed["shadow_is_not_spent"] is True
    assert branch.policy.digest()==before_hash
    # Freeze before result; only now score a known finish.
    settlement=branch.settle_and_learn(pre,(3,1,2),payload,sealed)
    assert settlement["actual"]["net_yen"]==0
    assert settlement["actual"]["stake_yen"]==0
    assert settlement["shadow_missed_profit_if_skipped_yen"]==settlement["shadow_net_yen"]
    assert branch.policy.count["enter"]==1


def test_shadow_updates_even_when_skipped():
    p=ShadowProfitPolicy()
    from behavior_policy import Decision
    ctx=policy_context(np.array([.25,.25,.25,.25]),10000,0,0).tolist()
    no=Decision("enter",False,0.5,ctx)
    before=p.weights["enter"].copy()
    gain=p.update_enter_shadow(no,20000)
    assert gain["p_after"]>gain["p_before"]
    assert p.weights["enter"][0]>before[0]
    loser=ShadowProfitPolicy()
    loss=loser.update_enter_shadow(no,-1000)
    assert loss["p_after"]<loss["p_before"]
    assert loser.weights["enter"][0]<0
    assert loser.count["enter"]==1
    assert p.count["enter"]==1


def test_calibration_causal():
    row=fixture();arms=feature_matrix(row["entries"],row["odds"])
    weights=np.zeros(D)
    t=update_temperature(1.,arms,weights,(2,1,3))
    assert 0.5<=t<=4.0
    assert math.isfinite(t)
    try:coherent_probabilities(arms,weights,0.0)
    except ValueError:pass
    else:raise AssertionError("bad temperature accepted")


def test_identical_parent_frozen():
    original,w,old,new=load_both_parents()
    assert np.array_equal(w,np.array(original["weights"],dtype=np.float64))
    assert old.digest()==new.digest()==original["behavior_policy_sha256"]
    assert old.count==new.count
    assert old.update_count==new.update_count
    assert old.weights is not new.weights
    assert all(not np.shares_memory(old.weights[k],new.weights[k]) for k in old.weights)
    assert "2023-08-01"==TRAIN_START
    assert "2023-09-30"==TRAIN_END
    assert TEST_START=="2023-10-01" and TEST_END=="2023-10-14"
    # Construction only: no training checkpoint for GEN-008 yet.
    assert not (OUT/"summary.json").exists()


if __name__=="__main__":
    test_probabilities()
    test_shadow_before_finish()
    test_shadow_not_cash()
    test_shadow_updates_even_when_skipped()
    test_calibration_causal()
    test_identical_parent_frozen()
    print("GEN008_CREATION_TESTS_PASS_NO_TRAINING")
