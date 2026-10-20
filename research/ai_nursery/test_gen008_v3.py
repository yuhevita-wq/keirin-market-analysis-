"""Synthetic executable tests: V3 probability, no-bet and positive-edge behavior."""
from __future__ import annotations
import itertools
import numpy as np
from gen008_v3 import (blend,trio_probabilities,joint_portfolio,settle_v3,
                      matrix,score,calibrate)


def sample(rid="7720230915010001",date="2023-09-15",special=False):
    cars=tuple(range(1,6))
    tickets={"-".join(map(str,x)):3. for x in itertools.permutations(cars,3)}
    trio={"=".join(map(str,x)):2. for x in itertools.combinations(cars,3)}
    if special:
        tickets["1-2-3"]=1000.
    return {
        "race_id":rid,
        "race":{"race_date":date,"track":"fixture","race_no":1,"start_time":"10:00"},
        "entries":{c:{"car_no":str(c),"age":str(23+3*c),"class":"A1"} for c in cars},
        "odds":{"3連単":tickets,"3連複":trio},
        "outcome":(1,2,3),
    }


def main():
    weight=np.zeros(256,dtype=np.float64)
    cold=sample()
    keys,model,market=matrix(cold,weight,1.)
    assert len(keys)==60
    assert abs(sum(model)-1)<1e-12
    assert abs(sum(market)-1)<1e-12
    assert abs(sum(trio_probabilities(keys,blend(model,market,0.3)).values())-1)<1e-12
    q=joint_portfolio(cold,weight,1.,1.)
    assert not q["selected"]
    assert q["actual_paper_stake_yen"]==0
    assert settle_v3(q,(1,2,3))["net_yen"]==0

    rich=sample(special=True)
    p1=joint_portfolio(rich,weight,1.,1.)
    p2=joint_portfolio(rich,weight,1.,1.)
    assert p1==p2,"V3 must not have random decision behavior"
    assert p1["actual_paper_stake_yen"]>0
    assert any(t["type"]=="3連単" and t["selection"]=="1-2-3" for t in p1["selected"])
    assert settle_v3(p1,(1,2,3))["payout_yen"]>0
    assert settle_v3(p1,(1,2,4))["net_yen"]<=0
    p_fit=calibrate(
       [sample(rid="1111111111111111")],
       [sample(rid="2222222222222222",date="2023-09-23")],weight,1.)
    assert p_fit["train_races"]==p_fit["validation_races"]==1
    assert 0<=p_fit["alpha"]<=1
    print("GEN008_V3_SYNTHETIC_TESTS_PASS")


if __name__=="__main__":
    main()
