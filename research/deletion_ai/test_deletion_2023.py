"""Synthetic deletion-first 2023 tests (all markets, no hindsight)."""
import itertools
import math
import numpy as np

from deletion_core import (
    BRANCHES,MARKETS,COVERAGES,ACTIVE,UNIT,books,new_weights,
    probabilities,deletion_plan,full_sheet_score,evaluate_plan,update,model_digest
)
from run_reset_2023 import choose_coverage


def fixture(cars=tuple(range(1,8))):
    p3={"-".join(map(str,t)):max(1.5,60+7*(i%13)+i%11)
        for i,t in enumerate(itertools.permutations(cars,3))}
    p2={"=".join(map(str,t)):max(1.5,6+3*(i%9))
        for i,t in enumerate(itertools.combinations(cars,3))}
    return {"race_id":"9120230101010001",
            "entries":{c:{"car_no":str(c)} for c in cars},
            "race":{"race_date":"2023-01-01"},
            "odds":{"3連単":p3,"3連複":p2}}


def main():
    p=fixture()
    q=books(p)
    assert len(q["3連単"]["keys"])==210
    assert len(q["3連複"]["keys"])==35
    w=new_weights()
    assert len({model_digest(w,b) for b in BRANCHES})==1
    source={b:model_digest(w,b) for b in BRANCHES}
    for branch in BRANCHES:
        for market in MARKETS:
            prior=probabilities(q[market],w[branch][market],branch)
            assert np.allclose(prior,q[market]["prior"],atol=1e-13)
            assert abs(sum(prior)-1)<1e-12
            for target in COVERAGES:
                out=deletion_plan(q[market],prior,target)
                assert not out["deleted"],"Without learned edge no ticket may be deleted"
                assert len(out["kept"])==len(q[market]["keys"])
        before={b:model_digest(w,b) for b in BRANCHES}
        update(w,q,branch,(1,2,3))
        assert model_digest(w,branch)!=before[branch]
        assert all(model_digest(w,other)==before[other]
                   for other in BRANCHES if other!=branch)
    full=full_sheet_score(q,(1,2,3))
    assert full["total_marks"]==245
    assert full["winning_marks"]==2
    assert full["stake_yen"]==24500
    old3=probabilities(q["3連単"],w["OLD-R"]["3連単"],"OLD-R")
    old2=probabilities(q["3連複"],w["OLD-R"]["3連複"],"OLD-R")
    plans={m:deletion_plan(q[m],pred,0.6)
           for m,pred in (("3連単",old3),("3連複",old2))}
    settled=evaluate_plan(q,plans,(1,2,3))
    assert settled["saved_stake_yen"]==UNIT*settled["deleted_marks"]
    assert settled["delta_net_vs_all_marks"]==(
        settled["saved_stake_yen"]-settled["forfeited_payout_yen"])
    assert settled["stake_yen"]<=full["stake_yen"]
    assert settled["payout_yen"]<=full["payout_yen"]
    assert 0<=settled["deleted_winning_tickets"]<=2
    assert settled["retained_winning_tickets"]+settled["deleted_winning_tickets"]==2

    class Choices:pass
    d={"monthly":{},"totals":{}}
    for m in ("2023-10","2023-11","2023-12"):
        d["monthly"][m]={"full":{"all":{"stake_yen":1000.,"payout_yen":750.}}}
        for b in BRANCHES:
            d["monthly"][m][b]={}
            for target in COVERAGES:
                # Best target is deliberately 0.8, others consistently inferior.
                ratio=.85 if target==.8 else .65 if target<1 else .75
                d["monthly"][m][b][str(target)]={
                    "stake_yen":400.,"payout_yen":400.*ratio,
                }
    d["totals"]={"full":{"all":{"stake_yen":3000.,"payout_yen":2250.}}}
    for b in BRANCHES:
        d["totals"][b]={str(t):{"stake_yen":1200.,
                 "payout_yen":1200.*(.85 if t==.8 else .65 if t<1 else .75),
                 "deleted_marks":30.,"forfeited_payout_yen":0.,
                 "saved_stake_yen":500.} for t in COVERAGES}
    selected,details=choose_coverage(d)
    assert all(selected[b]==.8 for b in BRANCHES)
    assert all(details[b]["chosen"]["minimum_monthly_uplift_points"]>0
               for b in BRANCHES)
    print("GEN008_DELETION_SCRATCH_2023_UNIT_TEST_PASS",selected)


if __name__=="__main__":
    main()
