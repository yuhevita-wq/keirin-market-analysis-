"""Unit checks for 100-yen exhaustive dual-market paper sheets and learning."""
from __future__ import annotations
import itertools,math
import numpy as np
from gen008_all_marks_experiment import (
    all_marks,paper_settlement,shadow_settlement,positive_shadow,
    gradient_full_sheet,revised_prior,metrics,UNIT
)


def fixture(odds=9.):
    cars=(1,2,3,4,5)
    triples=tuple("-".join(map(str,a)) for a in itertools.permutations(cars,3))
    trios=tuple("=".join(map(str,a)) for a in itertools.combinations(cars,3))
    trio_to_index={x:i for i,x in enumerate(trios)}
    group=np.array([trio_to_index["=".join(sorted(k.split("-"),key=int))] for k in triples])
    p={"race_id":"1120240201010001","entries":{c:{"car_no":str(c)} for c in cars},
       "race":{"race_date":"2024-02-01"},
       "odds":{"3連単":{k:odds for k in triples},
               "3連複":{k:2.0 for k in trios}}}
    x=np.stack([np.array([i/200.,math.sin(i/7),i%3,i%5,i%7],dtype=float)
                for i in range(len(triples))])
    feat={"keys":triples,"trio_names":trios,"trio_group":group,"x":x}
    return p,feat


def main():
    p,f=fixture()
    paper=all_marks(p,f)
    assert paper["marks"]==70
    assert paper["virtual_stake_yen"]==7000
    assert len(paper["markets"]["3連単"])==60
    assert len(paper["markets"]["3連複"])==10
    actual=paper_settlement(p,f,(1,2,3))
    assert actual["winning_marks"]==2 and actual["losing_marks"]==68
    assert actual["3連単_paid_yen"]==900
    assert actual["3連複_paid_yen"]==200
    assert actual["net_yen"]==1100-7000
    assert shadow_settlement({"3連単":[],"3連複":[]},(1,2,3))["net_yen"]==0
    q=np.ones(60)/60
    grad=gradient_full_sheet(f,q,0)
    assert len(grad)==5 and np.all(np.isfinite(grad))
    grad_other=gradient_full_sheet(f,q,1)
    assert not np.allclose(grad,grad_other)
    learned=revised_prior(q,f["x"],.018*grad)
    assert np.isclose(learned.sum(),1) and np.all(learned>0)
    assert metrics(f,q,0)["3連単_logloss"]>0
    positive=positive_shadow(p,f,q)
    assert len(positive["3連単"])==0
    assert len(positive["3連複"])==0
    print("GEN008_ALL_MARKS_UNIT_TESTS_PASS")


if __name__=="__main__":main()
