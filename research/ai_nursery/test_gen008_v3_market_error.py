"""Synthetic market-error tests. Real candidate must still pass independent time split."""
from __future__ import annotations
import itertools
import math
import numpy as np

from gen008_v3_market_error import (market_features,corrected_probability,
          loss_and_grad,portfolio,settle,fit,FEATURE_NAMES,logloss_report)


def race(i=1,day="2023-09-16",payout=20.):
    cars=tuple(range(1,6))
    ticket={ "-".join(map(str,p)):60. for p in itertools.permutations(cars,3)}
    trio={ "=".join(map(str,p)):10. for p in itertools.combinations(cars,3)}
    ticket["1-2-3"]=payout
    return {"race_id":str(i).zfill(16),
            "race":{"race_date":day,"track":"test","start_time":"10:00","race_no":1},
            "entries":{c:{"car_no":str(c),"age":str(c+22),"class":"A1"} for c in cars},
            "odds":{"3連単":ticket,"3連複":trio},"outcome":(1,2,3)}


def test():
    weights=np.zeros(256)
    fixture=race()
    f=market_features(fixture,weights,1.)
    assert f["x"].shape==(60,len(FEATURE_NAMES))
    assert len(f["trio_names"])==10
    prior=corrected_probability(f,np.zeros(len(FEATURE_NAMES)))
    assert np.allclose(prior,f["baseline"])
    assert abs(sum(prior)-1)<1e-12
    data=[(f,0)]
    v,grad=loss_and_grad(np.zeros(len(FEATURE_NAMES)),data)
    assert np.isfinite(v) and np.all(np.isfinite(grad))
    for j in range(len(grad)):
        test=np.zeros(len(grad));test[j]=1e-6
        vp=loss_and_grad(test,data)[0]
        vm=loss_and_grad(-test,data)[0]
        assert abs((vp-vm)/2e-6-grad[j])<1e-5,(j,grad[j],(vp-vm)/2e-6)
    fitted=fit([race(i) for i in range(1,111)],weights,1.)
    assert fitted["train_races"]==110
    assert len(fitted["coefficients"])==len(FEATURE_NAMES)
    item=portfolio(fixture,f,prior)
    assert item==portfolio(fixture,f,prior)
    assert not item["selected"],"Uniform fair odds offers no positive growth"
    assert settle(item,(1,2,3))["net_yen"]==0
    rich=race(payout=600.)
    ff=market_features(rich,weights,1.)
    p=corrected_probability(ff,np.zeros(len(FEATURE_NAMES)))
    chosen=portfolio(rich,ff,p)
    assert isinstance(chosen["selected"],list)
    assert all(x["stake_yen"]%100==0 for x in chosen["selected"])
    assert chosen["stake_yen"]<=10000
    assert settle(chosen,(1,2,3))["stake_yen"]==chosen["stake_yen"]
    print("GEN008_V3_MARKET_ERROR_SYNTHETIC_PASS",fitted["coefficients"])


if __name__=="__main__":
    test()
