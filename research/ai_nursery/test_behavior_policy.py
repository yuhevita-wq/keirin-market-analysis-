"""Tests: action policy learns from its own actions and credit assignment is mark-local."""
from __future__ import annotations
import json
import tempfile
from pathlib import Path
import numpy as np
from behavior_policy import (
    BehaviorPolicy, Decision, policy_context, mark_credit_weights,
    D_BEHAVIOR, learn_from_race,
)

def main():
    rng=np.random.default_rng(728)
    context=policy_context(np.array([0.4,0.3,0.2,0.1]),9000,0,0)
    assert context.shape==(D_BEHAVIOR,)
    assert np.isfinite(context).all()

    good=BehaviorPolicy()
    bad=BehaviorPolicy()
    for i in range(400):
        g=good.decide("continue",context,rng,0)
        b=bad.decide("continue",context,rng,0)
        learn_from_race(good,[g],[5000 if g.action else 0],5000)
        learn_from_race(bad,[b],[-5000 if b.action else 0],-5000)
    assert good.weights["continue"]@context >0
    assert bad.weights["continue"]@context <0
    assert good.decide("continue",context,rng).probability>0.57
    assert bad.decide("continue",context,rng).probability<0.43
    assert good.update_count==bad.update_count==400

    # Independent enter/skip also learns; skip yields 0 actual reward.
    enter=BehaviorPolicy()
    for i in range(450):
        a=enter.decide("enter",context,rng)
        learn_from_race(enter,[a],[],-3500 if a.action else 0)
    assert enter.decide("enter",context,rng).probability<0.45
    assert enter.weights["continue"]@context==0

    arms=[
        {"market":"3連単","selection":"1-2-3","odds":100.0,"x":np.array([1.0,0.0,0.0])},
        {"market":"3連単","selection":"2-3-1","odds":5.0,"x":np.array([0.0,1.0,0.0])},
        {"market":"3連複","selection":"1=2=3","odds":7.0,"x":np.array([0.0,0.0,1.0])}
    ]
    probs=np.array([0.2,0.5,0.3])
    papers=[{"type":"3連単","picks":["1-2-3","2-3-1"]}]
    true={("3連単","1-2-3"),("3連複","1=2=3")}
    w,credit=mark_credit_weights(arms,probs,papers,true)
    assert len(credit)==2 and credit[0]["signed_feedback"]>0 and credit[1]["signed_feedback"]<0
    assert credit[0]["net_proxy_yen"]==9900 and credit[1]["net_proxy_yen"]==-100
    assert np.linalg.norm(w)>0

    with tempfile.TemporaryDirectory() as directory:
        p=Path(directory)/"policy.json"
        saved=good.save(p)
        clone=BehaviorPolicy.load(p)
        assert clone.digest()==saved["weights_sha256"]
        assert clone.update_count==good.update_count
        bad_data=json.loads(p.read_text(encoding="utf-8"))
        bad_data["weights"]["continue"][0]=999
        p.write_text(json.dumps(bad_data),encoding="utf-8")
        try:BehaviorPolicy.load(p)
        except ValueError:pass
        else:raise AssertionError("tampered learned behavior accepted")
    print("BEHAVIOR_POLICY_LEARNING_TEST_PASS",good.update_count,"train samples")

if __name__=="__main__":
    main()
