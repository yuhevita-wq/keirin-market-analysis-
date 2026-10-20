"""Net-profit is the only objective. No-betting and funded wallets are not wins."""
from __future__ import annotations
import tempfile
from pathlib import Path
import numpy as np
from behavior_policy import BehaviorPolicy,Decision
from profit_objective import ProfitPolicy,net_profit_yen,successful,OBJECTIVE,signed_ticket_updates

def main():
    assert OBJECTIVE=="maximize_total_net_yen"
    assert net_profit_yen(5100,1200)==3900
    assert net_profit_yen(0,1200)==-1200
    assert net_profit_yen(0,0)==0
    assert successful(1,100)
    assert not successful(0,0)
    assert not successful(0,100)
    assert not successful(-1,100)
    assert not successful(100,0)
    c=np.array([1.,1.,0.2,0.8,0.,0.,0.6])
    seed=BehaviorPolicy()
    seed.weights["enter"]=np.full(7,0.05)
    seed.weights["continue"]=np.full(7,-0.03)
    seed.count={"enter":937,"continue":764}
    seed.update_count=1701
    with tempfile.TemporaryDirectory() as temp:
        f=Path(temp)/"parent_behavior.json"
        seed.save(f)
        child=ProfitPolicy.inherit(f)
        assert child.digest()==seed.digest()
        assert child.update_count==1701
        assert child.count==seed.count
        d=Decision("enter",True,0.5,c.tolist())
        before=child.weights["enter"]@c
        positive=child.update(d,80000)
        assert positive["uncompressed_net_signal"]==8.0
        assert positive["objective"]==OBJECTIVE
        assert positive["update_l2"]<=0.35000001
        assert child.weights["enter"]@c>before
        first=child.digest()
        for i in range(10):
            child.update(Decision("enter",True,0.5,c.tolist()),-3500)
        assert child.digest()!=first
        checkfile=Path(temp)/"child.json"
        child.save(checkfile)
        assert BehaviorPolicy.load(checkfile).digest()==child.digest()
    X=np.eye(3)
    arms=[{"market":"3連単","selection":"1-2-3","odds":15.0,"x":X[0]},
          {"market":"3連単","selection":"2-3-1","odds":4.0,"x":X[1]},
          {"market":"3連複","selection":"1=2=3","odds":3.0,"x":X[2]}]
    pages=[{"type":"3連単","picks":["1-2-3","2-3-1"]}]
    grad,hist=signed_ticket_updates(arms,np.array([0.2,0.5,0.3]),pages,{("3連単","1-2-3")})
    assert len(hist)==2
    assert hist[0]["net_proxy_yen"]==1400 and hist[0]["signed_feedback"]>0
    assert hist[1]["net_proxy_yen"]==-100 and hist[1]["signed_feedback"]<0
    assert np.linalg.norm(grad)<=0.10000001
    print("NET_PROFIT_OBJECTIVE_TEST_PASS",OBJECTIVE)

if __name__=="__main__":
    main()
