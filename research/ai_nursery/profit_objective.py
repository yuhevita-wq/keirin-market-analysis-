"""GEN-007 and later: single monetary objective, cumulative pure paper NET profit.

Net profit = cash-equivalent payout - stake. NO contribution, wallet, reward
vault, refill, stress, hit-rate, or ROI term enters the optimization reward.
A zero-mark agent obtains net=0 and never passes the strict positive-profit
success condition. Historical fixed checkpoints remain unchanged.
"""
from __future__ import annotations
import math
import numpy as np
from behavior_policy import BehaviorPolicy,BASELINE_RATE,LEARNING_RATE,YEN_UNIT


OBJECTIVE="maximize_total_net_yen"
SUCCESS_THRESHOLD_YEN=0
MAX_POLICY_STEP_NORM=0.35
MAX_TICKET_STEP_NORM=0.1
TICKET_LR=0.008

def net_profit_yen(payout_yen,stake_yen):
    payout=float(payout_yen);stake=float(stake_yen)
    if not math.isfinite(payout) or not math.isfinite(stake) or payout<0 or stake<0:
        raise ValueError("Invalid paper stake or payout")
    return payout-stake

def successful(total_net_yen,total_stake_yen):
    """Strictly-positive net AND at least one actual mark. Skips alone fail."""
    return total_stake_yen>0 and total_net_yen>0

class ProfitPolicy(BehaviorPolicy):
    """Continue from GEN-006 policy and apply signed uncompressed yen utility.

    Numerical clipping applies ONLY to the gradient vector, never to the
    objective, realized gains or the relative monetary preference.
    """
    @classmethod
    def inherit(cls,path):
        old=BehaviorPolicy.load(path)
        obj=cls()
        obj.weights={k:v.copy() for k,v in old.weights.items()}
        obj.baseline=dict(old.baseline)
        obj.count=dict(old.count)
        obj.update_count=old.update_count
        return obj

    def update(self,decision,realized_incremental_net_yen):
        if decision.kind not in self.weights:
            raise ValueError("Unknown decision class")
        if not math.isfinite(realized_incremental_net_yen):
            raise ValueError("Nonfinite net outcome")
        # Raw yen objective, measured in 10,000-yen units for step-size only.
        signal=float(realized_incremental_net_yen)/YEN_UNIT
        kind=decision.kind
        advantage=signal-self.baseline[kind]
        vector=np.asarray(decision.context,dtype=float)
        raw=LEARNING_RATE*(float(decision.action)-decision.probability)*advantage*vector
        norm=float(np.linalg.norm(raw))
        gain=min(1.0,MAX_POLICY_STEP_NORM/norm) if norm>0 else 1.0
        delta=raw*gain
        self.weights[kind]+=delta
        self.baseline[kind]=(1-BASELINE_RATE)*self.baseline[kind]+BASELINE_RATE*signal
        self.count[kind]+=1
        self.update_count+=1
        return {
            "kind":kind,"action_taken":bool(decision.action),
            "realized_incremental_net_yen":int(round(realized_incremental_net_yen)),
            "scaled_reward":signal,"uncompressed_net_signal":signal,
            "advantage":advantage,"update_l2":float(np.linalg.norm(delta)),
            "new_action_probability":float(1/(1+math.exp(-float(np.clip(self.weights[kind]@vector,-30,30))))),
            "new_baseline":self.baseline[kind],
            "objective":OBJECTIVE
        }

def signed_ticket_updates(arms,probabilities,pages,winners):
    """Attribute actual signed profits/losses to each purchased mark."""
    if not pages:
        return np.zeros(len(arms[0]["x"]),dtype=np.float64),[]
    X=np.stack([a["x"] for a in arms])
    p=np.asarray(probabilities,dtype=float)
    average=p@X
    lookup={(a["market"],a["selection"]):i for i,a in enumerate(arms)}
    grad=np.zeros(X.shape[1],dtype=float)
    journal=[]
    for page in pages:
        for t in page["picks"]:
            i=lookup[(page["type"],t)]
            payout=round(float(arms[i]["odds"])*100) if (page["type"],t) in winners else 0
            net=payout-100
            # Do not reward losing picks in winning bundles.
            grad+=(float(net)/YEN_UNIT)*(X[i]-average)*TICKET_LR
            journal.append({
                "type":page["type"],"selection":t,"stake_yen":100,
                "payout_proxy_yen":int(payout),"net_proxy_yen":int(net),
                "signed_feedback":float(net)/YEN_UNIT
            })
    norm=float(np.linalg.norm(grad))
    if norm>MAX_TICKET_STEP_NORM:
        grad*=MAX_TICKET_STEP_NORM/norm
    return grad,journal
