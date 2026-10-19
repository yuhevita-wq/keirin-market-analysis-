"""Generic learnable enter/skip and paper stop/continue policy for NEXT generations.

NO keirin tactics, formations, favorites, market-edge rules or page limits.
The actual trained behavior depends only on past PRE-decision state and POST-result
incremental monetary feedback. Does not rewrite completed GEN-001..005.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

D_BEHAVIOR=7
LEARNING_RATE=0.06
BASELINE_RATE=0.03
TICKET_CREDIT_LR=0.008
YEN_UNIT=10000.0
EPS=1e-7


def sigmoid(t):
    t=float(np.clip(t,-30,30))
    return 1/(1+math.exp(-t))


def policy_context(probs,bank_yen,paper_count,spent_yen):
    """Only racecard/odds-derived uncertainty and presently known wallet/papers."""
    p=np.asarray(probs,dtype=np.float64)
    if p.ndim!=1 or len(p)<2 or not np.all(np.isfinite(p)):
        raise ValueError("bad pre-decision probabilities")
    p=p/p.sum()
    entropy=-float(np.sum(p*np.log(np.maximum(p,1e-30))))
    norm=entropy/math.log(len(p))
    context=np.array([
        1.0,
        min(float(bank_yen)/10000.0,5.0),
        float(p.max()),
        norm,
        math.log1p(max(0,paper_count))/6.0,
        math.log1p(max(0,spent_yen)/100.0)/10.0,
        math.log(len(p))/10.0
    ],dtype=np.float64)
    assert np.all(np.isfinite(context))
    return context


def mark_credit_weights(arms,probs,papers,correct):
    """Per-mark realized gains: winning marks gain odds*stake; losing marks lose stake.

    This differs from old reward signal, which rewarded ALL chosen tickets whenever
    aggregate wallet went over 10k. All marks may overlap across sheets; each
    physical purchased mark is an individual exposure and counted once.
    Returns vector for ticket-wise REINFORCE, without needing race-result peek
    before the mark sheet is sealed.
    """
    X=np.stack([a["x"] for a in arms])
    p=np.asarray(probs,dtype=float)
    expected=p@X
    index={(a["market"],a["selection"]):i for i,a in enumerate(arms)}
    credit=np.zeros(X.shape[1],dtype=float)
    outcomes=[]
    for paper in papers:
        for item in paper["picks"]:
            idx=index[(paper["type"],item)]
            payout=float(arms[idx]["odds"])*100.0 if (paper["type"],item) in correct else 0.
            net=payout-100.0
            # Symmetric bounded signal; no positive reinforcement of losing marks.
            scaled=float(np.tanh(net/YEN_UNIT))
            credit+=TICKET_CREDIT_LR*scaled*(X[idx]-expected)
            outcomes.append({
                "type":paper["type"],"selection":item,"stake_yen":100,
                "payout_proxy_yen":int(round(payout)),
                "net_proxy_yen":int(round(net)),
                "signed_feedback":scaled,
            })
    return credit,outcomes


@dataclass
class Decision:
    kind:str
    action:bool
    probability:float
    context:list[float]
    sheet_index:int | None=None

    def json(self):
        return {
            "kind":self.kind,"action":"continue" if self.action else "stop",
            "probability_of_continue":self.probability,
            "context":self.context,"sheet_index":self.sheet_index
        }


class BehaviorPolicy:
    def __init__(self):
        self.weights={
            "enter":np.zeros(D_BEHAVIOR,dtype=np.float64),
            "continue":np.zeros(D_BEHAVIOR,dtype=np.float64)
        }
        self.baseline={"enter":0.0,"continue":0.0}
        self.count={"enter":0,"continue":0}
        self.update_count=0

    @classmethod
    def load(cls,path):
        d=json.loads(Path(path).read_text(encoding="utf-8"))
        if d.get("schema")!=1:raise ValueError("unknown behavior checkpoint")
        out=cls()
        for name in ("enter","continue"):
            x=np.asarray(d["weights"][name],dtype=np.float64)
            if x.shape!=(D_BEHAVIOR,) or not np.all(np.isfinite(x)):
                raise ValueError("invalid action policy weights")
            out.weights[name]=x
            out.baseline[name]=float(d["baseline"][name])
            out.count[name]=int(d["count"][name])
        out.update_count=int(d["update_count"])
        if out.digest()!=d["weights_sha256"]:raise ValueError("action checkpoint modified")
        return out

    def digest(self):
        packed=b"".join(self.weights[n].tobytes() for n in ("enter","continue"))
        return hashlib.sha256(packed).hexdigest()

    def save(self,path):
        obj={
            "schema":1,"decision_classes":["enter","continue"],
            "weights":{key:v.tolist() for key,v in self.weights.items()},
            "baseline":self.baseline,"count":self.count,
            "update_count":self.update_count,
            "weights_sha256":self.digest(),
            "description":"Learnable generic enter/skip and page continue/stop, not racing tactics",
        }
        Path(path).parent.mkdir(parents=True,exist_ok=True)
        Path(path).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
        return obj

    def decide(self,kind,context,rng,sheet_index=None):
        if kind not in self.weights:raise ValueError("unknown action")
        c=np.asarray(context,dtype=float)
        if c.shape!=(D_BEHAVIOR,):raise ValueError("bad context size")
        chance=sigmoid(self.weights[kind]@c)
        action=bool(rng.random()<chance)
        return Decision(kind,action,chance,c.tolist(),sheet_index)

    def update(self,decision,realized_incremental_net_yen):
        """Score function gradient with EMA advantage, counterfactual-free.

        Enter: total race net vs not entering (0).
        Continue: value of newly purchased sheet vs stopping (0).
        Each action is credited separately, not every mark in winning race.
        """
        kind=decision.kind
        if kind not in self.weights:raise ValueError("unknown decision")
        reward=float(np.tanh(realized_incremental_net_yen/YEN_UNIT))
        advantage=reward-self.baseline[kind]
        c=np.asarray(decision.context,dtype=np.float64)
        z=(float(decision.action)-decision.probability)*advantage
        delta=LEARNING_RATE*z*c
        self.weights[kind]+=delta
        self.baseline[kind]=(1-BASELINE_RATE)*self.baseline[kind]+BASELINE_RATE*reward
        self.count[kind]+=1
        self.update_count+=1
        return {
            "kind":kind,"action_taken":bool(decision.action),
            "realized_incremental_net_yen":int(round(realized_incremental_net_yen)),
            "scaled_reward":reward,"advantage":advantage,
            "update_l2":float(np.linalg.norm(delta)),
            "new_action_probability":sigmoid(self.weights[kind]@c),
            "new_baseline":self.baseline[kind]
        }


def learn_from_race(policy,decisions,sheet_returns,paper_net):
    """Read ONLY after predictions are sealed and the real result was revealed."""
    steps=[]
    for d in decisions:
        if d.kind=="enter":
            net=paper_net if d.action else 0
        else:
            net=sheet_returns[d.sheet_index] if d.action else 0
        steps.append(policy.update(d,net))
    return steps


def make_learned_sheets(arms,probabilities,odds,entries,bank_yen,rng,policy):
    """Generate marks using SAME old formation semantics, but learned skip/stop.

    Every decision, its probability and PRE-race context are returned for
    post-race learning. No fixed cap; finite seed options bound runtime.
    """
    from train_week2 import formation_from_seed
    x=np.asarray(probabilities,dtype=float)
    first=policy.decide("enter",policy_context(x,bank_yen,0,0),rng)
    decisions=[first]
    if not first.action:
        return [],decisions
    remaining=set(range(len(arms)))
    papers=[]
    stake=0
    while remaining:
        ids=np.fromiter(sorted(remaining),dtype=np.int32)
        weights=x[ids];weights=weights/weights.sum()
        index=int(rng.choice(ids,p=weights))
        seed=arms[index]
        pool_ids=[i for i,a in enumerate(arms) if a["market"]==seed["market"]]
        paper=formation_from_seed(seed,[arms[i] for i in pool_ids],x[pool_ids],
                                  odds,entries,rng)
        paper["sheet_number"]=len(papers)+1
        paper["seed_selection"]=seed["selection"]
        papers.append(paper)
        stake+=paper["total_yen"]
        remaining.remove(index)
        if not remaining:
            break
        control=policy.decide("continue",
            policy_context(x,bank_yen,len(papers),stake),rng,
            sheet_index=len(papers))
        decisions.append(control)
        if not control.action:
            break
    return papers,decisions
