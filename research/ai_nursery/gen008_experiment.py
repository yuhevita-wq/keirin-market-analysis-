"""Experimental GEN-008 primitives; old GEN-007 implementation is untouched.

1) Frozen-pre-result one-sheet shadow for EVERY race, even if skipped.
2) Two-market coherent probabilities: ordered top three sum to one; trio
   probability is the sum of its six ordered permutations.
3) Online temperature adapted ONLY after past finishes are unsealed.

Never confuse paper shadow P&L with money actually spent. A single shadow
is an observed opportunity for that PRE-COMMITTED plan only, not a claim
about every profitable bet in a skipped race.
"""
from __future__ import annotations

import hashlib
import math

import numpy as np

from behavior_policy import policy_context, sigmoid, BASELINE_RATE, LEARNING_RATE, YEN_UNIT
from profit_objective import ProfitPolicy, MAX_POLICY_STEP_NORM
from train_week2 import formation_from_seed

TEMPERATURE_LOWER=0.5
TEMPERATURE_UPPER=4.0
TEMPERATURE_RATE=0.012


def race_seed(race_id, namespace="decisions"):
    digest=hashlib.sha256((str(race_id)+"|GEN008|"+namespace).encode("utf-8")).digest()
    return int.from_bytes(digest[:8],"big")


def model_digest(weights):
    return hashlib.sha256(np.asarray(weights,dtype=np.float64).tobytes()).hexdigest()


def coherent_probabilities(arms, weights, temperature=1.0):
    """Return one sampling vector AND true per-market probabilities.

    The trained weights originate in GEN-007. They are not reset.
    For a trifecta: q(x) = softmax(score(x)/T) across all ordered outcomes.
    For each trio: q({a,b,c}) = sum q(all six permutations).
    The sampling vector uses 50% each MARKET to make both available for
    mark-sheet seeding; it is not itself a probability of a race outcome.
    """
    if not math.isfinite(temperature) or temperature<=0:raise ValueError("bad temperature")
    w=np.asarray(weights,dtype=np.float64)
    triple=[(i,a) for i,a in enumerate(arms) if a["market"]=="3連単"]
    trio=[(i,a) for i,a in enumerate(arms) if a["market"]=="3連複"]
    if not triple or not trio:raise ValueError("Both complete books required")
    if any(len(a["x"])!=len(w) for _,a in triple):
        raise ValueError("prediction feature size mismatch")
    X=np.stack([a["x"] for _,a in triple])
    prior=np.asarray([a["prior"] for _,a in triple])
    logits=(X@w+prior)/float(temperature)
    exp=np.exp(logits-np.max(logits))
    q=exp/np.sum(exp)
    trio_map={}
    for local,(_,arm) in enumerate(triple):
        selection=tuple(sorted(map(int,arm["selection"].split("-"))))
        k="=".join(map(str,selection))
        trio_map[k]=trio_map.get(k,0.0)+float(q[local])
    q_trio=np.asarray([trio_map.get(a["selection"],0.0) for _,a in trio])
    if np.any(q_trio<=0) or abs(float(q_trio.sum())-1)>1e-8:
        raise AssertionError("Cannot form coherent unordered market")
    if abs(float(q.sum())-1)>1e-8:
        raise AssertionError("Ordered outcome market not normalized")
    mix=np.zeros(len(arms),dtype=np.float64)
    for k,(i,_) in enumerate(triple):mix[i]=0.5*float(q[k])
    for k,(i,_) in enumerate(trio):mix[i]=0.5*float(q_trio[k])
    if abs(float(mix.sum())-1)>1e-8:
        raise AssertionError("Improper market-mixture sampling probabilities")
    return mix,{"3連単":dict((a["selection"],float(q[j])) for j,(_,a) in enumerate(triple)),
                "3連複":dict((a["selection"],float(q_trio[j])) for j,(_,a) in enumerate(trio))}


def calibration_metrics(market_probs, finish):
    ordered="-".join(map(str,finish))
    unordered="=".join(map(str,sorted(finish)))
    pt=float(market_probs["3連単"][ordered])
    pc=float(market_probs["3連複"][unordered])
    return {"trifecta_logloss":float(-math.log(max(pt,1e-30))),
            "trio_logloss":float(-math.log(max(pc,1e-30))),
            "trifecta_winner_probability":pt,
            "trio_winner_probability":pc,
            "trifecta_mass":float(sum(market_probs["3連単"].values())),
            "trio_mass":float(sum(market_probs["3連複"].values()))}


def update_temperature(previous_temperature, arms, weights_before, finish):
    """Online gradient on ordered-market log-loss, after sealed outcome only."""
    t=float(previous_temperature)
    eps=0.005
    lo=max(TEMPERATURE_LOWER,t*math.exp(-eps))
    hi=min(TEMPERATURE_UPPER,t*math.exp(eps))
    if hi==lo:return t
    lp=calibration_metrics(coherent_probabilities(arms,weights_before,hi)[1],finish)["trifecta_logloss"]
    lm=calibration_metrics(coherent_probabilities(arms,weights_before,lo)[1],finish)["trifecta_logloss"]
    slope=(lp-lm)/(math.log(hi)-math.log(lo))
    return float(np.clip(t*math.exp(-TEMPERATURE_RATE*slope),TEMPERATURE_LOWER,TEMPERATURE_UPPER))


def make_shadow_sheet(arms, probabilities, odds, entries, race_id):
    """Create one independent PRE-RESULT counterfactual page on EVERY race.

    This is not a real purchased page. Its randomized construction is fixed
    by race ID, independent of the actual skip/enter decision.
    """
    p=np.asarray(probabilities,dtype=np.float64)
    if len(arms)!=len(p) or abs(float(p.sum())-1)>1e-7:
        raise ValueError("invalid shadow input probability")
    rng=np.random.default_rng(race_seed(race_id,"shadow"))
    index=int(rng.choice(len(arms),p=p))
    seed=arms[index]
    indices=[j for j,a in enumerate(arms) if a["market"]==seed["market"]]
    page=formation_from_seed(seed,[arms[j] for j in indices],
                             p[indices],odds,entries,rng)
    page["sheet_number"]=1
    page["seed_selection"]=seed["selection"]
    return page


def score_papers(papers, odds, finish):
    """Realized paper settlement from a *sealed* list of papers, not policy EV."""
    won={("3連単","-".join(map(str,finish))),
         ("3連複","=".join(map(str,sorted(finish))))}
    spent=sum(100*len(page["picks"]) for page in papers)
    paid=int(round(sum(
        100*float(odds[page["type"]][key])
        for page in papers for key in page["picks"]
        if (page["type"],key) in won)))
    return {"stake_yen":int(spent),"payout_yen":paid,"net_yen":int(paid-spent),
            "marks":int(spent//100),"hit":bool(paid>0)}


class ShadowProfitPolicy(ProfitPolicy):
    """Separate action-policy class, initialized from GEN-007 weights.

    Shadow score changes ENTER probability even on skipped races. This is a
    deterministic surrogate for the value of the single sealed page, NOT an
    evaluation of all possible bets or the true return of an unplaced trade.
    CONTINUE still learns from actual page increments via ProfitPolicy.update.
    """
    def update_enter_shadow(self,decision,shadow_net):
        if decision.kind!="enter":raise ValueError("not an enter decision")
        if not math.isfinite(shadow_net):raise ValueError("invalid shadow return")
        p=float(decision.probability)
        ctx=np.asarray(decision.context,dtype=np.float64)
        signal=float(shadow_net)/YEN_UNIT
        baseline=self.baseline["enter"]
        # Estimated expected utility: p(enter|context) * net(one shadow page).
        # Gradient of expected utility uses p*(1-p) and the PRE-race context.
        # Importantly, skipped races also have a nonzero learning signal.
        gradient=LEARNING_RATE*p*(1-p)*(signal-baseline)*ctx
        norm=float(np.linalg.norm(gradient))
        if norm>MAX_POLICY_STEP_NORM:
            gradient*=MAX_POLICY_STEP_NORM/norm
        self.weights["enter"]+=gradient
        self.baseline["enter"]=(1-BASELINE_RATE)*baseline+BASELINE_RATE*signal
        self.count["enter"]+=1
        self.update_count+=1
        return {"kind":"enter","action_taken":bool(decision.action),
                "counterfactual_one_sheet_net_yen":int(shadow_net),
                "actual_enter_reward_not_substituted":True,
                "update_l2":float(np.linalg.norm(gradient)),
                "p_before":p,
                "p_after":sigmoid(self.weights["enter"]@ctx),
                "surrogate_value":"one independently pre-sealed page only"}


def inherited_shadow_policy(path):
    from behavior_policy import BehaviorPolicy
    original=BehaviorPolicy.load(path)
    policy=ShadowProfitPolicy()
    policy.weights={k:v.copy() for k,v in original.weights.items()}
    policy.baseline=dict(original.baseline)
    policy.count=dict(original.count)
    policy.update_count=original.update_count
    if policy.digest()!=original.digest():raise AssertionError("parent policy not inherited")
    return policy
