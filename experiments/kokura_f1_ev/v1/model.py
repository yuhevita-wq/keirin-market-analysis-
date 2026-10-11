"""Frozen v1: seven-rider outcome probabilities and snapshot EV calculation."""
import argparse
import csv
import itertools
import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

PERMS = np.array(list(itertools.permutations(range(7), 3)))
COMBOS = list(itertools.combinations(range(1, 8), 3))
GROUP = np.array([COMBOS.index(tuple(sorted(v + 1))) for v in PERMS])
FIELDS = ("score", "b_count", "win_rate", "top3_rate")


def validate_card(card):
    riders = sorted(card["riders"], key=lambda r: r["car_no"])
    if [r["car_no"] for r in riders] != list(range(1, 8)):
        raise ValueError("Exactly seven distinct riders numbered 1..7 are required")
    members = [c for line in card["lines"] for c in line]
    if any(not line for line in card["lines"]) or sorted(members) != list(range(1, 8)):
        raise ValueError("Lines must contain every rider exactly once")
    if not isinstance(card["special"], bool):
        raise ValueError("special must be a boolean")
    for r in riders:
        for f in FIELDS:
            if not math.isfinite(float(r[f])):
                raise ValueError(f"Nonfinite {f}")
        if r["b_count"] < 0 or any(not 0 <= r[f] <= 100 for f in ("win_rate", "top3_rate")):
            raise ValueError("Invalid B count or percentage")
    return riders


def features(card):
    riders = validate_card(card)
    sc = np.array([r["score"] for r in riders])
    b = np.array([r["b_count"] for r in riders])
    wr = np.array([r["win_rate"] for r in riders]) / 100
    tr = np.array([r["top3_rate"] for r in riders]) / 100
    loc = {c: (i, j + 1, len(line)) for i, line in enumerate(card["lines"]) for j, c in enumerate(line)}
    lp = np.array([loc[c][1] for c in range(1, 8)])
    sz = np.array([loc[c][2] for c in range(1, 8)])
    li = np.array([loc[c][0] for c in range(1, 8)])
    indiv = np.column_stack([(sc-sc.mean())/5, b/10, wr, tr,
                             ((lp == 2) & (sz > 1)).astype(float),
                             (lp >= 3).astype(float), (sz == 1).astype(float)])
    same = [(li[PERMS[:,a]] == li[PERMS[:,z]]).astype(float) for a,z in [(0,1),(0,2),(1,2)]]
    adj = [((li[PERMS[:,a]] == li[PERMS[:,z]]) &
            (lp[PERMS[:,z]] == lp[PERMS[:,a]] + 1)).astype(float)
           for a,z in [(0,1),(1,0),(1,2)]]
    special = card["special"]
    return np.column_stack([*[indiv[PERMS[:,k]] for k in range(3)],
                            np.column_stack(same + adj),
                            np.column_stack([special*indiv[PERMS[:,0],4],
                                             special*indiv[PERMS[:,0],5],
                                             special*indiv[PERMS[:,0],6], special*same[0]])])


def probabilities(card, model):
    if model["version"] != "kokura-f1-ev-v1":
        raise ValueError("Unknown model version")
    w = np.array(model["weights"])
    x = features(card)
    temperature = float(model["temperature"])
    if w.shape != (x.shape[1],) or not np.all(np.isfinite(w)) or not temperature > 0:
        raise ValueError("Invalid weights or temperature")
    z = x @ w / temperature
    p = np.exp(z-logsumexp(z))
    return p, np.bincount(GROUP, weights=p, minlength=35)


def expected_value(probability, odds, stake=100):
    if not math.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError("Invalid probability")
    if not math.isfinite(odds) or odds < 1 or not math.isfinite(stake) or stake < 0:
        raise ValueError("Invalid decimal odds or stake")
    return {"expected_return_ratio": probability*odds,
            "expected_net_yen": stake*(probability*odds-1),
            "break_even_probability": 1/odds}


def calculate(card, model, snapshot, stake=100):
    p, trio = probabilities(card, model)
    rows = []
    for kind, codes, pp in [("3連単", ["-".join(map(str,v+1)) for v in PERMS], p),
                            ("3連複", ["=".join(map(str,v)) for v in COMBOS], trio)]:
        odds = snapshot["odds"][kind]
        if set(odds) != set(codes):
            raise ValueError(f"Missing or extra combinations for {kind}")
        for code, probability in zip(codes, pp):
            price = float(odds[code])
            rows.append({"race": card["race"], "type": kind, "code": code,
                         "prob": float(probability), "odds": price,
                         "odds_timestamp_jst": snapshot["timestamp_jst"],
                         **expected_value(float(probability), price, stake)})
    return rows


def fit(training, ridge=1.0):
    x = np.array([features(r["card"]) for r in training])
    y = np.array([next(i for i,v in enumerate(PERMS+1) if list(v) == r["result"]) for r in training])
    def fg(w):
        z = x @ w
        p = np.exp(z-logsumexp(z, axis=1, keepdims=True))
        loss = np.sum(logsumexp(z,axis=1)-z[np.arange(len(y)),y]) + .5*ridge*(w@w)
        p[np.arange(len(y)),y] -= 1
        grad = np.einsum("no,nof->f",p,x) + ridge*w
        return loss, grad
    result = minimize(fg,np.zeros(x.shape[-1]),jac=True,method="L-BFGS-B",
                      options={"maxiter":150,"ftol":1e-8})
    if not result.success:
        raise RuntimeError(result.message)
    return result.x


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cards", type=Path, required=True)
    parser.add_argument("--odds", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path(__file__).with_name("frozen_model.json"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--stake", type=float, default=100)
    args = parser.parse_args()
    cards, odds, model = [json.loads(p.read_text()) for p in (args.cards,args.odds,args.model)]
    snapshots = {s["race"]: s for s in odds}
    rows = [r for card in cards for r in calculate(card,model,snapshots[card["race"]],args.stake)]
    with args.out.open("w",newline="",encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(cards)} races / {len(rows)} complete outcomes -> {args.out}")


if __name__ == "__main__":
    main()
