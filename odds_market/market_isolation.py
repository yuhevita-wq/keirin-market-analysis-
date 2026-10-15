"""Cross-market isolation research, odds-only. NOT a calibrated win model.

All sold combinations are required. Every signal is computed without lineups,
form, rider stats, race outcome, or favorite narratives.

The 3rentan anomaly estimate is explicitly OUT-OF-MARKET: IPF on 2shatan and
3renpuku probabilities. Other types are checked using exact event aggregation
from other markets. Support differences are not evidence of positive EV by
themselves; validate with chronological snapshots and observed payouts.
"""
import itertools
import math
from collections import defaultdict

from odds_market.collect_all_odds import expected

CAR_FRAME_9 = {1: 1, 2: 2, 3: 3, 4: 4, 5: 4,
               6: 5, 7: 5, 8: 6, 9: 6}
KINDS = ("3rentan", "3renpuku", "2shatan", "2shafuku",
         "wide", "2wakutan", "2wakufuku")


def normalize_quotes(records):
    quotes = defaultdict(dict)
    stamped = set()
    race_ids = set()
    for row in records:
        rid, _, kind, selection, low, high, stamp = row
        key = tuple(map(int, selection.split("-")))
        a, b = float(low), float(high)
        if kind not in KINDS or not 1 <= a <= b or not math.isfinite(b):
            raise ValueError("Invalid quote type or odds")
        if key in quotes[kind]:
            raise ValueError("Duplicate quote key")
        quotes[kind][key] = (a, b)
        stamped.add(stamp)
        race_ids.add(rid)
    if len(race_ids) != 1 or len(stamped) != 1:
        raise ValueError("Mixed race ID or market timestamp")
    nums = {num for triple in quotes["3rentan"] for num in triple}
    n = len(nums)
    if nums != set(range(1, n + 1)):
        raise ValueError("Invalid car numbers")
    exp = expected(n)
    if set(quotes) != set(exp) or any(set(quotes[k]) != v for k, v in exp.items()):
        raise ValueError("Refusing incomplete sold bet markets")
    probs = {}
    for kind, group in quotes.items():
        inverse = {k: 1 / math.sqrt(lo * hi) for k, (lo, hi) in group.items()}
        total = sum(inverse.values())
        if total <= 0:
            raise ValueError("Zero implied quote mass")
        # 3 wide pairs (not 1) win on an ordered trifecta; use scale 3.
        multiplier = 3 if kind == "wide" else 1
        probs[kind] = {key: multiplier * val / total
                       for key, val in inverse.items()}
    return quotes, probs, n, next(iter(race_ids)), next(iter(stamped))


def marginals_of_trifecta(p3, n):
    p2 = defaultdict(float)
    pu2 = defaultdict(float)
    pu3 = defaultdict(float)
    wide = defaultdict(float)
    wf2 = defaultdict(float)
    wfu2 = defaultdict(float)
    for (a, b, c), weight in p3.items():
        p2[a, b] += weight
        pu2[tuple(sorted((a, b)))] += weight
        pu3[tuple(sorted((a, b, c)))] += weight
        for pair in itertools.combinations(sorted((a, b, c)), 2):
            wide[pair] += weight
        if n == 9:
            fa, fb = CAR_FRAME_9[a], CAR_FRAME_9[b]
            wf2[fa, fb] += weight
            wfu2[tuple(sorted((fa, fb)))] += weight
    return {"2shatan": dict(p2), "2shafuku": dict(pu2),
            "3renpuku": dict(pu3), "wide": dict(wide),
            "2wakutan": dict(wf2), "2wakufuku": dict(wfu2)}


def fit_outside_trifecta(p_exacta, p_trio, n, iters=120):
    """Maximum entropy joint table using ONLY exacta/trio observed margins.

    Iterative proportional fitting preserves both observed marginals when
    they are jointly feasible; inconsistencies are separately recorded.
    """
    p = {(a, b, c): p_exacta[a, b] / (n - 2)
         for a, b, c in itertools.permutations(range(1, n + 1), 3)}
    for iteration in range(iters):
        by_trio = defaultdict(float)
        for key, w in p.items():
            by_trio[tuple(sorted(key))] += w
        for key in p:
            trio = tuple(sorted(key))
            p[key] *= p_trio[trio] / max(by_trio[trio], 1e-300)
        by_pair = defaultdict(float)
        for (a, b, _), w in p.items():
            by_pair[a, b] += w
        for key in p:
            p[key] *= p_exacta[key[:2]] / max(by_pair[key[:2]], 1e-300)
    trio_reconstruction = marginals_of_trifecta(p, n)["3renpuku"]
    max_trio_error = max(abs(trio_reconstruction[k] - v)
                         for k, v in p_trio.items())
    return p, max_trio_error


def scan(records, limit=25):
    quotes, observed, n, race, timestamp = normalize_quotes(records)
    p3 = observed["3rentan"]
    group3 = marginals_of_trifecta(p3, n)
    reconstructed, max_ipf_error = fit_outside_trifecta(
        observed["2shatan"], observed["3renpuku"], n)
    # Combine independent marginal routes; this is a relative support
    # diagnostic, NOT a market-proof estimate of probabilities.
    predictions = {
        "3rentan": reconstructed,
        "3renpuku": group3["3renpuku"],
        "2shatan": group3["2shatan"],
        "2shafuku": {
            k: (group3["2shafuku"][k] +
                observed["2shatan"].get((k[0], k[1]), 0) +
                observed["2shatan"].get((k[1], k[0]), 0)) / 2
            for k in observed["2shafuku"]
        },
        "wide": {
            k: (group3["wide"][k] +
                sum(v for t, v in observed["3renpuku"].items()
                    if k[0] in t and k[1] in t)) / 2
            for k in observed["wide"]
        },
    }
    if n == 9:
        predictions["2wakutan"] = group3["2wakutan"]
        predictions["2wakufuku"] = group3["2wakufuku"]
    rows = []
    all_rows = []
    for kind in KINDS:
        if kind not in observed:
            continue
        for key, observed_probability in observed[kind].items():
            predicted = predictions[kind].get(key, 0)
            if predicted <= 0:
                continue
            ratio = predicted / observed_probability
            lo, hi = quotes[kind][key]
            if kind == "3rentan":
                a, b, c = key
                supports = {
                    "2shatan": round(observed["2shatan"][a, b], 6),
                    "3renpuku": round(observed["3renpuku"][tuple(sorted(key))], 6),
                }
            elif kind == "2shafuku":
                supports = {
                    "2shatan_AB_BA": round(
                        observed["2shatan"][key] +
                        observed["2shatan"][tuple(reversed(key))], 6),
                    "3rentan_1st2": round(group3["2shafuku"][key], 6)
                }
            elif kind == "wide":
                supports = {
                    "3renpuku_includes_pair": round(sum(
                        v for t, v in observed["3renpuku"].items()
                        if set(key).issubset(t)), 6),
                    "3rentan_includes_pair": round(group3["wide"][key], 6),
                }
            else:
                supports = {"other_market_probability": round(predicted, 6)}
            cand = {
                "type": kind, "selection": "-".join(map(str, key)),
                "odds": round(lo, 2),
                "odds_max": round(hi, 2),
                "support_ratio": round(ratio, 3),
                "other_market_support": round(predicted, 6),
                "own_market_support": round(observed_probability, 6),
                # Explicitly experimental, not a positive-EV assertion.
                "diagnostic_fair_multiple": round(1 / predicted, 2),
                "supports": supports,
            }
            all_rows.append(cand)
            if predicted >= 0.002 and ratio >= 1.25:
                # Prefer material probability mass to pure 9999.9 longshots.
                cand["priority"] = round((ratio - 1) * math.sqrt(predicted), 6)
                rows.append(cand)
    rows.sort(key=lambda x: (-x["priority"], -x["support_ratio"], x["type"]))
    return {
        "race_id": race, "source_odds_time_jst": timestamp, "cars": n,
        "counts": {k: len(quotes[k]) for k in KINDS if k in quotes},
        "integrity": {"complete": True,
                      "same_timestamp": True,
                      "ipf_max_trio_margin_error": round(max_ipf_error, 8)},
        "method": "full market normalized reciprocal odds; odds-only; leave 3rentan out of its own projection",
        "warning": "Anomalous SUPPORT, not proven mispricing or positive expected value. Wide quotes are ranges; thin pools fluctuate.",
        "top": rows[:limit],
    }
