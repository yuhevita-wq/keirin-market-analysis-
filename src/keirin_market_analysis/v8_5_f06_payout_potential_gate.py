from __future__ import annotations

"""v8.5-F06: race-level payout-potential gate on top of v8.4-F05.

Core idea
---------
A race is not attractive merely because the selected formation has high hit
support. We also require the *price structure of the whole formation* to be
healthy if it hits.

This module deliberately does NOT delete individual low-odds tickets. The v8.4
formation remains intact. Price is used only to classify the whole race as
buy/skip.

No result, payout, or Q1-fitted threshold is used.

For a v8.4 formation with N tickets and 100 yen per ticket:

    race stake = 100 * N
    gross return multiple of ticket t if it wins = odds(t) / N

Let q(t) be normalized 3-rentan market mass and condition it on the selected
formation:

    w(t) = q(t) / sum_selected q(t)

We define two intrinsic payout-potential diagnostics:

1. Market-weighted geometric hit return multiple

    PPM = exp(sum_selected w(t) * log(odds(t)/N))

   PPM > 1 means the market-weighted geometric (typical, tail-resistant)
   payout conditional on a formation hit exceeds the total race stake.

2. Profitable-hit support share

    PHS = sum_{odds(t)/N >= 1} w(t)

   PHS > 0.5 means a strict majority of the selected formation's market hit
   support lies on tickets that at least return the whole race stake.

The race is bought only when BOTH hold. These are not Q1-tuned cut points:
1.0 is mathematical break-even and 0.5 is majority support.
"""

from math import exp, log

from v8_4_f05_incremental_growth import build_v8_4_f05

SCHEME_VERSION = "v8.5-F06"
BASE_FORMATION_VERSION = "v8.4-F05"
STAKE_PER_TICKET = 100
PRICE_CUT_ENABLED = False
FIXED_PLACE_COUNTS = False
FIXED_POINT_COUNT = False
RESULTS_USED = False
PAYOUTS_USED = False


def _normalized_q(trifecta_odds):
    inv = {}
    for t, o in trifecta_odds.items():
        x = float(o)
        if x <= 0:
            raise ValueError("non-positive trifecta odds")
        inv[t] = 1.0 / x
    z = sum(inv.values())
    if z <= 0:
        raise ValueError("invalid trifecta normalization")
    return {t: v / z for t, v in inv.items()}, z


def payout_potential_metrics(tickets, trifecta_odds):
    tickets = tuple(tickets)
    n = len(tickets)
    if n <= 0:
        raise ValueError("empty formation")

    q, total_inv = _normalized_q(trifecta_odds)
    missing = [t for t in tickets if t not in q]
    if missing:
        raise ValueError("selected ticket missing from trifecta market")

    q_mass = sum(q[t] for t in tickets)
    if q_mass <= 0:
        raise ValueError("non-positive selected q mass")

    weighted_log_return = 0.0
    profitable_support = 0.0
    weighted_arithmetic_return = 0.0
    min_return = float("inf")
    max_return = 0.0

    for t in tickets:
        w = q[t] / q_mass
        gross_return_multiple = float(trifecta_odds[t]) / n
        if gross_return_multiple <= 0:
            raise ValueError("non-positive gross return multiple")
        weighted_log_return += w * log(gross_return_multiple)
        weighted_arithmetic_return += w * gross_return_multiple
        if gross_return_multiple >= 1.0:
            profitable_support += w
        min_return = min(min_return, gross_return_multiple)
        max_return = max(max_return, gross_return_multiple)

    ppm = exp(weighted_log_return)
    conditional_payout_proxy_yen = 100.0 * n / (total_inv * q_mass)
    race_stake_yen = STAKE_PER_TICKET * n

    return {
        "ticket_count": n,
        "race_stake_yen": race_stake_yen,
        "q_mass": q_mass,
        "payout_potential_multiple": ppm,
        "payout_potential_yen": race_stake_yen * ppm,
        "profitable_hit_support_share": profitable_support,
        "market_weighted_arithmetic_hit_return_multiple": weighted_arithmetic_return,
        "conditional_payout_proxy_yen": conditional_payout_proxy_yen,
        "min_ticket_return_multiple": min_return,
        "max_ticket_return_multiple": max_return,
    }


def build_v8_5_f06(trio_odds, trifecta_odds, predicted_line_formation):
    base = build_v8_4_f05(trio_odds, trifecta_odds, predicted_line_formation)
    if not base.get("buy"):
        return {
            "scheme_version": SCHEME_VERSION,
            "base_formation_version": BASE_FORMATION_VERSION,
            "buy": False,
            "reason": base.get("reason", "BASE_SKIP"),
        }

    try:
        pm = payout_potential_metrics(base["tickets"], trifecta_odds)
    except ValueError as exc:
        return {
            "scheme_version": SCHEME_VERSION,
            "base_formation_version": BASE_FORMATION_VERSION,
            "buy": False,
            "reason": f"PAYOUT_METRIC_ERROR:{exc}",
        }

    geometric_break_even = pm["payout_potential_multiple"] > 1.0
    majority_profitable = pm["profitable_hit_support_share"] > 0.5
    buy = geometric_break_even and majority_profitable

    return {
        **base,
        "scheme_version": SCHEME_VERSION,
        "base_formation_version": BASE_FORMATION_VERSION,
        "buy": buy,
        "reason": "PASS" if buy else (
            "PAYOUT_POTENTIAL_LE_1X" if not geometric_break_even
            else "PROFITABLE_HIT_SUPPORT_NOT_MAJORITY"
        ),
        "price_cut": False,
        "payout_gate": {
            "rule": "PPM>1.0 AND profitable_hit_support_share>0.5",
            "threshold_source": "intrinsic break-even and majority; not fitted on Q1 outcomes",
            **pm,
        },
    }
