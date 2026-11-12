from __future__ import annotations

"""v8.7-F08: compact incremental growth by added-block purity.

Problem addressed
-----------------
v8.4 ranks rider additions by average added q per new ticket. A rider can create
several filler tickets at once; one or two strong tickets can subsidize weak
rectangle-fillers and cause the formation to balloon.

v8.7 evaluates the *whole newly created block* with the geometric mean of its
normalized 3-rentan supports. The normalized market has 210 exact tickets, so
1/210 is the intrinsic average support of one ticket.

A rider addition is legal only when:

    geometric_mean(q(t) for newly_created_tickets) > 1/210

Thus a rectangular expansion is accepted only if the whole added block is, in a
geometric sense, denser than the market-average exact ticket. No fixed point
count, place count, Q1-fitted cutoff, result, payout, or individual price cut is
used.
"""

from math import exp, log

from v7_0_f01_market_hierarchy import v6_1_entry_gate
from v8_0_f01_rectangular_hierarchy import _structural_pools
from v8_1_f02_cross_market_rectangular import cross_market_ticket_table
from v8_4_f05_incremental_growth import _seed_state, _possible_moves

SCHEME_VERSION = "v8.7-F08"
UNIFORM_Q = 1.0 / 210.0
PRICE_CUT_ENABLED = False
FIXED_POINT_COUNT = False
FIXED_PLACE_COUNTS = False
RESULTS_USED = False
PAYOUTS_USED = False


def _added_tickets(current, nxt):
    return tuple(sorted(set(nxt.tickets) - set(current.tickets)))


def _block_stats(current, move, values):
    added = _added_tickets(current, move.state)
    if not added:
        return None
    qs = [values[t]["q"] for t in added]
    gm_q = exp(sum(log(max(x, 1e-300)) for x in qs) / len(qs))
    min_q = min(qs)
    avg_q = sum(qs) / len(qs)
    bridges = [(values[t]["q"] * values[t]["q_star"]) ** 0.5 for t in added]
    gm_bridge = exp(sum(log(max(x, 1e-300)) for x in bridges) / len(bridges))
    return {
        "gm_q": gm_q,
        "avg_q": avg_q,
        "min_q": min_q,
        "gm_bridge": gm_bridge,
        "purity_ratio": gm_q / UNIFORM_Q,
        "added_count": len(added),
    }


def choose_block_purity_growth(trio_odds, trifecta_odds, gate):
    values = cross_market_ticket_table(trio_odds, trifecta_odds)
    p1, p2, p3 = _structural_pools(trio_odds, gate)
    cur = _seed_state(p1, p2, p3, values)
    states = [cur]
    trace = []

    while True:
        candidates = []
        for move in _possible_moves(cur, p1, p2, p3, values):
            stats = _block_stats(cur, move, values)
            if stats is None or stats["gm_q"] <= UNIFORM_Q:
                continue
            candidates.append((move, stats))
        if not candidates:
            break
        move, stats = max(
            candidates,
            key=lambda ms: (
                ms[1]["gm_q"],
                ms[1]["gm_bridge"],
                ms[1]["avg_q"],
                -ms[1]["added_count"],
                -ms[0].place,
                -ms[0].rider,
            ),
        )
        trace.append({
            "from": cur.display,
            "to": move.state.display,
            "place": move.place,
            "rider": move.rider,
            **stats,
        })
        cur = move.state
        states.append(cur)

    return cur, states, trace


def build_v8_7_f08(trio_odds, trifecta_odds, predicted_line_formation):
    gate = v6_1_entry_gate(trio_odds, trifecta_odds, predicted_line_formation)
    if not gate["entry_pass"]:
        return {"scheme_version":SCHEME_VERSION,"buy":False,"reason":gate["reason"]}
    try:
        chosen, states, trace = choose_block_purity_growth(trio_odds,trifecta_odds,gate)
    except ValueError as exc:
        return {"scheme_version":SCHEME_VERSION,"buy":False,"reason":str(exc)}
    return {
        "scheme_version":SCHEME_VERSION,"buy":True,"reason":"PASS",
        "formation":chosen.display,"first":chosen.first,"second":chosen.second,"third":chosen.third,
        "tickets":chosen.tickets,"ticket_count":chosen.ticket_count,"q_mass":chosen.q_mass,
        "growth_steps":len(states)-1,"growth_trace":trace,"entry_gate":gate,
        "formation_selector":"grow only when geometric mean q of the whole added rectangle block exceeds uniform 1/210",
        "uniform_q_boundary":UNIFORM_Q,"price_cut_enabled":False,"fixed_point_count":False,"fixed_place_counts":False,
    }
