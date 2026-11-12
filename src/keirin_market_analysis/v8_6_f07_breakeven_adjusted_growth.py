from __future__ import annotations

"""v8.6-F07: choose the v8.4 growth state by break-even-adjusted hit support.

Purpose
-------
v8.4 successfully increased hit rate but grew too wide. v8.5 tried to solve
that after formation construction with a race-level price gate, but it did not
materially improve ROI. v8.6 moves payout economics *inside* formation-width
selection without deleting individual tickets.

For every clean F1-F2-F3 state on the unchanged v8.4 incremental growth path:

    N = ticket count
    r(t) = odds(t) / N
    BAHS(F) = sum_{t in F} q(t) * min(1, r(t))

q(t) is normalized 3-rentan market mass.

Interpretation:
* if a winning ticket returns at least the full race stake (r>=1), its market
  hit support counts fully;
* if it is a losing hit (r<1), its support is discounted in proportion to the
  fraction of stake it would recover.

Thus BAHS rewards retained hit support while automatically penalizing excessive
width that turns popular hits into losing hits. The only boundary is 1.0,
mathematical break-even. There is no fixed ticket count, place count, Q1-fitted
cutoff, or individual price pruning.
"""

from v7_0_f01_market_hierarchy import v6_1_entry_gate
from v8_0_f01_rectangular_hierarchy import _structural_pools
from v8_1_f02_cross_market_rectangular import cross_market_ticket_table
from v8_4_f05_incremental_growth import _seed_state, _grow_path

SCHEME_VERSION = "v8.6-F07"
BASE_GROWTH_VERSION = "v8.4-F05"
PRICE_CUT_ENABLED = False
FIXED_PLACE_COUNTS = False
FIXED_POINT_COUNT = False
RESULTS_USED = False
PAYOUTS_USED = False


def _normalized_q(trifecta_odds):
    inv = {t: 1.0 / float(o) for t, o in trifecta_odds.items() if float(o) > 0}
    z = sum(inv.values())
    if z <= 0:
        raise ValueError("invalid trifecta market")
    return {t: v / z for t, v in inv.items()}


def _state_metrics(state, trifecta_odds, q):
    n = state.ticket_count
    if n <= 0:
        raise ValueError("empty state")
    bahs = 0.0
    profitable_q_mass = 0.0
    losing_q_mass = 0.0
    weighted_recovery_mass = 0.0
    for t in state.tickets:
        r = float(trifecta_odds[t]) / n
        qt = q[t]
        recovery = min(1.0, r)
        bahs += qt * recovery
        weighted_recovery_mass += qt * recovery
        if r >= 1.0:
            profitable_q_mass += qt
        else:
            losing_q_mass += qt
    profitable_share = profitable_q_mass / state.q_mass if state.q_mass > 0 else 0.0
    losing_share = losing_q_mass / state.q_mass if state.q_mass > 0 else 0.0
    return {
        "bahs": bahs,
        "q_mass": state.q_mass,
        "ticket_count": n,
        "profitable_q_mass": profitable_q_mass,
        "profitable_support_share": profitable_share,
        "losing_support_share": losing_share,
        "bahs_density": bahs / n,
    }


def choose_bahs_growth(trio_odds, trifecta_odds, gate):
    values = cross_market_ticket_table(trio_odds, trifecta_odds)
    p1, p2, p3 = _structural_pools(trio_odds, gate)
    seed = _seed_state(p1, p2, p3, values)
    states, moves = _grow_path(seed, p1, p2, p3, values)
    q = _normalized_q(trifecta_odds)
    metrics = [_state_metrics(s, trifecta_odds, q) for s in states]

    # Primary: maximum break-even-adjusted hit support.
    # Tie-breakers prefer more raw hit support, then more profitable support,
    # then the smaller formation.
    idx = max(
        range(len(states)),
        key=lambda i: (
            metrics[i]["bahs"],
            metrics[i]["q_mass"],
            metrics[i]["profitable_support_share"],
            -metrics[i]["ticket_count"],
        ),
    )
    return states[idx], states, moves, metrics, idx


def build_v8_6_f07(trio_odds, trifecta_odds, predicted_line_formation):
    gate = v6_1_entry_gate(trio_odds, trifecta_odds, predicted_line_formation)
    if not gate["entry_pass"]:
        return {"scheme_version": SCHEME_VERSION, "buy": False, "reason": gate["reason"]}

    try:
        chosen, states, moves, metrics, idx = choose_bahs_growth(
            trio_odds, trifecta_odds, gate
        )
    except ValueError as exc:
        return {"scheme_version": SCHEME_VERSION, "buy": False, "reason": str(exc)}

    return {
        "scheme_version": SCHEME_VERSION,
        "base_growth_version": BASE_GROWTH_VERSION,
        "buy": True,
        "reason": "PASS",
        "formation": chosen.display,
        "first": chosen.first,
        "second": chosen.second,
        "third": chosen.third,
        "tickets": chosen.tickets,
        "ticket_count": chosen.ticket_count,
        "q_mass": chosen.q_mass,
        "bahs": metrics[idx]["bahs"],
        "profitable_support_share": metrics[idx]["profitable_support_share"],
        "selected_growth_step": idx,
        "growth_steps_total": len(states) - 1,
        "price_cut_enabled": False,
        "fixed_place_counts": False,
        "fixed_point_count": False,
        "entry_gate": gate,
        "formation_selector": "maximize break-even-adjusted hit support BAHS across the unchanged v8.4 growth path",
        "selector_formula": "sum_selected q(t) * min(1, odds(t)/ticket_count)",
    }
