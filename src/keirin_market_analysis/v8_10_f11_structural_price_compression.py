from __future__ import annotations

"""v8.10-F11: v8.9 entry/formation + structural price compression.

Core order is intentionally strict:
1. Build the complete v8.9-F10 formation WITHOUT any price pruning.
2. Only after that formation exists, inspect trifecta prices.
3. Price correction may remove a rider from an entire place-set only.
4. Every resulting bet set must still be one clean rectangular F1-F2-F3 formation.
5. Individual tickets are never removed just because they are cheap.

The price phase therefore changes the FORMATION, not an arbitrary list of tickets.
No result or payout is used by this engine.
"""

from dataclasses import dataclass
from math import exp, log
from typing import Mapping

from v7_0_f01_market_hierarchy import implied_probabilities
from v8_8_f09_market_cliff import State, _state
from v8_9_f10_compact_entry import build_v8_9_f10

SCHEME_VERSION = "v8.10-F11"
BASE_SCHEME = "v8.9-F10"
INDIVIDUAL_TICKET_PRUNING = False
STRUCTURAL_PRICE_COMPRESSION = True
FIXED_PLACE_COUNTS = False
FIXED_POINT_COUNT = False
NESTED_REQUIRED = False


@dataclass(frozen=True)
class PriceState:
    state: State
    q_retention: float
    weighted_gm_odds: float
    gm_return_multiple: float
    profitable_q_share: float

    @property
    def display(self) -> str:
        return self.state.display


@dataclass(frozen=True)
class PriceMove:
    place: int
    rider: int
    q_loss: float
    log_return_gain: float
    efficiency: float
    next_state: PriceState


def _price_state(
    state: State,
    base_q_mass: float,
    q: Mapping[tuple[int, int, int], float],
    trifecta_odds: Mapping[tuple[int, int, int], float],
) -> PriceState:
    mass = state.q_mass
    weights = [q[t] / mass for t in state.tickets]
    gm_odds = exp(sum(w * log(float(trifecta_odds[t])) for w, t in zip(weights, state.tickets)))
    n = state.ticket_count
    gm_mult = gm_odds / n
    profitable_mass = sum(q[t] for t in state.tickets if float(trifecta_odds[t]) >= n)
    return PriceState(
        state=state,
        q_retention=mass / base_q_mass if base_q_mass > 0 else 0.0,
        weighted_gm_odds=gm_odds,
        gm_return_multiple=gm_mult,
        profitable_q_share=profitable_mass / mass if mass > 0 else 0.0,
    )


def _one_rider_removals(
    current: PriceState,
    base_q_mass: float,
    q,
    trifecta_odds,
):
    cur = current.state
    sets = {1: cur.first, 2: cur.second, 3: cur.third}
    for place in (1, 2, 3):
        if len(sets[place]) <= 1:
            continue
        for rider in sets[place]:
            f1, f2, f3 = cur.first, cur.second, cur.third
            if place == 1:
                f1 = tuple(x for x in f1 if x != rider)
            elif place == 2:
                f2 = tuple(x for x in f2 if x != rider)
            else:
                f3 = tuple(x for x in f3 if x != rider)
            nxt = _state(f1, f2, f3, q)
            if nxt is None:
                continue
            pnext = _price_state(nxt, base_q_mass, q, trifecta_odds)
            q_loss = cur.q_mass - nxt.q_mass
            if q_loss <= 0:
                continue
            gain = log(pnext.gm_return_multiple) - log(current.gm_return_multiple)
            if gain <= 0:
                continue
            yield PriceMove(
                place=place,
                rider=rider,
                q_loss=q_loss,
                log_return_gain=gain,
                efficiency=gain / q_loss,
                next_state=pnext,
            )


def _best_removal(current: PriceState, base_q_mass, q, trifecta_odds):
    moves = list(_one_rider_removals(current, base_q_mass, q, trifecta_odds))
    if not moves:
        return None
    # Primary: largest economic gain per unit of market support sacrificed.
    # Tie: smaller support loss, then earlier place, then smaller rider number.
    return max(
        moves,
        key=lambda m: (m.efficiency, -m.q_loss, -m.place, -m.rider),
    )


def _compression_path(base: PriceState, q, trifecta_odds):
    states = [base]
    moves = []
    cur = base
    base_q_mass = base.state.q_mass
    while True:
        move = _best_removal(cur, base_q_mass, q, trifecta_odds)
        if move is None:
            break
        moves.append(move)
        cur = move.next_state
        states.append(cur)
    return states, moves


def _select_structural_knee(states: list[PriceState]):
    """Choose a parameter-free knee between support retention and price gain.

    State 0 is the untouched completed formation.  Later states are clean
    rectangular subformations reached only by whole-rider removal.

    We normalize cumulative q-loss and cumulative log-return gain to [0, 1]
    across the realized compression path and choose the state with the largest
    vertical distance above the endpoint chord.  This avoids a hand-picked
    ticket limit or odds threshold.
    """
    if len(states) <= 1:
        return 0
    base = states[0]
    last = states[-1]
    total_loss = 1.0 - last.q_retention
    total_gain = log(last.gm_return_multiple) - log(base.gm_return_multiple)
    if total_loss <= 0 or total_gain <= 0:
        return 0

    best_i = 0
    best_dist = 0.0
    for i, s in enumerate(states):
        loss_norm = (1.0 - s.q_retention) / total_loss
        gain_norm = (log(s.gm_return_multiple) - log(base.gm_return_multiple)) / total_gain
        distance = gain_norm - loss_norm
        if distance > best_dist:
            best_dist = distance
            best_i = i
    return best_i


def build_v8_10_f11(trio_odds, trifecta_odds, predicted_line_formation):
    # Phase A: formation MUST be completed before the price phase starts.
    base = build_v8_9_f10(trio_odds, trifecta_odds, predicted_line_formation)
    if not base.get("buy"):
        return {
            **base,
            "scheme_version": SCHEME_VERSION,
            "base_scheme": BASE_SCHEME,
            "individual_ticket_pruning": INDIVIDUAL_TICKET_PRUNING,
            "structural_price_compression": STRUCTURAL_PRICE_COMPRESSION,
        }

    q = implied_probabilities(trifecta_odds)
    base_state = _state(base["first"], base["second"], base["third"], q)
    if base_state is None:
        return {
            "scheme_version": SCHEME_VERSION,
            "buy": False,
            "reason": "INVALID_BASE_FORMATION",
            "base_scheme": BASE_SCHEME,
        }

    p0 = _price_state(base_state, base_state.q_mass, q, trifecta_odds)
    states, moves = _compression_path(p0, q, trifecta_odds)
    selected_i = _select_structural_knee(states)
    chosen = states[selected_i]

    trace = []
    for i, ps in enumerate(states):
        row = {
            "step": i,
            "formation": ps.display,
            "ticket_count": ps.state.ticket_count,
            "q_mass": ps.state.q_mass,
            "q_retention": ps.q_retention,
            "weighted_gm_odds": ps.weighted_gm_odds,
            "gm_return_multiple": ps.gm_return_multiple,
            "profitable_q_share": ps.profitable_q_share,
            "selected": i == selected_i,
        }
        if i > 0:
            m = moves[i - 1]
            row.update({
                "removed_place": m.place,
                "removed_rider": m.rider,
                "q_loss": m.q_loss,
                "log_return_gain": m.log_return_gain,
                "price_efficiency": m.efficiency,
            })
        trace.append(row)

    return {
        **base,
        "scheme_version": SCHEME_VERSION,
        "base_scheme": BASE_SCHEME,
        "formation_before_price": base["formation"],
        "ticket_count_before_price": base["ticket_count"],
        "formation": chosen.display,
        "first": chosen.state.first,
        "second": chosen.state.second,
        "third": chosen.state.third,
        "tickets": chosen.state.tickets,
        "ticket_count": chosen.state.ticket_count,
        "q_mass": chosen.state.q_mass,
        "price_compression_steps": selected_i,
        "price_path_steps_total": len(moves),
        "price_q_retention": chosen.q_retention,
        "price_weighted_gm_odds": chosen.weighted_gm_odds,
        "price_gm_return_multiple": chosen.gm_return_multiple,
        "price_profitable_q_share": chosen.profitable_q_share,
        "price_compression_trace": trace,
        "individual_ticket_pruning": INDIVIDUAL_TICKET_PRUNING,
        "structural_price_compression": STRUCTURAL_PRICE_COMPRESSION,
        "price_phase_rule": "complete formation first; then whole-rider removals only; every candidate remains one clean F1-F2-F3 rectangle; select parameter-free support-vs-price knee",
        "development_status": "Q1_DESIGN_REQUIRES_DEVELOPMENT_SIMULATION_AND_LATER_OOS_VALIDATION",
    }
