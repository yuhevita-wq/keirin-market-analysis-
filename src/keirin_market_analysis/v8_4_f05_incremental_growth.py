from __future__ import annotations

"""v8.4-F05: grow one clean formation one rider at a time from the market center.

Design goals
------------
* Keep the v6.1 pre-formation entry gate unchanged.
* Keep one branch-free F1-F2-F3 formation.
* Do not fix rider counts at any place.
* Do not fix the final ticket count.
* Do not use odds>N price pruning.
* Do not use results or payouts anywhere in formation construction.

Why this differs from v8.0-v8.3
-------------------------------
Those versions enumerated completed rectangles and then chose one. That can make
one extra rider create many filler tickets at once. v8.4 instead starts from the
single strongest exact-order ticket inside the unchanged A/B structural pools
and grows the rectangle one rider at a time.

At every growth step we test every legal one-rider addition to F1, F2 or F3.
For a move m:

    dq(m) = added normalized 3-rentan market mass
    dn(m) = number of newly created exact tickets
    q_eff(m) = dq(m) / dn(m)

The move with greatest q_eff is taken. Cross-market information is only an
auxiliary tie-breaker, not the engine:

    q_star(t) = 3-renpuku-implied exact-order mass
    bridge(t) = sqrt(q(t) * q_star(t))

so equal/near-equal popularity expansions prefer the one supported by both
markets. The full deterministic growth path is then reduced to its parameter-
free knee on cumulative q mass versus ticket count. The selected prefix is the
final formation.
"""

from dataclasses import dataclass
from itertools import product
from math import sqrt
from typing import Iterable

from v7_0_f01_market_hierarchy import v6_1_entry_gate
from v8_0_f01_rectangular_hierarchy import _structural_pools
from v8_1_f02_cross_market_rectangular import cross_market_ticket_table

SCHEME_VERSION = "v8.4-F05"
PRICE_CUT_ENABLED = False
FIXED_PLACE_COUNTS = False
FIXED_POINT_COUNT = False
NESTED_REQUIRED = False


@dataclass(frozen=True)
class State:
    first: tuple[int, ...]
    second: tuple[int, ...]
    third: tuple[int, ...]
    tickets: tuple[tuple[int, int, int], ...]
    q_mass: float
    bridge_mass: float
    q_star_mass: float
    cross_excess: float

    @property
    def ticket_count(self) -> int:
        return len(self.tickets)

    @property
    def display(self) -> str:
        show = lambda xs: "".join(map(str, sorted(xs)))
        return f"{show(self.first)}-{show(self.second)}-{show(self.third)}"


@dataclass(frozen=True)
class GrowthMove:
    place: int
    rider: int
    added_ticket_count: int
    added_q_mass: float
    q_efficiency: float
    added_bridge_mass: float
    bridge_efficiency: float
    added_cross_excess: float
    cross_efficiency: float
    state: State


def _tickets(first: Iterable[int], second: Iterable[int], third: Iterable[int]):
    return tuple(sorted({
        (a, b, c)
        for a, b, c in product(first, second, third)
        if len({a, b, c}) == 3
    }))


def _state(first, second, third, values) -> State | None:
    f1 = tuple(sorted(set(first)))
    f2 = tuple(sorted(set(second)))
    f3 = tuple(sorted(set(third)))
    ts = _tickets(f1, f2, f3)
    if not ts:
        return None

    # A displayed rider must generate at least one real exact-order ticket.
    if {t[0] for t in ts} != set(f1):
        return None
    if {t[1] for t in ts} != set(f2):
        return None
    if {t[2] for t in ts} != set(f3):
        return None

    q_mass = sum(values[t]["q"] for t in ts)
    q_star_mass = sum(values[t]["q_star"] for t in ts)
    bridge_mass = sum(sqrt(values[t]["q"] * values[t]["q_star"]) for t in ts)
    return State(
        first=f1,
        second=f2,
        third=f3,
        tickets=ts,
        q_mass=q_mass,
        bridge_mass=bridge_mass,
        q_star_mass=q_star_mass,
        cross_excess=q_star_mass - q_mass,
    )


def _seed_state(p1, p2, p3, values) -> State:
    candidates: list[State] = []
    for a, b, c in product(p1, p2, p3):
        if len({a, b, c}) != 3:
            continue
        s = _state((a,), (b,), (c,), values)
        if s is not None:
            candidates.append(s)
    if not candidates:
        raise ValueError("no valid market-center seed")

    # Primary engine: exact 3-rentan support. Cross-market bridge is auxiliary.
    return max(
        candidates,
        key=lambda s: (
            s.q_mass,
            s.bridge_mass,
            s.cross_excess,
            tuple(-x for x in s.tickets[0]),
        ),
    )


def _possible_moves(current: State, p1, p2, p3, values):
    old_tickets = set(current.tickets)
    pools = {1: tuple(sorted(set(p1))), 2: tuple(sorted(set(p2))), 3: tuple(sorted(set(p3)))}
    current_sets = {1: current.first, 2: current.second, 3: current.third}

    for place in (1, 2, 3):
        for rider in pools[place]:
            if rider in current_sets[place]:
                continue
            f1, f2, f3 = current.first, current.second, current.third
            if place == 1:
                f1 = tuple(sorted(set(f1) | {rider}))
            elif place == 2:
                f2 = tuple(sorted(set(f2) | {rider}))
            else:
                f3 = tuple(sorted(set(f3) | {rider}))

            nxt = _state(f1, f2, f3, values)
            if nxt is None:
                continue
            added = tuple(sorted(set(nxt.tickets) - old_tickets))
            if not added:
                continue

            dn = len(added)
            dq = sum(values[t]["q"] for t in added)
            db = sum(sqrt(values[t]["q"] * values[t]["q_star"]) for t in added)
            de = sum(values[t]["q_star"] - values[t]["q"] for t in added)
            yield GrowthMove(
                place=place,
                rider=rider,
                added_ticket_count=dn,
                added_q_mass=dq,
                q_efficiency=dq / dn,
                added_bridge_mass=db,
                bridge_efficiency=db / dn,
                added_cross_excess=de,
                cross_efficiency=de / dn,
                state=nxt,
            )


def _best_move(current: State, p1, p2, p3, values) -> GrowthMove | None:
    moves = list(_possible_moves(current, p1, p2, p3, values))
    if not moves:
        return None

    # Cross-market value is intentionally only auxiliary after q-efficiency.
    return max(
        moves,
        key=lambda m: (
            m.q_efficiency,
            m.bridge_efficiency,
            m.cross_efficiency,
            -m.added_ticket_count,
            -m.place,
            -m.rider,
        ),
    )


def _grow_path(seed: State, p1, p2, p3, values):
    states = [seed]
    moves: list[GrowthMove] = []
    cur = seed
    while True:
        move = _best_move(cur, p1, p2, p3, values)
        if move is None:
            break
        moves.append(move)
        cur = move.state
        states.append(cur)
    return states, moves


def _select_growth_knee(states: list[State]) -> int:
    """Return selected state index on cumulative q-mass versus point-count path."""
    if not states:
        raise ValueError("empty growth path")
    if len(states) == 1:
        return 0

    first = states[0]
    last = states[-1]
    dn = last.ticket_count - first.ticket_count
    dq = last.q_mass - first.q_mass
    if dn <= 0 or dq <= 1e-15:
        return 0

    def knee_score(i: int):
        s = states[i]
        x = (s.ticket_count - first.ticket_count) / dn
        y = (s.q_mass - first.q_mass) / dq
        return y - x

    # The curve itself determines the width. Bridge density only resolves ties.
    return max(
        range(len(states)),
        key=lambda i: (
            knee_score(i),
            states[i].bridge_mass / states[i].ticket_count,
            states[i].q_mass / states[i].ticket_count,
            -states[i].ticket_count,
            tuple(-x for x in states[i].first),
            tuple(-x for x in states[i].second),
            tuple(-x for x in states[i].third),
        ),
    )


def choose_incremental_growth(trio_odds, trifecta_odds, gate):
    values = cross_market_ticket_table(trio_odds, trifecta_odds)
    p1, p2, p3 = _structural_pools(trio_odds, gate)
    seed = _seed_state(p1, p2, p3, values)
    states, moves = _grow_path(seed, p1, p2, p3, values)
    selected_index = _select_growth_knee(states)
    return states[selected_index], states, moves, selected_index


def build_v8_4_f05(trio_odds, trifecta_odds, predicted_line_formation):
    gate = v6_1_entry_gate(trio_odds, trifecta_odds, predicted_line_formation)
    if not gate["entry_pass"]:
        return {"scheme_version": SCHEME_VERSION, "buy": False, "reason": gate["reason"]}

    try:
        chosen, states, moves, selected_index = choose_incremental_growth(
            trio_odds, trifecta_odds, gate
        )
    except ValueError as exc:
        return {"scheme_version": SCHEME_VERSION, "buy": False, "reason": str(exc)}

    trace = []
    for i, s in enumerate(states):
        row = {
            "step": i,
            "formation": s.display,
            "ticket_count": s.ticket_count,
            "q_mass": s.q_mass,
            "bridge_mass": s.bridge_mass,
            "q_star_mass": s.q_star_mass,
            "cross_excess": s.cross_excess,
            "selected": i == selected_index,
        }
        if i > 0:
            m = moves[i - 1]
            row.update({
                "added_place": m.place,
                "added_rider": m.rider,
                "added_ticket_count": m.added_ticket_count,
                "added_q_mass": m.added_q_mass,
                "marginal_q_efficiency": m.q_efficiency,
                "marginal_bridge_efficiency": m.bridge_efficiency,
                "marginal_cross_efficiency": m.cross_efficiency,
            })
        trace.append(row)

    return {
        "scheme_version": SCHEME_VERSION,
        "buy": True,
        "reason": "PASS",
        "formation": chosen.display,
        "first": chosen.first,
        "second": chosen.second,
        "third": chosen.third,
        "tickets": chosen.tickets,
        "ticket_count": chosen.ticket_count,
        "q_mass": chosen.q_mass,
        "bridge_mass": chosen.bridge_mass,
        "q_star_mass": chosen.q_star_mass,
        "cross_excess": chosen.cross_excess,
        "selected_growth_step": selected_index,
        "growth_steps_total": len(states) - 1,
        "growth_trace": trace,
        "entry_gate": gate,
        "price_cut_enabled": False,
        "fixed_place_counts": False,
        "fixed_point_count": False,
        "nested_required": False,
        "formation_selector": "incremental one-rider growth; marginal q efficiency with cross-market auxiliary; cumulative q-mass knee",
    }
