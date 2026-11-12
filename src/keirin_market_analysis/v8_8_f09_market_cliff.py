from __future__ import annotations

"""v8.8-F09: PS_AB entry + market-cliff compact rectangular formation.

Decision inputs are pre-race market odds and predicted line formation only.
No result or payout is used by this engine.
"""

from dataclasses import dataclass
from itertools import product
from math import exp, log
from statistics import median
from typing import Iterable, Mapping

from v7_0_f01_market_hierarchy import (
    implied_probabilities,
    parse_lines,
    positional_support,
    trio_implied_probabilities,
)
from v8_0_f01_rectangular_hierarchy import _structural_pools

SCHEME_VERSION = "v8.8-F09"
PRICE_CUT_ENABLED = False
FIXED_PLACE_COUNTS = False
FIXED_POINT_COUNT = False
NESTED_REQUIRED = False
STAKE_YEN_PER_TICKET = 100


@dataclass(frozen=True)
class State:
    first: tuple[int, ...]
    second: tuple[int, ...]
    third: tuple[int, ...]
    tickets: tuple[tuple[int, int, int], ...]
    q_mass: float

    @property
    def ticket_count(self) -> int:
        return len(self.tickets)

    @property
    def display(self) -> str:
        show = lambda xs: "".join(str(x) for x in sorted(xs))
        return f"{show(self.first)}-{show(self.second)}-{show(self.third)}"


@dataclass(frozen=True)
class GrowthMove:
    place: int
    rider: int
    added_tickets: tuple[tuple[int, int, int], ...]
    added_q_mass: float
    marginal_density: float
    block_gm_q: float
    state: State

    @property
    def added_ticket_count(self) -> int:
        return len(self.added_tickets)


def _tickets(first: Iterable[int], second: Iterable[int], third: Iterable[int]):
    return tuple(sorted({
        (a, b, c)
        for a, b, c in product(first, second, third)
        if len({a, b, c}) == 3
    }))


def _state(first, second, third, q: Mapping[tuple[int, int, int], float]) -> State | None:
    f1 = tuple(sorted(set(first)))
    f2 = tuple(sorted(set(second)))
    f3 = tuple(sorted(set(third)))
    ts = _tickets(f1, f2, f3)
    if not ts:
        return None
    if {t[0] for t in ts} != set(f1):
        return None
    if {t[1] for t in ts} != set(f2):
        return None
    if {t[2] for t in ts} != set(f3):
        return None
    return State(f1, f2, f3, ts, sum(q[t] for t in ts))


def ps_ab_gate(
    trio_odds: Mapping[tuple[int, int, int], float],
    trifecta_odds: Mapping[tuple[int, int, int], float],
    predicted_line_formation: str,
) -> dict[str, object]:
    """New hard gate: only PS_AB rejects. H_AB/H_RATIO are diagnostics."""
    p3 = trio_implied_probabilities(trio_odds)
    q = implied_probabilities(trifecta_odds)
    cars = sorted({car for combo in trio_odds for car in combo})
    lines = parse_lines(predicted_line_formation)
    if lines is None or set(car for line in lines for car in line) != set(cars):
        return {"entry_pass": False, "reason": "INVALID_LINE"}

    s = {
        car: sum(prob for combo, prob in p3.items() if car in combo) / 3.0
        for car in cars
    }
    ls = [sum(s[car] for car in line) for line in lines]
    line_order = sorted(range(len(lines)), key=lambda idx: (-ls[idx], idx))
    if len(line_order) < 2:
        return {"entry_pass": False, "reason": "LT2_LINES"}
    ai, bi = line_order[:2]

    pair_rows: list[tuple[float, int, int, tuple[int, int]]] = []
    for li, line in enumerate(lines):
        for pos in range(len(line) - 1):
            pair = (line[pos], line[pos + 1])
            ps = sum(
                prob for combo, prob in p3.items()
                if pair[0] in combo and pair[1] in combo
            )
            pair_rows.append((ps, li, pos, pair))
    pair_rows.sort(key=lambda row: (-row[0], row[1], row[2]))
    if len(pair_rows) < 2 or {pair_rows[0][1], pair_rows[1][1]} != {ai, bi}:
        return {"entry_pass": False, "reason": "PS_TOP2_NOT_AB"}

    h1, _, _ = positional_support(q, cars)
    ranked_head = sorted(cars, key=lambda car: (-h1[car], car))
    top2 = ranked_head[:2]
    line_of = {car: li for li, line in enumerate(lines) for car in line}
    h_ab = {line_of[top2[0]], line_of[top2[1]]} == {ai, bi}
    h_ratio = h1[top2[0]] < 2.0 * h1[top2[1]]
    h_state = f"HAB{int(h_ab)}_HRATIO{int(h_ratio)}"

    return {
        "entry_pass": True,
        "reason": "PASS",
        "A_line": lines[ai],
        "B_line": lines[bi],
        "head_rank": tuple(ranked_head),
        "H_AB": h_ab,
        "H_RATIO": h_ratio,
        "H_state": h_state,
        "top2_H": tuple(top2),
        "H1_top": h1[top2[0]],
        "H1_second": h1[top2[1]],
    }


def _seed_state(p1, p2, p3, q) -> State:
    candidates: list[State] = []
    for a, b, c in product(p1, p2, p3):
        if len({a, b, c}) != 3:
            continue
        s = _state((a,), (b,), (c,), q)
        if s is not None:
            candidates.append(s)
    if not candidates:
        raise ValueError("NO_VALID_SEED")
    return max(
        candidates,
        key=lambda s: (s.q_mass, tuple(-x for x in s.tickets[0])),
    )


def _gm_q(added, q) -> float:
    vals = [q[t] for t in added if q[t] > 0]
    if not vals or len(vals) != len(added):
        return 0.0
    return exp(sum(log(v) for v in vals) / len(vals))


def _possible_moves(current: State, p1, p2, p3, q):
    old = set(current.tickets)
    pools = {1: tuple(sorted(set(p1))), 2: tuple(sorted(set(p2))), 3: tuple(sorted(set(p3)))}
    cursets = {1: current.first, 2: current.second, 3: current.third}
    for place in (1, 2, 3):
        for rider in pools[place]:
            if rider in cursets[place]:
                continue
            f1, f2, f3 = current.first, current.second, current.third
            if place == 1:
                f1 = tuple(sorted(set(f1) | {rider}))
            elif place == 2:
                f2 = tuple(sorted(set(f2) | {rider}))
            else:
                f3 = tuple(sorted(set(f3) | {rider}))
            nxt = _state(f1, f2, f3, q)
            if nxt is None:
                continue
            added = tuple(sorted(set(nxt.tickets) - old))
            if not added:
                continue
            dq = sum(q[t] for t in added)
            yield GrowthMove(
                place=place,
                rider=rider,
                added_tickets=added,
                added_q_mass=dq,
                marginal_density=dq / len(added),
                block_gm_q=_gm_q(added, q),
                state=nxt,
            )


def _best_move(current: State, p1, p2, p3, q) -> GrowthMove | None:
    moves = list(_possible_moves(current, p1, p2, p3, q))
    if not moves:
        return None
    # Primary: marginal q per added ticket. Tie: 1st -> 2nd -> 3rd, smaller rider.
    return max(moves, key=lambda m: (m.marginal_density, -m.place, -m.rider))


def _grow_path(seed: State, p1, p2, p3, q):
    states = [seed]
    moves: list[GrowthMove] = []
    cur = seed
    while True:
        move = _best_move(cur, p1, p2, p3, q)
        if move is None:
            break
        moves.append(move)
        cur = move.state
        states.append(cur)
    return states, moves


def _select_market_cliff(states: list[State], moves: list[GrowthMove]):
    """Select state immediately before largest adjacent GM-support drop."""
    if not states:
        raise ValueError("EMPTY_GROWTH_PATH")
    if not moves:
        return 0, None, tuple()
    if len(moves) == 1:
        return 1, None, tuple()

    drops = tuple(
        (moves[i].block_gm_q / moves[i + 1].block_gm_q)
        if moves[i + 1].block_gm_q > 0 else float("inf")
        for i in range(len(moves) - 1)
    )
    # Earlier tie wins. Drop i is E_{i+1}/E_{i+2}; keep state i+1.
    max_drop = max(drops)
    cliff_i = next(i for i, d in enumerate(drops) if d == max_drop)
    selected_state_index = cliff_i + 1
    return selected_state_index, max_drop, drops


def _weighted_gm_odds(tickets, q, trifecta_odds):
    mass = sum(q[t] for t in tickets)
    if mass <= 0:
        return None
    return exp(sum((q[t] / mass) * log(float(trifecta_odds[t])) for t in tickets))


def build_v8_8_f09(trio_odds, trifecta_odds, predicted_line_formation):
    gate = ps_ab_gate(trio_odds, trifecta_odds, predicted_line_formation)
    if not gate["entry_pass"]:
        return {"scheme_version": SCHEME_VERSION, "buy": False, "reason": gate["reason"]}

    q = implied_probabilities(trifecta_odds)
    p1, p2, p3 = _structural_pools(trio_odds, gate)
    try:
        seed = _seed_state(p1, p2, p3, q)
        states, moves = _grow_path(seed, p1, p2, p3, q)
        selected_index, cliff_drop, drops = _select_market_cliff(states, moves)
    except ValueError as exc:
        return {"scheme_version": SCHEME_VERSION, "buy": False, "reason": str(exc)}

    chosen = states[selected_index]
    n = chosen.ticket_count
    profit_mass_1x = sum(q[t] for t in chosen.tickets if float(trifecta_odds[t]) >= n)
    profit_mass_2x = sum(q[t] for t in chosen.tickets if float(trifecta_odds[t]) >= 2 * n)
    odds_vals = [float(trifecta_odds[t]) for t in chosen.tickets]

    trace = []
    for i, state in enumerate(states):
        row = {
            "step": i,
            "formation": state.display,
            "ticket_count": state.ticket_count,
            "q_mass": state.q_mass,
            "selected": i == selected_index,
        }
        if i > 0:
            move = moves[i - 1]
            row.update({
                "added_place": move.place,
                "added_rider": move.rider,
                "added_ticket_count": move.added_ticket_count,
                "added_q_mass": move.added_q_mass,
                "marginal_density": move.marginal_density,
                "block_gm_q": move.block_gm_q,
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
        "ticket_count": n,
        "q_mass": chosen.q_mass,
        "selected_growth_step": selected_index,
        "growth_steps_total": len(moves),
        "cliff_drop": cliff_drop,
        "all_cliff_drops": drops,
        "profit_mass_1x": profit_mass_1x,
        "profit_mass_2x": profit_mass_2x,
        "median_selected_odds": median(odds_vals),
        "weighted_gm_selected_odds": _weighted_gm_odds(chosen.tickets, q, trifecta_odds),
        "H_AB": gate["H_AB"],
        "H_RATIO": gate["H_RATIO"],
        "H_state": gate["H_state"],
        "entry_gate": gate,
        "growth_trace": trace,
        "price_cut_enabled": PRICE_CUT_ENABLED,
        "fixed_place_counts": FIXED_PLACE_COUNTS,
        "fixed_point_count": FIXED_POINT_COUNT,
        "nested_required": NESTED_REQUIRED,
        "formation_selector": "marginal q-density growth; stop immediately before largest adjacent geometric-mean q block-support drop",
    }
