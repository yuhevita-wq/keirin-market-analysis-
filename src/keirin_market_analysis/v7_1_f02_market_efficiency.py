from __future__ import annotations

"""v7.1-F02: variable-size nested formation chosen by market efficiency.

Core rule:
- Keep the unchanged v6.1 pre-formation entry gate.
- Do not pre-fix rider counts for 1st/2nd/3rd.
- Do not use any odds>N price cut.
- Enumerate market-ranked nested formations and choose the one that maximizes
  market concentration above a uniform 210-ticket baseline.

No result or payout data is used here.
"""

from dataclasses import dataclass
from typing import Mapping, Sequence

from v7_0_f01_market_hierarchy import (
    generate_nested_tickets,
    implied_probabilities,
    positional_support,
    v6_1_entry_gate,
)

SCHEME_VERSION = "v7.1-F02"
BASE_ENTRY_SCHEME_VERSION = "v6.1"
PRICE_CUT_ENABLED = False
FIXED_PLACE_COUNTS = False
STAKE_YEN_PER_TICKET = 100
FULL_TRIFECTA_TICKETS = 210


@dataclass(frozen=True)
class EfficiencyFormation:
    first: tuple[int, ...]
    second: tuple[int, ...]
    third: tuple[int, ...]
    tickets: tuple[tuple[int, int, int], ...]
    formation_mass: float
    uniform_mass: float
    excess_mass: float
    raw_k1: int
    raw_k2: int
    raw_k3: int

    @property
    def ticket_count(self) -> int:
        return len(self.tickets)

    @property
    def stake_yen(self) -> int:
        return self.ticket_count * STAKE_YEN_PER_TICKET

    @property
    def display(self) -> str:
        def show(xs: Sequence[int]) -> str:
            return "".join(str(x) for x in sorted(xs))
        return f"{show(self.first)}-{show(self.second)}-{show(self.third)}"


def _rank(scores: Mapping[int, float]) -> tuple[int, ...]:
    return tuple(sorted(scores, key=lambda car: (-scores[car], car)))


def _union(*groups: Sequence[int]) -> tuple[int, ...]:
    return tuple(sorted({car for group in groups for car in group}))


def choose_market_efficiency_formation(
    trifecta_odds: Mapping[tuple[int, int, int], float],
    cars: Sequence[int] | None = None,
) -> EfficiencyFormation:
    """Choose a clean nested F1-F2-F3 without fixed place counts.

    For each k1/k2/k3 from 1..number of riders:
      R1 = top-k1 by 1st-place support H1
      R2 = top-k2 by 2nd-place support H2
      R3 = top-k3 by 3rd-place support H3
      F1 = R1
      F2 = F1 union R2
      F3 = F2 union R3

    Each distinct valid nested formation gets:
      market_mass = sum normalized q(t) over its tickets
      uniform_mass = ticket_count / 210
      excess_mass = market_mass - uniform_mass

    The selected formation maximizes excess_mass.  This rewards concentrated
    market support but penalizes adding tickets at exactly the rate a uniform
    210-ticket market would gain mass.  Therefore no rider-count or ticket-count
    target is fixed in advance.
    """
    q = implied_probabilities(trifecta_odds)
    if cars is None:
        cars = tuple(sorted({car for ticket in trifecta_odds for car in ticket}))
    else:
        cars = tuple(sorted(set(cars)))
    if len(cars) < 3:
        raise ValueError("at least three riders are required")

    h1, h2, h3 = positional_support(q, cars)
    r1, r2, r3 = _rank(h1), _rank(h2), _rank(h3)
    ncar = len(cars)

    unique: dict[tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]], EfficiencyFormation] = {}
    for k1 in range(1, ncar + 1):
        raw1 = r1[:k1]
        for k2 in range(1, ncar + 1):
            raw2 = r2[:k2]
            f1 = tuple(sorted(raw1))
            f2 = _union(f1, raw2)
            for k3 in range(1, ncar + 1):
                raw3 = r3[:k3]
                f3 = _union(f2, raw3)
                tickets = generate_nested_tickets(f1, f2, f3)
                if not tickets:
                    continue
                key = (f1, f2, f3)
                if key in unique:
                    continue
                mass = sum(q.get(ticket, 0.0) for ticket in tickets)
                uniform = len(tickets) / FULL_TRIFECTA_TICKETS
                unique[key] = EfficiencyFormation(
                    first=f1,
                    second=f2,
                    third=f3,
                    tickets=tickets,
                    formation_mass=mass,
                    uniform_mass=uniform,
                    excess_mass=mass - uniform,
                    raw_k1=k1,
                    raw_k2=k2,
                    raw_k3=k3,
                )

    if not unique:
        raise ValueError("no valid nested formation generated")

    # Higher excess mass wins.  Exact ties prefer fewer tickets, then fewer
    # total rider slots, then deterministic lexical order.
    return max(
        unique.values(),
        key=lambda x: (
            x.excess_mass,
            -x.ticket_count,
            -(len(x.first) + len(x.second) + len(x.third)),
            tuple(-v for v in x.first + x.second + x.third),
        ),
    )


def build_v7_1_f02(
    trio_odds: Mapping[tuple[int, int, int], float],
    trifecta_odds: Mapping[tuple[int, int, int], float],
    predicted_line_formation: str,
) -> dict[str, object]:
    gate = v6_1_entry_gate(trio_odds, trifecta_odds, predicted_line_formation)
    if not gate["entry_pass"]:
        return {
            "scheme_version": SCHEME_VERSION,
            "buy": False,
            "reason": gate["reason"],
        }

    cars = sorted({car for combo in trio_odds for car in combo})
    f = choose_market_efficiency_formation(trifecta_odds, cars)
    return {
        "scheme_version": SCHEME_VERSION,
        "buy": True,
        "reason": "PASS",
        "formation": f.display,
        "first": f.first,
        "second": f.second,
        "third": f.third,
        "tickets": f.tickets,
        "ticket_count": f.ticket_count,
        "stake_yen": f.stake_yen,
        "formation_mass": f.formation_mass,
        "uniform_mass": f.uniform_mass,
        "excess_mass": f.excess_mass,
        "raw_rank_depths": (f.raw_k1, f.raw_k2, f.raw_k3),
        "entry_gate": gate,
        "price_cut_enabled": PRICE_CUT_ENABLED,
        "fixed_place_counts": FIXED_PLACE_COUNTS,
    }
