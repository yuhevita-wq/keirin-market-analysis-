"""3連単フォーメーション: structural grammar based on the 40-page user Library manual.

MANUAL: keirin_3rentan_formation_manual_latest.pdf (2026-10-30), especially
pp. 3-4, 9-21, 29-36. Treat the manual as a specification for representing
a scenario by a SET of ordered 3-rider outcomes, not a profitable-bet recipe.

This module is independent of V6 and V6-C1. It must never look at observed
results, payout labels, odds, or a trainer when CONSTRUCTING formations.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations, permutations, product
from typing import Mapping

Ticket = tuple[str, str, str]
ALL = "全"


def normalize_cars(cars):
    cars = tuple(str(x) for x in cars)
    if len(set(cars)) != len(cars) or len(cars) < 3 or len(cars) > 9:
        raise ValueError("Race must contain 3-9 unique riders")
    if any(not x.isascii() or len(x) != 1 or x not in "123456789" for x in cars):
        raise ValueError("Car numbers must be single-digit 1-9")
    return tuple(sorted(cars, key=int))


def _slot(x, universe):
    if isinstance(x, str):
        if x in (ALL, "ALL"):
            return universe
        x = tuple(x.replace("・", "").replace(" ", "").replace(",", ""))
    else:
        x = tuple(str(y) for y in x)
    if not x or len(set(x)) != len(x) or not set(x).issubset(universe):
        raise ValueError(f"Invalid slot {x}")
    return tuple(sorted(x, key=int))


@dataclass(frozen=True)
class Shape:
    """One named, separately expressible tactical scenario."""
    type: str
    why: str
    first: tuple[str, ...]
    second: tuple[str, ...]
    third: tuple[str, ...]
    tickets: frozenset[Ticket]
    notation: str

    @property
    def points(self):
        return len(self.tickets)


@dataclass(frozen=True)
class Package:
    """Union of multiple independent readings; no accidental cross product."""
    shapes: tuple[Shape, ...]
    tickets: frozenset[Ticket]
    overlap: frozenset[Ticket]
    ticket_sources: dict[Ticket, tuple[int, ...]] = field(compare=False)
    @property
    def points(self):
        return len(self.tickets)
    @property
    def separate_points(self):
        return sum(x.points for x in self.shapes)


def form(cars, first, second, third, *, type="custom", why=""):
    universe = normalize_cars(cars)
    a, b, c = (_slot(x, universe) for x in (first, second, third))
    tickets = frozenset(t for t in product(a, b, c) if len(set(t)) == 3)
    if not tickets:
        raise ValueError("No physically possible 3連単 outcomes")
    def label(x):
        return ALL if x == universe else "".join(x)
    return Shape(type, why, a, b, c, tickets,
                 "-".join((label(a), label(b), label(c))))


def box(cars, selected, *, why=""):
    universe = normalize_cars(cars)
    sel = _slot(selected, universe)
    if len(sel) < 3:
        raise ValueError("BOX needs 3+ distinct riders")
    tickets = frozenset(permutations(sel, 3))
    return Shape("box", why, sel, sel, sel, tickets, "".join(sel) + " BOX")


def top_two_flip(cars, pair, thirds=ALL, *, why="", type="one_two_flip"):
    pair = _slot(pair, normalize_cars(cars))
    if len(pair) != 2:
        raise ValueError("Exactly 2 one-two riders required")
    a, b = pair
    return form(cars, (a, b), (a, b), thirds, type=type, why=why)


def head_fixed(cars, head, others, *, why=""):
    return form(cars, head, others, others, type="first_fixed_with_23_swap", why=why)


def third_all(cars, first, second, *, why=""):
    return form(cars, first, second, ALL, type="third_all", why=why)


def sandwich(cars, first, third, seconds=ALL, *, why=""):
    return form(cars, first, seconds, third, type="sandwich_1st_3rd_fixed", why=why)


def second_fixed(cars, firsts, second, thirds, *, why=""):
    return form(cars, firsts, second, thirds, type="second_fixed", why=why)


def third_fixed(cars, firsts, seconds, third, *, why=""):
    return form(cars, firsts, seconds, third, type="third_fixed", why=why)


def together(shapes):
    shapes = tuple(shapes)
    if not shapes:
        raise ValueError("Empty bundle")
    sources = {}
    for i, sh in enumerate(shapes):
        for ticket in sh.tickets:
            sources.setdefault(ticket, []).append(i)
    overlap = frozenset(k for k, v in sources.items() if len(v) > 1)
    return Package(shapes, frozenset(sources), overlap,
                   {k: tuple(v) for k, v in sources.items()})


def from_manual_notation(cars, notation, *, why="manual example"):
    """Parse three numbered slots / ALL, BOX and exact 1=2 / 2=3 flips.

    Full complex mixed '=...' expressions cannot be interpreted without the
    author's intended expansion; reject rather than silently fabricate bets.
    """
    source = notation.replace("−", "-").replace("＝", "=").replace(" ", "")
    if source.upper().endswith("BOX"):
        return box(cars, source[:-3], why=why)
    parts = source.split("-")
    if len(parts) == 2 and "=" in parts[0] and parts[0].count("=") == 1:
        a, b = parts[0].split("=")
        c = parts[1]
        if len(a) != 1 or len(b) != 1:
            raise ValueError("Only unambiguous two-person 1st/2nd flips")
        return top_two_flip(cars, a + b, c, why=why)
    if len(parts) == 2 and "=" in parts[1] and parts[1].count("=") == 1:
        b, c = parts[1].split("=")
        a = parts[0]
        if len(b) != 1 or len(c) != 1:
            raise ValueError("Only unambiguous two-person 2nd/3rd flips")
        return form(cars, a, b+c, b+c, type="second_third_flip", why=why)
    if len(parts) == 3 and all("=" not in x for x in parts):
        return form(cars, *parts, type="manual_explicit_slots", why=why)
    raise ValueError("Notation ambiguous; specify explicit per-slot candidates")


def evaluate_set(tickets, probabilities: Mapping[Ticket, float], odds: Mapping[Ticket, float],
                 stake_yen=100):
    """Score the ENTIRE combined set, not a sum of isolated positive-EV tips.

    Stake is one stake_yen per DISTINCT ordered combination. A ticket with
    negative EV is allowed if the formation contains it as a structural hedge.
    """
    import math
    tickets = frozenset(tickets)
    if not tickets:raise ValueError("empty tickets")
    if set(probabilities)!=set(odds):
        raise ValueError("Not the same complete ordered market universe")
    if any(len(set(t))!=3 for t in probabilities):
        raise ValueError("Impossible repeated rider")
    if abs(sum(probabilities.values())-1)>1e-8 or any(
        not math.isfinite(p) or p<0 for p in probabilities.values()):
        raise ValueError("Probability model invalid")
    if any(not math.isfinite(o) or o<=1 for o in odds.values()):
        raise ValueError("Odds market invalid")
    if not tickets.issubset(probabilities):
        raise ValueError("Ticket outside market/model")
    if not isinstance(stake_yen, int) or stake_yen<=0:
        raise ValueError("Invalid per-ticket stake")
    total=stake_yen*len(tickets)
    pset=sum(probabilities[t] for t in tickets)
    expected_payout=sum(probabilities[t]*odds[t]*stake_yen for t in tickets)
    synth=sum(1/odds[t] for t in tickets)
    return {
        "points":len(tickets),
        "stake_yen":total,
        "model_hit_probability":pset,
        "model_expected_payout_yen":expected_payout,
        "model_expected_roi_pct":100*expected_payout/total,
        "composite_odds":1/synth,
        "ticket_ev_over_100_count":sum(probabilities[t]*odds[t]>=1 for t in tickets),
        "ticket_ev_under_100_count":sum(probabilities[t]*odds[t]<1 for t in tickets),
        "market_total_inverse_odds":sum(1/o for o in odds.values()),
    }


def verify_settlement(tickets, outcome: Ticket, official_payout_yen: int,
                      stake_yen=100):
    """Official payable winner is supplied separately, after construction."""
    if not isinstance(official_payout_yen,int) or official_payout_yen<0:
        raise ValueError("Invalid official paid winner")
    tickets=frozenset(tickets)
    hit=outcome in tickets
    spent=len(tickets)*stake_yen
    payout=official_payout_yen if hit else 0
    return {
        "points":len(tickets),
        "hit":hit,
        "investment_yen":spent,
        "actual_payout_yen":payout,
        "profit_yen":payout-spent,
        "actual_roi_pct":100*payout/spent if spent else None,
    }
