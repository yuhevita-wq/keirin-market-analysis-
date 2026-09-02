from __future__ import annotations

"""v12.0-N01: MARKET CONNECTION TOPOLOGY.

Fresh scheme root. It does not inherit v8/v9/v10/v11 betting logic.

Core idea:
- stop treating line labels as the answer;
- read the market as a directed connection network;
- for each plausible market head, evaluate every companion pair from three views:
  1) trifecta head-conditional pair attachment,
  2) trio set attachment,
  3) independent race-card fundamental pair strength;
- retain only pair connections supported by at least two informative views;
- buy both tail orders for every retained connection.

No result, payout, fitted threshold, race-type branch, or line-based ticket rule is used.
Line position/size enter only through the independent fundamental F utility.
"""

from itertools import combinations, permutations
from math import isfinite
from typing import Iterable, Mapping

from racecard_fundamentals_v1 import fundamental_scores

SCHEME_VERSION = "v12.0-N01"
STATUS = "DESIGN_FROZEN_PRE_SIMULATION"


def _positive_float(x) -> float | None:
    try:
        v = float(x)
    except Exception:
        return None
    return v if isfinite(v) and v > 0 else None


def _normalize_inverse_odds(odds: Mapping[tuple[int, ...], float]) -> dict[tuple[int, ...], float] | None:
    inv: dict[tuple[int, ...], float] = {}
    for key, value in odds.items():
        v = _positive_float(value)
        if v is None:
            return None
        inv[tuple(key)] = 1.0 / v
    z = sum(inv.values())
    if z <= 0:
        return None
    return {k: v / z for k, v in inv.items()}


def _probability_top_block(values: Mapping[object, float]) -> tuple[object, ...]:
    """Natural upper tier from the largest adjacent ratio drop.

    If the view is completely flat, the whole set is returned and the caller may
    treat that view as uninformative rather than inventing a cutoff.
    """
    order = sorted(values, key=lambda k: (-float(values[k]), repr(k)))
    if len(order) <= 1:
        return tuple(order)
    ranked = [float(values[k]) for k in order]
    if max(ranked) - min(ranked) <= 1e-15:
        return tuple(order)
    ratios = [ranked[i] / ranked[i + 1] for i in range(len(ranked) - 1)]
    largest = max(ratios)
    eps = 1e-12
    cut = max(i + 1 for i, r in enumerate(ratios) if abs(r - largest) <= eps)
    return tuple(order[:cut])


def _fundamental_top_block(values: Mapping[object, float]) -> tuple[object, ...]:
    """Natural upper tier from the largest adjacent absolute F gap."""
    order = sorted(values, key=lambda k: (-float(values[k]), repr(k)))
    if len(order) <= 1:
        return tuple(order)
    ranked = [float(values[k]) for k in order]
    gaps = [ranked[i] - ranked[i + 1] for i in range(len(ranked) - 1)]
    largest = max(gaps)
    if largest <= 1e-15:
        return tuple(order)
    eps = 1e-12
    cut = max(i + 1 for i, g in enumerate(gaps) if abs(g - largest) <= eps)
    return tuple(order[:cut])


def _validate_markets(
    trio_odds: Mapping[tuple[int, int, int], float],
    trifecta_odds: Mapping[tuple[int, int, int], float],
) -> tuple[bool, str, tuple[int, ...]]:
    if len(trio_odds) != 35:
        return False, "TRIO_NOT_COMPLETE_35", ()
    if len(trifecta_odds) != 210:
        return False, "TRIFECTA_NOT_COMPLETE_210", ()

    cars = tuple(sorted({c for t in trifecta_odds for c in t}))
    if len(cars) != 7:
        return False, "NOT_SEVEN_CARS", ()

    expected_trio = {tuple(c) for c in combinations(cars, 3)}
    actual_trio = {tuple(sorted(t)) for t in trio_odds}
    if actual_trio != expected_trio:
        return False, "TRIO_COMBINATION_SET_INVALID", ()

    expected_tf = set(permutations(cars, 3))
    actual_tf = {tuple(t) for t in trifecta_odds}
    if actual_tf != expected_tf:
        return False, "TRIFECTA_PERMUTATION_SET_INVALID", ()

    if any(_positive_float(v) is None for v in trio_odds.values()):
        return False, "TRIO_ODDS_INVALID", ()
    if any(_positive_float(v) is None for v in trifecta_odds.values()):
        return False, "TRIFECTA_ODDS_INVALID", ()

    return True, "OK", cars


def build_v12_0_n01(
    trio_odds: Mapping[tuple[int, int, int], float],
    trifecta_odds: Mapping[tuple[int, int, int], float],
    entry_rows: Iterable[dict],
    race_type: str | None = None,
) -> dict[str, object]:
    ok, reason, cars = _validate_markets(trio_odds, trifecta_odds)
    base: dict[str, object] = {
        "scheme_version": SCHEME_VERSION,
        "development_status": STATUS,
        "buy": False,
        "reason": reason,
        "tickets": [],
        "ticket_count": 0,
        "race_type_metadata_only": race_type,
        "race_type_specific_branch": False,
        "result_used": False,
        "payout_used": False,
        "fitted_numeric_cutoff": False,
        "line_based_ticket_rule": False,
        "prediction_mark_used": False,
        "evaluation_mark_used": False,
    }
    if not ok:
        return base

    p3_raw = _normalize_inverse_odds({tuple(sorted(k)): v for k, v in trio_odds.items()})
    q = _normalize_inverse_odds(trifecta_odds)
    if p3_raw is None or q is None:
        base["reason"] = "ODDS_NORMALIZATION_FAILED"
        return base

    fd = fundamental_scores(entry_rows)
    if not fd.get("ok"):
        base["reason"] = str(fd.get("reason", "FUNDAMENTAL_INPUT_INVALID"))
        return base
    f = {int(k): float(v) for k, v in fd["F"].items()}
    if set(f) != set(cars):
        base["reason"] = "FUNDAMENTAL_CAR_SET_MISMATCH"
        return base

    # 1st-place market support.
    h = {car: sum(prob for t, prob in q.items() if t[0] == car) for car in cars}
    head_block = _probability_top_block(h)
    head_informative = len(head_block) < len(cars)
    if not head_informative:
        base["reason"] = "HEAD_VIEW_FLAT_NO_STRUCTURE"
        base["head_support"] = h
        base["head_block"] = list(head_block)
        return base

    tickets: set[tuple[int, int, int]] = set()
    head_diagnostics: dict[str, object] = {}

    for head_obj in head_block:
        head = int(head_obj)
        others = [c for c in cars if c != head]
        pairs = [tuple(pair) for pair in combinations(others, 2)]

        h_mass = h[head]
        if h_mass <= 0:
            continue

        # View A: among outcomes where HEAD wins, which companion pair is attached?
        tf_pair = {
            pair: (q[(head, pair[0], pair[1])] + q[(head, pair[1], pair[0])]) / h_mass
            for pair in pairs
        }

        # View B: in the trio market, which companion pair is attached to HEAD?
        trio_raw = {pair: p3_raw[tuple(sorted((head, pair[0], pair[1])))] for pair in pairs}
        trio_z = sum(trio_raw.values())
        trio_pair = {pair: value / trio_z for pair, value in trio_raw.items()}

        # View C: independent race-card strength of the two companions.
        f_pair = {pair: f[pair[0]] + f[pair[1]] for pair in pairs}

        tf_top = set(_probability_top_block(tf_pair))
        trio_top = set(_probability_top_block(trio_pair))
        f_top = set(_fundamental_top_block(f_pair))

        views = {
            "TRIFECTA_HEAD_PAIR": tf_top,
            "TRIO_SET_PAIR": trio_top,
            "FUNDAMENTAL_PAIR": f_top,
        }
        informative = {
            name: block for name, block in views.items() if len(block) < len(pairs)
        }

        # A connection requires at least two genuinely informative views.
        if len(informative) < 2:
            head_diagnostics[str(head)] = {
                "status": "INSUFFICIENT_INFORMATIVE_PAIR_VIEWS",
                "tf_top": [list(x) for x in sorted(tf_top)],
                "trio_top": [list(x) for x in sorted(trio_top)],
                "f_top": [list(x) for x in sorted(f_top)],
            }
            continue

        accepted: list[tuple[int, int]] = []
        signatures: dict[str, list[str]] = {}
        for pair in pairs:
            supporters = [name for name, block in informative.items() if pair in block]
            if len(supporters) >= 2:
                accepted.append(pair)
                signatures[f"{pair[0]}-{pair[1]}"] = supporters
                # Do not pretend F can resolve 2nd vs 3rd. Keep both exact orders.
                tickets.add((head, pair[0], pair[1]))
                tickets.add((head, pair[1], pair[0]))

        head_diagnostics[str(head)] = {
            "status": "CONNECTIONS_FOUND" if accepted else "NO_TWO_VIEW_CONNECTION",
            "accepted_pairs": [list(x) for x in accepted],
            "connection_supporters": signatures,
            "tf_top": [list(x) for x in sorted(tf_top)],
            "trio_top": [list(x) for x in sorted(trio_top)],
            "f_top": [list(x) for x in sorted(f_top)],
            "informative_views": sorted(informative),
        }

    if not tickets:
        base["reason"] = "NO_TWO_VIEW_MARKET_CONNECTION"
        base["head_support"] = h
        base["head_block"] = list(head_block)
        base["head_diagnostics"] = head_diagnostics
        return base

    ordered_tickets = sorted(tickets)
    base.update(
        {
            "buy": True,
            "reason": "N01_CONNECTION_TOPOLOGY_BUY",
            "tickets": ordered_tickets,
            "ticket_count": len(ordered_tickets),
            "head_support": h,
            "head_block": list(head_block),
            "head_diagnostics": head_diagnostics,
            "fundamental_F": {str(k): v for k, v in f.items()},
            "formation_type": "DIRECTED_CONNECTION_PATH_SET",
            "ticket_generation_note": (
                "Each ticket comes from a companion pair supported by at least two informative views; "
                "both tail orders are retained. No line label chooses or deletes a ticket."
            ),
        }
    )
    return base
