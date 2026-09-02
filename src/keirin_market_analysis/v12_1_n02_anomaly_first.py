from __future__ import annotations

"""v12.1-N02: ANOMALY FIRST.

Race selection is the first decision.

The useful v12 market-connection feature idea is retained, but the rejected rule
"any two-view overlap => buy" is gone. A race is eligible only when the complete
market contains a robustly extreme cross-market connection anomaly with a coherent
pre-result interpretation.

No Q1/Q2/Q3 outcome, payout, race-type branch, fitted ticket cap, or line-based
betting rule is used.
"""

from statistics import median
from typing import Iterable, Mapping

from market_connection_features_v1 import extract_market_connection_features

SCHEME_VERSION = "v12.1-N02"
STATUS = "DESIGN_FROZEN_PRE_SIMULATION"
MODIFIED_Z_OUTLIER = 3.5  # conventional robust-outlier criterion, not fitted to keirin outcomes


def _modified_z(values: list[float]) -> list[float]:
    if not values:
        return []
    med = median(values)
    deviations = [abs(v - med) for v in values]
    mad = median(deviations)
    if mad <= 1e-15:
        return [0.0 for _ in values]
    return [0.6745 * (v - med) / mad for v in values]


def _probability_top_pairs(pair_values: dict[tuple[int, int], float]) -> set[tuple[int, int]]:
    """Descriptive natural upper tier for one head's 15 trio companion pairs."""
    order = sorted(pair_values, key=lambda k: (-pair_values[k], k))
    if len(order) <= 1:
        return set(order)
    ranked = [pair_values[k] for k in order]
    if max(ranked) - min(ranked) <= 1e-15:
        return set()
    ratios = [ranked[i] / ranked[i + 1] for i in range(len(ranked) - 1)]
    largest = max(ratios)
    eps = 1e-12
    cut = max(i + 1 for i, r in enumerate(ratios) if abs(r - largest) <= eps)
    return set(order[:cut])


def build_v12_1_n02(
    trio_odds: Mapping[tuple[int, int, int], float],
    trifecta_odds: Mapping[tuple[int, int, int], float],
    entry_rows: Iterable[dict],
    race_type: str | None = None,
) -> dict[str, object]:
    base: dict[str, object] = {
        "scheme_version": SCHEME_VERSION,
        "development_status": STATUS,
        "buy": False,
        "reason": "UNINITIALIZED",
        "tickets": [],
        "ticket_count": 0,
        "race_type_metadata_only": race_type,
        "race_type_specific_branch": False,
        "result_used": False,
        "payout_used": False,
        "q1_fitted_cutoff": False,
        "line_based_ticket_rule": False,
        "race_selection_first": True,
    }

    features = extract_market_connection_features(trio_odds, trifecta_odds, entry_rows)
    if not features.get("ok"):
        base["reason"] = str(features.get("reason", "FEATURE_EXTRACTION_FAILED"))
        return base

    connections = list(features["connections"])
    residuals = [float(c["market_connection_log_residual"]) for c in connections]
    zscores = _modified_z(residuals)

    market_heads = {int(x) for x in features["market_head_block"]}
    fundamental_heads = {int(x) for x in features["fundamental_head_block"]}

    # For each head, define which unordered three-rider sets the trio market itself
    # treats as its natural upper package. This is descriptive context, not enough
    # by itself to authorize a bet.
    trio_top_by_head: dict[int, set[tuple[int, int]]] = {}
    for head in features["cars"]:
        pair_values: dict[tuple[int, int], float] = {}
        for c in connections:
            if int(c["head"]) != int(head):
                continue
            pair = tuple(sorted((int(c["a"]), int(c["b"]))))
            pair_values[pair] = float(c["trio_set_pair"])
        trio_top_by_head[int(head)] = _probability_top_pairs(pair_values)

    selected: list[dict[str, object]] = []
    diagnostics: list[dict[str, object]] = []

    for c, z in zip(connections, zscores):
        head = int(c["head"])
        pair = tuple(sorted((int(c["a"]), int(c["b"]))))
        residual = float(c["market_connection_log_residual"])

        diag = {
            "head": head,
            "pair": list(pair),
            "residual": residual,
            "modified_z": z,
            "market_head_supported": head in market_heads,
            "fundamental_head_supported": head in fundamental_heads,
            "trio_pair_top_package": pair in trio_top_by_head.get(head, set()),
        }
        diagnostics.append(diag)

        # Race-selection thesis:
        # The unordered trio market strongly packages {head,a,b}, but the exact
        # trifecta market attaches that same pair to this head abnormally LESS than
        # the race's own normal connection pattern would imply. If both the market
        # head hierarchy and the independent race-card head hierarchy support h,
        # this becomes an under-attached head connection worth expressing.
        if z > -MODIFIED_Z_OUTLIER:
            continue
        if residual >= 0:
            continue
        if head not in market_heads:
            continue
        if head not in fundamental_heads:
            continue
        if pair not in trio_top_by_head.get(head, set()):
            continue

        selected.append({**diag, "tail_order_log_asymmetry": float(c["tail_order_log_asymmetry"])})

    if not selected:
        base.update(
            {
                "reason": "NO_ROBUST_UNDERATTACHED_CONNECTION_ANOMALY",
                "market_head_block": sorted(market_heads),
                "fundamental_head_block": sorted(fundamental_heads),
                "selected_connections": [],
                "connection_diagnostics": diagnostics,
            }
        )
        return base

    # Ticket construction happens only AFTER the race has passed the anomaly gate.
    # The thesis identifies the head and unordered companion pair, not exact tail
    # order, so both tail orders are kept for each selected anomalous connection.
    tickets: set[tuple[int, int, int]] = set()
    for c in selected:
        h = int(c["head"])
        a, b = (int(x) for x in c["pair"])
        tickets.add((h, a, b))
        tickets.add((h, b, a))

    ordered = sorted(tickets)
    base.update(
        {
            "buy": True,
            "reason": "N02_ROBUST_UNDERATTACHED_CONNECTION",
            "tickets": ordered,
            "ticket_count": len(ordered),
            "market_head_block": sorted(market_heads),
            "fundamental_head_block": sorted(fundamental_heads),
            "selected_connections": selected,
            "connection_diagnostics": diagnostics,
            "race_gate": {
                "type": "ROBUST_WITHIN_RACE_CONNECTION_ANOMALY",
                "modified_z_cutoff": MODIFIED_Z_OUTLIER,
                "cutoff_origin": "conventional robust-outlier criterion; not chosen from keirin results",
                "requires_negative_cross_market_residual": True,
                "requires_market_head_support": True,
                "requires_fundamental_head_support": True,
                "requires_trio_top_package": True,
            },
            "formation_type": "ANOMALOUS_CONNECTION_TAIL_SWAP",
        }
    )
    return base
