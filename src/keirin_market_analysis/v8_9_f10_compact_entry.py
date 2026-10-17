from __future__ import annotations

"""v8.9-F10: compact-entry wrapper around v8.8-F09 market-cliff formation.

Q1-development change from v8.8-F09:
- keep PS_AB as a hard entry condition;
- additionally require a concentrated first-place market:
      H1_top >= 2 * H1_second
  This is the complement of the old v6.1 H_RATIO condition.
- H_AB remains a classification/diagnostic variable, not a hard exclusion.
- H_POS remains removed.

Formation construction is unchanged from v8.8-F09:
market-center seed -> one-rider marginal-q growth -> stop before the largest
adjacent geometric-mean support drop (market cliff).

No result or payout is used by this engine. No fixed place counts, fixed ticket
count, individual odds pruning, or price cut is used.
"""

from v8_8_f09_market_cliff import build_v8_8_f09, ps_ab_gate

SCHEME_VERSION = "v8.9-F10"
BASE_FORMATION_SCHEME = "v8.8-F09"
PRICE_CUT_ENABLED = False
FIXED_PLACE_COUNTS = False
FIXED_POINT_COUNT = False
NESTED_REQUIRED = False


def compact_entry_gate(trio_odds, trifecta_odds, predicted_line_formation):
    """PS_AB + H-market concentration gate.

    H_CONCENTRATED is true when the strongest first-place support is at least
    twice the second strongest support. H_AB is retained only as a descriptor.
    """
    gate = ps_ab_gate(trio_odds, trifecta_odds, predicted_line_formation)
    if not gate.get("entry_pass"):
        return gate

    h_concentrated = float(gate["H1_top"]) >= 2.0 * float(gate["H1_second"])
    out = dict(gate)
    out["H_CONCENTRATED"] = h_concentrated
    out["entry_rule"] = "PS_AB and H1_top >= 2 * H1_second"

    if not h_concentrated:
        out["entry_pass"] = False
        out["reason"] = "H_NOT_CONCENTRATED"
        return out

    out["entry_pass"] = True
    out["reason"] = "PASS"
    return out


def build_v8_9_f10(trio_odds, trifecta_odds, predicted_line_formation):
    gate = compact_entry_gate(trio_odds, trifecta_odds, predicted_line_formation)
    if not gate.get("entry_pass"):
        return {
            "scheme_version": SCHEME_VERSION,
            "buy": False,
            "reason": gate.get("reason"),
            "entry_gate": gate,
            "price_cut_enabled": PRICE_CUT_ENABLED,
            "fixed_place_counts": FIXED_PLACE_COUNTS,
            "fixed_point_count": FIXED_POINT_COUNT,
            "nested_required": NESTED_REQUIRED,
        }

    # Formation logic stays exactly v8.8-F09 after the stricter entry passes.
    result = build_v8_8_f09(trio_odds, trifecta_odds, predicted_line_formation)
    if not result.get("buy"):
        return {
            **result,
            "scheme_version": SCHEME_VERSION,
            "entry_gate": gate,
        }

    result = dict(result)
    result["scheme_version"] = SCHEME_VERSION
    result["base_formation_scheme"] = BASE_FORMATION_SCHEME
    result["entry_gate"] = gate
    result["H_CONCENTRATED"] = True
    result["entry_rule"] = "PS_AB + H_CONCENTRATED (H1_top >= 2 * H1_second); H_AB diagnostic only"
    result["development_status"] = "Q1_DEVELOPED_ENTRY_RULE_REQUIRES_OOS_VALIDATION"
    return result
