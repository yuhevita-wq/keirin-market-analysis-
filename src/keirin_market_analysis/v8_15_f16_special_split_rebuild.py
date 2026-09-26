from __future__ import annotations

"""v8.15-F16: split the former SPECIAL bucket into three market branches.

Development design on 2024Q1. Decision inputs remain pre-race market data only.

Branches:
- SELECTION (S-class selection): quarantine / no bet. PS_AB, H-state and semantic
  formation-price viability all failed to identify a usable Q1 structure.
- SPECIAL (ordinary S-class special): PS_AB + H_CONCENTRATED, then keep the clean
  F09 market-cliff formation unchanged. H_AB remains diagnostic. No H1 anchor force.
- INITIAL_SPECIAL (S-class initial-special / first-day-special): PS_AB plus structural
  state consistency H_AB == H_CONCENTRATED, then keep F09 formation unchanged.

H_CONCENTRATED is the pre-existing definition H1_top >= 2 * H1_second.
No individual ticket pruning, ticket odds cutoff, fixed place count, or nesting rule.
"""

from v8_11_f12_race_type_adaptive import classify_race_type
from v8_14_f15_general_anchor_preservation import build_v8_14_f15
from v8_8_f09_market_cliff import build_v8_8_f09

SCHEME_VERSION = "v8.15-F16"
STATUS = "DEVELOPMENT_SPECIAL_SPLIT_REBUILD"


def special_subtype(race_type: str) -> str:
    text = (race_type or "").strip()
    if "初特選" in text or "初日特選" in text:
        return "INITIAL_SPECIAL"
    if "選抜" in text:
        return "SELECTION"
    if "特選" in text:
        return "SPECIAL"
    return "OTHER_SPECIAL"


def build_v8_15_f16(trio_odds, trifecta_odds, predicted_line_formation, race_type: str):
    group = classify_race_type(race_type)

    # Preserve all non-SPECIAL v8.14 branches exactly.
    if group != "SPECIAL":
        out = build_v8_14_f15(
            trio_odds,
            trifecta_odds,
            predicted_line_formation,
            race_type,
        )
        return {**out, "scheme_version": SCHEME_VERSION}

    subtype = special_subtype(race_type)
    base = build_v8_8_f09(trio_odds, trifecta_odds, predicted_line_formation)
    if not base.get("buy"):
        return {
            **base,
            "scheme_version": SCHEME_VERSION,
            "race_type": race_type,
            "race_type_group": "SPECIAL",
            "special_subtype": subtype,
            "development_status": STATUS,
        }

    entry_gate = dict(base.get("entry_gate") or {})
    h_concentrated = not bool(entry_gate.get("H_RATIO"))
    h_ab = bool(entry_gate.get("H_AB"))

    if subtype == "SELECTION":
        return {
            "scheme_version": SCHEME_VERSION,
            "buy": False,
            "reason": "SELECTION_BRANCH_QUARANTINED_NO_USABLE_MARKET_STRUCTURE",
            "race_type": race_type,
            "race_type_group": "SPECIAL",
            "special_subtype": subtype,
            "entry_gate": entry_gate,
            "diagnostic_formation": base.get("formation"),
            "diagnostic_ticket_count": base.get("ticket_count"),
            "branch_policy": "NO_BET_PENDING_SELECTION_REBUILD",
            "development_status": STATUS,
        }

    if subtype == "SPECIAL":
        if not h_concentrated:
            return {
                "scheme_version": SCHEME_VERSION,
                "buy": False,
                "reason": "SPECIAL_H_NOT_CONCENTRATED",
                "race_type": race_type,
                "race_type_group": "SPECIAL",
                "special_subtype": subtype,
                "entry_gate": entry_gate,
                "diagnostic_formation": base.get("formation"),
                "branch_policy": "PS_AB_PLUS_H_CONCENTRATED",
                "development_status": STATUS,
            }
        return {
            **base,
            "scheme_version": SCHEME_VERSION,
            "race_type": race_type,
            "race_type_group": "SPECIAL",
            "special_subtype": subtype,
            "branch_policy": "PS_AB_PLUS_H_CONCENTRATED_KEEP_F09_FORMATION",
            "special_h_concentrated": True,
            "special_h_ab_diagnostic": h_ab,
            "individual_ticket_pruning": False,
            "development_status": STATUS,
        }

    if subtype == "INITIAL_SPECIAL":
        if h_ab != h_concentrated:
            return {
                "scheme_version": SCHEME_VERSION,
                "buy": False,
                "reason": "INITIAL_SPECIAL_H_STATE_MISMATCH",
                "race_type": race_type,
                "race_type_group": "SPECIAL",
                "special_subtype": subtype,
                "entry_gate": entry_gate,
                "diagnostic_formation": base.get("formation"),
                "branch_policy": "PS_AB_PLUS_H_STATE_CONSISTENCY",
                "h_concentrated": h_concentrated,
                "h_ab": h_ab,
                "development_status": STATUS,
            }
        return {
            **base,
            "scheme_version": SCHEME_VERSION,
            "race_type": race_type,
            "race_type_group": "SPECIAL",
            "special_subtype": subtype,
            "branch_policy": "PS_AB_PLUS_H_STATE_CONSISTENCY_KEEP_F09_FORMATION",
            "initial_h_concentrated": h_concentrated,
            "initial_h_ab": h_ab,
            "initial_h_state_consistent": True,
            "individual_ticket_pruning": False,
            "development_status": STATUS,
        }

    return {
        "scheme_version": SCHEME_VERSION,
        "buy": False,
        "reason": "UNMAPPED_SPECIAL_SUBTYPE",
        "race_type": race_type,
        "race_type_group": "SPECIAL",
        "special_subtype": subtype,
        "development_status": STATUS,
    }
