from __future__ import annotations

"""v8.16-F17: adopt FINAL branch; keep Selection quarantined.

FINAL rule (development):
- PS_AB must pass.
- H_CONCENTRATED: H1_top >= 2 * H1_second.
- Keep the unchanged v8.8-F09 Market-Cliff rectangular formation.
- H_AB is diagnostic only.

All other race-type branches are inherited unchanged from v8.15-F16.
"""

from v8_11_f12_race_type_adaptive import classify_race_type
from v8_15_f16_special_split_rebuild import build_v8_15_f16
from v8_8_f09_market_cliff import build_v8_8_f09

SCHEME_VERSION = "v8.16-F17"
STATUS = "DEVELOPMENT_FINAL_ADOPTED"


def build_v8_16_f17(trio_odds, trifecta_odds, predicted_line_formation, race_type: str):
    group = classify_race_type(race_type)
    if group != "FINAL":
        out = build_v8_15_f16(trio_odds, trifecta_odds, predicted_line_formation, race_type)
        return {**out, "scheme_version": SCHEME_VERSION}

    base = build_v8_8_f09(trio_odds, trifecta_odds, predicted_line_formation)
    if not base.get("buy"):
        return {
            **base,
            "scheme_version": SCHEME_VERSION,
            "race_type": race_type,
            "race_type_group": "FINAL",
            "development_status": STATUS,
        }

    gate = dict(base.get("entry_gate") or {})
    h_concentrated = not bool(gate.get("H_RATIO"))
    if not h_concentrated:
        return {
            "scheme_version": SCHEME_VERSION,
            "buy": False,
            "reason": "FINAL_H_NOT_CONCENTRATED",
            "race_type": race_type,
            "race_type_group": "FINAL",
            "entry_gate": gate,
            "diagnostic_formation": base.get("formation"),
            "branch_policy": "PS_AB_PLUS_H_CONCENTRATED",
            "development_status": STATUS,
        }

    return {
        **base,
        "scheme_version": SCHEME_VERSION,
        "race_type": race_type,
        "race_type_group": "FINAL",
        "branch_policy": "PS_AB_PLUS_H_CONCENTRATED_KEEP_F09_FORMATION",
        "final_h_concentrated": True,
        "final_h_ab_diagnostic": bool(gate.get("H_AB")),
        "development_status": STATUS,
    }
