from __future__ import annotations

"""v8.12-F13: preserve strong branches, quarantine weak branches for rebuild.

This is a development architecture, not an OOS-validated final scheme.

Policy learned from the fixed-rule v8.11 Q1 development run:
- QUALIFYING and SEMIFINAL are close to usable and must only receive micro-adjustments.
- GENERAL, SPECIAL and FINAL must not inherit the v8.11 rules unchanged.
- Weak branches are explicitly quarantined instead of silently being tuned by payout/result.

The quarantine is deliberate: it prevents the failed v8.11 GENERAL/SPECIAL/FINAL
rules from continuing to generate bets while their new race-type-specific market
structures are researched.
"""

from v8_11_f12_race_type_adaptive import build_v8_11_f12, classify_race_type
from v8_8_f09_market_cliff import build_v8_8_f09

SCHEME_VERSION = "v8.12-F13"
STATUS = "DEVELOPMENT_BRANCH_REBUILD"

MICRO_TUNE_GROUPS = {"QUALIFYING", "SEMIFINAL"}
MAJOR_REBUILD_GROUPS = {"GENERAL", "SPECIAL", "FINAL"}

REBUILD_AXIS = {
    "GENERAL": (
        "retain race-type identity but redesign around formation-level price viability; "
        "high hit rate alone is insufficient when the clean formation is too expensive"
    ),
    "SPECIAL": (
        "split the former broad SPECIAL bucket before defining a new gate; "
        "selection/special/initial-special must not share one PS_AB-only rule"
    ),
    "FINAL": (
        "standalone final-race market model; do not reuse the failed H_BALANCED hypothesis "
        "and do not simply flip it to H_CONCENTRATED without separate evidence"
    ),
}


def build_v8_12_f13(trio_odds, trifecta_odds, predicted_line_formation, race_type: str):
    group = classify_race_type(race_type)

    # Preserve the two near-profitable branches exactly. Any later changes here
    # must be small, explicit and separately measured.
    if group in MICRO_TUNE_GROUPS:
        out = build_v8_11_f12(
            trio_odds,
            trifecta_odds,
            predicted_line_formation,
            race_type,
        )
        return {
            **out,
            "scheme_version": SCHEME_VERSION,
            "branch_policy": "MICRO_TUNE_ONLY",
            "branch_source": "v8.11-F12 unchanged at F13 architecture stage",
            "development_status": STATUS,
        }

    # The weak v8.11 branches are quarantined. We still compute the pre-price
    # F09 structure for diagnostics, but no bet is permitted until a dedicated
    # branch-specific rule is defined. This is not an individual-ticket cut.
    if group in MAJOR_REBUILD_GROUPS:
        diag = build_v8_8_f09(
            trio_odds,
            trifecta_odds,
            predicted_line_formation,
        )
        return {
            "scheme_version": SCHEME_VERSION,
            "buy": False,
            "reason": f"MAJOR_REBUILD_PENDING_{group}",
            "race_type": race_type,
            "race_type_group": group,
            "branch_policy": "MAJOR_REBUILD_REQUIRED",
            "rebuild_axis": REBUILD_AXIS[group],
            "diagnostic_ps_ab_pass": bool(diag.get("buy")),
            "diagnostic_entry_gate": diag.get("entry_gate"),
            "diagnostic_formation": diag.get("formation"),
            "diagnostic_ticket_count": diag.get("ticket_count"),
            "individual_ticket_pruning": False,
            "development_status": STATUS,
        }

    return {
        "scheme_version": SCHEME_VERSION,
        "buy": False,
        "reason": "UNMAPPED_RACE_TYPE",
        "race_type": race_type,
        "race_type_group": group,
        "development_status": STATUS,
    }
