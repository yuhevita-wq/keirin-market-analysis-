from __future__ import annotations

"""v8.11-F12: race-type adaptive market structure + structural price compression.

Development design only. No result or payout is used by this engine.

Order is fixed:
1. PS_AB market structure and clean market-cliff formation are built first.
2. The race type changes what H-market structure is considered acceptable.
3. Only after the clean formation exists, structural price compression is applied.
4. Price compression may remove whole riders from a place-set only.
5. Individual cheap-ticket pruning is prohibited.

Five race-type groups are used to avoid overfitting seven separate micro-schemes:
- QUALIFYING: S-class qualifying / preliminary races
- GENERAL: S-class general races
- SEMIFINAL: S-class semifinals
- SPECIAL: selection / special-selection / initial-special-selection races
- FINAL: S-class finals
"""

from v7_0_f01_market_hierarchy import implied_probabilities
from v8_8_f09_market_cliff import _state, build_v8_8_f09
from v8_10_f11_structural_price_compression import (
    _compression_path,
    _price_state,
    _select_structural_knee,
)

SCHEME_VERSION = "v8.11-F12"
BASE_FORMATION_SCHEME = "v8.8-F09"
BASE_PRICE_SCHEME = "v8.10-F11"
INDIVIDUAL_TICKET_PRUNING = False
STRUCTURAL_PRICE_COMPRESSION = True
RACE_TYPE_ADAPTIVE = True


def classify_race_type(race_type: str) -> str:
    text = (race_type or "").strip()
    # Order matters because 準決勝 contains 決勝.
    if "準決勝" in text:
        return "SEMIFINAL"
    if "決勝" in text:
        return "FINAL"
    if "予選" in text:
        return "QUALIFYING"
    if "一般" in text:
        return "GENERAL"
    if "選抜" in text or "特選" in text:
        return "SPECIAL"
    return "OTHER"


def race_type_market_gate(group: str, entry_gate: dict) -> tuple[bool, str]:
    """Interpret the same PS/H market differently by race type.

    H_RATIO=True means H1_top < 2 * H1_second (balanced first-place market).
    H_RATIO=False means H1_top >= 2 * H1_second (concentrated first-place market).
    H_AB=True means the top two H riders are split one each across A/B.

    These rules are conceptual development hypotheses, not Q1-fitted thresholds.
    """
    h_ratio = bool(entry_gate.get("H_RATIO"))
    h_ab = bool(entry_gate.get("H_AB"))

    if group == "QUALIFYING":
        return (not h_ratio), "PS_AB + H_CONCENTRATED"

    if group == "GENERAL":
        return (not h_ratio) and h_ab, "PS_AB + H_CONCENTRATED + H_AB"

    if group == "SEMIFINAL":
        return True, "PS_AB_ONLY; H_STATE_DIAGNOSTIC"

    if group == "SPECIAL":
        return True, "PS_AB_ONLY; H_STATE_DIAGNOSTIC"

    if group == "FINAL":
        return h_ratio, "PS_AB + H_BALANCED"

    return False, "UNMAPPED_RACE_TYPE"


def build_v8_11_f12(
    trio_odds,
    trifecta_odds,
    predicted_line_formation,
    race_type: str,
):
    group = classify_race_type(race_type)
    if group == "OTHER":
        return {
            "scheme_version": SCHEME_VERSION,
            "buy": False,
            "reason": "UNMAPPED_RACE_TYPE",
            "race_type": race_type,
            "race_type_group": group,
        }

    # Phase A: PS_AB + clean formation. No price pruning here.
    base = build_v8_8_f09(trio_odds, trifecta_odds, predicted_line_formation)
    if not base.get("buy"):
        return {
            **base,
            "scheme_version": SCHEME_VERSION,
            "race_type": race_type,
            "race_type_group": group,
            "individual_ticket_pruning": INDIVIDUAL_TICKET_PRUNING,
            "structural_price_compression": STRUCTURAL_PRICE_COMPRESSION,
        }

    entry_gate = dict(base.get("entry_gate") or {})
    type_pass, type_rule = race_type_market_gate(group, entry_gate)
    if not type_pass:
        return {
            "scheme_version": SCHEME_VERSION,
            "buy": False,
            "reason": f"RACE_TYPE_MARKET_GATE_FAIL_{group}",
            "race_type": race_type,
            "race_type_group": group,
            "race_type_rule": type_rule,
            "entry_gate": entry_gate,
            "formation_before_type_gate": base.get("formation"),
            "individual_ticket_pruning": INDIVIDUAL_TICKET_PRUNING,
            "structural_price_compression": STRUCTURAL_PRICE_COMPRESSION,
        }

    # Phase B: the clean formation is now fixed. Only now may prices compress it.
    q = implied_probabilities(trifecta_odds)
    base_state = _state(base["first"], base["second"], base["third"], q)
    if base_state is None:
        return {
            "scheme_version": SCHEME_VERSION,
            "buy": False,
            "reason": "INVALID_BASE_FORMATION",
            "race_type": race_type,
            "race_type_group": group,
        }

    p0 = _price_state(base_state, base_state.q_mass, q, trifecta_odds)
    states, moves = _compression_path(p0, q, trifecta_odds)
    selected_i = _select_structural_knee(states)
    chosen = states[selected_i]

    trace = []
    for i, ps in enumerate(states):
        row = {
            "step": i,
            "formation": ps.display,
            "ticket_count": ps.state.ticket_count,
            "q_mass": ps.state.q_mass,
            "q_retention": ps.q_retention,
            "weighted_gm_odds": ps.weighted_gm_odds,
            "gm_return_multiple": ps.gm_return_multiple,
            "profitable_q_share": ps.profitable_q_share,
            "selected": i == selected_i,
        }
        if i > 0:
            move = moves[i - 1]
            row.update({
                "removed_place": move.place,
                "removed_rider": move.rider,
                "q_loss": move.q_loss,
                "log_return_gain": move.log_return_gain,
                "price_efficiency": move.efficiency,
            })
        trace.append(row)

    n = chosen.state.ticket_count
    profit_mass_1x = sum(
        q[t] for t in chosen.state.tickets if float(trifecta_odds[t]) >= n
    )
    profit_mass_2x = sum(
        q[t] for t in chosen.state.tickets if float(trifecta_odds[t]) >= 2 * n
    )

    return {
        **base,
        "scheme_version": SCHEME_VERSION,
        "base_formation_scheme": BASE_FORMATION_SCHEME,
        "base_price_scheme": BASE_PRICE_SCHEME,
        "race_type": race_type,
        "race_type_group": group,
        "race_type_rule": type_rule,
        "race_type_adaptive": RACE_TYPE_ADAPTIVE,
        "formation_before_price": base["formation"],
        "ticket_count_before_price": base["ticket_count"],
        "formation": chosen.display,
        "first": chosen.state.first,
        "second": chosen.state.second,
        "third": chosen.state.third,
        "tickets": chosen.state.tickets,
        "ticket_count": n,
        "q_mass": chosen.state.q_mass,
        "profit_mass_1x": profit_mass_1x,
        "profit_mass_2x": profit_mass_2x,
        "price_compression_steps": selected_i,
        "price_path_steps_total": len(moves),
        "price_q_retention": chosen.q_retention,
        "price_weighted_gm_odds": chosen.weighted_gm_odds,
        "price_gm_return_multiple": chosen.gm_return_multiple,
        "price_profitable_q_share": chosen.profitable_q_share,
        "price_compression_trace": trace,
        "individual_ticket_pruning": INDIVIDUAL_TICKET_PRUNING,
        "structural_price_compression": STRUCTURAL_PRICE_COMPRESSION,
        "price_phase_rule": "complete clean formation first; then whole-rider removals only; never prune individual cheap tickets",
        "development_status": "RACE_TYPE_ADAPTIVE_DEVELOPMENT_DESIGN_REQUIRES_Q1_SIMULATION_AND_LATER_OOS_VALIDATION",
    }
