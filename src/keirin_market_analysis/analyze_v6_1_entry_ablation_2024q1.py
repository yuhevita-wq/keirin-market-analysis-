from __future__ import annotations

import json
import statistics
from collections import defaultdict

from simulate_v8_1_f02_2024q1 import load, pi, pl, STAKE
from v7_0_f01_market_hierarchy import (
    implied_probabilities,
    positional_support,
    trio_implied_probabilities,
)
from v8_4_f05_incremental_growth import choose_incremental_growth

DATASET = "2024Q1"


def diagnose_gate(trio_odds, trifecta_odds, predicted_line_formation):
    p3 = trio_implied_probabilities(trio_odds)
    q = implied_probabilities(trifecta_odds)
    cars = sorted({car for combo in trio_odds for car in combo})
    lines = pl(predicted_line_formation)
    if lines is None or set(car for line in lines for car in line) != set(cars):
        return None

    s = {car: sum(prob for combo, prob in p3.items() if car in combo) / 3.0 for car in cars}
    ls = [sum(s[car] for car in line) for line in lines]
    line_order = sorted(range(len(lines)), key=lambda idx: (-ls[idx], idx))
    if len(line_order) < 2:
        return None
    ai, bi = line_order[:2]

    pair_rows = []
    for li, line in enumerate(lines):
        for pos in range(len(line) - 1):
            pair = (line[pos], line[pos + 1])
            ps = sum(prob for combo, prob in p3.items() if pair[0] in combo and pair[1] in combo)
            pair_rows.append((ps, li, pos, pair))
    pair_rows.sort(key=lambda row: (-row[0], row[1], row[2]))
    c_ps = len(pair_rows) >= 2 and {pair_rows[0][1], pair_rows[1][1]} == {ai, bi}

    h1, _, _ = positional_support(q, cars)
    ranked_head = sorted(cars, key=lambda car: (-h1[car], car))
    top2 = ranked_head[:2]
    line_of = {car: li for li, line in enumerate(lines) for car in line}
    pos_of = {car: pos for line in lines for pos, car in enumerate(line)}

    c_h_ab = len(top2) == 2 and {line_of[top2[0]], line_of[top2[1]]} == {ai, bi}
    c_h_pos = len(top2) == 2 and all(pos_of[car] <= 1 for car in top2)
    c_h_ratio = len(top2) == 2 and h1[top2[0]] < 2.0 * h1[top2[1]]

    gate = {
        "entry_pass": True,
        "reason": "ABLATION_DIAGNOSTIC",
        "A_line": lines[ai],
        "B_line": lines[bi],
        "head_rank": tuple(ranked_head),
    }
    return {
        "gate": gate,
        "PS_AB": bool(c_ps),
        "H_AB": bool(c_h_ab),
        "H_POS": bool(c_h_pos),
        "H_RATIO": bool(c_h_ratio),
    }


VARIANTS = {
    "NONE_ALL": (),
    "ONLY_PS": ("PS_AB",),
    "ONLY_H_AB": ("H_AB",),
    "ONLY_H_POS": ("H_POS",),
    "ONLY_H_RATIO": ("H_RATIO",),
    "PS_H_AB": ("PS_AB", "H_AB"),
    "PS_H_AB_H_POS": ("PS_AB", "H_AB", "H_POS"),
    "H_AB_H_POS_H_RATIO": ("H_AB", "H_POS", "H_RATIO"),
    "DROP_PS": ("H_AB", "H_POS", "H_RATIO"),
    "DROP_H_AB": ("PS_AB", "H_POS", "H_RATIO"),
    "DROP_H_POS": ("PS_AB", "H_AB", "H_RATIO"),
    "DROP_H_RATIO": ("PS_AB", "H_AB", "H_POS"),
    "FULL_V6_1": ("PS_AB", "H_AB", "H_POS", "H_RATIO"),
}


def summarize(rows):
    races = len(rows)
    hits = sum(r["hit"] for r in rows)
    tickets = sum(r["ticket_count"] for r in rows)
    stake = tickets * STAKE
    payout = sum(r["payout_yen"] for r in rows)
    hit_payouts = [r["payout_yen"] for r in rows if r["hit"]]
    profitable_hits = sum(r["hit"] and r["payout_yen"] >= r["ticket_count"] * STAKE for r in rows)
    losing_hits = hits - profitable_hits
    return {
        "races": races,
        "hits": hits,
        "hit_rate_pct": 100 * hits / races if races else None,
        "tickets": tickets,
        "avg_tickets": tickets / races if races else None,
        "stake_yen": stake,
        "payout_yen": payout,
        "profit_yen": payout - stake,
        "roi_pct": 100 * payout / stake if stake else None,
        "profitable_hit_races": profitable_hits,
        "losing_hit_races": losing_hits,
        "profitable_hit_share_pct": 100 * profitable_hits / hits if hits else None,
        "avg_hit_payout_yen": sum(hit_payouts) / len(hit_payouts) if hit_payouts else None,
        "median_hit_payout_yen": statistics.median(hit_payouts) if hit_payouts else None,
        "hits_ge_5000": sum(v >= 5000 for v in hit_payouts),
        "hits_ge_10000": sum(v >= 10000 for v in hit_payouts),
        "payout_from_hits_ge_5000_yen": sum(v for v in hit_payouts if v >= 5000),
        "payout_from_hits_ge_10000_yen": sum(v for v in hit_payouts if v >= 10000),
    }


def main():
    races, trio, tf, pay = load()
    eligible = []
    condition_counts = defaultdict(int)

    for rid, r in sorted(races.items(), key=lambda kv: (kv[1].get("race_date", ""), kv[0])):
        if r.get("meeting_grade") != "F1" or not (r.get("race_type") or "").startswith("Ｓ級"):
            continue
        if pi(r.get("entry_count")) != 7 or len(trio.get(rid, {})) != 35 or len(tf.get(rid, {})) != 210:
            continue
        cars = sorted({v for c in trio[rid] for v in c})
        lines = pl(r.get("predicted_line_formation"))
        if len(cars) != 7 or lines is None or set(v for line in lines for v in line) != set(cars):
            continue
        if rid not in pay or not pay[rid]:
            continue

        diag = diagnose_gate(trio[rid], tf[rid], r.get("predicted_line_formation") or "")
        if diag is None:
            continue
        for c in ("PS_AB", "H_AB", "H_POS", "H_RATIO"):
            condition_counts[c] += int(diag[c])

        chosen, states, moves, selected_index = choose_incremental_growth(trio[rid], tf[rid], diag["gate"])
        wins = [t for t in chosen.tickets if t in pay[rid]]
        payout = sum(pay[rid][t] for t in wins)
        eligible.append({
            "race_id": rid,
            "conditions": diag,
            "ticket_count": chosen.ticket_count,
            "hit": int(bool(wins)),
            "payout_yen": payout,
        })

    summaries = {}
    selected_ids = {}
    for name, reqs in VARIANTS.items():
        rows = [r for r in eligible if all(r["conditions"][c] for c in reqs)]
        summaries[name] = summarize(rows)
        selected_ids[name] = {r["race_id"] for r in rows}

    full = selected_ids["FULL_V6_1"]
    removal_effect = {}
    for dropped in ("DROP_PS", "DROP_H_AB", "DROP_H_POS", "DROP_H_RATIO"):
        added_ids = selected_ids[dropped] - full
        added_rows = [r for r in eligible if r["race_id"] in added_ids]
        removal_effect[dropped] = {
            "newly_admitted_vs_full": summarize(added_rows),
            "newly_admitted_races": len(added_ids),
        }

    result = {
        "analysis": "v6.1 entry gate ablation",
        "dataset": DATASET,
        "formation_held_fixed": "v8.4-F05 incremental-growth formation for every race; only entry conditions vary",
        "population": len(eligible),
        "condition_pass_counts": dict(condition_counts),
        "condition_pass_rates_pct": {k: 100 * v / len(eligible) for k, v in condition_counts.items()},
        "variants": summaries,
        "drop_one_condition_increment": removal_effect,
        "notes": [
            "LS top-two lines are the A/B definition, not an independent pass/fail filter.",
            "No entry threshold was fitted to outcomes in this analysis.",
            "5000/10000 yen fields are descriptive payout bands, not selection rules.",
        ],
    }
    print("V6_1_ENTRY_ABLATION_Q1_BEGIN")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print("V6_1_ENTRY_ABLATION_Q1_END")


if __name__ == "__main__":
    main()
