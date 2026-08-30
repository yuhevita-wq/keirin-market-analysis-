from __future__ import annotations

import itertools
import json
from collections import defaultdict
from pathlib import Path

from fake_favorite_formation_sim_2023 import (
    load_audited_gate,
    load_trio_odds,
    load_trifecta_odds,
    load_results,
    compute_delta,
    entrants_from_trio,
    norm_combo,
)

ROOT = Path(__file__).resolve().parents[2]
AUDITS = ROOT / "data" / "audits"
OUT = AUDITS / "axis_drop_rule_search_2023.json"
STAKE = 100

RULES = ["positive_sum", "positive_count_then_sum", "all_delta_sum", "max_delta_then_sum"]


def pair_stats(pair, outsiders, delta):
    vals = [delta.get(norm_combo(pair[0], pair[1], x), float("-inf")) for x in outsiders]
    finite = [v for v in vals if v != float("-inf")]
    pos = [v for v in finite if v > 0]
    return {
        "positive_sum": sum(pos),
        "positive_count": len(pos),
        "all_delta_sum": sum(finite),
        "max_delta": max(finite) if finite else float("-inf"),
    }


def pair_key(rule, st, pair):
    tie = tuple(-x for x in pair)
    if rule == "positive_sum":
        return (st["positive_sum"], st["positive_count"], st["all_delta_sum"], tie)
    if rule == "positive_count_then_sum":
        return (st["positive_count"], st["positive_sum"], st["all_delta_sum"], tie)
    if rule == "all_delta_sum":
        return (st["all_delta_sum"], st["positive_sum"], st["positive_count"], tie)
    if rule == "max_delta_then_sum":
        return (st["max_delta"], st["positive_sum"], st["positive_count"], tie)
    raise ValueError(rule)


def summarize(acc):
    stake = acc["tickets"] * STAKE
    payout = acc["payout"]
    return {
        "bet_races": acc["bet_races"],
        "tickets": acc["tickets"],
        "hit_races": acc["hit_races"],
        "race_hit_rate_pct": 100 * acc["hit_races"] / acc["bet_races"] if acc["bet_races"] else None,
        "stake_yen": stake,
        "payout_yen": payout,
        "profit_yen": payout - stake,
        "roi_pct": 100 * payout / stake if stake else None,
        "axis_pair_survived_top3": acc["axis_survived"],
        "axis_pair_survival_pct": 100 * acc["axis_survived"] / acc["bet_races"] if acc["bet_races"] else None,
    }


def main():
    gate = load_audited_gate()
    trio = load_trio_odds()
    trifecta = load_trifecta_odds()
    results = load_results()

    flow = {r: defaultdict(int) for r in RULES}
    box = {r: defaultdict(int) for r in RULES}
    dropped = {r: defaultdict(int) for r in RULES}
    details = []

    for rid in sorted(gate):
        favorite = norm_combo(*gate[rid]["favorite"])
        entrants = entrants_from_trio(trio[rid])
        outsiders = sorted(set(entrants) - set(favorite))
        delta = compute_delta(trio[rid], trifecta[rid])
        result = norm_combo(*results[rid])
        result_set = set(result)

        pair_rows = []
        for pair in itertools.combinations(favorite, 2):
            pair = norm_combo(*pair)
            pair_rows.append((pair, pair_stats(pair, outsiders, delta)))

        for rule in RULES:
            pair, st = max(pair_rows, key=lambda x: pair_key(rule, x[1], x[0]))
            dropped_car = next(x for x in favorite if x not in pair)
            dropped[rule][str(dropped_car)] += 1

            # 2-head flow: keep the selected two favorite cars, exclude the dropped favorite car,
            # and flow only to actual outsiders. The original favorite ticket is therefore never bought.
            flow_tickets = [norm_combo(pair[0], pair[1], x) for x in outsiders]
            flow[rule]["bet_races"] += 1
            flow[rule]["tickets"] += len(flow_tickets)
            flow[rule]["axis_survived"] += int(set(pair).issubset(result_set))
            hit = result in flow_tickets
            flow[rule]["hit_races"] += int(hit)
            if hit:
                flow[rule]["payout"] += round(STAKE * trio[rid][result])

            # 4-car BOX: selected pair + top two outsiders by pair-specific delta.
            # Requires at least two outsiders; original favorite cannot be present because dropped_car is excluded.
            ranked_out = sorted(
                outsiders,
                key=lambda x: (delta.get(norm_combo(pair[0], pair[1], x), float("-inf")), -x),
                reverse=True,
            )
            if len(ranked_out) >= 2:
                four = sorted([pair[0], pair[1], ranked_out[0], ranked_out[1]])
                box_tickets = [norm_combo(*c) for c in itertools.combinations(four, 3)]
                box[rule]["bet_races"] += 1
                box[rule]["tickets"] += 4
                box[rule]["axis_survived"] += int(set(pair).issubset(result_set))
                bhit = result in box_tickets
                box[rule]["hit_races"] += int(bhit)
                if bhit:
                    box[rule]["payout"] += round(STAKE * trio[rid][result])

            details.append({
                "race_id": rid,
                "rule": rule,
                "favorite": favorite,
                "selected_pair": pair,
                "dropped_favorite_car": dropped_car,
                "pair_stats": st,
                "result": result,
            })

    out = {
        "status": "AXIS_DROP_RULE_SEARCH_2023",
        "year": 2023,
        "years_read": [2023],
        "evaluation_year_2024_used": False,
        "evaluation_year_2025_used": False,
        "evaluation_year_2026_used": False,
        "population": "Audited 208 fake-favorite races only.",
        "principle": "Choose 2 survivors only from the exact trio favorite's 3 cars. The third favorite car is dropped. Original favorite ticket is never bought.",
        "pair_scoring_rules": {
            "positive_sum": "Sum of positive delta for pair+outsider across all actual outsiders; ties by positive count, total delta.",
            "positive_count_then_sum": "Count of positive pair+outsider deltas first; ties by positive-delta sum, then total delta.",
            "all_delta_sum": "Sum of all pair+outsider deltas; ties by positive sum, positive count.",
            "max_delta_then_sum": "Largest pair+outsider delta first; ties by positive sum, positive count.",
        },
        "two_head_outsider_flow": {r: summarize(flow[r]) for r in RULES},
        "four_car_box": {r: summarize(box[r]) for r in RULES},
        "dropped_favorite_car_number_distribution": {r: dict(dropped[r]) for r in RULES},
        "selection_note": "Freeze one scoring rule after reviewing only this 2023 output; apply unchanged to 2024 later.",
    }
    AUDITS.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
