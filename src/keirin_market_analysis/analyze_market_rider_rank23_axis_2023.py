from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from fake_favorite_formation_sim_2023 import (
    load_audited_gate,
    load_trio_odds,
    load_results,
    entrants_from_trio,
    norm_combo,
)

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "audits" / "market_rider_rank23_axis_2023.json"
STAKE = 100


def rider_market_support(rider: int, trio_odds: dict[tuple[int, int, int], float]) -> float:
    """Marginal trio-market support for a rider: sum of 1/odds over all trio outcomes containing rider.

    The race-level normalization denominator is common to every rider, so it does not affect ranking.
    This is a market-implied marginal support score, not a calibrated true probability.
    """
    return sum((1.0 / odds) for combo, odds in trio_odds.items() if rider in combo and odds > 0)


def max_losing_streak(hits: list[bool]) -> int:
    best = cur = 0
    for hit in hits:
        if hit:
            cur = 0
        else:
            cur += 1
            best = max(best, cur)
    return best


def main():
    gate = load_audited_gate()
    trio = load_trio_odds()
    results = load_results()

    acc = defaultdict(int)
    hit_seq: list[bool] = []
    winning_odds: list[float] = []
    rank_top3 = {1: 0, 2: 0, 3: 0}
    examples = []

    for rid in sorted(gate):
        race_trio = trio[rid]
        entrants = entrants_from_trio(race_trio)
        result = norm_combo(*results[rid])
        result_set = set(result)

        ranked = sorted(
            [(r, rider_market_support(r, race_trio)) for r in entrants],
            key=lambda x: (x[1], -x[0]),
            reverse=True,
        )
        r1, r2, r3 = ranked[0][0], ranked[1][0], ranked[2][0]
        axis = norm_combo(r2, r3)
        tickets = [norm_combo(r2, r3, x) for x in entrants if x not in axis]
        hit = result in tickets

        acc["bet_races"] += 1
        acc["tickets"] += len(tickets)
        acc["hit_races"] += int(hit)
        acc["axis_survived"] += int(set(axis).issubset(result_set))
        hit_seq.append(hit)

        for rank, rider in ((1, r1), (2, r2), (3, r3)):
            rank_top3[rank] += int(rider in result_set)

        if hit:
            odds = race_trio[result]
            payout = round(STAKE * odds)
            acc["payout"] += payout
            winning_odds.append(odds)

        if len(examples) < 10:
            examples.append({
                "race_id": rid,
                "market_rank_1": r1,
                "market_rank_2": r2,
                "market_rank_3": r3,
                "axis": list(axis),
                "tickets": [list(t) for t in tickets],
                "result": list(result),
                "hit": hit,
            })

    stake = acc["tickets"] * STAKE
    payout = acc["payout"]
    winning_odds_sorted = sorted(winning_odds)
    median_odds = None
    if winning_odds_sorted:
        n = len(winning_odds_sorted)
        mid = n // 2
        median_odds = winning_odds_sorted[mid] if n % 2 else (winning_odds_sorted[mid - 1] + winning_odds_sorted[mid]) / 2

    out = {
        "status": "MARKET_RIDER_RANK23_AXIS_2023",
        "year": 2023,
        "years_read": [2023],
        "population": "Audited 208 fake-favorite races only.",
        "market_support_definition": "For each rider, sum 1/trio_final_odds across every valid trio combination containing that rider. Rank riders descending by this marginal trio-market support; lower car number only breaks exact ties.",
        "strategy": "Use market-supported rider rank #2 and #3 as the two-head trio axis. Rank #1 is not removed; it remains one of the literal full-flow targets.",
        "payout_method": "100 yen times final trio odds for apples-to-apples comparison with prior development simulations; actual payout CSV not yet used.",
        "result": {
            "bet_races": acc["bet_races"],
            "tickets": acc["tickets"],
            "hit_races": acc["hit_races"],
            "race_hit_rate_pct": 100 * acc["hit_races"] / acc["bet_races"],
            "axis_pair_survival_pct": 100 * acc["axis_survived"] / acc["bet_races"],
            "stake_yen": stake,
            "payout_yen": payout,
            "profit_yen": payout - stake,
            "roi_pct": 100 * payout / stake,
            "max_losing_streak": max_losing_streak(hit_seq),
            "mean_winning_odds": sum(winning_odds) / len(winning_odds) if winning_odds else None,
            "median_winning_odds": median_odds,
            "market_rank1_top3_pct": 100 * rank_top3[1] / acc["bet_races"],
            "market_rank2_top3_pct": 100 * rank_top3[2] / acc["bet_races"],
            "market_rank3_top3_pct": 100 * rank_top3[3] / acc["bet_races"],
        },
        "examples_first_10": examples,
        "note": "Development simulation only. No 2024/2025/2026 data read.",
    }

    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
