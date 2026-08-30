from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "2023" / "s_class_yosen"
AUDITS = ROOT / "data" / "audits"
GATE_JSON = AUDITS / "fake_favorite_true_middle_2023_2024.json"
TRIO_ODDS = DATA / "trio_final_odds.csv"
PAYOUTS = DATA / "payouts.csv"
OUT = AUDITS / "market_support_134_wide_box_2023.json"
STAKE = 100


def read_csv(path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def parse_trio_combo(s):
    s = s.strip().replace("=", "-").replace(",", "-")
    return tuple(sorted(int(x) for x in s.split("-") if x.strip()))


def parse_pair(s):
    s = s.strip().replace("=", "-").replace(",", "-")
    xs = [int(x) for x in s.split("-") if x.strip()]
    return tuple(sorted(xs))


def max_losing_streak(hits):
    best = cur = 0
    for h in hits:
        if h:
            cur = 0
        else:
            cur += 1
            best = max(best, cur)
    return best


def main():
    gate = json.loads(GATE_JSON.read_text(encoding="utf-8"))["years"]["2023"]["races"]
    race_ids = [str(r["race_id"]) for r in gate]
    assert len(race_ids) == 208

    trio_rows = read_csv(TRIO_ODDS)
    trio_by_race = defaultdict(list)
    for r in trio_rows:
        trio_by_race[r["race_id"]].append(r)

    wide_by_race = defaultdict(list)
    for r in read_csv(PAYOUTS):
        if r.get("ticket_type") == "ワイド" and r.get("status") == "paid":
            wide_by_race[r["race_id"]].append({
                "combo": parse_pair(r["combination"]),
                "payout_yen": int(r["payout_yen"]),
            })

    total_stake = total_payout = hit_races = 0
    race_hits = []
    race_payouts = []
    ticket_hit_counts = []
    examples = []

    for rid in race_ids:
        rows = trio_by_race[rid]
        support = defaultdict(float)
        for r in rows:
            combo = parse_trio_combo(r["combination"])
            odds = float(r["odds"])
            if odds <= 0:
                continue
            w = 1.0 / odds
            for car in combo:
                support[car] += w
        ranked = sorted(support, key=lambda c: (-support[c], c))
        if len(ranked) < 4:
            raise RuntimeError(f"Not enough ranked riders {rid}")
        c1, c3, c4 = ranked[0], ranked[2], ranked[3]
        tickets = {tuple(sorted(x)) for x in [(c1,c3),(c1,c4),(c3,c4)]}
        total_stake += 3 * STAKE

        paid_rows = wide_by_race[rid]
        selected_paid = [x for x in paid_rows if x["combo"] in tickets]
        payout = sum(x["payout_yen"] for x in selected_paid)
        total_payout += payout
        hits = len(selected_paid)
        ticket_hit_counts.append(hits)
        hit = hits > 0
        race_hits.append(hit)
        if hit:
            hit_races += 1
            race_payouts.append(payout)

        if len(examples) < 10:
            examples.append({
                "race_id": rid,
                "market_rank_1": c1,
                "market_rank_3": c3,
                "market_rank_4": c4,
                "tickets": [list(x) for x in sorted(tickets)],
                "winning_wide_rows": paid_rows,
                "selected_paid_rows": selected_paid,
                "hit_ticket_count": hits,
                "race_payout_yen": payout,
            })

    dist = defaultdict(int)
    for n in ticket_hit_counts:
        dist[str(n)] += 1
    race_payouts_sorted = sorted(race_payouts)
    median = None
    if race_payouts_sorted:
        n=len(race_payouts_sorted)
        median = race_payouts_sorted[n//2] if n%2 else (race_payouts_sorted[n//2-1]+race_payouts_sorted[n//2])/2

    out = {
        "status": "MARKET_SUPPORT_134_WIDE_BOX_2023",
        "year": 2023,
        "years_read": [2023],
        "population": "Audited 208 fake-favorite races only.",
        "market_support_definition": "For each rider, sum 1/trio_final_odds across every valid trio containing that rider; rank descending, lower car number only breaks exact ties.",
        "strategy": "Use market support ranks 1,3,4 as a 3-car wide BOX: pairs 1-3, 1-4, 3-4; exactly 3 tickets/race.",
        "payout_method": "Actual published ワイド payout_yen from payouts.csv; 100 yen per ticket. All selected paid wide rows are credited, so double/triple hits are summed.",
        "result": {
            "bet_races": 208,
            "tickets": 624,
            "avg_tickets_per_race": 3.0,
            "hit_races": hit_races,
            "race_hit_rate_pct": 100*hit_races/208,
            "stake_yen": total_stake,
            "payout_yen": total_payout,
            "profit_yen": total_payout-total_stake,
            "roi_pct": 100*total_payout/total_stake,
            "max_losing_streak": max_losing_streak(race_hits),
            "hit_ticket_count_distribution": dict(sorted(dist.items(), key=lambda kv:int(kv[0]))),
            "mean_hit_race_payout_yen": (sum(race_payouts)/len(race_payouts)) if race_payouts else None,
            "median_hit_race_payout_yen": median,
            "min_hit_race_payout_yen": min(race_payouts) if race_payouts else None,
            "max_hit_race_payout_yen": max(race_payouts) if race_payouts else None,
        },
        "examples_first_10": examples,
        "note": "2023 development simulation only. No 2024/2025/2026 data read."
    }
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out["result"], ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
