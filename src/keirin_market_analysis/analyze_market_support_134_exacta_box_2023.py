from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "2023" / "s_class_yosen"
AUDITS = ROOT / "data" / "audits"
GATE_JSON = AUDITS / "fake_favorite_true_middle_2023_2024.json"
OUT_JSON = AUDITS / "market_support_134_exacta_box_2023.json"
STAKE_PER_TICKET = 100


def read_csv(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def parse_trio_combo(s: str):
    s = s.strip().replace("=", "-").replace(",", "-")
    return tuple(sorted(int(x) for x in s.split("-") if x.strip()))


def load_gate():
    payload = json.loads(GATE_JSON.read_text(encoding="utf-8"))
    rows = payload["years"]["2023"]["races"]
    gate = {str(r["race_id"]): r for r in rows}
    if len(gate) != 208:
        raise RuntimeError(f"expected 208 gate races, got {len(gate)}")
    return gate


def load_trio_odds():
    rows = read_csv(DATA / "trio_final_odds.csv")
    out = defaultdict(dict)
    for r in rows:
        rid = r["race_id"]
        combo = parse_trio_combo(r.get("combo") or r.get("combination") or r.get("selection") or "")
        odds = float(r.get("odds") or r.get("trio_odds") or r.get("final_odds"))
        out[rid][combo] = odds
    return out


def rider_ranks(trio_odds):
    support = defaultdict(float)
    for combo, odds in trio_odds.items():
        if odds <= 0:
            continue
        w = 1.0 / odds
        for car in combo:
            support[car] += w
    return [car for car, _ in sorted(support.items(), key=lambda kv: (-kv[1], kv[0]))]


def parse_exacta_combo(s: str):
    s = s.strip().replace("→", "-").replace("=", "-").replace(",", "-")
    vals = [int(x) for x in s.split("-") if x.strip()]
    return tuple(vals[:2]) if len(vals) >= 2 else None


def load_exacta_payouts():
    rows = read_csv(DATA / "payouts.csv")
    out = defaultdict(list)
    for r in rows:
        if r.get("ticket_type") != "2車単" or r.get("status") != "paid":
            continue
        combo = parse_exacta_combo(r.get("combination", ""))
        if not combo:
            continue
        out[r["race_id"]].append((combo, int(r["payout_yen"])))
    return out


def main():
    gate = load_gate()
    trio = load_trio_odds()
    payouts = load_exacta_payouts()

    total = {"bet_races": 0, "tickets": 0, "hit_races": 0, "stake_yen": 0, "payout_yen": 0}
    hit_payouts = []
    losing_streak = 0
    max_losing_streak = 0
    examples = []

    for rid in sorted(gate):
        ranks = rider_ranks(trio[rid])
        if len(ranks) < 4:
            raise RuntimeError(f"not enough ranked riders for {rid}")
        a, b, c = ranks[0], ranks[2], ranks[3]
        tickets = {(a,b),(b,a),(a,c),(c,a),(b,c),(c,b)}
        race_payout = sum(p for combo,p in payouts.get(rid, []) if combo in tickets)
        hit = race_payout > 0

        total["bet_races"] += 1
        total["tickets"] += 6
        total["stake_yen"] += 600
        total["payout_yen"] += race_payout
        total["hit_races"] += int(hit)
        if hit:
            hit_payouts.append(race_payout)
            losing_streak = 0
        else:
            losing_streak += 1
            max_losing_streak = max(max_losing_streak, losing_streak)

        if len(examples) < 10:
            examples.append({
                "race_id": rid,
                "market_rank_1": a,
                "market_rank_3": b,
                "market_rank_4": c,
                "tickets": [list(x) for x in sorted(tickets)],
                "winning_exacta_rows": [{"combo": list(combo), "payout_yen": p} for combo,p in payouts.get(rid, [])],
                "hit": hit,
                "race_payout_yen": race_payout,
            })

    total["avg_tickets_per_race"] = total["tickets"] / total["bet_races"]
    total["race_hit_rate_pct"] = 100 * total["hit_races"] / total["bet_races"]
    total["profit_yen"] = total["payout_yen"] - total["stake_yen"]
    total["roi_pct"] = 100 * total["payout_yen"] / total["stake_yen"]
    total["max_losing_streak"] = max_losing_streak
    if hit_payouts:
        xs = sorted(hit_payouts)
        n = len(xs)
        total["mean_hit_payout_yen"] = sum(xs) / n
        total["median_hit_payout_yen"] = xs[n//2] if n % 2 else (xs[n//2-1]+xs[n//2])/2
        total["min_hit_payout_yen"] = xs[0]
        total["max_hit_payout_yen"] = xs[-1]

    output = {
        "status": "MARKET_SUPPORT_134_EXACTA_BOX_2023",
        "year": 2023,
        "years_read": [2023],
        "population": "Audited 208 fake-favorite races only.",
        "market_support_definition": "For each rider, sum 1/trio_final_odds across every valid trio containing that rider; rank descending, lower car number only breaks exact ties.",
        "strategy": "Use market support ranks 1,3,4 as a 3-car exacta BOX: 1->3,3->1,1->4,4->1,3->4,4->3; exactly 6 tickets/race.",
        "payout_method": "Actual published 2車単 payout_yen from payouts.csv; 100 yen per ticket. Multiple paid exacta rows in dead-heat cases are all credited if selected.",
        "result": total,
        "examples_first_10": examples,
        "note": "2023 development simulation only. No 2024/2025/2026 data read."
    }
    AUDITS.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

if __name__ == "__main__":
    main()
