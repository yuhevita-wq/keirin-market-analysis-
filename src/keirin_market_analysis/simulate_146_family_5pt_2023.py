from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "2023" / "s_class_yosen"
AUDITS = ROOT / "data" / "audits"
TRIO_ODDS = DATA / "trio_final_odds.csv"
PAYOUTS = DATA / "payouts.csv"
OUT = AUDITS / "market_support_146_family_5pt_2023.json"
STAKE = 100
PATTERNS = [(1,3,6),(1,4,5),(1,4,6),(1,4,7),(1,5,6)]


def read_csv(path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def parse_trio_combo(s):
    s = s.strip().replace("=", "-").replace(",", "-")
    return tuple(sorted(int(x) for x in s.split("-") if x.strip()))


def max_losing_streak(hits):
    best = cur = 0
    for h in hits:
        if h:
            cur = 0
        else:
            cur += 1
            best = max(best, cur)
    return best


def median(vals):
    xs = sorted(vals)
    if not xs:
        return None
    n = len(xs)
    return xs[n//2] if n % 2 else (xs[n//2-1] + xs[n//2]) / 2


def main():
    trio_by_race = defaultdict(list)
    for r in read_csv(TRIO_ODDS):
        trio_by_race[r["race_id"]].append(r)

    paid_trio_by_race = defaultdict(list)
    for r in read_csv(PAYOUTS):
        if r.get("ticket_type") == "3連複" and r.get("status") == "paid":
            paid_trio_by_race[r["race_id"]].append({
                "combo": parse_trio_combo(r["combination"]),
                "payout_yen": int(r["payout_yen"]),
            })

    race_ids = sorted(set(trio_by_race) & set(paid_trio_by_race))

    total_stake = 0
    total_payout = 0
    hit_races = 0
    race_hits = []
    hit_payouts = []
    pattern_stats = {
        "-".join(map(str,p)): {"tickets":0,"hits":0,"stake_yen":0,"payout_yen":0}
        for p in PATTERNS
    }

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

        tickets = []
        for pat in PATTERNS:
            if max(pat) > len(ranked):
                continue
            combo = tuple(sorted(ranked[i-1] for i in pat))
            tickets.append((pat, combo))

        paid_map = {x["combo"]: x["payout_yen"] for x in paid_trio_by_race[rid]}
        race_payout = 0
        race_hit = False
        for pat, combo in tickets:
            key = "-".join(map(str,pat))
            ps = pattern_stats[key]
            ps["tickets"] += 1
            ps["stake_yen"] += STAKE
            total_stake += STAKE
            payout = paid_map.get(combo, 0)
            if payout:
                ps["hits"] += 1
                ps["payout_yen"] += payout
                total_payout += payout
                race_payout += payout
                race_hit = True
        race_hits.append(race_hit)
        if race_hit:
            hit_races += 1
            hit_payouts.append(race_payout)

    for ps in pattern_stats.values():
        ps["profit_yen"] = ps["payout_yen"] - ps["stake_yen"]
        ps["roi_pct"] = 100 * ps["payout_yen"] / ps["stake_yen"] if ps["stake_yen"] else None
        ps["ticket_hit_rate_pct"] = 100 * ps["hits"] / ps["tickets"] if ps["tickets"] else None

    result = {
        "bet_races": len(race_ids),
        "tickets": sum(x["tickets"] for x in pattern_stats.values()),
        "avg_tickets_per_race": sum(x["tickets"] for x in pattern_stats.values()) / len(race_ids),
        "hit_races": hit_races,
        "race_hit_rate_pct": 100 * hit_races / len(race_ids),
        "stake_yen": total_stake,
        "payout_yen": total_payout,
        "profit_yen": total_payout - total_stake,
        "roi_pct": 100 * total_payout / total_stake,
        "max_losing_streak": max_losing_streak(race_hits),
        "mean_hit_race_payout_yen": sum(hit_payouts)/len(hit_payouts) if hit_payouts else None,
        "median_hit_race_payout_yen": median(hit_payouts),
        "max_hit_race_payout_yen": max(hit_payouts) if hit_payouts else None,
    }

    out = {
        "status": "MARKET_SUPPORT_146_FAMILY_5PT_2023",
        "year": 2023,
        "years_read": [2023],
        "population": "All 2023 exact-match S級予選 races with complete 3連複 odds and paid 3連複 result.",
        "support_definition": "Individual rider support = sum of 1/trio_final_odds over every valid trio containing rider; ranked descending, lower car number only exact tie-break.",
        "strategy": "Every race, buy 5 support-rank trios: 1-3-6, 1-4-5, 1-4-6, 1-4-7, 1-5-6. 100 yen each. No race gate.",
        "payout_method": "Actual published 3連複 payout_yen from payouts.csv.",
        "result": result,
        "by_pattern": pattern_stats,
        "warning": "2023 development simulation only. No 2024/2025/2026 data read; no gate or threshold tuning applied."
    }
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
