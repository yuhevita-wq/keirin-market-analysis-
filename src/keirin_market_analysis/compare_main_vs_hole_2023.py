from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from pathlib import Path

DETAIL = Path("data/audits/market_scenario_portfolio_2023_decisions.csv")
OUT = Path("data/audits/market_scenario_main_vs_hole_2023.json")

TICKET_RE = re.compile(r"^(main|hole):(\d+)-(\d+)-(\d+)@([0-9.]+)x(\d+)u$")


def median(xs):
    if not xs:
        return None
    ys = sorted(xs)
    n = len(ys)
    return ys[n//2] if n % 2 else (ys[n//2-1] + ys[n//2]) / 2


def main():
    stats = {
        "main": defaultdict(float),
        "hole": defaultdict(float),
    }
    stats["main"]["tickets"] = 0
    stats["hole"]["tickets"] = 0
    stats["main"]["wins"] = 0
    stats["hole"]["wins"] = 0
    stats["main"]["races_present"] = 0
    stats["hole"]["races_present"] = 0

    odds_lists = {"main": [], "hole": []}
    winning_odds = {"main": [], "hole": []}
    winning_profit = {"main": [], "hole": []}
    race_stake_by_kind = {"main": [], "hole": []}

    with DETAIL.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("purchased") != "1":
                continue
            winner = row.get("winner", "")
            winner_kind = row.get("winner_kind", "")
            kind_stake = {"main": 0.0, "hole": 0.0}
            kinds_present = set()
            for part in (row.get("tickets") or "").split(" | "):
                m = TICKET_RE.match(part.strip())
                if not m:
                    continue
                kind = m.group(1)
                cars = "-".join(m.group(i) for i in (2,3,4))
                odds = float(m.group(5))
                units = int(m.group(6))
                stake = units * 100.0
                stats[kind]["tickets"] += 1
                stats[kind]["stake_yen"] += stake
                kind_stake[kind] += stake
                kinds_present.add(kind)
                odds_lists[kind].append(odds)
                if cars == winner:
                    payout = stake * odds
                    stats[kind]["wins"] += 1
                    stats[kind]["payout_yen"] += payout
                    stats[kind]["profit_yen"] += payout - stake
                    winning_odds[kind].append(odds)
                    winning_profit[kind].append(payout - stake)
            for kind in kinds_present:
                stats[kind]["races_present"] += 1
                race_stake_by_kind[kind].append(kind_stake[kind])

    out = {
        "status": "MARKET_SCENARIO_MAIN_VS_HOLE_2023",
        "year": 2023,
        "source": str(DETAIL),
        "note": "Component ROI treats main and hole stakes as separate sub-portfolios using the actual 100-yen-unit dutching allocations from the joint portfolio. A race has only one winning trio set, so payout belongs to exactly one component.",
        "components": {},
    }
    for kind in ("main", "hole"):
        s = stats[kind]
        stake = s["stake_yen"]
        payout = s["payout_yen"]
        tickets = int(s["tickets"])
        wins = int(s["wins"])
        out["components"][kind] = {
            "tickets": tickets,
            "wins": wins,
            "ticket_hit_rate_pct": wins / tickets * 100 if tickets else 0,
            "races_present": int(s["races_present"]),
            "stake_yen": round(stake),
            "payout_yen": round(payout),
            "profit_yen": round(payout - stake),
            "roi_pct": payout / stake * 100 if stake else 0,
            "median_odds": median(odds_lists[kind]),
            "median_winning_odds": median(winning_odds[kind]),
            "mean_winning_odds": sum(winning_odds[kind])/len(winning_odds[kind]) if winning_odds[kind] else None,
            "median_race_stake_yen": median(race_stake_by_kind[kind]),
            "median_profit_on_component_hit_yen": median(winning_profit[kind]),
        }

    total_stake = sum(out["components"][k]["stake_yen"] for k in ("main","hole"))
    total_payout = sum(out["components"][k]["payout_yen"] for k in ("main","hole"))
    out["check"] = {
        "combined_stake_yen": total_stake,
        "combined_payout_yen": total_payout,
        "combined_roi_pct": total_payout / total_stake * 100 if total_stake else 0,
    }
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
