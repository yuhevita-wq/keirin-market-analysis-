from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "2023" / "s_class_yosen"
OUT = ROOT / "data" / "audits" / "trio_favorite_miss_replacement_sim_2023.json"
STAKE = 100
FAV_SHARE_MAX = 0.17724290354711825
ENTROPY_MIN = 0.8115383343655119


def read_csv(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def parse_combo(s: str):
    s = s.strip().replace("=", "-").replace(",", "-")
    return tuple(sorted(int(x) for x in s.split("-") if x.strip()))


def max_losing_streak(xs):
    best = cur = 0
    for x in xs:
        if x:
            cur = 0
        else:
            cur += 1
            best = max(best, cur)
    return best


def main():
    trio_by_race = defaultdict(list)
    for r in read_csv(DATA / "trio_final_odds.csv"):
        if r.get("odds_status") != "available":
            continue
        try:
            odds = float(r["odds"])
        except Exception:
            continue
        if odds <= 0:
            continue
        trio_by_race[r["race_id"]].append((parse_combo(r["combination"]), odds))

    payouts = defaultdict(list)
    for r in read_csv(DATA / "payouts.csv"):
        if r.get("ticket_type") == "3連複" and r.get("status") == "paid":
            payouts[r["race_id"]].append((parse_combo(r["combination"]), int(r["payout_yen"])))

    totals = {"bet_races":0,"tickets":0,"hit_races":0,"stake_yen":0,"payout_yen":0}
    race_hits = []
    decisions = []
    hit_by_replacement_rank = defaultdict(lambda: {"tickets":0,"hits":0,"stake_yen":0,"payout_yen":0})

    for rid in sorted(set(trio_by_race) & set(payouts)):
        rows = trio_by_race[rid]
        inv = {c:1.0/o for c,o in rows}
        z = sum(inv.values())
        p = {c:w/z for c,w in inv.items()}
        favorite = min(rows, key=lambda x:(x[1],x[0]))[0]
        fav_share = p[favorite]
        ent = -sum(v*math.log(v) for v in p.values())/math.log(len(p)) if len(p)>1 else 0.0
        if not (fav_share <= FAV_SHARE_MAX and ent >= ENTROPY_MIN):
            continue

        rider_support = defaultdict(float)
        for combo, prob in p.items():
            for car in combo:
                rider_support[car] += prob
        ranked = sorted(rider_support, key=lambda c:(-rider_support[c],c))
        rank_of = {c:i+1 for i,c in enumerate(ranked)}
        fav_ranked = sorted(favorite, key=lambda c:(-rider_support[c],c))
        dropped = fav_ranked[-1]
        retained = tuple(sorted(set(favorite)-{dropped}))
        favorite_set = set(favorite)

        repl_cars = [ranked[i-1] for i in (4,5,6) if i <= len(ranked) and ranked[i-1] not in favorite_set]
        tickets = []
        meta = []
        for rc in repl_cars:
            t = tuple(sorted((*retained, rc)))
            if t not in tickets:
                tickets.append(t)
                meta.append((rank_of[rc], t))

        totals["bet_races"] += 1
        totals["tickets"] += len(tickets)
        totals["stake_yen"] += STAKE * len(tickets)
        winning = payouts[rid]
        payout = sum(pay for combo,pay in winning if combo in tickets)
        hit = payout > 0
        totals["hit_races"] += int(hit)
        totals["payout_yen"] += payout
        race_hits.append(hit)

        for rr,t in meta:
            s = hit_by_replacement_rank[str(rr)]
            s["tickets"] += 1
            s["stake_yen"] += STAKE
            pay = sum(pay for combo,pay in winning if combo == t)
            if pay:
                s["hits"] += 1
                s["payout_yen"] += pay

        decisions.append({
            "race_id":rid,
            "favorite":list(favorite),
            "favorite_share":fav_share,
            "entropy":ent,
            "dropped_weakest_favorite_member":dropped,
            "retained":list(retained),
            "replacement_support_ranks":[x[0] for x in meta],
            "tickets":[list(x) for x in tickets],
            "winning_trios":[{"combo":list(c),"payout_yen":pay} for c,pay in winning],
            "hit":int(hit),
            "race_payout_yen":payout,
        })

    totals["profit_yen"] = totals["payout_yen"] - totals["stake_yen"]
    totals["roi_pct"] = 100*totals["payout_yen"]/totals["stake_yen"] if totals["stake_yen"] else None
    totals["race_hit_rate_pct"] = 100*totals["hit_races"]/totals["bet_races"] if totals["bet_races"] else None
    totals["avg_tickets_per_race"] = totals["tickets"]/totals["bet_races"] if totals["bet_races"] else None
    totals["max_losing_streak"] = max_losing_streak(race_hits)

    by_rank = {}
    for k,s in sorted(hit_by_replacement_rank.items(), key=lambda kv:int(kv[0])):
        s["profit_yen"] = s["payout_yen"] - s["stake_yen"]
        s["roi_pct"] = 100*s["payout_yen"]/s["stake_yen"] if s["stake_yen"] else None
        s["ticket_hit_rate_pct"] = 100*s["hits"]/s["tickets"] if s["tickets"] else None
        by_rank[k] = dict(s)

    out = {
        "status":"TRIO_FAVORITE_MISS_REPLACEMENT_SIM_2023_OUTSIDERS_ONLY",
        "year":2023,
        "years_read":[2023],
        "gate":{
            "favorite_share_max":FAV_SHARE_MAX,
            "entropy_min":ENTROPY_MIN,
            "definition":"Select races where final 3連複 favorite market share <= threshold AND normalized trio-market entropy >= threshold."
        },
        "strategy":"Within the 3連複 favorite trio, drop the member with the weakest individual marginal trio-market support. Retain the other two and buy outsider riders whose individual market-support rank is 4, 5, or 6. Any rider already in the original favorite trio is excluded from replacement. Flat 100 yen per unique trio ticket.",
        "payout_method":"Actual published 3連複 payout_yen from payouts.csv.",
        "result":totals,
        "diagnostic_by_replacement_support_rank":by_rank,
        "decisions":decisions,
        "warning":"This is a 2023 development simulation. Both gate thresholds and the replacement architecture were discovered from 2023 outcomes, so ROI is in-sample and not validation. No 2024/2025/2026 data read."
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
