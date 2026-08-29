from __future__ import annotations

import csv
import itertools
import json
import math
from collections import defaultdict
from pathlib import Path

YEAR = 2023
DATA = Path(f"data/{YEAR}/s_class_yosen")
OUT = Path("data/audits/market_scenario_portfolio_2023_v1.json")
DETAIL = Path("data/audits/market_scenario_portfolio_2023_v1_decisions.csv")

MIN_EFFECTIVE_ORDERS_HOLE = 3.0
MIN_SUPPORT_ORDERS_10PCT_HOLE = 3


def read_csv(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def norm_inv_odds(rows, separator):
    vals = {}
    total = 0.0
    for r in rows:
        if r.get("odds_status") != "available":
            continue
        try:
            odds = float(r["odds"])
        except Exception:
            continue
        if odds <= 1.0:
            continue
        comb = r.get("combination", "")
        if len(comb.split(separator)) != 3:
            continue
        q = 1.0 / odds
        vals[comb] = q
        total += q
    if total <= 0:
        return {}
    return {k: v / total for k, v in vals.items()}


def top3_results():
    by = defaultdict(list)
    for r in read_csv(DATA / "results.csv"):
        try:
            fp = int(r.get("finish_position") or 999)
            car = int(r.get("car_no") or 0)
        except Exception:
            continue
        if 1 <= fp <= 3 and car > 0:
            by[r["race_id"]].append((fp, car))
    out = {}
    for rid, xs in by.items():
        xs = sorted(xs)
        if len(xs) == 3 and [x[0] for x in xs] == [1, 2, 3]:
            out[rid] = tuple(sorted(x[1] for x in xs))
    return out


def load_markets():
    tri = defaultdict(list)
    trio = defaultdict(list)
    for r in read_csv(DATA / "trifecta_final_odds.csv"):
        tri[r["race_id"]].append(r)
    for r in read_csv(DATA / "trio_final_odds.csv"):
        trio[r["race_id"]].append(r)
    return tri, trio


def effective_order_count(order_probs):
    s = sum(order_probs)
    if s <= 0:
        return 0.0, 0
    shares = [p / s for p in order_probs if p > 0]
    entropy = -sum(x * math.log(x) for x in shares)
    eff = math.exp(entropy)
    support10 = sum(1 for x in shares if x >= 0.10)
    return eff, support10


def build_race(tri_rows, trio_rows):
    p_order = norm_inv_odds(tri_rows, "-")
    p_trio_raw = norm_inv_odds(trio_rows, "=")
    if not p_order or not p_trio_raw:
        return None

    set_order_probs = defaultdict(list)
    for comb, p in p_order.items():
        try:
            cars = tuple(sorted(int(x) for x in comb.split("-")))
        except Exception:
            continue
        if len(cars) == 3 and len(set(cars)) == 3:
            set_order_probs[cars].append(p)

    trio_probs = {}
    trio_odds = {}
    for r in trio_rows:
        if r.get("odds_status") != "available":
            continue
        try:
            cars = tuple(sorted(int(x) for x in r["combination"].split("=")))
            odds = float(r["odds"])
        except Exception:
            continue
        if odds <= 1.0 or r["combination"] not in p_trio_raw:
            continue
        trio_probs[cars] = p_trio_raw[r["combination"]]
        trio_odds[cars] = odds

    common = sorted(set(set_order_probs) & set(trio_probs))
    if len(common) < 10:
        return None

    out = {}
    for cars in common:
        orders = set_order_probs[cars]
        p_tri_set = sum(orders)
        p_trio = trio_probs[cars]
        eff, support10 = effective_order_count(orders)
        out[cars] = {
            "p_trio": p_trio,
            "p_tri_set": p_tri_set,
            "odds": trio_odds[cars],
            "consensus": math.sqrt(max(p_trio, 0.0) * max(p_tri_set, 0.0)),
            "delta": p_tri_set - p_trio,
            "ratio": p_tri_set / p_trio if p_trio > 0 else None,
            "effective_orders": eff,
            "support_orders_10pct": support10,
        }
    return out


def dutch_units(odds):
    """Find 100-yen units such that every selected ticket is strictly profitable."""
    if not odds or any(o <= 1.0 for o in odds):
        return None
    if sum(1.0 / o for o in odds) >= 1.0 - 1e-12:
        return None
    target = len(odds)
    for _ in range(2000):
        units = [math.floor(target / o) + 1 for o in odds]
        actual = sum(units)
        if actual <= target and all(u * o > actual + 1e-12 for u, o in zip(units, odds)):
            return units
        target = max(target + 1, actual)
        if target > 1_000_000:
            return None
    return None


def middle_scenarios(market, main1):
    main_set = set(main1)
    all_cars = sorted({c for cars in market for c in cars})
    outsiders = [c for c in all_cars if c not in main_set]
    scenarios = []
    for outsider in outsiders:
        family = []
        for dropped in main1:
            cars = tuple(sorted((main_set - {dropped}) | {outsider}))
            if cars in market:
                family.append(cars)
        positive = [c for c in family if market[c]["delta"] > 0]
        # A one-car-break scenario must be supported in at least two of the
        # three possible pairings with the main trio. This is a cluster rule,
        # not a popularity or odds-band rule.
        if len(positive) < 2:
            continue
        representative = max(
            positive,
            key=lambda c: (market[c]["delta"], market[c]["p_tri_set"], c),
        )
        score = sum(market[c]["delta"] for c in positive)
        scenarios.append({
            "kind": "middle",
            "key": f"outsider_{outsider}",
            "cars": representative,
            "score": score,
            "support_sets": len(positive),
        })
    return sorted(scenarios, key=lambda x: (-x["score"], x["cars"]))


def deep_hole_scenarios(market, main1):
    main_set = set(main1)
    all_cars = sorted({c for cars in market for c in cars})
    outsiders = [c for c in all_cars if c not in main_set]

    qualifying = []
    for cars, x in market.items():
        overlap = len(set(cars) & main_set)
        if overlap > 1:
            continue
        if x["delta"] <= 0:
            continue
        if x["effective_orders"] + 1e-12 < MIN_EFFECTIVE_ORDERS_HOLE:
            continue
        if x["support_orders_10pct"] < MIN_SUPPORT_ORDERS_10PCT_HOLE:
            continue
        qualifying.append(cars)

    by_rep = {}
    for pair in itertools.combinations(outsiders, 2):
        family = [c for c in qualifying if set(pair).issubset(c)]
        # A deep alternative must have the same outsider core pair supported
        # across at least two distinct 3-car sets.
        if len(family) < 2:
            continue
        representative = max(
            family,
            key=lambda c: (
                market[c]["delta"] * (market[c]["effective_orders"] / 6.0) * (3 - len(set(c) & main_set)),
                market[c]["p_tri_set"],
                c,
            ),
        )
        scenario_score = sum(market[c]["delta"] for c in family)
        item = {
            "kind": "hole",
            "key": f"core_{pair[0]}_{pair[1]}",
            "cars": representative,
            "score": scenario_score,
            "support_sets": len(family),
        }
        old = by_rep.get(representative)
        if old is None or item["score"] > old["score"]:
            by_rep[representative] = item
    return sorted(by_rep.values(), key=lambda x: (-x["score"], x["cars"]))


def candidate_portfolio(market):
    main1 = max(market, key=lambda c: (market[c]["consensus"], tuple(-x for x in c)))
    middle = middle_scenarios(market, main1)
    holes = deep_hole_scenarios(market, main1)

    selected = [{"kind": "main", "key": "main1", "cars": main1, "score": None, "support_sets": None}]
    rejected_no_trigami = 0

    # v1 priority: main -> ordinary one-car-break scenarios -> truly distinct scenarios.
    for cand in middle + holes:
        trial = selected + [cand]
        alloc = dutch_units([market[x["cars"]]["odds"] for x in trial])
        if alloc is None:
            rejected_no_trigami += 1
            continue
        selected = trial

    units = dutch_units([market[x["cars"]]["odds"] for x in selected])
    if units is None:
        return None, "final_no_trigami_failure"

    total_units = sum(units)
    tickets = []
    for item, u in zip(selected, units):
        x = market[item["cars"]]
        hit_roi = x["odds"] * u / total_units * 100.0
        tickets.append({
            **item,
            **x,
            "units": u,
            "stake_yen": u * 100,
            "hit_roi_pct": hit_roi,
        })
    if not all(t["hit_roi_pct"] > 100.0 + 1e-10 for t in tickets):
        raise AssertionError("trigami ticket leaked")
    return {
        "tickets": tickets,
        "total_stake_yen": total_units * 100,
        "min_hit_roi_pct": min(t["hit_roi_pct"] for t in tickets),
        "reciprocal_odds_sum": sum(1.0 / t["odds"] for t in tickets),
        "middle_candidates": len(middle),
        "hole_candidates": len(holes),
        "rejected_no_trigami": rejected_no_trigami,
    }, None


def median(xs):
    if not xs:
        return None
    ys = sorted(xs)
    n = len(ys)
    if n % 2:
        return ys[n // 2]
    return (ys[n // 2 - 1] + ys[n // 2]) / 2


def main():
    tri, trio = load_markets()
    winners = top3_results()
    race_ids = sorted(set(tri) & set(trio) & set(winners))

    eligible = purchased = hit_races = 0
    total_stake = total_payout = 0.0
    total_rejected_no_trigami = 0
    trigami_violations = 0
    losing_streak = max_losing_streak = 0
    race_stakes = []
    min_hit_rois = []
    decisions = []

    comp = {
        "main": {"tickets": 0, "wins": 0, "stake": 0.0, "payout": 0.0, "odds": [], "winning_odds": []},
        "middle": {"tickets": 0, "wins": 0, "stake": 0.0, "payout": 0.0, "odds": [], "winning_odds": []},
        "hole": {"tickets": 0, "wins": 0, "stake": 0.0, "payout": 0.0, "odds": [], "winning_odds": []},
    }

    for rid in race_ids:
        market = build_race(tri[rid], trio[rid])
        if market is None or winners[rid] not in market:
            continue
        eligible += 1
        portfolio, reason = candidate_portfolio(market)
        if portfolio is None:
            decisions.append({"race_id": rid, "purchased": 0, "reason": reason})
            continue

        purchased += 1
        stake = portfolio["total_stake_yen"]
        total_stake += stake
        race_stakes.append(stake)
        min_hit_rois.append(portfolio["min_hit_roi_pct"])
        total_rejected_no_trigami += portfolio["rejected_no_trigami"]
        winner = winners[rid]
        winning_ticket = None

        for t in portfolio["tickets"]:
            c = comp[t["kind"]]
            c["tickets"] += 1
            c["stake"] += t["stake_yen"]
            c["odds"].append(t["odds"])
            if t["hit_roi_pct"] <= 100.0 + 1e-10:
                trigami_violations += 1
            if t["cars"] == winner:
                winning_ticket = t
                c["wins"] += 1
                payout = t["stake_yen"] * t["odds"]
                c["payout"] += payout
                c["winning_odds"].append(t["odds"])

        if winning_ticket is not None:
            hit_races += 1
            payout = winning_ticket["stake_yen"] * winning_ticket["odds"]
            total_payout += payout
            losing_streak = 0
        else:
            losing_streak += 1
            max_losing_streak = max(max_losing_streak, losing_streak)

        tickets_desc = " | ".join(
            f"{t['kind']}:{'-'.join(map(str,t['cars']))}@{t['odds']:.1f}x{t['units']}u"
            for t in portfolio["tickets"]
        )
        decisions.append({
            "race_id": rid,
            "purchased": 1,
            "reason": "",
            "winner": "-".join(map(str, winner)),
            "winner_selected": int(winning_ticket is not None),
            "winner_kind": winning_ticket["kind"] if winning_ticket else "",
            "stake_yen": round(stake),
            "payout_yen": round(winning_ticket["stake_yen"] * winning_ticket["odds"], 1) if winning_ticket else 0,
            "profit_yen": round((winning_ticket["stake_yen"] * winning_ticket["odds"] if winning_ticket else 0) - stake, 1),
            "ticket_count": len(portfolio["tickets"]),
            "main_count": sum(t["kind"] == "main" for t in portfolio["tickets"]),
            "middle_count": sum(t["kind"] == "middle" for t in portfolio["tickets"]),
            "hole_count": sum(t["kind"] == "hole" for t in portfolio["tickets"]),
            "middle_candidates": portfolio["middle_candidates"],
            "hole_candidates": portfolio["hole_candidates"],
            "min_hit_roi_pct": round(portfolio["min_hit_roi_pct"], 6),
            "tickets": tickets_desc,
        })

    DETAIL.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "race_id", "purchased", "reason", "winner", "winner_selected", "winner_kind",
        "stake_yen", "payout_yen", "profit_yen", "ticket_count", "main_count",
        "middle_count", "hole_count", "middle_candidates", "hole_candidates",
        "min_hit_roi_pct", "tickets",
    ]
    with DETAIL.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(decisions)

    components = {}
    for kind, c in comp.items():
        components[kind] = {
            "tickets": c["tickets"],
            "wins": c["wins"],
            "ticket_hit_rate_pct": c["wins"] / c["tickets"] * 100 if c["tickets"] else 0.0,
            "stake_yen": round(c["stake"]),
            "payout_yen": round(c["payout"]),
            "profit_yen": round(c["payout"] - c["stake"]),
            "roi_pct": c["payout"] / c["stake"] * 100 if c["stake"] else 0.0,
            "median_odds": median(c["odds"]),
            "median_winning_odds": median(c["winning_odds"]),
        }

    total_tickets = sum(c["tickets"] for c in comp.values())
    summary = {
        "status": "MARKET_SCENARIO_PORTFOLIO_2023_V1_HOBBY_SIMULATION",
        "year": YEAR,
        "years_read": [2023],
        "evaluation_year_2024_used": False,
        "evaluation_year_2025_used": False,
        "evaluation_year_2026_used": False,
        "odds_phase": "final",
        "important_limit": "Exploratory 2023 in-sample hobby simulation using final odds. v1 rules were defined from prior 2023 exploratory findings, so this is not OOS evidence and not T-10 executable performance.",
        "strategy_v1_machine_definition": {
            "main": "Single highest-consensus set by sqrt(P_trio * P_trifecta_set).",
            "middle_scenario": "For each outsider relative to main1, inspect the three one-car-replacement sets. Require P_trifecta_set>P_trio in at least two of the three; buy only the strongest-delta representative.",
            "deep_hole_scenario": "Among sets sharing <=1 main car, require positive cross-market delta plus broad six-order support. A common outsider core pair must recur in at least two qualifying alternative sets; buy one representative per distinct supported core, deduplicated by representative ticket.",
            "selection_order": "main -> all qualifying middle scenario representatives by scenario score -> all qualifying deep-hole representatives by scenario score, each only if the full 100-yen-unit dutched portfolio remains strictly profitable on every ticket.",
            "popularity_or_odds_band_used_for_scenario_selection": False,
            "no_trigami": "Every selected ticket must pay strictly more than total race stake after integer 100-yen dutching."
        },
        "eligible_complete_races": eligible,
        "purchased_races": purchased,
        "purchase_rate_pct": purchased / eligible * 100 if eligible else 0.0,
        "hit_races": hit_races,
        "race_hit_rate_pct": hit_races / purchased * 100 if purchased else 0.0,
        "total_tickets": total_tickets,
        "avg_tickets_per_race": total_tickets / purchased if purchased else 0.0,
        "total_stake_yen": round(total_stake),
        "total_payout_yen": round(total_payout),
        "profit_yen": round(total_payout - total_stake),
        "roi_pct": total_payout / total_stake * 100 if total_stake else 0.0,
        "median_race_stake_yen": median(race_stakes),
        "max_race_stake_yen": max(race_stakes) if race_stakes else None,
        "minimum_theoretical_hit_roi_pct_across_purchased_races": min(min_hit_rois) if min_hit_rois else None,
        "median_minimum_hit_roi_pct": median(min_hit_rois),
        "max_consecutive_losing_purchased_races": max_losing_streak,
        "candidate_additions_rejected_by_no_trigami": total_rejected_no_trigami,
        "trigami_violations": trigami_violations,
        "components": components,
    }
    OUT.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
