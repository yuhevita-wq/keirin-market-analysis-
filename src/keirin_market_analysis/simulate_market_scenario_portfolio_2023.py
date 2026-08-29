from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

YEAR = 2023
DATA = Path(f"data/{YEAR}/s_class_yosen")
OUT = Path("data/audits/market_scenario_portfolio_2023.json")
DETAIL = Path("data/audits/market_scenario_portfolio_2023_decisions.csv")

MAX_HOLES = 3
MIN_EFFECTIVE_ORDERS = 3.0
MIN_SUPPORT_ORDERS_10PCT = 3
MAX_OVERLAP_WITH_MAIN = 1  # at least two of the three cars differ from the main scenario


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
        parts = comb.split(separator)
        if len(parts) != 3:
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
        consensus = math.sqrt(max(p_trio, 0.0) * max(p_tri_set, 0.0))
        delta = p_tri_set - p_trio
        ratio = p_tri_set / p_trio if p_trio > 0 else None
        out[cars] = {
            "p_trio": p_trio,
            "p_tri_set": p_tri_set,
            "odds": trio_odds[cars],
            "consensus": consensus,
            "delta": delta,
            "ratio": ratio,
            "effective_orders": eff,
            "support_orders_10pct": support10,
        }
    return out


def dutch_units(odds):
    """Return minimum-ish 100-yen units that make every selected ticket strictly profitable.

    The necessary continuous condition is sum(1/odds) < 1. Integer rounding is
    handled by a monotone fixed-point search. No result data are used.
    """
    if not odds or any(o <= 1.0 for o in odds):
        return None
    inv_sum = sum(1.0 / o for o in odds)
    if inv_sum >= 1.0 - 1e-12:
        return None

    target = len(odds)
    for _ in range(1000):
        units = [math.floor(target / o) + 1 for o in odds]
        actual = sum(units)
        if actual <= target:
            if all(u * o > actual + 1e-12 for u, o in zip(units, odds)):
                return units
        nxt = max(target + 1, actual)
        if nxt > 1_000_000:
            return None
        target = nxt
    return None


def candidate_portfolio(market):
    ranked_main = sorted(market, key=lambda c: (-market[c]["consensus"], c))
    if not ranked_main:
        return None, "no_main"
    main1 = ranked_main[0]
    main2 = ranked_main[1] if len(ranked_main) > 1 else None

    holes = []
    for cars, x in market.items():
        if cars == main1 or cars == main2:
            continue
        overlap = len(set(cars) & set(main1))
        distance = 3 - overlap
        if overlap > MAX_OVERLAP_WITH_MAIN:
            continue
        if x["delta"] <= 0:
            continue
        if x["effective_orders"] + 1e-12 < MIN_EFFECTIVE_ORDERS:
            continue
        if x["support_orders_10pct"] < MIN_SUPPORT_ORDERS_10PCT:
            continue
        breadth = x["effective_orders"] / 6.0
        score = x["delta"] * breadth * distance
        holes.append((score, cars))
    holes.sort(key=lambda z: (-z[0], z[1]))
    if not holes:
        return None, "no_alternative_scenario"

    # Core philosophy: main scenario first, then genuinely different alternatives.
    # Main2 is optional and is attempted only after at least one hole survives the
    # all-green constraint, so the portfolio cannot silently become "two favorites only".
    selected = [("main", main1, None)]
    hole_count = 0
    rejected_no_trigami = 0
    for score, cars in holes:
        if hole_count >= MAX_HOLES:
            break
        trial = selected + [("hole", cars, score)]
        alloc = dutch_units([market[c]["odds"] for _, c, _ in trial])
        if alloc is None:
            rejected_no_trigami += 1
            continue
        selected = trial
        hole_count += 1

    if hole_count == 0:
        return None, "holes_fail_no_trigami"

    if main2 is not None:
        trial = selected + [("main", main2, None)]
        if dutch_units([market[c]["odds"] for _, c, _ in trial]) is not None:
            selected = trial
        else:
            rejected_no_trigami += 1

    units = dutch_units([market[c]["odds"] for _, c, _ in selected])
    if units is None:
        return None, "final_no_trigami_failure"

    total_units = sum(units)
    tickets = []
    for (kind, cars, score), u in zip(selected, units):
        odds = market[cars]["odds"]
        hit_roi = odds * u / total_units * 100.0
        tickets.append({
            "kind": kind,
            "cars": cars,
            "score": score,
            "units": u,
            "stake_yen": u * 100,
            "odds": odds,
            "hit_roi_pct": hit_roi,
            **market[cars],
        })
    if not all(t["hit_roi_pct"] > 100.0 + 1e-10 for t in tickets):
        raise AssertionError("trigami ticket leaked into final portfolio")

    return {
        "tickets": tickets,
        "total_stake_yen": total_units * 100,
        "min_hit_roi_pct": min(t["hit_roi_pct"] for t in tickets),
        "reciprocal_odds_sum": sum(1.0 / t["odds"] for t in tickets),
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

    eligible = 0
    purchased = 0
    hit_races = 0
    main_hits = 0
    hole_hits = 0
    total_stake = 0.0
    total_payout = 0.0
    all_ticket_count = 0
    hole_ticket_count = 0
    main_ticket_count = 0
    main_odds = []
    hole_odds = []
    min_hit_rois = []
    race_stakes = []
    reject_reasons = defaultdict(int)
    total_rejected_no_trigami = 0
    losing_streak = 0
    max_losing_streak = 0
    decisions = []

    for rid in race_ids:
        market = build_race(tri[rid], trio[rid])
        if market is None or winners[rid] not in market:
            reject_reasons["incomplete_market"] += 1
            continue
        eligible += 1
        portfolio, reason = candidate_portfolio(market)
        if portfolio is None:
            reject_reasons[reason] += 1
            decisions.append({
                "race_id": rid,
                "purchased": 0,
                "reason": reason,
                "winner": "-".join(map(str, winners[rid])),
            })
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
            all_ticket_count += 1
            if t["kind"] == "hole":
                hole_ticket_count += 1
                hole_odds.append(t["odds"])
            else:
                main_ticket_count += 1
                main_odds.append(t["odds"])
            if t["cars"] == winner:
                winning_ticket = t

        if winning_ticket is not None:
            hit_races += 1
            payout = winning_ticket["stake_yen"] * winning_ticket["odds"]
            total_payout += payout
            if winning_ticket["kind"] == "hole":
                hole_hits += 1
            else:
                main_hits += 1
            losing_streak = 0
        else:
            losing_streak += 1
            max_losing_streak = max(max_losing_streak, losing_streak)

        ticket_desc = []
        for t in portfolio["tickets"]:
            ticket_desc.append(
                f"{t['kind']}:{'-'.join(map(str,t['cars']))}@{t['odds']:.1f}x{t['units']}u"
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
            "tickets": " | ".join(ticket_desc),
            "ticket_count": len(portfolio["tickets"]),
            "hole_count": sum(1 for t in portfolio["tickets"] if t["kind"] == "hole"),
            "main_count": sum(1 for t in portfolio["tickets"] if t["kind"] == "main"),
            "min_hit_roi_pct": round(portfolio["min_hit_roi_pct"], 6),
            "reciprocal_odds_sum": round(portfolio["reciprocal_odds_sum"], 9),
        })

    # CSV detail is intentionally result-bearing only for the 2023 hobby simulation.
    DETAIL.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "race_id", "purchased", "reason", "winner", "winner_selected", "winner_kind",
        "stake_yen", "payout_yen", "profit_yen", "tickets", "ticket_count", "hole_count",
        "main_count", "min_hit_roi_pct", "reciprocal_odds_sum",
    ]
    with DETAIL.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(decisions)

    roi = total_payout / total_stake * 100 if total_stake else 0.0
    summary = {
        "status": "MARKET_SCENARIO_PORTFOLIO_2023_V0_HOBBY_SIMULATION",
        "year": YEAR,
        "years_read": [2023],
        "evaluation_year_2024_used": False,
        "evaluation_year_2025_used": False,
        "evaluation_year_2026_used": False,
        "odds_phase": "final",
        "important_limit": "This is a 2023 in-sample hobby simulation using final odds. It is not OOS evidence and not executable T-10 performance.",
        "strategy_frozen_before_result_evaluation": {
            "main_definition": "Top market-consensus set by sqrt(normalized trio probability * normalized six-order trifecta-set probability).",
            "hole_definition": "Non-main set with trifecta-set probability above trio probability, effective six-order count >=3, at least three order shares >=10%, and overlap <=1 car with main1.",
            "hole_score": "(P_trifecta_set - P_trio) * (effective_order_count/6) * (3-overlap_with_main1). Popularity rank and odds band are not used.",
            "portfolio_order": "main1 -> up to 3 holes by score if all-green feasible -> optional main2 if still all-green.",
            "no_trigami_rule": "Every selected ticket must return strictly more than total race stake after 100-yen-unit dutching; otherwise it is excluded or the race is skipped.",
        },
        "parameters": {
            "max_holes": MAX_HOLES,
            "min_effective_orders": MIN_EFFECTIVE_ORDERS,
            "min_support_orders_10pct": MIN_SUPPORT_ORDERS_10PCT,
            "max_overlap_with_main1": MAX_OVERLAP_WITH_MAIN,
        },
        "eligible_complete_races": eligible,
        "purchased_races": purchased,
        "purchase_rate_pct": purchased / eligible * 100 if eligible else 0.0,
        "hit_races": hit_races,
        "race_hit_rate_pct": hit_races / purchased * 100 if purchased else 0.0,
        "main_hits": main_hits,
        "hole_hits": hole_hits,
        "total_tickets": all_ticket_count,
        "main_tickets": main_ticket_count,
        "hole_tickets": hole_ticket_count,
        "avg_tickets_per_purchased_race": all_ticket_count / purchased if purchased else 0.0,
        "avg_holes_per_purchased_race": hole_ticket_count / purchased if purchased else 0.0,
        "total_stake_yen": round(total_stake),
        "total_payout_yen": round(total_payout),
        "profit_yen": round(total_payout - total_stake),
        "roi_pct": roi,
        "median_race_stake_yen": median(race_stakes),
        "max_race_stake_yen": max(race_stakes) if race_stakes else None,
        "median_main_odds": median(main_odds),
        "median_hole_odds": median(hole_odds),
        "minimum_theoretical_hit_roi_pct_across_purchased_races": min(min_hit_rois) if min_hit_rois else None,
        "median_minimum_hit_roi_pct": median(min_hit_rois),
        "max_consecutive_losing_purchased_races": max_losing_streak,
        "candidate_additions_rejected_by_no_trigami": total_rejected_no_trigami,
        "reject_reasons": dict(sorted(reject_reasons.items())),
        "trigami_violations": 0,
        "note": "A trio race has only one winning three-car set, so a purchased race can be a main hit or a hole hit, never both.",
    }
    OUT.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
