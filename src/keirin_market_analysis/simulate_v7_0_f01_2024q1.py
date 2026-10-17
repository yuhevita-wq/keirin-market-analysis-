from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from v7_0_f01_market_hierarchy import build_v7_0_f01

SCHEME_VERSION = "v7.0-F01"
DATASET = "2024Q1"
STATUS = "Q1_DEVELOPMENT_SIMULATION"
STAKE_YEN_PER_TICKET = 100

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/2024/s_class_f1_all_parts/2024_q1"
OUT = ROOT / "artifacts/v7_0_f01_2024q1"


def read_rows(name: str):
    with (DATA / name).open("r", encoding="utf-8-sig", newline="") as f:
        yield from csv.DictReader(f)


def to_int(x):
    try:
        return int(float(x))
    except (TypeError, ValueError):
        return None


def to_float(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def parse_trio(text: str):
    s = (text or "").strip().replace("=", "-").replace(",", "-")
    xs = tuple(sorted(int(x) for x in s.split("-") if x.strip().isdigit()))
    return xs if len(xs) == 3 and len(set(xs)) == 3 else None


def parse_tf(text: str):
    s = (text or "").strip().replace("=", "-").replace(",", "-")
    xs = tuple(int(x) for x in s.split("-") if x.strip().isdigit())
    return xs if len(xs) == 3 and len(set(xs)) == 3 else None


def parse_lines(text: str):
    lines = []
    for raw in (text or "").strip().split("/"):
        raw = raw.strip()
        if not raw:
            continue
        parts = raw.split("-")
        if not all(p.strip().isdigit() for p in parts):
            return None
        line = tuple(int(p.strip()) for p in parts)
        if not line:
            return None
        lines.append(line)
    flat = [car for line in lines for car in line]
    if not lines or len(flat) != len(set(flat)):
        return None
    return tuple(lines)


def load_data():
    races = {r["race_id"]: r for r in read_rows("races.csv")}

    trio = defaultdict(dict)
    for r in read_rows("trio_final_odds.csv"):
        combo = parse_trio(r.get("combination"))
        odds = to_float(r.get("odds"))
        if combo and odds and odds > 0 and r.get("odds_status") == "available":
            trio[r["race_id"]][combo] = odds

    tf = defaultdict(dict)
    for r in read_rows("trifecta_final_odds.csv"):
        combo = parse_tf(r.get("combination"))
        odds = to_float(r.get("odds"))
        if combo and odds and odds > 0 and r.get("odds_status") == "available":
            tf[r["race_id"]][combo] = odds

    payouts = defaultdict(dict)
    for r in read_rows("payouts.csv"):
        if r.get("bet_code") != "trifecta" and r.get("ticket_type") != "3連単":
            continue
        if r.get("status") != "paid":
            continue
        combo = parse_tf(r.get("combination"))
        payout = to_int(r.get("payout_yen"))
        if combo and payout is not None:
            payouts[r["race_id"]][combo] = payout

    return races, trio, tf, payouts


def max_losing_streak(rows):
    cur = best = 0
    for r in sorted(rows, key=lambda x: (x["race_date"], x["race_id"])):
        if int(r["hit"]):
            cur = 0
        else:
            cur += 1
            best = max(best, cur)
    return best


def summarize(rows):
    bet_races = len(rows)
    hit_races = sum(int(r["hit"]) for r in rows)
    tickets = sum(int(r["ticket_count"]) for r in rows)
    payout = sum(int(r["payout_yen"]) for r in rows)
    stake = tickets * STAKE_YEN_PER_TICKET
    return {
        "bet_races": bet_races,
        "hit_races": hit_races,
        "miss_races": bet_races - hit_races,
        "hit_rate_pct": 100 * hit_races / bet_races if bet_races else None,
        "tickets": tickets,
        "avg_tickets_per_race": tickets / bet_races if bet_races else None,
        "min_tickets_per_race": min((int(r["ticket_count"]) for r in rows), default=None),
        "max_tickets_per_race": max((int(r["ticket_count"]) for r in rows), default=None),
        "stake_yen": stake,
        "payout_yen": payout,
        "profit_yen": payout - stake,
        "roi_pct": 100 * payout / stake if stake else None,
        "max_losing_streak": max_losing_streak(rows),
    }


def main():
    races, trio, tf, payouts = load_data()
    attr = defaultdict(int)
    fail = Counter()
    shapes = Counter()
    size_patterns = Counter()
    rows = []

    attr["races_csv"] = len(races)

    for rid, r in sorted(races.items(), key=lambda kv: (kv[1].get("race_date", ""), kv[0])):
        if r.get("meeting_grade") != "F1" or not (r.get("race_type") or "").startswith("Ｓ級"):
            continue
        attr["f1_s"] += 1

        if to_int(r.get("entry_count")) != 7:
            continue
        attr["seven_car"] += 1

        if len(trio.get(rid, {})) != 35:
            continue
        attr["complete_trio35"] += 1

        if len(tf.get(rid, {})) != 210:
            continue
        attr["complete_tf210"] += 1

        cars = sorted({car for combo in trio[rid] for car in combo})
        if len(cars) != 7:
            continue

        lines = parse_lines(r.get("predicted_line_formation"))
        if lines is None or set(car for line in lines for car in line) != set(cars):
            continue
        attr["complete_line"] += 1

        if rid not in payouts or not payouts[rid]:
            continue
        attr["has_tf_payout"] += 1
        attr["population"] += 1

        decision = build_v7_0_f01(trio[rid], tf[rid], r.get("predicted_line_formation") or "")
        if not decision.get("buy"):
            fail[str(decision.get("reason", "UNKNOWN"))] += 1
            continue

        tickets = tuple(decision["tickets"])
        if not tickets:
            fail["NO_VALID_TICKETS"] += 1
            continue

        paid = payouts[rid]
        winning = [ticket for ticket in tickets if ticket in paid]
        payout = sum(paid[ticket] for ticket in winning)
        hit = bool(winning)

        first = tuple(decision["first"])
        second = tuple(decision["second"])
        third = tuple(decision["third"])
        shape = str(decision["formation"])
        size_pattern = f"{len(first)}-{len(second)}-{len(third)}"
        shapes[shape] += 1
        size_patterns[size_pattern] += 1

        boundaries = decision["boundaries"]
        rows.append({
            "race_id": rid,
            "race_date": r.get("race_date"),
            "track": r.get("track"),
            "race_no": to_int(r.get("race_no")),
            "race_type": r.get("race_type"),
            "predicted_line_formation": r.get("predicted_line_formation"),
            "formation": shape,
            "size_pattern": size_pattern,
            "first": "".join(map(str, first)),
            "second": "".join(map(str, second)),
            "third": "".join(map(str, third)),
            "first_cut_rank": boundaries["first"].cut_after_rank,
            "second_cut_rank_raw": boundaries["second"].cut_after_rank,
            "third_cut_rank_raw": boundaries["third"].cut_after_rank,
            "first_boundary_ratio": boundaries["first"].boundary_ratio,
            "second_boundary_ratio": boundaries["second"].boundary_ratio,
            "third_boundary_ratio": boundaries["third"].boundary_ratio,
            "formation_mass": decision["formation_mass"],
            "ticket_count": decision["ticket_count"],
            "hit": int(hit),
            "payout_yen": payout,
            "winning_ticket": ";".join("-".join(map(str, t)) for t in winning),
        })

    attr["entry_pass"] = len(rows)
    attr["bet_races"] = len(rows)
    summary = summarize(rows)

    result = {
        "scheme_version": SCHEME_VERSION,
        "dataset": DATASET,
        "status": STATUS,
        "formation_rule": "market hierarchy; variable rider counts; F1 subset F2 subset F3",
        "entry_rule": "unchanged v6.1 pre-formation gate",
        "price_cut": False,
        "fixed_place_counts": False,
        "attrition": dict(attr),
        "entry_fail_reasons": dict(fail),
        "summary": summary,
        "size_pattern_distribution": dict(sorted(size_patterns.items(), key=lambda kv: (-kv[1], kv[0]))),
        "top_20_exact_formations": dict(shapes.most_common(20)),
    }

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "v7_0_f01_2024q1_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if rows:
        with (OUT / "v7_0_f01_2024q1_races.csv").open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    print("V7_0_F01_Q1_RESULT_BEGIN")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print("V7_0_F01_Q1_RESULT_END")


if __name__ == "__main__":
    main()
