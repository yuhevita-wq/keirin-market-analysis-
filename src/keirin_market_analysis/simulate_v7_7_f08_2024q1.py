from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from v7_7_f08_joint_prefix_hierarchy import build_v7_7_f08

SCHEME = "v7.7-F08"
STAKE = 100
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/2024/s_class_f1_all_parts/2024_q1"
OUT = ROOT / "artifacts/v7_7_f08_2024q1"


def rr(name):
    with (DATA / name).open("r", encoding="utf-8-sig", newline="") as f:
        yield from csv.DictReader(f)


def pi(x):
    try:
        return int(float(x))
    except (TypeError, ValueError):
        return None


def pf(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def pt(s):
    xs = tuple(sorted(int(v) for v in (s or "").replace("=", "-").replace(",", "-").split("-") if v.strip().isdigit()))
    return xs if len(xs) == 3 and len(set(xs)) == 3 else None


def p3(s):
    xs = tuple(int(v) for v in (s or "").replace("=", "-").replace(",", "-").split("-") if v.strip().isdigit())
    return xs if len(xs) == 3 and len(set(xs)) == 3 else None


def pl(s):
    lines = []
    for raw in (s or "").split("/"):
        if not raw.strip():
            continue
        parts = raw.strip().split("-")
        if not all(v.strip().isdigit() for v in parts):
            return None
        lines.append(tuple(int(v) for v in parts))
    flat = [v for line in lines for v in line]
    return tuple(lines) if lines and len(flat) == len(set(flat)) else None


def load():
    races = {r["race_id"]: r for r in rr("races.csv")}
    trio, tf, pay = defaultdict(dict), defaultdict(dict), defaultdict(dict)
    for r in rr("trio_final_odds.csv"):
        c, o = pt(r.get("combination")), pf(r.get("odds"))
        if c and o and o > 0 and r.get("odds_status") == "available":
            trio[r["race_id"]][c] = o
    for r in rr("trifecta_final_odds.csv"):
        c, o = p3(r.get("combination")), pf(r.get("odds"))
        if c and o and o > 0 and r.get("odds_status") == "available":
            tf[r["race_id"]][c] = o
    for r in rr("payouts.csv"):
        if r.get("bet_code") != "trifecta" and r.get("ticket_type") != "3連単":
            continue
        if r.get("status") != "paid":
            continue
        c, y = p3(r.get("combination")), pi(r.get("payout_yen"))
        if c and y is not None:
            pay[r["race_id"]][c] = y
    return races, trio, tf, pay


def streak(rows):
    cur = best = 0
    for r in sorted(rows, key=lambda x: (x["race_date"], x["race_id"])):
        if r["hit"]:
            cur = 0
        else:
            cur += 1
            best = max(best, cur)
    return best


def main():
    races, trio, tf, pay = load()
    out = []
    fail = Counter()
    sizes = Counter()
    prefix_shapes = Counter()
    attr = defaultdict(int)
    attr["races_csv"] = len(races)

    for rid, r in sorted(races.items(), key=lambda kv: (kv[1].get("race_date", ""), kv[0])):
        if r.get("meeting_grade") != "F1" or not (r.get("race_type") or "").startswith("Ｓ級"):
            continue
        attr["f1_s"] += 1
        if pi(r.get("entry_count")) != 7:
            continue
        attr["seven_car"] += 1
        if len(trio.get(rid, {})) != 35:
            continue
        attr["complete_trio35"] += 1
        if len(tf.get(rid, {})) != 210:
            continue
        attr["complete_tf210"] += 1
        cars = sorted({v for c in trio[rid] for v in c})
        lines = pl(r.get("predicted_line_formation"))
        if len(cars) != 7 or lines is None or set(v for line in lines for v in line) != set(cars):
            continue
        attr["complete_line"] += 1
        if rid not in pay or not pay[rid]:
            continue
        attr["has_tf_payout"] += 1
        attr["population"] += 1

        d = build_v7_7_f08(trio[rid], tf[rid], r.get("predicted_line_formation") or "")
        if not d.get("buy"):
            fail[str(d.get("reason"))] += 1
            continue
        attr["entry_pass"] += 1
        tickets = tuple(d["tickets"])
        wins = [t for t in tickets if t in pay[rid]]
        payout = sum(pay[rid][t] for t in wins)
        size = f"{len(d['first'])}-{len(d['second'])}-{len(d['third'])}"
        prefix = f"{len(d['first'])}-{len(d['second'])}"
        sizes[size] += 1
        prefix_shapes[prefix] += 1
        out.append({
            "race_id": rid,
            "race_date": r.get("race_date"),
            "track": r.get("track"),
            "race_no": pi(r.get("race_no")),
            "formation": d["formation"],
            "size_pattern": size,
            "prefix_count": d["prefix_count"],
            "prefix_mass": d["prefix_mass"],
            "ticket_count": d["ticket_count"],
            "market_mass": d["market_mass"],
            "hit": int(bool(wins)),
            "payout_yen": payout,
            "winning_ticket": ";".join("-".join(map(str, t)) for t in wins),
        })

    attr["bet_races"] = len(out)
    hits = sum(r["hit"] for r in out)
    tickets = sum(r["ticket_count"] for r in out)
    payout = sum(r["payout_yen"] for r in out)
    stake = tickets * STAKE
    result = {
        "scheme_version": SCHEME,
        "dataset": "2024Q1",
        "entry_rule": "unchanged v6.1 pre-formation gate",
        "formation_rule": "joint 1->2 prefix Pareto knee, then conditional 3rd-place Pareto knee",
        "fixed_place_counts": False,
        "price_cut": False,
        "attrition": dict(attr),
        "entry_fail_reasons": dict(fail),
        "summary": {
            "bet_races": len(out),
            "hit_races": hits,
            "miss_races": len(out) - hits,
            "hit_rate_pct": 100 * hits / len(out) if out else None,
            "tickets": tickets,
            "avg_tickets_per_race": tickets / len(out) if out else None,
            "min_tickets_per_race": min((r["ticket_count"] for r in out), default=None),
            "max_tickets_per_race": max((r["ticket_count"] for r in out), default=None),
            "stake_yen": stake,
            "payout_yen": payout,
            "profit_yen": payout - stake,
            "roi_pct": 100 * payout / stake if stake else None,
            "max_losing_streak": streak(out),
        },
        "prefix_size_distribution": dict(sorted(prefix_shapes.items(), key=lambda kv: (-kv[1], kv[0]))),
        "size_pattern_distribution": dict(sorted(sizes.items(), key=lambda kv: (-kv[1], kv[0]))),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "v7_7_f08_2024q1_summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    if out:
        with (OUT / "v7_7_f08_2024q1_races.csv").open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
            w.writeheader()
            w.writerows(out)
    print("V7_7_F08_Q1_RESULT_BEGIN")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print("V7_7_F08_Q1_RESULT_END")


if __name__ == "__main__":
    main()
