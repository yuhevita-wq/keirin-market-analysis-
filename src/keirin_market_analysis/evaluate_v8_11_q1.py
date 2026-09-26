from __future__ import annotations

import json
from collections import Counter, defaultdict

from simulate_v8_1_f02_2024q1 import load, pi, pl
from v8_11_f12_race_type_adaptive import build_v8_11_f12, classify_race_type

STAKE = 100
SCHEME = "v8.11-F12"
GROUP_ORDER = ("QUALIFYING", "GENERAL", "SEMIFINAL", "SPECIAL", "FINAL", "OTHER")


def _summary(rows):
    n = len(rows)
    hits = sum(r["hit"] for r in rows)
    tickets_before = sum(r["tickets_before"] for r in rows)
    tickets_after = sum(r["tickets_after"] for r in rows)
    payout = sum(r["payout_yen"] for r in rows)
    stake = tickets_after * STAKE
    profitable_hits = sum(1 for r in rows if r["hit"] and r["payout_yen"] > r["tickets_after"] * STAKE)
    losing_hits = sum(1 for r in rows if r["hit"] and r["payout_yen"] < r["tickets_after"] * STAKE)
    breakeven_hits = sum(1 for r in rows if r["hit"] and r["payout_yen"] == r["tickets_after"] * STAKE)
    return {
        "bet_races": n,
        "hits": hits,
        "hit_rate_pct": 100 * hits / n if n else None,
        "tickets_before_price": tickets_before,
        "tickets_after_price": tickets_after,
        "avg_tickets_before": tickets_before / n if n else None,
        "avg_tickets_after": tickets_after / n if n else None,
        "ticket_reduction_pct": 100 * (tickets_before - tickets_after) / tickets_before if tickets_before else None,
        "compressed_races": sum(r["compressed"] for r in rows),
        "stake_yen": stake,
        "payout_yen": payout,
        "profit_yen": payout - stake,
        "roi_pct": 100 * payout / stake if stake else None,
        "profitable_hit_races": profitable_hits,
        "losing_hit_races": losing_hits,
        "breakeven_hit_races": breakeven_hits,
        "profitable_hit_share_pct": 100 * profitable_hits / hits if hits else None,
    }


def _max_losing_streak(rows):
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
    rows = []
    fail = Counter()
    population_by_group = Counter()
    bet_rows_by_group = defaultdict(list)

    population = 0
    for rid, r in sorted(races.items(), key=lambda x: (x[1].get("race_date", ""), x[0])):
        if r.get("meeting_grade") != "F1" or not (r.get("race_type") or "").startswith("Ｓ級"):
            continue
        if pi(r.get("entry_count")) != 7 or len(trio.get(rid, {})) != 35 or len(tf.get(rid, {})) != 210:
            continue
        cars = sorted({v for c in trio[rid] for v in c})
        lines = pl(r.get("predicted_line_formation"))
        if len(cars) != 7 or lines is None or set(v for line in lines for v in line) != set(cars):
            continue
        if rid not in pay or not pay[rid]:
            continue

        population += 1
        race_type = r.get("race_type") or ""
        group = classify_race_type(race_type)
        population_by_group[group] += 1

        d = build_v8_11_f12(
            trio[rid],
            tf[rid],
            r.get("predicted_line_formation") or "",
            race_type,
        )
        if not d.get("buy"):
            fail[f"{group}:{d.get('reason')}"] += 1
            continue

        wins = [t for t in d["tickets"] if t in pay[rid]]
        payout = sum(pay[rid][t] for t in wins)
        before = int(d.get("ticket_count_before_price") or d["ticket_count"])
        after = int(d["ticket_count"])
        row = {
            "race_id": rid,
            "race_date": r.get("race_date") or "",
            "race_type": race_type,
            "group": group,
            "formation_before": d.get("formation_before_price"),
            "formation_after": d.get("formation"),
            "tickets_before": before,
            "tickets_after": after,
            "compressed": int(after < before),
            "hit": int(bool(wins)),
            "payout_yen": payout,
            "race_profit_yen": payout - after * STAKE,
            "H_AB": int(bool((d.get("entry_gate") or {}).get("H_AB"))),
            "H_RATIO": int(bool((d.get("entry_gate") or {}).get("H_RATIO"))),
            "race_type_rule": d.get("race_type_rule"),
        }
        rows.append(row)
        bet_rows_by_group[group].append(row)

    overall = _summary(rows)
    overall["max_losing_streak"] = _max_losing_streak(rows)

    group_results = {}
    for group in GROUP_ORDER:
        s = _summary(bet_rows_by_group[group])
        s["population"] = population_by_group[group]
        s["buy_rate_pct"] = 100 * s["bet_races"] / population_by_group[group] if population_by_group[group] else None
        group_results[group] = s

    result = {
        "scheme": SCHEME,
        "dataset": "2024Q1",
        "status": "FIXED_RULE_DEVELOPMENT_SIMULATION",
        "population": population,
        "population_by_group": dict(population_by_group),
        "overall": overall,
        "by_race_type_group": group_results,
        "entry_fail_reasons": dict(fail),
        "acceptance": "PASS_Q1_PROFIT" if overall["profit_yen"] > 0 else "REJECT_Q1_NOT_PROFITABLE",
        "notes": [
            "Race-type rules were fixed before this Q1 run and were not changed during simulation.",
            "Formation is built before price compression.",
            "Price compression removes whole riders from place-sets only; individual ticket pruning is prohibited.",
            "2024Q1 remains development data; later untouched OOS validation is required for any accepted scheme.",
        ],
    }

    print("V8_11_F12_Q1_RESULT_BEGIN")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print("V8_11_F12_Q1_RESULT_END")

    # Deterministic top/bottom group audit for quick visual inspection.
    audit = []
    for group in GROUP_ORDER:
        gr = bet_rows_by_group[group]
        if not gr:
            continue
        best = max(gr, key=lambda r: (r["race_profit_yen"], r["race_id"]))
        worst = min(gr, key=lambda r: (r["race_profit_yen"], r["race_id"]))
        audit.append({"group": group, "sample": "BEST", **best})
        if worst["race_id"] != best["race_id"]:
            audit.append({"group": group, "sample": "WORST", **worst})
    print("V8_11_F12_Q1_AUDIT_BEGIN")
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    print("V8_11_F12_Q1_AUDIT_END")


if __name__ == "__main__":
    main()
