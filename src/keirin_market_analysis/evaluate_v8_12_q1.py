from __future__ import annotations

from collections import Counter, defaultdict
import json

from simulate_v8_1_f02_2024q1 import load, pi, pl
from v8_11_f12_race_type_adaptive import classify_race_type
from v8_12_f13_branch_rebuild import build_v8_12_f13

STAKE = 100
SCHEME = "v8.12-F13"
GROUPS = ("QUALIFYING", "GENERAL", "SEMIFINAL", "SPECIAL", "FINAL", "OTHER")


def _streak(rows):
    cur = best = 0
    for r in sorted(rows, key=lambda x: (x["race_date"], x["race_id"])):
        if r["hit"]:
            cur = 0
        else:
            cur += 1
            best = max(best, cur)
    return best


def _summary(rows):
    n = len(rows)
    hits = sum(r["hit"] for r in rows)
    before = sum(r["ticket_count_before_price"] for r in rows)
    after = sum(r["ticket_count"] for r in rows)
    payout = sum(r["payout_yen"] for r in rows)
    stake = after * STAKE
    profitable_hits = sum(r["hit"] and r["race_profit_yen"] > 0 for r in rows)
    losing_hits = sum(r["hit"] and r["race_profit_yen"] < 0 for r in rows)
    breakeven_hits = sum(r["hit"] and r["race_profit_yen"] == 0 for r in rows)
    return {
        "bet_races": n,
        "hits": hits,
        "hit_rate_pct": 100 * hits / n if n else None,
        "tickets_before_price": before,
        "tickets_after_price": after,
        "avg_tickets_before": before / n if n else None,
        "avg_tickets_after": after / n if n else None,
        "ticket_reduction_pct": 100 * (before - after) / before if before else None,
        "compressed_races": sum(r["ticket_count"] < r["ticket_count_before_price"] for r in rows),
        "stake_yen": stake,
        "payout_yen": payout,
        "profit_yen": payout - stake,
        "roi_pct": 100 * payout / stake if stake else None,
        "profitable_hit_races": profitable_hits,
        "losing_hit_races": losing_hits,
        "breakeven_hit_races": breakeven_hits,
        "profitable_hit_share_pct": 100 * profitable_hits / hits if hits else None,
        "max_losing_streak": _streak(rows),
    }


def main():
    races, trio, tf, pay = load()
    rows = []
    population = Counter()
    fail = Counter()
    quarantine = Counter()
    exact_type_bets = Counter()
    exact_type_population = Counter()

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

        race_type = r.get("race_type") or ""
        group = classify_race_type(race_type)
        population[group] += 1
        exact_type_population[race_type] += 1

        d = build_v8_12_f13(
            trio[rid], tf[rid], r.get("predicted_line_formation") or "", race_type
        )
        if not d.get("buy"):
            reason = str(d.get("reason"))
            fail[f"{group}:{reason}"] += 1
            if reason.startswith("MAJOR_REBUILD_PENDING_"):
                quarantine[group] += 1
            continue

        wins = [t for t in d["tickets"] if t in pay[rid]]
        payout = sum(pay[rid][t] for t in wins)
        before = int(d.get("ticket_count_before_price") or d["ticket_count"])
        after = int(d["ticket_count"])
        hit = int(bool(wins))
        exact_type_bets[race_type] += 1
        rows.append({
            "race_id": rid,
            "race_date": r.get("race_date"),
            "race_type": race_type,
            "group": group,
            "formation_before": d.get("formation_before_price") or d.get("formation"),
            "formation_after": d.get("formation"),
            "ticket_count_before_price": before,
            "ticket_count": after,
            "hit": hit,
            "payout_yen": payout,
            "race_profit_yen": payout - after * STAKE,
            "price_q_retention": d.get("price_q_retention"),
            "entry_gate": d.get("entry_gate"),
        })

    overall = _summary(rows)
    by_group = {}
    for group in GROUPS:
        group_rows = [r for r in rows if r["group"] == group]
        s = _summary(group_rows)
        s["population"] = population[group]
        s["buy_rate_pct"] = 100 * len(group_rows) / population[group] if population[group] else None
        s["branch_policy"] = "MICRO_TUNE_ONLY" if group in ("QUALIFYING", "SEMIFINAL") else "QUARANTINED_REBUILD"
        by_group[group] = s

    result = {
        "scheme": SCHEME,
        "dataset": "2024Q1",
        "status": "FIXED_RULE_BRANCH_REBUILD_SIMULATION",
        "population": sum(population.values()),
        "population_by_group": dict(population),
        "overall": overall,
        "by_race_type_group": by_group,
        "quarantined_races": dict(quarantine),
        "exact_race_type_population": dict(exact_type_population),
        "exact_race_type_bets": dict(exact_type_bets),
        "entry_fail_reasons": dict(fail),
        "acceptance": "PASS_Q1_PROFIT" if overall["profit_yen"] > 0 else "REJECT_Q1_NOT_PROFITABLE",
        "notes": [
            "QUALIFYING and SEMIFINAL are inherited unchanged from v8.11-F12 at this architecture stage.",
            "GENERAL, SPECIAL and FINAL are quarantined and generate no bets until dedicated rebuild rules exist.",
            "No individual cheap-ticket pruning is allowed; price compression remains whole-rider structural only.",
            "2024Q1 is development data; this run is not out-of-sample validation.",
        ],
    }

    audit = []
    for group in ("QUALIFYING", "SEMIFINAL"):
        g = [r for r in rows if r["group"] == group]
        if not g:
            continue
        best = max(g, key=lambda r: (r["race_profit_yen"], r["race_id"]))
        worst = min(g, key=lambda r: (r["race_profit_yen"], r["race_id"]))
        for label, rr in (("BEST", best), ("WORST", worst)):
            audit.append({
                "group": group,
                "sample": label,
                "race_id": rr["race_id"],
                "race_date": rr["race_date"],
                "race_type": rr["race_type"],
                "formation_before": rr["formation_before"],
                "formation_after": rr["formation_after"],
                "tickets_before": rr["ticket_count_before_price"],
                "tickets_after": rr["ticket_count"],
                "hit": rr["hit"],
                "payout_yen": rr["payout_yen"],
                "race_profit_yen": rr["race_profit_yen"],
            })

    print("V8_12_F13_Q1_RESULT_BEGIN")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print("V8_12_F13_Q1_RESULT_END")
    print("V8_12_F13_Q1_AUDIT_BEGIN")
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    print("V8_12_F13_Q1_AUDIT_END")


if __name__ == "__main__":
    main()
