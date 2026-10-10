from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median

from simulate_v8_1_f02_2024q1 import STAKE, load, pi, pl, streak
from v8_8_f09_market_cliff import build_v8_8_f09

SCHEME = "v8.8-F09"
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts/v8_8_f09_2024q1"


def metrics(rows):
    rows = list(rows)
    races = len(rows)
    hits = sum(r["hit"] for r in rows)
    tickets = sum(r["ticket_count"] for r in rows)
    stake = tickets * STAKE
    payout = sum(r["payout_yen"] for r in rows)
    profitable_hits = sum(1 for r in rows if r["hit"] and r["payout_yen"] > r["ticket_count"] * STAKE)
    break_even_hits = sum(1 for r in rows if r["hit"] and r["payout_yen"] == r["ticket_count"] * STAKE)
    losing_hits = sum(1 for r in rows if r["hit"] and r["payout_yen"] < r["ticket_count"] * STAKE)
    hit_payouts = [r["payout_yen"] for r in rows if r["hit"]]
    return {
        "races": races,
        "hits": hits,
        "hit_rate_pct": 100 * hits / races if races else None,
        "tickets": tickets,
        "avg_tickets": tickets / races if races else None,
        "min_tickets": min((r["ticket_count"] for r in rows), default=None),
        "max_tickets": max((r["ticket_count"] for r in rows), default=None),
        "stake_yen": stake,
        "payout_yen": payout,
        "profit_yen": payout - stake,
        "roi_pct": 100 * payout / stake if stake else None,
        "profitable_hit_races": profitable_hits,
        "break_even_hit_races": break_even_hits,
        "losing_hit_races": losing_hits,
        "profitable_hit_share_pct": 100 * profitable_hits / hits if hits else None,
        "avg_hit_payout_yen": sum(hit_payouts) / len(hit_payouts) if hit_payouts else None,
        "median_hit_payout_yen": median(hit_payouts) if hit_payouts else None,
        "hits_ge_5000": sum(1 for y in hit_payouts if y >= 5000),
        "hits_ge_10000": sum(1 for y in hit_payouts if y >= 10000),
    }


def quantile_groups(rows, field, k=4):
    ordered = sorted(rows, key=lambda r: (r[field], r["race_date"], r["race_id"]))
    n = len(ordered)
    groups = {}
    for i in range(k):
        lo = i * n // k
        hi = (i + 1) * n // k
        part = ordered[lo:hi]
        label = f"Q{i+1}_{'LOW' if i == 0 else 'HIGH' if i == k-1 else 'MID'}"
        m = metrics(part)
        m["field_min"] = min((r[field] for r in part), default=None)
        m["field_max"] = max((r[field] for r in part), default=None)
        m["field_median"] = median([r[field] for r in part]) if part else None
        groups[label] = m
    return groups


def main():
    races, trio, tf, pay = load()
    out = []
    fail = Counter()
    sizes = Counter()
    hstates = Counter()
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

        d = build_v8_8_f09(trio[rid], tf[rid], r.get("predicted_line_formation") or "")
        if not d.get("buy"):
            fail[str(d.get("reason"))] += 1
            continue

        attr["ps_ab_pass"] += 1
        ts = tuple(d["tickets"])
        wins = [t for t in ts if t in pay[rid]]
        payout = sum(pay[rid][t] for t in wins)
        size = f"{len(d['first'])}-{len(d['second'])}-{len(d['third'])}"
        sizes[size] += 1
        hstates[d["H_state"]] += 1
        out.append({
            "race_id": rid,
            "race_date": r.get("race_date"),
            "track": r.get("track"),
            "race_no": pi(r.get("race_no")),
            "formation": d["formation"],
            "size_pattern": size,
            "ticket_count": d["ticket_count"],
            "selected_growth_step": d["selected_growth_step"],
            "growth_steps_total": d["growth_steps_total"],
            "cliff_drop": d["cliff_drop"],
            "q_mass": d["q_mass"],
            "profit_mass_1x": d["profit_mass_1x"],
            "profit_mass_2x": d["profit_mass_2x"],
            "median_selected_odds": d["median_selected_odds"],
            "weighted_gm_selected_odds": d["weighted_gm_selected_odds"],
            "H_AB": int(d["H_AB"]),
            "H_RATIO": int(d["H_RATIO"]),
            "H_state": d["H_state"],
            "hit": int(bool(wins)),
            "payout_yen": payout,
            "winning_ticket": ";".join("-".join(map(str, t)) for t in wins),
        })

    attr["bet_races"] = len(out)
    base = metrics(out)
    base["max_losing_streak"] = streak(out)

    by_h_state = {}
    for state in sorted(hstates):
        rows = [r for r in out if r["H_state"] == state]
        by_h_state[state] = metrics(rows)

    payout_bands = {
        "hit_lt_2000": sum(1 for r in out if r["hit"] and r["payout_yen"] < 2000),
        "hit_2000_4999": sum(1 for r in out if r["hit"] and 2000 <= r["payout_yen"] < 5000),
        "hit_5000_9999": sum(1 for r in out if r["hit"] and 5000 <= r["payout_yen"] < 10000),
        "hit_ge_10000": sum(1 for r in out if r["hit"] and r["payout_yen"] >= 10000),
    }

    result = {
        "scheme_version": SCHEME,
        "dataset": "2024Q1",
        "status": "Q1_DEVELOPMENT_SIMULATION_FIXED_RULE",
        "population": attr["population"],
        "entry_rule": "PS_AB only; H_AB and H_RATIO are classification variables, H_POS removed",
        "formation_rule": "market-center seed; one-rider marginal q-density growth; select state immediately before largest adjacent geometric-mean q block-support drop",
        "price_rule": "ProfitMass_1x and ProfitMass_2x are diagnostics only; no price-based ticket deletion or entry cutoff",
        "fixed_place_counts": False,
        "fixed_point_count": False,
        "price_cut": False,
        "nested_required": False,
        "attrition": dict(attr),
        "entry_fail_reasons": dict(fail),
        "H_state_counts": dict(hstates),
        "summary": base,
        "payout_bands": payout_bands,
        "size_pattern_distribution": dict(sorted(sizes.items(), key=lambda kv: (-kv[1], kv[0]))),
        "performance_by_H_state": by_h_state,
        "profit_mass_1x_quartiles": quantile_groups(out, "profit_mass_1x"),
        "profit_mass_2x_quartiles": quantile_groups(out, "profit_mass_2x"),
        "notes": [
            "Q1 is development data; no Q1 outcome-fitted cutoff is used in v8.8-F09.",
            "ProfitMass quartiles are descriptive analysis only and are not entry thresholds.",
        ],
    }

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "v8_8_f09_2024q1_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if out:
        with (OUT / "v8_8_f09_2024q1_races.csv").open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
            w.writeheader()
            w.writerows(out)

    print("V8_8_F09_Q1_RESULT_BEGIN")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print("V8_8_F09_Q1_RESULT_END")


if __name__ == "__main__":
    main()
