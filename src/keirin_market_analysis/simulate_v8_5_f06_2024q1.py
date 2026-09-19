from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from statistics import median

from simulate_v8_1_f02_2024q1 import load, pi, pl, streak, STAKE
from v8_4_f05_incremental_growth import build_v8_4_f05
from v8_5_f06_payout_potential_gate import build_v8_5_f06, payout_potential_metrics

SCHEME = "v8.5-F06"
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts/v8_5_f06_2024q1"


def summarize(rows):
    races = len(rows)
    hits = sum(r["hit"] for r in rows)
    tickets = sum(r["ticket_count"] for r in rows)
    stake = sum(r["stake_yen"] for r in rows)
    payout = sum(r["payout_yen"] for r in rows)
    losing_hits = sum(1 for r in rows if r["hit"] and r["payout_yen"] < r["stake_yen"])
    good_hits = sum(1 for r in rows if r["hit"] and r["payout_yen"] >= r["stake_yen"])
    return {
        "races": races,
        "hits": hits,
        "misses": races - hits,
        "hit_rate_pct": 100 * hits / races if races else None,
        "tickets": tickets,
        "avg_tickets_per_race": tickets / races if races else None,
        "stake_yen": stake,
        "payout_yen": payout,
        "profit_yen": payout - stake,
        "roi_pct": 100 * payout / stake if stake else None,
        "max_losing_streak": streak(rows) if rows else None,
        "losing_hit_races": losing_hits,
        "losing_hit_share_of_hits_pct": 100 * losing_hits / hits if hits else None,
        "breakeven_or_better_hit_races": good_hits,
        "median_ppm": median(r["ppm"] for r in rows) if rows else None,
        "median_phs": median(r["phs"] for r in rows) if rows else None,
    }


def main():
    races, trio, tf, pay = load()
    base_rows, kept_rows, removed_rows = [], [], []
    skip_reasons = Counter()
    attr = Counter()

    for rid, r in sorted(races.items(), key=lambda kv: (kv[1].get("race_date", ""), kv[0])):
        attr["races_csv"] += 1
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

        base = build_v8_4_f05(trio[rid], tf[rid], r.get("predicted_line_formation") or "")
        if not base.get("buy"):
            continue
        attr["v8_4_entry_pass"] += 1

        tickets = tuple(base["tickets"])
        wins = [t for t in tickets if t in pay[rid]]
        payout = sum(pay[rid][t] for t in wins)
        stake = len(tickets) * STAKE
        pm = payout_potential_metrics(tickets, tf[rid])
        row = {
            "race_id": rid,
            "race_date": r.get("race_date"),
            "track": r.get("track"),
            "race_no": pi(r.get("race_no")),
            "race_type": r.get("race_type"),
            "formation": base["formation"],
            "ticket_count": len(tickets),
            "stake_yen": stake,
            "hit": int(bool(wins)),
            "payout_yen": payout,
            "winning_ticket": ";".join("-".join(map(str, t)) for t in wins),
            "ppm": pm["payout_potential_multiple"],
            "phs": pm["profitable_hit_support_share"],
            "conditional_payout_proxy_yen": pm["conditional_payout_proxy_yen"],
        }
        base_rows.append(row)

        new = build_v8_5_f06(trio[rid], tf[rid], r.get("predicted_line_formation") or "")
        if new.get("buy"):
            attr["v8_5_buy"] += 1
            kept_rows.append(row)
        else:
            reason = str(new.get("reason"))
            skip_reasons[reason] += 1
            removed_rows.append({**row, "gate_reason": reason})

    base_summary = summarize(base_rows)
    kept_summary = summarize(kept_rows)
    removed_summary = summarize(removed_rows)
    base_losing_hits = [r for r in base_rows if r["hit"] and r["payout_yen"] < r["stake_yen"]]
    removed_losing_hits = [r for r in removed_rows if r["hit"] and r["payout_yen"] < r["stake_yen"]]
    removed_good_hits = [r for r in removed_rows if r["hit"] and r["payout_yen"] >= r["stake_yen"]]

    result = {
        "scheme_version": SCHEME,
        "dataset": "2024Q1",
        "status": "Q1_DEVELOPMENT_SIMULATION_FIXED_RULE",
        "entry_and_formation": "v6.1 pre-formation gate + unchanged v8.4-F05 formation",
        "new_race_gate": "PPM>1.0 AND PHS>0.5; thresholds intrinsic, not fitted to Q1 outcomes",
        "price_cut": False,
        "result_used_in_gate": False,
        "attrition": dict(attr),
        "v8_5_skip_reasons": dict(skip_reasons),
        "v8_4_baseline": base_summary,
        "v8_5_kept": kept_summary,
        "removed_by_payout_gate": removed_summary,
        "gate_effect": {
            "races_removed": len(removed_rows),
            "race_reduction_pct": 100 * len(removed_rows) / len(base_rows) if base_rows else None,
            "losing_hits_in_v8_4": len(base_losing_hits),
            "losing_hits_removed": len(removed_losing_hits),
            "losing_hit_removal_rate_pct": 100 * len(removed_losing_hits) / len(base_losing_hits) if base_losing_hits else None,
            "breakeven_or_better_hits_removed": len(removed_good_hits),
            "profit_change_yen_vs_v8_4": kept_summary["profit_yen"] - base_summary["profit_yen"],
            "roi_change_pp_vs_v8_4": kept_summary["roi_pct"] - base_summary["roi_pct"] if kept_summary["roi_pct"] is not None else None,
            "hit_rate_change_pp_vs_v8_4": kept_summary["hit_rate_pct"] - base_summary["hit_rate_pct"] if kept_summary["hit_rate_pct"] is not None else None,
        },
    }

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print("V8_5_F06_Q1_RESULT_BEGIN")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print("V8_5_F06_Q1_RESULT_END")


if __name__ == "__main__":
    main()
