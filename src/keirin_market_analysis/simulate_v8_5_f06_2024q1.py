from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from simulate_v8_1_f02_2024q1 import load, pi, pl, streak, STAKE
from v8_4_f05_incremental_growth import build_v8_4_f05
from v8_5_f06_payout_potential_gate import build_v8_5_f06

SCHEME = "v8.5-F06"
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts/v8_5_f06_2024q1"


def _summary(rows):
    b = len(rows)
    h = sum(r["hit"] for r in rows)
    n = sum(r["ticket_count"] for r in rows)
    payout = sum(r["payout_yen"] for r in rows)
    stake = n * STAKE
    return {
        "bet_races": b,
        "hit_races": h,
        "miss_races": b - h,
        "hit_rate_pct": 100 * h / b if b else None,
        "tickets": n,
        "avg_tickets_per_race": n / b if b else None,
        "stake_yen": stake,
        "payout_yen": payout,
        "profit_yen": payout - stake,
        "roi_pct": 100 * payout / stake if stake else None,
        "max_losing_streak": streak(rows) if rows else 0,
    }


def main():
    races, trio, tf, pay = load()
    base_rows = []
    selected_rows = []
    removed_rows = []
    fail = Counter()
    attr = defaultdict(int)

    for rid, r in sorted(races.items(), key=lambda kv: (kv[1].get("race_date", ""), kv[0])):
        if r.get("meeting_grade") != "F1" or not (r.get("race_type") or "").startswith("Ｓ級"):
            continue
        if pi(r.get("entry_count")) != 7:
            continue
        if len(trio.get(rid, {})) != 35 or len(tf.get(rid, {})) != 210:
            continue
        cars = sorted({v for c in trio[rid] for v in c})
        lines = pl(r.get("predicted_line_formation"))
        if len(cars) != 7 or lines is None or set(v for line in lines for v in line) != set(cars):
            continue
        if rid not in pay or not pay[rid]:
            continue
        attr["population"] += 1

        base = build_v8_4_f05(trio[rid], tf[rid], r.get("predicted_line_formation") or "")
        if not base.get("buy"):
            continue
        attr["v8_4_entry_pass"] += 1

        tickets = tuple(base["tickets"])
        wins = [t for t in tickets if t in pay[rid]]
        payout = sum(pay[rid][t] for t in wins)
        common = {
            "race_id": rid,
            "race_date": r.get("race_date"),
            "track": r.get("track"),
            "race_no": pi(r.get("race_no")),
            "race_type": r.get("race_type"),
            "formation": base["formation"],
            "ticket_count": base["ticket_count"],
            "hit": int(bool(wins)),
            "payout_yen": payout,
            "winning_ticket": ";".join("-".join(map(str, t)) for t in wins),
        }
        base_rows.append(common)

        d = build_v8_5_f06(trio[rid], tf[rid], r.get("predicted_line_formation") or "")
        gate = d.get("payout_gate") or {}
        row = {
            **common,
            "payout_potential_multiple": gate.get("payout_potential_multiple"),
            "payout_potential_yen": gate.get("payout_potential_yen"),
            "profitable_hit_support_share": gate.get("profitable_hit_support_share"),
            "conditional_payout_proxy_yen": gate.get("conditional_payout_proxy_yen"),
            "gate_reason": d.get("reason"),
        }
        if d.get("buy"):
            selected_rows.append(row)
        else:
            removed_rows.append(row)
            fail[str(d.get("reason"))] += 1

    result = {
        "scheme_version": SCHEME,
        "dataset": "2024Q1",
        "status": "Q1_DEVELOPMENT_SIMULATION",
        "formation_rule": "unchanged v8.4-F05 incremental-growth formation",
        "entry_change": "add race-level payout-potential gate after v8.4 formation",
        "payout_gate_rule": "market-weighted geometric hit return multiple > 1.0 AND profitable-hit support share > 0.5",
        "threshold_source": "intrinsic break-even and majority; not fitted on Q1 outcomes",
        "price_cut": False,
        "attrition": dict(attr),
        "payout_gate_fail_reasons": dict(fail),
        "v8_4_base": _summary(base_rows),
        "v8_5_f06": _summary(selected_rows),
        "removed_by_payout_gate": _summary(removed_rows),
    }

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    if selected_rows:
        with (OUT / "selected_races.csv").open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(selected_rows[0].keys()))
            w.writeheader(); w.writerows(selected_rows)
    if removed_rows:
        with (OUT / "removed_races.csv").open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(removed_rows[0].keys()))
            w.writeheader(); w.writerows(removed_rows)

    print("V8_5_F06_Q1_RESULT_BEGIN")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print("V8_5_F06_Q1_RESULT_END")


if __name__ == "__main__":
    main()
