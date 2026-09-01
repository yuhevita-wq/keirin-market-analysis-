from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from simulate_v8_1_f02_2024q1 import STAKE, load, pi, pl, streak
from v8_9_f10_compact_entry import build_v8_9_f10

SCHEME = "v8.9-F10"
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts/v8_9_f10_2024q1"


def main():
    races, trio, tf, pay = load()
    out = []
    fail = Counter()
    sizes = Counter()
    h_ab = Counter()
    attr = Counter()

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

        d = build_v8_9_f10(trio[rid], tf[rid], r.get("predicted_line_formation") or "")
        if not d.get("buy"):
            fail[str(d.get("reason"))] += 1
            continue

        ts = tuple(d["tickets"])
        wins = [t for t in ts if t in pay[rid]]
        payout = sum(pay[rid][t] for t in wins)
        size = f"{len(d['first'])}-{len(d['second'])}-{len(d['third'])}"
        sizes[size] += 1
        h_ab[str(int(bool(d.get("entry_gate", {}).get("H_AB"))))] += 1
        out.append({
            "race_id": rid,
            "race_date": r.get("race_date"),
            "formation": d["formation"],
            "ticket_count": d["ticket_count"],
            "size_pattern": size,
            "selected_growth_step": d.get("selected_growth_step"),
            "q_mass": d.get("q_mass"),
            "H_AB": int(bool(d.get("entry_gate", {}).get("H_AB"))),
            "hit": int(bool(wins)),
            "payout_yen": payout,
        })

    races_n = len(out)
    hits = sum(r["hit"] for r in out)
    tickets = sum(r["ticket_count"] for r in out)
    stake_yen = tickets * STAKE
    payout_yen = sum(r["payout_yen"] for r in out)
    result = {
        "scheme_version": SCHEME,
        "dataset": "2024Q1",
        "status": "Q1_DEVELOPMENT_SIMULATION_FIXED_RULE",
        "population": attr["population"],
        "entry_rule": "PS_AB + H1_top >= 2 * H1_second; H_AB diagnostic only",
        "bet_races": races_n,
        "hits": hits,
        "hit_rate_pct": 100 * hits / races_n if races_n else None,
        "tickets": tickets,
        "avg_tickets": tickets / races_n if races_n else None,
        "min_tickets": min((r["ticket_count"] for r in out), default=None),
        "max_tickets": max((r["ticket_count"] for r in out), default=None),
        "stake_yen": stake_yen,
        "payout_yen": payout_yen,
        "profit_yen": payout_yen - stake_yen,
        "roi_pct": 100 * payout_yen / stake_yen if stake_yen else None,
        "max_losing_streak": streak(out),
        "entry_fail_reasons": dict(fail),
        "H_AB_counts": dict(h_ab),
        "size_pattern_distribution": dict(sorted(sizes.items(), key=lambda kv: (-kv[1], kv[0]))),
        "acceptance": "PASS_Q1_PROFIT" if payout_yen > stake_yen else "REJECT_Q1_NOT_PROFITABLE",
        "notes": [
            "2024Q1 is development data, not out-of-sample validation.",
            "No result or payout is used by the entry or formation construction.",
            "No fixed place counts, fixed ticket count, nested requirement, or price cut is used.",
        ],
    }

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "v8_9_f10_2024q1_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("V8_9_F10_Q1_RESULT_BEGIN")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print("V8_9_F10_Q1_RESULT_END")


if __name__ == "__main__":
    main()
