from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from simulate_v8_1_f02_2024q1 import STAKE, load, pi, pl, streak
from v8_9_f10_compact_entry import build_v8_9_f10

SCHEME = "v8.9-F10"
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts/v8_9_f10_2024q1"


def _ticket_text(t):
    return "-".join(str(x) for x in t) if t else ""


def _sample_rows(rows):
    """Deterministic audit sample spanning compact, typical, wide and payout-rich buys."""
    if not rows:
        return []
    chosen = []
    seen = set()

    def add(row, label):
        if row["race_id"] in seen:
            return
        seen.add(row["race_id"])
        chosen.append({"sample_type": label, **row})

    by_points = sorted(rows, key=lambda r: (r["ticket_count"], r["race_date"], r["race_id"]))
    n = len(by_points)
    for idx, label in [
        (0, "MIN_POINTS"),
        (n // 4, "LOW_POINTS"),
        (n // 2, "MEDIAN_POINTS"),
        ((3 * n) // 4, "HIGH_POINTS"),
        (n - 1, "MAX_POINTS"),
    ]:
        add(by_points[idx], label)

    hits = sorted(
        (r for r in rows if r["hit"]),
        key=lambda r: (-r["payout_yen"], r["race_date"], r["race_id"]),
    )
    for i, row in enumerate(hits[:3], start=1):
        add(row, f"TOP_PAYOUT_HIT_{i}")

    misses = sorted(
        (r for r in rows if not r["hit"]),
        key=lambda r: (abs(r["ticket_count"] - 11), r["race_date"], r["race_id"]),
    )
    for i, row in enumerate(misses[:2], start=1):
        add(row, f"TYPICAL_MISS_{i}")

    return chosen


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
        actual_tickets = tuple(sorted(pay[rid]))
        actual_ticket = actual_tickets[0] if actual_tickets else None
        actual_payout = pay[rid].get(actual_ticket, 0) if actual_ticket else 0
        size = f"{len(d['first'])}-{len(d['second'])}-{len(d['third'])}"
        sizes[size] += 1
        gate = d.get("entry_gate", {})
        h_ab[str(int(bool(gate.get("H_AB"))))] += 1
        h1_top = float(gate.get("H1_top") or 0.0)
        h1_second = float(gate.get("H1_second") or 0.0)
        out.append({
            "race_id": rid,
            "race_date": r.get("race_date"),
            "track": r.get("track"),
            "race_no": pi(r.get("race_no")),
            "race_type": r.get("race_type"),
            "predicted_line_formation": r.get("predicted_line_formation"),
            "top2_H": list(gate.get("top2_H") or ()),
            "H1_ratio": h1_top / h1_second if h1_second > 0 else None,
            "H_AB": int(bool(gate.get("H_AB"))),
            "formation": d["formation"],
            "ticket_count": d["ticket_count"],
            "size_pattern": size,
            "selected_growth_step": d.get("selected_growth_step"),
            "q_mass": d.get("q_mass"),
            "profit_mass_1x": d.get("profit_mass_1x"),
            "profit_mass_2x": d.get("profit_mass_2x"),
            "hit": int(bool(wins)),
            "winning_ticket_in_bets": _ticket_text(wins[0]) if wins else "",
            "actual_result_ticket": _ticket_text(actual_ticket),
            "actual_result_payout_yen": actual_payout,
            "payout_yen": payout,
            "stake_yen": d["ticket_count"] * STAKE,
            "race_profit_yen": payout - d["ticket_count"] * STAKE,
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

    samples = _sample_rows(out)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "v8_9_f10_2024q1_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (OUT / "v8_9_f10_2024q1_samples.json").write_text(
        json.dumps(samples, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("V8_9_F10_Q1_RESULT_BEGIN")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print("V8_9_F10_Q1_RESULT_END")
    print("V8_9_F10_Q1_SAMPLES_BEGIN")
    print(json.dumps(samples, ensure_ascii=False, indent=2))
    print("V8_9_F10_Q1_SAMPLES_END")


if __name__ == "__main__":
    main()
