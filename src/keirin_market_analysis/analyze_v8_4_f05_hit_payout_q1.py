from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median

from simulate_v8_1_f02_2024q1 import load, pi, pl, STAKE
from v8_4_f05_incremental_growth import build_v8_4_f05

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts/v8_4_f05_2024q1_payout_analysis"


def qtile(xs, p):
    ys = sorted(xs)
    if not ys:
        return None
    if len(ys) == 1:
        return ys[0]
    pos = (len(ys) - 1) * p
    lo = int(pos)
    hi = min(lo + 1, len(ys) - 1)
    w = pos - lo
    return ys[lo] * (1 - w) + ys[hi] * w


def summarize(rows):
    b = len(rows)
    h = sum(r["hit"] for r in rows)
    stake = sum(r["stake_yen"] for r in rows)
    payout = sum(r["payout_yen"] for r in rows)
    return {
        "races": b,
        "hits": h,
        "hit_rate_pct": 100 * h / b if b else None,
        "stake_yen": stake,
        "payout_yen": payout,
        "profit_yen": payout - stake,
        "roi_pct": 100 * payout / stake if stake else None,
        "avg_ticket_count": sum(r["ticket_count"] for r in rows) / b if b else None,
        "avg_price_proxy_yen": sum(r["conditional_payout_proxy_yen"] for r in rows) / b if b else None,
        "median_price_proxy_yen": median(r["conditional_payout_proxy_yen"] for r in rows) if b else None,
    }


def main():
    races, trio, tf, pay = load()
    rows = []

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

        d = build_v8_4_f05(trio[rid], tf[rid], r.get("predicted_line_formation") or "")
        if not d.get("buy"):
            continue

        tickets = tuple(d["tickets"])
        odds = tf[rid]
        total_inv = sum(1.0 / float(o) for o in odds.values() if float(o) > 0)
        selected_odds = [float(odds[t]) for t in tickets]
        n = len(tickets)
        q_mass = float(d["q_mass"])

        # Conditional expected payout if the winner lies inside the selected formation,
        # under the normalized 3-rentan market distribution. Because q(t) ∝ 1/odds(t),
        # E[100*odds | selected hit] = 100*n/(total_inv*q_mass).
        cond_proxy = (100.0 * n / (total_inv * q_mass)) if total_inv > 0 and q_mass > 0 else 0.0

        wins = [t for t in tickets if t in pay[rid]]
        payout = sum(pay[rid][t] for t in wins)
        stake = n * STAKE

        so = sorted(selected_odds)
        med_odds = median(so)
        q_density = q_mass / n if n else 0.0

        rows.append({
            "race_id": rid,
            "race_date": r.get("race_date"),
            "track": r.get("track"),
            "race_no": pi(r.get("race_no")),
            "race_type": r.get("race_type"),
            "formation": d["formation"],
            "ticket_count": n,
            "stake_yen": stake,
            "q_mass": q_mass,
            "q_density": q_density,
            "min_selected_odds": min(so),
            "median_selected_odds": med_odds,
            "max_selected_odds": max(so),
            "conditional_payout_proxy_yen": cond_proxy,
            "hit": int(bool(wins)),
            "payout_yen": payout,
            "race_return_multiple": payout / stake if stake else 0.0,
            "winning_ticket": ";".join("-".join(map(str, t)) for t in wins),
        })

    hits = [r for r in rows if r["hit"]]
    misses = [r for r in rows if not r["hit"]]

    # Actual hit payout distribution.
    payout_vals = [r["payout_yen"] for r in hits]
    payout_distribution = {
        "hit_count": len(hits),
        "min_yen": min(payout_vals) if payout_vals else None,
        "q25_yen": qtile(payout_vals, 0.25),
        "median_yen": qtile(payout_vals, 0.50),
        "q75_yen": qtile(payout_vals, 0.75),
        "max_yen": max(payout_vals) if payout_vals else None,
        "mean_yen": sum(payout_vals) / len(payout_vals) if payout_vals else None,
    }

    abs_bands = [
        ("<2000", lambda x: x < 2000),
        ("2000-4999", lambda x: 2000 <= x < 5000),
        ("5000-9999", lambda x: 5000 <= x < 10000),
        ("10000-29999", lambda x: 10000 <= x < 30000),
        (">=30000", lambda x: x >= 30000),
    ]
    hit_payout_bands = {}
    for name, fn in abs_bands:
        rs = [r for r in hits if fn(r["payout_yen"])]
        hit_payout_bands[name] = {
            "hits": len(rs),
            "share_of_hits_pct": 100 * len(rs) / len(hits) if hits else None,
            "payout_yen": sum(r["payout_yen"] for r in rs),
            "share_of_total_payout_pct": 100 * sum(r["payout_yen"] for r in rs) / sum(payout_vals) if payout_vals else None,
            "avg_ticket_count": sum(r["ticket_count"] for r in rs) / len(rs) if rs else None,
            "median_price_proxy_yen": median(r["conditional_payout_proxy_yen"] for r in rs) if rs else None,
            "median_q_density": median(r["q_density"] for r in rs) if rs else None,
            "median_selected_odds": median(r["median_selected_odds"] for r in rs) if rs else None,
        }

    rel_bands = [
        ("hit_but_race_loss_<1x", lambda x: x < 1.0),
        ("1x_to_<2x", lambda x: 1.0 <= x < 2.0),
        ("2x_to_<5x", lambda x: 2.0 <= x < 5.0),
        (">=5x", lambda x: x >= 5.0),
    ]
    hit_return_bands = {}
    for name, fn in rel_bands:
        rs = [r for r in hits if fn(r["race_return_multiple"])]
        hit_return_bands[name] = {
            "hits": len(rs),
            "share_of_hits_pct": 100 * len(rs) / len(hits) if hits else None,
            "payout_yen": sum(r["payout_yen"] for r in rs),
            "avg_ticket_count": sum(r["ticket_count"] for r in rs) / len(rs) if rs else None,
            "median_price_proxy_yen": median(r["conditional_payout_proxy_yen"] for r in rs) if rs else None,
            "median_q_density": median(r["q_density"] for r in rs) if rs else None,
        }

    # Pre-race conditional payout proxy quartiles across all 260 selected races.
    proxies = [r["conditional_payout_proxy_yen"] for r in rows]
    cuts = [qtile(proxies, p) for p in (0.25, 0.50, 0.75)]
    proxy_quartiles = {}
    labels = ["Q1_lowest_price", "Q2", "Q3", "Q4_highest_price"]
    for i, label in enumerate(labels):
        lo = float("-inf") if i == 0 else cuts[i - 1]
        hi = float("inf") if i == 3 else cuts[i]
        if i == 0:
            rs = [r for r in rows if r["conditional_payout_proxy_yen"] <= hi]
        elif i == 3:
            rs = [r for r in rows if r["conditional_payout_proxy_yen"] > lo]
        else:
            rs = [r for r in rows if lo < r["conditional_payout_proxy_yen"] <= hi]
        proxy_quartiles[label] = {"range_yen": [None if lo == float("-inf") else lo, None if hi == float("inf") else hi], **summarize(rs)}

    # Diagnostic comparison: high actual payout hits (top quartile) vs other hits vs misses.
    hit_q75 = qtile(payout_vals, 0.75) if payout_vals else None
    groups = {
        "high_payout_hits_top25pct": [r for r in hits if r["payout_yen"] >= hit_q75] if hit_q75 is not None else [],
        "other_hits": [r for r in hits if r["payout_yen"] < hit_q75] if hit_q75 is not None else [],
        "misses": misses,
    }
    group_features = {}
    for name, rs in groups.items():
        group_features[name] = {
            "races": len(rs),
            "median_ticket_count": median(r["ticket_count"] for r in rs) if rs else None,
            "median_q_mass": median(r["q_mass"] for r in rs) if rs else None,
            "median_q_density": median(r["q_density"] for r in rs) if rs else None,
            "median_selected_odds": median(r["median_selected_odds"] for r in rs) if rs else None,
            "median_price_proxy_yen": median(r["conditional_payout_proxy_yen"] for r in rs) if rs else None,
        }

    top_hits = sorted(hits, key=lambda r: (-r["payout_yen"], r["race_date"], r["race_id"]))[:15]

    result = {
        "scheme": "v8.4-F05",
        "dataset": "2024Q1",
        "status": "EXPLORATORY_PAYOUT_ANALYSIS_DO_NOT_FREEZE_THRESHOLD_FROM_Q1_ALONE",
        "all_selected_races": summarize(rows),
        "actual_hit_payout_distribution": payout_distribution,
        "actual_hit_payout_bands": hit_payout_bands,
        "actual_hit_return_multiple_bands": hit_return_bands,
        "pre_race_conditional_payout_proxy_definition": "100 * selected_ticket_count / (sum_all_210(1/odds) * selected_q_mass)",
        "pre_race_proxy_quartile_performance": proxy_quartiles,
        "diagnostic_group_feature_medians": group_features,
        "top_15_actual_payout_hits": top_hits,
    }

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print("V8_4_F05_PAYOUT_ANALYSIS_BEGIN")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print("V8_4_F05_PAYOUT_ANALYSIS_END")


if __name__ == "__main__":
    main()
