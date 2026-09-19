#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/keirin_shogi/high_payout_hypothesis_study"
OUT.mkdir(parents=True, exist_ok=True)

BLOCKS = {
    "h1_oos": {
        "log": ROOT / "results/keirin_shogi/v37_shrunk_board_third/race_log.json",
        "payouts": ROOT / "data/2026_h1/s_class_f1_all/payouts.csv",
        "entries": ROOT / "data/2026_h1/s_class_f1_all/entries.csv",
    },
    "true_future": {
        "log": ROOT / "results/keirin_shogi/v37_true_future_block1/race_log.json",
        "payouts": ROOT / "data/2026_future_block1/s_class_f1_20260701_20260830/payouts.csv",
        "entries": ROOT / "data/2026_future_block1/s_class_f1_20260701_20260830/entries.csv",
    },
}

def read_csv(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))

def qtile(values, q):
    xs = sorted(values)
    if not xs:
        return None
    p = (len(xs) - 1) * q
    lo, hi = math.floor(p), math.ceil(p)
    if lo == hi:
        return xs[lo]
    return xs[lo] + (xs[hi] - xs[lo]) * (p - lo)

def stats(rows):
    p = [r["payout"] for r in rows]
    return {
        "n": len(rows),
        "mean": round(mean(p)) if p else None,
        "median": round(median(p)) if p else None,
        "q75": round(qtile(p, 0.75)) if p else None,
        "q90": round(qtile(p, 0.90)) if p else None,
        "ge5000": sum(x >= 5000 for x in p),
        "ge10000": sum(x >= 10000 for x in p),
        "ge20000": sum(x >= 20000 for x in p),
        "rate_ge10000": (sum(x >= 10000 for x in p) / len(p)) if p else None,
    }

def entropy(values):
    total = sum(values)
    if total <= 0:
        return 0.0
    out = 0.0
    for value in values:
        p = value / total
        if p > 0:
            out -= p * math.log(p)
    return out

def load_block(cfg):
    logs = json.loads(cfg["log"].read_text(encoding="utf-8"))
    payouts = read_csv(cfg["payouts"])
    entries = read_csv(cfg["entries"])

    payout_map = {
        str(r["race_id"]): int(r["payout_yen"])
        for r in payouts
        if r.get("ticket_type") == "3連単"
        and r.get("status") == "paid"
        and str(r.get("payout_yen", "")).isdigit()
    }

    entry_map = defaultdict(dict)
    for r in entries:
        entry_map[str(r["race_id"])][int(float(r["car_no"]))] = {
            "line_id": str(r.get("line_id", "")),
            "line_position": int(float(r.get("line_position") or 0)),
        }

    out = []
    for log in logs:
        rid = str(log["race_id"])
        if rid not in payout_map:
            continue
        first = int(log["actual_first"])
        second = int(log["actual_second"])
        third = int(log["actual_third"])
        fp = list(map(int, log.get("first_candidates", [])))
        sp = list(map(int, log.get("second_candidates", [])))
        tp = list(map(int, log.get("third_candidates", [])))
        first_in = first in fp
        second_in = second in sp
        third_in = third in tp
        hit_count = sum((first_in, second_in, third_in))

        em = entry_map.get(rid, {})
        ef, es = em.get(first), em.get(second)
        same_line = bool(ef and es and ef["line_id"] and ef["line_id"] == es["line_id"])
        reversal = bool(same_line and ef["line_position"] > es["line_position"])
        forward = bool(same_line and ef["line_position"] < es["line_position"])

        pairs = [
            {
                "a": int(x["a"]),
                "b": int(x["b"]),
                "p": float(x.get("probability", x.get("prob", 0.0))),
            }
            for x in log.get("top_pairs", [])
        ]
        a, b = sorted((first, second))
        pair_rank = next(
            (i + 1 for i, x in enumerate(pairs) if sorted((x["a"], x["b"])) == [a, b]),
            6,
        )

        third_ranking = log.get("third_ranking", [])
        third_rank = next(
            (i + 1 for i, x in enumerate(third_ranking) if int(x["no"]) == third),
            None,
        )
        membership = log.get("top2_membership", log.get("second_membership", []))
        second_rank = next(
            (i + 1 for i, x in enumerate(membership) if int(x["no"]) == second),
            None,
        )

        cells = len(fp) + len(sp) + len(tp)
        union_size = len(set(fp + sp + tp))
        missing_tier = None
        if hit_count == 2:
            missing_tier = "first" if not first_in else ("second" if not second_in else "third")

        missing_connected = False
        if hit_count == 2:
            missing_no = first if missing_tier == "first" else second if missing_tier == "second" else third
            me = em.get(missing_no)
            if me:
                for no in (first, second, third):
                    if no == missing_no:
                        continue
                    oe = em.get(no)
                    if oe and oe["line_id"] == me["line_id"]:
                        missing_connected = True
                        break

        top_pair = pairs[0]["p"] if pairs else 0.0
        pair2 = pairs[1]["p"] if len(pairs) > 1 else 0.0
        third_probs = [float(x["probability"]) for x in third_ranking]

        out.append({
            **log,
            "payout": payout_map[rid],
            "first_in": first_in,
            "second_in": second_in,
            "third_in": third_in,
            "hit_count": hit_count,
            "same_line": same_line,
            "reversal": reversal,
            "forward": forward,
            "actual_pair_rank": pair_rank,
            "third_rank": third_rank,
            "second_rank": second_rank,
            "cells": cells,
            "union_size": union_size,
            "overlap": cells - union_size,
            "missing_tier": missing_tier,
            "missing_connected": missing_connected,
            "top_pair_prob": top_pair,
            "top_pair_gap": top_pair - pair2,
            "top3_pair_mass": sum(x["p"] for x in pairs[:3]),
            "third_top1_prob": third_probs[0] if third_probs else 0.0,
            "third_entropy": entropy(third_probs),
        })
    return out

def quartile_compare(rows, key):
    vals = [float(r[key]) for r in rows]
    q1, q3 = qtile(vals, 0.25), qtile(vals, 0.75)
    return {
        "q25": q1,
        "q75": q3,
        "low": stats([r for r in rows if r[key] <= q1]),
        "high": stats([r for r in rows if r[key] >= q3]),
    }

def analyze(rows):
    hit_groups = {str(n): stats([r for r in rows if r["hit_count"] == n]) for n in range(4)}
    line = {
        "same_line_reversal": stats([r for r in rows if r["reversal"]]),
        "same_line_forward": stats([r for r in rows if r["forward"]]),
        "cross_line": stats([r for r in rows if not r["same_line"]]),
    }
    pair_rank = {
        "rank1": stats([r for r in rows if r["actual_pair_rank"] == 1]),
        "rank2to5": stats([r for r in rows if 2 <= r["actual_pair_rank"] <= 5]),
        "outside_top5": stats([r for r in rows if r["actual_pair_rank"] >= 6]),
    }
    third_rank = {
        "rank1to3": stats([r for r in rows if r["third_rank"] and r["third_rank"] <= 3]),
        "rank4to7": stats([r for r in rows if r["third_rank"] and r["third_rank"] >= 4]),
    }

    two = [r for r in rows if r["hit_count"] == 2]
    high_two = [r for r in two if r["payout"] >= 10000]
    missing = {}
    for tier in ("first", "second", "third"):
        missing[tier] = {
            "all": stats([r for r in two if r["missing_tier"] == tier]),
            "high_ge10000": stats([r for r in high_two if r["missing_tier"] == tier]),
        }

    return {
        "overall": stats(rows),
        "row_hit_count": hit_groups,
        "line_relation": line,
        "actual_top2_pair_rank": pair_rank,
        "actual_third_rank": third_rank,
        "pre_race_uncertainty": {
            "top_pair_prob": quartile_compare(rows, "top_pair_prob"),
            "top_pair_gap": quartile_compare(rows, "top_pair_gap"),
            "top3_pair_mass": quartile_compare(rows, "top3_pair_mass"),
            "third_top1_prob": quartile_compare(rows, "third_top1_prob"),
            "third_entropy": quartile_compare(rows, "third_entropy"),
        },
        "two_of_three_capture": {
            "missing_tier": missing,
            "high_ge10000_count": len(high_two),
            "second_missing_top4_membership": sum(
                r["missing_tier"] == "second" and r["second_rank"] and r["second_rank"] <= 4
                for r in high_two
            ),
            "third_missing_top4": sum(
                r["missing_tier"] == "third" and r["third_rank"] and r["third_rank"] <= 4
                for r in high_two
            ),
            "missing_line_connected": sum(r["missing_connected"] for r in high_two),
        },
        "top_payouts": [
            {
                "race_id": r["race_id"],
                "race_date": r["race_date"],
                "payout": r["payout"],
                "actual": [r["actual_first"], r["actual_second"], r["actual_third"]],
                "hit_count": r["hit_count"],
                "actual_pair_rank": r["actual_pair_rank"],
                "third_rank": r["third_rank"],
                "line_relation": "reversal" if r["reversal"] else ("forward" if r["forward"] else "cross_line"),
            }
            for r in sorted(rows, key=lambda x: -x["payout"])[:20]
        ],
    }

def main():
    blocks = {name: load_block(cfg) for name, cfg in BLOCKS.items()}
    report = {
        "study": "high_payout_hypothesis_study",
        "purpose": "Test why a hit-rate-oriented seven-car v37 model sometimes reaches high payouts without using target-race odds at prediction time.",
        "guards": {
            "prediction_time_odds_used": False,
            "prediction_time_popularity_used": False,
            "payout_used_only_as_post_race_reward": True,
            "h1_is_oos_log": True,
            "true_future_is_frozen_future_validation": True,
        },
        "counts": {k: len(v) for k, v in blocks.items()},
        "analysis": {k: analyze(v) for k, v in blocks.items()},
    }
    (OUT / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
