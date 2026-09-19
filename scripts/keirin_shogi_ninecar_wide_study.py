#!/usr/bin/env python3
from __future__ import annotations

"""Forward study: convert nine-car v3.2 boards / joint504 into wide tickets.

Purpose
-------
Test whether the nine-car board is more useful as a top-3 pair detector than
as an ordered trifecta formation source.

No odds/popularity are used to select tickets. Published wide payouts are used
only after the race to calculate realized return.

Forward protocol
----------------
2024 races: rider/pair/third models trained on races before 2024.
             strong-state overlay calibrated on 2023 OOS rows.
2025 races: models trained on races before 2025; overlay calibrated on 2024.
2026 H1:    models trained on races before 2026; overlay calibrated on 2025.

The production board policy is unchanged: greedy joint504 budget 7 plus the
v3.2 strong-rider state overlay and its participation filter.
"""

import csv
import importlib.util
import io
import json
import math
import re
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results/keirin_shogi/ninecar_wide_study"
OUT_JSON = OUT_DIR / "summary.json"
OUT_MD = OUT_DIR / "summary.md"
V32_PATH = ROOT / "scripts/keirin_shogi_ninecar_v32.py"
STAKE = 100


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


v32 = load_module("ninecar_v32_wide_study", V32_PATH)
v2 = v32.v2


def parse_pair(value: object):
    xs = [int(x) for x in re.findall(r"[1-9]", str(value))]
    if len(xs) != 2 or xs[0] == xs[1]:
        return None
    return tuple(sorted(xs))


def load_wide_payouts():
    by_race: dict[str, dict[tuple[int, int], int]] = defaultdict(dict)
    archive_stats = []
    for path in v2.archive_paths():
        paid = 0
        rows = 0
        has_payouts = False
        try:
            with zipfile.ZipFile(path) as z:
                try:
                    raw = z.read("payouts.csv")
                except KeyError:
                    archive_stats.append({
                        "archive": str(path.relative_to(ROOT)),
                        "has_payouts": False,
                        "rows": 0,
                        "paid_wide_rows": 0,
                    })
                    continue
                has_payouts = True
                text = raw.decode("utf-8-sig", errors="replace")
                for row in csv.DictReader(io.StringIO(text)):
                    rows += 1
                    if row.get("ticket_type") != "ワイド" or row.get("status") != "paid":
                        continue
                    pair = parse_pair(row.get("combination", ""))
                    payout = str(row.get("payout_yen", "")).replace(",", "").strip()
                    rid = str(row.get("race_id", ""))
                    if pair and rid and payout.isdigit():
                        by_race[rid][pair] = int(payout)
                        paid += 1
        except zipfile.BadZipFile:
            has_payouts = False
        archive_stats.append({
            "archive": str(path.relative_to(ROOT)),
            "has_payouts": has_payouts,
            "rows": rows,
            "paid_wide_rows": paid,
        })
    return by_race, archive_stats


def pair_key(a: int, b: int):
    return tuple(sorted((int(a), int(b))))


def union_pairs(board):
    cars = sorted(set(board[0]) | set(board[1]) | set(board[2]))
    return {pair_key(a, b) for i, a in enumerate(cars) for b in cars[i + 1 :]}


def row_pair(board, r1: int, r2: int):
    return {
        pair_key(a, b)
        for a in board[r1]
        for b in board[r2]
        if a != b
    }


def cross_pairs(board):
    return row_pair(board, 0, 1) | row_pair(board, 0, 2) | row_pair(board, 1, 2)


def adjacent_pairs(board):
    return row_pair(board, 0, 1) | row_pair(board, 1, 2)


def joint_pair_probabilities(joint):
    probs = defaultdict(float)
    for a, b, c, p in joint:
        probs[pair_key(a, b)] += float(p)
        probs[pair_key(a, c)] += float(p)
        probs[pair_key(b, c)] += float(p)
    # Every joint triple contributes exactly three unordered pairs.
    return dict(probs)


def make_tickets(strategy: str, board, pair_probs):
    if strategy == "board_union_all":
        return union_pairs(board)
    if strategy == "board_cross_all":
        return cross_pairs(board)
    if strategy == "board_adjacent":
        return adjacent_pairs(board)
    if strategy == "board_row12":
        return row_pair(board, 0, 1)
    if strategy == "board_row13":
        return row_pair(board, 0, 2)
    if strategy == "board_row23":
        return row_pair(board, 1, 2)

    m = re.fullmatch(r"(joint|cross_joint)_top(\d+)", strategy)
    if m:
        family, k_text = m.groups()
        k = int(k_text)
        candidates = set(pair_probs)
        if family == "cross_joint":
            candidates &= cross_pairs(board)
        ranked = sorted(candidates, key=lambda x: (-pair_probs.get(x, 0.0), x))
        return set(ranked[:k])

    m = re.fullmatch(r"(joint|cross_joint)_q(\d+)", strategy)
    if m:
        family, bp_text = m.groups()
        threshold = int(bp_text) / 1000.0
        candidates = set(pair_probs)
        if family == "cross_joint":
            candidates &= cross_pairs(board)
        return {p for p in candidates if pair_probs.get(p, 0.0) >= threshold}

    # "Cheap-looking cut" without odds: rank unordered wide pairs by the
    # model's pre-race probability that both riders finish in the top three.
    # Example rank2to5 deliberately removes the single most obvious pair.
    m = re.fullmatch(r"(joint|cross_joint)_rank(\d+)to(\d+)", strategy)
    if m:
        family, lo_text, hi_text = m.groups()
        lo, hi = int(lo_text), int(hi_text)
        candidates = set(pair_probs)
        if family == "cross_joint":
            candidates &= cross_pairs(board)
        ranked = sorted(candidates, key=lambda x: (-pair_probs.get(x, 0.0), x))
        return set(ranked[lo - 1:hi])

    raise ValueError(strategy)


STRATEGIES = [
    "board_union_all",
    "board_cross_all",
    "board_adjacent",
    "board_row12",
    "board_row13",
    "board_row23",
]
for prefix in ("joint", "cross_joint"):
    STRATEGIES += [f"{prefix}_top{k}" for k in (1, 2, 3, 4, 5, 6, 8)]
    STRATEGIES += [f"{prefix}_q{bp}" for bp in (40, 50, 60, 70, 80, 100, 120, 150)]
    STRATEGIES += [
        f"{prefix}_rank{lo}to{hi}"
        for lo, hi in ((2,4),(2,5),(2,6),(2,8),(3,5),(3,6),(3,8),(4,8))
    ]


def max_losing_streak(flags):
    best = cur = 0
    for hit in flags:
        if hit:
            cur = 0
        else:
            cur += 1
            best = max(best, cur)
    return best


def summarize(records):
    if not records:
        return {
            "eligible_races": 0,
            "bet_races": 0,
            "tickets": 0,
            "stake_yen": 0,
            "payout_yen": 0,
            "profit_yen": 0,
            "roi_pct": None,
            "race_hit_rate_pct": None,
            "avg_tickets_per_bet_race": None,
            "max_losing_streak": None,
        }
    bets = [r for r in records if r["ticket_count"] > 0]
    tickets = sum(r["ticket_count"] for r in bets)
    stake = tickets * STAKE
    payout = sum(r["payout_yen"] for r in bets)
    hits = [r["payout_yen"] > 0 for r in bets]
    return {
        "eligible_races": len(records),
        "bet_races": len(bets),
        "tickets": tickets,
        "stake_yen": stake,
        "payout_yen": payout,
        "profit_yen": payout - stake,
        "roi_pct": (100.0 * payout / stake) if stake else None,
        "race_hit_rate_pct": (100.0 * sum(hits) / len(hits)) if hits else None,
        "avg_tickets_per_bet_race": (tickets / len(bets)) if bets else None,
        "max_losing_streak": max_losing_streak(hits) if hits else None,
        "multi_hit_races": sum(r["hit_ticket_count"] >= 2 for r in bets),
        "triple_hit_races": sum(r["hit_ticket_count"] == 3 for r in bets),
        "max_race_payout_yen": max((r["payout_yen"] for r in bets), default=0),
        "top1_race_payout_share": (
            max((r["payout_yen"] for r in bets), default=0) / payout
            if payout else None
        ),
    }


def build_forward_scored(races, year):
    calibration_year = year - 1
    cal_rows = v32.build_oos_rows(calibration_year, races)
    if not cal_rows:
        raise RuntimeError(f"no calibration rows for {calibration_year}")
    state_model, state_names = v32.fit_state(cal_rows)
    cal_scored = v32.score_state(cal_rows, state_model, state_names)
    rules = v32.choose_rules(cal_scored)

    test_rows = v32.build_oos_rows(year, races)
    test_scored = v32.score_state(test_rows, state_model, state_names)
    return test_scored, rules, len(cal_rows)


def evaluate_year(year, scored, rules, payout_map):
    strategy_records = {s: [] for s in STRATEGIES}
    coverage = {
        "model_races": len(scored),
        "payout_covered_races": 0,
        "participant_races": 0,
        "participant_payout_covered_races": 0,
    }

    for row in scored:
        race = row["race"]
        paid = payout_map.get(race.race_id)
        if not paid:
            continue
        coverage["payout_covered_races"] += 1

        board, board_mass, participate, dominant, action = v32.apply_overlay(row, rules)
        if not participate:
            continue
        coverage["participant_races"] += 1
        coverage["participant_payout_covered_races"] += 1

        pair_probs = joint_pair_probabilities(row["joint"])
        actual_top3 = set(race.order)
        actual_wide_pairs = {
            pair_key(race.order[0], race.order[1]),
            pair_key(race.order[0], race.order[2]),
            pair_key(race.order[1], race.order[2]),
        }

        for strategy in STRATEGIES:
            tickets = make_tickets(strategy, board, pair_probs)
            hit_pairs = sorted(tickets & set(paid))
            payout = sum(paid[p] for p in hit_pairs)
            strategy_records[strategy].append({
                "race_id": race.race_id,
                "race_date": race.race_date,
                "grade": race.grade,
                "race_type": race.race_type,
                "board": [list(x) for x in board],
                "board_mass": float(board_mass),
                "overlay_action": action,
                "dominant": bool(dominant),
                "ticket_count": len(tickets),
                "hit_ticket_count": len(hit_pairs),
                "payout_yen": payout,
                "actual_top3": list(race.order),
                "actual_wide_pairs": [list(x) for x in sorted(actual_wide_pairs)],
                "selected_hit_pairs": [list(x) for x in hit_pairs],
                "selected_pairs": [list(x) for x in sorted(tickets)],
            })

    metrics = {s: summarize(rows) for s, rows in strategy_records.items()}
    by_grade = {}
    for g in ("G1", "G2", "G3"):
        by_grade[g] = {
            s: summarize([r for r in rows if r["grade"] == g])
            for s, rows in strategy_records.items()
        }
    return {
        "year": year,
        "coverage": coverage,
        "strategies": metrics,
        "by_grade": by_grade,
        "records": strategy_records,
    }



def payout_band_summary(values):
    if not values:
        return {"n": 0, "mean_payout": None, "median_payout": None}
    xs = sorted(values)
    n = len(xs)
    med = xs[n//2] if n % 2 else (xs[n//2-1] + xs[n//2]) / 2
    return {
        "n": n,
        "mean_payout": sum(xs) / n,
        "median_payout": med,
        "ge1000_rate": sum(x >= 1000 for x in xs) / n,
        "ge2000_rate": sum(x >= 2000 for x in xs) / n,
    }


def winning_pair_price_proxy(year, scored, rules, payout_map):
    # Post-race diagnostic only. Selection never sees payout.
    buckets = {
        "rank1to3": [],
        "rank4to8": [],
        "rank9to15": [],
        "rank16plus": [],
    }
    rows = []
    for row in scored:
        race = row["race"]
        paid = payout_map.get(race.race_id)
        if not paid:
            continue
        board, _mass, participate, _dominant, _action = v32.apply_overlay(row, rules)
        if not participate:
            continue
        pair_probs = joint_pair_probabilities(row["joint"])
        ranked = sorted(pair_probs, key=lambda x: (-pair_probs[x], x))
        rank_map = {pair: i + 1 for i, pair in enumerate(ranked)}
        for pair, payout in paid.items():
            rank = rank_map.get(pair)
            if rank is None:
                continue
            if rank <= 3:
                bucket = "rank1to3"
            elif rank <= 8:
                bucket = "rank4to8"
            elif rank <= 15:
                bucket = "rank9to15"
            else:
                bucket = "rank16plus"
            buckets[bucket].append(payout)
            rows.append({
                "race_id": race.race_id,
                "pair": list(pair),
                "pair_rank": rank,
                "pair_probability": pair_probs[pair],
                "payout_yen": payout,
                "bucket": bucket,
            })
    return {
        "year": year,
        "buckets": {k: payout_band_summary(v) for k, v in buckets.items()},
        "rows_n": len(rows),
    }


def development_selection(year_results):
    # Choose only on 2024. 2025 and 2026H1 are untouched validation.
    dev = year_results[0]
    candidates = []
    for strategy, m in dev["strategies"].items():
        if not m["bet_races"] or m["avg_tickets_per_bet_race"] is None:
            continue
        # Avoid winning by spraying the whole board: practical cap 8 pairs/race.
        if m["avg_tickets_per_bet_race"] > 8.0:
            continue
        # Require some recurrence rather than one lucky hit.
        if m["race_hit_rate_pct"] is None or m["race_hit_rate_pct"] < 20.0:
            continue
        candidates.append((m["roi_pct"], m["race_hit_rate_pct"], -m["avg_tickets_per_bet_race"], strategy))
    candidates.sort(reverse=True)
    chosen = [x[-1] for x in candidates[:5]]

    out = []
    for strategy in chosen:
        row = {"strategy": strategy}
        for y in year_results:
            m = y["strategies"][strategy]
            row[str(y["year"])] = {
                "roi_pct": m["roi_pct"],
                "profit_yen": m["profit_yen"],
                "hit_rate_pct": m["race_hit_rate_pct"],
                "avg_tickets": m["avg_tickets_per_bet_race"],
                "top1_race_payout_share": m["top1_race_payout_share"],
            }
        out.append(row)
    return out


def compact_year(year_result):
    return {
        "year": year_result["year"],
        "coverage": year_result["coverage"],
        "strategies": year_result["strategies"],
        "by_grade": year_result["by_grade"],
    }


def pooled_metrics(year_results):
    out = {}
    for strategy in STRATEGIES:
        rows = []
        for y in year_results:
            rows.extend(y["records"][strategy])
        out[strategy] = summarize(rows)
    return out


def stable_shortlist(year_results):
    shortlist = []
    for strategy in STRATEGIES:
        yearly = [y["strategies"][strategy] for y in year_results]
        rois = [m["roi_pct"] for m in yearly if m["roi_pct"] is not None]
        if len(rois) != len(year_results):
            continue
        pooled_rows = []
        for y in year_results:
            pooled_rows.extend(y["records"][strategy])
        pooled = summarize(pooled_rows)
        shortlist.append({
            "strategy": strategy,
            "roi_2024": rois[0],
            "roi_2025": rois[1],
            "roi_2026_h1": rois[2],
            "min_forward_roi": min(rois),
            "pooled_roi": pooled["roi_pct"],
            "pooled_profit": pooled["profit_yen"],
            "pooled_hit_rate": pooled["race_hit_rate_pct"],
            "pooled_avg_tickets": pooled["avg_tickets_per_bet_race"],
            "positive_all_periods": all(x > 100.0 for x in rois),
        })
    shortlist.sort(
        key=lambda x: (
            x["positive_all_periods"],
            x["min_forward_roi"],
            x["pooled_roi"] if x["pooled_roi"] is not None else -1,
        ),
        reverse=True,
    )
    return shortlist


def main():
    races = v2.load_races()
    payouts, archive_stats = load_wide_payouts()

    years = []
    calibration = {}
    for year in (2024, 2025, 2026):
        scored, rules, cal_n = build_forward_scored(races, year)
        result = evaluate_year(year, scored, rules, payouts)
        result["calibration_year"] = year - 1
        result["calibration_rows"] = cal_n
        result["rules"] = rules
        years.append(result)
        calibration[str(year)] = {
            "calibration_year": year - 1,
            "calibration_rows": cal_n,
            "rules": rules,
        }

    report = {
        "study": "ninecar_v32_wide_conversion_forward",
        "purpose": (
            "Test whether the 7-piece nine-car v3.2 board / joint504 distribution "
            "has more value when converted to wide tickets than when treated as an "
            "ordered trifecta formation."
        ),
        "protocol": {
            "stake_per_pair_yen": STAKE,
            "years": [2024, 2025, 2026],
            "2026_scope": "available 2026 H1 grade-race archives",
            "participation": "v3.2 collapse filter; skipped races bet 0 yen",
            "board": "v3.2 greedy joint504 budget 7 + strong-state overlay",
            "wide_pair_probability": (
                "sum joint504 probability mass of every ordered top3 triple "
                "containing both riders; order ignored"
            ),
            "payout_credit": (
                "published winning ワイド payout_yen; all selected winning wide "
                "pairs in a race are credited"
            ),
        },
        "guards": {
            "prediction_time_odds_used": False,
            "prediction_time_popularity_used": False,
            "payout_used_as_prediction_feature": False,
            "future_results_used_for_model_fit": False,
            "future_results_used_for_overlay_calibration": False,
            "selection_rules_predeclared_in_code": True,
        },
        "race_counts": {
            "loaded_grade_races": len(races),
            "wide_payout_races": len(payouts),
        },
        "archive_payout_coverage": archive_stats,
        "calibration": calibration,
        "years": [compact_year(y) for y in years],
        "pooled": pooled_metrics(years),
        "stable_shortlist": stable_shortlist(years),
        "development_2024_selection_then_validation": development_selection(years),
        "winning_pair_price_proxy": [
            winning_pair_price_proxy(y["year"], (
                build_forward_scored(races, y["year"])[0]
            ), y["rules"], payouts)
            for y in years
        ],
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    top = report["stable_shortlist"][:20]
    lines = [
        "# Nine-car v3.2 -> Wide forward study",
        "",
        "No prediction-time odds/popularity. 100 yen per selected wide pair.",
        "",
        "## Coverage",
    ]
    for y in years:
        c = y["coverage"]
        lines.append(
            f"- {y['year']}: model {c['model_races']}, payout-covered {c['payout_covered_races']}, "
            f"participants {c['participant_payout_covered_races']}"
        )
    lines += [
        "",
        "## Stable shortlist",
        "",
        "| strategy | 2024 ROI | 2025 ROI | 2026H1 ROI | pooled ROI | profit | hit rate | avg tickets | +all |",
        "|---|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for x in top:
        lines.append(
            f"| {x['strategy']} | {x['roi_2024']:.1f}% | {x['roi_2025']:.1f}% | "
            f"{x['roi_2026_h1']:.1f}% | {x['pooled_roi']:.1f}% | {x['pooled_profit']:,} | "
            f"{x['pooled_hit_rate']:.1f}% | {x['pooled_avg_tickets']:.2f} | "
            f"{'YES' if x['positive_all_periods'] else 'NO'} |"
        )
    lines += [
        "",
        "## Interpretation guard",
        "",
        "A strategy is not adopted merely because it leads this table. 2024/2025/2026H1 "
        "are shown separately to expose regime dependence and payout concentration. "
        "The development_2024_selection_then_validation section selects only on 2024 "
        "and treats 2025/2026H1 as untouched validation. Pair-probability rank is also "
        "checked as a no-odds proxy for cheap/expensive-looking wide combinations.",
        "",
    ]
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({
        "out": str(OUT_JSON),
        "loaded_races": len(races),
        "payout_races": len(payouts),
        "top10": top[:10],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
