from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def fnum(v: object) -> float | None:
    try:
        if v in (None, ""):
            return None
        return float(v)
    except (TypeError, ValueError):
        return None


def inum(v: object) -> int:
    try:
        return int(float(str(v)))
    except (TypeError, ValueError):
        return 0


def q(values: Iterable[float]) -> dict[str, float | int | None]:
    vals = sorted(float(v) for v in values)
    if not vals:
        return {"count": 0, "min": None, "median": None, "max": None}
    return {
        "count": len(vals),
        "min": vals[0],
        "median": statistics.median(vals),
        "max": vals[-1],
    }


def result_map(data_dir: Path) -> dict[str, dict[int, int]]:
    out: dict[str, dict[int, int]] = defaultdict(dict)
    for r in read_csv(data_dir / "results.csv"):
        pos = inum(r.get("finish_position"))
        car = inum(r.get("car_no"))
        if pos > 0 and car > 0:
            out[r["race_id"]][car] = pos
    return out


def race_financial(rows: list[dict[str, str]]) -> dict[str, object]:
    bought = [r for r in rows if inum(r.get("purchased")) == 1]
    hits = [r for r in bought if inum(r.get("hit")) == 1]
    stake = sum(inum(r.get("stake_yen")) for r in bought)
    payout = sum(inum(r.get("payout_yen")) for r in bought)
    hit_payouts = [inum(r.get("payout_yen")) for r in hits]
    sorted_hit_payouts = sorted(hit_payouts, reverse=True)
    top1 = sorted_hit_payouts[0] if sorted_hit_payouts else 0
    top3 = sum(sorted_hit_payouts[:3])
    profitable_hits = sum(1 for r in hits if inum(r.get("payout_yen")) > inum(r.get("stake_yen")))
    losing_or_flat_hits = len(hits) - profitable_hits
    return {
        "purchased_races": len(bought),
        "hits": len(hits),
        "hit_rate": len(hits) / len(bought) if bought else 0.0,
        "stake_yen": stake,
        "payout_yen": payout,
        "profit_yen": payout - stake,
        "roi": payout / stake if stake else 0.0,
        "avg_stake_per_purchase": stake / len(bought) if bought else 0.0,
        "avg_payout_per_hit": payout / len(hits) if hits else 0.0,
        "median_payout_per_hit": statistics.median(hit_payouts) if hit_payouts else 0.0,
        "max_payout_yen": top1,
        "top1_payout_share": top1 / payout if payout else 0.0,
        "top3_payout_share": top3 / payout if payout else 0.0,
        "roi_excluding_largest_hit": (payout - top1) / stake if stake else 0.0,
        "profitable_hit_races": profitable_hits,
        "losing_or_flat_hit_races": losing_or_flat_hits,
    }


def exact_order(pos: dict[int, int], cars: list[int]) -> bool:
    return all(pos.get(car) == i + 1 for i, car in enumerate(cars))


def top2_pair(pos: dict[int, int], a: int, b: int) -> bool:
    return {pos.get(a), pos.get(b)} == {1, 2}


def strategy_outcomes(rows: list[dict[str, str]], results: dict[str, dict[int, int]], strategy: str) -> dict[str, object]:
    bought = [r for r in rows if r.get("strategy") == strategy and inum(r.get("purchased")) == 1]
    c = Counter()
    for r in bought:
        pos = results.get(r["race_id"], {})
        a = inum(r.get("a_car")); b = inum(r.get("b_car")); m3 = inum(r.get("m3_car"))
        r1l = inum(r.get("r1l_car")); r1b = inum(r.get("r1b_car"))
        if a and b and top2_pair(pos, a, b):
            c["A_B_top2_any_order"] += 1
        if r1l and pos.get(r1l) == 1:
            c["R1L_winner"] += 1
        if r1l and r1b and top2_pair(pos, r1l, r1b):
            c["R1_pair_top2_any_order"] += 1
        if strategy == "early_v0":
            if r1l and r1b and top2_pair(pos, r1l, r1b) and pos.get(a) == 3 or r1l and r1b and top2_pair(pos, r1l, r1b) and pos.get(b) == 3:
                c["target_shape_hit"] += 1
        elif strategy == "middle_v0":
            if exact_order(pos, [a, b, r1b]):
                c["A-B-R1B"] += 1
            if exact_order(pos, [r1l, r1b, b]):
                c["R1L-R1B-B"] += 1
        elif strategy == "late_v1":
            decision = r.get("decision")
            c[f"decision::{decision}"] += 1
            if decision == "mainline_4pt" and top2_pair(pos, a, b):
                c["mainline_AB_top2"] += 1
                if pos.get(m3) == 3:
                    c["mainline_AB_top2_M3_third"] += 1
    return {"purchased_races": len(bought), "counts": dict(c)}


def decision_structure(rows: list[dict[str, str]], strategy: str) -> dict[str, object]:
    rs = [r for r in rows if r.get("strategy") == strategy]
    bought = [r for r in rs if inum(r.get("purchased")) == 1]
    out: dict[str, object] = {
        "decision_rows": len(rs),
        "purchased_races": len(bought),
        "purchase_rate": len(bought) / len(rs) if rs else 0.0,
        "decision_counts": dict(Counter(r.get("decision", "") for r in rs)),
    }
    if strategy == "early_v0":
        out["pair_win_all"] = q(v for r in rs if (v := fnum(r.get("pair_win_sum"))) is not None)
        out["pair_win_bought"] = q(v for r in bought if (v := fnum(r.get("pair_win_sum"))) is not None)
    elif strategy == "middle_v0":
        out["b_score_all"] = q(v for r in rs if (v := fnum(r.get("b_score"))) is not None)
        out["b_score_bought"] = q(v for r in bought if (v := fnum(r.get("b_score"))) is not None)
    elif strategy == "late_v1":
        out["pair_top3_all"] = q(v for r in rs if (v := fnum(r.get("pair_top3_sum"))) is not None)
        out["pair_win_all"] = q(v for r in rs if (v := fnum(r.get("pair_win_sum"))) is not None)
        out["b_top3_all"] = q(v for r in rs if (v := fnum(r.get("b_top3"))) is not None)
    return out


def branch_metrics(rows: list[dict[str, str]], strategy: str) -> dict[str, object]:
    rs = [r for r in rows if r.get("strategy") == strategy]
    decisions = sorted(set(r.get("decision", "") for r in rs))
    return {d: race_financial([r for r in rs if r.get("decision") == d]) for d in decisions if d}


def analyze(year: int, data_dir: Path, sim_dir: Path) -> dict[str, object]:
    decisions = read_csv(sim_dir / "race_decisions.csv")
    results = result_map(data_dir)
    strategies = ("early_v0", "middle_v0", "late_v1")
    return {
        "year": year,
        "dataset_races": len(read_csv(data_dir / "races.csv")),
        "strategies": {
            s: {
                "decision_structure": decision_structure(decisions, s),
                "financial": race_financial([r for r in decisions if r.get("strategy") == s]),
                "outcome_structure": strategy_outcomes(decisions, results, s),
                "branch_financial": branch_metrics(decisions, s),
            }
            for s in strategies
        },
        "combined_financial": race_financial(decisions),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-2024", default="data/2024/s_class_yosen")
    p.add_argument("--data-2025", default="data/2025/s_class_yosen")
    p.add_argument("--sim-2024", required=True)
    p.add_argument("--sim-2025", required=True)
    p.add_argument("--population-audit", default="data/audits/oos_population_2024_vs_2025.json")
    p.add_argument("--out", default="data/audits/frozen_structure_2024_vs_2025.json")
    args = p.parse_args()

    report = {
        "purpose": "Describe 2024 vs 2025 structural differences under the already-frozen strategies. No thresholds, branches, bets, or filters are changed.",
        "population": json.loads(Path(args.population_audit).read_text(encoding="utf-8")),
        "years": {
            "2024": analyze(2024, Path(args.data_2024), Path(args.sim_2024)),
            "2025": analyze(2025, Path(args.data_2025), Path(args.sim_2025)),
        },
        "warning": "2025 is the discovery/in-sample year and 2024 is OOS. Differences describe what happened; they do not prove a causal regime change and must not be used to retroactively tune the frozen OOS result.",
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
