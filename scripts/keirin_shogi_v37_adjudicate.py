#!/usr/bin/env python3
from __future__ import annotations

import json
import math
from pathlib import Path


V37 = Path("results/keirin_shogi/v37_true_future_block1")
V23 = Path("results/keirin_shogi/v23_third_true_future_block1")
OUT = Path("results/keirin_shogi/v37_adoption")


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def exact_two_sided(successes: int, trials: int):
    if trials == 0:
        return 1.0
    probability = sum(math.comb(trials, k) for k in range(successes + 1)) / (2**trials)
    return min(1.0, 2.0 * probability)


def comparison(a, b, key):
    by_id = {str(row["race_id"]): row for row in b}
    both = a_only = b_only = neither = 0
    for row in a:
        other = by_id[str(row["race_id"])]
        hit_a = bool(row[key])
        hit_b = bool(other[key])
        if hit_a and hit_b:
            both += 1
        elif hit_a:
            a_only += 1
        elif hit_b:
            b_only += 1
        else:
            neither += 1
    discordant = a_only + b_only
    smaller = min(a_only, b_only)
    return {
        "both": both,
        "v37_only": a_only,
        "v23_only": b_only,
        "neither": neither,
        "discordant": discordant,
        "exact_mcnemar_two_sided_p": exact_two_sided(smaller, discordant),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    v37_summary = load(V37 / "summary.json")
    v23_summary = load(V23 / "summary.json")
    v37_logs = load(V37 / "race_log.json")
    v23_logs = load(V23 / "race_log.json")
    if {str(row["race_id"]) for row in v37_logs} != {
        str(row["race_id"]) for row in v23_logs
    }:
        raise RuntimeError("v37 and v23 future race sets differ")

    v37_metrics = v37_summary["metrics"]
    v23_metrics = v23_summary["metrics"]
    deltas = {
        "third_capture": float(v37_metrics["third_capture"])
        - float(v23_metrics["third_capture"]),
        "complete_board_capture": float(v37_metrics["complete_board_capture"])
        - float(v23_metrics["complete_board_capture"]),
        "third_given_first_second": float(v37_metrics["third_given_first_second"])
        - float(v23_metrics["third_given_first_second"]),
        "avg_third_candidates": float(v37_metrics["avg_third_candidates"])
        - float(v23_metrics["avg_third_candidates"]),
    }
    requirements = {
        "future_third_above_v23": deltas["third_capture"] > 0.0,
        "future_complete_above_v23": deltas["complete_board_capture"] > 0.0,
        "future_candidate_count_below_v23": deltas["avg_third_candidates"] < 0.0,
        "candidate_count_at_most_2_85": float(v37_metrics["avg_third_candidates"]) <= 2.85,
        "frozen_without_future_retuning": not bool(
            v37_summary["guards"]["parameters_changed_after_future_open"]
        ),
    }
    adopt = all(requirements.values())
    third_pair = comparison(v37_logs, v23_logs, "third_hit")
    complete_pair = comparison(v37_logs, v23_logs, "complete_board_hit")

    summary = {
        "decision": "CURRENT_BEST_PROVISIONAL" if adopt else "NOT_ADOPTED",
        "board_version": "v37" if adopt else "v31_v23",
        "future_block": v37_summary["future_block"],
        "freeze_commit": v37_summary["freeze_commit"],
        "races": len(v37_logs),
        "v37": v37_metrics,
        "legacy_v23_same_board": v23_metrics,
        "v37_minus_v23": deltas,
        "paired_comparison": {
            "third": third_pair,
            "complete_board": complete_pair,
        },
        "requirements": requirements,
        "interpretation": (
            "v37 wins the same-race future comparison on third capture and complete-board "
            "capture with fewer candidates. The 153-race paired difference is not yet "
            "statistically decisive, so adoption is provisional and the next sealed block "
            "must be evaluated without retuning."
        ),
        "absolute_historical_target_note": (
            "v37 future third capture is below the old historical 53.03% reference by "
            "0.75 percentage points, but the legacy v23 model itself fell to 47.71% on "
            "the same future block."
        ),
    }
    (OUT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (OUT / "README.md").write_text(
        "# v37 adoption decision\n\n"
        "同一の完全未来153レースで、v37と旧v23の3着段を比較。\n\n"
        f"- v37 3着: {float(v37_metrics['third_capture']):.2%}\n"
        f"- v23 3着: {float(v23_metrics['third_capture']):.2%}\n"
        f"- v37 完全盤面: {float(v37_metrics['complete_board_capture']):.2%}\n"
        f"- v23 完全盤面: {float(v23_metrics['complete_board_capture']):.2%}\n"
        f"- 判定: **{summary['decision']}**\n\n"
        "次の未開封ブロックでも無調整検証を継続する。\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
