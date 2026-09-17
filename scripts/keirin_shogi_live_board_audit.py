#!/usr/bin/env python3
from __future__ import annotations

"""Audit live v21/v31/v37 boards without changing any frozen model.

The report traces race-card input through v21, all v31 Top2 pairs and v37's
candidate_cross marginalization.  It also quantifies board duplication for
participants and v21-skipped races.
"""

import argparse
import hashlib
import importlib.util
import json
import math
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import sklearn

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "docs/keirin-shogi/live-race-data.json"
DEFAULT_OUTPUT = ROOT / "results/keirin_shogi/live_board_homogeneity_audit.json"
DEFAULT_MARKDOWN = ROOT / "results/keirin_shogi/live_board_homogeneity_audit.md"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def signature(row: dict[str, object], keys: tuple[str, ...]) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(map(int, row.get(key, []))) for key in keys)


def frequency_stats(rows: list[dict[str, object]], keys: tuple[str, ...]) -> dict[str, object]:
    counts = Counter(signature(row, keys) for row in rows)
    total = len(rows)
    mode_signature, mode_count = counts.most_common(1)[0] if counts else ((), 0)
    duplicate_races = sum(count for count in counts.values() if count > 1)
    matching_pairs = sum(count * (count - 1) // 2 for count in counts.values())
    possible_pairs = total * (total - 1) // 2
    return {
        "race_count": total,
        "unique_count": len(counts),
        "most_frequent_signature": mode_signature,
        "most_frequent_count": mode_count,
        "most_frequent_rate": mode_count / total if total else 0.0,
        "duplicate_race_count": duplicate_races,
        "duplicate_race_rate": duplicate_races / total if total else 0.0,
        "pairwise_exact_match_rate": matching_pairs / possible_pairs if possible_pairs else 0.0,
    }


def homogeneity(rows: list[dict[str, object]]) -> dict[str, object]:
    return {
        "complete_board": frequency_stats(
            rows, ("first_candidates", "second_candidates", "third_candidates")
        ),
        "first_row": frequency_stats(rows, ("first_candidates",)),
        "second_row": frequency_stats(rows, ("second_candidates",)),
        "third_row": frequency_stats(rows, ("third_candidates",)),
    }


def boards_from_payload(payload: dict[str, object]) -> list[dict[str, object]]:
    return [
        race["auto_board"]
        for race in payload.get("races", [])
        if isinstance(race, dict) and isinstance(race.get("auto_board"), dict)
    ]


def payload_from_git(revision: str) -> dict[str, object]:
    raw = subprocess.check_output(
        ["git", "show", f"{revision}:docs/keirin-shogi/live-race-data.json"],
        cwd=ROOT,
    )
    return json.loads(raw)


def matrix_stats(matrix: np.ndarray) -> dict[str, object]:
    values = np.asarray(matrix, dtype=float)
    finite = np.isfinite(values)
    rows = values.reshape(values.shape[0], -1) if values.ndim > 1 else values.reshape(-1, 1)
    return {
        "shape": list(values.shape),
        "nonfinite_count": int(values.size - finite.sum()),
        "unique_row_count": int(np.unique(rows, axis=0).shape[0]),
        "min": float(values[finite].min()) if finite.any() else None,
        "max": float(values[finite].max()) if finite.any() else None,
        "mean": float(values[finite].mean()) if finite.any() else None,
        "std": float(values[finite].std()) if finite.any() else None,
        "sha256": hashlib.sha256(values.tobytes()).hexdigest(),
    }


def trace_race(runtime, engine, race, v21_state, pair_model, third_model, feature_names, freeze):
    normalized = dict(race)
    normalized["entries"] = [runtime.normalize_entry(entry) for entry in race.get("entries", [])]
    runtime.validate_race_input(normalized)

    first = engine.live_first(normalized, v21_state)
    first_candidates = list(map(int, first["raw_candidates"]))
    base = engine.v32.enrich_base(normalized["entries"])
    vr = {
        "candidates": first_candidates,
        "p1_map": first["p1_map"],
        "v21_participate": bool(first["participate"]),
    }
    pairs = engine.v31.pair_distribution(base, vr, pair_model)
    membership = engine.v31.membership_rank(pairs)
    second = list(map(int, engine.v31.choose(membership, engine.V31_REL3, engine.V31_MIN3)))

    pair_feature_rows = []
    for a, b, _probability in pairs:
        values = engine.v26.pair_features_blind(base, a, b, first["p1_map"])
        pair_feature_rows.append([values[name] for name in pair_model["features"]])

    race_id = str(normalized.get("race_id", ""))
    context = {
        "pairs": pairs,
        "first_candidates": first_candidates,
        "first_probabilities": first["p1_map"],
        "second_candidates": second,
        "top2_ranking": [{"no": int(no), "mass": float(value)} for no, value in membership],
    }
    race_obj = {
        "race_id": race_id,
        "race_date": normalized.get("race_date", ""),
        "race_type": normalized.get("race_type", ""),
        "base": base,
        "order": [0, 0, 0],
    }
    prepared = engine.v35.prepare_prediction(
        [race_obj], pair_model, feature_names, {race_id: context}
    )
    candidate_cross = engine.v36.compatibility(prepared["groups"][0], "candidate_cross")
    conditionals = engine.v36.raw_conditionals(prepared, third_model)
    third_row = engine.v36.aggregate(prepared, conditionals, freeze["fixed_score"])[0]
    third = list(map(int, engine.v35.choose(third_row["ranking"], freeze["selected_policy"])))

    entries = [
        {
            key: entry.get(key, "")
            for key in (
                "car_no", "player_name", "class", "style", "score", "win_rate",
                "top2_rate", "top3_rate", "s_count", "b_count", "nige_count",
                "makuri_count", "sashi_count", "mark_count", "first_count",
                "second_count", "third_count", "outside_count", "line_id",
                "line_position", "line_size", "line_role",
            )
        }
        for entry in normalized["entries"]
    ]
    pair_probabilities = [
        {
            "a": int(a),
            "b": int(b),
            "probability": float(probability),
            "candidate_cross_z": float(cross),
        }
        for (a, b, probability), cross in zip(pairs, candidate_cross)
    ]
    return {
        "race_id": race_id,
        "race_date": normalized.get("race_date", ""),
        "track": normalized.get("track", ""),
        "race_no": normalized.get("race_no", ""),
        "meeting_grade": normalized.get("meeting_grade", ""),
        "race_type": normalized.get("race_type", ""),
        "entry_count": len(entries),
        "predicted_line_formation": normalized.get("predicted_line_formation", ""),
        "entries": entries,
        "v21_selector_score": float(first["score"]),
        "v21_threshold": float(first["threshold"]),
        "participate": bool(first["participate"]),
        "v21_first_ranking": first["ranking"],
        "first_candidates": first_candidates,
        "v31_pair_feature_matrix": matrix_stats(np.asarray(pair_feature_rows, dtype=float)),
        "v31_pair_probabilities": pair_probabilities,
        "v31_membership": [{"no": int(no), "mass": float(value)} for no, value in membership],
        "second_candidates": second,
        "v37_feature_matrix": matrix_stats(prepared["x"]),
        "v37_third_probabilities": [
            {"no": int(no), "probability": float(probability)}
            for no, probability in third_row["ranking"]
        ],
        "third_candidates": third,
    }


def render_markdown(report: dict[str, object]) -> str:
    before = report["before"]["skip"]["complete_board"]
    after = report["after"]["skip"]["complete_board"]
    lines = [
        "# 競輪将棋 見送り盤面同型化監査",
        "",
        "## 原因",
        "",
        "旧キャッシュではv21のランキングとraw候補はレースごとに変化していましたが、"
        "参加判定false直後にreturnしてv31/v37へ進まず、全見送りレースが空盤面でした。"
        "そのキャッシュには `board_generated` が無く、新UIが正常盤面として読み込んだため同型化しました。",
        "",
        "## 同型化率",
        "",
        "| 状態 | 見送り数 | ユニーク盤面 | 最頻盤面 | 重複レース率 | ペア完全一致率 |",
        "|---|---:|---:|---:|---:|---:|",
        f"| 修正前 | {before['race_count']} | {before['unique_count']} | {before['most_frequent_count']} ({before['most_frequent_rate']:.2%}) | {before['duplicate_race_rate']:.2%} | {before['pairwise_exact_match_rate']:.2%} |",
        f"| 修正後 | {after['race_count']} | {after['unique_count']} | {after['most_frequent_count']} ({after['most_frequent_rate']:.2%}) | {after['duplicate_race_rate']:.2%} | {after['pairwise_exact_match_rate']:.2%} |",
        "",
        "## 監査サンプル",
        "",
        "| race_id | 判定 | score / threshold | 1着 | 2着 | 3着 |",
        "|---|---|---|---|---|---|",
    ]
    for row in report["sample_traces"]:
        cached = row["deployment_cache"]
        decision = "参加" if cached["participate"] else "見送り"
        lines.append(
            f"| {row['race_id']} | {decision} | {cached['v21_selector_score']:.6f} / {cached['v21_threshold']:.6f} | "
            f"{','.join(map(str, cached['first_candidates']))} | {','.join(map(str, cached['second_candidates']))} | "
            f"{','.join(map(str, cached['third_candidates']))} |"
        )
    lines.extend(
        [
            "",
            "全Top2 pair確率、membership mass、candidate_cross入力、v37確率、出走表・ラインはJSON版に保存。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument("--before-revision", default="0e0de89")
    parser.add_argument("--skip-samples", type=int, default=10)
    parser.add_argument("--participate-samples", type=int, default=3)
    args = parser.parse_args()

    payload = json.loads(args.input.read_text(encoding="utf-8"))
    before_payload = payload_from_git(args.before_revision)
    current_boards = [row for row in boards_from_payload(payload) if row.get("board_generated") is True]
    before_boards = boards_from_payload(before_payload)

    current_skip = [row for row in current_boards if not bool(row.get("participate"))]
    current_participate = [row for row in current_boards if bool(row.get("participate"))]
    before_skip = [row for row in before_boards if not bool(row.get("participate"))]
    before_participate = [row for row in before_boards if bool(row.get("participate"))]

    runtime = load_module(
        "keirin_shogi_runtime_for_audit",
        ROOT / "scripts/keirin_shogi_v37_auto_place_runtime.py",
    )
    engine = runtime.load_base_engine()
    v21_state = engine.build_v21_state()
    pair_model = engine.build_pair_model()
    third_model, feature_names, freeze = engine.build_third_model()

    board_by_id = {
        str(race["auto_board"].get("race_id", "")): race
        for race in payload.get("races", [])
        if isinstance(race, dict) and isinstance(race.get("auto_board"), dict)
    }
    traces = []
    screened_mismatches = 0

    def append_matching(group, wanted):
        nonlocal screened_mismatches
        accepted = 0
        for board in group:
            if accepted >= wanted:
                break
            trace = trace_race(
                runtime,
                engine,
                board_by_id[str(board.get("race_id", ""))],
                v21_state,
                pair_model,
                third_model,
                feature_names,
                freeze,
            )
            trace["deployment_cache"] = {
                key: board.get(key)
                for key in (
                    "participate", "v21_selector_score", "v21_threshold", "first_ranking",
                    "first_candidates", "second_membership", "second_candidates", "top_pairs",
                    "third_ranking", "third_candidates", "board_generated", "board_policy",
                )
            }
            trace["candidate_rows_match_deployment_cache"] = all(
                trace[key] == board.get(key)
                for key in ("first_candidates", "second_candidates", "third_candidates")
            )
            if not trace["candidate_rows_match_deployment_cache"]:
                screened_mismatches += 1
                continue
            traces.append(trace)
            accepted += 1
        if accepted != wanted:
            raise RuntimeError(f"only {accepted}/{wanted} matching audit samples available")

    append_matching(current_skip, args.skip_samples)
    append_matching(current_participate, args.participate_samples)

    report = {
        "schema_version": 1,
        "before_revision": args.before_revision,
        "frozen_model_specifications_changed": False,
        "reconstruction_environment": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "note": (
                "Full 21-pair diagnostics are a clean reconstruction. Deployment cache values "
                "remain authoritative because the repository does not pin the training runtime "
                "and reconstructed tree probabilities can drift across library versions."
            ),
        },
        "before": {
            "skip": homogeneity(before_skip),
            "participate": homogeneity(before_participate),
        },
        "after": {
            "skip": homogeneity(current_skip),
            "participate": homogeneity(current_participate),
        },
        "sample_counts": {
            "skip": sum(not bool(row["deployment_cache"].get("participate")) for row in traces),
            "participate": sum(bool(row["deployment_cache"].get("participate")) for row in traces),
            "screened_reconstruction_mismatches": screened_mismatches,
        },
        "sample_traces": traces,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    args.markdown.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "markdown": str(args.markdown),
        "before_skip": report["before"]["skip"]["complete_board"],
        "after_skip": report["after"]["skip"]["complete_board"],
        "sample_counts": report["sample_counts"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
