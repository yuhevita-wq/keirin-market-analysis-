from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import defaultdict
from pathlib import Path

from .s_yosen_2025 import RaceRef, extract_entries, fetch_html as fetch_race_html, make_session as make_race_session
from .line_formation_2025 import parse_line_formation_html

RELEVANT_NUMERIC = ("score", "win_rate", "top2_rate", "top3_rate")
ROLE_FIELDS = ("a_car", "b_car", "m3_car", "r1l_car", "r1b_car")


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        r = csv.DictReader(f)
        return list(r.fieldnames or []), list(r)


def fnum(v: object) -> float:
    try:
        return float(str(v))
    except (TypeError, ValueError):
        return float("-inf")


def line_groups(entries: list[dict[str, str]]) -> dict[int, list[dict[str, str]]]:
    by: dict[int, list[dict[str, str]]] = defaultdict(list)
    for e in entries:
        if e.get("line_id", "").isdigit() and e.get("line_position", "").isdigit():
            by[int(e["line_id"])].append(e)
    for lid in by:
        by[lid].sort(key=lambda x: int(x["line_position"]))
    return dict(by)


def independent_main(entries: list[dict[str, str]]) -> tuple[int, list[dict[str, str]]] | None:
    cands = []
    for lid, mem in line_groups(entries).items():
        if len(mem) < 2:
            continue
        key = (
            fnum(mem[0].get("score")) + fnum(mem[1].get("score")),
            fnum(mem[0].get("score")),
            1 if len(mem) >= 3 else 0,
            -lid,
        )
        cands.append((key, lid, mem))
    if not cands:
        return None
    _, lid, mem = max(cands, key=lambda x: x[0])
    return lid, mem


def independent_rival(entries: list[dict[str, str]], main_id: int, mode: str) -> list[dict[str, str]] | None:
    cands = []
    for lid, mem in line_groups(entries).items():
        if lid == main_id or len(mem) < 2:
            continue
        pair = fnum(mem[0].get("score")) + fnum(mem[1].get("score"))
        if mode in {"early", "middle"}:
            key = (pair, -lid)
        elif mode == "late":
            key = (pair, fnum(mem[0].get("score")), -lid)
        else:
            raise ValueError(mode)
        cands.append((key, mem))
    return max(cands, key=lambda x: x[0])[1] if cands else None


def expected_roles(entries: list[dict[str, str]], mode: str) -> dict[str, str] | None:
    main = independent_main(entries)
    if not main:
        return None
    main_id, mem = main
    if len(mem) < 3:
        return None
    rival = independent_rival(entries, main_id, mode)
    return {
        "a_car": mem[0]["car_no"],
        "b_car": mem[1]["car_no"],
        "m3_car": mem[2]["car_no"],
        "r1l_car": rival[0]["car_no"] if rival else "",
        "r1b_car": rival[1]["car_no"] if rival else "",
    }


def evenly_spaced(rows: list[dict[str, str]], count: int) -> list[dict[str, str]]:
    if count <= 0 or not rows:
        return []
    if count >= len(rows):
        return rows
    if count == 1:
        return [rows[len(rows) // 2]]
    idxs = sorted({round(i * (len(rows) - 1) / (count - 1)) for i in range(count)})
    return [rows[i] for i in idxs]


def audit_year(year: int, data_dir: Path, decisions_path: Path, live_samples: int, sleep_seconds: float) -> dict[str, object]:
    race_fields, races = read_csv(data_dir / "races.csv")
    entry_fields, entries = read_csv(data_dir / "entries.csv")
    _, decisions = read_csv(decisions_path)

    entries_by: dict[str, list[dict[str, str]]] = defaultdict(list)
    for e in entries:
        entries_by[e["race_id"]].append(e)

    missing = {k: 0 for k in RELEVANT_NUMERIC}
    nonnumeric = {k: 0 for k in RELEVANT_NUMERIC}
    vals: dict[str, list[float]] = {k: [] for k in RELEVANT_NUMERIC}
    rate_order_violations = 0
    raw_snapshot_missing_values = 0
    raw_snapshot_bad_json = 0

    for e in entries:
        for k in RELEVANT_NUMERIC:
            v = e.get(k, "")
            if v == "":
                missing[k] += 1
            try:
                vals[k].append(float(v))
            except (TypeError, ValueError):
                nonnumeric[k] += 1
        try:
            w, t2, t3 = float(e["win_rate"]), float(e["top2_rate"]), float(e["top3_rate"])
            if not (0 <= w <= t2 <= t3 <= 100):
                rate_order_violations += 1
        except (TypeError, ValueError, KeyError):
            rate_order_violations += 1
        try:
            raw = json.loads(e.get("raw_row_json", ""))
            cells = [str(x) for x in raw.get("cells", [])]
            # These exact stored values must still be present in the raw row snapshot,
            # and in the natural table order score -> win -> top2 -> top3.
            positions = []
            start = 0
            for k in RELEVANT_NUMERIC:
                target = str(e.get(k, ""))
                pos = next((i for i in range(start, len(cells)) if cells[i] == target), None)
                if pos is None:
                    raw_snapshot_missing_values += 1
                    positions = []
                    break
                positions.append(pos)
                start = pos + 1
        except Exception:
            raw_snapshot_bad_json += 1

    line_invariant_failures = 0
    formation_mismatches = 0
    main_not_found = 0
    rival_mode_disagreements = 0
    rival_pair_score_ties = 0

    for race in races:
        es = entries_by[race["race_id"]]
        groups = line_groups(es)
        all_cars = {int(e["car_no"]) for e in es}
        line_cars = {int(e["car_no"]) for mem in groups.values() for e in mem}
        ok = line_cars == all_cars and sorted(groups) == list(range(1, len(groups) + 1))
        rebuilt_parts = []
        for lid in sorted(groups):
            mem = groups[lid]
            positions = [int(e["line_position"]) for e in mem]
            sizes = {int(e["line_size"]) for e in mem if e.get("line_size", "").isdigit()}
            if positions != list(range(1, len(mem) + 1)) or sizes != {len(mem)}:
                ok = False
            rebuilt_parts.append("-".join(e["car_no"] for e in mem))
        if not ok:
            line_invariant_failures += 1
        rebuilt = "/".join(rebuilt_parts)
        if rebuilt != race.get("predicted_line_formation", "") or any(e.get("predicted_line_formation", "") != rebuilt for e in es):
            formation_mismatches += 1

        main = independent_main(es)
        if not main:
            main_not_found += 1
            continue
        main_id, _ = main
        rivals = [mem for lid, mem in groups.items() if lid != main_id and len(mem) >= 2]
        if rivals:
            pair_scores = [fnum(m[0].get("score")) + fnum(m[1].get("score")) for m in rivals]
            mx = max(pair_scores)
            if sum(abs(x - mx) < 1e-12 for x in pair_scores) > 1:
                rival_pair_score_ties += 1
        er = independent_rival(es, main_id, "early")
        lr = independent_rival(es, main_id, "late")
        ekey = tuple(x["car_no"] for x in er[:2]) if er else ()
        lkey = tuple(x["car_no"] for x in lr[:2]) if lr else ()
        if ekey != lkey:
            rival_mode_disagreements += 1

    role_mismatches = 0
    role_checked = 0
    decision_segment_bad = 0
    for d in decisions:
        mode = {"early_v0": "early", "middle_v0": "middle", "late_v1": "late"}.get(d.get("strategy", ""))
        if mode is None:
            continue
        role_checked += 1
        expected = expected_roles(entries_by[d["race_id"]], mode)
        if expected is None:
            role_mismatches += 1
            continue
        for k in ROLE_FIELDS:
            if str(d.get(k, "")) != str(expected.get(k, "")):
                role_mismatches += 1
                break
        expected_seg = {"early_v0": "前半", "middle_v0": "中盤", "late_v1": "後半"}[d["strategy"]]
        if d.get("segment") != expected_seg:
            decision_segment_bad += 1

    # Re-read a deterministic set of historical KDreams pages and compare the
    # stored pre-race inputs and published line forecast to a fresh parse.
    live_mismatches: list[dict[str, object]] = []
    live_fetch_failures: list[dict[str, object]] = []
    session = make_race_session()
    ordered_races = sorted(races, key=lambda r: (r["race_date"], r["track"], int(r["race_no"])))
    sample = evenly_spaced(ordered_races, live_samples)
    for race in sample:
        try:
            html = fetch_race_html(session, race["source_url"])
            ref = RaceRef(discovered_on=race["race_date"], race_no=int(race["race_no"]), url=race["source_url"])
            _, fresh_entries = extract_entries(html, ref)
            fresh_by = {str(e["car_no"]): e for e in fresh_entries}
            stored_by = {str(e["car_no"]): e for e in entries_by[race["race_id"]]}
            issues = []
            if set(fresh_by) != set(stored_by):
                issues.append("entrant_set")
            for car in sorted(set(fresh_by) & set(stored_by), key=int):
                for k in ("player_name",) + RELEVANT_NUMERIC:
                    if str(fresh_by[car].get(k, "")) != str(stored_by[car].get(k, "")):
                        issues.append(f"car{car}:{k}")
            fresh_line = parse_line_formation_html(html)
            if str(fresh_line.get("formation", "")) != str(race.get("predicted_line_formation", "")):
                issues.append("predicted_line_formation")
            if issues:
                live_mismatches.append({"race_id": race["race_id"], "date": race["race_date"], "issues": sorted(set(issues))})
        except Exception as exc:
            live_fetch_failures.append({"race_id": race["race_id"], "date": race["race_date"], "error": f"{type(exc).__name__}: {exc}"})
        if sleep_seconds:
            time.sleep(sleep_seconds)

    stats = {}
    for k, xs in vals.items():
        ys = sorted(xs)
        stats[k] = {
            "count": len(ys),
            "min": ys[0] if ys else None,
            "median": ys[len(ys)//2] if ys else None,
            "max": ys[-1] if ys else None,
        }

    core_ok = (
        all(missing[k] == 0 and nonnumeric[k] == 0 for k in RELEVANT_NUMERIC)
        and rate_order_violations == 0
        and raw_snapshot_bad_json == 0
        and raw_snapshot_missing_values == 0
        and line_invariant_failures == 0
        and formation_mismatches == 0
        and main_not_found == 0
        and role_mismatches == 0
        and decision_segment_bad == 0
        and len(live_mismatches) == 0
        and len(live_fetch_failures) == 0
    )
    return {
        "year": year,
        "race_rows": len(races),
        "entry_rows": len(entries),
        "required_columns_present": {k: (k in entry_fields) for k in ("score","win_rate","top2_rate","top3_rate","raw_row_json","line_id","line_position","line_size","predicted_line_formation")},
        "missing_numeric": missing,
        "nonnumeric": nonnumeric,
        "numeric_distribution": stats,
        "rate_order_violations": rate_order_violations,
        "raw_snapshot_bad_json": raw_snapshot_bad_json,
        "raw_snapshot_missing_values": raw_snapshot_missing_values,
        "line_invariant_failures": line_invariant_failures,
        "formation_mismatches": formation_mismatches,
        "main_not_found": main_not_found,
        "role_decisions_checked": role_checked,
        "role_mismatches": role_mismatches,
        "decision_segment_mismatches": decision_segment_bad,
        "rival_pair_score_tie_races": rival_pair_score_ties,
        "rival_tiebreak_mode_disagreements": rival_mode_disagreements,
        "live_source_samples_requested": live_samples,
        "live_source_samples_checked": len(sample) - len(live_fetch_failures),
        "live_source_mismatches": live_mismatches,
        "live_source_fetch_failures": live_fetch_failures,
        "status": "PASS" if core_ok else "FAIL",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--decisions-2024", required=True)
    ap.add_argument("--decisions-2025", required=True)
    ap.add_argument("--live-samples", type=int, default=24)
    ap.add_argument("--sleep", type=float, default=0.08)
    ap.add_argument("--out", default="data/audits/input_role_audit_2024_2025.json")
    args = ap.parse_args()

    _, e24 = read_csv(Path("data/2024/s_class_yosen/entries.csv"))
    _, e25 = read_csv(Path("data/2025/s_class_yosen/entries.csv"))
    relevant_schema_24 = [k for k in e24[0].keys() if k not in {"captured_at_utc"}] if e24 else []
    relevant_schema_25 = [k for k in e25[0].keys() if k not in {"captured_at_utc"}] if e25 else []

    report = {
        "purpose": "Audit stored pre-race inputs and line-to-role transformation only. No strategy thresholds, bets, results, or payouts are modified.",
        "collector_identity": "2024 collector imports the frozen 2025 extract_entries parser directly.",
        "entry_schema_same_except_capture_timestamp": relevant_schema_24 == relevant_schema_25,
        "years": {
            "2024": audit_year(2024, Path("data/2024/s_class_yosen"), Path(args.decisions_2024), args.live_samples, args.sleep),
            "2025": audit_year(2025, Path("data/2025/s_class_yosen"), Path(args.decisions_2025), args.live_samples, args.sleep),
        },
    }
    report["status"] = "PASS" if report["entry_schema_same_except_capture_timestamp"] and all(v["status"] == "PASS" for v in report["years"].values()) else "FAIL"
    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
