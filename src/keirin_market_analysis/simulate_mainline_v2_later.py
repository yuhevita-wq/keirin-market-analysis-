from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

from .simulate_mainline_v1 import choose_main_line, num, segment_for

DATA_DIR = Path("data/2025/s_class_yosen")
OUT_DIR = DATA_DIR / "simulations" / "mainline_v2_later"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def choose_third_candidates(
    entries: list[dict[str, str]],
    main_line_id: int,
    main_members: list[dict[str, str]],
) -> tuple[list[int], str, bool]:
    a = int(main_members[0]["car_no"])
    b = int(main_members[1]["car_no"])

    if len(main_members) >= 3:
        main3 = int(main_members[2]["car_no"])
        remaining = [e for e in entries if int(e["car_no"]) not in {a, b, main3}]
        best_remaining = min(
            remaining,
            key=lambda e: (-num(e.get("score", "")), int(e["car_no"])),
        )
        return [main3, int(best_remaining["car_no"])], "main3_plus_best_remaining_score", False

    by_line: dict[int, list[dict[str, str]]] = defaultdict(list)
    for e in entries:
        if e.get("line_id", "").isdigit() and e.get("line_position", "").isdigit():
            by_line[int(e["line_id"])].append(e)
    for lid in by_line:
        by_line[lid].sort(key=lambda e: int(e["line_position"]))

    rival_candidates: list[tuple[tuple[float, float, int], list[dict[str, str]]]] = []
    for lid, members in by_line.items():
        if lid == main_line_id or len(members) < 2:
            continue
        leader, second = members[0], members[1]
        strength = num(leader.get("score", "")) + num(second.get("score", ""))
        leader_score = num(leader.get("score", ""))
        rival_candidates.append(((strength, leader_score, -lid), members))

    if rival_candidates:
        _, rival = max(rival_candidates, key=lambda x: x[0])
        return [int(rival[0]["car_no"]), int(rival[1]["car_no"])], "strongest_rival_leader_second", False

    # Defensive fallback: guarantee two third-place candidates even if no other 2+ rider line exists.
    remaining = [e for e in entries if int(e["car_no"]) not in {a, b}]
    ranked = sorted(remaining, key=lambda e: (-num(e.get("score", "")), int(e["car_no"])))
    return [int(ranked[0]["car_no"]), int(ranked[1]["car_no"])], "fallback_top2_remaining_score", True


def summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    n = len(rows)
    stake = sum(int(r["stake_yen"]) for r in rows)
    payout = sum(int(r["payout_yen"]) for r in rows)
    top2 = sum(int(r["mainline_top2"]) for r in rows)
    top2_rows = [r for r in rows if int(r["mainline_top2"]) == 1]
    third_cap = sum(int(r["third_captured_given_top2"]) for r in top2_rows)
    hits = sum(int(r["hit"]) for r in rows)
    return {
        "races": n,
        "mainline_top2_races": top2,
        "mainline_top2_rate": top2 / n if n else 0,
        "third_captured_races_given_top2": third_cap,
        "third_capture_rate_given_top2": third_cap / len(top2_rows) if top2_rows else 0,
        "average_third_candidates": sum(int(r["third_candidate_count"]) for r in rows) / n if n else 0,
        "average_bet_count": sum(int(r["bet_count"]) for r in rows) / n if n else 0,
        "hit_races": hits,
        "hit_rate": hits / n if n else 0,
        "stake_yen": stake,
        "payout_yen": payout,
        "profit_yen": payout - stake,
        "roi": payout / stake if stake else 0,
    }


def main() -> None:
    races = read_csv(DATA_DIR / "races.csv")
    entries = read_csv(DATA_DIR / "entries.csv")
    results = read_csv(DATA_DIR / "results.csv")
    payouts = read_csv(DATA_DIR / "payouts.csv")

    entries_by_race: dict[str, list[dict[str, str]]] = defaultdict(list)
    results_by_race: dict[str, list[dict[str, str]]] = defaultdict(list)
    trifecta_by_race: dict[str, dict[str, int]] = defaultdict(dict)
    races_by_day_track: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)

    for e in entries:
        entries_by_race[e["race_id"]].append(e)
    for r in results:
        results_by_race[r["race_id"]].append(r)
    for p in payouts:
        if p.get("ticket_type") == "3連単" and p.get("status") == "paid" and p.get("combination"):
            try:
                trifecta_by_race[p["race_id"]][p["combination"]] = int(p["payout_yen"])
            except ValueError:
                pass
    for r in races:
        races_by_day_track[(r["race_date"], r["track"])].append(r)

    segment_by_race: dict[str, str] = {}
    ordinal_by_race: dict[str, int] = {}
    group_size_by_race: dict[str, int] = {}
    for group in races_by_day_track.values():
        group.sort(key=lambda r: int(r["race_no"]))
        total = len(group)
        for pos, r in enumerate(group, 1):
            segment_by_race[r["race_id"]] = segment_for(pos, total)
            ordinal_by_race[r["race_id"]] = pos
            group_size_by_race[r["race_id"]] = total

    race_rows: list[dict[str, object]] = []
    bet_rows: list[dict[str, object]] = []
    fallback_count = 0

    for race in sorted(races, key=lambda r: (r["race_date"], r["track"], int(r["race_no"]))):
        race_id = race["race_id"]
        if segment_by_race[race_id] != "後半":
            continue

        es = entries_by_race[race_id]
        main_line = choose_main_line(es)
        if main_line is None:
            continue
        line_id, members = main_line
        a = int(members[0]["car_no"])
        b = int(members[1]["car_no"])
        xs, third_rule, fallback = choose_third_candidates(es, line_id, members)
        fallback_count += int(fallback)

        result_map = {int(r["car_no"]): r for r in results_by_race[race_id]}
        fa = result_map.get(a, {}).get("finish_position", "")
        fb = result_map.get(b, {}).get("finish_position", "")
        mainline_top2 = {str(fa), str(fb)} == {"1", "2"}
        third_finishers = sorted(int(r["car_no"]) for r in results_by_race[race_id] if r.get("finish_position") == "3")
        third_captured = bool(set(third_finishers) & set(xs)) if mainline_top2 else False

        stake = 0
        returned = 0
        hits: list[str] = []
        for x in xs:
            for combo in (f"{a}-{b}-{x}", f"{b}-{a}-{x}"):
                stake += 100
                payout = trifecta_by_race[race_id].get(combo, 0)
                if payout:
                    returned += payout
                    hits.append(combo)
                bet_rows.append({
                    "race_id": race_id,
                    "race_date": race["race_date"],
                    "track": race["track"],
                    "race_no": race["race_no"],
                    "main_line_size": len(members),
                    "third_rule": third_rule,
                    "combination": combo,
                    "stake_yen": 100,
                    "payout_yen": payout,
                    "profit_yen": payout - 100 if payout else -100,
                    "hit": 1 if payout else 0,
                })

        race_rows.append({
            "race_id": race_id,
            "race_date": race["race_date"],
            "half": "H1" if race["race_date"] <= "2025-06-30" else "H2",
            "track": race["track"],
            "race_no": race["race_no"],
            "segment": "後半",
            "segment_ordinal": ordinal_by_race[race_id],
            "s_yosen_count_that_day_track": group_size_by_race[race_id],
            "main_line_id": line_id,
            "main_line_size": len(members),
            "main_line_formation": "-".join(e["car_no"] for e in members),
            "a_car": a,
            "b_car": b,
            "third_rule": third_rule,
            "third_candidates": "-".join(map(str, xs)),
            "third_candidate_count": len(xs),
            "finish_a": fa,
            "finish_b": fb,
            "mainline_top2": 1 if mainline_top2 else 0,
            "actual_third": "-".join(map(str, third_finishers)),
            "third_captured_given_top2": 1 if third_captured else 0,
            "bet_count": len(xs) * 2,
            "stake_yen": stake,
            "payout_yen": returned,
            "profit_yen": returned - stake,
            "hit": 1 if returned > 0 else 0,
            "hit_combinations": "/".join(hits),
        })

    by_mainline_size = {
        "2": summarize([r for r in race_rows if int(r["main_line_size"]) == 2]),
        "3plus": summarize([r for r in race_rows if int(r["main_line_size"]) >= 3]),
    }
    by_half = {
        "H1": summarize([r for r in race_rows if r["half"] == "H1"]),
        "H2": summarize([r for r in race_rows if r["half"] == "H2"]),
    }

    summary = {
        "strategy": "mainline_v2_later",
        "scope": "2025 exact S級予選, 後半 only",
        "rules": {
            "main_line": "same as mainline_v1: 2+ rider line maximizing leader score + second rider score; ties by leader score, then 3+ rider line, then lower line_id",
            "mainline_3plus_third": "main-line third rider + highest-score remaining rider excluding A/B/main3",
            "mainline_2_third": "leader + second rider of strongest other 2+ rider line by their score sum; defensive fallback to top-2 remaining score only if no such rival line exists",
            "bets": "A-B-X and B-A-X for both X candidates, 100 yen each; normally 4 points per race",
            "segment": "within each race_date x track, S級予選 sorted by race_no and split into equal-width terciles; buy 後半 only",
            "no_post_result_filters": True,
        },
        "fallback_races": fallback_count,
        "overall": summarize(race_rows),
        "by_mainline_size": by_mainline_size,
        "by_half": by_half,
    }

    race_fields = [
        "race_id","race_date","half","track","race_no","segment","segment_ordinal","s_yosen_count_that_day_track",
        "main_line_id","main_line_size","main_line_formation","a_car","b_car","third_rule","third_candidates",
        "third_candidate_count","finish_a","finish_b","mainline_top2","actual_third","third_captured_given_top2",
        "bet_count","stake_yen","payout_yen","profit_yen","hit","hit_combinations"
    ]
    bet_fields = ["race_id","race_date","track","race_no","main_line_size","third_rule","combination","stake_yen","payout_yen","profit_yen","hit"]
    write_csv(OUT_DIR / "race_simulation.csv", race_rows, race_fields)
    write_csv(OUT_DIR / "bets.csv", bet_rows, bet_fields)
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
