from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from .simulate_branching_v1 import classify, mainline_bets, read_csv, strongest_rival, summarize, write_csv
from .simulate_mainline_v1 import choose_main_line, segment_for

DATA_DIR = Path("data/2025/s_class_yosen")
OUT_DIR = DATA_DIR / "simulations" / "branching_v3_confidence"


def f(row: dict[str, str], key: str) -> float:
    try:
        return float(row.get(key, "") or 0)
    except ValueError:
        return 0.0


def rough_roles(entries: list[dict[str, str]], main_line_id: int, main_members: list[dict[str, str]]) -> tuple[int, int, int, int, int] | None:
    rival = strongest_rival(entries, main_line_id)
    if not rival or len(rival) < 2 or len(main_members) < 3:
        return None
    return (
        int(main_members[0]["car_no"]),
        int(main_members[1]["car_no"]),
        int(main_members[2]["car_no"]),
        int(rival[0]["car_no"]),
        int(rival[1]["car_no"]),
    )


def formation(first: list[int], second: list[int], third: list[int]) -> list[str]:
    combos = {
        f"{x}-{y}-{z}"
        for x in first
        for y in second
        for z in third
        if len({x, y, z}) == 3
    }
    return sorted(combos, key=lambda s: tuple(map(int, s.split("-"))))


def rough_bets(entries: list[dict[str, str]], main_line_id: int, main_members: list[dict[str, str]], narrow_to_a: bool) -> list[str]:
    roles = rough_roles(entries, main_line_id, main_members)
    if roles is None:
        return []
    a, b, m3, r1l, r1b = roles
    first = [a] if narrow_to_a else [a, r1l]
    second = [a, b, r1l, r1b]
    third = [a, b, m3, r1l, r1b]
    return formation(first, second, third)


def main() -> None:
    races = read_csv(DATA_DIR / "races.csv")
    entries = read_csv(DATA_DIR / "entries.csv")
    payouts = read_csv(DATA_DIR / "payouts.csv")

    entries_by_race: dict[str, list[dict[str, str]]] = defaultdict(list)
    trifecta_by_race: dict[str, dict[str, int]] = defaultdict(dict)
    races_by_day_track: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)

    for e in entries:
        entries_by_race[e["race_id"]].append(e)
    for p in payouts:
        if p.get("ticket_type") == "3連単" and p.get("status") == "paid" and p.get("combination"):
            try:
                trifecta_by_race[p["race_id"]][p["combination"]] = int(p["payout_yen"])
            except ValueError:
                pass
    for r in races:
        races_by_day_track[(r["race_date"], r["track"])].append(r)

    segment_by_race: dict[str, str] = {}
    for group in races_by_day_track.values():
        group.sort(key=lambda r: int(r["race_no"]))
        total = len(group)
        for pos, r in enumerate(group, 1):
            segment_by_race[r["race_id"]] = segment_for(pos, total)

    race_rows: list[dict[str, object]] = []
    bet_rows: list[dict[str, object]] = []

    for race in sorted(races, key=lambda r: (r["race_date"], r["track"], int(r["race_no"]))):
        rid = race["race_id"]
        if segment_by_race.get(rid) != "後半":
            continue
        es = entries_by_race[rid]
        main = choose_main_line(es)
        if not main:
            continue
        main_id, members = main
        if len(members) < 3:
            continue

        branch, pair_top3, pair_win, b_top3 = classify(members)
        score_gap = None
        if branch == "A_mainline":
            combos = mainline_bets(es, members)
            strategy = "mainline_4pt"
        elif branch == "B_rough":
            rival = strongest_rival(es, main_id)
            if rival and len(rival) >= 2:
                score_gap = f(members[0], "score") - f(rival[0], "score")
                narrow = score_gap >= 10.0
                combos = rough_bets(es, main_id, members, narrow_to_a=narrow)
                strategy = "rough_9pt_A_confidence" if narrow else "rough_18pt"
            else:
                combos = []
                strategy = "skip_no_rival"
        else:
            combos = []
            strategy = "skip_middle"

        stake = len(combos) * 100
        returned = 0
        hit_combos: list[str] = []
        for combo in combos:
            payout = trifecta_by_race[rid].get(combo, 0)
            returned += payout
            if payout:
                hit_combos.append(combo)
            bet_rows.append({
                "race_id": rid,
                "race_date": race["race_date"],
                "half": "H1" if race["race_date"] <= "2025-06-30" else "H2",
                "track": race["track"],
                "race_no": race["race_no"],
                "branch": branch,
                "strategy": strategy,
                "score_gap_A_vs_R1L": "" if score_gap is None else round(score_gap, 4),
                "combination": combo,
                "stake_yen": 100,
                "payout_yen": payout,
                "profit_yen": payout - 100 if payout else -100,
                "hit": 1 if payout else 0,
            })

        race_rows.append({
            "race_id": rid,
            "race_date": race["race_date"],
            "half": "H1" if race["race_date"] <= "2025-06-30" else "H2",
            "track": race["track"],
            "race_no": race["race_no"],
            "branch": branch,
            "strategy": strategy,
            "main_line_formation": "-".join(e["car_no"] for e in members),
            "pair_top3_sum": pair_top3,
            "pair_win_sum": pair_win,
            "main_second_top3": b_top3,
            "score_gap_A_vs_R1L": "" if score_gap is None else round(score_gap, 4),
            "bet_count": len(combos),
            "stake_yen": stake,
            "payout_yen": returned,
            "profit_yen": returned - stake,
            "hit": 1 if returned else 0,
            "hit_combinations": "/".join(hit_combos),
        })

    strategies = ("mainline_4pt", "rough_9pt_A_confidence", "rough_18pt", "skip_middle", "skip_no_rival")
    summary = {
        "strategy": "branching_v3_confidence",
        "scope": "2025 exact S級予選, 後半, main line size >=3",
        "rules": {
            "A_mainline": "existing mainline branch; 4 points",
            "B_rough_confidence": "if A score - R1L score >=10.0, first=A only; 9 points",
            "B_rough_normal": "if A score - R1L score <10.0, first=A/R1L; 18 points",
            "C_middle": "skip",
            "second": "A/B/R1L/R1B",
            "third": "A/B/M3/R1L/R1B",
            "stake": "100 yen per combination",
            "no_result_or_payout_features": True,
        },
        "overall": summarize(race_rows),
        "by_strategy": {s: summarize([r for r in race_rows if r["strategy"] == s]) for s in strategies},
        "by_half": {h: summarize([r for r in race_rows if r["half"] == h]) for h in ("H1", "H2")},
        "by_half_strategy": {
            h: {s: summarize([r for r in race_rows if r["half"] == h and r["strategy"] == s]) for s in strategies}
            for h in ("H1", "H2")
        },
        "validation": {
            "race_rows": len(race_rows),
            "bet_races": sum(1 for r in race_rows if int(r["bet_count"]) > 0),
            "skip_races": sum(1 for r in race_rows if int(r["bet_count"]) == 0),
            "nine_point_races": sum(1 for r in race_rows if r["strategy"] == "rough_9pt_A_confidence"),
            "eighteen_point_races": sum(1 for r in race_rows if r["strategy"] == "rough_18pt"),
            "four_point_races": sum(1 for r in race_rows if r["strategy"] == "mainline_4pt"),
            "all_9pt_are_9": all(int(r["bet_count"]) == 9 for r in race_rows if r["strategy"] == "rough_9pt_A_confidence"),
            "all_18pt_are_18": all(int(r["bet_count"]) == 18 for r in race_rows if r["strategy"] == "rough_18pt"),
            "all_4pt_are_4": all(int(r["bet_count"]) == 4 for r in race_rows if r["strategy"] == "mainline_4pt"),
        },
        "warning": "The >=10 score-gap confidence rule was discovered within 2025 exploratory analysis. H2 is not a fresh untouched sample for the threshold itself. Treat ROI as exploratory and validate on another year before operational adoption.",
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    race_fields = [
        "race_id","race_date","half","track","race_no","branch","strategy","main_line_formation",
        "pair_top3_sum","pair_win_sum","main_second_top3","score_gap_A_vs_R1L","bet_count","stake_yen","payout_yen","profit_yen","hit","hit_combinations"
    ]
    bet_fields = [
        "race_id","race_date","half","track","race_no","branch","strategy","score_gap_A_vs_R1L","combination","stake_yen","payout_yen","profit_yen","hit"
    ]
    write_csv(OUT_DIR / "race_simulation.csv", race_rows, race_fields)
    write_csv(OUT_DIR / "bets.csv", bet_rows, bet_fields)
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    assert len(race_rows) == 353
    assert summary["validation"]["all_9pt_are_9"]
    assert summary["validation"]["all_18pt_are_18"]
    assert summary["validation"]["all_4pt_are_4"]
    assert summary["validation"]["nine_point_races"] == 8
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
