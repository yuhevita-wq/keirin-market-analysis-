from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

from .simulate_branching_v1 import classify, mainline_bets, read_csv, strongest_rival, summarize, write_csv
from .simulate_mainline_v1 import choose_main_line, segment_for

DATA_DIR = Path("data/2025/s_class_yosen")
OUT_DIR = DATA_DIR / "simulations" / "branching_v2"


def rough_bets_27(entries: list[dict[str, str]], main_line_id: int, main_members: list[dict[str, str]]) -> list[str]:
    a = int(main_members[0]["car_no"])
    b = int(main_members[1]["car_no"])
    m3 = int(main_members[2]["car_no"])
    rival = strongest_rival(entries, main_line_id)
    if not rival:
        return []
    r1l = int(rival[0]["car_no"])
    r1b = int(rival[1]["car_no"])

    first = [a, b, r1l]
    second = [a, b, r1l, r1b]
    third = [a, b, m3, r1l, r1b]
    combos = {
        f"{x}-{y}-{z}"
        for x in first
        for y in second
        for z in third
        if len({x, y, z}) == 3
    }
    return sorted(combos, key=lambda s: tuple(map(int, s.split("-"))))


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
        if branch == "A_mainline":
            combos = mainline_bets(es, members)
            strategy = "mainline_4pt"
        elif branch == "B_rough":
            combos = rough_bets_27(es, main_id, members)
            strategy = "rough_27pt" if combos else "skip_no_rival"
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
            "bet_count": len(combos),
            "stake_yen": stake,
            "payout_yen": returned,
            "profit_yen": returned - stake,
            "hit": 1 if returned else 0,
            "hit_combinations": "/".join(hit_combos),
        })

    branches = ("A_mainline", "B_rough", "C_middle")
    summary = {
        "strategy": "branching_v2",
        "scope": "2025 exact S級予選, 後半, main line size >=3",
        "rules": {
            "A_mainline": "pair top3 rate sum > 106.1 and pair win rate sum > 54.7; existing 4-point mainline strategy",
            "B_rough": "pair top3 rate sum <= 106.1 and main second top3 rate <= 35.7; 1st=A/B/R1L, 2nd=A/B/R1L/R1B, 3rd=A/B/M3/R1L/R1B; 27 points",
            "C_middle": "all remaining races; skip",
            "stake": "100 yen per combination",
            "no_result_or_payout_features": True,
        },
        "overall": summarize(race_rows),
        "by_branch": {b: summarize([r for r in race_rows if r["branch"] == b]) for b in branches},
        "by_half": {h: summarize([r for r in race_rows if r["half"] == h]) for h in ("H1", "H2")},
        "by_half_branch": {
            h: {b: summarize([r for r in race_rows if r["half"] == h and r["branch"] == b]) for b in branches}
            for h in ("H1", "H2")
        },
        "warning": "Branch thresholds originated from H1 exploratory analysis. H2 remains the cleaner temporal validation view. Rough 27-point formation was selected from structural 2025 analysis, so full-year ROI is exploratory, not out-of-sample validation.",
    }

    race_fields = [
        "race_id","race_date","half","track","race_no","branch","strategy","main_line_formation",
        "pair_top3_sum","pair_win_sum","main_second_top3","bet_count","stake_yen","payout_yen","profit_yen","hit","hit_combinations"
    ]
    bet_fields = [
        "race_id","race_date","half","track","race_no","branch","strategy","combination","stake_yen","payout_yen","profit_yen","hit"
    ]
    write_csv(OUT_DIR / "race_simulation.csv", race_rows, race_fields)
    write_csv(OUT_DIR / "bets.csv", bet_rows, bet_fields)
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
