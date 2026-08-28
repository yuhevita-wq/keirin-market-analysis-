from __future__ import annotations

import csv
import itertools
import json
from collections import defaultdict
from pathlib import Path

from .simulate_mainline_v1 import choose_main_line, num, segment_for

DATA_DIR = Path("data/2025/s_class_yosen")
OUT_DIR = DATA_DIR / "simulations" / "branching_v1"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def fnum(v: str) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def strongest_rival(entries: list[dict[str, str]], main_line_id: int) -> list[dict[str, str]] | None:
    by_line: dict[int, list[dict[str, str]]] = defaultdict(list)
    for e in entries:
        if e.get("line_id", "").isdigit() and e.get("line_position", "").isdigit():
            by_line[int(e["line_id"])].append(e)
    candidates = []
    for lid, members in by_line.items():
        if lid == main_line_id:
            continue
        members = sorted(members, key=lambda e: int(e["line_position"]))
        if len(members) < 2:
            continue
        leader, second = members[0], members[1]
        pair_score = num(leader.get("score", "")) + num(second.get("score", ""))
        leader_score = num(leader.get("score", ""))
        candidates.append(((pair_score, leader_score, -lid), members))
    if not candidates:
        return None
    return max(candidates, key=lambda x: x[0])[1]


def classify(main_members: list[dict[str, str]]) -> tuple[str, float, float, float]:
    a = main_members[0]
    b = main_members[1]
    pair_top3 = fnum(a.get("top3_rate", "")) + fnum(b.get("top3_rate", ""))
    pair_win = fnum(a.get("win_rate", "")) + fnum(b.get("win_rate", ""))
    b_top3 = fnum(b.get("top3_rate", ""))
    if pair_top3 > 106.1 and pair_win > 54.7:
        return "A_mainline", pair_top3, pair_win, b_top3
    if pair_top3 <= 106.1 and b_top3 <= 35.7:
        return "B_rough", pair_top3, pair_win, b_top3
    return "C_middle", pair_top3, pair_win, b_top3


def mainline_bets(entries: list[dict[str, str]], main_members: list[dict[str, str]]) -> list[str]:
    a = int(main_members[0]["car_no"])
    b = int(main_members[1]["car_no"])
    c = int(main_members[2]["car_no"])
    remaining = [e for e in entries if int(e["car_no"]) not in {a, b, c}]
    best_remaining = min(remaining, key=lambda e: (-num(e.get("score", "")), int(e["car_no"])))
    x2 = int(best_remaining["car_no"])
    combos = []
    for x in (c, x2):
        combos.extend((f"{a}-{b}-{x}", f"{b}-{a}-{x}"))
    return combos


def rough_bets(entries: list[dict[str, str]], main_line_id: int, main_members: list[dict[str, str]]) -> list[str]:
    a = int(main_members[0]["car_no"])
    rival = strongest_rival(entries, main_line_id)
    if not rival:
        return []
    r1 = int(rival[0]["car_no"])
    r2 = int(rival[1]["car_no"])
    return ["-".join(map(str, p)) for p in itertools.permutations((a, r1, r2), 3)]


def summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    n = len(rows)
    bet_rows = [r for r in rows if int(r["bet_count"]) > 0]
    stake = sum(int(r["stake_yen"]) for r in rows)
    payout = sum(int(r["payout_yen"]) for r in rows)
    hits = sum(int(r["hit"]) for r in rows)
    return {
        "races": n,
        "bet_races": len(bet_rows),
        "skip_races": n - len(bet_rows),
        "tickets": sum(int(r["bet_count"]) for r in rows),
        "stake_yen": stake,
        "payout_yen": payout,
        "profit_yen": payout - stake,
        "roi": payout / stake if stake else 0,
        "hit_races": hits,
        "hit_rate_all": hits / n if n else 0,
        "hit_rate_bet_races": hits / len(bet_rows) if bet_rows else 0,
    }


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
            combos = rough_bets(es, main_id, members)
            strategy = "A_plus_strongest_rival_box6" if combos else "skip_no_rival"
        else:
            combos = []
            strategy = "skip_middle"

        stake = len(combos) * 100
        returned = 0
        hit_combos = []
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
        "strategy": "branching_v1",
        "scope": "2025 exact S級予選, 後半, main line size >=3",
        "rules": {
            "A_mainline": "pair top3 rate sum > 106.1 and pair win rate sum > 54.7; buy existing mainline 4-point strategy",
            "B_rough": "not A and pair top3 rate sum <= 106.1 and main second top3 rate <= 35.7; select main leader A + strongest rival line leader/second and buy all 6 trifecta permutations",
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
        "warning": "Branch thresholds originated from H1 exploratory analysis. H2 is the cleaner temporal validation view; full-year ROI is not out-of-sample.",
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
