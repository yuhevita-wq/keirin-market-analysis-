from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

DATA_DIR = Path("data/2025/s_class_yosen")
OUT_DIR = DATA_DIR / "simulations" / "mainline_v1"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def num(v: str) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("-inf")


def segment_for(position: int, total: int) -> str:
    # Equal-width terciles by ordinal position within the day's S級予選 card.
    idx = min(2, ((position - 1) * 3) // total)
    return ("前半", "中盤", "後半")[idx]


def choose_main_line(entries: list[dict[str, str]]) -> tuple[int, list[dict[str, str]]] | None:
    by_line: dict[int, list[dict[str, str]]] = defaultdict(list)
    for e in entries:
        if e.get("line_id", "").isdigit() and e.get("line_position", "").isdigit():
            by_line[int(e["line_id"])].append(e)

    candidates: list[tuple[tuple[float, float, int, int], int, list[dict[str, str]]]] = []
    for line_id, members in by_line.items():
        members = sorted(members, key=lambda e: int(e["line_position"]))
        if len(members) < 2:
            continue
        leader, second = members[0], members[1]
        pair_score = num(leader.get("score", "")) + num(second.get("score", ""))
        leader_score = num(leader.get("score", ""))
        # tie-break: leader score, then 3+ rider line, then stable lower line_id
        key = (pair_score, leader_score, 1 if len(members) >= 3 else 0, -line_id)
        candidates.append((key, line_id, members))
    if not candidates:
        return None
    _, line_id, members = max(candidates, key=lambda x: x[0])
    return line_id, members


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
    for key, group in races_by_day_track.items():
        group.sort(key=lambda r: int(r["race_no"]))
        total = len(group)
        for pos, r in enumerate(group, 1):
            segment_by_race[r["race_id"]] = segment_for(pos, total)
            ordinal_by_race[r["race_id"]] = pos
            group_size_by_race[r["race_id"]] = total

    race_rows: list[dict[str, object]] = []
    bet_rows: list[dict[str, object]] = []

    for race in sorted(races, key=lambda r: (r["race_date"], r["track"], int(r["race_no"]))):
        race_id = race["race_id"]
        es = entries_by_race[race_id]
        main_line = choose_main_line(es)
        if main_line is None:
            continue
        line_id, members = main_line
        a = int(members[0]["car_no"])
        b = int(members[1]["car_no"])

        third_candidates: set[int] = set()
        # Main-line third rider.
        if len(members) >= 3:
            third_candidates.add(int(members[2]["car_no"]))
        # Leaders of every other published line, including singletons only when they have a line_id.
        for e in es:
            if e.get("line_id", "").isdigit() and e.get("line_position") == "1" and int(e["line_id"]) != line_id:
                third_candidates.add(int(e["car_no"]))
        # Overall top four by pre-race race score.
        scored = sorted(es, key=lambda e: (-num(e.get("score", "")), int(e["car_no"])))
        for e in scored[:4]:
            third_candidates.add(int(e["car_no"]))
        third_candidates.discard(a)
        third_candidates.discard(b)
        xs = sorted(third_candidates)

        result_map = {int(r["car_no"]): r for r in results_by_race[race_id]}
        fa = result_map.get(a, {}).get("finish_position", "")
        fb = result_map.get(b, {}).get("finish_position", "")
        mainline_top2 = {str(fa), str(fb)} == {"1", "2"}
        third_finishers = sorted(
            int(r["car_no"]) for r in results_by_race[race_id] if r.get("finish_position") == "3"
        )
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
                    "segment": segment_by_race[race_id],
                    "combination": combo,
                    "stake_yen": 100,
                    "payout_yen": payout,
                    "profit_yen": payout - 100 if payout else -100,
                    "hit": 1 if payout else 0,
                })

        returned_profit = returned - stake
        race_rows.append({
            "race_id": race_id,
            "race_date": race["race_date"],
            "track": race["track"],
            "race_no": race["race_no"],
            "segment": segment_by_race[race_id],
            "segment_ordinal": ordinal_by_race[race_id],
            "s_yosen_count_that_day_track": group_size_by_race[race_id],
            "main_line_id": line_id,
            "main_line_formation": "-".join(e["car_no"] for e in members),
            "a_car": a,
            "b_car": b,
            "a_score": members[0].get("score", ""),
            "b_score": members[1].get("score", ""),
            "pair_score_sum": num(members[0].get("score", "")) + num(members[1].get("score", "")),
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
            "profit_yen": returned_profit,
            "hit": 1 if returned > 0 else 0,
            "hit_combinations": "/".join(hits),
        })

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

    by_segment = {seg: summarize([r for r in race_rows if r["segment"] == seg]) for seg in ("前半", "中盤", "後半")}
    summary = {
        "strategy": "mainline_v1",
        "rules": {
            "main_line": "2+ rider line maximizing leader score + second rider score; ties: higher leader score, then 3+ rider line, then lower line_id",
            "third_candidates": "main-line third rider + leaders of other lines + overall top-4 score riders; remove A/B and duplicates",
            "bets": "A-B-X and B-A-X, 100 yen each",
            "segment": "within each race_date x track, sort S級予選 by race_no and split ordinal positions into equal-width terciles",
            "no_post_result_filters": True,
        },
        "overall": summarize(race_rows),
        "by_segment": by_segment,
    }

    race_fields = [
        "race_id","race_date","track","race_no","segment","segment_ordinal","s_yosen_count_that_day_track",
        "main_line_id","main_line_formation","a_car","b_car","a_score","b_score","pair_score_sum",
        "third_candidates","third_candidate_count","finish_a","finish_b","mainline_top2","actual_third",
        "third_captured_given_top2","bet_count","stake_yen","payout_yen","profit_yen","hit","hit_combinations"
    ]
    bet_fields = ["race_id","race_date","track","race_no","segment","combination","stake_yen","payout_yen","profit_yen","hit"]
    write_csv(OUT_DIR / "race_simulation.csv", race_rows, race_fields)
    write_csv(OUT_DIR / "bets.csv", bet_rows, bet_fields)
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
