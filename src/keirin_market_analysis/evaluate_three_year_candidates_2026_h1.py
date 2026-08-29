from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

from .simulate_mainline_v1 import choose_main_line, segment_for
from .simulate_early_candidate_v0 import strongest_rival
from .search_three_year_conditions_v1 import feature_row, formations

DATA = Path("data/2026_h1/s_class_yosen")
MANIFEST = Path("data/audits/three_year_candidates_frozen_for_2026_h1.json")
OUT = Path("data/audits/three_year_candidates_2026_h1_oos.json")
DETAILS = Path("data/audits/three_year_candidates_2026_h1_oos_details.csv")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def strategy_for(segment: str, row: dict[str, object]) -> tuple[str, str] | None:
    if segment == "前半":
        if float(row["pair_score_gap"]) <= 6 and float(row["pair_win_gap"]) <= 0:
            return "early", "MIX2"
        return None
    if segment == "中盤":
        if float(row["rival_top3"]) >= 80 and float(row["pair_top3_gap"]) >= 10:
            return "middle", "MIX2"
        return None
    if segment == "後半":
        if float(row["r1l_score"]) >= 100 and float(row["main_top2"]) <= 70:
            return "late", "MAIN4_X"
        return None
    return None


def blank_stats() -> dict[str, object]:
    return {
        "races": 0,
        "hits": 0,
        "stake_yen": 0,
        "payout_yen": 0,
        "profit_yen": 0,
        "roi": 0.0,
        "hit_rate": 0.0,
        "max_hit_payout_yen": 0,
        "top1_payout_share": 0.0,
        "max_losing_streak": 0,
    }


def summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    if not rows:
        return blank_stats()
    stake = sum(int(r["stake_yen"]) for r in rows)
    payout = sum(int(r["payout_yen"]) for r in rows)
    hits = sum(1 for r in rows if int(r["payout_yen"]) > 0)
    hit_pays = [int(r["payout_yen"]) for r in rows if int(r["payout_yen"]) > 0]
    losing = best_losing = 0
    for r in sorted(rows, key=lambda x: (str(x["race_date"]), str(x["track"]), int(x["race_no"]))):
        if int(r["payout_yen"]) > 0:
            losing = 0
        else:
            losing += 1
            best_losing = max(best_losing, losing)
    return {
        "races": len(rows),
        "hits": hits,
        "stake_yen": stake,
        "payout_yen": payout,
        "profit_yen": payout - stake,
        "roi": payout / stake if stake else 0.0,
        "hit_rate": hits / len(rows),
        "max_hit_payout_yen": max(hit_pays) if hit_pays else 0,
        "top1_payout_share": max(hit_pays) / payout if payout and hit_pays else 0.0,
        "max_losing_streak": best_losing,
    }


def main() -> int:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert manifest["status"] == "FROZEN_FOR_2026_H1_FORWARD_OOS"
    assert manifest["strategies"]["early"]["formation"] == "MIX2"
    assert manifest["strategies"]["middle"]["formation"] == "MIX2"
    assert manifest["strategies"]["late"]["formation"] == "MAIN4_X"

    base = json.loads((DATA / "summary.json").read_text(encoding="utf-8"))
    line = json.loads((DATA / "line_summary.json").read_text(encoding="utf-8"))
    result = json.loads((DATA / "result_summary.json").read_text(encoding="utf-8"))
    assert base["start_date"] == "2026-01-01" and base["end_date"] == "2026-06-30"
    assert base["failures"] == 0
    assert line["complete_line_races"] == base["parsed_races"]
    assert line["line_parse_failures"] == 0
    assert result["complete_result_races"] == base["parsed_races"]
    assert result["complete_payout_races"] == base["parsed_races"]
    assert result["parse_failures"] == 0

    races = read_csv(DATA / "races.csv")
    entries = read_csv(DATA / "entries.csv")
    payouts = read_csv(DATA / "payouts.csv")

    entries_by_race: dict[str, list[dict[str, str]]] = defaultdict(list)
    for e in entries:
        entries_by_race[e["race_id"]].append(e)

    tri: dict[str, dict[str, int]] = defaultdict(dict)
    for p in payouts:
        if p.get("ticket_type") == "3連単" and p.get("status") == "paid" and p.get("combination"):
            tri[p["race_id"]][p["combination"]] = int(p["payout_yen"])

    groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for r in races:
        groups[(r["race_date"], r["track"])].append(r)
    segment_by_race: dict[str, str] = {}
    for group in groups.values():
        group.sort(key=lambda r: int(r["race_no"]))
        for i, r in enumerate(group, 1):
            segment_by_race[r["race_id"]] = segment_for(i, len(group))

    structural_counts = {"前半": 0, "中盤": 0, "後半": 0}
    detail_rows: list[dict[str, object]] = []

    for race in sorted(races, key=lambda r: (r["race_date"], r["track"], int(r["race_no"]))):
        rid = race["race_id"]
        segment = segment_by_race.get(rid)
        if segment not in structural_counts:
            continue
        es = entries_by_race[rid]
        chosen = choose_main_line(es)
        if not chosen:
            continue
        main_id, main = chosen
        if len(main) < 3:
            continue
        rival = strongest_rival(es, main_id)
        if not rival or len(rival) < 2:
            continue
        structural_counts[segment] += 1
        row = feature_row(race, es, main_id, main, rival)
        selected = strategy_for(segment, row)
        if not selected:
            continue
        strategy, form = selected
        all_forms = formations(row, es)
        bets = all_forms[form]
        paid = tri.get(rid, {})
        race_payout = sum(paid.get(b, 0) for b in bets)
        detail_rows.append({
            "strategy": strategy,
            "segment": segment,
            "race_id": rid,
            "race_date": race["race_date"],
            "month": race["race_date"][:7],
            "track": race["track"],
            "race_no": int(race["race_no"]),
            "formation": form,
            "bets": "|".join(bets),
            "stake_yen": 100 * len(bets),
            "payout_yen": race_payout,
            "profit_yen": race_payout - 100 * len(bets),
            "hit": int(race_payout > 0),
            "A": row["A"], "B": row["B"], "M3": row["M3"], "R1L": row["R1L"], "R1B": row["R1B"],
            "pair_score_gap": row["pair_score_gap"],
            "pair_win_gap": row["pair_win_gap"],
            "rival_top3": row["rival_top3"],
            "pair_top3_gap": row["pair_top3_gap"],
            "r1l_score": row["r1l_score"],
            "main_top2": row["main_top2"],
        })

    strategy_rows = {s: [r for r in detail_rows if r["strategy"] == s] for s in ("early", "middle", "late")}
    monthly: dict[str, dict[str, object]] = {}
    for month in [f"2026-{m:02d}" for m in range(1, 7)]:
        monthly[month] = {
            s: summarize([r for r in strategy_rows[s] if r["month"] == month])
            for s in ("early", "middle", "late")
        }
        monthly[month]["combined"] = summarize([r for r in detail_rows if r["month"] == month])

    output = {
        "status": "FORWARD_OOS_EVALUATED_WITHOUT_RETUNING",
        "development_window": "2023-2025",
        "evaluation_window": "2026-01-01..2026-06-30",
        "dataset_races": len(races),
        "structural_eligible_counts": structural_counts,
        "strategies": {s: summarize(strategy_rows[s]) for s in ("early", "middle", "late")},
        "combined": summarize(detail_rows),
        "monthly": monthly,
        "frozen_manifest": str(MANIFEST),
        "note": "No 2026 H1 outcome was used to alter thresholds, formations, role definitions, or stakes. Results are evaluation only."
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    fields = [
        "strategy", "segment", "race_id", "race_date", "month", "track", "race_no", "formation", "bets",
        "stake_yen", "payout_yen", "profit_yen", "hit", "A", "B", "M3", "R1L", "R1B",
        "pair_score_gap", "pair_win_gap", "rival_top3", "pair_top3_gap", "r1l_score", "main_top2"
    ]
    with DETAILS.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(detail_rows)

    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
