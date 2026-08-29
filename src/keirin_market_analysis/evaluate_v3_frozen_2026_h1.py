from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

from .search_three_year_conditions_v1 import choose_main_line, strongest_rival, feature_row, formations, segment_for

SPEC = Path("data/strategy_specs/v3_proposed_pre2026.json")
DATA = Path("data/2026_h1/s_class_yosen")
OUT = DATA / "v3_frozen_evaluation.json"


def read_csv(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def load_2026_h1():
    races = read_csv(DATA / "races.csv")
    entries = read_csv(DATA / "entries.csv")
    payouts = read_csv(DATA / "payouts.csv")

    eb = defaultdict(list)
    for e in entries:
        eb[e["race_id"]].append(e)

    tri = defaultdict(dict)
    for p in payouts:
        if p.get("ticket_type") == "3連単" and p.get("status") == "paid" and p.get("combination"):
            try:
                tri[p["race_id"]][p["combination"]] = int(p["payout_yen"])
            except (TypeError, ValueError):
                pass

    groups = defaultdict(list)
    for r in races:
        groups[(r["race_date"], r["track"])].append(r)
    seg = {}
    for g in groups.values():
        g.sort(key=lambda r: int(r["race_no"]))
        for i, r in enumerate(g, 1):
            seg[r["race_id"]] = segment_for(i, len(g))
    return races, eb, tri, seg


def exact_spec_assertions(spec):
    assert spec["status"] == "PROPOSED_AWAITING_USER_GO_BEFORE_2026_EVALUATION"
    assert spec["development_years"] == [2023, 2024, 2025]
    assert spec["evaluation_year"] == 2026
    assert spec["evaluation_locked"] is True

    early = spec["segments"]["early"]
    middle = spec["segments"]["middle"]
    late = spec["segments"]["late"]

    assert early["conditions"] == ["A+B top2 rate sum >= 70", "A score - R1L score <= 2"]
    assert early["formation"] == "MIX2"
    assert early["bets_yen_100_each"] == ["A-B-R1B", "R1L-R1B-B"]
    assert early["stake_yen_per_race"] == 200

    assert middle["conditions"] == ["A+B top2 rate sum >= 80", "R1L+R1B top3 rate sum >= 80"]
    assert middle["formation"] == "MIX2"
    assert middle["bets_yen_100_each"] == ["A-B-R1B", "R1L-R1B-B"]
    assert middle["stake_yen_per_race"] == 200

    assert late["conditions"] == ["R1L score >= 100", "A+B top2 rate sum <= 70"]
    assert late["formation"] == "MAIN4_X"
    assert late["bets_yen_100_each"] == ["A-B-M3", "B-A-M3", "A-B-X", "B-A-X"]
    assert late["stake_yen_per_race"] == 400


def qualifies(key: str, row: dict) -> bool:
    if key == "early":
        return row["main_top2"] >= 70 and row["leader_score_gap"] <= 2
    if key == "middle":
        return row["main_top2"] >= 80 and row["rival_top3"] >= 80
    if key == "late":
        return row["r1l_score"] >= 100 and row["main_top2"] <= 70
    raise KeyError(key)


def summarize(rows):
    stake = sum(r["stake_yen"] for r in rows)
    payout = sum(r["payout_yen"] for r in rows)
    hits = sum(1 for r in rows if r["payout_yen"] > 0)
    hit_pays = [r["payout_yen"] for r in rows if r["payout_yen"] > 0]
    monthly = {}
    for month in range(1, 7):
        mr = [r for r in rows if int(r["race_date"][5:7]) == month]
        ms = sum(r["stake_yen"] for r in mr)
        mp = sum(r["payout_yen"] for r in mr)
        mh = sum(1 for r in mr if r["payout_yen"] > 0)
        monthly[f"2026-{month:02d}"] = {
            "races": len(mr), "hits": mh, "stake_yen": ms, "payout_yen": mp,
            "profit_yen": mp-ms, "roi": mp/ms if ms else 0.0,
        }
    return {
        "races": len(rows),
        "hits": hits,
        "hit_rate": hits/len(rows) if rows else 0.0,
        "stake_yen": stake,
        "payout_yen": payout,
        "profit_yen": payout-stake,
        "roi": payout/stake if stake else 0.0,
        "max_hit_payout_yen": max(hit_pays) if hit_pays else 0,
        "top1_payout_share": max(hit_pays)/payout if payout and hit_pays else 0.0,
        "monthly": monthly,
    }


def main():
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    exact_spec_assertions(spec)

    races, eb, tri, seg = load_2026_h1()
    buckets = {"early": [], "middle": [], "late": []}
    seg_name = {"early": "前半", "middle": "中盤", "late": "後半"}
    form_name = {"early": "MIX2", "middle": "MIX2", "late": "MAIN4_X"}

    structural_base = {"early": 0, "middle": 0, "late": 0}
    for race in races:
        rid = race["race_id"]
        s = seg.get(rid)
        if s not in ("前半", "中盤", "後半"):
            continue
        chosen = choose_main_line(eb[rid])
        if not chosen:
            continue
        main_id, main = chosen
        if len(main) < 3:
            continue
        rival = strongest_rival(eb[rid], main_id)
        if not rival or len(rival) < 2:
            continue

        row = feature_row(race, eb[rid], main_id, main, rival)
        row_forms = formations(row, eb[rid])

        for key in ("early", "middle", "late"):
            if s != seg_name[key]:
                continue
            structural_base[key] += 1
            if not qualifies(key, row):
                continue
            bets = row_forms[form_name[key]]
            payout = sum(tri[rid].get(b, 0) for b in bets)
            buckets[key].append({
                "race_id": rid,
                "race_date": race["race_date"],
                "track": race["track"],
                "race_no": int(race["race_no"]),
                "bets": bets,
                "stake_yen": 100 * len(bets),
                "payout_yen": payout,
                "hit": payout > 0,
            })

    sections = {}
    all_rows = []
    for key in ("early", "middle", "late"):
        stats = summarize(buckets[key])
        stats["eligible_base"] = structural_base[key]
        stats["purchase_rate"] = stats["races"] / structural_base[key] if structural_base[key] else 0.0
        sections[key] = stats
        all_rows.extend(buckets[key])

    combined = summarize(all_rows)
    combined["eligible_base"] = sum(structural_base.values())
    combined["purchase_rate"] = combined["races"] / combined["eligible_base"] if combined["eligible_base"] else 0.0

    out = {
        "status": "FROZEN_V3_2026_H1_EVALUATION_COMPLETE_AFTER_USER_GO",
        "strategy_spec": str(SPEC),
        "strategy_spec_asserted_unchanged": True,
        "development_years": [2023, 2024, 2025],
        "evaluation_period": "2026-01-01 through 2026-06-30",
        "dataset_races": len(races),
        "no_threshold_or_formation_changes_after_go": True,
        "sections": sections,
        "combined": combined,
    }
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
