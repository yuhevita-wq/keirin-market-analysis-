from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from .search_three_year_conditions_v1 import choose_main_line, strongest_rival, feature_row, segment_for, num

YEARS = (2023, 2024, 2025)
OUT = Path("data/audits/high_payout_structures_2023_2025_v1.json")
THRESHOLDS = (5000, 10000, 20000)


def read_csv(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def f(v):
    x = num(v)
    return 0.0 if x == float("-inf") else float(x)


def load_year(year: int):
    d = Path(f"data/{year}/s_class_yosen")
    races = read_csv(d / "races.csv")
    entries = read_csv(d / "entries.csv")
    payouts = read_csv(d / "payouts.csv")

    eb = defaultdict(list)
    for e in entries:
        eb[e["race_id"]].append(e)

    win3 = {}
    for p in payouts:
        if p.get("ticket_type") == "3連単" and p.get("status") == "paid" and p.get("combination"):
            try:
                win3[p["race_id"]] = (p["combination"], int(p["payout_yen"]))
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
    return races, eb, win3, seg


def exclusive_roles(es, main, rival):
    roles = {}
    ordered = [
        ("A", main[0]), ("B", main[1]), ("M3", main[2]),
        ("R1L", rival[0]), ("R1B", rival[1]),
    ]
    used = set()
    for name, e in ordered:
        car = int(e["car_no"])
        if car not in used:
            roles[car] = name
            used.add(car)

    rest = [e for e in es if int(e["car_no"]) not in used]
    rest.sort(key=lambda e: (-f(e.get("score")), int(e["car_no"])))
    for i, e in enumerate(rest, 1):
        roles[int(e["car_no"])] = f"O{i}"
    return roles


def parse_combination(s: str):
    return tuple(int(x) for x in s.split("-"))


def motif(role_order):
    r = set(role_order)
    first, second, third = role_order
    motifs = []
    if first in {"R1L", "R1B"}: motifs.append("RIVAL_HEAD")
    if first in {"A", "B", "M3"}: motifs.append("MAIN_HEAD")
    if first.startswith("O"): motifs.append("OUTSIDER_HEAD")
    if {first, second} == {"R1L", "R1B"}: motifs.append("RIVAL_PAIR_TOP2")
    if first in {"R1L", "R1B"} and second in {"A", "B", "M3"}: motifs.append("RIVAL_THEN_MAIN")
    if first in {"A", "B"} and second in {"R1L", "R1B"}: motifs.append("MAIN_THEN_RIVAL")
    if any(x.startswith("O") for x in role_order): motifs.append("OUTSIDER_INCLUDED")
    if sum(x in {"A", "B", "M3"} for x in role_order) <= 1: motifs.append("MAIN_COLLAPSE")
    if sum(x in {"R1L", "R1B"} for x in role_order) == 2: motifs.append("BOTH_RIVALS_INCLUDED")
    return motifs


def summarize_records(records):
    payouts = [r["payout_yen"] for r in records]
    if not payouts:
        return {"races": 0}
    ps = sorted(payouts)
    return {
        "races": len(records),
        "median_payout_yen": ps[len(ps)//2],
        "mean_payout_yen": sum(ps)/len(ps),
        "max_payout_yen": max(ps),
        **{f"ge_{t}_count": sum(p >= t for p in ps) for t in THRESHOLDS},
        **{f"ge_{t}_rate": sum(p >= t for p in ps)/len(ps) for t in THRESHOLDS},
    }


def main():
    all_records = []
    structural_counts = {}
    for year in YEARS:
        races, eb, win3, seg = load_year(year)
        yc = Counter()
        for race in races:
            rid = race["race_id"]
            s = seg.get(rid)
            if s not in ("前半", "中盤", "後半") or rid not in win3:
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
            roles = exclusive_roles(eb[rid], main, rival)
            combo, payout = win3[rid]
            cars = parse_combination(combo)
            if any(c not in roles for c in cars):
                continue
            role_order = tuple(roles[c] for c in cars)
            row = feature_row(race, eb[rid], main_id, main, rival)
            rec = {
                "year": year,
                "segment": s,
                "race_id": rid,
                "race_date": race["race_date"],
                "track": race["track"],
                "race_no": int(race["race_no"]),
                "combination": combo,
                "payout_yen": payout,
                "role_order": "-".join(role_order),
                "motifs": motif(role_order),
                "features": {k: row[k] for k in (
                    "a_score","b_score","r1l_score","r1b_score","main_score","rival_score",
                    "main_win","rival_win","main_top2","rival_top2","main_top3","rival_top3",
                    "leader_score_gap","second_score_gap","pair_score_gap","pair_win_gap",
                    "pair_top2_gap","pair_top3_gap"
                )},
            }
            all_records.append(rec)
            yc[s] += 1
        structural_counts[str(year)] = dict(yc)

    by_segment = {}
    for s in ("前半", "中盤", "後半"):
        rs = [r for r in all_records if r["segment"] == s]
        perm = defaultdict(list)
        mot = defaultdict(list)
        for r in rs:
            perm[r["role_order"]].append(r)
            for m in r["motifs"]:
                mot[m].append(r)
        perm_rows = []
        for name, x in perm.items():
            z = summarize_records(x)
            z["role_order"] = name
            z["year_counts"] = {str(y): sum(r["year"] == y for r in x) for y in YEARS}
            perm_rows.append(z)
        perm_rows.sort(key=lambda z: (z.get("ge_10000_count",0), z.get("ge_5000_count",0), z.get("races",0)), reverse=True)

        motif_rows = []
        for name, x in mot.items():
            z = summarize_records(x)
            z["motif"] = name
            z["year_counts"] = {str(y): sum(r["year"] == y for r in x) for y in YEARS}
            motif_rows.append(z)
        motif_rows.sort(key=lambda z: (z.get("ge_10000_rate",0), z.get("ge_5000_rate",0), z.get("races",0)), reverse=True)

        high = {}
        for t in THRESHOLDS:
            hr = [r for r in rs if r["payout_yen"] >= t]
            top_perm = Counter(r["role_order"] for r in hr).most_common(20)
            top_motif = Counter(m for r in hr for m in r["motifs"]).most_common()
            high[str(t)] = {
                "count": len(hr),
                "rate": len(hr)/len(rs) if rs else 0.0,
                "year_counts": {str(y): sum(r["year"] == y for r in hr) for y in YEARS},
                "top_role_orders": top_perm,
                "motifs": top_motif,
            }

        by_segment[s] = {
            "overall": summarize_records(rs),
            "high_payout": high,
            "role_order_stats": perm_rows[:60],
            "motif_stats": motif_rows,
        }

    out = {
        "status": "HIGH_PAYOUT_STRUCTURE_RESEARCH_2023_2025_ONLY",
        "years_read": list(YEARS),
        "evaluation_year_2026_used": False,
        "thresholds_yen_per_100": list(THRESHOLDS),
        "structural_counts": structural_counts,
        "segments": by_segment,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({s:{"overall":by_segment[s]["overall"],"high":by_segment[s]["high_payout"]} for s in by_segment}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
