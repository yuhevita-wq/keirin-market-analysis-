from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOTS = [
    ("2024", Path("data/2024/s_class_f1_all_parts")),
    ("2025", Path("data/2025/s_class_f1_all_parts")),
    ("2026H1", Path("data/2026_h1/s_class_f1_all")),
]
TARGET_RACE_TYPE = "Ｓ級一般"


def _i(v):
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return None


def load_rows(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        yield from csv.DictReader(f)


def rider_role(e, winning_line_id):
    if e.get("line_id") == winning_line_id:
        pos = _i(e.get("line_position"))
        if pos == 1:
            return "自ライン先頭①"
        if pos == 2:
            return "自ライン番手②"
        if pos == 3:
            return "自ライン3番手③"
        return f"自ライン{pos}番手"

    size = _i(e.get("line_size"))
    pos = _i(e.get("line_position"))
    if size == 1:
        return "単騎"
    if pos == 1:
        return "別線先頭①"
    if pos == 2:
        return "別線番手②"
    if pos == 3:
        return "別線3番手③"
    return f"別線{pos}番手"


def empty_bucket():
    return {
        "s_general_races": 0,
        "three_line_races": 0,
        "bante_win_races": 0,
        "finish_structure": Counter(),
        "ordered_top3_roles": Counter(),
        "winner_method": Counter(),
        "leader_finish": Counter(),
        "third_finish": Counter(),
        "outside_pair_relation_when_bante_only": Counter(),
    }


def pct(n, d):
    return round(100.0 * n / d, 2) if d else 0.0


def analyze_scope(only_7=False):
    buckets = {"ALL": empty_bucket()}
    for year, _ in ROOTS:
        buckets[year] = empty_bucket()

    seen_races = set()

    for year, root in ROOTS:
        if not root.exists():
            continue
        for entries_path in sorted(root.rglob("entries.csv")):
            results_path = entries_path.with_name("results.csv")
            if not results_path.exists():
                continue

            entries_by_race = defaultdict(dict)
            race_meta = {}
            for e in load_rows(entries_path):
                if e.get("race_type") != TARGET_RACE_TYPE:
                    continue
                rid = e.get("race_id")
                car = _i(e.get("car_no"))
                if not rid or car is None:
                    continue
                entries_by_race[rid][car] = e
                race_meta[rid] = e

            if not entries_by_race:
                continue

            results_by_race = defaultdict(dict)
            for r in load_rows(results_path):
                rid = r.get("race_id")
                if rid not in entries_by_race:
                    continue
                car = _i(r.get("car_no"))
                fp = _i(r.get("finish_position"))
                if car is None or fp is None:
                    continue
                results_by_race[rid][fp] = r

            for rid, cars in entries_by_race.items():
                dedupe_key = (year, rid, only_7)
                if dedupe_key in seen_races:
                    continue
                seen_races.add(dedupe_key)

                entry_count = len(cars)
                if only_7 and entry_count != 7:
                    continue

                targets = [buckets[year], buckets["ALL"]]
                for b in targets:
                    b["s_general_races"] += 1

                three_line_bantes = [
                    car for car, e in cars.items()
                    if _i(e.get("line_size")) == 3 and _i(e.get("line_position")) == 2
                ]
                if not three_line_bantes:
                    continue
                for b in targets:
                    b["three_line_races"] += 1

                top = results_by_race.get(rid, {})
                if not all(k in top for k in (1, 2, 3)):
                    continue
                winner_car = _i(top[1].get("car_no"))
                if winner_car not in three_line_bantes:
                    continue

                winner_entry = cars[winner_car]
                win_line = winner_entry.get("line_id")
                same_line = {
                    _i(e.get("line_position")): car
                    for car, e in cars.items()
                    if e.get("line_id") == win_line and _i(e.get("line_size")) == 3
                }
                if not all(p in same_line for p in (1, 2, 3)):
                    continue

                for b in targets:
                    b["bante_win_races"] += 1
                    method = (top[1].get("winning_method") or "不明").strip() or "不明"
                    b["winner_method"][method] += 1

                top3_cars = [_i(top[p].get("car_no")) for p in (1, 2, 3)]
                own_leader = same_line[1]
                own_third = same_line[3]

                if own_leader in top3_cars and own_third in top3_cars:
                    structure = "①②③丸残り"
                elif own_leader in top3_cars:
                    structure = "①②残り＋別線1"
                elif own_third in top3_cars:
                    structure = "②③残り＋別線1"
                else:
                    structure = "②のみ＋別線2"

                role2 = rider_role(cars[top3_cars[1]], win_line)
                role3 = rider_role(cars[top3_cars[2]], win_line)
                ordered = f"②→{role2}→{role3}"

                # own leader / third actual finish ranks
                finish_by_car = {
                    _i(r.get("car_no")): fp
                    for fp, r in results_by_race[rid].items()
                    if _i(r.get("car_no")) is not None
                }
                lf = finish_by_car.get(own_leader)
                tf = finish_by_car.get(own_third)

                for b in targets:
                    b["finish_structure"][structure] += 1
                    b["ordered_top3_roles"][ordered] += 1
                    b["leader_finish"][str(lf) if lf is not None else "不明"] += 1
                    b["third_finish"][str(tf) if tf is not None else "不明"] += 1

                if structure == "②のみ＋別線2":
                    e2 = cars[top3_cars[1]]
                    e3 = cars[top3_cars[2]]
                    l2 = e2.get("line_id")
                    l3 = e3.get("line_id")
                    if _i(e2.get("line_size")) == 1 or _i(e3.get("line_size")) == 1:
                        rel = "単騎を含む"
                    elif l2 and l3 and l2 == l3:
                        rel = "同じ別線から2人"
                    else:
                        rel = "別々のラインから1人ずつ"
                    for b in targets:
                        b["outside_pair_relation_when_bante_only"][rel] += 1

    out = {}
    for key, b in buckets.items():
        n = b["bante_win_races"]
        out[key] = {
            "s_general_races": b["s_general_races"],
            "three_line_races": b["three_line_races"],
            "bante_win_races": n,
            "bante_win_rate_among_three_line_races_pct": pct(n, b["three_line_races"]),
            "finish_structure": [
                {"pattern": k, "count": v, "pct": pct(v, n)}
                for k, v in b["finish_structure"].most_common()
            ],
            "ordered_top3_roles_top15": [
                {"pattern": k, "count": v, "pct": pct(v, n)}
                for k, v in b["ordered_top3_roles"].most_common(15)
            ],
            "winner_method": [
                {"method": k, "count": v, "pct": pct(v, n)}
                for k, v in b["winner_method"].most_common()
            ],
            "leader_finish": [
                {"finish": k, "count": v, "pct": pct(v, n)}
                for k, v in sorted(b["leader_finish"].items(), key=lambda kv: (999 if kv[0] == "不明" else int(kv[0])))
            ],
            "third_finish": [
                {"finish": k, "count": v, "pct": pct(v, n)}
                for k, v in sorted(b["third_finish"].items(), key=lambda kv: (999 if kv[0] == "不明" else int(kv[0])))
            ],
            "outside_pair_relation_when_bante_only": [
                {"pattern": k, "count": v, "pct_of_bante_only": pct(v, b["finish_structure"].get("②のみ＋別線2", 0))}
                for k, v in b["outside_pair_relation_when_bante_only"].most_common()
            ],
        }
    return out


def render_markdown(name, data):
    a = data["ALL"]
    lines = [
        f"# S級一般 3車ライン番手1着 集計 ({name})",
        "",
        f"- S級一般: {a['s_general_races']}R",
        f"- 3車ラインあり: {a['three_line_races']}R",
        f"- その3車ライン番手が1着: {a['bante_win_races']}R ({a['bante_win_rate_among_three_line_races_pct']}%)",
        "",
        "## 結末構造",
    ]
    for x in a["finish_structure"]:
        lines.append(f"- {x['pattern']}: {x['count']}R ({x['pct']}%)")
    lines += ["", "## 3連単の役割順 TOP15"]
    for x in a["ordered_top3_roles_top15"]:
        lines.append(f"- {x['pattern']}: {x['count']}R ({x['pct']}%)")
    lines += ["", "## 年別"]
    for y in ("2024", "2025", "2026H1"):
        d = data[y]
        lines.append(f"- {y}: 3車ラインあり {d['three_line_races']}R / 番手1着 {d['bante_win_races']}R ({d['bante_win_rate_among_three_line_races_pct']}%)")
        if d["finish_structure"]:
            top = ", ".join(f"{x['pattern']} {x['pct']}%" for x in d["finish_structure"])
            lines.append(f"  - 構造: {top}")
    return "\n".join(lines)


def main():
    out = {
        "all_entry_counts": analyze_scope(only_7=False),
        "seven_car_only": analyze_scope(only_7=True),
    }
    report_dir = Path("results/s_general_bante_win")
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "summary.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    md = render_markdown("全車立て", out["all_entry_counts"]) + "\n\n" + render_markdown("7車立てのみ", out["seven_car_only"])
    (report_dir / "summary.md").write_text(md, encoding="utf-8")
    print(md)
    print("\n===JSON===")
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
