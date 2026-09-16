#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

BASE = Path("data/2024/s_class_f1_all_parts/2024_q1")
START_DATE = "2024-01-01"
END_DATE = "2024-01-07"
OUT_DIR = Path("results/keirin_shogi/v1_role_first/2024-01-01_2024-01-07")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 2着・3着はv0据え置き。今回の変更点は1着のみ。
WEIGHTS = {
    "second": {"b": 0.10, "win": 0.20, "line": 0.20, "ban": 0.30, "cross": 0.20},
    "third":  {"b": 0.05, "win": 0.10, "line": 0.20, "ban": 0.20, "cross": 0.45},
}

def fnum(v, default=0.0):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default

def inum(v, default=0):
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return default

def normalize(values):
    if not values:
        return []
    lo = min(values)
    hi = max(values)
    if hi == lo:
        return [50.0] * len(values)
    return [((v - lo) / (hi - lo)) * 100.0 for v in values]

def read_csv(path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))

def derive_features(rows):
    # すべて事前出走表由来。結果・オッズは不使用。
    raw_cols = {
        "b": [fnum(r.get("b_count")) for r in rows],
        "win": [fnum(r.get("win_rate")) for r in rows],
        "line": [fnum(r.get("line_size"), 1.0) for r in rows],
        "score": [fnum(r.get("score")) for r in rows],
        "nige": [fnum(r.get("nige_count")) for r in rows],
        "makuri": [fnum(r.get("makuri_count")) for r in rows],
        "sashi": [fnum(r.get("sashi_count")) for r in rows],
        "mark": [fnum(r.get("mark_count")) for r in rows],
        "top2": [fnum(r.get("top2_rate")) for r in rows],
    }
    norm = {k: normalize(v) for k, v in raw_cols.items()}

    # ライン先頭を引けるようにline_idで索引。
    leader_idx_by_line = {}
    for i, r in enumerate(rows):
        line_id = str(r.get("line_id") or "")
        pos = inum(r.get("line_position"))
        if line_id and pos == 1:
            leader_idx_by_line[line_id] = i

    features = []
    for i, r in enumerate(rows):
        line_pos = inum(r.get("line_position"))
        line_size = fnum(r.get("line_size"), 1.0)
        line_id = str(r.get("line_id") or "")

        if line_size <= 1:
            role = "solo"
        elif line_pos == 1:
            role = "leader"
        elif line_pos == 2:
            role = "second"
        else:
            role = "third_plus"

        ban = 0.0
        if line_pos == 2:
            ban = 0.70 * norm["sashi"][i] + 0.30 * norm["mark"][i]

        cross = 0.50 * norm["score"][i] + 0.30 * norm["makuri"][i] + 0.20 * norm["top2"][i]

        # 番手評価で「前の選手」を見る。
        leader_strength = 0.0
        leader_idx = leader_idx_by_line.get(line_id)
        if leader_idx is not None and leader_idx != i:
            leader_strength = (
                0.35 * norm["b"][leader_idx]
                + 0.30 * norm["nige"][leader_idx]
                + 0.20 * norm["makuri"][leader_idx]
                + 0.15 * norm["score"][leader_idx]
            )

        features.append({
            "no": inum(r.get("car_no")),
            "name": r.get("player_name", ""),
            "role": role,
            "line_id": line_id,
            "line_position": line_pos,
            "line_size": line_size,
            "b": raw_cols["b"][i],
            "win": raw_cols["win"][i],
            "line": raw_cols["line"][i],
            "score": raw_cols["score"][i],
            "nige_count": raw_cols["nige"][i],
            "makuri_count": raw_cols["makuri"][i],
            "sashi_count": raw_cols["sashi"][i],
            "mark_count": raw_cols["mark"][i],
            "top2_rate": raw_cols["top2"][i],
            "ban": ban,
            "cross": cross,
            "leader_strength": leader_strength,
            "norm": {k: norm[k][i] for k in norm},
        })
    return features

def first_score(x):
    n = x["norm"]
    role = x["role"]

    # 役割ごとに「1着になる型」を分ける。
    if role == "leader":
        parts = {
            "勝率": n["win"] * 0.25,
            "逃げ実績": n["nige"] * 0.25,
            "捲り実績": n["makuri"] * 0.25,
            "B数": n["b"] * 0.15,
            "競走得点": n["score"] * 0.10,
        }
    elif role == "second":
        parts = {
            "勝率": n["win"] * 0.20,
            "差し実績": n["sashi"] * 0.30,
            "マーク実績": n["mark"] * 0.10,
            "競走得点": n["score"] * 0.20,
            "前の選手の先行力": x["leader_strength"] * 0.20,
        }
    elif role == "third_plus":
        parts = {
            "勝率": n["win"] * 0.25,
            "差し実績": n["sashi"] * 0.25,
            "マーク実績": n["mark"] * 0.15,
            "競走得点": n["score"] * 0.20,
            "捲り実績": n["makuri"] * 0.15,
        }
    else:  # solo
        parts = {
            "勝率": n["win"] * 0.30,
            "競走得点": n["score"] * 0.25,
            "捲り実績": n["makuri"] * 0.25,
            "2連対率": n["top2"] * 0.15,
            "B数": n["b"] * 0.05,
        }

    return sum(parts.values()), parts

def score_race(features):
    nb = normalize([x["b"] for x in features])
    nwin = normalize([x["win"] for x in features])
    nline = normalize([x["line"] for x in features])
    nban = normalize([x["ban"] for x in features])
    ncross = normalize([x["cross"] for x in features])

    out = []
    for i, x in enumerate(features):
        item = dict(x)
        item["scores"] = {}
        item["parts"] = {}

        fs, fp = first_score(x)
        item["scores"]["first"] = fs
        item["parts"]["first"] = fp

        common = {"b": nb[i], "win": nwin[i], "line": nline[i], "ban": nban[i], "cross": ncross[i]}
        for rank in ("second", "third"):
            w = WEIGHTS[rank]
            parts = {k: common[k] * w[k] for k in w}
            item["parts"][rank] = parts
            item["scores"][rank] = sum(parts.values())

        out.append(item)
    return out

def choose(scored, rank):
    s = sorted(scored, key=lambda x: (-x["scores"][rank], x["no"]))
    if rank == "first":
        count = 2
        if len(s) >= 2 and s[0]["scores"][rank] - s[1]["scores"][rank] >= 15:
            count = 1
    elif rank == "second":
        count = min(3, len(s))
        if len(s) >= 3 and s[1]["scores"][rank] - s[2]["scores"][rank] >= 15:
            count = 2
    else:
        count = min(4, len(s))
        if len(s) >= 4 and s[2]["scores"][rank] - s[3]["scores"][rank] >= 15:
            count = 3
    return sorted([x["no"] for x in s[:count]])

def main():
    races = read_csv(BASE / "races.csv")
    entries = read_csv(BASE / "entries.csv")
    results = read_csv(BASE / "results.csv")

    target_races = {
        r["race_id"]: r
        for r in races
        if START_DATE <= r.get("race_date", "") <= END_DATE
        and inum(r.get("entry_count")) == 7
        and r.get("meeting_grade") == "F1"
        and "Ｓ級" in r.get("race_type", "")
    }

    entries_by_race = defaultdict(list)
    for r in entries:
        if r.get("race_id") in target_races:
            entries_by_race[r["race_id"]].append(r)

    results_by_race = defaultdict(list)
    for r in results:
        if r.get("race_id") in target_races:
            results_by_race[r["race_id"]].append(r)

    rows_out = []
    detail = []

    for race_id, race in sorted(target_races.items(), key=lambda kv: (kv[1]["race_date"], kv[1]["track"], inum(kv[1]["race_no"]))):
        erows = sorted(entries_by_race.get(race_id, []), key=lambda x: inum(x.get("car_no")))
        rrows = results_by_race.get(race_id, [])
        if len(erows) != 7:
            continue

        finish = {}
        for rr in rrows:
            pos = inum(rr.get("finish_position"))
            if pos in (1, 2, 3):
                finish[pos] = inum(rr.get("car_no"))
        if set(finish) != {1, 2, 3}:
            continue

        features = derive_features(erows)
        scored = score_race(features)
        first = choose(scored, "first")
        second = choose(scored, "second")
        third = choose(scored, "third")

        hit1 = finish[1] in first
        hit2 = finish[2] in second
        hit3 = finish[3] in third
        complete = hit1 and hit2 and hit3

        winner = next((x for x in scored if x["no"] == finish[1]), None)
        winner_role = winner["role"] if winner else ""

        rows_out.append({
            "race_id": race_id,
            "race_date": race["race_date"],
            "track": race["track"],
            "race_no": race["race_no"],
            "race_type": race["race_type"],
            "first_candidates": "-".join(map(str, first)),
            "second_candidates": "-".join(map(str, second)),
            "third_candidates": "-".join(map(str, third)),
            "actual_1st": finish[1],
            "actual_1st_role": winner_role,
            "actual_2nd": finish[2],
            "actual_3rd": finish[3],
            "hit_1st": int(hit1),
            "hit_2nd": int(hit2),
            "hit_3rd": int(hit3),
            "complete_capture": int(complete),
            "candidate_cells": len(first) + len(second) + len(third),
        })

        detail.append({
            "race": {
                "race_id": race_id,
                "race_date": race["race_date"],
                "track": race["track"],
                "race_no": race["race_no"],
                "race_type": race["race_type"],
            },
            "board": {"first": first, "second": second, "third": third},
            "actual": {"first": finish[1], "second": finish[2], "third": finish[3]},
            "hits": {"first": hit1, "second": hit2, "third": hit3, "complete": complete},
            "scored": scored,
        })

    n = len(rows_out)
    role_counts = defaultdict(lambda: {"wins": 0, "captured": 0})
    for r in rows_out:
        role = r["actual_1st_role"]
        role_counts[role]["wins"] += 1
        role_counts[role]["captured"] += r["hit_1st"]

    summary = {
        "period": {"start": START_DATE, "end": END_DATE},
        "scope": "F1 S級・7車立てのみ",
        "algorithm": "keirin_shogi_v1_role_first",
        "races_evaluated": n,
        "first_hit_rate": (sum(r["hit_1st"] for r in rows_out) / n if n else 0),
        "second_hit_rate": (sum(r["hit_2nd"] for r in rows_out) / n if n else 0),
        "third_hit_rate": (sum(r["hit_3rd"] for r in rows_out) / n if n else 0),
        "complete_capture_rate": (sum(r["complete_capture"] for r in rows_out) / n if n else 0),
        "avg_candidate_cells": (mean(r["candidate_cells"] for r in rows_out) if n else 0),
        "winner_role_capture": dict(role_counts),
        "first_algorithm": {
            "leader": "勝率25% + 逃げ25% + 捲り25% + B15% + 競走得点10%",
            "second": "勝率20% + 差し30% + マーク10% + 競走得点20% + 前の選手の先行力20%",
            "third_plus": "勝率25% + 差し25% + マーク15% + 競走得点20% + 捲り15%",
            "solo": "勝率30% + 競走得点25% + 捲り25% + 2連対率15% + B5%",
        },
        "leader_strength": "前の選手の B35% + 逃げ30% + 捲り20% + 競走得点15%",
        "note": "1着のみ役割別ロジックへ変更。2着・3着はv0据え置き。結果・オッズは特徴量計算に未使用。",
    }

    with (OUT_DIR / "race_log.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        w.writeheader()
        w.writerows(rows_out)

    (OUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT_DIR / "detail.json").write_text(json.dumps(detail, ensure_ascii=False, indent=2), encoding="utf-8")

    md = [
        "# 競輪将棋 v1 役割別1着評価",
        "",
        f"- 期間: {START_DATE}〜{END_DATE}",
        "- 対象: F1 S級・7車立て",
        f"- 評価レース数: {n}",
        "",
        "## 結果",
        "",
        f"- 1着段捕捉率: {summary['first_hit_rate']:.1%}",
        f"- 2着段捕捉率: {summary['second_hit_rate']:.1%}",
        f"- 3着段捕捉率: {summary['third_hit_rate']:.1%}",
        f"- 3段完全捕捉率: {summary['complete_capture_rate']:.1%}",
        f"- 平均配置マス数: {summary['avg_candidate_cells']:.2f}",
        "",
        "## 1着ロジック",
        "",
        "- 先頭: 勝率25% + 逃げ25% + 捲り25% + B15% + 競走得点10%",
        "- 番手: 勝率20% + 差し30% + マーク10% + 競走得点20% + 前の選手の先行力20%",
        "- 3番手以降: 勝率25% + 差し25% + マーク15% + 競走得点20% + 捲り15%",
        "- 単騎: 勝率30% + 競走得点25% + 捲り25% + 2連対率15% + B5%",
        "",
        "## 重要",
        "",
        "- 番手の1着評価では前の選手のB・逃げ・捲り・競走得点を見る",
        "- オッズ不使用",
        "- 結果は評価にのみ使用",
    ]
    (OUT_DIR / "README.md").write_text("\n".join(md), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
