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
OUT_DIR = Path("results/keirin_shogi/v0/2024-01-01_2024-01-07")
OUT_DIR.mkdir(parents=True, exist_ok=True)

WEIGHTS = {
    "first":  {"b": 0.30, "win": 0.35, "line": 0.15, "ban": 0.05, "cross": 0.15},
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
    score_norm = normalize([fnum(r.get("score")) for r in rows])
    makuri_norm = normalize([fnum(r.get("makuri_count")) for r in rows])
    top2_norm = normalize([fnum(r.get("top2_rate")) for r in rows])
    sashi_norm = normalize([fnum(r.get("sashi_count")) for r in rows])
    mark_norm = normalize([fnum(r.get("mark_count")) for r in rows])

    features = []
    for i, r in enumerate(rows):
        line_pos = inum(r.get("line_position"))
        # 番手差し評価:
        # 2番手だけを「番手」として扱い、差し実績70% + マーク実績30%。
        # 2番手以外は0。v0の明示的な仮説で、後続週で壊す前提。
        ban = 0.0
        if line_pos == 2:
            ban = 0.70 * sashi_norm[i] + 0.30 * mark_norm[i]

        # 別線能力:
        # ライン位置に依存しない地力の代理。
        # 競走得点50% + 捲り実績30% + 2連対率20%。
        cross = 0.50 * score_norm[i] + 0.30 * makuri_norm[i] + 0.20 * top2_norm[i]

        features.append({
            "no": inum(r.get("car_no")),
            "name": r.get("player_name", ""),
            "b": fnum(r.get("b_count")),
            "win": fnum(r.get("win_rate")),
            "line": fnum(r.get("line_size"), 1.0),
            "ban": ban,
            "cross": cross,
            "line_position": line_pos,
            "score": fnum(r.get("score")),
            "makuri_count": fnum(r.get("makuri_count")),
            "top2_rate": fnum(r.get("top2_rate")),
            "sashi_count": fnum(r.get("sashi_count")),
            "mark_count": fnum(r.get("mark_count")),
        })
    return features

def score_race(features):
    nb = normalize([x["b"] for x in features])
    nwin = normalize([x["win"] for x in features])
    nline = normalize([x["line"] for x in features])
    nban = normalize([x["ban"] for x in features])
    ncross = normalize([x["cross"] for x in features])

    out = []
    for i, x in enumerate(features):
        n = {"b": nb[i], "win": nwin[i], "line": nline[i], "ban": nban[i], "cross": ncross[i]}
        item = dict(x)
        item["normalized"] = n
        item["scores"] = {}
        item["parts"] = {}
        for rank, w in WEIGHTS.items():
            parts = {k: n[k] * w[k] for k in w}
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

        missing = []
        if not hit1: missing.append("1着")
        if not hit2: missing.append("2着")
        if not hit3: missing.append("3着")

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
            "actual_2nd": finish[2],
            "actual_3rd": finish[3],
            "hit_1st": int(hit1),
            "hit_2nd": int(hit2),
            "hit_3rd": int(hit3),
            "complete_capture": int(complete),
            "missing_ranks": "/".join(missing),
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
    summary = {
        "period": {"start": START_DATE, "end": END_DATE},
        "scope": "F1 S級・7車立てのみ",
        "algorithm": "keirin_shogi_v0",
        "races_evaluated": n,
        "first_hit_rate": (sum(r["hit_1st"] for r in rows_out) / n if n else 0),
        "second_hit_rate": (sum(r["hit_2nd"] for r in rows_out) / n if n else 0),
        "third_hit_rate": (sum(r["hit_3rd"] for r in rows_out) / n if n else 0),
        "complete_capture_rate": (sum(r["complete_capture"] for r in rows_out) / n if n else 0),
        "avg_candidate_cells": (mean(r["candidate_cells"] for r in rows_out) if n else 0),
        "missing_rank_counts": {
            "1着": sum(1 for r in rows_out if not r["hit_1st"]),
            "2着": sum(1 for r in rows_out if not r["hit_2nd"]),
            "3着": sum(1 for r in rows_out if not r["hit_3rd"]),
        },
        "feature_definition": {
            "B数": "entries.b_count",
            "勝率": "entries.win_rate",
            "ライン長": "entries.line_size",
            "番手差し評価": "line_position==2 のとき、レース内正規化 sashi_count×0.70 + mark_count×0.30。その他は0",
            "別線能力": "レース内正規化 score×0.50 + makuri_count×0.30 + top2_rate×0.20",
        },
        "weights": WEIGHTS,
        "note": "2024-01-01〜01-07は同一v0で固定。結果・オッズは特徴量計算に使用していない。",
    }

    csv_path = OUT_DIR / "race_log.csv"
    fieldnames = list(rows_out[0].keys()) if rows_out else [
        "race_id","race_date","track","race_no","race_type","first_candidates","second_candidates","third_candidates",
        "actual_1st","actual_2nd","actual_3rd","hit_1st","hit_2nd","hit_3rd","complete_capture","missing_ranks","candidate_cells"
    ]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows_out)

    (OUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT_DIR / "detail.json").write_text(json.dumps(detail, ensure_ascii=False, indent=2), encoding="utf-8")

    md = [
        "# 競輪将棋 v0 週次試走",
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
        f"- 抜け: 1着 {summary['missing_rank_counts']['1着']} / 2着 {summary['missing_rank_counts']['2着']} / 3着 {summary['missing_rank_counts']['3着']}",
        "",
        "## 特徴量定義",
        "",
        "- B数: b_count",
        "- 勝率: win_rate",
        "- ライン長: line_size",
        "- 番手差し評価: 2番手のみ、sashi_count 70% + mark_count 30%（各レース内正規化）",
        "- 別線能力: score 50% + makuri_count 30% + top2_rate 20%（各レース内正規化）",
        "",
        "## 固定条件",
        "",
        "- 1週間の途中で重み変更なし",
        "- オッズ不使用",
        "- 結果は評価にのみ使用",
        "",
    ]
    (OUT_DIR / "README.md").write_text("\n".join(md), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("\nPER-RACE")
    for r in rows_out:
        print(
            f"{r['race_date']} {r['track']} {r['race_no']}R "
            f"[{r['first_candidates']}]-[{r['second_candidates']}]-[{r['third_candidates']}] "
            f"=> {r['actual_1st']}-{r['actual_2nd']}-{r['actual_3rd']} "
            f"hits={r['hit_1st']}{r['hit_2nd']}{r['hit_3rd']} complete={r['complete_capture']}"
        )

if __name__ == "__main__":
    main()
