from __future__ import annotations

import csv
import io
import json
import math
import statistics
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_ROOT = ROOT / "data" / "grade_races" / "g3"
OUT = ROOT / "results" / "g3_day3_reality"


def decode(raw: bytes) -> str:
    for enc in ("utf-8-sig", "utf-8", "cp932", "shift_jis"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            pass
    return raw.decode("utf-8", errors="replace")


def read_member(zf: zipfile.ZipFile, basename: str) -> list[dict[str, str]]:
    matches = [n for n in zf.namelist() if Path(n).name == basename]
    if not matches:
        return []
    return list(csv.DictReader(io.StringIO(decode(zf.read(matches[0])))))


def num(v: str | None) -> float:
    try:
        x = float(v or "")
        return x if math.isfinite(x) else float("-inf")
    except (TypeError, ValueError):
        return float("-inf")


def int_or_none(v: str | None) -> int | None:
    try:
        return int(v or "")
    except (TypeError, ValueError):
        return None


def meeting_day(race_id: str) -> int | None:
    if len(race_id) == 16 and race_id.isdigit():
        try:
            return int(race_id[10:12])
        except ValueError:
            return None
    return None


def line_maps(entries: list[dict[str, str]]) -> dict[int, list[dict[str, str]]]:
    by_line: dict[int, list[dict[str, str]]] = defaultdict(list)
    for e in entries:
        if e.get("line_id", "").isdigit() and e.get("line_position", "").isdigit():
            by_line[int(e["line_id"])].append(e)
    for lid in by_line:
        by_line[lid].sort(key=lambda e: int(e["line_position"]))
    return by_line


def line_strength(members: list[dict[str, str]]) -> float:
    if len(members) >= 2:
        return num(members[0].get("score")) + num(members[1].get("score"))
    if members:
        return num(members[0].get("score"))
    return float("-inf")


def choose_main_line(entries: list[dict[str, str]]) -> tuple[int, list[dict[str, str]]] | None:
    by_line = line_maps(entries)
    candidates = []
    for lid, members in by_line.items():
        if len(members) < 2:
            continue
        key = (line_strength(members), num(members[0].get("score")), 1 if len(members) >= 3 else 0, -lid)
        candidates.append((key, lid, members))
    if not candidates:
        return None
    _, lid, members = max(candidates, key=lambda x: x[0])
    return lid, members


def percentile_median(xs: list[int]) -> float | None:
    return statistics.median(xs) if xs else None


def pct(n: int, d: int) -> float:
    return n / d if d else 0.0


def bucket_gap(gap: float | None) -> str:
    if gap is None or not math.isfinite(gap):
        return "singleton/unknown"
    if gap <= 1:
        return "<=1"
    if gap <= 3:
        return "1-3"
    if gap <= 5:
        return "3-5"
    if gap <= 10:
        return "5-10"
    return ">10"


def bucket_pop(rank: int | None) -> str:
    if rank is None:
        return "unknown"
    if rank <= 10:
        return "1-10"
    if rank <= 30:
        return "11-30"
    if rank <= 100:
        return "31-100"
    return "101+"


def bucket_score_rank(rank: int | None) -> str:
    if rank is None:
        return "unknown"
    if rank <= 3:
        return "1-3"
    if rank <= 6:
        return "4-6"
    return "7-9"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    races: dict[str, dict[str, str]] = {}
    entries_by_race: dict[str, list[dict[str, str]]] = defaultdict(list)
    results_by_race: dict[str, list[dict[str, str]]] = defaultdict(list)
    payouts_by_race: dict[str, list[dict[str, str]]] = defaultdict(list)
    trifecta_odds_by_race: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)

    for path in sorted(ARCHIVE_ROOT.rglob("*.zip")):
        with zipfile.ZipFile(path) as zf:
            for r in read_member(zf, "races.csv"):
                races.setdefault(r["race_id"], r)
            for e in read_member(zf, "entries.csv"):
                entries_by_race[e["race_id"]].append(e)
            for rr in read_member(zf, "results.csv"):
                results_by_race[rr["race_id"]].append(rr)
            for p in read_member(zf, "payouts.csv"):
                payouts_by_race[p["race_id"]].append(p)
            for o in read_member(zf, "trifecta_final_odds.csv"):
                trifecta_odds_by_race[o["race_id"]][o.get("combination", "")] = o

    rows: list[dict[str, object]] = []
    all_valid: list[dict[str, object]] = []

    for race in sorted(races.values(), key=lambda r: (r.get("race_date", ""), r.get("track", ""), int(r.get("race_no", "0") or 0))):
        rid = race["race_id"]
        if meeting_day(rid) != 3:
            continue
        es = entries_by_race.get(rid, [])
        rs = results_by_race.get(rid, [])
        main = choose_main_line(es)
        if not main or not rs:
            continue
        main_id, main_members = main
        main_cars = [int(e["car_no"]) for e in main_members]
        by_line = line_maps(es)
        entry_by_car = {int(e["car_no"]): e for e in es if e.get("car_no", "").isdigit()}

        finish = []
        for rr in rs:
            pos = int_or_none(rr.get("finish_position")); car = int_or_none(rr.get("car_no"))
            if pos in (1, 2, 3) and car is not None:
                finish.append((pos, car))
        if len(finish) != 3 or sorted(p for p, _ in finish) != [1, 2, 3]:
            continue
        top3 = [c for _, c in sorted(finish)]
        winner = top3[0]

        # Ranking of riders by pre-race score (ties use lower car number for stability).
        scored = sorted(es, key=lambda e: (-num(e.get("score")), int(e.get("car_no", "99") or 99)))
        score_rank_by_car = {int(e["car_no"]): i + 1 for i, e in enumerate(scored) if e.get("car_no", "").isdigit()}

        # Rank 2+ rider lines by the same pair-strength concept used to choose the main line.
        ranked_lines = sorted(
            [(line_strength(mem), num(mem[0].get("score")), lid, mem) for lid, mem in by_line.items() if len(mem) >= 2],
            key=lambda x: (-x[0], -x[1], x[2]),
        )
        line_rank = {lid: i + 1 for i, (_, _, lid, _) in enumerate(ranked_lines)}
        main_strength = line_strength(main_members)

        we = entry_by_car.get(winner, {})
        wlid = int(we["line_id"]) if we.get("line_id", "").isdigit() else None
        wpos = int(we["line_position"]) if we.get("line_position", "").isdigit() else None
        wmembers = by_line.get(wlid, []) if wlid is not None else []
        if len(wmembers) >= 2:
            w_relation = "rival_rank2" if line_rank.get(wlid) == 2 else "other_line"
            w_role = "leader" if wpos == 1 else ("second" if wpos == 2 else "third_plus")
            gap = main_strength - line_strength(wmembers)
            w_line_rank = line_rank.get(wlid)
        else:
            w_relation = "singleton"
            w_role = "singleton"
            gap = None
            w_line_rank = None

        payouts = payouts_by_race.get(rid, [])
        winning_tf = f"{top3[0]}-{top3[1]}-{top3[2]}"
        paid_tf = [p for p in payouts if p.get("ticket_type") == "3連単" and p.get("status") == "paid" and p.get("combination") == winning_tf]
        tf_payout = int(float(paid_tf[0].get("payout_yen", "0") or 0)) if paid_tf else 0
        tf_pop = int_or_none(paid_tf[0].get("popularity")) if paid_tf else None

        abc_rank = None; abc_odds = None; abc_exact = False
        if len(main_cars) >= 3:
            abc = f"{main_cars[0]}-{main_cars[1]}-{main_cars[2]}"
            oo = trifecta_odds_by_race.get(rid, {}).get(abc, {})
            abc_rank = int_or_none(oo.get("market_rank"))
            try:
                abc_odds = float(oo.get("odds", "")) if oo.get("odds", "") else None
            except ValueError:
                abc_odds = None
            abc_exact = top3 == main_cars[:3]

        base = {
            "race_id": rid,
            "race_date": race.get("race_date", ""),
            "track": race.get("track", ""),
            "race_no": race.get("race_no", ""),
            "race_type": race.get("race_type", ""),
            "main_line": "-".join(map(str, main_cars)),
            "winner": winner,
            "winner_name": we.get("player_name", ""),
            "winner_in_mainline": int(winner in main_cars),
            "winner_relation": w_relation,
            "winner_role": w_role,
            "winner_line_rank": w_line_rank or "",
            "winner_score_rank": score_rank_by_car.get(winner, ""),
            "winner_score": we.get("score", ""),
            "main_pair_strength": main_strength,
            "winner_line_gap": gap if gap is not None else "",
            "main_top3_count": sum(c in main_cars for c in top3),
            "top3": winning_tf,
            "winning_trifecta_payout": tf_payout,
            "winning_trifecta_popularity": tf_pop or "",
            "abc_market_rank": abc_rank or "",
            "abc_final_odds": abc_odds or "",
            "abc_exact": int(abc_exact),
        }
        all_valid.append(base)
        if winner not in main_cars:
            rows.append(base)

    n_all = len(all_valid)
    n = len(rows)
    main_survivors = Counter(int(r["main_top3_count"]) for r in rows)
    relation = Counter(str(r["winner_relation"]) for r in rows)
    roles = Counter(str(r["winner_role"]) for r in rows)
    score_buckets = Counter(bucket_score_rank(int(r["winner_score_rank"]) if str(r["winner_score_rank"]).isdigit() else None) for r in rows)
    gap_buckets = Counter(bucket_gap(float(r["winner_line_gap"]) if r["winner_line_gap"] != "" else None) for r in rows)
    pop_buckets = Counter(bucket_pop(int(r["winning_trifecta_popularity"]) if str(r["winning_trifecta_popularity"]).isdigit() else None) for r in rows)
    pops = [int(r["winning_trifecta_popularity"]) for r in rows if str(r["winning_trifecta_popularity"]).isdigit()]
    payouts = [int(r["winning_trifecta_payout"]) for r in rows if int(r["winning_trifecta_payout"]) > 0]

    race_type = {}
    for typ, grp in defaultdict(list).items():
        pass
    by_type: dict[str, list[dict[str, object]]] = defaultdict(list)
    for r in rows:
        by_type[str(r["race_type"])].append(r)
    type_summary = {}
    for typ, grp in sorted(by_type.items(), key=lambda kv: -len(kv[1])):
        if len(grp) < 20:
            continue
        type_summary[typ] = {
            "offmain_wins": len(grp),
            "share_of_valid_type": pct(len(grp), sum(1 for x in all_valid if x["race_type"] == typ)),
            "main0_top3_rate": pct(sum(int(x["main_top3_count"]) == 0 for x in grp), len(grp)),
            "winner_rival_rank2_rate": pct(sum(x["winner_relation"] == "rival_rank2" for x in grp), len(grp)),
            "median_trifecta_popularity": percentile_median([int(x["winning_trifecta_popularity"]) for x in grp if str(x["winning_trifecta_popularity"]).isdigit()]),
            "median_trifecta_payout": percentile_median([int(x["winning_trifecta_payout"]) for x in grp if int(x["winning_trifecta_payout"]) > 0]),
        }

    def favorite_slice(pred) -> dict[str, object]:
        eligible = [r for r in all_valid if pred(r)]
        outside = [r for r in eligible if int(r["winner_in_mainline"]) == 0]
        exact = [r for r in eligible if int(r["abc_exact"]) == 1]
        return {
            "races": len(eligible),
            "abc_exact_hits": len(exact),
            "abc_exact_rate": pct(len(exact), len(eligible)),
            "winner_outside_main": len(outside),
            "winner_outside_main_rate": pct(len(outside), len(eligible)),
        }

    favorite_pressure = {
        "abc_rank1": favorite_slice(lambda r: str(r["abc_market_rank"]) == "1"),
        "abc_rank1_3": favorite_slice(lambda r: str(r["abc_market_rank"]).isdigit() and int(r["abc_market_rank"]) <= 3),
        "abc_odds_le_10": favorite_slice(lambda r: r["abc_final_odds"] != "" and float(r["abc_final_odds"]) <= 10),
        "abc_odds_le_15": favorite_slice(lambda r: r["abc_final_odds"] != "" and float(r["abc_final_odds"]) <= 15),
        "abc_odds_le_20": favorite_slice(lambda r: r["abc_final_odds"] != "" and float(r["abc_final_odds"]) <= 20),
    }

    top_big = sorted(rows, key=lambda r: int(r["winning_trifecta_payout"]), reverse=True)[:20]
    full_wipe = sorted([r for r in rows if int(r["main_top3_count"]) == 0], key=lambda r: int(r["winning_trifecta_payout"]), reverse=True)[:20]

    summary = {
        "valid_day3_races": n_all,
        "offmain_winner_races": n,
        "offmain_winner_rate": pct(n, n_all),
        "main_line_survivors_in_top3_when_winner_offmain": dict(sorted(main_survivors.items())),
        "winner_relation": dict(relation),
        "winner_role": dict(roles),
        "winner_score_rank_bucket": dict(score_buckets),
        "main_vs_winner_line_pair_score_gap": dict(gap_buckets),
        "winning_trifecta_popularity_bucket": dict(pop_buckets),
        "median_winning_trifecta_popularity": percentile_median(pops),
        "median_winning_trifecta_payout": percentile_median(payouts),
        "favorite_pressure": favorite_pressure,
        "by_race_type": type_summary,
    }

    fields = list(rows[0].keys()) if rows else []
    with (OUT / "offmain_races.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
    (OUT / "offmain_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# G3三日目 主力ライン外勝者の49.7%を分解",
        "",
        f"- 有効三日目: {n_all}R",
        f"- 勝者が主力ライン外: {n}R = {pct(n,n_all):.1%}",
        "",
        "## 主力ラインは何人残ったか",
    ]
    for k, v in sorted(main_survivors.items()):
        lines.append(f"- 主力ラインTOP3残存 {k}人: {v}R = {pct(v,n):.1%}")
    lines += ["", "## 外から勝った選手はどこにいたか"]
    for k, v in relation.most_common():
        lines.append(f"- {k}: {v}R = {pct(v,n):.1%}")
    lines += ["", "### ライン内役割"]
    for k, v in roles.most_common():
        lines.append(f"- {k}: {v}R = {pct(v,n):.1%}")
    lines += ["", "## 勝者の競走得点順位"]
    for k, v in score_buckets.items():
        lines.append(f"- {k}位帯: {v}R = {pct(v,n):.1%}")
    lines += ["", "## 主力ラインと勝者ラインの先頭+番手得点差"]
    for k, v in gap_buckets.items():
        lines.append(f"- {k}: {v}R = {pct(v,n):.1%}")
    lines += ["", "## 実際の3連単"]
    lines.append(f"- 的中組合せの人気中央値: {summary['median_winning_trifecta_popularity']}")
    lines.append(f"- 払戻中央値: {summary['median_winning_trifecta_payout']:,}円" if summary['median_winning_trifecta_payout'] else "- 払戻中央値: n/a")
    for k, v in pop_buckets.items():
        lines.append(f"- {k}人気帯: {v}R = {pct(v,n):.1%}")
    lines += ["", "## ABCが安くても外勝者になる圧力"]
    for k, d in favorite_pressure.items():
        lines.append(f"- {k}: {d['races']}R | ABCそのまま {d['abc_exact_hits']}R={d['abc_exact_rate']:.1%} | 勝者主力外 {d['winner_outside_main']}R={d['winner_outside_main_rate']:.1%}")
    lines += ["", "## レース種別（20R以上）"]
    for typ, d in type_summary.items():
        lines.append(f"- {typ}: 外勝者 {d['offmain_wins']}R ({d['share_of_valid_type']:.1%}) | 主力全滅 {d['main0_top3_rate']:.1%} | 2番手強度ライン勝者 {d['winner_rival_rank2_rate']:.1%} | 3連単人気中央値 {d['median_trifecta_popularity']} | 払戻中央値 {int(d['median_trifecta_payout'] or 0):,}円")
    lines += ["", "## 高配当の外勝者例"]
    for r in top_big[:10]:
        lines.append(f"- {r['race_date']} {r['track']} {r['race_no']}R {r['race_type']} | 主力 {r['main_line']} → {r['top3']} | 勝者 {r['winner_name']} (得点順位{r['winner_score_rank']}) | {int(r['winning_trifecta_payout']):,}円 / {r['winning_trifecta_popularity']}人気")
    lines += ["", "## 主力ライン0人でTOP3全崩壊の高配当例"]
    for r in full_wipe[:10]:
        lines.append(f"- {r['race_date']} {r['track']} {r['race_no']}R {r['race_type']} | 主力 {r['main_line']} → {r['top3']} | {int(r['winning_trifecta_payout']):,}円 / {r['winning_trifecta_popularity']}人気")

    (OUT / "offmain_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
