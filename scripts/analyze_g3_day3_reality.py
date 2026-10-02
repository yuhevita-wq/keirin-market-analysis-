from __future__ import annotations

import csv
import io
import itertools
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
    text = decode(zf.read(matches[0]))
    return list(csv.DictReader(io.StringIO(text)))


def num(v: str | None) -> float:
    try:
        return float(v or "")
    except (TypeError, ValueError):
        return float("-inf")


def choose_main_line(entries: list[dict[str, str]]) -> tuple[int, list[dict[str, str]]] | None:
    """Exact rule used by existing simulate_mainline_v1.py."""
    by_line: dict[int, list[dict[str, str]]] = defaultdict(list)
    for e in entries:
        if e.get("line_id", "").isdigit() and e.get("line_position", "").isdigit():
            by_line[int(e["line_id"])].append(e)
    candidates = []
    for line_id, members in by_line.items():
        members = sorted(members, key=lambda e: int(e["line_position"]))
        if len(members) < 2:
            continue
        leader, second = members[0], members[1]
        key = (
            num(leader.get("score")) + num(second.get("score")),
            num(leader.get("score")),
            1 if len(members) >= 3 else 0,
            -line_id,
        )
        candidates.append((key, line_id, members))
    if not candidates:
        return None
    _, line_id, members = max(candidates, key=lambda x: x[0])
    return line_id, members


def meeting_day(race_id: str) -> int | None:
    # KDreams archive IDs are TT + YYYYMMDD(start) + DD(meeting day) + RRRR(race no).
    if len(race_id) == 16 and race_id.isdigit():
        try:
            return int(race_id[10:12])
        except ValueError:
            return None
    return None


def int_or_none(v: str | None) -> int | None:
    try:
        return int(v or "")
    except ValueError:
        return None


def float_or_none(v: str | None) -> float | None:
    try:
        x = float(v or "")
        return x if math.isfinite(x) else None
    except ValueError:
        return None


def pct(x: int, n: int) -> float | None:
    return (x / n) if n else None


def money(v: str | None) -> int:
    try:
        return int(float(v or "0"))
    except ValueError:
        return 0


def summarize_strategy(rows: list[dict], prefix: str) -> dict:
    eligible = [r for r in rows if r.get(f"{prefix}_eligible")]
    n = len(eligible)
    stake = sum(int(r.get(f"{prefix}_stake", 0)) for r in eligible)
    returned = sum(int(r.get(f"{prefix}_return", 0)) for r in eligible)
    hits = sum(int(r.get(f"{prefix}_hit", 0)) for r in eligible)
    return {
        "eligible_races": n,
        "hit_races": hits,
        "hit_rate": pct(hits, n),
        "stake_yen": stake,
        "return_yen": returned,
        "profit_yen": returned - stake,
        "roi": (returned / stake) if stake else None,
    }


def median_or_none(xs: list[float]) -> float | None:
    return statistics.median(xs) if xs else None


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    races: dict[str, dict[str, str]] = {}
    entries_by_race: dict[str, list[dict[str, str]]] = defaultdict(list)
    results_by_race: dict[str, list[dict[str, str]]] = defaultdict(list)
    payouts_by_race: dict[str, list[dict[str, str]]] = defaultdict(list)
    trifecta_odds_by_race: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    trio_odds_by_race: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    archive_count = 0

    for path in sorted(ARCHIVE_ROOT.rglob("*.zip")):
        archive_count += 1
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
            for o in read_member(zf, "trio_final_odds.csv"):
                trio_odds_by_race[o["race_id"]][o.get("combination", "")] = o

    day3_races = [r for r in races.values() if meeting_day(r["race_id"]) == 3]
    day3_races.sort(key=lambda r: (r.get("race_date", ""), r.get("track", ""), int(r.get("race_no", "0") or 0)))

    rows: list[dict] = []
    invalid_strict_top3 = 0
    missing_results = 0
    missing_mainline = 0

    for race in day3_races:
        rid = race["race_id"]
        es = entries_by_race.get(rid, [])
        rs = results_by_race.get(rid, [])
        if not rs:
            missing_results += 1
            continue
        main = choose_main_line(es)
        if not main:
            missing_mainline += 1
            continue
        main_id, members = main
        main_cars = [int(e["car_no"]) for e in members]
        a = main_cars[0]
        b = main_cars[1]
        c = main_cars[2] if len(main_cars) >= 3 else None

        finish_pairs = []
        for rr in rs:
            pos = int_or_none(rr.get("finish_position"))
            car = int_or_none(rr.get("car_no"))
            if pos is not None and car is not None and pos <= 3:
                finish_pairs.append((pos, car))
        strict_top3_ok = len(finish_pairs) == 3 and sorted(p for p, _ in finish_pairs) == [1, 2, 3]
        if not strict_top3_ok:
            invalid_strict_top3 += 1
            continue
        top3 = [car for _, car in sorted(finish_pairs)]
        winner = top3[0]
        main_top3_count = sum(car in main_cars for car in top3)
        ab_both_top3 = a in top3 and b in top3
        ab_top2_any_order = set(top3[:2]) == {a, b}
        abc_all_top3 = c is not None and set(top3) == {a, b, c}
        abc_exact = c is not None and top3 == [a, b, c]
        bac_exact = c is not None and top3 == [b, a, c]
        core_survived_third_broke = c is not None and ab_top2_any_order and top3[2] != c
        winner_mainline_but_full_break = c is not None and winner in main_cars and not abc_all_top3

        payout_rows = payouts_by_race.get(rid, [])
        tf_paid = [p for p in payout_rows if p.get("ticket_type") == "3連単" and p.get("status") == "paid"]
        tr_paid = [p for p in payout_rows if p.get("ticket_type") == "3連複" and p.get("status") == "paid"]
        winning_tf = f"{top3[0]}-{top3[1]}-{top3[2]}"
        winning_tr = "=".join(map(str, sorted(top3)))
        tf_payout = next((money(p.get("payout_yen")) for p in tf_paid if p.get("combination") == winning_tf), 0)
        tf_pop = next((int_or_none(p.get("popularity")) for p in tf_paid if p.get("combination") == winning_tf), None)
        tr_payout = next((money(p.get("payout_yen")) for p in tr_paid if p.get("combination") == winning_tr), 0)
        tr_pop = next((int_or_none(p.get("popularity")) for p in tr_paid if p.get("combination") == winning_tr), None)

        entry_by_car = {int(e["car_no"]): e for e in es if e.get("car_no", "").isdigit()}
        names = {car: entry_by_car.get(car, {}).get("player_name", "") for car in top3}

        row = {
            "race_id": rid,
            "race_date": race.get("race_date", ""),
            "track": race.get("track", ""),
            "race_no": race.get("race_no", ""),
            "race_type": race.get("race_type", ""),
            "entry_count": race.get("entry_count", ""),
            "published_line": race.get("predicted_line_formation", ""),
            "main_line_id": main_id,
            "main_line_size": len(main_cars),
            "main_line": "-".join(map(str, main_cars)),
            "main_line_names": "/".join(entry_by_car.get(x, {}).get("player_name", "") for x in main_cars),
            "a": a,
            "b": b,
            "c": c or "",
            "top3": winning_tf,
            "top3_names": "/".join(names.get(x, "") for x in top3),
            "winner_in_mainline": int(winner in main_cars),
            "main_top3_count": main_top3_count,
            "ab_both_top3": int(ab_both_top3),
            "ab_top2_any_order": int(ab_top2_any_order),
            "abc_all_top3": int(bool(abc_all_top3)),
            "abc_exact": int(bool(abc_exact)),
            "bac_exact": int(bool(bac_exact)),
            "core_survived_third_broke": int(bool(core_survived_third_broke)),
            "winner_mainline_but_full_break": int(bool(winner_mainline_but_full_break)),
            "winning_trifecta_payout": tf_payout,
            "winning_trifecta_popularity": tf_pop or "",
            "winning_trio_payout": tr_payout,
            "winning_trio_popularity": tr_pop or "",
        }

        # Strategy 1: line-order one point A-B-C.
        if c is not None:
            combo = f"{a}-{b}-{c}"
            payout = next((money(p.get("payout_yen")) for p in tf_paid if p.get("combination") == combo), 0)
            o = trifecta_odds_by_race.get(rid, {}).get(combo, {})
            row.update({
                "abc1_eligible": 1,
                "abc1_combo": combo,
                "abc1_market_rank": int_or_none(o.get("market_rank")) or "",
                "abc1_final_odds": float_or_none(o.get("odds")) or "",
                "abc1_stake": 100,
                "abc1_return": payout,
                "abc1_hit": int(payout > 0),
            })

            # Strategy 2: A-B-C / B-A-C two-point core flip.
            combos2 = [f"{a}-{b}-{c}", f"{b}-{a}-{c}"]
            ret2 = sum(next((money(p.get("payout_yen")) for p in tf_paid if p.get("combination") == combo), 0) for combo in combos2)
            row.update({
                "abflip2_eligible": 1,
                "abflip2_combos": "/".join(combos2),
                "abflip2_stake": 200,
                "abflip2_return": ret2,
                "abflip2_hit": int(ret2 > 0),
            })

            # Strategy 3: all 6 permutations of first three main-line riders.
            combos6 = ["-".join(map(str, p)) for p in itertools.permutations([a, b, c], 3)]
            ret6 = sum(next((money(p.get("payout_yen")) for p in tf_paid if p.get("combination") == combo), 0) for combo in combos6)
            row.update({
                "box6_eligible": 1,
                "box6_stake": 600,
                "box6_return": ret6,
                "box6_hit": int(ret6 > 0),
            })

            # Strategy 4: trio A=B=C one point.
            tr_combo = "=".join(map(str, sorted([a, b, c])))
            tr_ret = next((money(p.get("payout_yen")) for p in tr_paid if p.get("combination") == tr_combo), 0)
            tr_o = trio_odds_by_race.get(rid, {}).get(tr_combo, {})
            row.update({
                "trio1_eligible": 1,
                "trio1_combo": tr_combo,
                "trio1_market_rank": int_or_none(tr_o.get("market_rank")) or "",
                "trio1_final_odds": float_or_none(tr_o.get("odds")) or "",
                "trio1_stake": 100,
                "trio1_return": tr_ret,
                "trio1_hit": int(tr_ret > 0),
            })
        else:
            for pref in ("abc1", "abflip2", "box6", "trio1"):
                row[f"{pref}_eligible"] = 0
                row[f"{pref}_stake"] = 0
                row[f"{pref}_return"] = 0
                row[f"{pref}_hit"] = 0

        rows.append(row)

    n = len(rows)
    n3_rows = [r for r in rows if int(r["main_line_size"]) >= 3]
    n3 = len(n3_rows)
    survivor_dist = Counter(int(r["main_top3_count"]) for r in n3_rows)

    abc_ranks = [float(r["abc1_market_rank"]) for r in n3_rows if str(r.get("abc1_market_rank", "")).strip()]
    trio_ranks = [float(r["trio1_market_rank"]) for r in n3_rows if str(r.get("trio1_market_rank", "")).strip()]

    structural = {
        "valid_day3_races": n,
        "mainline_size_3plus_races": n3,
        "winner_in_mainline": sum(int(r["winner_in_mainline"]) for r in rows),
        "winner_in_mainline_rate": pct(sum(int(r["winner_in_mainline"]) for r in rows), n),
        "mainline_3plus_top3_survivor_distribution": dict(sorted(survivor_dist.items())),
        "ab_both_top3": sum(int(r["ab_both_top3"]) for r in n3_rows),
        "ab_both_top3_rate": pct(sum(int(r["ab_both_top3"]) for r in n3_rows), n3),
        "ab_top2_any_order": sum(int(r["ab_top2_any_order"]) for r in n3_rows),
        "ab_top2_any_order_rate": pct(sum(int(r["ab_top2_any_order"]) for r in n3_rows), n3),
        "abc_all_top3": sum(int(r["abc_all_top3"]) for r in n3_rows),
        "abc_all_top3_rate": pct(sum(int(r["abc_all_top3"]) for r in n3_rows), n3),
        "abc_exact": sum(int(r["abc_exact"]) for r in n3_rows),
        "abc_exact_rate": pct(sum(int(r["abc_exact"]) for r in n3_rows), n3),
        "core_survived_third_broke": sum(int(r["core_survived_third_broke"]) for r in n3_rows),
        "core_survived_third_broke_rate": pct(sum(int(r["core_survived_third_broke"]) for r in n3_rows), n3),
        "winner_mainline_but_full_break": sum(int(r["winner_mainline_but_full_break"]) for r in n3_rows),
        "winner_mainline_but_full_break_rate": pct(sum(int(r["winner_mainline_but_full_break"]) for r in n3_rows), n3),
    }

    market = {
        "winning_trifecta_popularity_1": sum(1 for r in rows if r.get("winning_trifecta_popularity") == 1),
        "winning_trifecta_popularity_1_rate": pct(sum(1 for r in rows if r.get("winning_trifecta_popularity") == 1), n),
        "winning_trio_popularity_1": sum(1 for r in rows if r.get("winning_trio_popularity") == 1),
        "winning_trio_popularity_1_rate": pct(sum(1 for r in rows if r.get("winning_trio_popularity") == 1), n),
        "abc_straight_market_rank_median": median_or_none(abc_ranks),
        "abc_straight_ranked_top10_rate": pct(sum(1 for x in abc_ranks if x <= 10), len(abc_ranks)),
        "trio_main3_market_rank_median": median_or_none(trio_ranks),
        "trio_main3_ranked_top3_rate": pct(sum(1 for x in trio_ranks if x <= 3), len(trio_ranks)),
    }

    strategies = {
        "trifecta_A_B_C_one_point": summarize_strategy(n3_rows, "abc1"),
        "trifecta_A_B_C_and_B_A_C_two_points": summarize_strategy(n3_rows, "abflip2"),
        "trifecta_main3_box_six_points": summarize_strategy(n3_rows, "box6"),
        "trio_main3_one_point": summarize_strategy(n3_rows, "trio1"),
    }

    # Cruel examples: core A/B survive first-second but C is replaced, sorted by actual trifecta payout desc.
    cruel = [r for r in n3_rows if int(r["core_survived_third_broke"]) == 1]
    cruel.sort(key=lambda r: (int(r.get("winning_trifecta_payout", 0)), -int(r.get("race_no", 0) or 0)), reverse=True)
    cruel_examples = [
        {k: r.get(k) for k in (
            "race_id","race_date","track","race_no","race_type","published_line","main_line","main_line_names","top3","top3_names",
            "winning_trifecta_payout","winning_trifecta_popularity","winning_trio_payout","winning_trio_popularity",
            "abc1_market_rank","trio1_market_rank"
        )}
        for r in cruel[:20]
    ]

    # Another set: A-B-C itself was very highly ranked but failed.
    popular_burn = [r for r in n3_rows if r.get("abc1_market_rank") not in ("", None) and int(r["abc1_market_rank"]) <= 5 and not int(r["abc1_hit"])]
    popular_burn.sort(key=lambda r: (int(r["abc1_market_rank"]), -int(r.get("winning_trifecta_payout", 0))))
    popular_burn_examples = [
        {k: r.get(k) for k in (
            "race_id","race_date","track","race_no","race_type","published_line","main_line","main_line_names","top3","top3_names",
            "abc1_market_rank","abc1_final_odds","winning_trifecta_payout","winning_trifecta_popularity"
        )}
        for r in popular_burn[:20]
    ]

    summary = {
        "scope": {
            "archive_count": archive_count,
            "all_unique_g3_races": len(races),
            "day3_races_by_race_id": len(day3_races),
            "valid_strict_top3_mainline_races": n,
            "excluded_missing_results": missing_results,
            "excluded_missing_mainline": missing_mainline,
            "excluded_non_strict_top3_or_deadheat": invalid_strict_top3,
            "period_min": min((r.get("race_date", "") for r in day3_races), default=""),
            "period_max": max((r.get("race_date", "") for r in day3_races), default=""),
            "day3_definition": "KDreams race_id meeting-day field == 03",
            "mainline_definition": "existing simulate_mainline_v1 rule: among 2+ rider published lines, maximize leader score + second score; ties by leader score, then 3+ line, then lower line_id",
            "note": "This mainline is a pre-race score-defined main line, not a claim that it equals the market favorite line. Market popularity is reported separately from final odds/payout data.",
        },
        "structural": structural,
        "market": market,
        "strategies": strategies,
        "cruel_examples_core_survived_third_broke": cruel_examples,
        "popular_mainline_straight_burn_examples": popular_burn_examples,
    }

    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    fields = sorted({k for r in rows for k in r.keys()})
    with (OUT / "race_level.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    def fmt_rate(x: float | None) -> str:
        return "n/a" if x is None else f"{x*100:.1f}%"

    md = [
        "# G3三日目、本命ラインという安心料",
        "",
        f"- 対象期間: {summary['scope']['period_min']} ～ {summary['scope']['period_max']}",
        f"- G3 ZIP: {archive_count}本",
        f"- 三日目レース: {len(day3_races)}R / 有効分析: {n}R",
        f"- うち主力ライン3車以上: {n3}R",
        "",
        "## 定義",
        summary['scope']['mainline_definition'],
        "市場人気とは別物。市場人気は確定オッズ/払戻の popularity を別集計する。",
        "",
        "## 構造",
        f"- 勝者が主力ライン所属: {structural['winner_in_mainline']}/{n} = {fmt_rate(structural['winner_in_mainline_rate'])}",
        f"- A/Bがともに3着以内: {structural['ab_both_top3']}/{n3} = {fmt_rate(structural['ab_both_top3_rate'])}",
        f"- A/Bが1・2着を占有（順不同）: {structural['ab_top2_any_order']}/{n3} = {fmt_rate(structural['ab_top2_any_order_rate'])}",
        f"- A/B/Cが3着以内を独占（順不同）: {structural['abc_all_top3']}/{n3} = {fmt_rate(structural['abc_all_top3_rate'])}",
        f"- A-B-Cのライン順そのまま: {structural['abc_exact']}/{n3} = {fmt_rate(structural['abc_exact_rate'])}",
        f"- A/Bが1・2着なのにCが消える: {structural['core_survived_third_broke']}/{n3} = {fmt_rate(structural['core_survived_third_broke_rate'])}",
        f"- 勝者は主力ラインなのにA/B/C独占は崩壊: {structural['winner_mainline_but_full_break']}/{n3} = {fmt_rate(structural['winner_mainline_but_full_break_rate'])}",
        f"- 主力ラインのTOP3残存人数分布: {structural['mainline_3plus_top3_survivor_distribution']}",
        "",
        "## 市場",
        f"- 実際の3連単1番人気決着: {market['winning_trifecta_popularity_1']}/{n} = {fmt_rate(market['winning_trifecta_popularity_1_rate'])}",
        f"- 実際の3連複1番人気決着: {market['winning_trio_popularity_1']}/{n} = {fmt_rate(market['winning_trio_popularity_1_rate'])}",
        f"- A-B-C筋1点の市場人気中央値: {market['abc_straight_market_rank_median']}",
        f"- A-B-C筋1点が3連単10番人気以内: {fmt_rate(market['abc_straight_ranked_top10_rate'])}",
        f"- A=B=Cライン3車の3連複人気中央値: {market['trio_main3_market_rank_median']}",
        f"- A=B=Cが3連複3番人気以内: {fmt_rate(market['trio_main3_ranked_top3_rate'])}",
        "",
        "## 100円均等の単純シミュレーション",
    ]
    for name, s in strategies.items():
        md.append(f"- {name}: {s['hit_races']}/{s['eligible_races']}的中 ({fmt_rate(s['hit_rate'])}), 投資{s['stake_yen']:,}円 → 払戻{s['return_yen']:,}円, ROI {fmt_rate(s['roi'])}")
    md += ["", "## A/Bが1・2着なのにCだけ消えた高配当例"]
    for r in cruel_examples[:10]:
        md.append(
            f"- {r['race_date']} {r['track']} {r['race_no']}R {r['race_type']} | 主力 {r['main_line']} ({r['main_line_names']}) → {r['top3']} ({r['top3_names']}) | 3連単 {int(r['winning_trifecta_payout']):,}円 / 人気 {r['winning_trifecta_popularity']}"
        )
    md += ["", "## A-B-Cが3連単5番人気以内なのに飛んだ例"]
    for r in popular_burn_examples[:10]:
        md.append(
            f"- {r['race_date']} {r['track']} {r['race_no']}R {r['race_type']} | 主力 {r['main_line']} ({r['main_line_names']}) A-B-C人気 {r['abc1_market_rank']}位 ({r['abc1_final_odds']}倍) → 実着 {r['top3']} | 3連単 {int(r['winning_trifecta_payout']):,}円 / 人気 {r['winning_trifecta_popularity']}"
        )
    md += ["", "注: 的中/ROIは戦略推奨ではなく、保存済み確定結果に対する構造監査。"]
    (OUT / "report.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
