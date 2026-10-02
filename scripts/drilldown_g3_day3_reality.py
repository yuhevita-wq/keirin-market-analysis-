from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results" / "g3_day3_reality"
SRC = BASE / "race_level.csv"


def i(v, default=0):
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return default


def f(v, default=None):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def rate(a, b):
    return a / b if b else None


def fmt(x):
    return "n/a" if x is None else f"{x*100:.1f}%"


def strategy(rows, prefix):
    rows = [r for r in rows if i(r.get(f"{prefix}_eligible")) == 1]
    stake = sum(i(r.get(f"{prefix}_stake")) for r in rows)
    ret = sum(i(r.get(f"{prefix}_return")) for r in rows)
    hits = sum(i(r.get(f"{prefix}_hit")) for r in rows)
    return {"races": len(rows), "hits": hits, "hit_rate": rate(hits, len(rows)), "stake": stake, "return": ret, "profit": ret-stake, "roi": rate(ret, stake)}


def market_bucket(rows, prefix, rank_pred):
    xs = []
    for r in rows:
        rank = i(r.get(f"{prefix}_market_rank"), -1)
        if rank > 0 and rank_pred(rank):
            xs.append(r)
    return strategy(xs, prefix)


def main():
    with SRC.open("r", encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    n3 = [r for r in rows if i(r.get("main_line_size")) >= 3]

    core_top2 = [r for r in n3 if i(r.get("ab_top2_any_order")) == 1]
    third_broke = [r for r in core_top2 if i(r.get("core_survived_third_broke")) == 1]
    winner_main = [r for r in n3 if i(r.get("winner_in_mainline")) == 1]
    winner_main_full_broke = [r for r in winner_main if i(r.get("abc_all_top3")) == 0]
    ab_both_top3 = [r for r in n3 if i(r.get("ab_both_top3")) == 1]
    ab_both_c_missing = [r for r in ab_both_top3 if i(r.get("abc_all_top3")) == 0]

    rank_buckets = {
        "rank_1": lambda x: x == 1,
        "rank_1_3": lambda x: x <= 3,
        "rank_1_5": lambda x: x <= 5,
        "rank_1_10": lambda x: x <= 10,
        "rank_11_plus": lambda x: x >= 11,
    }
    abc_market = {k: market_bucket(n3, "abc1", pred) for k, pred in rank_buckets.items()}
    trio_market = {k: market_bucket(n3, "trio1", pred) for k, pred in rank_buckets.items()}

    by_year = {}
    years = sorted({r.get("race_date", "")[:4] for r in n3 if r.get("race_date")})
    for y in years:
        ys = [r for r in n3 if r.get("race_date", "").startswith(y)]
        by_year[y] = {
            "races": len(ys),
            "abc1": strategy(ys, "abc1"),
            "abflip2": strategy(ys, "abflip2"),
            "box6": strategy(ys, "box6"),
            "trio1": strategy(ys, "trio1"),
            "core_top2": sum(i(r.get("ab_top2_any_order")) for r in ys),
            "core_top2_third_broke": sum(i(r.get("core_survived_third_broke")) for r in ys),
        }

    by_type = {}
    type_groups = defaultdict(list)
    for r in n3:
        type_groups[r.get("race_type", "")].append(r)
    for typ, xs in sorted(type_groups.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        if len(xs) < 20:
            continue
        core = [r for r in xs if i(r.get("ab_top2_any_order")) == 1]
        broke = [r for r in core if i(r.get("core_survived_third_broke")) == 1]
        by_type[typ] = {
            "races": len(xs),
            "abc1": strategy(xs, "abc1"),
            "abflip2": strategy(xs, "abflip2"),
            "box6": strategy(xs, "box6"),
            "trio1": strategy(xs, "trio1"),
            "core_top2": len(core),
            "core_top2_third_broke": len(broke),
            "third_break_given_core_top2": rate(len(broke), len(core)),
        }

    # Rank-1 ABC misses, for the purest 'favorite line burned' sample.
    rank1 = [r for r in n3 if i(r.get("abc1_market_rank"), -1) == 1]
    rank1_misses = [r for r in rank1 if i(r.get("abc1_hit")) == 0]
    rank1_misses.sort(key=lambda r: i(r.get("winning_trifecta_payout")), reverse=True)
    rank1_examples = [
        {k: r.get(k) for k in (
            "race_date","track","race_no","race_type","main_line","main_line_names","abc1_final_odds",
            "top3","top3_names","winning_trifecta_payout","winning_trifecta_popularity",
            "winner_in_mainline","ab_both_top3","ab_top2_any_order"
        )}
        for r in rank1_misses[:20]
    ]

    out = {
        "conditional_reality": {
            "core_AB_top2_races": len(core_top2),
            "C_replaced_when_core_AB_top2": len(third_broke),
            "C_replaced_rate_given_core_AB_top2": rate(len(third_broke), len(core_top2)),
            "AB_both_top3_races": len(ab_both_top3),
            "ABC_not_full_when_AB_both_top3": len(ab_both_c_missing),
            "ABC_not_full_rate_given_AB_both_top3": rate(len(ab_both_c_missing), len(ab_both_top3)),
            "winner_in_mainline_3plus_races": len(winner_main),
            "ABC_full_failed_despite_mainline_winner": len(winner_main_full_broke),
            "ABC_full_failed_rate_given_mainline_winner": rate(len(winner_main_full_broke), len(winner_main)),
        },
        "abc_market_rank_buckets": abc_market,
        "trio_market_rank_buckets": trio_market,
        "by_year": by_year,
        "by_race_type_min20": by_type,
        "abc_market_rank1_miss_examples": rank1_examples,
    }
    (BASE / "drilldown.json").write_text(json.dumps(out, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")

    lines = ["# G3三日目 追加分解", "", "## 読みは合って車券が死ぬ"]
    c = out["conditional_reality"]
    lines += [
        f"- A/Bが実際に1・2着: {c['core_AB_top2_races']}R。そのうちCが3着から消えた: {c['C_replaced_when_core_AB_top2']}R = {fmt(c['C_replaced_rate_given_core_AB_top2'])}",
        f"- A/Bがともに3着以内: {c['AB_both_top3_races']}R。そのうちA/B/Cの3人独占にならなかった: {c['ABC_not_full_when_AB_both_top3']}R = {fmt(c['ABC_not_full_rate_given_AB_both_top3'])}",
        f"- 勝者が主力ラインだった3車以上ライン: {c['winner_in_mainline_3plus_races']}R。そのうちA/B/C独占失敗: {c['ABC_full_failed_despite_mainline_winner']}R = {fmt(c['ABC_full_failed_rate_given_mainline_winner'])}",
        "", "## A-B-C筋1点 市場人気別"
    ]
    for k, s in abc_market.items():
        lines.append(f"- {k}: {s['hits']}/{s['races']}的中 ({fmt(s['hit_rate'])}), 投資{s['stake']:,}円→払戻{s['return']:,}円, ROI {fmt(s['roi'])}")
    lines += ["", "## A=B=C 3連複 市場人気別"]
    for k, s in trio_market.items():
        lines.append(f"- {k}: {s['hits']}/{s['races']}的中 ({fmt(s['hit_rate'])}), 投資{s['stake']:,}円→払戻{s['return']:,}円, ROI {fmt(s['roi'])}")
    lines += ["", "## 年別"]
    for y, d in by_year.items():
        lines.append(f"- {y}: {d['races']}R | ABC1 ROI {fmt(d['abc1']['roi'])} | AB折返2 ROI {fmt(d['abflip2']['roi'])} | BOX6 ROI {fmt(d['box6']['roi'])} | 3連複 ROI {fmt(d['trio1']['roi'])}")
    lines += ["", "## レース種別（20R以上）"]
    for typ, d in by_type.items():
        lines.append(f"- {typ}: {d['races']}R | A/B 1-2時C消失 {d['core_top2_third_broke']}/{d['core_top2']}={fmt(d['third_break_given_core_top2'])} | ABC1 ROI {fmt(d['abc1']['roi'])} | BOX6 ROI {fmt(d['box6']['roi'])} | 3連複 ROI {fmt(d['trio1']['roi'])}")
    lines += ["", "## A-B-Cが3連単1番人気なのに死亡した高配当例"]
    for r in rank1_examples[:10]:
        lines.append(f"- {r['race_date']} {r['track']} {r['race_no']}R {r['race_type']} | 主力 {r['main_line']} ({r['main_line_names']}) 1番人気 {r['abc1_final_odds']}倍 → {r['top3']} ({r['top3_names']}) | 払戻 {i(r['winning_trifecta_payout']):,}円 / {r['winning_trifecta_popularity']}人気")
    (BASE / "drilldown.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
