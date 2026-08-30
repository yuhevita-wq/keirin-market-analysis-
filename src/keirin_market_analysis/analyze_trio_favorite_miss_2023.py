from __future__ import annotations

import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "2023" / "s_class_yosen"
OUT = ROOT / "data" / "audits" / "trio_favorite_miss_2023.json"


def read_csv(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def parse_combo(s: str):
    s = s.strip().replace("=", "-").replace(",", "-")
    return tuple(sorted(int(x) for x in s.split("-") if x.strip()))


def median(xs):
    return statistics.median(xs) if xs else None


def mean(xs):
    return statistics.mean(xs) if xs else None


def quantile(xs, q):
    if not xs:
        return None
    a = sorted(xs)
    if len(a) == 1:
        return a[0]
    p = (len(a) - 1) * q
    lo = int(math.floor(p)); hi = int(math.ceil(p))
    if lo == hi:
        return a[lo]
    return a[lo] + (a[hi] - a[lo]) * (p - lo)


def summarize_values(rows, key):
    hit = [r[key] for r in rows if not r["favorite_miss"]]
    miss = [r[key] for r in rows if r["favorite_miss"]]
    return {
        "favorite_hit": {"n": len(hit), "mean": mean(hit), "median": median(hit)},
        "favorite_miss": {"n": len(miss), "mean": mean(miss), "median": median(miss)},
    }


def condition_summary(rows, pred, baseline):
    selected = [r for r in rows if pred(r)]
    misses = sum(r["favorite_miss"] for r in selected)
    rate = misses / len(selected) if selected else None
    return {
        "races": len(selected),
        "coverage_pct": 100 * len(selected) / len(rows) if rows else None,
        "misses": misses,
        "miss_rate_pct": 100 * rate if rate is not None else None,
        "lift_vs_baseline": rate / baseline if rate is not None and baseline else None,
    }


def main():
    trio_by_race = defaultdict(list)
    for r in read_csv(DATA / "trio_final_odds.csv"):
        if r.get("odds_status") != "available":
            continue
        try:
            odds = float(r["odds"])
        except Exception:
            continue
        if odds <= 0:
            continue
        trio_by_race[r["race_id"]].append({
            "combo": parse_combo(r["combination"]),
            "odds": odds,
            "market_rank": int(r["market_rank"]) if r.get("market_rank") else None,
        })

    winners_by_race = defaultdict(list)
    for r in read_csv(DATA / "payouts.csv"):
        if r.get("ticket_type") != "3連複" or r.get("status") != "paid":
            continue
        winners_by_race[r["race_id"]].append({
            "combo": parse_combo(r["combination"]),
            "payout_yen": int(r["payout_yen"]),
            "popularity": int(r["popularity"]) if r.get("popularity") else None,
        })

    rows = []
    for rid in sorted(set(trio_by_race) & set(winners_by_race)):
        tr = trio_by_race[rid]
        if not tr:
            continue
        odds_map = {x["combo"]: x["odds"] for x in tr}
        by_odds = sorted(tr, key=lambda x: (x["odds"], x["combo"]))
        favorite = by_odds[0]["combo"]
        fav_odds = by_odds[0]["odds"]
        second_odds = by_odds[1]["odds"] if len(by_odds) > 1 else fav_odds
        inv = {c: 1.0/o for c,o in odds_map.items()}
        z = sum(inv.values())
        p = {c: w/z for c,w in inv.items()}
        ranked_p = sorted(p.items(), key=lambda kv: (-kv[1], kv[0]))
        fav_share = p[favorite]
        top3_conc = sum(v for _, v in ranked_p[:3])
        ent = -sum(v * math.log(v) for v in p.values()) / math.log(len(p)) if len(p) > 1 else 0.0

        rider_support = defaultdict(float)
        for combo, w in inv.items():
            for car in combo:
                rider_support[car] += w / z
        rider_ranked = sorted(rider_support, key=lambda c: (-rider_support[c], c))
        rider_rank = {c: i+1 for i,c in enumerate(rider_ranked)}
        fav_supports = sorted((rider_support[c] for c in favorite), reverse=True)

        winners = winners_by_race[rid]
        miss = not any(w["combo"] == favorite for w in winners)
        rows.append({
            "race_id": rid,
            "favorite": favorite,
            "favorite_odds": fav_odds,
            "second_to_fav_odds_ratio": second_odds / fav_odds,
            "favorite_share": fav_share,
            "top3_concentration": top3_conc,
            "entropy": ent,
            "favorite_weakest_rider_support": fav_supports[-1],
            "favorite_support_spread": fav_supports[0] - fav_supports[-1],
            "favorite_miss": miss,
            "winners": winners,
            "odds_map": odds_map,
            "rider_rank": rider_rank,
            "rider_support": dict(rider_support),
        })

    total = len(rows)
    miss_count = sum(r["favorite_miss"] for r in rows)
    baseline = miss_count / total

    feature_directions = {
        "favorite_odds": "high",
        "second_to_fav_odds_ratio": "low",
        "favorite_share": "low",
        "top3_concentration": "low",
        "entropy": "high",
        "favorite_weakest_rider_support": "low",
        "favorite_support_spread": "high",
    }

    comparisons = {k: summarize_values(rows, k) for k in feature_directions}
    quintiles = {}
    for k in feature_directions:
        vals = [r[k] for r in rows]
        cuts = [quantile(vals, q) for q in (0.2,0.4,0.6,0.8)]
        bins = []
        for i in range(5):
            lo = -float("inf") if i == 0 else cuts[i-1]
            hi = float("inf") if i == 4 else cuts[i]
            sel = [r for r in rows if (r[k] >= lo and (r[k] < hi if i < 4 else r[k] <= hi))]
            m = sum(r["favorite_miss"] for r in sel)
            bins.append({"bin": i+1, "min": min((r[k] for r in sel), default=None), "max": max((r[k] for r in sel), default=None), "races": len(sel), "miss_rate_pct": 100*m/len(sel) if sel else None})
        quintiles[k] = bins

    scans = {}
    min_n = math.ceil(total * 0.15)
    best_defs = {}
    for k, direction in feature_directions.items():
        vals = [r[k] for r in rows]
        candidates = []
        for q in [i/10 for i in range(1,10)]:
            t = quantile(vals, q)
            if direction == "high":
                pred = lambda r, k=k, t=t: r[k] >= t
                expr = f"{k} >= {t}"
            else:
                pred = lambda r, k=k, t=t: r[k] <= t
                expr = f"{k} <= {t}"
            s = condition_summary(rows, pred, baseline)
            if s["races"] >= min_n:
                s.update({"threshold": t, "expression": expr, "quantile": q})
                candidates.append(s)
        candidates.sort(key=lambda x: (x["miss_rate_pct"], x["races"]), reverse=True)
        scans[k] = candidates
        if candidates:
            best_defs[k] = candidates[0]

    pair_conditions = []
    keys = list(best_defs)
    for a,b in combinations(keys,2):
        ca, cb = best_defs[a], best_defs[b]
        da, db = feature_directions[a], feature_directions[b]
        ta, tb = ca["threshold"], cb["threshold"]
        def pred(r, a=a,b=b,da=da,db=db,ta=ta,tb=tb):
            xa = r[a] >= ta if da == "high" else r[a] <= ta
            xb = r[b] >= tb if db == "high" else r[b] <= tb
            return xa and xb
        s = condition_summary(rows, pred, baseline)
        if s["races"] >= math.ceil(total * 0.10):
            s.update({"features": [a,b], "expressions": [ca["expression"], cb["expression"]]})
            pair_conditions.append(s)
    pair_conditions.sort(key=lambda x: (x["miss_rate_pct"], x["races"]), reverse=True)

    old_f1 = 0.1884985310418076
    old_f2 = 1.2058823529411764
    old_gate = condition_summary(rows, lambda r: r["favorite_share"] <= old_f1 and r["second_to_fav_odds_ratio"] <= old_f2, baseline)
    old_gate["expression"] = f"favorite_share <= {old_f1} AND second_to_fav_odds_ratio <= {old_f2}"

    survivor_dist = Counter()
    winning_popularity = Counter()
    popularity_buckets = Counter()
    payout_vals = []
    winning_odds_vals = []
    winning_rank_patterns = Counter()
    rider_rank_presence = Counter()
    one_replacement_outsider_rank = Counter()
    one_replacement_dropped_role = Counter()

    miss_examples = []
    for r in rows:
        if not r["favorite_miss"]:
            continue
        fav = set(r["favorite"])
        fav_strength_order = sorted(r["favorite"], key=lambda c: (-r["rider_support"][c], c))
        role = {fav_strength_order[0]: "strongest", fav_strength_order[1]: "middle", fav_strength_order[2]: "weakest"}
        for w in r["winners"]:
            combo = w["combo"]
            survivors = len(fav & set(combo))
            survivor_dist[str(survivors)] += 1
            pop = w["popularity"] or next((x["market_rank"] for x in trio_by_race[r["race_id"]] if x["combo"] == combo), None)
            if pop is not None:
                winning_popularity[str(pop)] += 1
                if pop <= 3: bucket="2-3" if pop >= 2 else "1"
                elif pop <= 5: bucket="4-5"
                elif pop <= 10: bucket="6-10"
                elif pop <= 15: bucket="11-15"
                else: bucket="16+"
                popularity_buckets[bucket] += 1
            payout_vals.append(w["payout_yen"])
            if combo in r["odds_map"]:
                winning_odds_vals.append(r["odds_map"][combo])
            ranks = tuple(sorted(r["rider_rank"][c] for c in combo))
            winning_rank_patterns["-".join(map(str,ranks))] += 1
            for rr in ranks:
                rider_rank_presence[str(rr)] += 1
            if survivors == 2:
                outsider = next(iter(set(combo)-fav))
                dropped = next(iter(fav-set(combo)))
                one_replacement_outsider_rank[str(r["rider_rank"][outsider])] += 1
                one_replacement_dropped_role[role[dropped]] += 1
            if len(miss_examples) < 20:
                miss_examples.append({
                    "race_id": r["race_id"], "favorite": list(r["favorite"]), "favorite_odds": r["favorite_odds"],
                    "favorite_share": r["favorite_share"], "second_to_fav_odds_ratio": r["second_to_fav_odds_ratio"],
                    "winner": list(combo), "winner_popularity": pop, "payout_yen": w["payout_yen"],
                    "favorite_members_survived": survivors, "winner_rider_support_ranks": list(ranks),
                })

    out = {
        "status": "TRIO_FAVORITE_MISS_2023_FULL_POPULATION",
        "year": 2023,
        "years_read": [2023],
        "population": "All exact-match S級予選 races with complete 2023 final trio odds and published trio payout labels.",
        "data_note": "Market features use historical KDreams FINAL odds, not T-10-minute snapshots. Results/payouts are labels only.",
        "baseline": {"analyzable_races": total, "favorite_miss_races": miss_count, "favorite_miss_rate_pct": 100*baseline},
        "feature_comparison_hit_vs_miss": comparisons,
        "feature_quintiles": quintiles,
        "exploratory_single_feature_threshold_scans": scans,
        "exploratory_top_pair_conditions": pair_conditions[:15],
        "old_F1_F2_benchmark_on_full_2023": old_gate,
        "miss_outcomes": {
            "favorite_members_survived_distribution": dict(survivor_dist),
            "winning_trio_popularity_exact": dict(sorted(winning_popularity.items(), key=lambda kv:int(kv[0]))),
            "winning_trio_popularity_buckets": dict(popularity_buckets),
            "payout_yen": {"mean": mean(payout_vals), "median": median(payout_vals), "min": min(payout_vals) if payout_vals else None, "max": max(payout_vals) if payout_vals else None},
            "winning_final_odds": {"mean": mean(winning_odds_vals), "median": median(winning_odds_vals), "min": min(winning_odds_vals) if winning_odds_vals else None, "max": max(winning_odds_vals) if winning_odds_vals else None},
            "winning_rider_support_rank_patterns_top20": winning_rank_patterns.most_common(20),
            "rider_support_rank_presence": dict(sorted(rider_rank_presence.items(), key=lambda kv:int(kv[0]))),
            "one_replacement_outsider_support_rank": dict(sorted(one_replacement_outsider_rank.items(), key=lambda kv:int(kv[0]))),
            "one_replacement_dropped_favorite_member_role": dict(one_replacement_dropped_role),
        },
        "miss_examples_first20": miss_examples,
        "methodology_warning": "Threshold searches are exploratory development on 2023 and must not be called validated. Do not tune on 2024/2025/2026 from this output.",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
