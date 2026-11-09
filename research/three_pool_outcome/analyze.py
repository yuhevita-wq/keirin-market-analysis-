#!/usr/bin/env python3
"""Prospective-cutoff audit: 2車複 / 2車単 / 3連複 odds movement versus actual payouts.

The observable object is a conditional association, NOT identified causality.
No final odds, winner, or future information is used to define a signal.
Prices at T-15 and T-5 are actual archived timestamped observations.
All figures apply only to tickets observed at both times; never fill top-50 tails.
"""
import argparse
import csv
import gzip
import json
import math
import random
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from itertools import combinations, permutations
from pathlib import Path

JST = timezone(timedelta(hours=9))
POOLS = ("umaren", "umatan", "trio")
NAMES = {"umaren": "2車複", "umatan": "2車単", "trio": "3連複"}
CUTOFF_STALENESS_SECONDS = 120
SHORT = math.log(0.95)
DRIFT = math.log(1.05)
BINS = (5, 10, 20, 50, 100, 200, 500)
PHASE_DATE = "2026-07-01"


def canonical(pool, ticket):
    parts = str(ticket).replace("=", "-").split("-")
    need = 3 if pool == "trio" else 2
    if len(parts) != need or any(not s.isdigit() for s in parts):
        raise ValueError("invalid ticket")
    vals = [int(v) for v in parts]
    if len(set(vals)) != need or not all(1 <= v <= 9 for v in vals):
        raise ValueError("invalid distinct selection")
    if pool != "umatan":
        vals.sort()
    return "-".join(map(str, vals))


def price_band(price):
    for bound in BINS:
        if price < bound:
            return f"<{bound}"
    return "500+"


def movement(pre, late):
    if not all(math.isfinite(v) and v >= 1 for v in (pre, late)):
        raise ValueError("invalid observed odds")
    return math.log(late / pre)


def signal(chg):
    return "short" if chg <= SHORT else ("drift" if chg >= DRIFT else "stable")


def observe(record, pool, target_minute):
    official = record.get("official") or {}
    date = str(record.get("race_date"))
    deadline_text = official.get("deadline")
    if not isinstance(deadline_text, str) or not deadline_text.startswith(date):
        return None, "no_verified_deadline"
    try:
        deadline = datetime.fromisoformat(deadline_text)
        if deadline.utcoffset() != timedelta(hours=9):
            return None, "bad_cutoff_timezone"
        target = deadline - timedelta(minutes=target_minute)
    except (ValueError, TypeError):
        return None, "invalid_cutoff"
    options = []
    for frame in (record.get("pools", {}).get(pool, {}).get("frames") or []):
        if frame.get("final"):
            continue
        label = frame.get("label")
        if not isinstance(label, str) or len(label) != 5 or label[2] != ":":
            continue
        try:
            h, m = map(int, label.split(":"))
            observed = deadline.replace(hour=h, minute=m, second=0, microsecond=0)
            stale = (target - observed).total_seconds()
            if 0 <= stale <= CUTOFF_STALENESS_SECONDS and isinstance(frame.get("odds"), dict):
                options.append((observed, frame["odds"]))
        except ValueError:
            pass
    if not options:
        return None, f"tminus{target_minute}_not_observed"
    raw = max(options, key=lambda z: z[0])[1]
    odds = {}
    try:
        for key, val in raw.items():
            norm = canonical(pool, key)
            num = float(val)
            if not math.isfinite(num) or num < 1 or norm in odds:
                return None, "invalid_or_duplicate_quote"
            odds[norm] = num
    except (ValueError, TypeError):
        return None, "invalid_or_duplicate_quote"
    return odds, "ok"


def three_pool_quotes(r):
    official = r.get("official") or {}
    result = {}
    errors = {}
    for pool in POOLS:
        if official.get("status") != "complete" or pool not in official.get("complete_pools", []):
            errors[pool] = "official_payout_unverified"
            continue
        payouts = (official.get("payouts_per_100_yen") or {}).get(pool)
        if not isinstance(payouts, dict) or not payouts:
            errors[pool] = "payout_missing"
            continue
        at15, reason15 = observe(r, pool, 15)
        at5, reason5 = observe(r, pool, 5)
        if at15 is None or at5 is None:
            errors[pool] = reason15 if at15 is None else reason5
            continue
        try:
            payouts = {canonical(pool, k): int(v) for k, v in payouts.items()}
            if any(v <= 0 for v in payouts.values()):
                raise ValueError("bad payout")
        except (ValueError, TypeError):
            errors[pool] = "payout_invalid"
            continue
        common = at15.keys() & at5.keys()
        if not common:
            errors[pool] = "no_same_ticket_at_both_times"
            continue
        # Filtering requires both observable quotes only, never final-market membership.
        result[pool] = {
            key: (at5[key], movement(at15[key], at5[key]),
                  float(payouts.get(key, 0)) / 100.0, int(key in payouts))
            for key in common
        }
        errors[pool] = "ok"
    return result, errors


def category_add(aggregate, key, hit, gross):
    row = aggregate[key]
    row[0] += 1
    row[1] += hit
    row[2] += gross


def new_stats():
    return defaultdict(lambda: [0, 0, 0.0])


def normalize_summary(stats):
    output = {}
    for k, values in sorted(stats.items()):
        n, hits, gross = values
        output[k] = {"tickets": n, "hits": hits,
                     "hit_rate_pct": round(100 * hits / n, 5) if n else None,
                     "roi_pct": round(100 * gross / n, 5) if n else None}
    return output


def summarize_buckets(buckets):
    stats = new_stats()
    for (phase, market, day, band, cars, cat), values in buckets.items():
        out = stats[(phase, market, cat)]
        for i in range(3):
            out[i] += values[i]
    return {phase: {
        pool: normalize_summary({cat: vals for (p, q, cat), vals in stats.items()
                                  if p == phase and q == pool})
        for pool in POOLS}
        for phase in ("development", "holdout")}


def strat_diff(day_buckets, pool, phase, left="short", right="drift",
               boots=200, seed=2109):
    days = sorted({d for p, m, d, _, _, _ in day_buckets
                   if p == phase and m == pool})
    if len(days) < 15:
        return {"status": "insufficient_days", "days": len(days)}
    groups = defaultdict(list)
    for (p, m, day, price, cars, category), vals in day_buckets.items():
        if p == phase and m == pool and category in (left, right):
            groups[day].append(((price, cars, day[:7], category), vals))

    def effect(draw):
        by_stratum = defaultdict(lambda: [0, 0, 0.0])
        for day in draw:
            for key, val in groups[day]:
                current = by_stratum[key]
                current[0] += val[0]
                current[1] += val[1]
                current[2] += val[2]
        total_weight = 0.0
        sum_hit = 0.0
        sum_roi = 0.0
        eligible = 0
        total_tickets = 0
        strata = {(price, cars, ym) for price, cars, ym, _ in by_stratum}
        for s in strata:
            a = by_stratum.get(s + (left,), (0, 0, 0.0))
            b = by_stratum.get(s + (right,), (0, 0, 0.0))
            if min(a[0], b[0]) < 8:
                continue
            w = a[0] * b[0] / (a[0] + b[0])
            total_weight += w
            sum_hit += w * (a[1] / a[0] - b[1] / b[0])
            sum_roi += w * (a[2] / a[0] - b[2] / b[0])
            eligible += 1
            total_tickets += a[0] + b[0]
        if not total_weight:
            return None
        return (100 * sum_hit / total_weight,
                100 * sum_roi / total_weight, eligible, total_tickets)
    estimate = effect(days)
    if estimate is None:
        return {"status": "no_matched_price_car_month_strata", "days": len(days)}
    rng = random.Random(seed)
    reps = []
    for _ in range(boots):
        drawn = [days[rng.randrange(len(days))] for _ in days]
        got = effect(drawn)
        if got is not None:
            reps.append(got)
    def interval(i):
        if len(reps) < max(50, boots // 2):
            return None
        z = sorted(x[i] for x in reps)
        return [round(z[int(0.025 * (len(z)-1))], 5),
                round(z[int(0.975 * (len(z)-1))], 5)]
    return {"status": "estimated", "days": len(days), "matched_strata": estimate[2],
            "matched_ticket_rows": estimate[3], "contrast": f"{left}_minus_{right}",
            "matched_odd_bin_car_count_month": True,
            "hit_rate_difference_points": round(estimate[0], 5),
            "hit_rate_cluster_bootstrap_95ci": interval(0),
            "roi_difference_points": round(estimate[1], 5),
            "roi_cluster_bootstrap_95ci": interval(1),
            "bootstrap_draws": len(reps)}


def cross_market_category(trio_delta, pair_delta, order_delta):
    s = (signal(trio_delta), signal(pair_delta), signal(order_delta))
    if s == ("short", "short", "short"):
        return "all_short"
    if s == ("drift", "drift", "drift"):
        return "all_drift"
    if s[0] != "short" and s[1] == "short" and s[2] == "short":
        return "other_two_short_trio_not"
    if s[0] != "drift" and s[1] == "drift" and s[2] == "drift":
        return "other_two_drift_trio_not"
    return "other_mixed"


def research(root, out, boots):
    root = Path(root)
    files = sorted(root.rglob("????-??-??.json.gz"))
    if not files:
        raise ValueError("No downloaded archive files found")
    daily = new_stats()
    joint_daily = new_stats()
    counters = Counter()
    seen_races = set()
    date_range = [None, None]
    archived_coverage = Counter()
    joint_examples = defaultdict(list)
    for path in files:
        payload = json.loads(gzip.decompress(path.read_bytes()))
        if not isinstance(payload, dict):
            raise ValueError("Archive file must be date-keyed dict: " + str(path))
        for r in payload.values():
            day = r["race_date"]
            ident = (day, r["track"], str(r["race_no"]))
            if ident in seen_races:
                raise ValueError("Duplicate race across archives: " + repr(ident))
            seen_races.add(ident)
            date_range[0] = min(day, date_range[0]) if date_range[0] else day
            date_range[1] = max(day, date_range[1]) if date_range[1] else day
            counters["all_races"] += 1
            phase = "development" if day < PHASE_DATE else "holdout"
            cars = len(r.get("cars", []))
            if not (5 <= cars <= 9):
                counters["excluded_unknown_car_count"] += 1
                continue
            books, reasons = three_pool_quotes(r)
            for pool in POOLS:
                archived_coverage[(pool, reasons[pool])] += 1
                if pool not in books:
                    continue
                counters[phase + "_" + pool + "_races"] += 1
                for ticket, (price, delta, gross, hit) in books[pool].items():
                    category_add(daily, (phase, pool, day, price_band(price), cars, signal(delta)), hit, gross)
                    counters[phase + "_" + pool + "_tickets"] += 1
            if not all(p in books for p in POOLS):
                continue
            trio_book, pair_book, order_book = books["trio"], books["umaren"], books["umatan"]
            for triple, (price, trio_d, gross, hit) in trio_book.items():
                abc = tuple(map(int, triple.split("-")))
                pair_keys = ["-".join(map(str, pair)) for pair in combinations(abc, 2)]
                order_keys = ["-".join(map(str, pair)) for pair in permutations(abc, 2)]
                if any(k not in pair_book for k in pair_keys) or any(k not in order_book for k in order_keys):
                    counters["joint_missing_all_three_pairs_or_six_ordered"] += 1
                    continue
                pair_d = sum(pair_book[k][1] for k in pair_keys) / 3
                order_d = sum(order_book[k][1] for k in order_keys) / 6
                cat = cross_market_category(trio_d, pair_d, order_d)
                category_add(joint_daily, (phase, "joint", day, price_band(price), cars, cat), hit, gross)
                counters[phase + "_joint_tickets"] += 1
                if len(joint_examples[(phase, cat)]) < 3:
                    joint_examples[(phase, cat)].append({
                        "date":day, "track":r["track"], "race":r["race_no"], "trio":triple,
                        "trio_t5":price, "trio_delta_pct":round(100*(math.exp(trio_d)-1),2),
                        "pairs_mean_delta_pct":round(100*(math.exp(pair_d)-1),2),
                        "ordered_mean_delta_pct":round(100*(math.exp(order_d)-1),2),
                        "hit":bool(hit)})

    by_market = summarize_buckets(daily)
    joint_stats = new_stats()
    for (phase, market, day, band, cars, cat), vals in joint_daily.items():
        z = joint_stats[(phase, cat)]
        for i in range(3):
            z[i] += vals[i]
    joint_summary = {phase:normalize_summary({cat: v for (p,cat),v in joint_stats.items() if p==phase})
                     for phase in ("development","holdout")}
    contrasts = {}
    for phase in ("development", "holdout"):
        for pool in POOLS:
            contrasts[phase + "_" + pool] = strat_diff(daily, pool, phase, boots=boots)
        for label, left, right in [
            ("cross_all_short_vs_all_drift", "all_short", "all_drift"),
            ("cross_other_two_short_vs_other_two_drift",
             "other_two_short_trio_not", "other_two_drift_trio_not")]:
            contrasts[phase + "_" + label] = strat_diff(
                joint_daily, "joint", phase, left, right, boots=boots, seed=2046)
    report = {
        "research_question": "Does the observed T-15 to T-5 price path predict later official outcomes beyond T-5 price level, and do movements agree across the three markets?",
        "interpretation": "Observational, conditional associations; cannot establish a causal law or executable betting edge.",
        "data": {"files":len(files),"unique_races":len(seen_races),
                 "start":date_range[0],"end":date_range[1],
                 "phase_boundary":PHASE_DATE,"quality_reasons":{
                     pool:{reason:count for (p,reason),count in archived_coverage.items() if p==pool}
                     for pool in POOLS},"sample_counts":dict(counters)},
        "methods": {"price_points": "last source observation T-15 and T-5, each at most 120 seconds stale; source HH:MM is not retrieval time",
                    "pool_definitions":NAMES,
                    "change":"log(Tminus5_odds/Tminus15_odds)",
                    "short":"Tminus5/Tminus15 <=0.95",
                    "drift":"Tminus5/Tminus15 >=1.05",
                    "gross_return":"official yen payout per 100 yen / 100; if losing 0; ROI reported as gross return percentage, not net profit",
                    "outcome":"Official final results only, never used in eligibility or signal creation",
                    "eligibility":"Pool official status complete, valid cutoff, same ticket observed at both preclose times. Never require final-market top50 membership.",
                    "joint":"3連複 ticket and its 3 2車複 pairs and all 6 directed 2車単 tickets observed at both times; mean log movement for each component.",
                    "stratification":"Pre-close T-5 price band, entrants, calendar month; harmonic-weighted contrast of short versus drift; 8+ tickets in each group per stratum.",
                    "ci":"Cluster bootstrap resampling whole dates; not individual tickets; 2.5%-97.5% empirical intervals",
                    "caveats":["Limited top50 truncated markets", "Nonrandom availability of exact timestamps",
                               "Final settlement payout is not guaranteed as executable at T-5",
                               "Results can correlate with market information and common external variables",
                               "Development and holdout are chronological within existing archived dates",
                               "Multiple exploratory contrasts are not corrected for selection"]},
        "pool_direction_summary":by_market,
        "joint_three_market_summary":joint_summary,
        "conditional_matched_contrasts":contrasts,
        "limited_joint_examples":{f"{phase}_{cat}":vals for (phase,cat),vals in joint_examples.items()},
    }
    out = Path(out)
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("REPORT_PATH",str(out),flush=True)
    print(json.dumps({k:report[k] for k in (
        "research_question", "interpretation", "data", "pool_direction_summary",
        "joint_three_market_summary", "conditional_matched_contrasts")},
        ensure_ascii=False),flush=True)
    return report


def self_test():
    # A positive payout is observed only when an official combination wins.
    def record():
        def mk(keys):
            return {"frames":[{"label":"12:40","final":False,"odds":{k:10.0 for k in keys}},
                              {"label":"12:50","final":False,"odds":{k:8.0 for k in keys}},
                              {"label":"最終","final":True,"odds":{k:2.0 for k in keys}}]}
        return {"race_date":"2026-05-04","track":"TEST","race_no":1,"cars":list(range(1,8)),
          "pools":{"umaren":mk(["1-2"]),"umatan":mk(["1-2"]),
                   "trio":mk(["1-2-3"])},
          "official":{"status":"complete","deadline":"2026-05-04T12:55:00+09:00",
            "complete_pools":list(POOLS),
            "payouts_per_100_yen":{"umaren":{"1-2":550},"umatan":{"1-2":800},
                                   "trio":{"1-2-3":1600}}}}
    r=record()
    # 12:40 is T-15, 12:50 is T-5; final price 2.0 must never enter changes.
    books,reasons=three_pool_quotes(r)
    assert all(reasons[p]=="ok" for p in POOLS),reasons
    assert all(len(books[p])==1 for p in POOLS)
    assert books["trio"]["1-2-3"][0]==8.0
    assert abs(books["trio"]["1-2-3"][1]-math.log(.8))<1e-9
    assert books["trio"]["1-2-3"][2]==16.0
    assert cross_market_category(-.1,-.12,-.2)=="all_short"
    r["pools"]["trio"]["frames"][1]["label"]="12:53"
    b,e=three_pool_quotes(r)
    assert "trio" not in b and e["trio"]=="tminus5_not_observed"
    print("PASS synthetic cutoff, finalized payout, no future-price leakage, truncated ticket safety",flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--input",default="work/path_odds")
    parser.add_argument("--out",default="work/three_pool_outcome/report.json")
    parser.add_argument("--bootstrap",type=int,default=200)
    parser.add_argument("--self-test",action="store_true")
    args=parser.parse_args()
    if args.self_test:
        self_test()
    else:
        research(args.input,args.out,args.bootstrap)


if __name__=="__main__":
    main()
