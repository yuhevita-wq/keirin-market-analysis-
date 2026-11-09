#!/usr/bin/env python3
"""Flat-buy discovery of ticket-generating policies, conditional race filters and OOS tests.

V2 deliberately does not use the V1 profitable-single-atom gate.
Only features known before the finish may select wagers. See README for coverage limits.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import itertools
import json
import math
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TYPES = ("3連単", "3連複")
GIRLS = re.compile("ガールズ|女子|L級|ガ予|ガ決|ガ一般|ガ選|GIRLS", re.I)


def norm(s):
    return unicodedata.normalize("NFKC", str(s or "")).strip()


def race_day(rid):
    rid = str(rid)
    if not re.fullmatch(r"\d{16}", rid):
        return None
    try:
        meeting_day = int(rid[10:12])
        if not 1 <= meeting_day <= 7:
            return None
        return date(int(rid[2:6]), int(rid[6:8]), int(rid[8:10])) + timedelta(days=meeting_day - 1)
    except ValueError:
        return None


def gz_json(path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def stage(label):
    s = norm(label)
    for part, v in (("初日特選", "初日特選"), ("準決", "準決勝"), ("決勝", "決勝"),
                    ("二次予", "二次予選"), ("二予", "二次予選"),
                    ("一次予", "一次予選"), ("一予", "一次予選"),
                    ("特予", "予選"), ("予選", "予選"), ("特一般", "一般"),
                    ("一般", "一般"), ("選抜", "選抜"), ("特選", "特選")):
        if part in s:
            return v
    return "その他"


def price_band(n):
    for a in (3.5, 6, 10, 20, 30, 50, 100, 200, 500):
        if n < a:
            return "<" + str(a)
    return ">=500"


def race_band(n):
    for a in (4, 7, 10, 15, 25, 50):
        if n < a:
            return "<" + str(a)
    return ">=50"


def number(value):
    try:
        s = norm(value).replace(",", "")
        x = float(s)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None


@dataclass
class Stat:
    races: int = 0
    bets: int = 0
    hits: int = 0
    payout: int = 0
    largest_win: int = 0

    def add(self, n, hit, pay):
        self.races += 1
        self.bets += n
        self.hits += hit
        self.payout += pay
        self.largest_win = max(self.largest_win, pay)

    def result(self):
        invested = self.bets * 100
        return {
            "races": self.races, "bets": self.bets, "hits": self.hits,
            "stake_yen": invested, "return_yen": self.payout,
            "profit_yen": self.payout - invested,
            "roi": round(self.payout / invested, 6) if invested else None,
            "largest_single_race_win_share": round(self.largest_win / self.payout, 6)
            if self.payout else None,
        }


def total(values):
    s = Stat()
    for x in values:
        s.races += x.races
        s.bets += x.bets
        s.hits += x.hits
        s.payout += x.payout
        s.largest_win = max(s.largest_win, x.largest_win)
    return s


def read_result_files(root, first, last):
    by_year = {}
    for y in range(first, last + 1):
        p = root / "data/results_only" / f"results_{y}.jsonl.gz"
        if not p.exists():
            raise FileNotFoundError(p)
        result = {}
        for r in gz_json(p):
            rid = str(r["race_id"])
            d = race_day(rid)
            if d is None or d.year != y:
                raise ValueError(f"bad finish date: {rid}")
            finish = tuple(map(int, r["finish_order"]))
            if len(finish) != 3 or len(set(finish)) != 3 or min(finish) < 1 or max(finish) > 9:
                raise ValueError(f"bad finish order: {rid}")
            if rid in result and result[rid] != finish:
                raise ValueError(f"conflicting results: {rid}")
            result[rid] = finish
        by_year[y] = result
    return by_year


def load_riders(root, years):
    """Only pre-race entry columns. Archive subset, never infer missing line IDs."""
    book = {}
    conflicts = set()
    folders = []
    base = root / "data"
    for y in years:
        if y in (2023, 2024, 2025):
            folders.extend(sorted((base / str(y) / "s_class_f1_all_parts").glob("*/entries.csv")))
            p = base / str(y) / "s_class_yosen" / "entries.csv"
            if p.exists():
                folders.append(p)
        elif y == 2026:
            folders.extend(sorted((base / "2026_h1").glob("*/entries.csv")))
        for folder in ("a_class_12", "a3_challenge"):
            folders.extend(sorted((base / folder).glob(f"{y}_*/entries.csv")))
    for p in folders:
        with p.open("r", encoding="utf-8-sig", newline="") as handle:
            for r in csv.DictReader(handle):
                rid = str(r.get("race_id", ""))
                day = race_day(rid)
                if day is None or day.year not in years:
                    continue
                car = norm(r.get("car_no"))
                if not car.isdigit() or not 1 <= int(car) <= 9:
                    continue
                key = (rid, int(car))
                # The field must be present in a true pre-race entry record.
                line = norm(r.get("line_id"))
                pos = norm(r.get("line_position"))
                age = number(r.get("age"))
                score = number(r.get("score"))
                bcount = number(r.get("b_count"))
                old = book.get(key)
                x = {"line": line if line.isdigit() else "",
                     "pos": int(pos) if pos.isdigit() and int(pos) >= 1 else None,
                     "age": age, "score": score, "bcount": bcount}
                if old is not None:
                    if old["line"] and x["line"] and old["line"] != x["line"]:
                        conflicts.add(rid)
                    if old["pos"] and x["pos"] and old["pos"] != x["pos"]:
                        conflicts.add(rid)
                    for k, v in x.items():
                        if old[k] in ("", None) and v not in ("", None):
                            old[k] = v
                else:
                    book[key] = x
    # Avoid assigning an incorrect line for a conflicted race.
    for rid in conflicts:
        for key, row in list(book.items()):
            if key[0] == rid:
                row["line"] = ""
    race_riders = defaultdict(dict)
    for (rid, car), row in book.items():
        race_riders[rid][car] = row
    return race_riders, {"source_entry_files": len(folders),
                         "races_with_any_entry": len(race_riders),
                         "conflicting_line_races": len(conflicts),
                         "note": "2021-22 have no compatible entry archive; 2023-26 source coverage is partial"}


def parse_market(raw, wager):
    if not isinstance(raw, dict) or not raw:
        return None
    entries = {}
    cars = set()
    for text, value in raw.items():
        pieces = re.findall(r"\d+", str(text))
        if len(pieces) != 3:
            return None
        ids = tuple(map(int, pieces))
        if len(set(ids)) != 3 or min(ids) < 1 or max(ids) > 9:
            return None
        key = tuple(sorted(ids)) if wager == "3連複" else ids
        odd = number(value)
        if odd is None or odd <= 0 or key in entries:
            return None
        entries[key] = odd
        cars.update(ids)
    n = len(cars)
    expected = math.perm(n, 3) if wager == "3連単" else math.comb(n, 3)
    if n not in (5, 6, 7, 8, 9) or len(entries) != expected:
        return None
    return sorted(entries.items(), key=lambda item: (item[1], item[0]))


def valid_lines(riders, cars):
    if len(riders) != len(cars) or set(riders) != set(cars):
        return None
    if any(not riders[c]["line"] or riders[c]["pos"] is None for c in cars):
        return None
    groups = defaultdict(list)
    for c in cars:
        groups[riders[c]["line"]].append((riders[c]["pos"], c))
    lines = []
    for g in groups.values():
        g.sort()
        if [p for p, _ in g] != list(range(1, len(g) + 1)):
            return None
        lines.append([c for _, c in g])
    return sorted(lines, key=lambda xs: (-len(xs), xs))


def race_features(race_type, parsed, lines, riders):
    wager = next(iter(parsed))
    items = parsed[wager]
    cars = {c for combo, odd in items for c in combo}
    top = items[0][1]
    p10 = items[min(len(items)-1, 9)][1]
    s = norm(race_type).upper()
    grade = re.search(r"(^|/)(GP|G[123]|F[12])(/|$)", s)
    result = {
        f"n={len(cars)}",
        f"stage={stage(race_type)}",
        f"fav_odds={race_band(top)}",
        f"spread10={race_band(p10 / top)}",
        f"wager={wager}",
    }
    if grade:
        result.add("grade=" + grade.group(2))
    if lines is not None:
        sizes = "".join(map(str, sorted((len(x) for x in lines), reverse=True)))
        result.update((f"line_shape={sizes}", f"line_max={len(lines[0])}",
                       f"line_count={len(lines)}"))
    if riders and set(riders) == cars:
        ages = [riders[c]["age"] for c in cars if riders[c]["age"] is not None]
        scores = [riders[c]["score"] for c in cars if riders[c]["score"] is not None]
        if len(ages) == len(cars):
            result.add(f"youngest_age={'<30' if min(ages) < 30 else '>=30'}")
            result.add(f"age_span={'<10' if max(ages)-min(ages) < 10 else '>=10'}")
        if len(scores) == len(cars):
            gap = max(scores)-min(scores)
            result.add(f"score_spread={'<5' if gap < 5 else '<10' if gap < 10 else '>=10'}")
            result.add(f"top_score={'<95' if max(scores) < 95 else '<110' if max(scores) < 110 else '>=110'}")
        if lines is not None and len(lines) >= 2:
            leader_scores = [riders[x[0]]["score"] for x in lines]
            if all(x is not None for x in leader_scores):
                gap = max(leader_scores)-min(leader_scores)
                result.add(f"line_leader_gap={'<5' if gap < 5 else '<10' if gap < 10 else '>=10'}")
    return sorted(result)


def select_policies(wager, items, winner, lines, riders=None):
    """Return diverse pre-race policy bets, not every ticket in the book."""
    if wager == "3連単":
        schemes = {
            "top1": items[:1],
            "top3": items[:3],
            "top6": items[:6],
            "top12": items[:12],
            "top24": items[:24],
            "rank4_12": items[3:12],
            "rank13_30": items[12:30],
            "rank31_70": items[30:70],
        }
        for low, high in ((3.5, 10), (10, 30), (30, 100), (100, 300), (300, 1000)):
            schemes[f"odds_{low:g}_{high:g}"] = [x for x in items if low <= x[1] < high]
        head_weights = Counter()
        for combo, odd in items:
            head_weights[combo[0]] += 1 / odd
        best_heads = [c for c, _ in sorted(head_weights.items(),
                                            key=lambda x: (-x[1], x[0]))]
        schemes["head_top1_best6"] = [x for x in items if x[0][0] == best_heads[0]][:6]
        schemes["head_top2_best6"] = [x for x in items if x[0][0] in best_heads[:2]][:6]
        schemes["head_outside_top2_best6"] = [x for x in items if x[0][0] not in best_heads[:2]][:6]
        cars_in_book = {c for combo, _ in items for c in combo}
        if riders and set(riders) == cars_in_book:
            scores = {c: riders[c]["score"] for c in cars_in_book}
            ages = {c: riders[c]["age"] for c in cars_in_book}
            if all(v is not None for v in scores.values()):
                ordered = sorted(cars_in_book, key=lambda c: (-scores[c], c))
                schemes["score_top1_head_best6"] = [x for x in items if x[0][0] == ordered[0]][:6]
                schemes["score_top2_head_best8"] = [x for x in items if x[0][0] in ordered[:2]][:8]
                schemes["score_bottom2_head_best6"] = [x for x in items if x[0][0] in ordered[-2:]][:6]
                schemes["score_top_two_pair_best8"] = [
                    x for x in items if set(x[0][:2]) == set(ordered[:2])
                ][:8]
            if all(v is not None for v in ages.values()):
                ordered_age = sorted(cars_in_book, key=lambda c: (ages[c], c))
                schemes["youngest_head_best6"] = [x for x in items if x[0][0] == ordered_age[0]][:6]
                schemes["oldest_head_best6"] = [x for x in items if x[0][0] == ordered_age[-1]][:6]
        if lines is not None:
            group_of = {c: k for k, line in enumerate(lines) for c in line}
            position = {c: p+1 for line in lines for p, c in enumerate(line)}
            filters = {
                "line_pair_1st2nd": lambda c: group_of[c[0]] == group_of[c[1]],
                "cross_pair_1st2nd": lambda c: group_of[c[0]] != group_of[c[1]],
                "line_head_wins": lambda c: position[c[0]] == 1,
                "line_bante_wins": lambda c: position[c[0]] == 2,
                "line_3same": lambda c: group_of[c[0]] == group_of[c[1]] == group_of[c[2]],
                "line_pair_cross3": lambda c: group_of[c[0]] == group_of[c[1]] != group_of[c[2]],
                "cross_1st2nd_line3": lambda c: group_of[c[0]] != group_of[c[1]] == group_of[c[2]],
                "line_1st3rd": lambda c: group_of[c[0]] == group_of[c[2]] != group_of[c[1]],
            }
            for name, fn in filters.items():
                schemes[name+"_best6"] = [x for x in items if fn(x[0])][:6]
            main = lines[0]
            if len(main) >= 2:
                schemes["mainline_head_bante_best8"] = [
                    x for x in items if x[0][:2] in ((main[0], main[1]), (main[1], main[0]))
                ][:8]
                schemes["mainline_head_best8"] = [x for x in items if x[0][0] == main[0]][:8]
    else:
        schemes = {
            "top1": items[:1], "top3": items[:3], "top6": items[:6],
            "top12": items[:12], "rank4_12": items[3:12],
        }
        for low, high in ((3.5, 10), (10, 30), (30, 100), (100, 300)):
            schemes[f"odds_{low:g}_{high:g}"] = [x for x in items if low <= x[1] < high]
        if lines is not None:
            group_of = {c: k for k, line in enumerate(lines) for c in line}
            schemes["at_least_2same_best6"] = [
                x for x in items if len({group_of[c] for c in x[0]}) < 3
            ][:6]
            schemes["all_different_best6"] = [
                x for x in items if len({group_of[c] for c in x[0]}) == 3
            ][:6]
            schemes["all_same_best6"] = [
                x for x in items if len({group_of[c] for c in x[0]}) == 1
            ][:6]
    for name, selection in schemes.items():
        if not selection:
            continue
        won = [odd for combo, odd in selection if combo == winner]
        yield f"{wager}/{name}", selection, len(won), int(round(100 * won[0])) if won else 0


def field_cells(features, max_conditions=3):
    """Every dimensionally valid subset up to depth; never gate on single-feature ROI."""
    yield ()
    for depth in range(1, max_conditions + 1):
        for conditions in itertools.combinations(features, depth):
            if len({x.split("=", 1)[0] for x in conditions}) == depth:
                yield conditions


def scan(root, first, last, rider_index, max_conditions=3):
    results = read_result_files(root, first, last)
    stats = defaultdict(lambda: defaultdict(Stat))
    audit = {}
    for year in range(first, last + 1):
        p = root / "data/odds_only" / f"odds_{year}.jsonl.gz"
        if not p.exists():
            raise FileNotFoundError(p)
        seen = set()
        flags = Counter()
        for r in gz_json(p):
            rid = str(r.get("race_id", ""))
            if rid in seen:
                raise ValueError(f"duplicate odds race ID: {rid}")
            seen.add(rid)
            dt = race_day(rid)
            if dt is None or dt.year != year:
                flags["invalid_id_or_year"] += 1
                continue
            if year == 2026 and dt > date(2026, 6, 30):
                flags["2026_out_of_scope"] += 1
                continue
            rt = r.get("race_type", "")
            if GIRLS.search(norm(rt)):
                flags["girls_skipped"] += 1
                continue
            outcome = results[year].get(rid)
            if outcome is None:
                flags["unverified_results"] += 1
                continue
            markets = {}
            for wager in TYPES:
                m = parse_market((r.get("odds") or {}).get(wager), wager)
                if m is not None:
                    cars = {c for combo, _ in m for c in combo}
                    if set(outcome).issubset(cars):
                        markets[wager] = m
            if not markets:
                flags["invalid_markets"] += 1
                continue
            flags["verified_races"] += 1
            riders = rider_index.get(rid, {})
            if riders:
                flags["races_with_entry_subset"] += 1
            for wager, items in markets.items():
                flags[f"{wager}_valid"] += 1
                active_cars = {c for combo, _ in items for c in combo}
                lines = valid_lines(riders, active_cars)
                if lines is not None:
                    flags[f"{wager}_valid_line"] += 1
                features = race_features(rt, {wager: items}, lines, riders)
                winner = outcome if wager == "3連単" else tuple(sorted(outcome))
                for policy, selection, hit, pay in select_policies(wager, items, winner, lines, riders):
                    for cell in field_cells(features, max_conditions=max_conditions):
                        stats[(policy, cell)][year].add(len(selection), hit, pay)
        flags["odds_races"] = len(seen)
        flags["matched_results_available"] = len(results[year])
        audit[str(year)] = dict(flags)
    return stats, audit


def performance(year_stats, years):
    z = total(year_stats.get(y, Stat()) for y in years)
    return {"overall": z.result(),
            "years": {str(y): year_stats.get(y, Stat()).result() for y in years}}


def filter_training(perf, *, min_races, min_hits, profitable_years):
    s = perf["overall"]
    if s["races"] < min_races or s["hits"] < min_hits or s["roi"] is None or s["roi"] <= 1:
        return False
    viable = [v for v in perf["years"].values() if v["races"] >= 10 and v["hits"] >= 1]
    return sum(v["roi"] is not None and v["roi"] > 1 for v in viable) >= profitable_years


def rank_train(stats, years, min_races, min_hits, min_years):
    good = []
    all_positive = 0
    for key, yearly in stats.items():
        p = performance(yearly, years)
        s = p["overall"]
        if s["roi"] is None:
            continue
        if s["roi"] > 1:
            all_positive += 1
        if filter_training(p, min_races=min_races, min_hits=min_hits,
                           profitable_years=min_years):
            good.append((key, p))
    # Avoid relying exclusively on one large hit and avoid overly complex filters.
    good.sort(key=lambda pair: (
        -sum(v["roi"] is not None and v["roi"] > 1 for v in pair[1]["years"].values()),
        pair[1]["overall"]["largest_single_race_win_share"] or 1,
        len(pair[0][1]), -pair[1]["overall"]["races"], -pair[1]["overall"]["roi"],
        pair[0]
    ))
    return good, all_positive


def key_json(key):
    return {"policy": key[0], "race_conditions": list(key[1])}


def run(root, out, first=2021, last=2025, holdout=2026,
        min_races=100, min_hits=5, min_years=2, max_conditions=3):
    if first >= last or holdout <= last:
        raise ValueError("Forward test must follow >=2 historical years")
    if max_conditions < 1 or max_conditions > 4:
        raise ValueError("max_conditions must be from 1 to 4")
    out.mkdir(parents=True, exist_ok=True)
    riders, rider_audit = load_riders(root, range(first, holdout+1))
    stats, audit = scan(root, first, holdout, riders, max_conditions=max_conditions)
    # All cells are discovered even when every single feature is below 100%.
    discovered, positive_n = rank_train(stats, list(range(first, last+1)),
                                         min_races, min_hits, min_years)
    # Record ALL result-bearing cells, not just those selected by screening.
    all_keys = sorted(stats, key=lambda k: (k[0], k[1]))
    with (out / "all_flatbuy_cells.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["policy", "race_conditions", "train_races", "train_bets", "train_hits",
                    "train_stake_yen", "train_return_yen", "train_roi",
                    "yearly_roi_json", "eligible_for_selection"])
        eligible_keys = {key for key, _ in discovered}
        for key in all_keys:
            p = performance(stats[key], range(first, last+1))
            s = p["overall"]
            w.writerow([key[0], ";".join(key[1]), s["races"], s["bets"],
                        s["hits"], s["stake_yen"], s["return_yen"], s["roi"],
                        json.dumps({y: x["roi"] for y, x in p["years"].items()}),
                        int(key in eligible_keys)])
    walks = []
    # Every forward year independently freezes a rule from earlier years.
    for test_year in range(first+2, holdout+1):
        train_years = list(range(first, min(test_year, last+1)))
        ranked, positive = rank_train(stats, train_years, min_races,
                                      min_hits, min_years)
        choice = ranked[0][0] if ranked else None
        p = performance(stats[choice], [test_year]) if choice else performance({}, [test_year])
        walks.append({"test_year": test_year, "train_years": train_years,
                      "candidate_positive_cells": positive,
                      "qualified_candidates": len(ranked),
                      "frozen_choice": key_json(choice) if choice else None,
                      "train_performance": ranked[0][1] if ranked else None,
                      "forward": p["overall"]})
    report = {
        "method": "pattern_policy_flatbuy_v2",
        "discovery": [first, last], "holdout": holdout,
        "scope": "available archived men's 3連単 and 3連複 complete markets; not all national races",
        "main_fix": "Search all predefined policies and 1..max_conditions conjunctions, without profitability gates",
        "max_conjoined_race_conditions": max_conditions,
        "exploration_counts_by_depth": {
            str(depth): sum(len(k[1]) == depth for k in stats)
            for depth in range(max_conditions + 1)
        },
        "rider_audit": rider_audit, "source_audit": audit,
        "distinct_cells_tested": len(stats),
        "profitable_train_cells_including_fragile": positive_n,
        "eligible_train_cells": len(discovered),
        "train_top30": [
            {**key_json(k), "train": p} for k, p in discovered[:30]],
        "walk_forward": walks,
        "final_rule": key_json(discovered[0][0]) if discovered else None,
        "holdout_result": walks[-1] if walks else None,
        "disclaimer": "Training ROI is exploratory; prices are final-odds payout proxy; live purchase was not simulated.",
        "data_caveats": [
            "2021-2022 rider/line information unavailable in equivalent archive",
            "2023-2026 entry/line archives are non-exhaustive and require coverage audit",
            "Does not infer same-line relationships from only position/size",
            "No official payout/refund audit; final odds differ from execution-time odds",
            "Only predefined deterministic policies and up to two race filters were searched; not all possible formations",
        ],
    }
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    with (out / "walk_forward.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["test_year", "train_years", "policy", "race_conditions",
                    "race_count", "bet_count", "hit_count", "stake_yen",
                    "return_yen", "profit_yen", "roi"])
        for row in walks:
            choice = row["frozen_choice"] or {}
            s = row["forward"]
            w.writerow([row["test_year"], ";".join(map(str, row["train_years"])),
                        choice.get("policy"), ";".join(choice.get("race_conditions", [])),
                        s["races"], s["bets"], s["hits"], s["stake_yen"],
                        s["return_yen"], s["profit_yen"], s["roi"]])
    print(json.dumps({"verified_races": sum(y.get("verified_races", 0) for y in audit.values()),
                      "cells_evaluated": len(stats), "training_positive_cells": positive_n,
                      "qualified_candidates": len(discovered),
                      "frozen_2026": report["holdout_result"],
                      "report_path": str(out/"report.json")}, ensure_ascii=False))
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=ROOT)
    p.add_argument("--out", type=Path, default=Path("results/flatbuy_pattern_v2"))
    p.add_argument("--first", type=int, default=2021)
    p.add_argument("--last", type=int, default=2025)
    p.add_argument("--holdout", type=int, default=2026)
    p.add_argument("--min-races", type=int, default=100)
    p.add_argument("--min-hits", type=int, default=5)
    p.add_argument("--min-years", type=int, default=2)
    p.add_argument("--max-conditions", type=int, default=3)
    a = p.parse_args()
    run(a.root, a.out, a.first, a.last, a.holdout, a.min_races, a.min_hits, a.min_years, a.max_conditions)


if __name__ == "__main__":
    main()
