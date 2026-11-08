#!/usr/bin/env python3
"""Five-year exhaustive flat-buy -> profitable-condition -> frozen forward test.

No existing HARUQ prediction, marks or formations are used. Every archived
exact ticket is evaluated as if purchased for 100 yen per qualifying race.
Unverifiable races are excluded, not assigned a zero payout.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WAGERS = ("3連単", "3連複")
GIRLS = re.compile(r"ガールズ|女子|L級|ガ予|ガ決|ガ一般|ガ選", re.I)


@dataclass
class Stat:
    bets: int = 0
    hits: int = 0
    paid: int = 0
    max_paid: int = 0
    races: int = 0

    def add(self, payout: int, hit: bool) -> None:
        self.bets += 1
        self.hits += int(hit)
        self.paid += payout
        self.max_paid = max(self.max_paid, payout)

    def as_dict(self) -> dict:
        stake = self.bets * 100
        return {
            "bets": self.bets, "hits": self.hits, "stake_yen": stake,
            "return_yen": self.paid, "profit_yen": self.paid - stake,
            "roi": round(self.paid / stake, 6) if stake else None,
            "hit_rate": round(self.hits / self.bets, 6) if self.bets else None,
            "largest_payout_share": round(self.max_paid / self.paid, 6) if self.paid else None,
        }


def combined(stats) -> Stat:
    t = Stat()
    for s in stats:
        t.bets += s.bets
        t.hits += s.hits
        t.paid += s.paid
        t.max_paid = max(t.max_paid, s.max_paid)
    return t


def race_date(rid: str) -> date | None:
    # The race ID's embedded date is the meeting START, not always the race day.
    if not re.fullmatch(r"\d{16}", str(rid)):
        return None
    try:
        day_index = int(rid[10:12])
        if not 1 <= day_index <= 7:
            return None
        return date(int(rid[2:6]), int(rid[6:8]), int(rid[8:10])) + timedelta(days=day_index - 1)
    except ValueError:
        return None


def rows(path: Path):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def labels(root: Path, year: int) -> dict[str, tuple[int, int, int]]:
    path = root / "data/results_only" / f"results_{year}.jsonl.gz"
    if not path.is_file():
        raise FileNotFoundError(path)
    out = {}
    for row in rows(path):
        rid = str(row.get("race_id", ""))
        order = row.get("finish_order")
        if race_date(rid) is None or race_date(rid).year != year:
            raise ValueError(f"invalid result race ID/year: {rid}")
        if not isinstance(order, list) or len(order) != 3:
            raise ValueError(f"invalid result: {rid}")
        finish = tuple(int(c) for c in order)
        if len(set(finish)) != 3 or any(not 1 <= x <= 9 for x in finish):
            raise ValueError(f"invalid podium: {rid}")
        if rid in out and out[rid] != finish:
            raise ValueError(f"conflicting podium: {rid}")
        out[rid] = finish
    return out


def band(v: float, edges: tuple[float, ...]) -> str:
    for edge in edges:
        if v < edge:
            return f"<{edge:g}"
    return f">={edges[-1]:g}"


def rank_band(r: int) -> str:
    if r == 1:
        return "1"
    for high in (3, 6, 12, 24, 48, 96, 192):
        if r <= high:
            low = (1 if high == 3 else {6: 4, 12: 7, 24: 13, 48: 25, 96: 49, 192: 97}[high])
            return f"{low}-{high}"
    return "193+"


def stage(value: str) -> str:
    s = unicodedata.normalize("NFKC", value or "")
    if "チャレンジ" in s:
        return "チャレンジ"
    for pattern, name in (
        ("初日特選", "初日特選"), ("準決", "準決勝"), ("決勝", "決勝"),
        ("二次予", "二次予選"), ("二予", "二次予選"),
        ("一次予", "一次予選"), ("一予", "一次予選"),
        ("特予", "予選"), ("予選", "予選"),
        ("特一般", "一般"), ("一般", "一般"),
        ("特選", "特選"), ("選抜", "選抜"),
    ):
        if pattern in s:
            return name
    return "不明"


def grade(value: str) -> str | None:
    s = unicodedata.normalize("NFKC", value or "").upper()
    m = re.search(r"(^|/)(GP|G[123]|F[12])(?:/|$)", s)
    return m.group(2) if m else None


def normalize_combo(combo: str, wager: str) -> tuple[int, int, int] | None:
    nums = re.findall(r"\d+", combo)
    if len(nums) != 3:
        return None
    v = tuple(map(int, nums))
    if len(set(v)) != 3 or any(not 1 <= x <= 9 for x in v):
        return None
    return tuple(sorted(v)) if wager == "3連複" else v


def market_tickets(record: dict, finish: tuple[int, int, int]):
    """Yield pre-result attributes and real winner flags, with complete-book checks."""
    book = record.get("odds") or {}
    race_type = str(record.get("race_type") or "")
    common = [f"レース種別={stage(race_type)}"]
    g = grade(race_type)
    if g:
        common.append(f"開催格={g}")
    for wager in WAGERS:
        raw = book.get(wager)
        if not isinstance(raw, dict) or not raw:
            continue
        entries = []
        cars = set()
        seen = set()
        valid = True
        for combo, price in raw.items():
            parsed = normalize_combo(str(combo), wager)
            try:
                val = float(price)
            except (TypeError, ValueError):
                valid = False
                break
            if parsed is None or parsed in seen or not math.isfinite(val) or val <= 0:
                valid = False
                break
            seen.add(parsed)
            cars.update(parsed)
            entries.append((str(combo), parsed, val))
        n = len(cars)
        expected = n * (n - 1) * (n - 2) if wager == "3連単" else n * (n - 1) * (n - 2) // 6
        if not valid or n < 3 or len(entries) != expected or not set(finish).issubset(cars):
            continue
        winner = finish if wager == "3連単" else tuple(sorted(finish))
        if winner not in seen:
            continue
        entries.sort(key=lambda x: (x[2], x[1]))
        top_odds = entries[0][2]
        race_tokens = (*common, f"券種={wager}", f"車立て={n}",
                       f"最人気オッズ帯={band(top_odds, (3.5, 6, 10, 20, 40))}")
        first_support = defaultdict(float)
        if wager == "3連単":
            for _, parsed, odd in entries:
                first_support[parsed[0]] += 1 / odd
            first_order = {car: i + 1 for i, (car, _) in enumerate(
                sorted(first_support.items(), key=lambda kv: (-kv[1], kv[0])))}
        last_price = None
        tie_rank = 0
        for idx, (combo, parsed, odd) in enumerate(entries, start=1):
            if last_price != odd:
                tie_rank = idx
            last_price = odd
            features = [*race_tokens,
                        f"オッズ帯={band(odd, (3.5, 6, 10, 20, 30, 50, 100, 200, 500))}",
                        f"人気順位帯={rank_band(tie_rank)}"]
            if wager == "3連単":
                place_rank = first_order[parsed[0]]
                features.append(f"1着候補支持順位={band(place_rank, (2, 4, 7))}")
            hit = parsed == winner
            # Decimal market odds are a proxy for 100-yen payout; never invent refunds.
            yield wager, combo, odd, tuple(features), hit, int(round(100 * odd)) if hit else 0


def scan(root: Path, years: list[int], rule: tuple[str, ...] = (),
         expand: bool = False, decision_writer=None):
    """Stream all complete, result-verified tickets. Never peek into next year's labels."""
    totals = defaultdict(Stat)
    expansions = defaultdict(lambda: defaultdict(Stat))
    audits = {}
    rule_set = set(rule)
    used_fields = {tok.split("=", 1)[0] for tok in rule}
    for year in years:
        path = root / "data/odds_only" / f"odds_{year}.jsonl.gz"
        if not path.is_file():
            raise FileNotFoundError(path)
        outcomes = labels(root, year)
        audit = Counter()
        matched_races = set()
        seen_races = set()
        for row in rows(path):
            rid = str(row.get("race_id", ""))
            if rid in seen_races:
                raise ValueError(f"duplicate odds race_id in {year}: {rid}")
            seen_races.add(rid)
            dt = race_date(rid)
            if dt is None or dt.year != year:
                audit["wrong_year_or_invalid_date"] += 1
                continue
            if GIRLS.search(str(row.get("race_type", ""))):
                audit["girls_excluded"] += 1
                continue
            finish = outcomes.get(rid)
            if finish is None:
                audit["no_verified_result"] += 1
                continue
            counted_types = set()
            purchased = False
            for wager, combo, odds, features, hit, payout in market_tickets(row, finish):
                counted_types.add(wager)
                fset = set(features)
                if not rule_set.issubset(fset):
                    continue
                purchased = True
                totals[year].add(payout, hit)
                if decision_writer is not None:
                    decision_writer.writerow([year, rid, wager, combo, odds,
                                              int(hit), payout, ";".join(rule)])
                if expand:
                    for feature in features:
                        if feature in rule_set or feature.split("=", 1)[0] in used_fields:
                            continue
                        expansions[feature][year].add(payout, hit)
            if counted_types:
                matched_races.add(rid)
                for wager in counted_types:
                    audit[f"valid_{wager}_race"] += 1
            else:
                audit["no_complete_winning_market"] += 1
            if purchased:
                audit["buy_races"] += 1
            else:
                audit["no_buy_races"] += 1
        audit["total_odds_races"] = len(seen_races)
        audit["verified_market_races"] = len(matched_races)
        audit["verified_results_in_file"] = len(outcomes)
        audits[str(year)] = dict(audit)
    return totals, expansions, audits


def pack_by_year(stats: dict[int, Stat], years: list[int]) -> dict:
    overall = combined(stats.get(y, Stat()) for y in years)
    return {"overall": overall.as_dict(),
            "by_year": {str(y): stats.get(y, Stat()).as_dict() for y in years}}


def eligible(packed: dict, *, min_bets: int, min_hits: int,
             minimum_profitable_years: int) -> bool:
    a = packed["overall"]
    if a["bets"] < min_bets or a["hits"] < min_hits or a["roi"] is None or a["roi"] <= 1:
        return False
    active = [v for v in packed["by_year"].values() if v["bets"] >= 30 and v["hits"] >= 1]
    profitable = sum(v["roi"] is not None and v["roi"] > 1 for v in active)
    return len(active) >= minimum_profitable_years and profitable >= minimum_profitable_years


def candidates(groups: dict, years: list[int], min_bets: int, min_hits: int,
               min_profitable_years: int):
    found = []
    for feature, by_year in groups.items():
        p = pack_by_year(by_year, years)
        if not eligible(p, min_bets=min_bets, min_hits=min_hits,
                        minimum_profitable_years=min_profitable_years):
            continue
        active = [v for v in p["by_year"].values() if v["bets"] >= 30 and v["hits"] >= 1]
        profitable = sum(v["roi"] > 1 for v in active)
        # Stability first, then payout concentration and ROI; no result-derived
        # ticket selection within a race. Complexity is limited by max depth.
        concentration = p["overall"]["largest_payout_share"] or 0
        score = (profitable / len(active), -concentration,
                 p["overall"]["roi"], p["overall"]["bets"])
        found.append((score, feature, p))
    found.sort(reverse=True)
    return [{"feature": k, "score": list(score), "performance": p}
            for score, k, p in found]


def learn(root: Path, years: list[int], *, depth: int, min_bets: int,
          min_hits: int, min_profitable_years: int, min_gain: float):
    rule: list[str] = []
    history = []
    for iteration in range(1, depth + 1):
        baselines, groups, audit = scan(root, years, tuple(rule), expand=True)
        base = pack_by_year(baselines, years)
        ranked = candidates(groups, years, min_bets, min_hits, min_profitable_years)
        chosen = next((
            c for c in ranked
            if c["performance"]["overall"]["roi"] > (base["overall"]["roi"] or 0) + (min_gain if rule else 0)
        ), None)
        step = {
            "iteration": iteration, "current_rule": list(rule),
            "current_performance": base, "qualifying_expansions": len(ranked),
            "top_candidates": ranked[:15], "chosen": chosen,
        }
        history.append(step)
        if chosen is None:
            break
        rule.append(chosen["feature"])
    frozen, _, audit = scan(root, years, tuple(rule))
    return {"train_years": years, "rule": rule, "depth": len(rule),
            "training": pack_by_year(frozen, years), "iterations": history,
            "data_audit": audit, "buy_enabled": bool(rule)}


def emit_decisions(root: Path, year: int, rule: list[str], output: Path):
    output.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(output, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["year", "race_id", "wager", "ticket", "final_odds",
                         "hit", "proxy_payout_yen", "rule"])
        return scan(root, [year], tuple(rule), decision_writer=writer)


def run(root: Path, out: Path, args):
    training = list(range(args.first_year, args.last_year + 1))
    if len(training) < 3:
        raise ValueError("At least 3 historical years are required")
    if args.holdout_year <= args.last_year:
        raise ValueError("Holdout must be strictly later than discovery window")
    out.mkdir(parents=True, exist_ok=True)

    # Full-period flat-buy census is descriptive only, NOT used to claim
    # historical out-of-sample performance or to select past-year tickets.
    root_stats, single_groups, census = scan(root, training, expand=True)
    ranked_full = candidates(single_groups, training, args.min_bets, args.min_hits,
                             args.min_profitable_years)
    with (out / "flat_buy_profitable_conditions.csv").open("w", newline="", encoding="utf-8") as handle:
        w = csv.writer(handle)
        w.writerow(["feature", "bets", "hits", "stake_yen", "return_yen",
                    "profit_yen", "roi", "profitable_years", "largest_payout_share"])
        for candidate in ranked_full:
            s = candidate["performance"]
            w.writerow([candidate["feature"], s["overall"]["bets"], s["overall"]["hits"],
                        s["overall"]["stake_yen"], s["overall"]["return_yen"],
                        s["overall"]["profit_yen"], s["overall"]["roi"],
                        sum(v["roi"] is not None and v["roi"] > 1 for v in s["by_year"].values()),
                        s["overall"]["largest_payout_share"]])

    frozen_tests = []
    for test_year in range(args.first_year + 2, args.holdout_year + 1):
        years = list(range(args.first_year, min(test_year, args.last_year + 1)))
        learned = learn(root, years, depth=args.depth, min_bets=args.min_bets,
                        min_hits=args.min_hits, min_profitable_years=args.min_profitable_years,
                        min_gain=args.min_gain)
        # If a rule cannot be established, this year is a genuine NO BUY.
        if not learned["rule"]:
            eval_stats, _, eval_audit = scan(root, [test_year],
                                            ("購入不可=全対象除外",))
        else:
            eval_stats, _, eval_audit = emit_decisions(
                root, test_year, learned["rule"],
                out / "decisions" / f"frozen_{test_year}.csv.gz")
        frozen_tests.append({
            "year": test_year, "training_years": years,
            "selected_rule": learned["rule"], "training": learned["training"],
            "forward": pack_by_year(eval_stats, [test_year]),
            "audit": eval_audit, "iterations": learned["iterations"]
        })
    final_model = learn(root, training, depth=args.depth, min_bets=args.min_bets,
                        min_hits=args.min_hits, min_profitable_years=args.min_profitable_years,
                        min_gain=args.min_gain)
    # The holdout rule must always be learned ONLY from 2021..2025.
    if final_model["rule"]:
        test_stats, _, test_audit = emit_decisions(root, args.holdout_year,
            final_model["rule"], out / "decisions" / f"frozen_{args.holdout_year}.csv.gz")
    else:
        test_stats, _, test_audit = scan(root, [args.holdout_year],
                                        ("購入不可=全対象除外",))

    final = {
        "study": "five_year_flatbuy_repeat_frozen_forward_v1",
        "discovery_window": [args.first_year, args.last_year],
        "holdout_year": args.holdout_year, "bets_are_flat_100_yen_each": True,
        "wagers_supported": list(WAGERS), "all_wager_types_supported": False,
        "no_girls": True, "no_race_outcome_used_as_feature": True,
        "source_odds": "FINAL market odds; not historical order-time odds",
        "payout_method": "winning final_odds * 100, rounded to yen; NOT official payout/refund reconciliation",
        "coverage_warning": "Only archived complete markets with verified finishes; NOT all actual nationwide races",
        "full_period_descriptive_only": {
            "all_ticket_flat_buy": pack_by_year(root_stats, training),
            "eligible_race_audit": census,
            "profitable_single_conditions_count": len(ranked_full),
            "profitable_single_conditions_top50": ranked_full[:50],
        },
        "chronological_frozen_backtests": frozen_tests,
        "final_rule_from_5_years": final_model,
        "untouched_later_year_evaluation": {
            "year": args.holdout_year,
            "performance": pack_by_year(test_stats, [args.holdout_year]),
            "audit": test_audit,
        },
        "interpretation": "ROI>100% in discovery is a hypothesis, never evidence of forward profitability.",
    }
    (out / "report.json").write_text(json.dumps(final, ensure_ascii=False, indent=2) + "\n",
                                     encoding="utf-8")
    with (out / "walk_forward.csv").open("w", newline="", encoding="utf-8") as handle:
        w = csv.writer(handle)
        w.writerow(["forward_year", "train_years", "rule", "bets", "hits", "stake_yen",
                    "return_yen", "profit_yen", "roi", "buy_races"])
        for test in frozen_tests + [{
            "year": args.holdout_year,
            "training_years": training,
            "selected_rule": final_model["rule"],
            "forward": pack_by_year(test_stats, [args.holdout_year]), "audit": test_audit,
        }]:
            s = test["forward"]["overall"]
            w.writerow([test["year"], ";".join(map(str, test["training_years"])),
                        ";".join(test["selected_rule"]), s["bets"], s["hits"],
                        s["stake_yen"], s["return_yen"], s["profit_yen"], s["roi"],
                        test["audit"][str(test["year"])].get("buy_races", 0)])
    print(json.dumps({
        "five_year_eligible_races": sum(v.get("verified_market_races", 0) for v in census.values()),
        "profitable_flat_conditions": len(ranked_full),
        "last_rule": final_model["rule"],
        "out_of_sample": final["untouched_later_year_evaluation"]["performance"]["overall"],
        "report": str(out / "report.json")
    }, ensure_ascii=False))
    return final


def cli():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=ROOT)
    p.add_argument("--out", type=Path, default=Path("results/five_year_flatbuy_loop"))
    p.add_argument("--first-year", type=int, default=2021)
    p.add_argument("--last-year", type=int, default=2025)
    p.add_argument("--holdout-year", type=int, default=2026)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--min-bets", type=int, default=500)
    p.add_argument("--min-hits", type=int, default=10)
    p.add_argument("--min-profitable-years", type=int, default=2)
    p.add_argument("--min-gain", type=float, default=0.02)
    a = p.parse_args()
    if a.depth < 1 or a.min_bets < 1 or a.min_profitable_years < 1:
        p.error("depth, min-bets and min-profitable-years must be positive")
    run(a.root, a.out, a)


if __name__ == "__main__":
    cli()
