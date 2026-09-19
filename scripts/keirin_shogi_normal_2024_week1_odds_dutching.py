#!/usr/bin/env python3
from __future__ import annotations

import csv
import importlib.util
import json
import math
import re
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data/2024/s_class_f1_all_parts/2024_q1"
START, END = "2024-01-01", "2024-01-07"
NORMAL_RUNTIME = ROOT / "scripts/.normal_v37_runtime_snapshot.py"
OUT = ROOT / "results/keirin_shogi/normal_board_2024_week1_odds_dutching"
STAKE_PER_RACE = 5000
UNIT = 100
TOTAL_UNITS = STAKE_PER_RACE // UNIT
EXPECTED_PARTICIPATION_RACES = 16
EXPECTED_PARTICIPATION_TICKETS = 112


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def read_csv(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def ino(v, default=0):
    try:
        return int(float(v))
    except Exception:
        return default


def norm(v):
    return unicodedata.normalize("NFKC", str(v or "")).strip()


def parse_triple(v):
    xs = [int(x) for x in re.findall(r"[1-9]", norm(v))]
    if len(xs) < 3:
        return None
    t = tuple(xs[:3])
    return t if len(set(t)) == 3 else None


def board_tickets(placed):
    return sorted({
        (int(a), int(b), int(c))
        for a in placed.get("first_candidates", [])
        for b in placed.get("second_candidates", [])
        for c in placed.get("third_candidates", [])
        if len({int(a), int(b), int(c)}) == 3
    })


def bounded_inverse_ideal(odds, total_units):
    n = len(odds)
    if n == 0:
        return []
    if n > total_units:
        raise ValueError(f"ticket_count={n} exceeds total 100-yen units={total_units}")

    free = set(range(n))
    fixed = {}
    remaining = float(total_units)
    ideals = [0.0] * n

    while free:
        denom = sum(1.0 / odds[i] for i in free)
        trial = {i: remaining * (1.0 / odds[i]) / denom for i in free}
        low = [i for i, u in trial.items() if u < 1.0]
        if not low:
            for i, u in trial.items():
                ideals[i] = u
            break
        for i in low:
            ideals[i] = 1.0
            fixed[i] = 1.0
            free.remove(i)
            remaining -= 1.0

    return ideals


def dutch_units(odds, total_units=TOTAL_UNITS):
    ideals = bounded_inverse_ideal(odds, total_units)
    units = [max(1, int(math.floor(x))) for x in ideals]
    # Correct any pathological floor overflow.
    while sum(units) > total_units:
        candidates = [i for i, u in enumerate(units) if u > 1]
        if not candidates:
            raise RuntimeError("cannot fit minimum stakes into total budget")
        i = max(candidates, key=lambda j: units[j] - ideals[j])
        units[i] -= 1

    def quality(candidate):
        gross = [candidate[i] * odds[i] for i in range(len(odds))]
        mean = sum(gross) / len(gross)
        spread = max(gross) - min(gross)
        sse = sum((x - mean) ** 2 for x in gross)
        return (spread, sse)

    while sum(units) < total_units:
        best_i = None
        best_q = None
        for i in range(len(units)):
            cand = list(units)
            cand[i] += 1
            q = quality(cand)
            if best_q is None or q < best_q:
                best_q = q
                best_i = i
        units[best_i] += 1

    return units


def load_target_odds(target_ids):
    out = defaultdict(dict)
    path = BASE / "trifecta_final_odds.csv"
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            rid = str(row.get("race_id", ""))
            if rid not in target_ids:
                continue
            if norm(row.get("odds_status")).lower() != "available":
                continue
            combo = parse_triple(row.get("combination"))
            try:
                odd = float(row.get("odds", ""))
            except Exception:
                continue
            if combo and odd > 0:
                out[rid][combo] = odd
    return out


def partial_odds_fallback(race, entrant_numbers):
    """Recover available fixed-first tables even if one first-car table is missing."""
    try:
        hist = load_module(
            "historical_final_odds_partial_fallback",
            ROOT / "src/keirin_market_analysis/historical_final_odds.py",
        )
        from bs4 import BeautifulSoup

        url = hist.odds_url(race.get("source_url", ""))
        html = hist.fetch_html(hist.session(), url)
        soup = BeautifulSoup(html, "lxml")
        merged = {}
        for table in soup.select("table.odds_table"):
            got = hist.parse_fixed_first_trifecta_table(table, entrant_numbers)
            if got is None:
                continue
            _first, block = got
            for combo, (odd, status) in block.items():
                if status == "available":
                    merged[tuple(map(int, combo))] = float(odd)
        return merged, url
    except Exception as exc:
        return {}, f"{type(exc).__name__}: {exc}"


def main():
    if not NORMAL_RUNTIME.exists():
        raise FileNotFoundError(
            f"{NORMAL_RUNTIME} missing. Workflow must materialize the frozen pre-v38 normal runtime snapshot."
        )
    runtime = load_module("normal_v37_runtime_snapshot", NORMAL_RUNTIME)

    races = read_csv(BASE / "races.csv")
    entries = read_csv(BASE / "entries.csv")
    results = read_csv(BASE / "results.csv")
    payouts = read_csv(BASE / "payouts.csv")

    targets = {
        r["race_id"]: r
        for r in races
        if START <= r.get("race_date", "") <= END
        and ino(r.get("entry_count")) == 7
        and r.get("meeting_grade") == "F1"
        and "Ｓ級" in str(r.get("race_type", ""))
    }
    target_ids = set(targets)

    entries_by = defaultdict(list)
    results_by = defaultdict(list)
    for e in entries:
        if e.get("race_id") in targets:
            entries_by[e["race_id"]].append(e)
    for rr in results:
        if rr.get("race_id") in targets:
            results_by[rr["race_id"]].append(rr)

    payout_by_race = {}
    for p in payouts:
        rid = str(p.get("race_id", ""))
        if rid not in targets:
            continue
        if norm(p.get("ticket_type")) not in {"3連単", "三連単"}:
            continue
        if norm(p.get("status")).lower() not in {"", "paid"}:
            continue
        combo = parse_triple(p.get("combination"))
        py = re.sub(r"[^0-9]", "", norm(p.get("payout_yen")))
        if combo and py:
            payout_by_race[rid] = (combo, int(py))

    odds_by_race = load_target_odds(target_ids)

    engine = runtime.load_base_engine()
    nine = runtime.load_ninecar_engine()
    overlay = runtime.load_sevencar_overlay_engine()
    v21_state = engine.build_v21_state()
    pair_model = engine.build_pair_model()
    third_model, feature_names, freeze = engine.build_third_model()

    rows = []
    failures = []
    for rid, race in sorted(
        targets.items(),
        key=lambda kv: (
            kv[1].get("race_date", ""),
            kv[1].get("track", ""),
            ino(kv[1].get("race_no")),
        ),
    ):
        erows = sorted(entries_by.get(rid, []), key=lambda x: ino(x.get("car_no")))
        if len(erows) != 7:
            failures.append({"race_id": rid, "stage": "entries", "error": f"entries={len(erows)}"})
            continue

        finish = {}
        for rr in results_by.get(rid, []):
            pos = ino(rr.get("finish_position"))
            if pos in (1, 2, 3):
                finish[pos] = ino(rr.get("car_no"))
        if set(finish) != {1, 2, 3}:
            failures.append({"race_id": rid, "stage": "results", "error": f"top3={finish}"})
            continue

        live = {
            "race_id": rid,
            "race_date": race.get("race_date", ""),
            "track": race.get("track", ""),
            "race_no": race.get("race_no", ""),
            "race_type": race.get("race_type", ""),
            "meeting_grade": race.get("meeting_grade", ""),
            "entries": [runtime.normalize_entry(e) for e in erows],
        }
        placed = runtime.runtime_place_one(
            engine,
            nine,
            overlay,
            live,
            v21_state,
            pair_model,
            third_model,
            feature_names,
            freeze,
        )
        if not placed.get("board_generated"):
            failures.append({
                "race_id": rid,
                "stage": placed.get("failure_stage", "board"),
                "error": placed.get("error", placed.get("skip_reason", "")),
            })
            continue

        tickets = board_tickets(placed)
        actual = (finish[1], finish[2], finish[3])
        payout_combo, payout100 = payout_by_race.get(rid, (actual, 0))

        race_odds = dict(odds_by_race.get(rid, {}))
        missing = [t for t in tickets if t not in race_odds]
        fallback_note = ""
        if missing:
            partial, fallback_note = partial_odds_fallback(
                race, [ino(e.get("car_no")) for e in erows]
            )
            race_odds.update(partial)
            missing = [t for t in tickets if t not in race_odds]

        detail = []
        strict_stake = 0
        strict_return = 0
        if not missing and tickets:
            odds = [race_odds[t] for t in tickets]
            units = dutch_units(odds, TOTAL_UNITS)
            strict_stake = sum(units) * UNIT
            for t, odd, u in zip(tickets, odds, units):
                stake = u * UNIT
                gross_if_hit = stake * odd
                is_hit = t == actual
                if is_hit:
                    strict_return = u * payout100
                detail.append({
                    "combination": "-".join(map(str, t)),
                    "odds": odd,
                    "stake_yen": stake,
                    "gross_at_final_odds_yen": round(gross_if_hit, 1),
                    "hit": is_hit,
                })

        rows.append({
            "race_id": rid,
            "race_date": race.get("race_date", ""),
            "track": race.get("track", ""),
            "race_no": ino(race.get("race_no")),
            "race_type": race.get("race_type", ""),
            "participate": bool(placed.get("participate")),
            "board": {
                "first": list(map(int, placed.get("first_candidates", []))),
                "second": list(map(int, placed.get("second_candidates", []))),
                "third": list(map(int, placed.get("third_candidates", []))),
            },
            "ticket_count": len(tickets),
            "actual": list(actual),
            "official_trifecta": "-".join(map(str, payout_combo)),
            "official_payout_per_100_yen": payout100,
            "odds_complete": not missing,
            "missing_ticket_odds": ["-".join(map(str, x)) for x in missing],
            "fallback_note": fallback_note if missing else "",
            "stake_yen": strict_stake,
            "return_yen": strict_return,
            "profit_yen": strict_return - strict_stake if not missing else None,
            "allocations": detail,
        })

    participation = [r for r in rows if r["participate"]]
    board_all = list(rows)

    part_ticket_count = sum(r["ticket_count"] for r in participation)
    guard_ok = (
        len(participation) == EXPECTED_PARTICIPATION_RACES
        and part_ticket_count == EXPECTED_PARTICIPATION_TICKETS
    )

    def summarize(group):
        covered = [r for r in group if r["odds_complete"]]
        unresolved = [r for r in group if not r["odds_complete"]]
        stake = sum(r["stake_yen"] for r in covered)
        ret = sum(r["return_yen"] for r in covered)
        hits = sum(r["return_yen"] > 0 for r in covered)
        return {
            "races_total": len(group),
            "races_odds_complete": len(covered),
            "races_odds_unresolved": len(unresolved),
            "ticket_count": sum(r["ticket_count"] for r in group),
            "stake_yen_strict_covered": stake,
            "return_yen_strict_covered": ret,
            "profit_yen_strict_covered": ret - stake,
            "roi_pct_strict_covered": 100.0 * ret / stake if stake else None,
            "hit_races_strict_covered": hits,
            "hit_rate_pct_strict_covered": 100.0 * hits / len(covered) if covered else None,
            "unresolved_races": [
                {
                    "race_id": r["race_id"],
                    "race_date": r["race_date"],
                    "track": r["track"],
                    "race_no": r["race_no"],
                    "missing_ticket_odds": r["missing_ticket_odds"],
                }
                for r in unresolved
            ],
        }

    report = {
        "study": "normal_board_2024_week1_odds_dutching",
        "period": [START, END],
        "scope": "F1 S級 7車立て",
        "normal_algorithm_snapshot": {
            "runtime_commit": "2bd26e8aaa4508936882e6f09925889f32c4c0b9",
            "policy": "v21/v31/v37+ SOFT_FAIL third append, pre-v38 normal runtime",
        },
        "stake_rule": {
            "per_race_yen": STAKE_PER_RACE,
            "unit_yen": UNIT,
            "method": "全盤面3連単を最低100円ずつ買い、残りを1/確定オッズ比例で配分。100円単位の丸め後、想定払戻のばらつきが小さくなるよう端数調整。",
            "odds_source": "楽天Kドリームス 確定オッズ",
            "warning": "確定オッズでありT-10分オッズではない。",
        },
        "baseline_guard": {
            "expected_participation_races": EXPECTED_PARTICIPATION_RACES,
            "actual_participation_races": len(participation),
            "expected_participation_tickets": EXPECTED_PARTICIPATION_TICKETS,
            "actual_participation_tickets": part_ticket_count,
            "ok": guard_ok,
        },
        "participation_only": summarize(participation),
        "all_boards_reference": summarize(board_all),
        "failures": failures,
        "participation_races": participation,
    }

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    with (OUT / "race_log.csv").open("w", encoding="utf-8-sig", newline="") as f:
        fields = [
            "race_id","race_date","track","race_no","race_type","participate",
            "first","second","third","actual","ticket_count","odds_complete",
            "stake_yen","return_yen","profit_yen"
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({
                "race_id": r["race_id"],
                "race_date": r["race_date"],
                "track": r["track"],
                "race_no": r["race_no"],
                "race_type": r["race_type"],
                "participate": int(r["participate"]),
                "first": "-".join(map(str, r["board"]["first"])),
                "second": "-".join(map(str, r["board"]["second"])),
                "third": "-".join(map(str, r["board"]["third"])),
                "actual": "-".join(map(str, r["actual"])),
                "ticket_count": r["ticket_count"],
                "odds_complete": int(r["odds_complete"]),
                "stake_yen": r["stake_yen"],
                "return_yen": r["return_yen"],
                "profit_yen": "" if r["profit_yen"] is None else r["profit_yen"],
            })

    md = [
        "# 通常駒置き × 2024 Week1 × 1R 5,000円オッズ配分",
        "",
        f"- 期間: {START}〜{END}",
        "- 対象: F1 S級 7車立て",
        "- 通常版: v21 / v31 / v37+（v38王席化前）",
        "- 車券: 盤面の1着×2着×3着 3連単",
        "- 予算: 参加1Rあたり5,000円",
        "- 配分: 最低100円 + 1/確定オッズ比例ダッチング",
        "",
        "## 参加判定込み",
        "",
    ]
    s = report["participation_only"]
    md += [
        f'- 対象 {s["races_total"]}R / オッズ完全 {s["races_odds_complete"]}R / 未解決 {s["races_odds_unresolved"]}R',
        f'- 投資（厳密集計可能分） {s["stake_yen_strict_covered"]:,}円',
        f'- 払戻（厳密集計可能分） {s["return_yen_strict_covered"]:,}円',
        f'- 損益（厳密集計可能分） {s["profit_yen_strict_covered"]:+,}円',
        f'- 回収率（厳密集計可能分） {s["roi_pct_strict_covered"]:.1f}%' if s["roi_pct_strict_covered"] is not None else "- 回収率 n/a",
        f'- 的中 {s["hit_races_strict_covered"]}R',
        "",
        f'- baseline guard: {"OK" if guard_ok else "MISMATCH"} '
        f'({len(participation)}R / {part_ticket_count}点; expected 16R / 112点)',
        "",
        "## レース別",
        "",
    ]
    for r in participation:
        if r["odds_complete"]:
            md.append(
                f'- {r["race_date"]} {r["track"]}{r["race_no"]}R '
                f'{r["actual"]} / {r["ticket_count"]}点 / '
                f'投資{r["stake_yen"]:,} / 払戻{r["return_yen"]:,} / '
                f'損益{r["profit_yen"]:+,}'
            )
        else:
            md.append(
                f'- {r["race_date"]} {r["track"]}{r["race_no"]}R '
                f'オッズ未解決: {r["missing_ticket_odds"]}'
            )
    (OUT / "README.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    print(json.dumps({
        "guard": report["baseline_guard"],
        "participation_only": report["participation_only"],
        "all_boards_reference": report["all_boards_reference"],
        "out": str(OUT),
    }, ensure_ascii=False, indent=2))

    return 0 if guard_ok else 4


if __name__ == "__main__":
    raise SystemExit(main())
