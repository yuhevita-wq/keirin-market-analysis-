#!/usr/bin/env python3
from __future__ import annotations

"""
Retrospective diagnostic: apply the CURRENT seven-car v21/v31/v37+ runtime
board snapshot to 2024-01-01..2024-01-07, then convert each 3-row board
mechanically to trifecta tickets.

IMPORTANT:
- This is intentionally a backward/current-snapshot diagnostic.
- The current production models were trained/tuned after 2024, so this is NOT
  a leakage-free forward backtest and must never be cited as proof of live ROI.
- Target-race odds/popularity are not used for board placement.
- Payouts/results are read only after board generation for scoring.

Labels (fixed before reading outcomes):
- DEATH: trifecta payout = 0
- SLAVE: 0 < trifecta payout < 10,000 yen
- KING: trifecta payout >= 10,000 yen ("万車券")
"""

import csv
import importlib.util
import json
import re
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data/2024/s_class_f1_all_parts/2024_q1"
START, END = "2024-01-01", "2024-01-07"
OUT = ROOT / "results/keirin_shogi/current_board_2024_week1_king_diagnostic"
OUT.mkdir(parents=True, exist_ok=True)
STAKE_PER_TICKET = 100
KING_PAYOUT_YEN = 10_000


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


runtime = load_module(
    "keirin_shogi_current_runtime_for_2024_king_diag",
    ROOT / "scripts/keirin_shogi_v37_auto_place_runtime.py",
)


def ino(v, default=0):
    try:
        return int(float(v))
    except Exception:
        return default


def read_csv(path: Path):
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def norm_text(v):
    return unicodedata.normalize("NFKC", str(v or "")).strip()


def parse_triple(v):
    xs = [int(x) for x in re.findall(r"[1-9]", norm_text(v))]
    if len(xs) < 3:
        return None
    t = tuple(xs[:3])
    if len(set(t)) != 3:
        return None
    return t


def board_tickets(placed):
    first = [int(x) for x in placed.get("first_candidates", [])]
    second = [int(x) for x in placed.get("second_candidates", [])]
    third = [int(x) for x in placed.get("third_candidates", [])]
    return {
        (a, b, c)
        for a in first
        for b in second
        for c in third
        if len({a, b, c}) == 3
    }


def label_for(payout):
    if payout <= 0:
        return "DEATH"
    if payout < KING_PAYOUT_YEN:
        return "SLAVE"
    return "KING"


def max_death_streak(rows):
    best = cur = 0
    for r in rows:
        if r["label"] == "DEATH":
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def scenario_summary(rows):
    bets = [r for r in rows if r["bet"]]
    stake = sum(r["stake_yen"] for r in bets)
    payout = sum(r["payout_yen"] for r in bets)
    deaths = sum(r["label"] == "DEATH" for r in bets)
    slaves = sum(r["label"] == "SLAVE" for r in bets)
    kings = sum(r["label"] == "KING" for r in bets)
    hit = slaves + kings
    king_payout = sum(r["payout_yen"] for r in bets if r["label"] == "KING")
    maxp = max((r["payout_yen"] for r in bets), default=0)
    return {
        "bet_races": len(bets),
        "tickets": sum(r["ticket_count"] for r in bets),
        "stake_yen": stake,
        "payout_yen": payout,
        "profit_yen": payout - stake,
        "roi_pct": (100.0 * payout / stake) if stake else None,
        "hit_races": hit,
        "hit_rate_pct": (100.0 * hit / len(bets)) if bets else None,
        "death_races": deaths,
        "slave_races": slaves,
        "king_races": kings,
        "king_rate_pct": (100.0 * kings / len(bets)) if bets else None,
        "king_payout_yen": king_payout,
        "king_share_of_total_payout": (king_payout / payout) if payout else None,
        "max_race_payout_yen": maxp,
        "top1_payout_share": (maxp / payout) if payout else None,
        "max_death_streak": max_death_streak(bets),
        "avg_tickets_per_bet": (
            sum(r["ticket_count"] for r in bets) / len(bets) if bets else None
        ),
    }


def king_sequence(rows):
    out = []
    deaths_since_king = 0
    cum_profit = 0
    for r in rows:
        if not r["bet"]:
            continue
        before = cum_profit
        cum_profit += r["profit_yen"]
        if r["label"] == "DEATH":
            deaths_since_king += 1
        elif r["label"] == "KING":
            out.append({
                "race_id": r["race_id"],
                "race_date": r["race_date"],
                "track": r["track"],
                "race_no": r["race_no"],
                "board": r["board"],
                "actual": r["actual"],
                "ticket_count": r["ticket_count"],
                "stake_yen": r["stake_yen"],
                "payout_yen": r["payout_yen"],
                "profit_yen": r["profit_yen"],
                "deaths_since_previous_king": deaths_since_king,
                "cumulative_profit_before_king": before,
                "cumulative_profit_after_king": cum_profit,
                "return_multiple_vs_race_stake": (
                    r["payout_yen"] / r["stake_yen"] if r["stake_yen"] else None
                ),
            })
            deaths_since_king = 0
    return out


def main():
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
    entries_by = defaultdict(list)
    results_by = defaultdict(list)
    for e in entries:
        if e.get("race_id") in targets:
            entries_by[e["race_id"]].append(e)
    for rr in results:
        if rr.get("race_id") in targets:
            results_by[rr["race_id"]].append(rr)

    trifecta_by_race = defaultdict(dict)
    trifecta_seen_races = set()
    ticket_type_counts = defaultdict(int)
    for p in payouts:
        rid = str(p.get("race_id", ""))
        if rid not in targets:
            continue
        tt = norm_text(p.get("ticket_type", ""))
        ticket_type_counts[tt] += 1
        if tt not in {"3連単", "三連単"}:
            continue
        trifecta_seen_races.add(rid)
        if norm_text(p.get("status", "")).lower() not in {"", "paid"}:
            continue
        triple = parse_triple(p.get("combination", ""))
        py = re.sub(r"[^0-9]", "", norm_text(p.get("payout_yen", "")))
        if triple and py:
            trifecta_by_race[rid][triple] = int(py)

    engine = runtime.load_base_engine()
    ninecar_engine = runtime.load_ninecar_engine()
    overlay_engine = runtime.load_sevencar_overlay_engine()
    v21_state = engine.build_v21_state()
    pair_model = engine.build_pair_model()
    third_model, feature_names, freeze = engine.build_third_model()

    placed_rows = []
    failures = []
    for rid, race in sorted(
        targets.items(),
        key=lambda kv: (kv[1].get("race_date", ""), kv[1].get("track", ""), ino(kv[1].get("race_no"))),
    ):
        erows = sorted(entries_by.get(rid, []), key=lambda x: ino(x.get("car_no")))
        if len(erows) != 7:
            failures.append({"race_id": rid, "stage": "historical_entries", "error": f"entries={len(erows)}"})
            continue
        finish = {}
        for rr in results_by.get(rid, []):
            pos = ino(rr.get("finish_position"))
            if pos in (1, 2, 3):
                finish[pos] = ino(rr.get("car_no"))
        if set(finish) != {1, 2, 3}:
            failures.append({"race_id": rid, "stage": "historical_results", "error": f"top3={finish}"})
            continue

        live_race = {
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
            ninecar_engine,
            overlay_engine,
            live_race,
            v21_state,
            pair_model,
            third_model,
            feature_names,
            freeze,
        )
        if not placed.get("board_generated"):
            failures.append({
                "race_id": rid,
                "stage": placed.get("failure_stage", "board_generation"),
                "error": placed.get("error", placed.get("skip_reason", "")),
            })
            continue

        tickets = board_tickets(placed)
        paid = trifecta_by_race.get(rid, {})
        hit_tickets = sorted(tickets & set(paid))
        payout_yen = sum(paid[t] for t in hit_tickets)
        ticket_count = len(tickets)
        stake_yen = ticket_count * STAKE_PER_TICKET
        actual = [finish[1], finish[2], finish[3]]
        board = {
            "first": placed["first_candidates"],
            "second": placed["second_candidates"],
            "third": placed["third_candidates"],
        }
        placed_rows.append({
            "race_id": rid,
            "race_date": race.get("race_date", ""),
            "track": race.get("track", ""),
            "race_no": ino(race.get("race_no")),
            "race_type": race.get("race_type", ""),
            "participate": bool(placed.get("participate")),
            "board": board,
            "overlay_action": placed.get("sevencar_state_overlay_action"),
            "actual": actual,
            "ticket_count": ticket_count,
            "selected_tickets": [list(x) for x in sorted(tickets)],
            "winning_ticket_selected": actual in [list(x) for x in tickets],
            "hit_tickets": [list(x) for x in hit_tickets],
            "payout_yen": payout_yen,
            "stake_yen": stake_yen,
            "profit_yen": payout_yen - stake_yen,
            "label": label_for(payout_yen),
            "payout_coverage": rid in trifecta_seen_races,
        })

    covered = [r for r in placed_rows if r["payout_coverage"]]

    current_policy = []
    board_all = []
    for r in covered:
        cp = dict(r)
        cp["bet"] = bool(r["participate"])
        if not cp["bet"]:
            cp["stake_yen"] = 0
            cp["payout_yen"] = 0
            cp["profit_yen"] = 0
            cp["label"] = "SKIP"
        current_policy.append(cp)

        ba = dict(r)
        ba["bet"] = True
        board_all.append(ba)

    skipped_board_kings = [
        r for r in covered
        if (not r["participate"]) and r["label"] == "KING"
    ]
    skipped_board_slaves = [
        r for r in covered
        if (not r["participate"]) and r["label"] == "SLAVE"
    ]

    report = {
        "study": "current_board_2024_week1_king_diagnostic",
        "period": [START, END],
        "scope": "F1 S級 7車立て",
        "board_snapshot": {
            "policy": runtime.SEVENCAR_POLICY,
            "base": "v21 first / v31 second / v37 third",
            "overlay": "validated SOFT_FAIL strong-rider append to third row",
            "current_v21_history_end": v21_state.get("history_end"),
            "current_v21_state_week": v21_state.get("state_week"),
        },
        "method": {
            "ticket": "3連単。1着段×2着段×3着段のうち同一車重複を除く全組合せ",
            "stake_per_ticket_yen": STAKE_PER_TICKET,
            "labels": {
                "DEATH": "払戻0円",
                "SLAVE": "的中かつ払戻1〜9,990円",
                "KING": "払戻10,000円以上（万車券）",
            },
            "scenario_current_policy": "v21参加判定=参加のレースだけ購入",
            "scenario_board_all": "参加/見送りに関係なく生成盤面を全購入",
        },
        "critical_warning": (
            "現在のproduction snapshotを2024年へ逆適用しているため未来情報を含む。"
            "これは現行盤面が王を拾う/捨てる構造の診断であり、forward ROI検証ではない。"
        ),
        "data": {
            "target_races": len(targets),
            "boards_generated": len(placed_rows),
            "trifecta_payout_covered_races": len(covered),
            "payout_ticket_type_counts": dict(ticket_type_counts),
            "failures": failures,
        },
        "current_policy": scenario_summary(current_policy),
        "board_all": scenario_summary(board_all),
        "current_policy_kings": king_sequence(current_policy),
        "board_all_kings": king_sequence(board_all),
        "skipped_board_kings": [
            {
                "race_id": r["race_id"], "race_date": r["race_date"], "track": r["track"],
                "race_no": r["race_no"], "board": r["board"], "actual": r["actual"],
                "ticket_count": r["ticket_count"], "stake_yen": r["stake_yen"],
                "payout_yen": r["payout_yen"], "profit_yen": r["profit_yen"],
            }
            for r in skipped_board_kings
        ],
        "skipped_board_slave_count": len(skipped_board_slaves),
        "races": covered,
    }

    (OUT / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (OUT / "race_log.csv").open("w", encoding="utf-8-sig", newline="") as f:
        fields = [
            "race_id","race_date","track","race_no","race_type","participate",
            "first","second","third","actual","ticket_count","payout_yen",
            "stake_yen","profit_yen","label","overlay_action"
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in covered:
            w.writerow({
                "race_id": r["race_id"], "race_date": r["race_date"], "track": r["track"],
                "race_no": r["race_no"], "race_type": r["race_type"],
                "participate": int(r["participate"]),
                "first": "-".join(map(str, r["board"]["first"])),
                "second": "-".join(map(str, r["board"]["second"])),
                "third": "-".join(map(str, r["board"]["third"])),
                "actual": "-".join(map(str, r["actual"])),
                "ticket_count": r["ticket_count"], "payout_yen": r["payout_yen"],
                "stake_yen": r["stake_yen"], "profit_yen": r["profit_yen"],
                "label": r["label"], "overlay_action": r["overlay_action"],
            })

    def fmt(m):
        roi = "n/a" if m["roi_pct"] is None else f'{m["roi_pct"]:.1f}%'
        return (
            f'bet={m["bet_races"]} / tickets={m["tickets"]} / '
            f'stake={m["stake_yen"]:,} / payout={m["payout_yen"]:,} / '
            f'profit={m["profit_yen"]:+,} / ROI={roi} / '
            f'DEATH={m["death_races"]} SLAVE={m["slave_races"]} KING={m["king_races"]}'
        )

    md = [
        "# 現行盤面 × 2024 Week 1 王・奴隷・死 診断",
        "",
        f"- 期間: {START}〜{END}",
        "- 対象: F1 S級・7車立て",
        f"- 現行盤面: {runtime.SEVENCAR_POLICY}",
        "- 車券化: 3連単 1着段×2着段×3着段、同一車重複除外、各100円",
        "- 死: 払戻0円",
        "- 奴隷: 払戻1〜9,990円",
        "- 王: 払戻10,000円以上（万車券）",
        "",
        "## 重要",
        "",
        "これは現在のproduction snapshotを2024年へ逆適用した構造診断です。",
        "2024年より後の情報を含むため、forwardの収益性証明には使えません。",
        "",
        "## 集計",
        "",
        f'- 現行参加判定込み: {fmt(report["current_policy"])}',
        f'- 全盤面購入: {fmt(report["board_all"])}',
        f'- 見送りに埋まっていた王: {len(report["skipped_board_kings"])}',
        "",
        "## 現行参加判定で拾った王",
        "",
    ]
    if report["current_policy_kings"]:
        for k in report["current_policy_kings"]:
            md.append(
                f'- {k["race_date"]} {k["track"]}{k["race_no"]}R '
                f'{k["actual"]} 払戻{k["payout_yen"]:,}円 '
                f'({k["return_multiple_vs_race_stake"]:.2f}倍) '
                f'前王から死{k["deaths_since_previous_king"]}R'
            )
    else:
        md.append("- なし")
    md += ["", "## 見送りに埋まっていた王", ""]
    if report["skipped_board_kings"]:
        for k in report["skipped_board_kings"]:
            md.append(
                f'- {k["race_date"]} {k["track"]}{k["race_no"]}R '
                f'{k["actual"]} 払戻{k["payout_yen"]:,}円'
            )
    else:
        md.append("- なし")
    (OUT / "README.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    print(json.dumps({
        "out": str(OUT),
        "target_races": len(targets),
        "boards_generated": len(placed_rows),
        "covered": len(covered),
        "failures": len(failures),
        "current_policy": report["current_policy"],
        "board_all": report["board_all"],
        "current_policy_kings": report["current_policy_kings"],
        "skipped_board_kings": report["skipped_board_kings"],
        "ticket_type_counts": dict(ticket_type_counts),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
