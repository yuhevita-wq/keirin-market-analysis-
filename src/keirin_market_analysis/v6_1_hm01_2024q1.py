from __future__ import annotations

import csv
import json
import math
import subprocess
from collections import defaultdict
from pathlib import Path

BASE_SCHEME_VERSION = "v6.1"
VALIDATION_VERSION = "v6.3-D01-Q3-OOS"
VALIDATION_DATASET = "2024Q3"
ENTRY_FILTER_THRESHOLD = 0.35640013538348414
STAKE = 100

ROOT = Path(__file__).resolve().parents[2]
DATA_REL = Path("data/2024/s_class_f1_all_parts/2024_q3")
DATA = ROOT / DATA_REL
OUT = ROOT / "artifacts" / "v6_1_hm01_2024q1"
Q3_OUT = ROOT / "artifacts" / "v6_3_d01_2024q3_oos"


def ensure_q3_checkout():
    if not (DATA / "races.csv").exists():
        subprocess.run(
            ["git", "sparse-checkout", "add", DATA_REL.as_posix()],
            cwd=ROOT,
            check=True,
        )


def read_rows(name: str):
    with (DATA / name).open("r", encoding="utf-8-sig", newline="") as f:
        yield from csv.DictReader(f)


def pfloat(x):
    try:
        v = float(x)
        return v if math.isfinite(v) and v > 0 else None
    except (TypeError, ValueError):
        return None


def pint(x):
    try:
        return int(float(x))
    except (TypeError, ValueError):
        return None


def parse_trio(s: str):
    s = (s or "").strip().replace("=", "-").replace(",", "-")
    xs = tuple(sorted(int(x) for x in s.split("-") if x.strip().isdigit()))
    return xs if len(xs) == 3 and len(set(xs)) == 3 else None


def parse_tf(s: str):
    s = (s or "").strip().replace("=", "-").replace(",", "-")
    xs = tuple(int(x) for x in s.split("-") if x.strip().isdigit())
    return xs if len(xs) == 3 and len(set(xs)) == 3 else None


def parse_lines(s: str):
    lines = []
    for raw in (s or "").strip().split("/"):
        raw = raw.strip()
        if not raw:
            continue
        xs = []
        for x in raw.split("-"):
            x = x.strip()
            if not x.isdigit():
                return None
            xs.append(int(x))
        if not xs:
            return None
        lines.append(tuple(xs))
    flat = [x for line in lines for x in line]
    if not lines or len(flat) != len(set(flat)):
        return None
    return tuple(lines)


def implied(odds_map):
    inv = {k: 1.0 / v for k, v in odds_map.items() if v and v > 0}
    z = sum(inv.values())
    return {k: v / z for k, v in inv.items()} if z else {}


def load_data():
    races = {r["race_id"]: r for r in read_rows("races.csv")}

    trio = defaultdict(dict)
    for r in read_rows("trio_final_odds.csv"):
        c, o = parse_trio(r.get("combination")), pfloat(r.get("odds"))
        if c and o and r.get("odds_status") == "available":
            trio[r["race_id"]][c] = o

    tf = defaultdict(dict)
    for r in read_rows("trifecta_final_odds.csv"):
        c, o = parse_tf(r.get("combination")), pfloat(r.get("odds"))
        if c and o and r.get("odds_status") == "available":
            tf[r["race_id"]][c] = o

    payouts = defaultdict(dict)
    for r in read_rows("payouts.csv"):
        if r.get("bet_code") != "trifecta" and r.get("ticket_type") != "3連単":
            continue
        if r.get("status") != "paid":
            continue
        c, y = parse_tf(r.get("combination")), pint(r.get("payout_yen"))
        if c and y is not None:
            payouts[r["race_id"]][c] = y

    return races, trio, tf, payouts


def analyze_race(r, trio_odds, tf_odds):
    p3 = implied(trio_odds)
    q = implied(tf_odds)
    cars = sorted({x for c in trio_odds for x in c})
    lines = parse_lines(r.get("predicted_line_formation"))
    if lines is None or set(x for line in lines for x in line) != set(cars):
        return None

    S = {i: sum(p for c, p in p3.items() if i in c) / 3.0 for i in cars}
    LS = [sum(S[i] for i in line) for line in lines]
    line_order = sorted(range(len(lines)), key=lambda i: (-LS[i], i))
    if len(line_order) < 2:
        return None
    ai, bi = line_order[:2]
    A, B = lines[ai], lines[bi]

    pair_rows = []
    for li, line in enumerate(lines):
        for pos in range(len(line) - 1):
            pair = (line[pos], line[pos + 1])
            ps = sum(p for c, p in p3.items() if pair[0] in c and pair[1] in c)
            pair_rows.append((ps, li, pos, pair))
    pair_rows.sort(key=lambda x: (-x[0], x[1], x[2]))
    if len(pair_rows) < 2 or {pair_rows[0][1], pair_rows[1][1]} != {ai, bi}:
        return {"entry_pass": False, "reason": "PS_TOP2_NOT_AB"}

    H = {i: sum(p for t, p in q.items() if t[0] == i) for i in cars}
    hrank = sorted(cars, key=lambda i: (-H[i], i))
    top2 = hrank[:2]
    line_of = {car: li for li, line in enumerate(lines) for car in line}
    if {line_of[top2[0]], line_of[top2[1]]} != {ai, bi}:
        return {"entry_pass": False, "reason": "H_TOP2_NOT_AB"}

    pos_of = {car: pos for line in lines for pos, car in enumerate(line)}
    if any(pos_of[x] > 1 for x in top2):
        return {"entry_pass": False, "reason": "H_TOP_NOT_HEAD2"}

    H1, H2 = H[top2[0]], H[top2[1]]
    if not (H1 < 2.0 * H2):
        return {"entry_pass": False, "reason": "H1_GE_2H2"}

    if len(A) < 2 or len(B) < 2:
        return {"entry_pass": False, "reason": "AB_TOO_SHORT"}

    AH = max(A[:2], key=lambda i: (H[i], -i))
    BH = max(B[:2], key=lambda i: (H[i], -i))
    A_other = A[1] if AH == A[0] else A[0]
    B_other = B[1] if BH == B[0] else B[0]

    outside = [i for i in cars if i not in A and i not in B]
    ext = max(outside, key=lambda i: (S[i], -i)) if outside else None

    third_pool = set(A[:3]) | set(B[:3])
    if ext is not None:
        third_pool.add(ext)

    pre = set()
    for first, seconds in ((AH, (A_other, BH)), (BH, (B_other, AH))):
        for second in seconds:
            if second == first:
                continue
            for third in third_pool:
                if third not in (first, second):
                    pre.add((first, second, third))

    kept = set(pre)
    while True:
        n = len(kept)
        nxt = {t for t in kept if t in tf_odds and tf_odds[t] > n}
        if nxt == kept:
            break
        kept = nxt
        if not kept:
            break

    m_pre = sum(q.get(t, 0.0) for t in pre)
    return {
        "entry_pass": True,
        "pre": pre,
        "kept": kept,
        "generated_N": len(pre),
        "final_bet_count": len(kept),
        "M_pre": m_pre,
    }


def max_losing_streak(rows):
    ordered = sorted(rows, key=lambda r: (r["race_date"], r["race_id"]))
    cur = best = 0
    for r in ordered:
        if int(r["hit"]):
            cur = 0
        else:
            cur += 1
            best = max(best, cur)
    return best


def summarize(rows):
    bet_races = len(rows)
    hits = sum(int(r["hit"]) for r in rows)
    tickets = sum(int(r["final_bet_count"]) for r in rows)
    payout = sum(int(r["payout_yen"]) for r in rows)
    stake = tickets * STAKE
    return {
        "bet_races": bet_races,
        "hit_races": hits,
        "miss_races": bet_races - hits,
        "hit_rate_pct": (100 * hits / bet_races) if bet_races else None,
        "tickets": tickets,
        "avg_tickets_per_race": (tickets / bet_races) if bet_races else None,
        "stake_yen": stake,
        "payout_yen": payout,
        "profit_yen": payout - stake,
        "roi_pct": (100 * payout / stake) if stake else None,
        "max_losing_streak": max_losing_streak(rows),
    }


def main():
    ensure_q3_checkout()
    races, trio, tf, payouts = load_data()

    attr = defaultdict(int)
    attr["races_csv"] = len(races)
    entry_fail_reasons = defaultdict(int)
    rows = []

    for rid, r in sorted(
        races.items(),
        key=lambda kv: (kv[1].get("race_date", ""), kv[0]),
    ):
        if r.get("meeting_grade") != "F1" or not (r.get("race_type") or "").startswith("Ｓ級"):
            continue
        attr["f1_s"] += 1

        if pint(r.get("entry_count")) != 7:
            continue
        attr["seven_car"] += 1

        if len(trio.get(rid, {})) != 35:
            continue
        attr["complete_trio35"] += 1

        if len(tf.get(rid, {})) != 210:
            continue
        attr["complete_tf210"] += 1

        cars = sorted({x for c in trio[rid] for x in c})
        if len(cars) != 7:
            continue

        lines = parse_lines(r.get("predicted_line_formation"))
        if lines is None or set(x for line in lines for x in line) != set(cars):
            continue
        attr["complete_line"] += 1

        if rid not in payouts or not payouts[rid]:
            continue
        attr["has_tf_payout"] += 1
        attr["population"] += 1

        x = analyze_race(r, trio[rid], tf[rid])
        if not x or not x.get("entry_pass"):
            entry_fail_reasons[(x or {}).get("reason", "ANALYZE_FAIL")] += 1
            continue
        attr["entry_pass"] += 1

        if x["final_bet_count"] < 2:
            entry_fail_reasons["FINAL_LT2"] += 1
            continue
        attr["bet_races"] += 1

        paid = payouts[rid]
        winning_kept = [t for t in x["kept"] if t in paid]
        hit = bool(winning_kept)
        payout = sum(paid[t] for t in winning_kept)

        rows.append(
            {
                "race_id": rid,
                "race_date": r.get("race_date"),
                "track": r.get("track"),
                "race_no": pint(r.get("race_no")),
                "hit": int(hit),
                "payout_yen": payout,
                "generated_N": x["generated_N"],
                "final_bet_count": x["final_bet_count"],
                "M_pre": x["M_pre"],
            }
        )

    base = summarize(rows)
    selected = [r for r in rows if float(r["M_pre"]) >= ENTRY_FILTER_THRESHOLD]
    removed = [r for r in rows if float(r["M_pre"]) < ENTRY_FILTER_THRESHOLD]
    d01 = summarize(selected)
    removed_summary = summarize(removed)

    result = {
        "scheme_version": VALIDATION_VERSION,
        "base_scheme_version": BASE_SCHEME_VERSION,
        "validation_dataset": VALIDATION_DATASET,
        "validation_status": "OUT_OF_SAMPLE",
        "entry_filter": {
            "name": "M_PRE_FORMATION_MASS",
            "rule": f"M_pre >= {ENTRY_FILTER_THRESHOLD}",
            "threshold": ENTRY_FILTER_THRESHOLD,
            "threshold_source": "fixed on 2024Q1 before Q2/Q3 validation",
        },
        "attrition": dict(attr),
        "entry_fail_reasons": dict(entry_fail_reasons),
        "base_v6_1": base,
        "v6_3_d01": d01,
        "removed_by_filter": removed_summary,
        "delta": {
            "bet_races": d01["bet_races"] - base["bet_races"],
            "hit_races": d01["hit_races"] - base["hit_races"],
            "hit_rate_pp": d01["hit_rate_pct"] - base["hit_rate_pct"],
            "roi_pp": d01["roi_pct"] - base["roi_pct"],
            "profit_yen": d01["profit_yen"] - base["profit_yen"],
            "max_losing_streak": d01["max_losing_streak"] - base["max_losing_streak"],
        },
        "verdict_by_user_rule": (
            "PASS_HIT_RATE_IMPROVED"
            if d01["hit_rate_pct"] > base["hit_rate_pct"]
            else "FAIL_HIT_RATE_NOT_IMPROVED"
        ),
        "no_q3_tuning": True,
    }

    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "v6_1_hm01_races.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            w.writeheader()
            w.writerows(rows)
    (OUT / "v6_1_hm01_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    Q3_OUT.mkdir(parents=True, exist_ok=True)
    (Q3_OUT / "v6_3_d01_2024q3_oos_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if selected:
        with (Q3_OUT / "v6_3_d01_2024q3_oos_selected_races.csv").open(
            "w", encoding="utf-8-sig", newline=""
        ) as f:
            w = csv.DictWriter(f, fieldnames=list(selected[0].keys()))
            w.writeheader()
            w.writerows(selected)

    print("Q3_OOS_RESULT_BEGIN")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print("Q3_OOS_RESULT_END")


if __name__ == "__main__":
    main()
