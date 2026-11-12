from __future__ import annotations

import csv
import json
import math
import statistics
import subprocess
from collections import defaultdict
from pathlib import Path

BASE_SCHEME_VERSION = "v6.1"
PARENT_SCHEME_VERSION = "v6.3-D01"
SCHEME_VERSION = "v6.4-D02"
DEVELOPMENT_DATASET = "2024Q2+2024Q3"
FINAL_HOLDOUT = "2024Q4"
D01_M_PRE_THRESHOLD = 0.35640013538348414
STAKE = 100

ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT / "data" / "2024" / "s_class_f1_all_parts"
OUT = ROOT / "artifacts" / "v6_4_d02_q2q3_dev"

CANDIDATES = {
    "AB_SHARE": "higher",
    "LS2_LS3": "higher",
    "Q1": "higher",
    "Q12": "higher",
    "HEADGAP_MIN": "higher",
    "PruneDamage": "lower",
}


def ensure_checkout(quarter: str):
    rel = Path("data/2024/s_class_f1_all_parts") / quarter
    if not (ROOT / rel / "races.csv").exists():
        subprocess.run(["git", "sparse-checkout", "add", rel.as_posix()], cwd=ROOT, check=True)


def read_rows(data: Path, name: str):
    with (data / name).open("r", encoding="utf-8-sig", newline="") as f:
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


def load_data(quarter: str):
    ensure_checkout(quarter)
    data = DATA_ROOT / quarter
    races = {r["race_id"]: r for r in read_rows(data, "races.csv")}

    trio = defaultdict(dict)
    for r in read_rows(data, "trio_final_odds.csv"):
        c, o = parse_trio(r.get("combination")), pfloat(r.get("odds"))
        if c and o and r.get("odds_status") == "available":
            trio[r["race_id"]][c] = o

    tf = defaultdict(dict)
    for r in read_rows(data, "trifecta_final_odds.csv"):
        c, o = parse_tf(r.get("combination")), pfloat(r.get("odds"))
        if c and o and r.get("odds_status") == "available":
            tf[r["race_id"]][c] = o

    payouts = defaultdict(dict)
    for r in read_rows(data, "payouts.csv"):
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
        return {"entry_pass": False}

    H = {i: sum(p for t, p in q.items() if t[0] == i) for i in cars}
    hrank = sorted(cars, key=lambda i: (-H[i], i))
    top2 = hrank[:2]
    line_of = {car: li for li, line in enumerate(lines) for car in line}
    if {line_of[top2[0]], line_of[top2[1]]} != {ai, bi}:
        return {"entry_pass": False}
    pos_of = {car: pos for line in lines for pos, car in enumerate(line)}
    if any(pos_of[x] > 1 for x in top2):
        return {"entry_pass": False}
    H1, H2 = H[top2[0]], H[top2[1]]
    if not (H1 < 2.0 * H2):
        return {"entry_pass": False}
    if len(A) < 2 or len(B) < 2:
        return {"entry_pass": False}

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

    ls_sorted = sorted(LS, reverse=True)
    m_pre = sum(q.get(t, 0.0) for t in pre)
    m_keep = sum(q.get(t, 0.0) for t in kept)
    q1 = H[AH] + H[BH]
    allowed12 = {(AH, A_other), (AH, BH), (BH, B_other), (BH, AH)}
    q12 = sum(p for t, p in q.items() if (t[0], t[1]) in allowed12)
    prune_damage = (m_pre - m_keep) / m_pre if m_pre > 0 else None

    return {
        "entry_pass": True,
        "kept": kept,
        "final_bet_count": len(kept),
        "M_pre": m_pre,
        "AB_SHARE": LS[ai] + LS[bi],
        "LS2_LS3": (ls_sorted[1] / ls_sorted[2]) if len(ls_sorted) > 2 and ls_sorted[2] > 0 else None,
        "Q1": q1,
        "Q12": q12,
        "HEADGAP_MIN": min(H[AH] - H[A_other], H[BH] - H[B_other]),
        "PruneDamage": prune_damage,
    }


def max_losing_streak(rows):
    ordered = sorted(rows, key=lambda r: (r["race_date"], r["race_id"]))
    cur = best = 0
    for r in ordered:
        if r["hit"]:
            cur = 0
        else:
            cur += 1
            best = max(best, cur)
    return best


def summarize(rows):
    n = len(rows)
    hits = sum(int(r["hit"]) for r in rows)
    tickets = sum(int(r["final_bet_count"]) for r in rows)
    payout = sum(int(r["payout_yen"]) for r in rows)
    stake = tickets * STAKE
    return {
        "bet_races": n,
        "hit_races": hits,
        "miss_races": n - hits,
        "hit_rate_pct": 100 * hits / n if n else None,
        "tickets": tickets,
        "avg_tickets_per_race": tickets / n if n else None,
        "stake_yen": stake,
        "payout_yen": payout,
        "profit_yen": payout - stake,
        "roi_pct": 100 * payout / stake if stake else None,
        "max_losing_streak": max_losing_streak(rows),
    }


def cliffs_delta(hit, miss):
    if not hit or not miss:
        return None
    gt = lt = 0
    for a in hit:
        for b in miss:
            gt += a > b
            lt += a < b
    return (gt - lt) / (len(hit) * len(miss))


def collect(quarter: str):
    races, trio, tf, payouts = load_data(quarter)
    rows = []
    for rid, r in sorted(races.items(), key=lambda kv: (kv[1].get("race_date", ""), kv[0])):
        if r.get("meeting_grade") != "F1" or not (r.get("race_type") or "").startswith("Ｓ級"):
            continue
        if pint(r.get("entry_count")) != 7 or len(trio.get(rid, {})) != 35 or len(tf.get(rid, {})) != 210:
            continue
        cars = sorted({x for c in trio[rid] for x in c})
        if len(cars) != 7:
            continue
        lines = parse_lines(r.get("predicted_line_formation"))
        if lines is None or set(x for line in lines for x in line) != set(cars):
            continue
        if rid not in payouts or not payouts[rid]:
            continue
        x = analyze_race(r, trio[rid], tf[rid])
        if not x or not x.get("entry_pass") or x["final_bet_count"] < 2:
            continue
        if x["M_pre"] < D01_M_PRE_THRESHOLD:
            continue
        paid = payouts[rid]
        wins = [t for t in x["kept"] if t in paid]
        row = {
            "quarter": quarter,
            "race_id": rid,
            "race_date": r.get("race_date"),
            "track": r.get("track"),
            "race_no": pint(r.get("race_no")),
            "hit": bool(wins),
            "payout_yen": sum(paid[t] for t in wins),
            "final_bet_count": x["final_bet_count"],
            "M_pre": x["M_pre"],
        }
        for k in CANDIDATES:
            row[k] = x[k]
        rows.append(row)
    return rows


def median_values(rows, key, hit_value):
    vals = [float(r[key]) for r in rows if r[key] is not None and bool(r["hit"]) == hit_value]
    return statistics.median(vals) if vals else None


def select_rule(rows_by_q):
    all_rows = rows_by_q["2024_q2"] + rows_by_q["2024_q3"]
    diagnostics = {}
    eligible = []
    for key, direction in CANDIDATES.items():
        per_q = {}
        same_direction = True
        adjusted_deltas = []
        for q in ("2024_q2", "2024_q3"):
            rows = rows_by_q[q]
            h = [float(r[key]) for r in rows if r[key] is not None and r["hit"]]
            m = [float(r[key]) for r in rows if r[key] is not None and not r["hit"]]
            hm = statistics.median(h) if h else None
            mm = statistics.median(m) if m else None
            raw_delta = cliffs_delta(h, m)
            adjusted = raw_delta if direction == "higher" else (-raw_delta if raw_delta is not None else None)
            expected = (hm is not None and mm is not None and (hm > mm if direction == "higher" else hm < mm))
            same_direction = same_direction and expected
            if adjusted is not None:
                adjusted_deltas.append(adjusted)
            per_q[q] = {
                "hit_median": hm,
                "miss_median": mm,
                "cliffs_delta_raw": raw_delta,
                "cliffs_delta_adjusted": adjusted,
                "direction_ok": expected,
            }

        pooled_hit = median_values(all_rows, key, True)
        pooled_miss = median_values(all_rows, key, False)
        threshold = (pooled_hit + pooled_miss) / 2 if pooled_hit is not None and pooled_miss is not None else None
        stability_score = min(adjusted_deltas) if len(adjusted_deltas) == 2 else None
        diagnostics[key] = {
            "direction": direction,
            "per_quarter": per_q,
            "pooled_hit_median": pooled_hit,
            "pooled_miss_median": pooled_miss,
            "threshold_midpoint": threshold,
            "same_direction_q2_q3": same_direction,
            "stability_score_min_adjusted_cliffs": stability_score,
        }
        if same_direction and stability_score is not None and stability_score > 0 and threshold is not None:
            eligible.append((stability_score, key, direction, threshold))

    eligible.sort(reverse=True)
    if not eligible:
        return diagnostics, None
    _, key, direction, threshold = eligible[0]
    return diagnostics, {"metric": key, "direction": direction, "threshold": threshold}


def apply_rule(rows, rule):
    key, direction, threshold = rule["metric"], rule["direction"], rule["threshold"]
    if direction == "higher":
        return [r for r in rows if r[key] is not None and float(r[key]) >= threshold]
    return [r for r in rows if r[key] is not None and float(r[key]) <= threshold]


def main():
    rows_by_q = {q: collect(q) for q in ("2024_q2", "2024_q3")}
    diagnostics, rule = select_rule(rows_by_q)
    if rule is None:
        result = {
            "scheme_version": SCHEME_VERSION,
            "parent_scheme_version": PARENT_SCHEME_VERSION,
            "development_dataset": DEVELOPMENT_DATASET,
            "final_holdout": FINAL_HOLDOUT,
            "status": "NO_STABLE_SECOND_FILTER_FOUND",
            "diagnostics": diagnostics,
        }
    else:
        quarter_results = {}
        for q, rows in rows_by_q.items():
            selected = apply_rule(rows, rule)
            removed = [r for r in rows if r not in selected]
            quarter_results[q] = {
                "d01_before": summarize(rows),
                "d02_after": summarize(selected),
                "removed_by_d02": summarize(removed),
            }
        pooled = rows_by_q["2024_q2"] + rows_by_q["2024_q3"]
        pooled_selected = apply_rule(pooled, rule)
        result = {
            "scheme_version": SCHEME_VERSION,
            "base_scheme_version": BASE_SCHEME_VERSION,
            "parent_scheme_version": PARENT_SCHEME_VERSION,
            "development_dataset": DEVELOPMENT_DATASET,
            "final_holdout": FINAL_HOLDOUT,
            "status": "DEVELOPMENT_COMPLETE_Q4_UNTOUCHED",
            "fixed_parent_filter": f"M_pre >= {D01_M_PRE_THRESHOLD}",
            "selection_protocol": "Among predeclared candidates, require same hit-vs-miss median direction in Q2 and Q3; choose largest minimum direction-adjusted Cliff's delta; threshold is pooled hit/miss median midpoint.",
            "added_entry_filter": rule,
            "diagnostics": diagnostics,
            "quarter_results": quarter_results,
            "pooled_q2_q3": {
                "d01_before": summarize(pooled),
                "d02_after": summarize(pooled_selected),
                "removed_by_d02": summarize([r for r in pooled if r not in pooled_selected]),
            },
            "q4_used": False,
        }

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "v6_4_d02_q2q3_summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print("V6_4_D02_DEV_BEGIN")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print("V6_4_D02_DEV_END")


if __name__ == "__main__":
    main()
