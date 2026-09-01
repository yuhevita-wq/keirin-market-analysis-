from __future__ import annotations

import csv
import itertools
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

BASE_SCHEME_VERSION = "v6.1"
ANALYSIS_VERSION = "v6.1-HM01"
ANALYSIS_PURPOSE = "Hit 72 vs miss 188 diagnostics; no scheme changes"
IMPLEMENTATION_STATUS = "reconstructed_from_canonical_spec_pending_benchmark_reproduction"
DEVELOPMENT_DATASET = "2024Q1"
STAKE = 100

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "2024" / "s_class_f1_all_parts" / "2024_q1"
OUT = ROOT / "artifacts" / "v6_1_hm01_2024q1"

CANONICAL = {
    "population": 1191,
    "bet_races": 260,
    "hit_races": 72,
    "tickets": 3725,
    "stake_yen": 372500,
    "payout_yen": 341200,
    "max_losing_streak": 18,
}


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


def quantile(xs, p):
    ys = sorted(x for x in xs if x is not None and math.isfinite(x))
    if not ys:
        return None
    if len(ys) == 1:
        return ys[0]
    pos = (len(ys) - 1) * p
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return ys[lo]
    return ys[lo] * (hi - pos) + ys[hi] * (pos - lo)


def stat_block(xs):
    ys = [x for x in xs if x is not None and math.isfinite(x)]
    if not ys:
        return {"n": 0, "mean": None, "median": None, "q25": None, "q75": None, "iqr": None}
    q25, q75 = quantile(ys, .25), quantile(ys, .75)
    return {
        "n": len(ys),
        "mean": statistics.fmean(ys),
        "median": statistics.median(ys),
        "q25": q25,
        "q75": q75,
        "iqr": q75 - q25,
    }


def cliffs_delta(hit, miss):
    a = [x for x in hit if x is not None and math.isfinite(x)]
    b = [x for x in miss if x is not None and math.isfinite(x)]
    if not a or not b:
        return None
    gt = lt = 0
    for x in a:
        for y in b:
            gt += x > y
            lt += x < y
    return (gt - lt) / (len(a) * len(b))


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

    result_rows = defaultdict(list)
    for r in read_rows("results.csv"):
        pos, car = pint(r.get("finish_position")), pint(r.get("car_no"))
        if pos is not None and car is not None:
            result_rows[r["race_id"]].append((pos, car))
    results = {}
    for rid, xs in result_rows.items():
        ys = sorted(xs)
        if len(ys) >= 3 and len({p for p, _ in ys[:3]}) == 3:
            results[rid] = tuple(car for _, car in ys[:3])

    payouts = defaultdict(dict)
    for r in read_rows("payouts.csv"):
        if r.get("bet_code") != "trifecta" and r.get("ticket_type") != "3連単":
            continue
        if r.get("status") != "paid":
            continue
        c, y = parse_tf(r.get("combination")), pint(r.get("payout_yen"))
        if c and y is not None:
            payouts[r["race_id"]][c] = y

    return races, trio, tf, results, payouts


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

    ls_sorted = sorted(LS, reverse=True)
    hvals = [H[i] for i in hrank]
    psvals = [x[0] for x in pair_rows]
    outside_mass = 1.0 - (LS[ai] + LS[bi])
    o1s = S[ext] if ext is not None else 0.0
    out_conc = o1s / outside_mass if outside_mass > 1e-15 else None
    q1 = H[AH] + H[BH]
    allowed12 = {(AH, A_other), (AH, BH), (BH, B_other), (BH, AH)}
    q12 = sum(p for t, p in q.items() if (t[0], t[1]) in allowed12)
    m_pre = sum(q.get(t, 0.0) for t in pre)
    m_keep = sum(q.get(t, 0.0) for t in kept)
    prune_damage = (m_pre - m_keep) / m_pre if m_pre > 0 else None
    la = LS[ai] / (LS[ai] + LS[bi]) if LS[ai] + LS[bi] > 0 else None
    ha = H[AH] / (H[AH] + H[BH]) if H[AH] + H[BH] > 0 else None

    return {
        "entry_pass": True,
        "A": A, "B": B, "AH": AH, "BH": BH, "A_other": A_other, "B_other": B_other,
        "pre": pre, "kept": kept, "generated_N": len(pre), "final_bet_count": len(kept),
        "LS1": ls_sorted[0] if len(ls_sorted) > 0 else None,
        "LS2": ls_sorted[1] if len(ls_sorted) > 1 else None,
        "LS3": ls_sorted[2] if len(ls_sorted) > 2 else None,
        "LS2_LS3": (ls_sorted[1] / ls_sorted[2]) if len(ls_sorted) > 2 and ls_sorted[2] > 0 else None,
        "AB_SHARE": LS[ai] + LS[bi],
        "H1": hvals[0], "H2": hvals[1], "H3": hvals[2],
        "H1_H2": hvals[0] / hvals[1] if hvals[1] > 0 else None,
        "H2_H3": hvals[1] / hvals[2] if hvals[2] > 0 else None,
        "H1_PLUS_H2": hvals[0] + hvals[1],
        "PS1": psvals[0] if len(psvals) > 0 else None,
        "PS2": psvals[1] if len(psvals) > 1 else None,
        "PS3": psvals[2] if len(psvals) > 2 else None,
        "PS2_PS3": psvals[1] / psvals[2] if len(psvals) > 2 and psvals[2] > 0 else None,
        "HeadGap_A": H[AH] - H[A_other],
        "HeadGap_B": H[BH] - H[B_other],
        "OUTSIDE_MASS": outside_mass,
        "O1_S": o1s,
        "OutConcentration": out_conc,
        "Q1": q1,
        "Q12": q12,
        "Q12_Q1": q12 / q1 if q1 > 0 else None,
        "M_pre": m_pre,
        "M_keep": m_keep,
        "PruneDamage": prune_damage,
        "Alignment": abs(la - ha) if la is not None and ha is not None else None,
    }


def miss_stage(order, x):
    if order in x["kept"]:
        return "HIT"
    if order in x["pre"]:
        return "PRICE_DROP"
    first, second, _ = order
    if first not in (x["AH"], x["BH"]):
        return "FIRST_COLLAPSE"
    allowed = (x["A_other"], x["BH"]) if first == x["AH"] else (x["B_other"], x["AH"])
    if second not in allowed:
        return "SECOND_COLLAPSE"
    return "THIRD_COLLAPSE"


def main():
    races, trio, tf, results, payouts = load_data()
    attr = defaultdict(int)
    attr["races_csv"] = len(races)
    rows = []
    entry_reasons = defaultdict(int)

    for rid, r in sorted(races.items(), key=lambda kv: (kv[1].get("race_date", ""), int(kv[1].get("race_no") or 0), kv[0])):
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
            entry_reasons[(x or {}).get("reason", "ANALYZE_FAIL")] += 1
            continue
        attr["entry_pass"] += 1
        if x["final_bet_count"] < 2:
            entry_reasons["FINAL_LT2"] += 1
            continue
        attr["bet_races"] += 1

        paid = payouts[rid]
        winning_kept = [t for t in x["kept"] if t in paid]
        hit = bool(winning_kept)
        payout = sum(paid[t] for t in winning_kept)
        actual_orders = list(paid)
        if hit:
            mtype = "HIT"
        else:
            priority = {"FIRST_COLLAPSE": 1, "SECOND_COLLAPSE": 2, "THIRD_COLLAPSE": 3, "PRICE_DROP": 4, "HIT": 5}
            stages = [miss_stage(t, x) for t in actual_orders]
            mtype = max(stages, key=lambda z: priority[z]) if stages else "NO_RESULT"

        row = {
            "race_id": rid,
            "race_date": r.get("race_date"),
            "track": r.get("track"),
            "race_no": pint(r.get("race_no")),
            "hit": int(hit),
            "miss_type": mtype,
            "actual_trifecta": "/".join("-".join(map(str, t)) for t in actual_orders),
            "payout_yen": payout,
        }
        for k, v in x.items():
            if k in {"entry_pass", "pre", "kept", "A", "B", "AH", "BH", "A_other", "B_other"}:
                continue
            row[k] = v
        row.update({"AH": x["AH"], "BH": x["BH"]})
        rows.append(row)

    tickets = sum(r["final_bet_count"] for r in rows)
    hits = sum(r["hit"] for r in rows)
    payout = sum(r["payout_yen"] for r in rows)
    stake = tickets * STAKE
    streak = max_streak = 0
    for r in rows:
        if r["hit"]:
            streak = 0
        else:
            streak += 1
            max_streak = max(max_streak, streak)

    observed = {
        "population": attr["population"],
        "bet_races": len(rows),
        "hit_races": hits,
        "tickets": tickets,
        "stake_yen": stake,
        "payout_yen": payout,
        "profit_yen": payout - stake,
        "hit_rate_pct": 100 * hits / len(rows) if rows else None,
        "roi_pct": 100 * payout / stake if stake else None,
        "max_losing_streak": max_streak,
    }
    checks = {k: observed.get(k) == v for k, v in CANONICAL.items()}
    baseline_match = all(checks.values())

    metrics = [
        "LS2_LS3", "AB_SHARE", "H1_H2", "H2_H3", "H1_PLUS_H2", "PS2_PS3",
        "HeadGap_A", "HeadGap_B", "OUTSIDE_MASS", "O1_S", "OutConcentration",
        "Q1", "Q12", "Q12_Q1", "M_pre", "M_keep", "PruneDamage", "Alignment",
        "generated_N", "final_bet_count",
    ]
    hrows = [r for r in rows if r["hit"]]
    mrows = [r for r in rows if not r["hit"]]
    hm = {}
    for metric in metrics:
        hv = [r.get(metric) for r in hrows]
        mv = [r.get(metric) for r in mrows]
        hs, ms = stat_block(hv), stat_block(mv)
        hm[metric] = {
            "hit": hs,
            "miss": ms,
            "median_diff_hit_minus_miss": (hs["median"] - ms["median"]) if hs["median"] is not None and ms["median"] is not None else None,
            "cliffs_delta_hit_vs_miss": cliffs_delta(hv, mv),
        }

    miss_types = defaultdict(int)
    for r in mrows:
        miss_types[r["miss_type"]] += 1

    OUT.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with (OUT / "v6_1_hm01_races.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if fieldnames:
            w.writeheader(); w.writerows(rows)

    summary = {
        "base_scheme_version": BASE_SCHEME_VERSION,
        "analysis_version": ANALYSIS_VERSION,
        "analysis_purpose": ANALYSIS_PURPOSE,
        "implementation_status": IMPLEMENTATION_STATUS,
        "development_dataset": DEVELOPMENT_DATASET,
        "canonical_benchmark": CANONICAL,
        "observed": observed,
        "baseline_match_canonical": baseline_match,
        "benchmark_checks": checks,
        "attrition": dict(attr),
        "entry_fail_reasons": dict(entry_reasons),
        "hit_miss_counts": {"hit": len(hrows), "miss": len(mrows)},
        "miss_type_counts": dict(miss_types),
        "hm01_metrics": hm,
        "upgrade_guard": "Do not derive or evaluate a new entry threshold unless baseline_match_canonical is true.",
    }
    (OUT / "v6_1_hm01_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
