from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

YEARS = (2023, 2024, 2025)
OUT = Path("data/audits/odds_contradiction_analysis_2023_2025.json")
OUT_COMPACT = Path("data/audits/odds_contradiction_analysis_2023_2025_compact.json")


def read_csv(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        yield from csv.DictReader(f)


def parse_combo(text: str, sep: str) -> tuple[int, ...]:
    return tuple(int(x) for x in text.split(sep))


def normalize_market(items: dict[tuple[int, ...], float]) -> dict[tuple[int, ...], float]:
    inv = {k: 1.0 / v for k, v in items.items() if v and v > 0}
    s = sum(inv.values())
    return {k: v / s for k, v in inv.items()} if s else {}


def safe_log_ratio(a: float, b: float) -> float:
    eps = 1e-15
    return math.log(max(a, eps) / max(b, eps))


def fit_set_maxent(cars: list[int], combos: list[tuple[int, int, int]], target: dict[int, float]) -> dict[tuple[int, int, int], float]:
    # Maximum-entropy 3-subset model q(S) proportional to product_i w_i,
    # fitted only to each rider's market-implied top-3 inclusion marginal.
    w = {c: 1.0 for c in cars}
    probs: dict[tuple[int, int, int], float] = {}
    for _ in range(80):
        raw = {s: w[s[0]] * w[s[1]] * w[s[2]] for s in combos}
        z = sum(raw.values())
        probs = {s: x / z for s, x in raw.items()}
        cur = {c: 0.0 for c in cars}
        for s, p in probs.items():
            for c in s:
                cur[c] += p
        err = max(abs(cur[c] - target[c]) for c in cars)
        if err < 1e-10:
            break
        for c in cars:
            if cur[c] > 0 and target[c] > 0:
                r = max(0.05, min(20.0, target[c] / cur[c]))
                w[c] *= r ** 0.55
        # Scale is unidentified; renormalize weights for numerical stability.
        gm = math.exp(sum(math.log(max(w[c], 1e-300)) for c in cars) / len(cars))
        for c in cars:
            w[c] /= gm
    return probs


def fit_order_maxent(cars: list[int], orders: list[tuple[int, int, int]], q: dict[tuple[int, int, int], float]) -> tuple[dict[tuple[int, int, int], float], dict[int, float], dict[int, float], dict[int, float]]:
    # Position-only log-linear model with structural zeros (same car cannot occupy two places).
    t1 = {c: 0.0 for c in cars}; t2 = {c: 0.0 for c in cars}; t3 = {c: 0.0 for c in cars}
    for (a, b, c), p in q.items():
        t1[a] += p; t2[b] += p; t3[c] += p
    w1 = {c: 1.0 for c in cars}; w2 = {c: 1.0 for c in cars}; w3 = {c: 1.0 for c in cars}
    probs: dict[tuple[int, int, int], float] = {}
    for _ in range(80):
        raw = {o: w1[o[0]] * w2[o[1]] * w3[o[2]] for o in orders}
        z = sum(raw.values())
        probs = {o: x / z for o, x in raw.items()}
        c1 = {c: 0.0 for c in cars}; c2 = {c: 0.0 for c in cars}; c3 = {c: 0.0 for c in cars}
        for (a, b, c), p in probs.items():
            c1[a] += p; c2[b] += p; c3[c] += p
        err = max(
            max(abs(c1[c]-t1[c]) for c in cars),
            max(abs(c2[c]-t2[c]) for c in cars),
            max(abs(c3[c]-t3[c]) for c in cars),
        )
        if err < 1e-10:
            break
        for c in cars:
            if c1[c] > 0 and t1[c] > 0: w1[c] *= max(0.05, min(20.0, t1[c]/c1[c])) ** 0.45
            if c2[c] > 0 and t2[c] > 0: w2[c] *= max(0.05, min(20.0, t2[c]/c2[c])) ** 0.45
            if c3[c] > 0 and t3[c] > 0: w3[c] *= max(0.05, min(20.0, t3[c]/c3[c])) ** 0.45
        for ww in (w1, w2, w3):
            gm = math.exp(sum(math.log(max(ww[c], 1e-300)) for c in cars) / len(cars))
            for c in cars: ww[c] /= gm
    return probs, t1, t2, t3


def decile_map(values: dict[tuple[int, ...], float]) -> dict[tuple[int, ...], int]:
    ordered = sorted(values, key=lambda k: (values[k], k))
    n = len(ordered)
    return {k: min(9, (i * 10) // n) for i, k in enumerate(ordered)}


def agg_new():
    return {"bets": 0, "wins": 0, "expected_market": 0.0, "expected_reference": 0.0, "gross_odds_return": 0.0}


def agg_add(a, win: bool, odds: float, p_market: float, p_ref: float):
    a["bets"] += 1
    a["wins"] += int(win)
    a["expected_market"] += p_market
    a["expected_reference"] += p_ref
    if win:
        a["gross_odds_return"] += odds


def finish_agg(a):
    bets = a["bets"]
    expm = a["expected_market"]
    expr = a["expected_reference"]
    return {
        **a,
        "hit_rate_pct": (a["wins"] / bets * 100.0) if bets else None,
        "roi_pct_final_odds": (a["gross_odds_return"] / bets * 100.0) if bets else None,
        "observed_to_market_expected": (a["wins"] / expm) if expm else None,
        "observed_to_reference_expected": (a["wins"] / expr) if expr else None,
    }


def load_results(base: Path) -> dict[str, tuple[int, int, int]]:
    tmp: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for r in read_csv(base / "results.csv"):
        try:
            pos = int(r["finish_position"]); car = int(r["car_no"])
        except Exception:
            continue
        if pos <= 3:
            tmp[r["race_id"]].append((pos, car))
    out = {}
    for rid, xs in tmp.items():
        xs = sorted(xs)
        if [p for p, _ in xs] == [1, 2, 3]:
            out[rid] = tuple(c for _, c in xs)
    return out


def load_odds(base: Path, filename: str, sep: str) -> dict[str, dict[tuple[int, ...], float]]:
    out: dict[str, dict[tuple[int, ...], float]] = defaultdict(dict)
    for r in read_csv(base / filename):
        if r.get("odds_status") != "available":
            continue
        try:
            odds = float(r["odds"])
            combo = parse_combo(r["combination"], sep)
        except Exception:
            continue
        out[r["race_id"]][combo] = odds
    return out


def summarize_table(d: dict, keys):
    return {str(k): finish_agg(d[k]) for k in keys}


def extreme_label(i: int, n: int) -> list[str]:
    labels = []
    if i == 0: labels.append("bottom1")
    if i < 3: labels.append("bottom3")
    if i == n - 1: labels.append("top1")
    if i >= n - 3: labels.append("top3")
    return labels


def analyze_year(year: int):
    base = Path(f"data/{year}/s_class_yosen")
    results = load_results(base)
    trio_odds = load_odds(base, "trio_final_odds.csv", "=")
    tri3_odds = load_odds(base, "trifecta_final_odds.csv", "-")

    d_dec = defaultdict(agg_new); c_dec = defaultdict(agg_new); h_dec = defaultdict(agg_new)
    d_ext = defaultdict(agg_new); c_ext = defaultdict(agg_new); h_ext = defaultdict(agg_new)
    indiv = defaultdict(lambda: {"cars":0,"top3":0,"expected_trio":0.0,"expected_trifecta":0.0})
    winner_rows = []
    eligible = 0

    common_ids = sorted(set(results) & set(trio_odds) & set(tri3_odds))
    for rid in common_ids:
        win_order = results[rid]
        win_set = tuple(sorted(win_order))
        trio_o = trio_odds[rid]; tri3_o = tri3_odds[rid]
        cars = sorted({c for s in trio_o for c in s})
        if len(cars) < 5:
            continue
        expected_trio_n = math.comb(len(cars), 3)
        expected_tri3_n = len(cars) * (len(cars)-1) * (len(cars)-2)
        if len(trio_o) != expected_trio_n or len(tri3_o) != expected_tri3_n:
            continue
        if win_set not in trio_o or win_order not in tri3_o:
            continue
        q_trio = normalize_market(trio_o)
        q_tri3 = normalize_market(tri3_o)
        if len(q_trio) != len(trio_o) or len(q_tri3) != len(tri3_o):
            continue
        eligible += 1

        # D: same 3-rider set, 3連単 aggregate versus 3連複 market.
        q_tri3_set = {s: 0.0 for s in q_trio}
        for o, p in q_tri3.items():
            s = tuple(sorted(o))
            if s in q_tri3_set:
                q_tri3_set[s] += p
        d = {s: safe_log_ratio(q_tri3_set[s], q_trio[s]) for s in q_trio}
        dmap = decile_map(d)
        ordered_d = sorted(d, key=lambda k: (d[k], k))
        for s in ordered_d:
            win = s == win_set
            agg_add(d_dec[dmap[s]], win, trio_o[s], q_trio[s], q_tri3_set[s])
        for i, s in enumerate(ordered_d):
            for label in extreme_label(i, len(ordered_d)):
                agg_add(d_ext[label], s == win_set, trio_o[s], q_trio[s], q_tri3_set[s])

        # I: individual top-3 inclusion disagreement between the two pools.
        m_trio = {c: 0.0 for c in cars}; m_tri3 = {c: 0.0 for c in cars}
        for s, p in q_trio.items():
            for c in s: m_trio[c] += p
        for o, p in q_tri3.items():
            for c in o: m_tri3[c] += p
        iv = {c: safe_log_ratio(m_tri3[c], m_trio[c]) for c in cars}
        ordered_i = sorted(cars, key=lambda c: (iv[c], c))
        ncar = len(cars)
        for i, c in enumerate(ordered_i):
            if i < 2: grp = "trio_stronger_bottom2"
            elif i >= ncar - 2: grp = "trifecta_stronger_top2"
            else: grp = "middle"
            a = indiv[grp]
            a["cars"] += 1
            a["top3"] += int(c in win_set)
            a["expected_trio"] += m_trio[c]
            a["expected_trifecta"] += m_tri3[c]

        # C: trio combination residual after fitting all individual inclusion marginals.
        qhat_set = fit_set_maxent(cars, list(q_trio), m_trio)
        cval = {s: safe_log_ratio(q_trio[s], qhat_set[s]) for s in q_trio}
        cmap = decile_map(cval)
        ordered_c = sorted(cval, key=lambda k: (cval[k], k))
        for s in ordered_c:
            agg_add(c_dec[cmap[s]], s == win_set, trio_o[s], q_trio[s], qhat_set[s])
        for i, s in enumerate(ordered_c):
            for label in extreme_label(i, len(ordered_c)):
                agg_add(c_ext[label], s == win_set, trio_o[s], q_trio[s], qhat_set[s])

        # H: trifecta order residual after fitting first/second/third marginals.
        qhat_order, _, _, _ = fit_order_maxent(cars, list(q_tri3), q_tri3)
        hval = {o: safe_log_ratio(q_tri3[o], qhat_order[o]) for o in q_tri3}
        hmap = decile_map(hval)
        ordered_h = sorted(hval, key=lambda k: (hval[k], k))
        for o in ordered_h:
            agg_add(h_dec[hmap[o]], o == win_order, tri3_o[o], q_tri3[o], qhat_order[o])
        for i, o in enumerate(ordered_h):
            for label in extreme_label(i, len(ordered_h)):
                agg_add(h_ext[label], o == win_order, tri3_o[o], q_tri3[o], qhat_order[o])

        winner_rows.append({
            "race_id": rid,
            "winner_order": "-".join(map(str, win_order)),
            "winner_set": "=".join(map(str, win_set)),
            "trio_odds": trio_o[win_set],
            "trifecta_odds": tri3_o[win_order],
            "D_log_ratio": d[win_set], "D_decile": dmap[win_set],
            "C_log_residual": cval[win_set], "C_decile": cmap[win_set],
            "H_log_residual": hval[win_order], "H_decile": hmap[win_order],
        })

    for k, a in indiv.items():
        a["top3_rate_pct"] = a["top3"] / a["cars"] * 100 if a["cars"] else None
        a["observed_to_trio_expected"] = a["top3"] / a["expected_trio"] if a["expected_trio"] else None
        a["observed_to_trifecta_expected"] = a["top3"] / a["expected_trifecta"] if a["expected_trifecta"] else None

    return {
        "year": year,
        "eligible_complete_races": eligible,
        "D_cross_market_set": {"deciles": summarize_table(d_dec, range(10)), "extremes": summarize_table(d_ext, ["bottom1","bottom3","top3","top1"])},
        "I_individual_cross_market": dict(indiv),
        "C_trio_combination_residual": {"deciles": summarize_table(c_dec, range(10)), "extremes": summarize_table(c_ext, ["bottom1","bottom3","top3","top1"])},
        "H_trifecta_order_residual": {"deciles": summarize_table(h_dec, range(10)), "extremes": summarize_table(h_ext, ["bottom1","bottom3","top3","top1"])},
        "winner_rows": winner_rows,
    }


def combine_aggs(year_results, metric: str, section: str, labels):
    out = {}
    for label in labels:
        a = agg_new()
        for yr in year_results:
            x = yr[metric][section][str(label)]
            for k in ("bets","wins"):
                a[k] += x[k]
            for k in ("expected_market","expected_reference","gross_odds_return"):
                a[k] += x[k]
        out[str(label)] = finish_agg(a)
    return out


def combine_indiv(year_results):
    out = {}
    groups = ("trio_stronger_bottom2", "middle", "trifecta_stronger_top2")
    for g in groups:
        a = {"cars":0,"top3":0,"expected_trio":0.0,"expected_trifecta":0.0}
        for yr in year_results:
            x = yr["I_individual_cross_market"].get(g, {})
            for k in a: a[k] += x.get(k, 0)
        a["top3_rate_pct"] = a["top3"] / a["cars"] * 100 if a["cars"] else None
        a["observed_to_trio_expected"] = a["top3"] / a["expected_trio"] if a["expected_trio"] else None
        a["observed_to_trifecta_expected"] = a["top3"] / a["expected_trifecta"] if a["expected_trifecta"] else None
        out[g] = a
    return out


def stable_findings(year_results):
    findings = []
    for metric in ("D_cross_market_set", "C_trio_combination_residual", "H_trifecta_order_residual"):
        for section, labels in (("deciles", [str(i) for i in range(10)]), ("extremes", ["bottom1","bottom3","top3","top1"])):
            for label in labels:
                rows = [yr[metric][section][label] for yr in year_results]
                if all(r["bets"] > 0 for r in rows):
                    cal = [r["observed_to_market_expected"] for r in rows]
                    roi = [r["roi_pct_final_odds"] for r in rows]
                    if all(x is not None and x > 1.0 for x in cal):
                        findings.append({
                            "metric": metric, "section": section, "bucket": label,
                            "all_years_observed_above_market_expected": True,
                            "all_years_roi_above_100": all(x is not None and x > 100.0 for x in roi),
                            "yearly_calibration_ratios": cal,
                            "yearly_roi_pct_final_odds": roi,
                        })
    return findings


def compact_from(full):
    def rows_for(metric):
        rows = []
        comb = full["combined"][metric]
        for sec in ("deciles", "extremes"):
            for bucket, x in comb[sec].items():
                rows.append({"section":sec,"bucket":bucket,**x})
        return rows
    return {
        "status": full["status"],
        "years_read": full["years_read"],
        "evaluation_year_2026_used": False,
        "odds_phase": "final",
        "important_limit": full["important_limit"],
        "eligible_complete_races_by_year": {str(y["year"]): y["eligible_complete_races"] for y in full["years"]},
        "definitions": full["definitions"],
        "D_cross_market_set": rows_for("D_cross_market_set"),
        "I_individual_cross_market": full["combined"]["I_individual_cross_market"],
        "C_trio_combination_residual": rows_for("C_trio_combination_residual"),
        "H_trifecta_order_residual": rows_for("H_trifecta_order_residual"),
        "stable_findings": full["stable_findings"],
    }


def main():
    years = [analyze_year(y) for y in YEARS]
    combined = {
        "eligible_complete_races": sum(y["eligible_complete_races"] for y in years),
        "D_cross_market_set": {
            "deciles": combine_aggs(years, "D_cross_market_set", "deciles", range(10)),
            "extremes": combine_aggs(years, "D_cross_market_set", "extremes", ["bottom1","bottom3","top3","top1"]),
        },
        "I_individual_cross_market": combine_indiv(years),
        "C_trio_combination_residual": {
            "deciles": combine_aggs(years, "C_trio_combination_residual", "deciles", range(10)),
            "extremes": combine_aggs(years, "C_trio_combination_residual", "extremes", ["bottom1","bottom3","top3","top1"]),
        },
        "H_trifecta_order_residual": {
            "deciles": combine_aggs(years, "H_trifecta_order_residual", "deciles", range(10)),
            "extremes": combine_aggs(years, "H_trifecta_order_residual", "extremes", ["bottom1","bottom3","top3","top1"]),
        },
    }
    full = {
        "status": "ODDS_CONTRADICTION_EXPLORATORY_ANALYSIS_2023_2025",
        "years_read": list(YEARS),
        "evaluation_year_2026_used": False,
        "odds_phase": "final",
        "important_limit": "Archived final odds only, not T-10 snapshots. Exploratory research, not a frozen betting strategy.",
        "definitions": {
            "market_probability": "Within each ticket pool, normalize inverse final odds: q_j=(1/odds_j)/sum(1/odds).",
            "D": "log(sum normalized trifecta probability over the six orders of a 3-rider set / normalized trio probability of that set). Positive means trifecta market rates the set more strongly than trio market.",
            "I": "For each rider, log(trifecta-derived top3 inclusion marginal / trio-derived top3 inclusion marginal).",
            "C": "log(actual normalized trio probability / maximum-entropy trio probability fitted only to all riders' trio top3 inclusion marginals). Negative means the combination is underbought relative to its members' individual popularity.",
            "H": "log(actual normalized trifecta probability / maximum-entropy order probability fitted only to each rider's 1st/2nd/3rd-place marginals). Negative means the exact order is underbought relative to position popularity.",
            "deciles": "Rank metric within each race, 0=lowest contradiction value, 9=highest.",
            "roi_pct_final_odds": "Exploratory gross ROI if 100 yen were staked on every cell in the bucket at archived final odds. Not a T-10 executable backtest.",
        },
        "years": years,
        "combined": combined,
        "stable_findings": stable_findings(years),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(full, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    compact = compact_from(full)
    OUT_COMPACT.write_text(json.dumps(compact, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(compact, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
