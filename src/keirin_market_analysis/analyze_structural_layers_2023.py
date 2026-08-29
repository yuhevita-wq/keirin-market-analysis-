from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

YEAR = 2023
DATA = Path(f"data/{YEAR}/s_class_yosen")
OUT = Path("data/audits/structural_layers_2023.json")


def read_csv(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def norm_inv_odds(rows, sep):
    vals = {}
    total = 0.0
    for r in rows:
        if r.get("odds_status") != "available":
            continue
        try:
            o = float(r["odds"])
        except Exception:
            continue
        if o <= 1.0:
            continue
        comb = r.get("combination", "")
        if len(comb.split(sep)) != 3:
            continue
        q = 1.0 / o
        vals[comb] = q
        total += q
    return {k: v / total for k, v in vals.items()} if total > 0 else {}


def top3_results():
    by = defaultdict(list)
    for r in read_csv(DATA / "results.csv"):
        try:
            fp = int(r.get("finish_position") or 999)
            car = int(r.get("car_no") or 0)
        except Exception:
            continue
        if 1 <= fp <= 3 and car > 0:
            by[r["race_id"]].append((fp, car))
    out = {}
    for rid, xs in by.items():
        xs = sorted(xs)
        if len(xs) == 3 and [x[0] for x in xs] == [1, 2, 3]:
            out[rid] = tuple(sorted(x[1] for x in xs))
    return out


def load_markets():
    tri, trio = defaultdict(list), defaultdict(list)
    for r in read_csv(DATA / "trifecta_final_odds.csv"):
        tri[r["race_id"]].append(r)
    for r in read_csv(DATA / "trio_final_odds.csv"):
        trio[r["race_id"]].append(r)
    return tri, trio


def effective_order_count(order_probs):
    s = sum(order_probs)
    if s <= 0:
        return 0.0, 0
    shares = [p / s for p in order_probs if p > 0]
    entropy = -sum(x * math.log(x) for x in shares)
    eff = math.exp(entropy)
    support10 = sum(1 for x in shares if x >= 0.10)
    return eff, support10


def build_race(tri_rows, trio_rows):
    p_order = norm_inv_odds(tri_rows, "-")
    p_trio_raw = norm_inv_odds(trio_rows, "=")
    if not p_order or not p_trio_raw:
        return None
    set_orders = defaultdict(list)
    for comb, p in p_order.items():
        try:
            cars = tuple(sorted(int(x) for x in comb.split("-")))
        except Exception:
            continue
        if len(cars) == 3 and len(set(cars)) == 3:
            set_orders[cars].append(p)
    trio_probs, trio_odds = {}, {}
    for r in trio_rows:
        if r.get("odds_status") != "available":
            continue
        try:
            cars = tuple(sorted(int(x) for x in r["combination"].split("=")))
            o = float(r["odds"])
        except Exception:
            continue
        if o <= 1.0 or r["combination"] not in p_trio_raw:
            continue
        trio_probs[cars] = p_trio_raw[r["combination"]]
        trio_odds[cars] = o
    common = sorted(set(set_orders) & set(trio_probs))
    if len(common) < 10:
        return None
    out = {}
    for c in common:
        ptri = sum(set_orders[c])
        ptrio = trio_probs[c]
        eff, support10 = effective_order_count(set_orders[c])
        out[c] = {
            "p_trio": ptrio,
            "p_tri_set": ptri,
            "delta": ptri - ptrio,
            "ratio": ptri / ptrio if ptrio > 0 else None,
            "consensus": math.sqrt(max(ptri, 0) * max(ptrio, 0)),
            "odds": trio_odds[c],
            "effective_orders": eff,
            "support10": support10,
        }
    return out


def blank():
    return {"tickets": 0, "wins": 0, "stake_yen": 0.0, "payout_yen": 0.0, "odds": [], "winning_odds": []}


def add(st, x, win):
    st["tickets"] += 1
    st["stake_yen"] += 100
    st["odds"].append(x["odds"])
    if win:
        st["wins"] += 1
        st["payout_yen"] += x["odds"] * 100
        st["winning_odds"].append(x["odds"])


def median(xs):
    if not xs:
        return None
    ys = sorted(xs)
    n = len(ys)
    return ys[n//2] if n % 2 else (ys[n//2 - 1] + ys[n//2]) / 2


def finish(st):
    stake = st["stake_yen"]
    return {
        "tickets": st["tickets"],
        "wins": st["wins"],
        "ticket_hit_rate_pct": st["wins"] / st["tickets"] * 100 if st["tickets"] else 0,
        "stake_yen": round(stake),
        "payout_yen": round(st["payout_yen"]),
        "profit_yen": round(st["payout_yen"] - stake),
        "roi_pct": st["payout_yen"] / stake * 100 if stake else 0,
        "median_odds": median(st["odds"]),
        "median_winning_odds": median(st["winning_odds"]),
    }


def main():
    tri, trio = load_markets()
    winners = top3_results()
    stats = {k: blank() for k in [
        "main1",
        "middle_overlap2_all",
        "middle_overlap2_tri3_stronger",
        "middle_overlap2_supported_tri3_stronger",
        "alternative_overlap0or1_all",
    ]}
    eligible = 0
    middle_winner_races = 0
    middle_count_per_race = []

    for rid in sorted(set(tri) & set(trio) & set(winners)):
        market = build_race(tri[rid], trio[rid])
        if market is None or winners[rid] not in market:
            continue
        eligible += 1
        main1 = max(market, key=lambda c: (market[c]["consensus"], tuple(-x for x in c)))
        winner = winners[rid]
        if winner == main1:
            add(stats["main1"], market[main1], True)
        else:
            add(stats["main1"], market[main1], False)

        middle = []
        for c, x in market.items():
            if c == main1:
                continue
            overlap = len(set(c) & set(main1))
            iswin = c == winner
            if overlap == 2:
                middle.append(c)
                add(stats["middle_overlap2_all"], x, iswin)
                if x["delta"] > 0:
                    add(stats["middle_overlap2_tri3_stronger"], x, iswin)
                    if x["effective_orders"] >= 3.0 and x["support10"] >= 3:
                        add(stats["middle_overlap2_supported_tri3_stronger"], x, iswin)
            elif overlap <= 1:
                add(stats["alternative_overlap0or1_all"], x, iswin)
        middle_count_per_race.append(len(middle))
        if winner in middle:
            middle_winner_races += 1

    out = {
        "status": "STRUCTURAL_LAYER_AUDIT_2023",
        "year": YEAR,
        "years_read": [2023],
        "odds_phase": "final",
        "important_limit": "Exploratory 2023 in-sample structural audit. Equal 100-yen stake per ticket; not the joint no-trigami portfolio and not OOS.",
        "definition": {
            "main1": "Highest consensus sqrt(P_trio * P_trifecta_set).",
            "middle": "Any non-main 3-car set sharing exactly 2 cars with main1, i.e. a one-car replacement of the main scenario. Popularity rank and odds band are not used.",
            "middle_tri3_stronger": "Middle set with P_trifecta_set > P_trio.",
            "middle_supported_tri3_stronger": "Middle set with P_trifecta_set > P_trio plus effective six-order count >=3 and at least three order shares >=10%.",
            "alternative": "Any set sharing at most 1 car with main1."
        },
        "eligible_complete_races": eligible,
        "middle_winner_races": middle_winner_races,
        "middle_winner_race_rate_pct": middle_winner_races / eligible * 100 if eligible else 0,
        "avg_middle_sets_per_race": sum(middle_count_per_race) / len(middle_count_per_race) if middle_count_per_race else 0,
        "layers": {k: finish(v) for k, v in stats.items()},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
