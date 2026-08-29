from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

YEAR = 2023
DATA = Path(f"data/{YEAR}/s_class_yosen")
OUT = Path("data/audits/market_correction_sim_2023.json")


def read_csv(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def norm_inv_odds(rows):
    vals = {}
    s = 0.0
    for r in rows:
        if r.get("odds_status") != "available":
            continue
        try:
            o = float(r["odds"])
        except Exception:
            continue
        if o <= 0:
            continue
        q = 1.0 / o
        vals[r["combination"]] = q
        s += q
    if s <= 0:
        return {}
    return {k:v/s for k,v in vals.items()}


def result_top3_sets():
    by = defaultdict(list)
    for r in read_csv(DATA / "results.csv"):
        try:
            fp = int(r.get("finish_position") or 999)
            car = int(r.get("car_no") or 0)
        except Exception:
            continue
        if 1 <= fp <= 3 and car > 0:
            by[r["race_id"]].append((fp,car))
    out = {}
    for rid, xs in by.items():
        xs = sorted(xs)
        if len(xs) == 3 and [x[0] for x in xs] == [1,2,3]:
            out[rid] = tuple(sorted(x[1] for x in xs))
    return out


def load_markets():
    tri = defaultdict(list)
    trio = defaultdict(list)
    for r in read_csv(DATA / "trifecta_final_odds.csv"):
        tri[r["race_id"]].append(r)
    for r in read_csv(DATA / "trio_final_odds.csv"):
        trio[r["race_id"]].append(r)
    return tri,trio


def build_race_probs(tri_rows, trio_rows):
    pt = norm_inv_odds(tri_rows)
    pp = norm_inv_odds(trio_rows)
    if not pt or not pp:
        return None
    pset = defaultdict(float)
    for comb,p in pt.items():
        try:
            cars = tuple(sorted(int(x) for x in comb.split("-")))
        except Exception:
            continue
        if len(cars)==3 and len(set(cars))==3:
            pset[cars] += p
    trio_probs = {}
    trio_odds = {}
    for r in trio_rows:
        if r.get("odds_status") != "available":
            continue
        try:
            cars = tuple(sorted(int(x) for x in r["combination"].split("=")))
            o = float(r["odds"])
        except Exception:
            continue
        if r["combination"] in pp:
            trio_probs[cars] = pp[r["combination"]]
            trio_odds[cars] = o
    common = sorted(set(pset) & set(trio_probs))
    if len(common) < 10:
        return None
    return {c:(trio_probs[c], pset[c], trio_odds[c]) for c in common}


def main():
    tri,trio = load_markets()
    winners = result_top3_sets()
    races = {}
    for rid in sorted(set(tri)&set(trio)&set(winners)):
        got = build_race_probs(tri[rid], trio[rid])
        if got is not None and winners[rid] in got:
            races[rid] = got

    # Development-year alpha fit by logloss. Predeclared grid, no ROI used.
    alpha_grid = [round(i/20,2) for i in range(21)]
    alpha_scores = []
    for a in alpha_grid:
        loss = 0.0
        n = 0
        for rid, probs in races.items():
            win = winners[rid]
            p_trio,p_tri,_ = probs[win]
            p = (1-a)*p_trio + a*p_tri
            loss += -math.log(max(p,1e-15))
            n += 1
        alpha_scores.append({"alpha":a,"mean_logloss":loss/n if n else None})
    best = min(alpha_scores, key=lambda x:x["mean_logloss"])
    alpha = best["alpha"]

    thresholds = [1.00,1.05,1.10,1.15,1.20]
    stats = {}
    for t in thresholds:
        bets=wins=0
        stake=payout=0.0
        race_bets=defaultdict(int)
        race_wins=defaultdict(int)
        bet_odds=[]
        max_consecutive_losing_bets=cur_loss=0
        for rid in sorted(races):
            win = winners[rid]
            # Stable order for reproducibility
            for cars,(p_trio,p_tri,odds) in sorted(races[rid].items()):
                pstar=(1-alpha)*p_trio+alpha*p_tri
                ev=pstar*odds
                if ev + 1e-12 < t:
                    continue
                bets += 1
                stake += 100
                race_bets[rid] += 1
                bet_odds.append(odds)
                if cars == win:
                    wins += 1
                    payout += odds*100
                    race_wins[rid] += 1
                    cur_loss=0
                else:
                    cur_loss += 1
                    max_consecutive_losing_bets=max(max_consecutive_losing_bets,cur_loss)
        purchased_races=len(race_bets)
        hit_races=sum(1 for rid in race_bets if race_wins.get(rid,0)>0)
        stats[f"{t:.2f}"]={
            "threshold_pstar_times_trio_odds":t,
            "bets":bets,
            "wins":wins,
            "hit_rate_per_bet_pct":wins/bets*100 if bets else 0,
            "purchased_races":purchased_races,
            "race_purchase_rate_pct":purchased_races/len(races)*100 if races else 0,
            "hit_races":hit_races,
            "race_hit_rate_pct":hit_races/purchased_races*100 if purchased_races else 0,
            "avg_bets_per_purchased_race":bets/purchased_races if purchased_races else 0,
            "stake_yen":round(stake),
            "payout_yen":round(payout),
            "profit_yen":round(payout-stake),
            "roi_pct":payout/stake*100 if stake else 0,
            "median_bet_odds": sorted(bet_odds)[len(bet_odds)//2] if bet_odds else None,
            "max_consecutive_losing_bets":max_consecutive_losing_bets,
        }

    out={
        "status":"MARKET_CORRECTION_2023_DEVELOPMENT_SIMULATION",
        "year":2023,
        "years_read":[2023],
        "evaluation_year_2024_used":False,
        "evaluation_year_2025_used":False,
        "evaluation_year_2026_used":False,
        "odds_phase":"final",
        "important_limit":"Alpha is fitted and evaluated on the same 2023 outcomes. This is an in-sample development simulation, not OOS evidence and not T-10 executable data.",
        "eligible_complete_races":len(races),
        "formula":"P*=(1-alpha)*P_trio + alpha*P_trifecta_set; buy when P* * trio_final_odds >= threshold",
        "alpha_selection":"Choose alpha on grid 0.00..1.00 step 0.05 by lowest 2023 mean logloss only; ROI is not used to choose alpha.",
        "alpha_grid":alpha_scores,
        "selected_alpha":alpha,
        "selected_alpha_mean_logloss":best["mean_logloss"],
        "simulations":stats,
        "primary_reference_threshold":"1.10 is shown as the primary safety-margin reference because it was the previously stated example, but thresholds were predeclared before this run and no threshold is frozen from 2023 ROI.",
    }
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(out,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(out,ensure_ascii=False,indent=2))

if __name__ == "__main__":
    main()
