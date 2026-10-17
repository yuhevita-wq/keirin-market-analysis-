from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from pathlib import Path

DATA = Path("data/2024/s_class_f1_all_parts/2024_q1")
OUT = Path("results/mce_v1/2024_q1")
WARMUP_END = date(2024, 1, 14)
SIM_START = date(2024, 1, 15)
SIM_END = date(2024, 3, 31)

MIN_TRAIN_RACES = 100
SHRINK_TO_MARKET = 0.50
MIN_CONSERVATIVE_EV = 1.10
MIN_MODEL_MARKET_RATIO = 1.12
MAX_TICKETS = 3
STAKE_PER_TICKET = 100
L2 = 0.05
WARMUP_EPOCHS = 100
DAILY_EPOCHS = 8
LR = 0.05
BETA_CLIP = 2.0

FEATURE_NAMES = [
    "logp_center",
    "rank_center",
    "single_resid",
    "pair_resid",
    "min_car_center",
    "min_pair_center",
    "single_x_entropy",
    "pair_x_top1",
]

def read_csv(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))

def fnum(v, default=None):
    try:
        return float(str(v).replace(",", ""))
    except (TypeError, ValueError):
        return default

def inum(v, default=None):
    try:
        return int(float(str(v).replace(",", "")))
    except (TypeError, ValueError):
        return default

def parse_date(s: str) -> date:
    y, m, d = map(int, s.split("-"))
    return date(y, m, d)

def norm_combo(s: str) -> str:
    parts = [int(x) for x in str(s).replace("-", "=").split("=") if x.strip().isdigit()]
    return "=".join(map(str, sorted(parts)))

def quantile(xs, q):
    if not xs:
        return None
    ys = sorted(xs)
    if len(ys) == 1:
        return ys[0]
    pos = (len(ys)-1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return ys[lo]
    w = pos - lo
    return ys[lo] * (1-w) + ys[hi] * w

def clip(x, lo, hi):
    return max(lo, min(hi, x))

def softmax(scores):
    m = max(scores)
    ex = [math.exp(clip(s-m, -50, 50)) for s in scores]
    z = sum(ex)
    return [v/z for v in ex]

@dataclass
class RaceObs:
    race_id: str
    race_date: date
    track: str
    race_no: int
    race_type: str
    combos: list[str]
    odds: list[float]
    pm: list[float]
    features: list[list[float]]
    entropy_norm: float
    top1: float
    winners: dict[str, int]

def make_features(combos, odds):
    inv = [1.0/o for o in odds]
    z = sum(inv)
    pm = [v/z for v in inv]
    logs = [math.log(max(p, 1e-12)) for p in pm]
    mean_log = sum(logs)/len(logs)

    car_q = {c: 0.0 for c in range(1, 8)}
    pair_q = defaultdict(float)
    parsed = []
    for combo, p in zip(combos, pm):
        cs = tuple(map(int, combo.split("=")))
        parsed.append(cs)
        for c in cs:
            car_q[c] += p
        for i in range(3):
            for j in range(i+1, 3):
                pair_q[tuple(sorted((cs[i], cs[j])))] += p

    single_raw = []
    pair_raw = []
    min_car = []
    min_pair = []
    for cs in parsed:
        single_raw.append(max(1e-15, car_q[cs[0]] * car_q[cs[1]] * car_q[cs[2]]))
        pairs = [
            pair_q[tuple(sorted((cs[0], cs[1])))],
            pair_q[tuple(sorted((cs[0], cs[2])))],
            pair_q[tuple(sorted((cs[1], cs[2])))],
        ]
        pair_raw.append(max(1e-15, (pairs[0]*pairs[1]*pairs[2]) ** (1/3)))
        min_car.append(min(car_q[c] for c in cs))
        min_pair.append(min(pairs))

    zs = sum(single_raw)
    zp = sum(pair_raw)
    single_base = [v/zs for v in single_raw]
    pair_base = [v/zp for v in pair_raw]
    single_resid = [clip(math.log(max(pm[i],1e-12)/max(single_base[i],1e-12)), -2.5, 2.5) for i in range(len(pm))]
    pair_resid = [clip(math.log(max(pm[i],1e-12)/max(pair_base[i],1e-12)), -2.5, 2.5) for i in range(len(pm))]
    mean_mc = sum(min_car)/len(min_car)
    mean_mp = sum(min_pair)/len(min_pair)

    entropy = -sum(p*math.log(max(p,1e-12)) for p in pm)
    entropy_norm = entropy / math.log(len(pm))
    top1 = max(pm)
    ent_center = clip((entropy_norm - 0.80) * 5.0, -1.5, 1.5)
    top_center = clip((top1 - 0.10) * 5.0, -1.5, 1.5)

    order = sorted(range(len(odds)), key=lambda i: (odds[i], combos[i]))
    rank = [0]*len(odds)
    for r, idx in enumerate(order, 1):
        rank[idx] = r

    feats = []
    for i in range(len(pm)):
        logp_center = clip((logs[i]-mean_log)/2.0, -3.0, 3.0)
        rank_center = clip((18.0-rank[i])/17.0, -1.0, 1.0)
        mc = clip((min_car[i]-mean_mc)*5.0, -2.0, 2.0)
        mp = clip((min_pair[i]-mean_mp)*8.0, -2.0, 2.0)
        sr = single_resid[i]
        pr = pair_resid[i]
        feats.append([
            logp_center,
            rank_center,
            sr,
            pr,
            mc,
            mp,
            clip(sr*ent_center, -3.0, 3.0),
            clip(pr*top_center, -3.0, 3.0),
        ])
    return pm, feats, entropy_norm, top1

def model_probs(obs: RaceObs, beta):
    scores = []
    for p, x in zip(obs.pm, obs.features):
        adj = sum(b*v for b, v in zip(beta, x))
        scores.append(math.log(max(p,1e-12)) + adj)
    return softmax(scores)

def target_vector(obs: RaceObs):
    winners = [i for i,c in enumerate(obs.combos) if c in obs.winners]
    if not winners:
        return None
    y = [0.0]*len(obs.combos)
    w = 1.0/len(winners)
    for i in winners:
        y[i] = w
    return y

def fit(history, beta, epochs):
    beta = list(beta)
    for _ in range(epochs):
        grad = [0.0]*len(beta)
        used = 0
        for obs in history:
            y = target_vector(obs)
            if y is None:
                continue
            pred = model_probs(obs, beta)
            used += 1
            for i, x in enumerate(obs.features):
                d = pred[i]-y[i]
                for k in range(len(beta)):
                    grad[k] += d*x[k]
        if not used:
            break
        for k in range(len(beta)):
            g = grad[k]/used + L2*beta[k]
            beta[k] = clip(beta[k] - LR*g, -BETA_CLIP, BETA_CLIP)
    return beta

def summarize_bets(rows):
    stake = sum(r["stake_yen"] for r in rows)
    ret = sum(r["return_yen"] for r in rows)
    hit_races = sum(1 for r in rows if r["return_yen"] > 0)
    cum = 0
    peak = 0
    max_dd = 0
    loss_run = 0
    max_loss_run = 0
    for r in rows:
        cum += r["return_yen"] - r["stake_yen"]
        peak = max(peak, cum)
        max_dd = max(max_dd, peak-cum)
        if r["return_yen"] <= 0:
            loss_run += 1
            max_loss_run = max(max_loss_run, loss_run)
        else:
            loss_run = 0
    return {
        "bet_races": len(rows),
        "tickets": sum(r["ticket_count"] for r in rows),
        "hit_races": hit_races,
        "hit_rate": hit_races/len(rows) if rows else 0.0,
        "stake_yen": stake,
        "return_yen": ret,
        "profit_yen": ret-stake,
        "roi": ret/stake if stake else 0.0,
        "avg_tickets_per_bet_race": (sum(r["ticket_count"] for r in rows)/len(rows)) if rows else 0.0,
        "max_drawdown_yen": max_dd,
        "max_consecutive_losing_bet_races": max_loss_run,
    }

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    races = read_csv(DATA/"races.csv")
    odds_rows = read_csv(DATA/"trio_final_odds.csv")
    payouts = read_csv(DATA/"payouts.csv")

    race_meta = {}
    source_f1 = 0
    source_7car = 0
    for r in races:
        if r.get("meeting_grade") != "F1":
            continue
        source_f1 += 1
        if inum(r.get("entry_count"), 0) != 7:
            continue
        source_7car += 1
        race_meta[r["race_id"]] = r

    winners = defaultdict(dict)
    for p in payouts:
        if p.get("ticket_type") == "3連複" and p.get("status") == "paid" and p.get("combination"):
            rid = p["race_id"]
            combo = norm_combo(p["combination"])
            pay = inum(p.get("payout_yen"), 0) or 0
            winners[rid][combo] = pay

    odds_by = defaultdict(list)
    for o in odds_rows:
        rid = o.get("race_id")
        if rid not in race_meta:
            continue
        if o.get("ticket_type") != "3連複":
            continue
        combo = norm_combo(o.get("combination",""))
        odd = fnum(o.get("odds"))
        if len(combo.split("=")) != 3 or odd is None or odd <= 0:
            continue
        odds_by[rid].append((combo, odd))

    observations = []
    incomplete_odds = 0
    no_payout = 0
    for rid, r in race_meta.items():
        rows = odds_by.get(rid, [])
        bycombo = {}
        for c,o in rows:
            bycombo[c] = o
        if len(bycombo) != 35:
            incomplete_odds += 1
            continue
        if not winners.get(rid):
            no_payout += 1
            continue
        combos = sorted(bycombo, key=lambda c: tuple(map(int,c.split("="))))
        oddsv = [bycombo[c] for c in combos]
        pm, feats, ent, top1 = make_features(combos, oddsv)
        observations.append(RaceObs(
            race_id=rid,
            race_date=parse_date(r["race_date"]),
            track=r["track"],
            race_no=inum(r["race_no"],0),
            race_type=r["race_type"],
            combos=combos,
            odds=oddsv,
            pm=pm,
            features=feats,
            entropy_norm=ent,
            top1=top1,
            winners=winners[rid],
        ))

    observations.sort(key=lambda o:(o.race_date,o.track,o.race_no,o.race_id))
    warmup = [o for o in observations if o.race_date <= WARMUP_END]
    simulated = [o for o in observations if SIM_START <= o.race_date <= SIM_END]

    beta = [0.0]*len(FEATURE_NAMES)
    beta = fit(warmup, beta, WARMUP_EPOCHS)
    history = list(warmup)

    by_date = defaultdict(list)
    for o in simulated:
        by_date[o.race_date].append(o)

    bet_rows = []
    baseline_rows = []
    all_pred_rows = []
    ood_skips = 0
    no_candidate = 0
    model_logloss = 0.0
    market_logloss = 0.0
    scored_races = 0

    for d in sorted(by_date):
        day = sorted(by_date[d], key=lambda o:(o.track,o.race_no,o.race_id))
        ents = [h.entropy_norm for h in history]
        tops = [h.top1 for h in history]
        ent_lo, ent_hi = quantile(ents,0.01), quantile(ents,0.99)
        top_lo, top_hi = quantile(tops,0.01), quantile(tops,0.99)

        for obs in day:
            pred = model_probs(obs, beta)
            y = target_vector(obs)
            if y is not None:
                market_logloss += -sum(y[i]*math.log(max(obs.pm[i],1e-12)) for i in range(len(y)))
                model_logloss += -sum(y[i]*math.log(max(pred[i],1e-12)) for i in range(len(y)))
                scored_races += 1

            is_ood = (
                len(history) < MIN_TRAIN_RACES
                or obs.entropy_norm < ent_lo or obs.entropy_norm > ent_hi
                or obs.top1 < top_lo or obs.top1 > top_hi
            )
            candidates = []
            for i,(combo,odd,pm_i,pm_model) in enumerate(zip(obs.combos,obs.odds,obs.pm,pred)):
                ratio = pm_model/max(pm_i,1e-12)
                p_cons = SHRINK_TO_MARKET*pm_i + (1.0-SHRINK_TO_MARKET)*pm_model
                ev_cons = p_cons*odd
                candidates.append((ev_cons, ratio, combo, odd, pm_i, pm_model, p_cons, i))
            candidates.sort(reverse=True)

            if is_ood:
                ood_skips += 1
                selected = []
                skip_reason = "OOD"
            else:
                selected = [x for x in candidates if x[0] >= MIN_CONSERVATIVE_EV and x[1] >= MIN_MODEL_MARKET_RATIO][:MAX_TICKETS]
                skip_reason = "" if selected else "NO_EDGE"
                if not selected:
                    no_candidate += 1

            all_pred_rows.append({
                "race_id": obs.race_id,
                "race_date": obs.race_date.isoformat(),
                "track": obs.track,
                "race_no": obs.race_no,
                "race_type": obs.race_type,
                "entropy_norm": obs.entropy_norm,
                "top1_market_prob": obs.top1,
                "ood": int(is_ood),
                "bet": int(bool(selected)),
                "skip_reason": skip_reason,
                "selected": "|".join(x[2] for x in selected),
                "selected_conservative_ev": "|".join(f"{x[0]:.6f}" for x in selected),
                "selected_model_market_ratio": "|".join(f"{x[1]:.6f}" for x in selected),
            })

            if selected:
                chosen = {x[2]: x for x in selected}
                return_yen = sum(obs.winners.get(c,0) for c in chosen)
                row = {
                    "race_id": obs.race_id,
                    "race_date": obs.race_date.isoformat(),
                    "track": obs.track,
                    "race_no": obs.race_no,
                    "race_type": obs.race_type,
                    "ticket_count": len(selected),
                    "tickets": [x[2] for x in selected],
                    "stake_yen": STAKE_PER_TICKET*len(selected),
                    "return_yen": return_yen,
                    "profit_yen": return_yen-STAKE_PER_TICKET*len(selected),
                    "winner_combos": list(obs.winners.keys()),
                    "winner_payouts": obs.winners,
                    "selected_detail": [
                        {
                            "combo":x[2], "odds":x[3], "market_p":x[4],
                            "model_p":x[5], "conservative_p":x[6],
                            "conservative_ev":x[0], "model_market_ratio":x[1],
                        } for x in selected
                    ],
                }
                bet_rows.append(row)

                k = len(selected)
                fav_idx = sorted(range(len(obs.odds)), key=lambda i:(obs.odds[i],obs.combos[i]))[:k]
                favs = [obs.combos[i] for i in fav_idx]
                base_ret = sum(obs.winners.get(c,0) for c in favs)
                baseline_rows.append({
                    "race_id":obs.race_id,
                    "race_date":obs.race_date.isoformat(),
                    "track":obs.track,
                    "race_no":obs.race_no,
                    "race_type":obs.race_type,
                    "ticket_count":k,
                    "tickets":favs,
                    "stake_yen":STAKE_PER_TICKET*k,
                    "return_yen":base_ret,
                    "profit_yen":base_ret-STAKE_PER_TICKET*k,
                })

        history.extend(day)
        beta = fit(history, beta, DAILY_EPOCHS)

    bet_rows.sort(key=lambda r:(r["race_date"],r["track"],r["race_no"],r["race_id"]))
    baseline_rows.sort(key=lambda r:(r["race_date"],r["track"],r["race_no"],r["race_id"]))

    overall = summarize_bets(bet_rows)
    baseline = summarize_bets(baseline_rows)

    by_month = {}
    for m in ["2024-01","2024-02","2024-03"]:
        by_month[m] = summarize_bets([r for r in bet_rows if r["race_date"].startswith(m)])

    by_type = {}
    for rt in sorted({r["race_type"] for r in bet_rows}):
        by_type[rt] = summarize_bets([r for r in bet_rows if r["race_type"] == rt])

    hit_returns = sorted([r["return_yen"] for r in bet_rows if r["return_yen"]>0], reverse=True)
    top1_removed_return = overall["return_yen"] - (hit_returns[0] if hit_returns else 0)
    top3_removed_return = overall["return_yen"] - sum(hit_returns[:3])
    jackpot = {
        "largest_hit_return_yen": hit_returns[0] if hit_returns else 0,
        "top3_hit_returns_yen": hit_returns[:3],
        "roi_without_largest_hit": top1_removed_return/overall["stake_yen"] if overall["stake_yen"] else 0.0,
        "roi_without_top3_hits": top3_removed_return/overall["stake_yen"] if overall["stake_yen"] else 0.0,
        "top3_share_of_total_return": (sum(hit_returns[:3])/overall["return_yen"]) if overall["return_yen"] else 0.0,
    }

    summary = {
        "experiment":"MCE v1 frozen reality check - 2024 Q1 F1 S-class 7-car",
        "scope":{
            "source":"data/2024/s_class_f1_all_parts/2024_q1",
            "meeting_grade":"F1",
            "class":"S-class all race types",
            "entry_count":7,
            "warmup":"2024-01-01..2024-01-14",
            "simulation":"2024-01-15..2024-03-31",
            "source_f1_races":source_f1,
            "source_7car_races":source_7car,
            "complete_observations":len(observations),
            "warmup_races":len(warmup),
            "simulated_complete_races":len(simulated),
            "incomplete_35_odds_races":incomplete_odds,
            "missing_trio_payout_races":no_payout,
        },
        "frozen_rules":{
            "features":FEATURE_NAMES,
            "market_probability":"normalize inverse odds across all 35 trios",
            "model":"multinomial softmax correction on top of log market probability",
            "training":"expanding walk-forward; predict whole day before adding that day's outcomes",
            "shrink_to_market":SHRINK_TO_MARKET,
            "minimum_conservative_ev":MIN_CONSERVATIVE_EV,
            "minimum_model_market_ratio":MIN_MODEL_MARKET_RATIO,
            "max_tickets_per_race":MAX_TICKETS,
            "stake_yen_per_ticket":STAKE_PER_TICKET,
            "ood":"skip if entropy or top1 market probability outside prior-history 1st..99th percentile",
            "warmup_epochs":WARMUP_EPOCHS,
            "daily_epochs":DAILY_EPOCHS,
            "learning_rate":LR,
            "l2":L2,
            "final_odds_warning":"Historical final odds, not T-10. Research simulation only.",
        },
        "prediction_quality":{
            "scored_races":scored_races,
            "market_logloss":market_logloss/scored_races if scored_races else None,
            "mce_model_logloss":model_logloss/scored_races if scored_races else None,
            "logloss_improvement_vs_market":(market_logloss-model_logloss)/scored_races if scored_races else None,
        },
        "selection":{
            "ood_skips":ood_skips,
            "no_edge_skips":no_candidate,
            **overall,
        },
        "equal_ticket_market_favorite_baseline":baseline,
        "by_month":by_month,
        "by_race_type":by_type,
        "jackpot_dependency":jackpot,
        "final_beta":{name:beta[i] for i,name in enumerate(FEATURE_NAMES)},
        "leakage_guard":"For every simulation date, all predictions are produced before that date is appended to training history.",
    }

    (OUT/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    (OUT/"bets.json").write_text(json.dumps(bet_rows,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")

    with (OUT/"predictions.csv").open("w",encoding="utf-8-sig",newline="") as f:
        fields = list(all_pred_rows[0].keys()) if all_pred_rows else ["race_id"]
        w=csv.DictWriter(f,fieldnames=fields)
        w.writeheader()
        w.writerows(all_pred_rows)

    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__ == "__main__":
    main()
