from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np
import pandas as pd

from .prediction_models import distribution_metrics


@dataclass(frozen=True)
class TicketPolicy:
    """A price-blind ticket policy. It never reads final odds."""

    cumulative_mass: float = 0.48
    max_tickets: int = 12
    max_normalized_entropy: float = 0.88
    min_top10_mass: float = 0.36
    max_model_tv_distance: float = 0.60
    unit_yen: int = 100


def model_tv_distance(stat: pd.DataFrame, similarity: pd.DataFrame) -> float:
    if similarity.empty:
        return 1.0
    a = stat[["combo", "probability"]].rename(columns={"probability": "a"})
    b = similarity[["combo", "probability"]].rename(columns={"probability": "b"})
    j = a.merge(b, on="combo", how="outer").fillna(0.0)
    if j["a"].sum() > 0:
        j["a"] /= j["a"].sum()
    if j["b"].sum() > 0:
        j["b"] /= j["b"].sum()
    return float(0.5 * np.abs(j["a"] - j["b"]).sum())


def select_tickets(
    blended: pd.DataFrame,
    stat: pd.DataFrame,
    similarity: pd.DataFrame,
    policy: TicketPolicy,
) -> dict:
    metrics = distribution_metrics(blended)
    tv = model_tv_distance(stat, similarity)
    reasons: list[str] = []
    if metrics["normalized_entropy"] > policy.max_normalized_entropy:
        reasons.append("entropy")
    if metrics["top10_mass"] < policy.min_top10_mass:
        reasons.append("diffuse_top10")
    if tv > policy.max_model_tv_distance:
        reasons.append("model_disagreement")
    if reasons:
        return {
            "buy": False,
            "tickets": [],
            "stake_yen": 0,
            "metrics": {**metrics, "model_tv_distance": tv},
            "skip_reasons": reasons,
        }

    ordered = blended.sort_values("probability", ascending=False).reset_index(drop=True)
    tickets: list[str] = []
    mass = 0.0
    for r in ordered.itertuples(index=False):
        if len(tickets) >= policy.max_tickets:
            break
        tickets.append(str(r.combo))
        mass += float(r.probability)
        if mass >= policy.cumulative_mass:
            break
    if not tickets:
        return {
            "buy": False,
            "tickets": [],
            "stake_yen": 0,
            "metrics": {**metrics, "model_tv_distance": tv},
            "skip_reasons": ["no_ticket"],
        }
    return {
        "buy": True,
        "tickets": tickets,
        "covered_probability_mass": mass,
        "stake_yen": len(tickets) * policy.unit_yen,
        "metrics": {**metrics, "model_tv_distance": tv},
        "skip_reasons": [],
    }


def trifecta_payout_map(payouts: pd.DataFrame) -> dict[str, tuple[str, int]]:
    p = payouts.copy()
    p = p[(p["ticket_type"] == "3連単") & (p["status"] == "paid") & p["combination"].notna()].copy()
    p["payout_yen"] = pd.to_numeric(p["payout_yen"], errors="coerce").fillna(0).astype(int)
    out: dict[str, tuple[str, int]] = {}
    for r in p.itertuples(index=False):
        combo = str(r.combination).strip().replace("-", "")
        if len(combo) == 3:
            out[str(r.race_id)] = (combo, int(r.payout_yen))
    return out


def evaluate_decisions(decisions: list[dict], payouts: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    paid = trifecta_payout_map(payouts)
    rows: list[dict] = []
    for d in decisions:
        rid = str(d["race_id"])
        if rid not in paid:
            continue
        actual, payout_yen = paid[rid]
        buy = bool(d["buy"])
        tickets = [str(x).replace("-", "") for x in d.get("tickets", [])]
        stake = int(d.get("stake_yen", 0)) if buy else 0
        hit = int(buy and actual in tickets)
        ret = payout_yen if hit else 0
        rows.append({
            "race_id": rid,
            "race_date": d.get("race_date"),
            "track": d.get("track"),
            "race_no": d.get("race_no"),
            "race_type": d.get("race_type"),
            "buy": buy,
            "bet_count": len(tickets),
            "stake_yen": stake,
            "actual_combo": actual,
            "hit": hit,
            "return_yen": ret,
            "profit_yen": ret - stake,
            **{f"metric_{k}": v for k, v in d.get("metrics", {}).items()},
        })
    df = pd.DataFrame(rows)
    bought = df[df["buy"]] if not df.empty else df
    stake = int(bought["stake_yen"].sum()) if not bought.empty else 0
    ret = int(bought["return_yen"].sum()) if not bought.empty else 0
    peak = 0
    equity = 0
    max_dd = 0
    max_losing_streak = 0
    streak = 0
    for r in bought.sort_values([c for c in ["race_date", "track", "race_no"] if c in bought.columns]).itertuples(index=False):
        equity += int(r.profit_yen)
        peak = max(peak, equity)
        max_dd = max(max_dd, peak - equity)
        if int(r.hit):
            streak = 0
        else:
            streak += 1
            max_losing_streak = max(max_losing_streak, streak)
    summary = {
        "evaluated_races": int(len(df)),
        "bought_races": int(len(bought)),
        "purchase_rate": float(len(bought) / len(df)) if len(df) else 0.0,
        "hit_races": int(bought["hit"].sum()) if not bought.empty else 0,
        "hit_rate": float(bought["hit"].mean()) if not bought.empty else 0.0,
        "stake_yen": stake,
        "return_yen": ret,
        "profit_yen": ret - stake,
        "roi": float(ret / stake) if stake else 0.0,
        "max_drawdown_yen": int(max_dd),
        "max_losing_streak": int(max_losing_streak),
        "average_bet_count": float(bought["bet_count"].mean()) if not bought.empty else 0.0,
    }
    return df, summary


def policy_grid() -> list[TicketPolicy]:
    """Small pre-declared grid for 2024-only development; never expanded after 2025 is opened."""
    return [
        TicketPolicy(cumulative_mass=mass, max_tickets=max_t, max_normalized_entropy=ent, min_top10_mass=top10, max_model_tv_distance=tv)
        for mass, max_t, ent, top10, tv in product(
            (0.42, 0.48, 0.54),
            (8, 12, 16),
            (0.82, 0.88, 0.94),
            (0.30, 0.36, 0.42),
            (0.45, 0.60, 0.75),
        )
    ]


def robust_policy_choice(q3_scores: pd.DataFrame, q4_scores: pd.DataFrame, min_bought_races: int = 60) -> pd.Series:
    """Choose by worst-quarter ROI, then sample size, then lower drawdown. No 2025 information allowed."""
    keys = ["policy_id", "bought_races", "roi", "max_drawdown_yen"]
    a = q3_scores[keys].rename(columns={
        "bought_races": "q3_bought", "roi": "q3_roi", "max_drawdown_yen": "q3_dd"
    })
    b = q4_scores[keys].rename(columns={
        "bought_races": "q4_bought", "roi": "q4_roi", "max_drawdown_yen": "q4_dd"
    })
    j = a.merge(b, on="policy_id", how="inner")
    j = j[(j["q3_bought"] >= min_bought_races) & (j["q4_bought"] >= min_bought_races)].copy()
    if j.empty:
        raise ValueError("No policy has enough bought races in both Q3 and Q4")
    j["worst_roi"] = j[["q3_roi", "q4_roi"]].min(axis=1)
    j["total_bought"] = j["q3_bought"] + j["q4_bought"]
    j["worst_dd"] = j[["q3_dd", "q4_dd"]].max(axis=1)
    j = j.sort_values(["worst_roi", "total_bought", "worst_dd"], ascending=[False, False, True])
    return j.iloc[0]
