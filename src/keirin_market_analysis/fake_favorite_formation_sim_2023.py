from __future__ import annotations

import csv
import itertools
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "2023" / "s_class_yosen"
AUDITS = ROOT / "data" / "audits"
GATE_JSON = AUDITS / "fake_favorite_true_middle_2023_2024.json"
OUT_JSON = AUDITS / "fake_favorite_formation_sim_2023.json"
OUT_CSV = AUDITS / "fake_favorite_formation_sim_2023_decisions.csv"

FAV_CONSENSUS_MAX = 0.1884985310418076
SECOND_TO_FAV_ODDS_RATIO_MAX = 1.2058823529411764
STAKE_PER_TICKET = 100


def norm_combo(*xs: int) -> tuple[int, ...]:
    return tuple(sorted(int(x) for x in xs))


def parse_combo(value: str) -> tuple[int, ...]:
    s = value.strip().replace("=", "-").replace(",", "-")
    return tuple(sorted(int(x) for x in s.split("-") if x.strip()))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def pick_col(fieldnames, candidates):
    lower = {x.lower(): x for x in fieldnames}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    return None


def load_audited_gate() -> dict[str, dict]:
    payload = json.loads(GATE_JSON.read_text(encoding="utf-8"))
    rows = payload["years"]["2023"]["races"]
    gate = {str(r["race_id"]): r for r in rows}
    if len(gate) != 208:
        raise RuntimeError(f"Expected audited 2023 fake-favorite gate of 208 races, got {len(gate)}")
    return gate


def load_trio_odds() -> dict[str, dict[tuple[int, int, int], float]]:
    rows = read_csv(DATA / "trio_final_odds.csv")
    if not rows:
        raise RuntimeError("trio_final_odds.csv is empty")
    f = rows[0].keys()
    rid = pick_col(f, ["race_id"])
    odds = pick_col(f, ["odds", "trio_odds", "final_odds"])
    combo = pick_col(f, ["combo", "combination", "numbers", "selection"])
    c1, c2, c3 = (pick_col(f, [x]) for x in ["first", "second", "third"])
    if rid is None or odds is None:
        raise RuntimeError(f"Unsupported trio columns: {list(f)}")
    out = defaultdict(dict)
    for r in rows:
        if combo:
            key = parse_combo(r[combo])
        elif c1 and c2 and c3:
            key = norm_combo(r[c1], r[c2], r[c3])
        else:
            numbered = [pick_col(f, [x]) for x in ["n1", "n2", "n3"]]
            if not all(numbered):
                raise RuntimeError(f"Cannot identify trio combination columns: {list(f)}")
            key = norm_combo(*(r[x] for x in numbered))
        out[r[rid]][key] = float(r[odds])
    return out


def load_trifecta_odds() -> dict[str, dict[tuple[int, int, int], float]]:
    rows = read_csv(DATA / "trifecta_final_odds.csv")
    if not rows:
        raise RuntimeError("trifecta_final_odds.csv is empty")
    f = rows[0].keys()
    rid = pick_col(f, ["race_id"])
    odds = pick_col(f, ["odds", "trifecta_odds", "final_odds"])
    combo = pick_col(f, ["combo", "combination", "numbers", "selection"])
    c1, c2, c3 = (pick_col(f, [x]) for x in ["first", "second", "third"])
    if rid is None or odds is None:
        raise RuntimeError(f"Unsupported trifecta columns: {list(f)}")
    out = defaultdict(dict)
    for r in rows:
        if combo:
            raw = r[combo].strip().replace("=", "-").replace(",", "-")
            key = tuple(int(x) for x in raw.split("-") if x.strip())
        elif c1 and c2 and c3:
            key = (int(r[c1]), int(r[c2]), int(r[c3]))
        else:
            numbered = [pick_col(f, [x]) for x in ["n1", "n2", "n3"]]
            if not all(numbered):
                raise RuntimeError(f"Cannot identify trifecta columns: {list(f)}")
            key = tuple(int(r[x]) for x in numbered)
        out[r[rid]][key] = float(r[odds])
    return out


def load_results() -> dict[str, tuple[int, int, int]]:
    rows = read_csv(DATA / "results.csv")
    if not rows:
        raise RuntimeError("results.csv is empty")
    f = rows[0].keys()
    rid = pick_col(f, ["race_id"])
    rank = pick_col(f, ["finish_position", "finish", "rank", "placing", "arrival_order", "着順"])
    number = pick_col(f, ["number", "car_no", "bike_no", "frame_no", "車番"])
    if rid and rank and number:
        grouped = defaultdict(list)
        for r in rows:
            try:
                grouped[r[rid]].append((int(float(r[rank])), int(float(r[number]))))
            except (ValueError, TypeError):
                continue
        return {k: tuple(x[1] for x in sorted(v)[:3]) for k, v in grouped.items() if len(v) >= 3}
    first, second, third = (pick_col(f, [x]) for x in ["first", "second", "third"])
    if rid and first and second and third:
        return {r[rid]: (int(r[first]), int(r[second]), int(r[third])) for r in rows}
    raise RuntimeError(f"Unsupported results columns: {list(f)}")


def implied_distribution(odds_map):
    inv = {k: 1.0 / v for k, v in odds_map.items() if v and v > 0}
    z = sum(inv.values())
    return {k: v / z for k, v in inv.items()} if z else {}


def trifecta_set_distribution(tf_odds):
    order_p = implied_distribution(tf_odds)
    grouped = defaultdict(float)
    for order, p in order_p.items():
        grouped[norm_combo(*order)] += p
    return dict(grouped)


def compute_delta(trio_odds, tf_odds):
    p_trio = implied_distribution(trio_odds)
    p_tf_set = trifecta_set_distribution(tf_odds)
    common = set(p_trio) & set(p_tf_set)
    return {c: p_tf_set[c] - p_trio[c] for c in common}


def entrants_from_trio(trio_odds) -> tuple[int, ...]:
    entrants = sorted({car for combo in trio_odds for car in combo})
    return tuple(entrants)


def two_of_three_middle_candidates(favorite, delta, entrants):
    fav = set(favorite)
    outsiders = sorted(set(entrants) - fav)
    rows = []
    for o in outsiders:
        replacements = []
        for removed in favorite:
            c = norm_combo(*(fav - {removed}), o)
            replacements.append((c, delta.get(c, float("-inf"))))
        positive = [(c, d) for c, d in replacements if d > 0]
        if len(positive) >= 2:
            strongest = max(positive, key=lambda x: (x[1], x[0]))
            rows.append({"outsider": o, "positive_count": len(positive), "representative": strongest[0], "delta": strongest[1]})
    return rows


def select_core_pair(candidates):
    score = defaultdict(float)
    support = defaultdict(int)
    for c in candidates:
        combo = c["representative"]
        d = c["delta"]
        for pair in itertools.combinations(combo, 2):
            pair = norm_combo(*pair)
            score[pair] += d
            support[pair] += 1
    if not score:
        return None
    return max(score, key=lambda p: (support[p], score[p], tuple(-x for x in p)))


def select_fourth(core, favorite, candidates, delta):
    pool = set(favorite)
    for c in candidates:
        pool.update(c["representative"])
    pool -= set(core)
    if not pool:
        return None
    return max(pool, key=lambda car: (delta.get(norm_combo(core[0], core[1], car), float("-inf")), -car))


def payout_for_hit(result_set, ticket, trio_odds):
    if norm_combo(*result_set) != ticket:
        return 0
    odds = trio_odds.get(ticket)
    return int(round(STAKE_PER_TICKET * odds)) if odds else 0


def main():
    gate = load_audited_gate()
    trio = load_trio_odds()
    trifecta = load_trifecta_odds()
    results = load_results()

    available = set(trio) & set(trifecta) & set(results)
    missing_gate = sorted(set(gate) - available)
    if missing_gate:
        raise RuntimeError(f"Audited gate races missing required data: {missing_gate}")
    race_ids = sorted(gate)

    names = ["A_2of3_representatives", "B_core_pair_all", "C_four_car_box", "D_four_car_without_favorite"]
    totals = {k: {"bet_races": 0, "tickets": 0, "hit_races": 0, "stake_yen": 0, "payout_yen": 0} for k in names}
    decisions = []

    for rid in race_ids:
        gate_row = gate[rid]
        favorite = norm_combo(*gate_row["favorite"])
        fav_odds = float(gate_row["favorite_odds"])
        fav_consensus = float(gate_row["favorite_consensus"])
        ratio = float(gate_row["second_to_fav_odds_ratio"])
        delta = compute_delta(trio[rid], trifecta[rid])
        entrants = entrants_from_trio(trio[rid])
        candidates = two_of_three_middle_candidates(favorite, delta, entrants)
        core = select_core_pair(candidates)

        methods = {}
        methods["A_2of3_representatives"] = sorted({c["representative"] for c in candidates})
        methods["B_core_pair_all"] = [norm_combo(core[0], core[1], x) for x in entrants if x not in core] if core else []

        fourth = select_fourth(core, favorite, candidates, delta) if core else None
        four = sorted(set(core or ()) | ({fourth} if fourth else set()) | set(favorite))
        if len(four) > 4:
            extra = [x for x in four if x not in core]
            extra.sort(key=lambda x: (delta.get(norm_combo(core[0], core[1], x), float("-inf")), -x), reverse=True)
            four = sorted(set(core) | set(extra[:2]))
        if len(four) == 4:
            methods["C_four_car_box"] = [norm_combo(*x) for x in itertools.combinations(four, 3)]
            methods["D_four_car_without_favorite"] = [x for x in methods["C_four_car_box"] if x != favorite]
        else:
            methods["C_four_car_box"] = []
            methods["D_four_car_without_favorite"] = []

        result_set = norm_combo(*results[rid])
        row = {
            "race_id": rid,
            "entrant_count": len(entrants),
            "entrants": "-".join(map(str, entrants)),
            "favorite": "-".join(map(str, favorite)),
            "favorite_odds": fav_odds,
            "favorite_consensus": fav_consensus,
            "second_to_fav_odds_ratio": ratio,
            "candidate_count": len(candidates),
            "core_pair": "-".join(map(str, core)) if core else "",
            "four_cars": "-".join(map(str, four)) if len(four) == 4 else "",
            "result": "-".join(map(str, result_set)),
        }

        for name, tickets in methods.items():
            tickets = sorted(set(tickets))
            if tickets:
                totals[name]["bet_races"] += 1
            totals[name]["tickets"] += len(tickets)
            stake = STAKE_PER_TICKET * len(tickets)
            payout = sum(payout_for_hit(result_set, t, trio[rid]) for t in tickets)
            hit = int(any(t == result_set for t in tickets))
            totals[name]["hit_races"] += hit
            totals[name]["stake_yen"] += stake
            totals[name]["payout_yen"] += payout
            row[f"{name}_tickets"] = "|".join("-".join(map(str, t)) for t in tickets)
            row[f"{name}_stake_yen"] = stake
            row[f"{name}_payout_yen"] = payout
            row[f"{name}_hit"] = hit
        decisions.append(row)

    for s in totals.values():
        s["profit_yen"] = s["payout_yen"] - s["stake_yen"]
        s["roi_pct"] = (100.0 * s["payout_yen"] / s["stake_yen"]) if s["stake_yen"] else None
        s["race_hit_rate_pct"] = (100.0 * s["hit_races"] / s["bet_races"]) if s["bet_races"] else None
        s["avg_tickets_per_bet_race"] = (s["tickets"] / s["bet_races"]) if s["bet_races"] else None

    candidate_distribution = defaultdict(int)
    for r in decisions:
        candidate_distribution[str(r["candidate_count"])] += 1

    output = {
        "status": "FAKE_FAVORITE_FORMATION_SIM_2023_AUDITED_GATE",
        "year": 2023,
        "years_read": [2023],
        "evaluation_year_2024_used": False,
        "evaluation_year_2025_used": False,
        "evaluation_year_2026_used": False,
        "staking": "Flat 100 yen per trio ticket; no dutching.",
        "gate_source": "data/audits/fake_favorite_true_middle_2023_2024.json exact 2023 race list",
        "fake_gate": {"fav_consensus_max": FAV_CONSENSUS_MAX, "second_to_fav_odds_ratio_max": SECOND_TO_FAV_ODDS_RATIO_MAX},
        "definitions": {
            "A_2of3_representatives": "For each actual outsider where at least 2 of the 3 one-car replacements of the audited trio favorite have positive delta=P_trifecta_set-P_trio, buy the strongest-delta representative.",
            "B_core_pair_all": "Choose the pair with greatest candidate support, breaking ties by summed positive delta, then buy that pair with every other actual entrant.",
            "C_four_car_box": "Core pair plus two strongest relevant additional cars, using only actual entrants, translated to a 4-car trio BOX (4 tickets).",
            "D_four_car_without_favorite": "Same four cars as C, but remove the exact audited trio-market favorite ticket if present.",
        },
        "fake_favorite_races": len(race_ids),
        "candidate_count_distribution": dict(sorted(candidate_distribution.items(), key=lambda kv: int(kv[0]))),
        "methods": totals,
    }

    AUDITS.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    with OUT_CSV.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(decisions[0].keys()))
        w.writeheader()
        w.writerows(decisions)
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
