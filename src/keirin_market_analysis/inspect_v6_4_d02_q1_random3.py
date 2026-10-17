from __future__ import annotations

import json
import random

from develop_v6_4_d02_q2q3 import analyze_race, load_data, pint

SCHEME_VERSION = "v6.4-D02-Q1-RANDOM3"
DATASET = "2024Q1"
RANDOM_SEED = 6403
SAMPLE_SIZE = 3
D01_M_PRE_THRESHOLD = 0.35640013538348414
D02_PRUNE_DAMAGE_THRESHOLD = 0.3317657935777282


def main():
    races, trio, tf, payouts = load_data("2024_q1")
    eligible = []

    for rid, r in sorted(races.items(), key=lambda kv: (kv[1].get("race_date", ""), kv[0])):
        if r.get("meeting_grade") != "F1" or not (r.get("race_type") or "").startswith("Ｓ級"):
            continue
        if pint(r.get("entry_count")) != 7 or len(trio.get(rid, {})) != 35 or len(tf.get(rid, {})) != 210:
            continue
        if rid not in payouts or not payouts[rid]:
            continue
        x = analyze_race(r, trio[rid], tf[rid])
        if not x or not x.get("entry_pass") or x["final_bet_count"] < 2:
            continue
        if x["M_pre"] < D01_M_PRE_THRESHOLD:
            continue
        if x.get("PruneDamage") is None or x["PruneDamage"] > D02_PRUNE_DAMAGE_THRESHOLD:
            continue
        eligible.append((rid, r, x))

    rng = random.Random(RANDOM_SEED)
    chosen = rng.sample(eligible, SAMPLE_SIZE)

    samples = []
    for rid, r, x in chosen:
        paid = payouts[rid]
        actual_paid = [
            {"combination": "-".join(map(str, t)), "payout_yen": y}
            for t, y in sorted(paid.items())
        ]
        bets = [
            {
                "combination": "-".join(map(str, t)),
                "odds": tf[rid][t],
                "stake_yen": 100,
                "hit": t in paid,
            }
            for t in sorted(x["kept"])
        ]
        samples.append({
            "race_id": rid,
            "race_date": r.get("race_date"),
            "track": r.get("track"),
            "race_no": pint(r.get("race_no")),
            "race_type": r.get("race_type"),
            "predicted_line_formation": r.get("predicted_line_formation"),
            "M_pre": x["M_pre"],
            "PruneDamage": x["PruneDamage"],
            "final_bet_count": x["final_bet_count"],
            "bets": bets,
            "actual_paid": actual_paid,
            "race_hit": any(t in paid for t in x["kept"]),
        })

    result = {
        "scheme_version": SCHEME_VERSION,
        "dataset": DATASET,
        "selection_pool": "2024Q1 races purchased by fixed v6.4-D02",
        "selection_pool_size": len(eligible),
        "random_seed": RANDOM_SEED,
        "sample_size": SAMPLE_SIZE,
        "filters": {
            "M_pre_min": D01_M_PRE_THRESHOLD,
            "PruneDamage_max": D02_PRUNE_DAMAGE_THRESHOLD,
        },
        "samples": samples,
    }
    print("V6_4_D02_Q1_RANDOM3_BEGIN")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print("V6_4_D02_Q1_RANDOM3_END")


if __name__ == "__main__":
    main()
