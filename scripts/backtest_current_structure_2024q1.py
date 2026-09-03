from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

DATA = Path("data/2024/s_class_f1_all_parts/2024_q1")
OUT = Path("results/current_structure_v0/2024_q1")


def read_csv(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def fnum(v, default=float("-inf")):
    try:
        return float(str(v).replace(",", ""))
    except (TypeError, ValueError):
        return default


def inum(v, default=0):
    try:
        return int(float(str(v).replace(",", "")))
    except (TypeError, ValueError):
        return default


def build_lines(es):
    by = defaultdict(list)
    for e in es:
        lid = e.get("line_id", "")
        pos = e.get("line_position", "")
        if str(lid).isdigit() and str(pos).isdigit():
            by[int(lid)].append(e)
    out = []
    for lid, members in by.items():
        members = sorted(members, key=lambda e: int(e["line_position"]))
        if not members:
            continue
        leader = members[0]
        second = members[1] if len(members) >= 2 else None
        pair_mean = None
        structural_strength = None
        if second is not None:
            a = fnum(leader.get("score"))
            b = fnum(second.get("score"))
            if a != float("-inf") and b != float("-inf"):
                pair_mean = (a + b) / 2.0
                # Frozen before results are read: a modest 1.5-point structural bonus
                # for a 3+ rider line, reflecting third-rider residual value without
                # treating the third rider as equal to the front pair.
                structural_strength = pair_mean + (1.5 if len(members) >= 3 else 0.0)
        out.append({
            "line_id": lid,
            "members": members,
            "leader": leader,
            "second": second,
            "size": len(members),
            "pair_mean": pair_mean,
            "structural_strength": structural_strength,
            "leader_b": fnum(leader.get("b_count"), 0.0),
        })
    return out


def car(e):
    return int(e["car_no"])


def choose_tickets(race, es):
    lines = build_lines(es)
    multi = [x for x in lines if x["size"] >= 2 and x["structural_strength"] is not None]
    if not multi:
        return None

    # Core line: strongest front pair, with the frozen 3+ line residual bonus.
    core = max(
        multi,
        key=lambda x: (
            x["structural_strength"],
            fnum(x["leader"].get("score")),
            x["size"],
            -x["line_id"],
        ),
    )
    # Pace line: pre-race B count of the line leader. Ties go to structural strength.
    pace = max(
        multi,
        key=lambda x: (
            x["leader_b"],
            x["structural_strength"],
            fnum(x["leader"].get("score")),
            -x["line_id"],
        ),
    )

    A = car(core["leader"])
    B = car(core["second"])
    xs = []
    reasons = {}

    def add_x(e, reason):
        if e is None:
            return
        c = car(e)
        if c in (A, B) or c in xs:
            return
        xs.append(c)
        reasons[str(c)] = reason

    if pace["line_id"] != core["line_id"]:
        if core["size"] >= 3:
            add_x(core["members"][2], "core_third")
        add_x(pace["leader"], "pace_leader")
        add_x(pace["second"], "pace_second")
    else:
        if core["size"] >= 3:
            add_x(core["members"][2], "core_third")
        rivals = sorted(
            [x for x in multi if x["line_id"] != core["line_id"]],
            key=lambda x: (x["structural_strength"], fnum(x["leader"].get("score"))),
            reverse=True,
        )
        for r in rivals:
            add_x(r["leader"], "rival_leader")
            if len(xs) >= 3:
                break

    # Always fill to three third-place candidates from highest pre-race score outsiders.
    for e in sorted(es, key=lambda e: (fnum(e.get("score")), fnum(e.get("b_count"), 0.0)), reverse=True):
        if len(xs) >= 3:
            break
        add_x(e, "score_fill")

    xs = xs[:3]
    tickets = []
    ticket_reason = {}

    def add_ticket(t, reason):
        if len(set(t)) != 3:
            return
        s = "-".join(map(str, t))
        if s not in tickets:
            tickets.append(s)
            ticket_reason[s] = reason

    for x in xs:
        add_ticket((A, B, x), "core_ABX")
        add_ticket((B, A, x), "core_BAX")

    # If a distinct 3+ rider line is the pace line, keep two pace-survival routes.
    if pace["line_id"] != core["line_id"] and pace["size"] >= 3:
        P1 = car(pace["leader"])
        P2 = car(pace["second"])
        P3 = car(pace["members"][2])
        add_ticket((P1, P2, A), "pace_P1P2_coreleader")
        add_ticket((P1, P2, P3), "pace_full_line")

    return {
        "race_id": race["race_id"],
        "race_date": race["race_date"],
        "track": race["track"],
        "race_no": int(race["race_no"]),
        "race_type": race["race_type"],
        "formation": race.get("predicted_line_formation", ""),
        "core_line_id": core["line_id"],
        "core_line": "-".join(str(car(e)) for e in core["members"]),
        "core_strength": round(core["structural_strength"], 5),
        "core_A": A,
        "core_B": B,
        "pace_line_id": pace["line_id"],
        "pace_line": "-".join(str(car(e)) for e in pace["members"]),
        "pace_leader_b": pace["leader_b"],
        "third_candidates": xs,
        "third_reasons": reasons,
        "tickets": tickets,
        "ticket_reasons": ticket_reason,
        "bet_count": len(tickets),
    }


def summarize(rows):
    n = len(rows)
    stake = sum(r["stake_yen"] for r in rows)
    ret = sum(r["return_yen"] for r in rows)
    hits = sum(r["hit"] for r in rows)
    return {
        "races": n,
        "hit_races": hits,
        "hit_rate": hits / n if n else 0.0,
        "average_bet_count": sum(r["bet_count"] for r in rows) / n if n else 0.0,
        "stake_yen": stake,
        "return_yen": ret,
        "profit_yen": ret - stake,
        "roi": ret / stake if stake else 0.0,
        "core_pair_top2_races": sum(r["core_pair_top2"] for r in rows),
        "core_pair_top2_rate": sum(r["core_pair_top2"] for r in rows) / n if n else 0.0,
        "core_pair_both_top3_races": sum(r["core_pair_both_top3"] for r in rows),
        "core_pair_both_top3_rate": sum(r["core_pair_both_top3"] for r in rows) / n if n else 0.0,
    }


def infer_col(fieldnames, names):
    fmap = {str(x).lower(): x for x in fieldnames or []}
    for n in names:
        if n.lower() in fmap:
            return fmap[n.lower()]
    return None


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    races = read_csv(DATA / "races.csv")
    entries = read_csv(DATA / "entries.csv")
    eb = defaultdict(list)
    for e in entries:
        eb[e["race_id"]].append(e)

    # ---------------- PRE-RESULT STAGE ----------------
    predictions = []
    eligible_races = []
    for race in sorted(races, key=lambda r: (r["race_date"], r["track"], int(r["race_no"]))):
        if int(race.get("entry_count") or 0) != 7:
            continue
        p = choose_tickets(race, eb[race["race_id"]])
        if not p or not p["tickets"]:
            continue
        predictions.append(p)
        eligible_races.append(race)

    # Materialize predictions before any result/payout file is opened.
    (OUT / "predictions_pre_result.json").write_text(
        json.dumps(predictions, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    # ---------------- RESULT STAGE ----------------
    payouts = read_csv(DATA / "payouts.csv")
    actual = {}
    payout_map = {}
    for p in payouts:
        if p.get("ticket_type") == "3連単" and p.get("status") == "paid" and p.get("combination"):
            rid = p["race_id"]
            combo = p["combination"].strip()
            actual[rid] = combo
            payout_map[rid] = inum(p.get("payout_yen"), 0)

    evaluated = []
    for p in predictions:
        rid = p["race_id"]
        combo = actual.get(rid, "")
        if not combo:
            continue
        parts = [inum(x, -1) for x in combo.split("-")]
        tickets = p["tickets"]
        hit = int(combo in tickets)
        ret = payout_map.get(rid, 0) if hit else 0
        A, B = p["core_A"], p["core_B"]
        pair_top2 = int(len(parts) >= 2 and set(parts[:2]) == {A, B})
        pair_top3 = int(len(parts) >= 3 and A in parts[:3] and B in parts[:3])
        evaluated.append({
            **p,
            "actual_trifecta": combo,
            "actual_payout_yen": payout_map.get(rid, 0),
            "hit": hit,
            "hit_ticket": combo if hit else "",
            "stake_yen": 100 * len(tickets),
            "return_yen": ret,
            "profit_yen": ret - 100 * len(tickets),
            "core_pair_top2": pair_top2,
            "core_pair_both_top3": pair_top3,
        })

    by_type = {}
    for rt in sorted({r["race_type"] for r in evaluated}):
        by_type[rt] = summarize([r for r in evaluated if r["race_type"] == rt])

    # ---------------- MARKET BASELINE, COMPARISON ONLY ----------------
    # This is explicitly NOT used to create Current-Structure tickets.
    odds_path = DATA / "trifecta_final_odds.csv"
    market = {
        "available": False,
        "note": "Final odds are comparison-only, never used by Current-Structure v0.",
    }
    if odds_path.exists():
        with odds_path.open("r", encoding="utf-8-sig", newline="") as f:
            rdr = csv.DictReader(f)
            fields = rdr.fieldnames or []
            rid_col = infer_col(fields, ["race_id"])
            combo_col = infer_col(fields, ["combination", "combo", "bet_combination"])
            odds_col = infer_col(fields, ["odds", "final_odds", "odds_value"])
            market["fieldnames"] = fields
            market["inferred_columns"] = {"race_id": rid_col, "combination": combo_col, "odds": odds_col}
            if rid_col and combo_col and odds_col:
                ob = defaultdict(list)
                for row in rdr:
                    o = fnum(row.get(odds_col), None)
                    if o is None or o <= 0:
                        continue
                    ob[row[rid_col]].append((o, row[combo_col].strip()))
                base_rows = []
                eval_by_id = {r["race_id"]: r for r in evaluated}
                for rid, er in eval_by_id.items():
                    nbet = er["bet_count"]
                    opts = sorted(ob.get(rid, []), key=lambda x: x[0])
                    if len(opts) < nbet:
                        continue
                    ts = [c for _, c in opts[:nbet]]
                    combo = er["actual_trifecta"]
                    hit = int(combo in ts)
                    ret = er["actual_payout_yen"] if hit else 0
                    base_rows.append({
                        "race_id": rid,
                        "race_type": er["race_type"],
                        "bet_count": nbet,
                        "stake_yen": 100 * nbet,
                        "return_yen": ret,
                        "hit": hit,
                        "core_pair_top2": 0,
                        "core_pair_both_top3": 0,
                    })
                if base_rows:
                    b = summarize(base_rows)
                    market.update({"available": True, "comparable_races": len(base_rows), **b})
                    market["by_race_type"] = {
                        rt: summarize([r for r in base_rows if r["race_type"] == rt])
                        for rt in sorted({r["race_type"] for r in base_rows})
                    }

    summary = {
        "experiment": "Current-Structure v0 reality check",
        "frozen_before_result_join": True,
        "scope": {
            "period": "2024-01-01..2024-03-31",
            "meeting_grade": "F1",
            "class": "S-class all race types",
            "entry_count": 7,
            "source_dir": str(DATA),
            "q1_races_in_source": len(races),
            "q1_7car_races": sum(1 for r in races if int(r.get("entry_count") or 0) == 7),
            "predicted_races": len(predictions),
            "evaluated_races_with_paid_trifecta": len(evaluated),
        },
        "rules": {
            "inputs_only": ["predicted line/line_id/line_position", "score", "b_count"],
            "core_line": "among 2+ rider lines: mean(score leader, score second) + 1.5 if line has 3+ riders; ties leader score, line size, lower line_id",
            "pace_line": "highest leader b_count among 2+ rider lines; ties structural strength then leader score",
            "third_candidates_if_distinct_pace": "core third + pace leader + pace second; fill by highest score to 3 unique outsiders",
            "third_candidates_if_same_pace": "core third + strongest rival line leaders; fill by highest score to 3 unique outsiders",
            "core_bets": "A-B-X and B-A-X for three X candidates",
            "pace_extras": "if distinct pace line has 3+ riders: P1-P2-A and P1-P2-P3",
            "stake": "100 yen per ticket, no result-dependent skipping or resizing",
            "odds_use": "none in structure predictions; final odds only for equal-ticket-count market baseline",
        },
        "current_structure": {
            "overall": summarize(evaluated),
            "by_race_type": by_type,
        },
        "market_same_ticket_count_baseline": market,
    }

    (OUT / "evaluated.json").write_text(json.dumps(evaluated, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
