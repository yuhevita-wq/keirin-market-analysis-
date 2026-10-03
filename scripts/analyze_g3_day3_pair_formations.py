from __future__ import annotations

import csv
import io
import itertools
import json
import math
import statistics
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_ROOT = ROOT / "data" / "grade_races" / "g3"
OUT = ROOT / "results" / "g3_day3_reality"


def decode(raw: bytes) -> str:
    for enc in ("utf-8-sig", "utf-8", "cp932", "shift_jis"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            pass
    return raw.decode("utf-8", errors="replace")


def read_member(zf: zipfile.ZipFile, basename: str) -> list[dict[str, str]]:
    matches = [n for n in zf.namelist() if Path(n).name == basename]
    if not matches:
        return []
    return list(csv.DictReader(io.StringIO(decode(zf.read(matches[0])))))


def fnum(v: str | None, default: float = float("-inf")) -> float:
    try:
        x = float(v or "")
        return x if math.isfinite(x) else default
    except (TypeError, ValueError):
        return default


def inum(v: str | None) -> int | None:
    try:
        return int(v or "")
    except (TypeError, ValueError):
        return None


def meeting_day(race_id: str) -> int | None:
    if len(race_id) == 16 and race_id.isdigit():
        try:
            return int(race_id[10:12])
        except ValueError:
            return None
    return None


def line_rank(entries: list[dict[str, str]]) -> list[tuple[int, list[dict[str, str]], float]]:
    by_line: dict[int, list[dict[str, str]]] = defaultdict(list)
    for e in entries:
        if e.get("line_id", "").isdigit() and e.get("line_position", "").isdigit():
            by_line[int(e["line_id"])].append(e)
    rows = []
    for lid, mem in by_line.items():
        mem = sorted(mem, key=lambda x: int(x["line_position"]))
        if len(mem) < 2:
            continue
        pair = fnum(mem[0].get("score")) + fnum(mem[1].get("score"))
        key = (pair, fnum(mem[0].get("score")), 1 if len(mem) >= 3 else 0, -lid)
        rows.append((key, lid, mem, pair))
    rows.sort(reverse=True, key=lambda x: x[0])
    return [(lid, mem, pair) for _, lid, mem, pair in rows]


def unique(seq: list[int | None]) -> list[int]:
    out: list[int] = []
    seen = set()
    for x in seq:
        if x is None or x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


def trio_key(a: int, b: int, c: int) -> str:
    return "=".join(str(x) for x in sorted((a, b, c)))


def paid_trio(payouts: list[dict[str, str]]) -> tuple[str | None, int]:
    for p in payouts:
        if p.get("ticket_type") == "3連複" and p.get("status") == "paid" and p.get("combination"):
            try:
                return p["combination"], int(float(p.get("payout_yen") or 0))
            except ValueError:
                return p["combination"], 0
    return None, 0


def candidate_order_market(pair: tuple[int, int], pool: list[int], odds: dict[str, dict[str, str]]) -> list[int]:
    a, b = pair
    return sorted(pool, key=lambda x: (fnum(odds.get(trio_key(a, b, x), {}).get("odds"), 1e18), x))


def classify_third(x: int, A: int, B: int, C: int, D: int, main_mem: list[int], rival_mem: list[int], entry_by_car: dict[int, dict[str, str]]) -> str:
    if len(main_mem) >= 3 and x == main_mem[2]:
        return "main3"
    if x == C:
        return "C_rival_leader"
    if x == D:
        return "D_rival_second"
    if x in main_mem[3:]:
        return "main4plus"
    if x in rival_mem[2:]:
        return "rival3plus"
    e = entry_by_car.get(x, {})
    lp = e.get("line_position", "")
    ls = e.get("line_size", "")
    if ls == "1" or not e.get("line_id", "").isdigit():
        return "singleton"
    if lp == "1":
        return "other_line_leader"
    if lp == "2":
        return "other_line_second"
    return "other"


def summarize(records: list[dict], which: str, split: str) -> dict:
    rows = [r for r in records if r["split"] == split]
    stake = sum(r[f"{which}_points"] * 100 for r in rows)
    ret = sum(r[f"{which}_return"] for r in rows)
    hits = sum(r[f"{which}_hit"] for r in rows)
    points = sum(r[f"{which}_points"] for r in rows)
    return {
        "races": len(rows),
        "avg_points": points / len(rows) if rows else 0,
        "hit_rate": hits / len(rows) if rows else 0,
        "stake": stake,
        "return": ret,
        "roi": ret / stake if stake else 0,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    races: dict[str, dict[str, str]] = {}
    entries_by_race: dict[str, list[dict[str, str]]] = defaultdict(list)
    results_by_race: dict[str, list[dict[str, str]]] = defaultdict(list)
    payouts_by_race: dict[str, list[dict[str, str]]] = defaultdict(list)
    trio_odds_by_race: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)

    for path in sorted(ARCHIVE_ROOT.rglob("*.zip")):
        with zipfile.ZipFile(path) as zf:
            for r in read_member(zf, "races.csv"):
                races.setdefault(r["race_id"], r)
            for e in read_member(zf, "entries.csv"):
                entries_by_race[e["race_id"]].append(e)
            for rr in read_member(zf, "results.csv"):
                results_by_race[rr["race_id"]].append(rr)
            for p in read_member(zf, "payouts.csv"):
                payouts_by_race[p["race_id"]].append(p)
            for o in read_member(zf, "trio_final_odds.csv"):
                trio_odds_by_race[o["race_id"]][o.get("combination", "")] = o

    # strategy_name -> per-race records
    strat_records: dict[str, list[dict]] = defaultdict(list)
    ab_third_roles = Counter()
    cd_third_roles = Counter()
    pair_presence = Counter()
    pair_presence_by_split = defaultdict(Counter)
    race_rows = []

    for race in sorted(races.values(), key=lambda r: (r.get("race_date", ""), r.get("track", ""), int(r.get("race_no", "0") or 0))):
        rid = race["race_id"]
        if meeting_day(rid) != 3:
            continue
        es = entries_by_race.get(rid, [])
        ranked = line_rank(es)
        if len(ranked) < 2:
            continue
        main_lid, main_entries, main_pair_score = ranked[0]
        rival_lid, rival_entries, rival_pair_score = ranked[1]
        main_mem = [int(e["car_no"]) for e in main_entries]
        rival_mem = [int(e["car_no"]) for e in rival_entries]
        A, B = main_mem[:2]
        C, D = rival_mem[:2]
        entry_by_car = {int(e["car_no"]): e for e in es if e.get("car_no", "").isdigit()}
        all_cars = sorted(entry_by_car)
        M3 = main_mem[2] if len(main_mem) >= 3 else None
        R3 = rival_mem[2] if len(rival_mem) >= 3 else None

        finish = []
        for rr in results_by_race.get(rid, []):
            p = inum(rr.get("finish_position")); c = inum(rr.get("car_no"))
            if p is not None and c is not None and p <= 3:
                finish.append((p, c))
        if len(finish) != 3 or sorted(p for p, _ in finish) != [1, 2, 3]:
            continue
        top3 = [c for _, c in sorted(finish)]
        win_combo, win_pay = paid_trio(payouts_by_race.get(rid, []))
        if not win_combo:
            continue
        year = int(race["race_date"][:4])
        split = "train" if year <= 2024 else "test"
        odds = trio_odds_by_race.get(rid, {})

        ab_present = A in top3 and B in top3
        cd_present = C in top3 and D in top3
        pair_presence["races"] += 1
        pair_presence["AB"] += int(ab_present)
        pair_presence["CD"] += int(cd_present)
        pair_presence["AB_or_CD"] += int(ab_present or cd_present)
        pair_presence["AB_and_CD"] += int(ab_present and cd_present)
        pair_presence_by_split[split]["races"] += 1
        pair_presence_by_split[split]["AB"] += int(ab_present)
        pair_presence_by_split[split]["CD"] += int(cd_present)
        pair_presence_by_split[split]["AB_or_CD"] += int(ab_present or cd_present)

        if ab_present:
            x = next(c for c in top3 if c not in {A, B})
            ab_third_roles[classify_third(x, A, B, C, D, main_mem, rival_mem, entry_by_car)] += 1
        if cd_present:
            x = next(c for c in top3 if c not in {C, D})
            # classify CD third primarily by A/B relation
            if x == A: role = "A_main_leader"
            elif x == B: role = "B_main_second"
            elif M3 is not None and x == M3: role = "main3"
            elif R3 is not None and x == R3: role = "rival3"
            else:
                e = entry_by_car.get(x, {})
                lp = e.get("line_position", "")
                if lp == "1": role = "other_line_leader"
                elif lp == "2": role = "other_line_second"
                elif not e.get("line_id", "").isdigit() or e.get("line_size") == "1": role = "singleton"
                else: role = "other"
            cd_third_roles[role] += 1

        # Generic remaining-rider rankings.
        ab_pool = [x for x in all_cars if x not in {A, B}]
        cd_pool = [x for x in all_cars if x not in {C, D}]
        ab_score = sorted(ab_pool, key=lambda x: (-fnum(entry_by_car[x].get("score")), x))
        ab_top3 = sorted(ab_pool, key=lambda x: (-fnum(entry_by_car[x].get("top3_rate")), -fnum(entry_by_car[x].get("score")), x))
        ab_market = candidate_order_market((A, B), ab_pool, odds)
        cd_score = sorted(cd_pool, key=lambda x: (-fnum(entry_by_car[x].get("score")), x))
        cd_top3 = sorted(cd_pool, key=lambda x: (-fnum(entry_by_car[x].get("top3_rate")), -fnum(entry_by_car[x].get("score")), x))
        cd_market = candidate_order_market((C, D), cd_pool, odds)

        # AB: exactly the user's requested 2-3 third-slot candidates.
        ab_defs: dict[str, list[int]] = {
            "AB2_CD": unique([C, D])[:2],
            "AB2_M3C": unique([M3, C, D])[:2],
            "AB2_M3D": unique([M3, D, C])[:2],
            "AB2_score": ab_score[:2],
            "AB2_top3": ab_top3[:2],
            "AB2_market": ab_market[:2],
            "AB3_M3CD": unique([M3, C, D] + ab_score)[:3],
            "AB3_CDscore": unique([C, D] + ab_score)[:3],
            "AB3_score": ab_score[:3],
            "AB3_top3": ab_top3[:3],
            "AB3_market": ab_market[:3],
        }

        # CD: second-strongest line fixed pair. Test how far the third slot should extend.
        cd_defs: dict[str, list[int]] = {}
        for k in range(2, 7):
            cd_defs[f"CD{k}_ABscore"] = unique([A, B, M3] + cd_score)[:k]
            cd_defs[f"CD{k}_score"] = cd_score[:k]
            cd_defs[f"CD{k}_top3"] = cd_top3[:k]
            cd_defs[f"CD{k}_market"] = cd_market[:k]

        for name, xs in {**ab_defs, **cd_defs}.items():
            pair = (A, B) if name.startswith("AB") else (C, D)
            combos = [trio_key(pair[0], pair[1], x) for x in xs if x not in pair]
            combos = list(dict.fromkeys(combos))
            points = len(combos)
            hit = int(win_combo in combos)
            ret = win_pay if hit else 0
            strat_records[name].append({
                "split": split,
                "points": points,
                "hit": hit,
                "return": ret,
                "offwinner": int(top3[0] not in main_mem),
                "mainwinner": int(top3[0] in main_mem),
            })

        race_rows.append({
            "race_id": rid, "date": race["race_date"], "track": race.get("track", ""), "race_no": race.get("race_no", ""),
            "A": A, "B": B, "C": C, "D": D, "M3": M3 or "", "main_pair_score": main_pair_score,
            "rival_pair_score": rival_pair_score, "pair_gap": main_pair_score-rival_pair_score,
            "top3": "-".join(map(str, top3)), "ab_present": int(ab_present), "cd_present": int(cd_present),
            "winning_trio": win_combo, "payout": win_pay, "split": split,
        })

    # Summaries.
    strategy_summary = {}
    for name, recs in strat_records.items():
        strategy_summary[name] = {}
        for split in ("train", "test"):
            rows = [r for r in recs if r["split"] == split]
            stake = sum(r["points"] * 100 for r in rows)
            ret = sum(r["return"] for r in rows)
            hits = sum(r["hit"] for r in rows)
            off_rows = [r for r in rows if r["offwinner"]]
            main_rows = [r for r in rows if r["mainwinner"]]
            strategy_summary[name][split] = {
                "races": len(rows),
                "avg_points": sum(r["points"] for r in rows)/len(rows) if rows else 0,
                "hit_rate": hits/len(rows) if rows else 0,
                "roi": ret/stake if stake else 0,
                "off_capture": sum(r["hit"] for r in off_rows)/len(off_rows) if off_rows else 0,
                "main_capture": sum(r["hit"] for r in main_rows)/len(main_rows) if main_rows else 0,
            }

    def rate(counter: Counter, key: str) -> float:
        n = counter["races"]
        return counter[key]/n if n else 0

    out = {
        "pair_presence": {
            "all": {k: pair_presence[k] for k in ("races","AB","CD","AB_or_CD","AB_and_CD")},
            "train": dict(pair_presence_by_split["train"]),
            "test": dict(pair_presence_by_split["test"]),
        },
        "ab_third_roles": dict(ab_third_roles.most_common()),
        "cd_third_roles": dict(cd_third_roles.most_common()),
        "strategies": strategy_summary,
        "definitions": {
            "A_B": "strongest published 2+ rider line leader/second by pair score",
            "C_D": "second-strongest published 2+ rider line leader/second by pair score",
            "train": "2022-2024",
            "test": "2025-2026H1",
        }
    }
    (OUT / "pair_formations.json").write_text(json.dumps(out, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")

    # Markdown: prioritize stable strategies by min(train,test ROI), then low points.
    lines = ["# G3三日目 3連複 固定ペア再設計", "", f"有効2ライン以上: {pair_presence['races']}R", ""]
    lines += ["## 固定ペアが実際にTOP3へ2人とも残る率"]
    for label, cnt in (("AB", pair_presence["AB"]),("CD", pair_presence["CD"]),("ABまたはCD", pair_presence["AB_or_CD"])):
        lines.append(f"- {label}: {cnt}/{pair_presence['races']} = {cnt/pair_presence['races']:.1%}")
    lines += ["", "## ABが残った時、3人目は誰だったか"]
    total_ab = sum(ab_third_roles.values())
    for role, cnt in ab_third_roles.most_common():
        lines.append(f"- {role}: {cnt}/{total_ab} = {cnt/total_ab:.1%}")
    lines += ["", "## CDが残った時、3人目は誰だったか"]
    total_cd = sum(cd_third_roles.values())
    for role, cnt in cd_third_roles.most_common():
        lines.append(f"- {role}: {cnt}/{total_cd} = {cnt/total_cd:.1%}")

    def emit(title: str, prefix: str):
        lines.extend(["", f"## {title}"])
        names = [n for n in strategy_summary if n.startswith(prefix)]
        names.sort(key=lambda n: (min(strategy_summary[n]["train"]["roi"], strategy_summary[n]["test"]["roi"]), -strategy_summary[n]["test"]["avg_points"]), reverse=True)
        for n in names:
            tr = strategy_summary[n]["train"]; te = strategy_summary[n]["test"]
            lines.append(
                f"- {n}: train {tr['avg_points']:.1f}点 hit {tr['hit_rate']:.1%} ROI {tr['roi']:.1%} / "
                f"test {te['avg_points']:.1f}点 hit {te['hit_rate']:.1%} ROI {te['roi']:.1%} | "
                f"test外勝者捕捉 {te['off_capture']:.1%} 主力勝者捕捉 {te['main_capture']:.1%}"
            )
    emit("AB固定 2〜3点", "AB")
    emit("CD固定 相手2〜6人", "CD")
    lines += ["", "注: 100円均等。候補選定はすべてレース前情報のみ。市場順は確定3連複オッズを使用。結果は候補選定に未使用。"]
    (OUT / "pair_formations.md").write_text("\n".join(lines)+"\n", encoding="utf-8")

    # Save compact race rows for auditing.
    if race_rows:
        with (OUT / "pair_formations_races.csv").open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(race_rows[0].keys()))
            w.writeheader(); w.writerows(race_rows)

    print("\n".join(lines))


if __name__ == "__main__":
    main()
