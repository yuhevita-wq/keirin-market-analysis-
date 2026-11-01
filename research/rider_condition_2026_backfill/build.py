#!/usr/bin/env python3
"""Read-only consolidation of 2025 through 2026-H1 rider/race records.

Inputs remain unchanged. No predictive model, odds, or post-race feature is modified.
Sources are selected from canonical directories, never data/all_classes aliases.
"""
import csv
import gzip
import io
import json
import re
import unicodedata
import zipfile
from collections import Counter, defaultdict
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
START, END = "2025-01-01", "2026-10-31"
OUT = ROOT / "data/rider_condition_2025_202610"
DATE_RE = re.compile(r"^\d{4}-\d\d-\d\d$")

def norm(v):
    return unicodedata.normalize("NFKC", str(v or "")).strip()

def actual_date(rid):
    if not re.fullmatch(r"\d{16}", rid):
        return ""
    try:
        day = date(int(rid[2:6]), int(rid[6:8]), int(rid[8:10]))
        n = int(rid[10:12])
        if not 1 <= n <= 7 or not 1 <= int(rid[-2:]) <= 12:
            return ""
        return (day + timedelta(days=n-1)).isoformat()
    except ValueError:
        return ""

def eligible(r):
    rid = norm(r.get("race_id"))
    day = norm(r.get("race_date")) or actual_date(rid)
    typ = norm(r.get("race_type")).upper()
    klass = norm(r.get("class_scope")).upper()
    return (START <= day <= END and actual_date(rid) == day
            and not any(x in typ for x in ("ガールズ", "女子", "GIRLS", "ガ予", "ガ決"))
            and not ("L級" in typ or klass.startswith("L")))

def csv_rows(handle):
    yield from csv.DictReader(handle)

def folders():
    d = ROOT / "data"
    sources = list(sorted((d / "2025/s_class_f1_all_parts").glob("*/")))
    sources += [d / "2025/s_class_yosen",
                d / "2026_h1/s_class_f1_all", d / "2026_h1/s_class_yosen",
                d / "2026_future_block1/s_class_f1_20260701_20260830"]
    sources += sorted((d / "rider_condition_2026_backfill").glob("*/"))
    for prefix in ("a_class_12", "a3_challenge"):
        base = d / prefix
        sources += sorted(p for p in base.iterdir()
                          if p.is_dir() and (p.name.startswith("2025_") or
                                             p.name in {"2026_01_02", "2026_03_04", "2026_05_06", "2026_07_08", "2026_09_1002", "2026_09_1007"}))
    for p in sources:
        if (p / "races.csv").exists():
            yield p

def archives():
    base = ROOT / "data/grade_races"
    yield from sorted((base / "g1g2").glob("*2025*.zip"))
    yield from sorted((base / "g3/2025").glob("*.zip"))
    yield from sorted((base / "g3/2026_h1").glob("*.zip"))

def zipped_sets(p):
    with zipfile.ZipFile(p) as z:
        files = [n for n in z.namelist() if n.rsplit("/",1)[-1] == "races.csv"]
        for name in files:
            directory = name.rpartition("/")[0]
            prefix = directory + "/" if directory else ""
            found = set(z.namelist())
            payload = {}
            for kind in ("races", "entries", "results"):
                member = prefix + kind + ".csv"
                if member not in found:
                    payload[kind] = []
                    continue
                with z.open(member) as b, io.TextIOWrapper(b, encoding="utf-8-sig", newline="") as f:
                    payload[kind] = list(csv_rows(f))
            yield str(p.relative_to(ROOT)) + "!" + prefix, payload

def csv_set(p):
    payload = {}
    for kind in ("races", "entries", "results"):
        f = p / (kind + ".csv")
        if f.exists():
            with f.open(encoding="utf-8-sig", newline="") as h:
                payload[kind] = list(csv_rows(h))
        else:
            payload[kind] = []
    return str(p.relative_to(ROOT)), payload

def distinct_key(r, kind):
    rid = norm(r.get("race_id"))
    if kind == "races":
        return rid
    car = norm(r.get("car_no"))
    if not re.fullmatch(r"[1-9]", car):
        return ""
    return rid + ":" + car

def conflict_fields(kind):
    return {
        "races": ("race_date", "track", "race_type", "meeting_grade"),
        "entries": ("race_date", "car_no", "player_name", "term", "prefecture", "class"),
        "results": ("race_date", "car_no", "player_name", "finish_position"),
    }[kind]

def merge():
    items = {k:{} for k in ("races", "entries", "results")}
    provenance = defaultdict(set)
    disputes = []
    rejected = Counter()
    scanned = []
    bad_results = set()
    def consume(source, payload):
        stats = Counter()
        for kind in ("races", "entries", "results"):
            for row in payload[kind]:
                if not eligible(row):
                    stats[kind + "_out_of_scope"] += 1
                    continue
                key = distinct_key(row, kind)
                if not key:
                    stats[kind + "_invalid_key"] += 1
                    continue
                provenance[kind + ":" + key].add(source)
                prev = items[kind].get(key)
                if prev is None:
                    items[kind][key] = row
                    stats[kind + "_accepted"] += 1
                else:
                    diffs = [f for f in conflict_fields(kind)
                             if norm(prev.get(f)) and norm(row.get(f))
                             and norm(prev.get(f)) != norm(row.get(f))]
                    if diffs:
                        disputes.append({"kind":kind, "key":key, "source":source,
                                         "disagreeing_fields":diffs})
                        if kind == "results": bad_results.add(key)
                    else:
                        stats[kind + "_duplicate"] += 1
        scanned.append({"source":source, **stats})
    for folder in folders():
        name, payload = csv_set(folder)
        consume(name, payload)
    for path in archives():
        try:
            for source, payload in zipped_sets(path):
                consume(source, payload)
        except (OSError, zipfile.BadZipFile) as exc:
            rejected[str(path.relative_to(ROOT))] += 1
            disputes.append({"kind":"source_error","source":str(path.relative_to(ROOT)),
                             "error":str(exc)})
    for key in bad_results:
        items["results"].pop(key, None)
    return items, scanned, disputes, provenance, rejected

def published_podium():
    podiums = {}
    invalid = set()
    for year in (2025, 2026):
        p = ROOT / f"data/result_universe_v2/combined/results_{year}.jsonl.gz"
        if not p.exists():
            continue
        with gzip.open(p, "rt", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                rid = str(r["race_id"])
                if not START <= actual_date(rid) <= END:
                    continue
                order = tuple(str(x) for x in r["finish_order"])
                if len(order) != 3 or len(set(order)) != 3:
                    invalid.add(rid)
                    continue
                if rid in podiums and podiums[rid] != order:
                    invalid.add(rid)
                podiums[rid] = order
    for rid in invalid:
        podiums.pop(rid, None)
    return podiums, invalid

def gz_csv(path, header, rows):
    with path.open("wb") as file:
        with gzip.GzipFile(fileobj=file, mode="wb", filename="", mtime=0) as encoded:
            with io.TextIOWrapper(encoded, encoding="utf-8", newline="") as text:
                writer = csv.DictWriter(text, fieldnames=header, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(rows)

def run():
    OUT.mkdir(parents=True, exist_ok=True)
    items, scanned, disputes, provenance, rejected = merge()
    podium, podium_invalid = published_podium()
    races, entries, results = (items[x] for x in ("races","entries","results"))
    by_race = defaultdict(list)
    for k, entry in entries.items():
        rid = norm(entry["race_id"])
        if rid in races:
            by_race[rid].append((k, entry))
    header = ["race_id","race_date","track","meeting_grade","race_type","race_no",
              "meeting_key","meeting_day_no","car_no","player_name","prefecture",
              "term","provisional_player_key","identity_status","class","style",
              "finish_position","finish_label","in_top3","result_status",
              "line_position","line_size","line_role","source_count"]
    rider_rows = []
    race_rows = []
    counters = Counter()
    unmatched_podium = []
    for rid in sorted(races, key=lambda x: (norm(races[x].get("race_date")),x)):
        race = races[rid]
        participants = sorted(by_race.get(rid, []), key=lambda x: int(x[1]["car_no"]))
        if not participants:
            counters["races_without_entries"] += 1
            continue
        results_for = {k:results[k] for k,_ in participants if k in results}
        full = len(results_for) == len(participants)
        top3 = podium.get(rid)
        if top3 and not set(top3).issubset({norm(e.get("car_no")) for _,e in participants}):
            unmatched_podium.append(rid)
            top3 = None
        # If a published top-three conflicts with the full results, do not
        # use this race for top-three rates until it is manually reconciled.
        inconsistent = bool(top3 and any(
            str(pos) != norm(results_for[k].get("finish_position"))
            for pos, car in enumerate(top3,1)
            for k,e in participants if norm(e.get("car_no")) == car and k in results_for))
        if inconsistent:
            counters["podium_result_conflict_races"] += 1
        valid_podium = bool(top3 and not inconsistent)
        if full: counters["races_with_full_results"] += 1
        if valid_podium: counters["races_with_verified_podium"] += 1
        race_rows.append({
            "race_id":rid, "race_date":race.get("race_date"),"track":race.get("track"),
            "race_type":race.get("race_type"),"meeting_grade":race.get("meeting_grade"),
            "meeting_day_no":int(rid[10:12]),"entry_rows":len(participants),
            "full_results":int(full),"verified_podium":int(valid_podium),
            "source_count":len(provenance["races:"+rid])
        })
        for k,entry in participants:
            result = results_for.get(k)
            car = norm(entry.get("car_no"))
            pos = norm(result.get("finish_position")) if result else ""
            # Official full result row is required to count as a starter.
            status = "full_result" if result else ("podium_only" if valid_podium else "unverified")
            if valid_podium and car in top3:
                expected = str(top3.index(car)+1)
                if result and pos and pos != expected:
                    status = "conflict_unverified"
                    pos = ""
                elif not result:
                    pos = expected
            elif inconsistent:
                status = "conflict_unverified"
                pos = ""
            if pos and not pos.isdigit() and any(x in pos for x in ("欠","取消","中止")):
                status = "nonstarter"
            name = norm(entry.get("player_name")).replace(" ", "")
            pref = norm(entry.get("prefecture")).replace(" ", "")
            term = norm(entry.get("term"))
            player_key = name + "|" + pref + "|" + term if all((name,pref,term)) else ""
            identity_status = "provisional_name_prefecture_term" if player_key else "unresolved"
            top3_flag = (int(pos) <= 3 if pos.isdigit()
                         else None) if status not in ("unverified","nonstarter","conflict_unverified") else None
            rider_rows.append({
                "race_id":rid, "race_date":race.get("race_date"),
                "track":race.get("track"),"meeting_grade":race.get("meeting_grade"),
                "race_type":race.get("race_type"),"race_no":race.get("race_no"),
                "meeting_key":rid[:10], "meeting_day_no":int(rid[10:12]),
                "car_no":car,"player_name":entry.get("player_name"),
                "prefecture":entry.get("prefecture"),"term":entry.get("term"),
                "provisional_player_key":player_key,"identity_status":identity_status,
                "class":entry.get("class"),"style":entry.get("style"),
                "finish_position":pos,
                "finish_label":norm(result.get("finish_text")) if result else "",
                "in_top3":"" if top3_flag is None else int(top3_flag),
                "result_status":status,"line_position":entry.get("line_position"),
                "line_size":entry.get("line_size"),"line_role":entry.get("line_role"),
                "source_count":len(provenance["entries:"+k])
            })
            counters["rider_rows"] += 1
            counters["rider_status_"+status] += 1
    race_header = ["race_id","race_date","track","meeting_grade","race_type",
                   "meeting_day_no","entry_rows","full_results","verified_podium","source_count"]
    for year, start, end in ((2025,"2025-01-01","2025-12-31"),
                             (2026,"2026-01-01","2026-10-31")):
        gz_csv(OUT/f"rider_races_{year}.csv.gz",header,
               (r for r in rider_rows if start <= r["race_date"] <= end))
        gz_csv(OUT/f"races_{year}.csv.gz",race_header,
               (r for r in race_rows if start <= r["race_date"] <= end))
    report = {
        "status":"CONSOLIDATED_SAVED_SOURCE_SUBSET_NOT_NATIONWIDE_CERTIFIED",
        "window":[START,END],"girls_excluded":True,
        "canonical_source_paths_only":True,"odds_or_models_changed":False,
        "race_ids_with_records":len(races),
        "race_ids_with_entrant_data":len(race_rows),
        "rider_race_rows":len(rider_rows),
        "unique_provisional_player_keys":len({r["provisional_player_key"] for r in rider_rows
                                                if r["provisional_player_key"]}),
        "rider_identity":"Name + prefecture + term is provisional, NOT a verified official rider ID.",
        "counters":dict(counters),
        "results_present_without_entry_match":sum(1 for k in results if k not in entries),
        "published_podiums_without_race_entry_match":len(set(podium)-set(by_race)),
        "unmatched_podium_car_ids":unmatched_podium[:100],
        "invalid_podium_race_ids":sorted(podium_invalid)[:100],
        "source_groups":scanned,
        "conflict_count":len(disputes),
        "conflicts_sample":disputes[:150],
        "archive_errors":dict(rejected),
        "limitations":[
            "All-class entry coverage is unverified; source duplicates were keyed by race_id and car_no.",
            "2026 G1/G2 standalone source could be incomplete; compare the daily-index missing-race ledger.",
            "Final-day flag cannot be inferred from observed meeting days without authoritative schedule.",
            "No start is assumed when a result row is absent; podium-only entries are not full results.",
            "No official rider ID exists in the inspected entries.csv source schema.",
            "Do not use outcomes from a future date in historical predictions."
        ]
    }
    (OUT/"summary.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",
                                    encoding="utf-8")
    print(json.dumps({k:report[k] for k in ("status","race_ids_with_entrant_data",
         "rider_race_rows","unique_provisional_player_keys","conflict_count")},ensure_ascii=False))

if __name__ == "__main__":
    run()
