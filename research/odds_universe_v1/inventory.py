"""Inventory EVERY stored historical race source before fitting market models.

Never infer odds coverage from result/payout availability. ZIP archives are
opened and their true contents inspected, without modifying originals.
The year and class identifiers below are SOURCE INVENTORY ONLY, never features.
"""
from __future__ import annotations
import csv
import io
import json
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OUTPUT=ROOT/"data/audits/odds_universe_inventory.json"


def read_csv(fp):
    with fp.open(encoding="utf-8-sig",newline="") as f:
        return list(csv.DictReader(f))


def tally(race_rows,result_ids,source,odds_flag,totals):
    years=defaultdict(lambda:{"races":0,"race_ids":set(),"result_label":0,"odds_candidate":0})
    dup=0
    for row in race_rows:
        rid=str(row.get("race_id","")).strip()
        dt=str(row.get("race_date","")).strip()
        if not rid or len(dt)<4:continue
        year=dt[:4]
        v=years[year]
        v["races"]+=1
        if rid in v["race_ids"]:dup+=1
        v["race_ids"].add(rid)
        v["result_label"]+=int(rid in result_ids)
        v["odds_candidate"]+=int(odds_flag)
        totals["total_source_race_rows"]+=1
        totals["unique_ids"].add(rid)
    report={}
    for year,s in sorted(years.items()):
        report[year]={"races":s["races"],"unique_race_ids":len(s["race_ids"]),
                      "result_label_races":s["result_label"],
                      "races_with_odds_file_present_NOT_verified_per_race":s["odds_candidate"]}
    return report,dup


def inspect_csv_dataset(folder,totals):
    rfile=folder/"races.csv"
    if not rfile.exists():return None
    races=read_csv(rfile)
    res_ids=set()
    resfile=folder/"results.csv"
    if resfile.exists():
        for r in read_csv(resfile):
            if str(r.get("finish_position","")).strip() in {"1","2","3"}:
                res_ids.add(str(r.get("race_id","")))
    has3=bool(list(folder.glob("trifecta_final_odds*.csv")))
    has2=(folder/"trio_final_odds.csv").exists()
    by_year,dup=tally(races,res_ids,str(folder.relative_to(ROOT)),has3 and has2,totals)
    return {"source":str(folder.relative_to(ROOT)),"format":"csv",
            "has_races_csv":True,"has_entries_csv":(folder/"entries.csv").exists(),
            "has_results_csv":resfile.exists(),
            "has_trifecta_file":has3,"has_trio_file":has2,
            "per_race_odds_coverage_verified":False,
            "source_duplicate_rows":dup,
            "by_year":by_year}


def inspect_zip(zippath,totals):
    zreport={"source":str(zippath.relative_to(ROOT)),"format":"zip"}
    try:
        with zipfile.ZipFile(zippath) as z:
            names=[a for a in z.namelist() if not a.endswith("/")]
            zreport["member_names_sample"]=names[:25]
            zreport["member_count"]=len(names)
            members=[a for a in names if a.endswith("/races.csv") or a=="races.csv"]
            resultsets={}
            for name in names:
                if name.endswith("/results.csv") or name=="results.csv":
                    parent=name.rsplit("/",1)[0] if "/" in name else ""
                    rr=csv.DictReader(io.TextIOWrapper(z.open(name),encoding="utf-8-sig",newline=""))
                    resultsets[parent]={str(row.get("race_id","")) for row in rr if str(row.get("finish_position","")) in ("1","2","3")}
            zreport["results_member_count"]=len(resultsets)
            zreport["race_members"]=[]
            for name in members:
                parent=name.rsplit("/",1)[0] if "/" in name else ""
                result_ids=resultsets.get(parent,set())
                with z.open(name) as raw:
                    f=io.TextIOWrapper(raw,encoding="utf-8-sig",newline="")
                    rows=list(csv.DictReader(f))
                has3=any((x.rsplit("/",1)[0] if "/" in x else "") == parent and x.rsplit("/",1)[-1].startswith("trifecta_final_odds") for x in names)
                has2=any((x.rsplit("/",1)[0] if "/" in x else "") == parent and x.rsplit("/",1)[-1]=="trio_final_odds.csv" for x in names)
                yy,dup=tally(rows,result_ids,name,has3 and has2,totals)
                zreport["race_members"].append({"member":name,"by_year":yy,
                    "has_entries_csv":(parent+"/entries.csv" if parent else "entries.csv") in names,
                    "has_result":bool(result_ids),"has_trifecta_file":has3,
                    "has_trio_file":has2,"duplicates":dup})
            zreport["unreadable_as_races_csv"]=not bool(members)
    except Exception as exc:
        zreport["error"]=f"{type(exc).__name__}: {exc}"
    return zreport


def run():
    totals={"total_source_race_rows":0,"unique_ids":set()}
    csvdirs=set()
    for pref in ("data/a_class_12","data/a3_challenge","data/2023","data/2024",
                 "data/2025","data/2026_h1","data/2026_future_block1"):
        for p in sorted((ROOT/pref).rglob("races.csv")):
            csvdirs.add(p.parent)
    csvlist=[inspect_csv_dataset(folder,totals) for folder in sorted(csvdirs)]
    archives=[inspect_zip(p,totals) for p in sorted((ROOT/"data/grade_races").rglob("*.zip"))]
    asof=datetime.now(timezone.utc).isoformat()
    family={}
    for d in csvlist:
        prefix=d["source"].split("/")[1]
        family.setdefault(prefix,{"datasets":0,"record_rows":0,"rows_with_result_labels":0,
                                  "rows_from_datasets_with_full_odds_files_NOT_per_race":0})
        fam=family[prefix];fam["datasets"]+=1
        for v in d["by_year"].values():
            fam["record_rows"]+=v["races"]
            fam["rows_with_result_labels"]+=v["result_label_races"]
            fam["rows_from_datasets_with_full_odds_files_NOT_per_race"]+=v["races_with_odds_file_present_NOT_verified_per_race"]
    rec={
        "status":"SOURCE_INVENTORY_NOT_COMPLETE_MARKET_COVERAGE",
        "generated_at_utc":asof,
        "source_counts":{"source_race_rows_in_csv_or_zips":totals["total_source_race_rows"],
                         "distinct_race_ids_across_sources":len(totals["unique_ids"]),
                         "csv_datasets":len(csvlist),"zip_archives":len(archives)},
        "families":family,
        "csv_sources":csvlist,"grade_archives":archives,
        "coverage_warning":"Having an odds CSV/ZIP does not establish a complete full-odds market for every race; verify each key and status.",
        "missing_scope_warning":"No global all-Japan authoritative race calendar has yet been reconciled; girls and excluded historical years might be missing.",
        "forecast_rule":"Only archived odds distributions are eligible model features. These class/grade labels serve audit stratification ONLY."
    }
    OUTPUT.parent.mkdir(parents=True,exist_ok=True)
    OUTPUT.write_text(json.dumps(rec,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("INVENTORY",json.dumps({"counts":rec["source_counts"],"family":family,
       "archives":[{"name":a["source"],"count":a.get("member_count"),
                     "race_members":len(a.get("race_members",[])),"sample":a.get("member_names_sample",[])[:5]}
                     for a in archives]},ensure_ascii=False),flush=True)

if __name__=="__main__":
    run()
