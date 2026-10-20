"""Run full existing-repository inventory for the learner bridge; NO training."""
from __future__ import annotations
import json
from datetime import datetime,timezone
from collections import Counter

from source_bridge import ROOT,YEARS,joined_races

RESULT=ROOT/"research/ai_nursery/source_connection"
PUBLIC=ROOT/"docs/ai-lab/nursery-source-connection.json"

def run():
    years=[]
    total=Counter()
    hashes={}
    for year in YEARS:
        data,report=joined_races(year)
        if len(data)!=report["complete_races"]:
            raise AssertionError("Inconsistent per-year joined count")
        if data:
            d=[x["race"]["race_date"] for x in data]
            assert all(x[:4]==str(year) for x in d)
        years.append(report)
        hashes[str(year)]=report["joined_race_ids_sha256"]
        total["joined_races"]+=len(data)
        total["complete_market_races"]+=report["market_count"]["archive_complete_races"]
        total["racecards"]+=report["source_count"]["unique_cards"]
        total["odds_missing_on_racecards"]+=report["join_status"].get("no_complete_market",0)
        total["result_missing_on_racecards"]+=report["join_status"].get("no_result",0)
        print("SOURCE_BRIDGE_YEAR",year,
              "cards",report["source_count"]["unique_cards"],
              "odds_market",report["market_count"]["archive_complete_races"],
              "joined",len(data),
              "odds_gap",report["join_status"].get("no_complete_market",0),
              flush=True)
    report={
        "schema":"nursery_source_connection_v1",
        "status":"CONNECTED_VALIDATED_SOURCES_ONLY",
        "created_at_utc":datetime.now(timezone.utc).isoformat(),
        "repository":"yuhevita-wq/keirin-market-analysis-",
        "coverage_years":list(YEARS),
        "notes":[
          "This is exact race-id linkage of existing racecards, results and complete odds, not new source collection.",
          "The total of 39,934 complete odds races is not equal to fully joined training races.",
          "Original GEN-001..007 and locked future examinations remain unchanged.",
          "Historical odds are FINAL, never mislabeled as pre-deadline.",
          "Only completely enumerated 3連単 and 3連複 are connected. No synthetic missing markets.",
          "2021 and 2022 have grade-racecard archives only in this source loader; not all nationwide racecards.",
          "A3 source archive is missing March-April 2023; report missing coverage as-is.",
          "Do not treat the report's race-type, coverage and label metadata as future betting predictors."
        ],
        "totals":dict(total),
        "yearly":years,
        "per_year_join_sha256":hashes,
        "training_api":"from source_bridge import joined_races_for_training",
        "training_runs_performed":0,
        "model_weights_updated":False
    }
    RESULT.mkdir(parents=True,exist_ok=True)
    (RESULT/"coverage.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    PUBLIC.parent.mkdir(parents=True,exist_ok=True)
    PUBLIC.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("NURSERY_SOURCE_CONNECTION_COMPLETE",
          json.dumps({"years":len(years),**report["totals"]},ensure_ascii=False),
          flush=True)
    return report

if __name__=="__main__":
    run()
