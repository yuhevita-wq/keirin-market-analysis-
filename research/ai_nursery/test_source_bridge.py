"""No results leak, exact ID joins, all-years coverage and frozen exams."""
from __future__ import annotations
import json
from collections import Counter
from datetime import date

from source_bridge import (
    ROOT,YEARS,MARKETS,permitted_race_date,public_entries,joined_races_for_training,
)
from train_week import eligible

def check():
    assert permitted_race_date("2023-04-06",2023)
    assert not permitted_race_date("2023-04-06",2024)
    assert not permitted_race_date("2026-07-01",2026)
    assert not permitted_race_date("2023-02-31",2023)
    assert public_entries({
        "race_id":"123","car_no":"1","score":"102.3",
        "finish_position":"1","result_comment":"won","winning_method":"捲り",
        "raw_row_json":"PODIUM","source_url":"https://example.com"
    })=={"car_no":"1","score":"102.3"}
    # A complete 5-car both-market book must pass existing learner eligibility.
    from itertools import permutations,combinations
    cars=range(1,6)
    record={
      "entries":{c:{"car_no":str(c),"score":str(90+c)} for c in cars},
      "outcome":(3,1,4),
      "odds":{
          "3連単":{"-".join(map(str,p)):15.0 for p in permutations(cars,3)},
          "3連複":{"=".join(map(str,p)):5.0 for p in combinations(cars,3)}
      }
    }
    assert eligible(record)
    record["odds"]["3連単"].pop("1-2-3")
    assert not eligible(record)
    for d1,d2 in (("2023-06-01","2023-06-03"),("2023-06-06","2023-06-09"),
                  ("2023-07-01","2023-07-07"),("2023-06-30","2023-07-01")):
        try:
            joined_races_for_training(d1,d2)
        except ValueError:
            pass
        else:
            raise AssertionError("Training allowed to see protected exam "+d1+".."+d2)
    for day in ("2026-07-02","2027-01-01"):
        try:joined_races_for_training("2026-06-30",day)
        except ValueError:pass
        else:raise AssertionError("Prohibited future date "+day)

    p=ROOT/"research/ai_nursery/source_connection/coverage.json"
    a=json.loads(p.read_text(encoding="utf-8"))
    b=json.loads((ROOT/"docs/ai-lab/nursery-source-connection.json").read_text(encoding="utf-8"))
    assert a["status"]==b["status"]=="CONNECTED_VALIDATED_SOURCES_ONLY"
    assert a["per_year_join_sha256"]==b["per_year_join_sha256"]
    assert a["training_runs_performed"]==0 and a["model_weights_updated"] is False
    assert a["coverage_years"]==list(YEARS)
    archive=json.loads((ROOT/"data/audits/odds_only_export.json").read_text(encoding="utf-8"))
    assert len(a["yearly"])==len(YEARS)
    assert sum(r["complete_races"] for r in a["yearly"])==a["totals"]["joined_races"]
    assert sum(r["market_count"]["archive_complete_races"] for r in a["yearly"])==archive["unique_complete_races"]==39934
    assert a["totals"]["complete_market_races"]==39934
    assert a["totals"]["joined_races"]>0
    assert a["totals"]["odds_missing_on_racecards"]>=0
    for r in a["yearly"]:
        assert r["market_count"]["hash_verified"] is True
        assert r["complete_races"]<=r["source_count"]["unique_cards"]
        assert r["complete_races"]<=r["market_count"]["matched_candidate_ids"]
        assert r["complete_races"]<=archive["years"][str(r["year"])]["complete_races"]
        assert len(r["joined_race_ids_sha256"])==64
        assert r["price_phase"].startswith("FINAL")
        assert r["result_used_only_as_label"] is True
    print("NURSERY_SOURCE_BRIDGE_TEST_PASS",a["totals"]["joined_races"],"races linked")

if __name__=="__main__":
    check()
