"""Validate expanded connector against exact persisted counts and locked exams."""
from __future__ import annotations
import json
from source_bridge import ROOT,supervised_training_rows,public_entries,YEARS

def main():
    report=json.loads((ROOT/"research/ai_nursery/source_connection/coverage-expanded.json").read_text(encoding="utf-8"))
    public=json.loads((ROOT/"docs/ai-lab/nursery-source-connection-expanded.json").read_text(encoding="utf-8"))
    assert public["totals"]==report["totals"]
    assert report["schema"]=="nursery_git_sources_expanded_v2"
    assert report["models_retrained"] is False
    assert len(report["years"])==len(YEARS)==6
    assert sum(x["priced_and_result_labeled_races"] for x in report["years"])==report["totals"]["priced_and_labeled"]
    assert sum(x["result_labeled_without_usable_full_prices"] for x in report["years"])==report["totals"]["labeled_without_full_odds"]
    assert sum(x["new_complete_market_races_vs_previous_connector"] for x in report["years"])==report["totals"]["incremental_complete_markets"]
    assert report["totals"]["priced_and_labeled"]>=report["old_connector_complete_races"]
    assert report["totals"]["full_odds_archived"]==39934
    for row in report["years"]:
        assert row["priced_and_result_labeled_races"]+row["result_labeled_without_usable_full_prices"]==row["total_result_labeled_racecards"]
        assert row["total_result_labeled_racecards"]<=row["racecards_with_entries"]
        assert len(row["full_market_race_ids_sha256"])==64
        assert len(row["result_only_race_ids_sha256"])==64
        assert row["priced_and_result_labeled_races"]<=row["full_odds_archive_total"]
    for start,end in (
        ("2023-06-01","2023-06-07"),
        ("2023-07-01","2023-07-07"),
        ("2023-05-30","2023-06-02"),
        ("2026-06-30","2026-07-01"),
    ):
        try:
            supervised_training_rows(start,end)
        except ValueError:
            pass
        else:
            raise AssertionError("Protected exam or future data allowed "+start+" "+end)
    item=public_entries({"car_no":"1","score":"100","finish_position":"2",
                          "winning_method":"逃げ","payout":"340","result_comment":"won"})
    assert "finish_position" not in item and "winning_method" not in item and "result_comment" not in item
    assert item["score"]=="100"
    print("EXPANDED_SOURCE_BRIDGE_VALIDATED",report["totals"]["labeled_total"],
          "labelled cards",report["totals"]["priced_and_labeled"],"complete price books")

if __name__=="__main__":
    main()
