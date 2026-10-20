"""Second-stage expanded-source census: exact full-market vs price-less labels.

No model training. Every count derived by reading actual archived bytes.
Original GEN-001..007 checkpoints, old exam sets and odds-only manifest remain
immutable. The older 37,441 baseline remains an independently saved comparator.
"""
from __future__ import annotations
import json
from datetime import datetime,timezone
from collections import Counter

from source_bridge import ROOT,YEARS,supervised_races_year

ROOTOUT=ROOT/"research/ai_nursery/source_connection"
WEBOUT=ROOT/"docs/ai-lab"
OLD=WEBOUT/"nursery-source-connection.json"
PUBLIC=WEBOUT/"nursery-source-connection-expanded.json"


def main():
    baseline=json.loads(OLD.read_text(encoding="utf-8"))
    baseline_by_year={int(y["year"]):int(y["complete_races"]) for y in baseline["yearly"]}
    years=[]
    totals=Counter()
    for year in YEARS:
        prices,labels,source=supervised_races_year(year)
        priced_ids={x["race_id"] for x in prices}
        label_ids={x["race_id"] for x in labels}
        assert not priced_ids&label_ids
        assert len(prices)==source["complete_market_races"]
        assert len(labels)==source["result_labeled_no_full_market_races"]
        assert source["all_labeled_cards"]==len(prices)+len(labels)
        assert all(x["race"]["race_date"][:4]==str(year) for x in prices+labels)
        before=baseline_by_year[year]
        increment=len(prices)-before
        item={
            "year":year,
            "racecards_with_entries":source["card_source_counts"]["unique_cards"],
            "full_odds_archive_total":source["odds_source_counts"]["archive_complete_races"],
            "priced_and_result_labeled_races":len(prices),
            "result_labeled_without_usable_full_prices":len(labels),
            "total_result_labeled_racecards":len(prices)+len(labels),
            "previous_priced_and_labeled_races":before,
            "new_complete_market_races_vs_previous_connector":increment,
            "racecard_sources":source["sources"],
            "source_containers_processed":source["card_source_counts"].get("source_containers",0),
            "duplicate_racecard_rows":source["card_source_counts"].get("duplicate_race_id",0),
            "invalid_entry_or_podium":source["missing_reasons"],
            "full_market_race_ids_sha256":source["complete_race_ids_sha256"],
            "result_only_race_ids_sha256":source["no_odds_labeled_ids_sha256"],
        }
        years.append(item)
        totals["racecards_with_entries"]+=item["racecards_with_entries"]
        totals["full_odds_archived"]+=item["full_odds_archive_total"]
        totals["priced_and_labeled"]+=len(prices)
        totals["labeled_without_full_odds"]+=len(labels)
        totals["labeled_total"]+=len(prices)+len(labels)
        totals["incremental_complete_markets"]+=increment
        print("EXPANDED_SOURCE_YEAR",year,"price_books",len(prices),
              "supervised_without_prices",len(labels),
              "new_complete_vs_old",increment,flush=True)
    report={
        "schema":"nursery_git_sources_expanded_v2",
        "generated_utc":datetime.now(timezone.utc).isoformat(),
        "status":"EXPANDED_CONNECTOR_AUDITED",
        "scope":"2021-01-01 to 2026-06-30 non-girls",
        "old_connector_complete_races":baseline["totals"]["joined_races"],
        "added_source_families":[
            "data/unarchived_races_v1/batch_*/races.csv.gz",
            "data/unarchived_races_v1/batch_*/entries.csv.gz"
        ],
        "original_price_data":"data/odds_only/odds_YEAR.jsonl.gz; verified SHA256 and exact 3連単+3連複 odds",
        "original_result_labels":"data/result_universe_v2/combined/results_YEAR.jsonl.gz",
        "catalogs_present_not_used_as_bettable_prices":[
            "data/nationwide_calendar/*/race_index.csv.gz: index only; not rider entries",
            "data/odds_universe_v1/**: underlying recovered 3連単/3連複 odds batches already represented in canonical odds-only export",
            "data/odds_ciphertext/**: P7M ciphertext; not usable odds without authorized decryption; no keys requested or stored",
        ],
        "totals":dict(totals),
        "years":years,
        "training_interfaces":{
            "profit_betting":"from source_bridge import joined_races_for_training; only fully priced and result-labeled",
            "two_tier_future_models":"from source_bridge import supervised_training_rows; keys full_market and result_only",
        },
        "strict_invariants":[
            "Never create synthetic prices for races without full market books",
            "An outcome-labeled race without odds cannot train the price-conditioned profit bettor",
            "Race-card features have no post-event result attributes",
            "Exact 16-digit race ID matching; deduplication and source audit",
            "Do not train on reserved 2023-06-01..07 and 2023-07-01..07 common test sets",
            "No model checkpoint updated by this audit; this is connection only",
        ],
        "models_retrained":False,
        "previous_generations_unchanged":True
    }
    ROOTOUT.mkdir(parents=True,exist_ok=True)
    (ROOTOUT/"coverage-expanded.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    PUBLIC.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("EXPANDED_CONNECTION_SUCCESS",json.dumps(report["totals"],ensure_ascii=False),flush=True)


if __name__=="__main__":
    main()
