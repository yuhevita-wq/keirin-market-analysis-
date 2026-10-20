"""Truthful preflight: compare public 2-pool trainer against encrypted 7-pool archive.

Counts of sealed races are LOWER BOUNDS (per-day max); distinct race IDs require
secure decryption. Never call the smaller odds_only export "full repository".
No decryption, no raw odds, no collection, no model changes.
"""
from __future__ import annotations
import json
from datetime import datetime,timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/"data/audits/deletion_full_archive_source_gap.json"

def run():
    if OUT.exists():
        raise RuntimeError("Audited source-gaps record already exists: read only; no overwrite")
    seven=json.loads((ROOT/"data/audits/odds_seven_bet_archive_count.json").read_text())
    prices=json.loads((ROOT/"data/audits/odds_only_export.json").read_text())
    prior=json.loads((ROOT/"research/deletion_ai/outputs/GEN-008-DELETION-RESET-2023/summary.json").read_text())
    continuation=json.loads((ROOT/"research/deletion_ai/outputs/GEN-008-DELETION-CONTINUE-2024/summary.json").read_text())
    assert seven["confirmed_distinct_race_lower_bound"]>100000
    assert seven["unverified_run_manifest_count"]==0
    assert seven["distinct_days"]>=2000
    assert prices["supported_odds_types"]==["3連単","3連複"]
    years={}
    for year in ("2023","2024","2025"):
        bound=seven["year"][year]["confirmed_distinct_race_lower_bound"]
        odds=prices["years"][year]["complete_races"]
        years[year]={
          "encrypted_seven_bet_distinct_races_at_least":bound,
          "unencrypted_two_bet_complete_races":odds,
          "difference_in_archive_reported_counts_lower_bound_minus_plaintext":bound-odds,
          "ratio_plaintext_count_to_sealed_lower_bound_pct":round(100*odds/bound,2),
          "warning":"This ratio is NOT exact race-ID coverage: independently assembled archives require a secure per-ID join",
        }
    years["2023"]["old_2023_study_train_and_validation_races"] = (
        prior["train_races"]+prior["validation_races"])
    years["2024"]["old_2024_study_train_races"]=continuation["train_races"]
    assert years["2023"]["old_2023_study_train_and_validation_races"] < years["2023"]["encrypted_seven_bet_distinct_races_at_least"]
    assert years["2024"]["old_2024_study_train_races"] < years["2024"]["encrypted_seven_bet_distinct_races_at_least"]
    out={
      "status":"PUBLIC_SOURCE_GAP_AUDITED",
      "at_utc":datetime.now(timezone.utc).isoformat(),
      "years":years,
      "all_years_2021_2026h1_encrypted_seven_bet_lower_bound":
          seven["confirmed_distinct_race_lower_bound"],
      "all_years_2021_2026h1_plaintext_two_bet_race_count":
          prices["unique_complete_races"],
      "source_seven_bet":"data/audits/odds_seven_bet_archive_count.json",
      "source_old_model":"data/odds_only/odds_YEAR.jsonl.gz",
      "encrypted_private_key_is_not_in_public_repository":True,
      "private_key_must_not_be_disclosed_in_chat_or_committed":True,
      "the_ciphertexts_cannot_be_read_as_market_pricing_without_RSA_private_key":True,
      "authorized_training_plan":"Execute decrypt-and-train in ephemeral runner with privately installed GitHub Actions secret; publish only aggregate coverage, aggregate ROI and model coefficients",
      "public_repo_must_never_commit_decrypted_quote_files":True,
      "no_source_collection_triggered":True,
      "caveat":"Encrypted manifests count verified complete seven-bet races but do not reveal their exact IDs; exact overlaps and usable labels are unknowable until authorized secure decryption.",
    }
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(out,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("DELETION_DATA_OMISSION_AUDIT",json.dumps({
       "years":years,
       "encrypted_total_lower_bound":out["all_years_2021_2026h1_encrypted_seven_bet_lower_bound"],
       "previous_training_scope_not_full":True},ensure_ascii=False),flush=True)

if __name__=="__main__":
    run()
