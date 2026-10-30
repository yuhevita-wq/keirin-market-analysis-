#!/usr/bin/env python3
"""2023 W01 (Jan 01-07) RESTART: verify quoted full odds against official payout.

This is an integrity *prerequisite*, not a profit backtest.
- First read all 7-market sealed source and independently archived official paid winners.
- Compare the precise WINNING ORDERED trifecta quote with paid yen/100.
- Compare public subset on overlapping race IDs as independent secondary trace.
- Fail closed (NO purchase/no ROI) on major price discrepancy.
- Never upload decrypted odds, private keys, or plaintext price tables.
- Only aggregate statistics, hashes, and anonymized discrepancy diagnostics.
- A passing price test would not prove *pre-race* executability: source quote was
  collected retrospectively (historical "final" prices).
"""
from __future__ import annotations
import gzip,hashlib,json,math,statistics,sys,tempfile
from collections import Counter,defaultdict
from datetime import date,timedelta
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
for p in ("","research/independent_trifecta_2023","research/deletion_ai"):
    sys.path.insert(0,str(ROOT/p))
import research.deletion_ai.secure_seven_bet_bridge as bridge
import v6_settled_payout_only_2023_2026 as payout

START="2023-01-01"
END="2023-01-07"
OUTPUT=ROOT/"data/audits/v6_c1_weekly_restart_2023w01_integrity.json"
PUBLIC_ODDS=ROOT/"data/odds_only/odds_2023.jsonl.gz"
ABS_MATCH_YEN=1.1
FRACTION_MATCH=.02
MIN_CHECKED_PAYOUTS=40
REQUIRED_MATCH_SHARE=.98

def pack_amount_ratio(x):
    if not x:return None
    y=sorted(x)
    return {"count":len(y),"min":round(y[0],5),"p10":round(y[int(.10*(len(y)-1))],5),
            "median":round(statistics.median(y),5),"p90":round(y[int(.9*(len(y)-1))],5),
            "max":round(y[-1],5)}

def main():
    day_files,_=bridge.catalog(START,END)
    expected=[(date(2023,1,1)+timedelta(days=i)).isoformat() for i in range(7)]
    assert sorted(day_files)==expected,("W01 lacks authenticated daily original",sorted(day_files))
    stats=Counter()
    daily=defaultdict(Counter)
    odds={}
    # Catalog full sealed 7-market books on every Jan 1-7 day;
    # only *the winning ticket* is kept in memory for diagnosis after
    # all books have been verified, not saved or sent as artifact.
    temp,key=bridge.secure_key_temp()
    try:
        for day,files in day_files.items():
            uniq={};conflict=set()
            with tempfile.TemporaryDirectory(prefix="v6_c1_w01_verified_cipher_day_") as td:
                for path,sha in files:
                    records=bridge.decrypt_sealed(path,sha,key,td)
                    stats["authenticated_ciphertext_files"]+=1
                    stats["full_seven_bet_books_with_duplicate_retry"]+=len(records)
                    for rid,rows in records.items():
                        r=sorted(rows)
                        if rid in conflict:continue
                        if rid in uniq and uniq[rid]!=r:
                            uniq.pop(rid,None);conflict.add(rid)
                        else:uniq[rid]=r
            stats["days_decrypted"]+=1
            stats["conflicting_retry_races"]+=len(conflict)
            daily[day]["complete_sealed_races"]+=len(uniq)
            for rid,rows in sorted(uniq.items()):
                book,why=bridge.normalize_book(rid,day,rows)
                if why:
                    stats["source_excluded_"+why]+=1
                    daily[day]["source_excluded_"+why]+=1
                    continue
                stamps={row[-1] for row in rows}
                if len(stamps)!=1:
                    stats["timestamp_conflicting_within_race"]+=1
                    continue
                stamp=next(iter(stamps))
                if stamp and stamp[:10]!=day:
                    stats["timestamp_date_inconsistent"]+=1
                dist=book["odds"]["3連単"]
                inverse=sum(1/float(x) for x in dist.values())
                stats["verified_full_seven_type_books"]+=1
                daily[day]["verified_full_seven_type_books"]+=1
                odds[rid]={"day":day,
                           "3rentan":dist,
                           "stamp_day":stamp[:10] if stamp else None,
                           "market_inverse_total":inverse}
        print("W01_SEALED_ARCHIVE_VERIFIED",json.dumps({
            "days":stats["days_decrypted"],
            "books":stats["verified_full_seven_type_books"],
            "cipher_files":stats["authenticated_ciphertext_files"],
            "by_day":{k:dict(v) for k,v in sorted(daily.items())}
        },ensure_ascii=False),flush=True)
    finally:
        temp.cleanup()

    official,conflicts,source_audit=payout.load_settled_payments(set(odds))
    podium,podium_audit=bridge.verified_results(2023)
    public={}
    with gzip.open(PUBLIC_ODDS,"rt",encoding="utf-8") as fh:
        for line in fh:
            d=json.loads(line)
            rid=str(d.get("race_id",""))
            if rid in odds:
                if rid in public:raise AssertionError("duplicate public market race "+rid)
                public[rid]=d
    ratios=[]; p_ratios=[]; invs=[]
    same_combo_public_vs_vault=0
    exact_on_source=Counter()
    audit_examples=[]
    for rid,rec in odds.items():
        day=rec["day"];daily[day]["all_verified_books"]+=1
        invs.append(rec["market_inverse_total"])
        if rid not in podium:
            stats["missing_official_podium"]+=1
            continue
        winner="-".join(map(str,podium[rid]))
        if winner not in rec["3rentan"]:
            stats["winner_absent_from_complete_vault_book"]+=1
            continue
        stats["official_result_matchable"]+=1
        if rid in conflicts:
            stats["paid_conflict"]+=1;continue
        if rid not in official:
            stats["paid_missing"]+=1;continue
        paid_combo,paid_yen=official[rid]
        if tuple(map(str,paid_combo))!=tuple(map(str,podium[rid])):
            stats["official_paid_vs_official_result_disagreement"]+=1
            continue
        stats["official_paid_winner_verified"]+=1
        daily[day]["official_paid_winner_verified"]+=1
        # 100-yen normalized, exact ordered ticket only. Payout itself does
        # not appear in original odds teacher or candidate formation logic.
        saved=float(rec["3rentan"][winner])
        proxy=100*saved
        ratio=proxy/paid_yen
        ratios.append(ratio)
        diff=abs(proxy-paid_yen)
        same=diff<=ABS_MATCH_YEN
        near=diff<=max(ABS_MATCH_YEN,FRACTION_MATCH*paid_yen)
        within5=diff<=max(ABS_MATCH_YEN,.05*paid_yen)
        for label,flag in (("exact_yen",same),("within2pct",near),("within5pct",within5)):
            stats["vault_"+label]+=int(flag)
            daily[day]["vault_"+label]+=int(flag)
        if not same and len(audit_examples)<12:
            # No plaintext odds values, winner tickets, keys, or private data
            # are written to the report. Only aggregate discrepancy ratios.
            audit_examples.append({"day":day,
               "vault_odds_to_official_paid_ratio":round(ratio,4),
               "stored_odds_date_equals_race_day":rec["stamp_day"]==day,
               "overround_sum_inverse":round(rec["market_inverse_total"],5)})
        pub=public.get(rid)
        if pub:
            stats["public_overlap_races"]+=1
            book=(pub.get("odds") or {}).get("3連単",{})
            if winner in book:
                stats["public_winner_quote_present"]+=1
                p=float(book[winner]);pubproxy=100*p
                pdiff=abs(pubproxy-paid_yen)
                stats["public_exact_yen"]+=int(pdiff<=ABS_MATCH_YEN)
                stats["public_within2pct"]+=int(pdiff<=max(ABS_MATCH_YEN,FRACTION_MATCH*paid_yen))
                p_ratios.append(pubproxy/paid_yen)
                if abs(p-saved)<=1e-8:
                    same_combo_public_vs_vault+=1

    checked=stats["official_paid_winner_verified"]
    share=(stats["vault_within2pct"]/checked) if checked else 0
    viable=(checked>=MIN_CHECKED_PAYOUTS and share>=REQUIRED_MATCH_SHARE
            and stats["official_paid_vs_official_result_disagreement"]==0
            and stats["timestamp_date_inconsistent"]==0)
    # An audit can succeed computationally yet BLOCK all betting/ROI.
    status="PRICE_INTEGRITY_OK_FOR_EXPLORATORY_WEEKLY_FORMATIONS" if viable else "BLOCKED_PRICE_VS_OFFICIAL_PAYOUT"
    report={
      "week_no":1,"period":[START,END],
      "status":status,"actual_backtest_roi_computed":False,
      "do_not_buy_or_issue_profit_until_integrity_passes":not viable,
      "original_V6_and_C1_unchanged":True,
      "secrets_or_raw_quoted_odds_exported":False,
      "number_of_authenticated_files":stats["authenticated_ciphertext_files"],
      "coverage":dict(stats),
      "daily":{k:dict(v) for k,v in sorted(daily.items())},
      "official_payout_audit":source_audit,
      "official_podium_audit":podium_audit,
      "quote_to_official_payout_ratio":pack_amount_ratio(ratios),
      "public_quote_to_official_ratio":pack_amount_ratio(p_ratios),
      "market_full_inverse_odds_mass":pack_amount_ratio(invs),
      "public_winning_quote_identical_to_vault_count":same_combo_public_vs_vault,
      "anonymized_diagnostic_examples":audit_examples,
      "integrity_standard":{
         "min_matched_paid_races":MIN_CHECKED_PAYOUTS,
         "require_within_2pct_fraction":REQUIRED_MATCH_SHARE,
         "exact_yen_tolerance":ABS_MATCH_YEN,
         "must_have_no_official_result_conflicts":True,
         "must_have_same_archive_odds_date":True
      },
      "limitations":[
        "2023 Jan is part of V6 and V6-C1 training data; not forward validation.",
        "This test diagnoses quotes and official paid winner, not prospective preclosing availability.",
        "Do not treat any high odds/ROI proxy as real payout if matching fails.",
        "Invalid price archive means candidate selection cannot be trusted.",
        "Public 3ren-tan odds archive and sealed 7-bet source are separately checked."
      ]}
    OUTPUT.parent.mkdir(parents=True,exist_ok=True)
    OUTPUT.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("W01_INTEGRITY_FINAL",json.dumps({
        "period":[START,END],"status":status,"counts":dict(stats),
        "ratio":report["quote_to_official_payout_ratio"],
        "public_ratio":report["public_quote_to_official_ratio"],
        "by_day":report["daily"]
    },ensure_ascii=False),flush=True)
    if not viable:
        print("W01_FAIL_CLOSED_NO_PURCHASE_NO_ROI",flush=True)
        return 3
    return 0

if __name__=="__main__":
    raise SystemExit(main())
