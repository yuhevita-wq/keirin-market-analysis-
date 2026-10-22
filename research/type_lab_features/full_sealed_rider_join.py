#!/usr/bin/env python3
"""Combine 134,336 verified seven-market odds with real riders and label audits BY RACE ID.

Inputs remain owner's CMS-encrypted artifacts. Outputs also sealed with original
public certificate. This step NEVER changes a retrospective feature to a
pre-race-eligible one and does not authorize model training.
"""
from __future__ import annotations
import argparse, collections, contextlib, gzip, hashlib, json, os, subprocess
import tempfile
from pathlib import Path

BASE=Path(__file__).resolve().parents[2]
CERT=BASE/"odds_market/vault_public_cert.pem"
BASELINE_RACES={2021:23797,2022:24445,2023:25085,2024:24700,2025:24372,2026:11937}
BASELINE_MATCHES={2021:23669,2022:24331,2023:24966,2024:24598,2025:24234,2026:11875}
BASELINE_7_9={2021:17968,2022:20668,2023:23138,2024:22834,2025:22752,2026:11167}

def sha256(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for chunk in iter(lambda:f.read(1048576),b""):
            h.update(chunk)
    return h.hexdigest()

@contextlib.contextmanager
def owner_key():
    pem=os.environ.pop("KEIRIN_SEVEN_BET_PRIVATE_KEY_PEM","")
    if not pem: raise RuntimeError("OWNER_RSA_KEY_MISSING")
    with tempfile.TemporaryDirectory(prefix="type_lab_owner_key_") as td:
        path=Path(td)/"key.pem"
        path.write_bytes(pem.encode("utf-8").replace(b"\r\n",b"\n"))
        path.chmod(0o600)
        pem=""
        yield path

def iter_sealed(path,key):
    cmd=["openssl","cms","-decrypt","-inform","DER","-in",str(path),
         "-recip",str(CERT),"-inkey",str(key)]
    p=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL)
    try:
        with gzip.GzipFile(fileobj=p.stdout,mode="rb") as fin:
            for ln in fin:
                if ln.strip():
                    yield json.loads(ln)
        if p.wait(timeout=120)!=0:raise RuntimeError("CMS_DECRYPT_ERROR")
    finally:
        if p.poll() is None:p.kill()
        p.wait()

class Seal:
    def __init__(self,path):self.path=Path(path);self.proc=None;self.gzip=None
    def __enter__(self):
        self.path.parent.mkdir(parents=True,exist_ok=True)
        self.proc=subprocess.Popen(["openssl","cms","-encrypt","-binary",
          "-aes-256-gcm","-stream","-outform","DER","-out",str(self.path),str(CERT)],
          stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        self.gzip=gzip.GzipFile(fileobj=self.proc.stdin,mode="wb",mtime=0)
        return self
    def add(self,x):
        self.gzip.write((json.dumps(x,separators=(",",":"),ensure_ascii=False)+"\n").encode())
    def __exit__(self,exc,_,__):
        self.gzip.close();self.proc.stdin.close()
        code=self.proc.wait(timeout=120)
        if exc is not None or code!=0:
            self.path.unlink(missing_ok=True)
            if exc is None:raise RuntimeError("CMS_OUTPUT_ENCRYPT_ERROR")
        return False

def run(year,full,feature,out):
    full=Path(full);feature=Path(feature);out=Path(out)
    out.mkdir(parents=True,exist_ok=True)
    ref=json.loads((full/"connection_report.json").read_text())
    audit=json.loads((feature/"audit_summary.json").read_text())
    if ref["year"]!=year or audit["year"]!=year:raise RuntimeError("WRONG_YEAR_ARTIFACT")
    if ref["counts"]["all_seven_market_prices_verified"]!=BASELINE_RACES[year]:
        raise RuntimeError("UNEXPECTED_FULL_ARCHIVE_COUNT")
    if ref["counts"]["odds_result_race_id_joined"]!=BASELINE_MATCHES[year]:
        raise RuntimeError("UNEXPECTED_FULL_LABEL_COUNT")
    checks={a["name"]:a["sha256"] for a in ref["encrypted_artifacts"]}
    for filename,digest in checks.items():
        if sha256(full/filename)!=digest:raise RuntimeError("FULL_ARCHIVE_DIGEST_FAILED")

    detailed=out/f"full_seven_market_with_riders_{year}.jsonl.gz.cms"
    counter=collections.Counter()
    joint_ids=set()
    with owner_key() as key:
        labels={}
        for obj in iter_sealed(full/f"verified_result_labels_{year}.jsonl.gz.cms",key):
            rid=obj["race_id"]
            if rid in labels:raise RuntimeError("DUPLICATE_RESULT_ID")
            labels[rid]=obj["finish_order"]
        features={}
        for obj in iter_sealed(feature/f"rider_settlement_audit_{year}.jsonl.gz.cms",key):
            rid=obj["race_id"]
            if rid in features:raise RuntimeError("DUPLICATE_FEATURE_RACE_ID")
            features[rid]=obj
        counter["historical_results_loaded"]=len(labels)
        counter["rider_audit_records_loaded"]=len(features)
        with Seal(detailed) as dest:
            for obj in iter_sealed(full/f"all_seven_bet_odds_{year}.jsonl.gz.cms",key):
                rid=obj["race_id"]
                if rid in joint_ids:raise RuntimeError("DUPLICATE_SEALED_MARKET_ID")
                joint_ids.add(rid)
                counter["all_seven_market_books_visited"]+=1
                n=obj["n_cars"]
                order=labels.get(rid)
                riders=features.get(rid)
                if order is not None:counter["market_and_result_connected"]+=1
                supported=n in (7,9)
                if supported and order is not None:counter["model_7_9_market_result"]+=1
                roster=(riders is not None and bool(riders["complete_7_9_roster"])
                        and riders["n_entry_cars"]==n)
                if roster and supported and order is not None:
                    counter["market_result_and_full_roster_connected"]+=1
                unsafe=list((riders or {}).get("settlement_quarantine",[]))
                if riders is not None and riders.get("result_archive_order") != order:
                    unsafe.append("RESULT_ARCHIVE_ORDER_DISAGREEMENT")
                    counter["cross_archive_result_label_conflicts"]+=1
                if unsafe:counter["payout_or_podium_suspect"]+=1
                payout=bool((riders or {}).get("distinct_trifecta_winner_count",0))
                if roster and order is not None and payout and not unsafe:
                    counter["rider_result_with_single_payout_code_preliminary"]+=1
                asof=bool((riders or {}).get("all_values_asof_pre_race",False))
                if roster and order is not None and asof:
                    counter["pre_race_timestamp_verified_with_rider_and_result"]+=1
                if not supported:state="TYPE_LAB_NOT_5_6_8_CARS"
                elif order is None:state="MISSING_VERIFIED_PODIUM"
                elif unsafe:state="SETTLEMENT_QUARANTINE"
                elif not roster:state="MISSING_OR_CONFLICTING_RIDER_ROSTER"
                elif not payout:state="NO_INDEPENDENT_TRIFECTA_PAYOUT_VERIFICATION"
                elif not asof:state="RETROSPECTIVE_RIDER_VALUES"
                else:state="PROVISIONAL_UNDER_REVIEW_188_TIES"
                counter["state_"+state]+=1
                dest.add({
                    "race_id":rid,"race_date":obj["race_date"],"race_type":obj["race_type"],
                    "n_cars":n,"price_phase":obj["price_phase"],
                    "all_markets":obj["markets"],"actual_order_label_not_feature":order,
                    "rider_entries":(riders or {}).get("riders"),
                    "race_card_metadata":(riders or {}).get("source_race"),
                    "settlement_quarantine_evidence":unsafe,
                    "payout_verification_present":payout,
                    "all_sources_timestamped_pre_race":asof,
                    "feature_origin":("stored_historical_racecard" if riders else None),
                    "status":state,
                    "approved_for_training":False
                })
    if len(joint_ids)!=BASELINE_RACES[year]:raise RuntimeError("INCOMPLETE_MARKET_JOIN")
    if counter["market_and_result_connected"]!=BASELINE_MATCHES[year]:
        raise RuntimeError("RESULT_MATCH_COUNT_MISMATCH")
    if counter["model_7_9_market_result"]!=BASELINE_7_9[year]:
        raise RuntimeError("CAR_COUNT_CHANGED_FROM_FROZEN_RUN")
    summary={
       "schema":"type_lab_full_sealed_market_actual_rider_join_v1",
       "year":year,"original_full_archive":"2021-2026H1 ALL 2579 files",
       "counts":dict(sorted(counter.items())),
       "encrypted_output":detailed.name,
       "encrypted_output_sha256":sha256(detailed),
       "full_archive_source_commit":ref["source_commit"],
       "rider_source_audit":audit["schema"],
       "settlement_188_ids_completely_recovered":False,
       "independent_188_tie_exclusion_required_before_training":True,
       "rider_features_asof_timestamp_enforced":True,
       "approved_for_training":False,
       "notes":["Full seven-market row attached by exact race_id; no 42,975 plaintext substitution.",
                "Results always separate from prediction features; payout classification is quarantine evidence.",
                "Pre-race feature eligibility requires source observation captured before official deadline.",
                "No historical row is deemed clean solely because no payout evidence was available."]
    }
    (out/"full_join_summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n")
    print("TYPE_LAB_FULL_RIDER_JOIN",json.dumps({
         "year":year,"all_races":counter["all_seven_market_books_visited"],
         "outcomes":counter["market_and_result_connected"],
         "seven_nine":counter["model_7_9_market_result"],
         "rider_connected":counter["market_result_and_full_roster_connected"],
         "payout_suspect":counter["payout_or_podium_suspect"],
         "strict_pre_race":counter["pre_race_timestamp_verified_with_rider_and_result"]}),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--year",type=int,required=True,choices=list(BASELINE_RACES))
    p.add_argument("--full",required=True);p.add_argument("--feature",required=True)
    p.add_argument("--output",required=True)
    args=p.parse_args();run(args.year,args.full,args.feature,args.output)
