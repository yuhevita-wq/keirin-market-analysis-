#!/usr/bin/env python3
"""Classify historical p1/p3 using the EXISTING, unmodified Type Lab A-F rule.

Canonical rule: nyanta-kun/kiseki keirin/src/type_lab.py
Blob SHA 50a4bd2c2c1861702e13cbd46bc1f38b9b065d85
axis_sum = top two *raw* p3, threshold 1.44;
arare = line cardinality + lead-behind + style + event day + point comparison.
Never synthesize missing racecard metadata. Missing any verified as-of
arare input produces A/B/C or D/E/F *candidate family*, NOT a fake type.

This script classifies both:
- full-refit probabilities on all 80,882, explicitly in-sample
- frozen historical heldout probabilities on 33,487 races, true OOS
All per-race details are owner CMS-encrypted. Plain report only aggregates.
"""
from __future__ import annotations
import argparse, collections, contextlib, gzip, hashlib, json, math, os
import subprocess, tempfile
from pathlib import Path
from datetime import datetime

ORIGINAL_SOURCE="https://github.com/nyanta-kun/kiseki/blob/main/keirin/src/type_lab.py"
ORIGINAL_BLOB="50a4bd2c2c1861702e13cbd46bc1f38b9b065d85"
ROOT=Path(__file__).resolve().parents[2]
CERT=ROOT/"odds_market/vault_public_cert.pem"
EXPECTED_FULL=80882
EXPECTED_HOLDOUT=33487
AXIS_SUM_FIRM=1.44
BEHIND_MID=11.0

@contextlib.contextmanager
def secret_key():
    secret=os.environ.pop("KEIRIN_SEVEN_BET_PRIVATE_KEY_PEM","")
    if not secret:raise RuntimeError("OWNER_KEY_REQUIRED_FOR_ORIGINAL_CMS_ARTIFACTS")
    with tempfile.TemporaryDirectory(prefix="type_lab_af_key_") as root:
        path=Path(root)/"key.pem"
        path.write_bytes(secret.encode("utf8").replace(b"\r\n",b"\n"))
        path.chmod(0o600)
        secret=""
        yield path

def stream_encrypted(path,key):
    cmd=["openssl","cms","-decrypt","-inform","DER","-in",str(path),
          "-recip",str(CERT),"-inkey",str(key)]
    p=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL)
    try:
        with gzip.GzipFile(fileobj=p.stdout,mode="rb") as fd:
            for ln in fd:
                if ln.strip():yield json.loads(ln)
        if p.wait(timeout=120):raise RuntimeError("SOURCE_CMS_DECRYPT_VERIFY_ERROR")
    finally:
        if p.poll() is None:p.kill()
        p.wait()

class Sealed:
    def __init__(self,file):self.path=Path(file)
    def __enter__(self):
        self.path.parent.mkdir(parents=True,exist_ok=True)
        self.p=subprocess.Popen(["openssl","cms","-encrypt","-binary","-aes-256-gcm",
            "-stream","-outform","DER","-out",str(self.path),str(CERT)],
            stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        self.f=gzip.GzipFile(fileobj=self.p.stdin,mode="wb",mtime=0)
        self.n=0
        return self
    def write(self,v):
        self.f.write((json.dumps(v,separators=(",",":"),ensure_ascii=False,allow_nan=False)+"\n").encode())
        self.n+=1
    def __exit__(self,err,_,__):
        self.f.close();self.p.stdin.close()
        rc=self.p.wait(timeout=120)
        if err is not None or rc:
            self.path.unlink(missing_ok=True)
            if err is None:raise RuntimeError("CMS_OUTPUT_FAILED")
        return False

def sha256(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""):h.update(block)
    return h.hexdigest()

def type_label_of(axis_sum,arare):
    """VERBATIM behavior of canonical keirin/src/type_lab.py:type_label_of."""
    firm=float(axis_sum)>=AXIS_SUM_FIRM
    if int(arare)<=-1:return "A" if firm else "D"
    if int(arare)==0:return "B" if firm else "E"
    return "C" if firm else "F"

def safe_float(v):
    if v is None or isinstance(v,bool):return None
    try:f=float(v)
    except (TypeError,ValueError):return None
    return f if math.isfinite(f) else None

def certified_arare(extra,topcar,allcars):
    """Only original inputs observed BEFORE CLOSE; otherwise None.
    The upstream classifier uses default values for absences in live data.
    Historical backfill must NOT use those defaults, since the true value
    could have changed after the race.
    """
    if not extra or not isinstance(extra,dict):return None,"NO_PINNED_HISTORICAL_RACECARD"
    if extra.get("original_classifier_fields_verified_asof") is not True:
        return None,"NO_ASOF_VERIFIED_CLASSIFIER_FIELDS"
    # Validated by source collector; we still check all mandatory attributes.
    group=extra.get("line_group")
    position=extra.get("line_pos")
    style=extra.get("style")
    point=extra.get("race_point")
    behind=extra.get("behind_pct")
    di=extra.get("day_index")
    if not all(isinstance(o,dict) for o in (group,position,style,point,behind)):
        return None,"MISSING_LINE_STYLE_POINT_BEHIND_MAPS"
    cars={str(x) for x in allcars}
    for obj in (group,position,style,point,behind):
        if set(obj)!=cars:return None,"INCOMPLETE_FIELD_CAR_COVERAGE"
    if not isinstance(di,int) or not (1<=di<=6):
        return None,"MISSING_OR_INVALID_EVENT_DAY_INDEX"
    for car in cars:
        b=safe_float(behind[car])
        if safe_float(point[car]) is None or b is None or not 0<=b<=100:
            return None,"MISSING_BETTING_PRICE_OR_BEHIND"
        if not str(style[car]).strip():return None,"MISSING_STYLE"
    g=group[str(topcar)]
    if g is None or str(g) in ("","0"):
        size=1;lead=None;second=None
    else:
        members=[c for c in allcars if group[str(c)]==g]
        size=len(members)
        leads=[c for c in members if str(position[str(c)]) in ("1","1.0")]
        seconds=[c for c in members if str(position[str(c)]) in ("2","2.0")]
        if len(leads)!=1 or (len(members)>=2 and len(seconds)!=1):
            return None,"MISSING_OR_AMBIGUOUS_LINE_POSITION"
        lead=leads[0];second=seconds[0] if seconds else None
    result=1 if size==2 else (-1 if size>=4 else 0)
    if lead is not None:
        result+=(-1 if safe_float(behind[str(lead)])>=BEHIND_MID else 1)
        result+=(2 if str(style[str(lead)])=="追" else 0)
    result+=di-2
    if lead is not None and second is not None:
        result+=int(safe_float(point[str(second)])>safe_float(point[str(lead)]))
    return result,None

def classify_row(item,metadata=None):
    raceid=str(item["race_id"]);day=str(item["race_date"])
    cars=list(range(1,int(item["n_cars"])+1))
    p1={int(k):float(v) for k,v in item["p1"].items()}
    p3={int(k):float(v) for k,v in item["p3"].items()}
    if len(cars) not in (7,9) or set(p1)!=set(cars) or set(p3)!=set(cars):
        raise RuntimeError("INCOMPLETE_PROBABILITY_FIELD")
    if any(not (math.isfinite(v) and 0<=v<=1) for v in list(p1.values())+list(p3.values())):
        raise RuntimeError("INVALID_PROBABILITY")
    if abs(sum(p1.values())-1)>1e-8 or abs(sum(p3.values())-3)>1e-8:
        raise RuntimeError("PROBABILITY_MARGINAL_TOTAL_INCONSISTENT")
    if any(p3[c]<p1[c]-1e-8 for c in cars):
        raise RuntimeError("P3_LESS_THAN_P1")
    order=sorted(cars,key=lambda c:(-p3[c],c))
    ax1,ax2=order[:2]
    axis_sum=p3[ax1]+p3[ax2]
    family="ABC" if axis_sum>=AXIS_SUM_FIRM else "DEF"
    arare,block=certified_arare(metadata,ax1,cars)
    result=type_label_of(axis_sum,arare) if arare is not None else None
    return {
      "race_id":raceid,"race_date":day,"n_cars":len(cars),
      "top2_cars_by_raw_p3":[ax1,ax2],
      "axis_sum_raw_p3":round(axis_sum,12),
      "axis_family":family,"arare":arare,
      "type_label":result,"type_resolved":result is not None,
      "missing_reason":block,"reconstructed_from_complete_live_asof_inputs":result is not None
    }

def run(input_dir,output_dir,metadata_path=None):
    output_dir.mkdir(parents=True,exist_ok=True)
    input_dir=Path(input_dir)
    metadata={}
    summary={
        "schema":"type_lab_original_af_strict_no_missing_values_v1",
        "canonical_classifier":ORIGINAL_SOURCE,
        "canonical_classifier_blob_sha":ORIGINAL_BLOB,
        "unchanged_thresholds":{"axis_sum":AXIS_SUM_FIRM,"lead_behind_pct":BEHIND_MID},
        "arare_formula":"1 if top-axis line length=2, -1 if >=4; +(-1 if lead behind>=11 else +1) +2 if lead style=追; +(event day-2); +1 if second score>lead score",
        "model_probability_provenance":{
            "production":"in-sample probability predictions, 2021-2026H1, not an unbiased evaluation",
            "test":"frozen trained through 2024H1, 2024H2-calibrated predictions for 2025-2026H1"
        },
        "type_production_accuracy_out_of_sample_claimed":False,
        "full_abcdef_no_missing_source_claimed":False,
        "coverage":{},
        "independently_tested_axis_pair_podium_rate":{},
        "unresolved_reason_counts":{},
        "source_metadata_unavailable_retroactive_not_imputed":True
    }
    with secret_key() as key:
        if metadata_path:
            for row in stream_encrypted(metadata_path,key):
                rid=str(row["race_id"])
                if rid in metadata:raise RuntimeError("DUPLICATE_METADATA_ID")
                metadata[rid]=row
        inp=[
            ("full_history_in_sample","production_in_sample_all_80882_p1p3.jsonl.gz.cms",EXPECTED_FULL),
            ("frozen_2025_2026_heldout","heldout_test_predictions.jsonl.gz.cms",EXPECTED_HOLDOUT)]
        for name,basename,n_expected in inp:
            f=input_dir/basename
            if not f.exists():raise RuntimeError("MISSING_PRODUCTION_OR_OOS_CMS_"+basename)
            counters=collections.Counter();podium_seen=0;pair_hits=0;weighted=collections.Counter()
            group_total=collections.Counter();group_hits=collections.Counter()
            out=output_dir/(name+"_six_type_classification.jsonl.gz.cms")
            with Sealed(out) as writer:
                for row in stream_encrypted(f,key):
                    item=classify_row(row,metadata.get(str(row["race_id"])))
                    y=row.get("target_finish_order_kept_separate_from_features")
                    if y is not None:
                        podium_seen+=1
                        group_total[item["axis_family"]]+=1
                        if set(item["top2_cars_by_raw_p3"]).issubset(set(y)):
                            pair_hits+=1
                            group_hits[item["axis_family"]]+=1
                    if item["type_label"] is not None:counters["type_"+item["type_label"]]+=1
                    else:counters["unresolved_full_type"]+=1
                    counters["family_"+item["axis_family"]]+=1
                    counters["n"+str(item["n_cars"])]+=1
                    if item["missing_reason"]:weighted[item["missing_reason"]]+=1
                    writer.write({
                        **item,"model_sample_status":name,
                        "p1_p3_probability_vector_not_republished":True,
                        "type_is_original":True,
                        "unseen_target_podium_label_used_for_type":False
                    })
            if writer.n!=n_expected:raise RuntimeError("HISTORICAL_CLASSIFICATION_COUNT_MISMATCH_"+name)
            summary["coverage"][name]={
                "race_count":writer.n,"candidate_firm_A_B_C":counters["family_ABC"],
                "candidate_mixed_D_E_F":counters["family_DEF"],
                "exact_labels":{k:counters["type_"+k] for k in "ABCDEF"},
                "six_type_full_confirmed":sum(counters["type_"+k] for k in "ABCDEF"),
                "six_type_blocked_missing_asof_inputs":counters["unresolved_full_type"],
                "seven_cars":counters["n7"],"nine_cars":counters["n9"],
                "encrypted_file":out.name,"encrypted_sha256":sha256(out)
            }
            if podium_seen:
                summary["independently_tested_axis_pair_podium_rate"]={
                    "samples":podium_seen,"both_p3_top2_podium_hits":pair_hits,
                    "frequency":pair_hits/podium_seen,
                    "by_original_axis_family":{
                        grp:{"n":group_total[grp],"top2_both_in_podium":group_hits[grp],
                             "empirical_rate":group_hits[grp]/group_total[grp]
                             if group_total[grp] else None}
                        for grp in ("ABC","DEF")}
                }
            summary["unresolved_reason_counts"][name]=dict(weighted)
            print("EXACT_TYPE_LAB_A_F_COVERAGE",json.dumps({
                "sample":name,"races":writer.n,
                "family_ABC":counters["family_ABC"],
                "family_DEF":counters["family_DEF"],
                "full_A_F":sum(counters["type_"+k] for k in "ABCDEF"),
                "unresolved":counters["unresolved_full_type"],
                "axis_pair_podium_hits_in_heldout":pair_hits if podium_seen else None,
                "axis_pair_podium_hits_ABC":group_hits["ABC"] if podium_seen else None,
                "axis_pair_podium_hits_DEF":group_hits["DEF"] if podium_seen else None
            }),flush=True)
    (output_dir/"type_lab_af_strict_classification_report.json").write_text(
        json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    print("TYPE_LAB_A_F_AUDIT_DONE",json.dumps({
        "full_history":summary["coverage"]["full_history_in_sample"]["race_count"],
        "heldout":summary["coverage"]["frozen_2025_2026_heldout"]["race_count"],
        "actual_six_type_labels_full":
            summary["coverage"]["full_history_in_sample"]["six_type_full_confirmed"],
        "never_imputed_missing_line_attributes":True
    }),flush=True)

def selftest():
    # Both exact original boundaries and the no-missing-attributes rule.
    assert [type_label_of(1.5,r) for r in (-2,0,3)]==list("ABC")
    assert [type_label_of(1.4,r) for r in (-2,0,3)]==list("DEF")
    assert type_label_of(1.44,0)=="B" and type_label_of(1.439999,0)=="E"
    assert [type_label_of(1.44,r) for r in (-1,0,1)]==list("ABC")
    p1={str(i):(1/7) for i in range(1,8)}
    p3={str(i):(3/7) for i in range(1,8)}
    row={"race_id":"0120240101010001","race_date":"2024-01-01","n_cars":7,
         "p1":p1,"p3":p3}
    got=classify_row(row)
    assert got["type_label"] is None and got["axis_family"]=="DEF"
    assert got["missing_reason"]=="NO_PINNED_HISTORICAL_RACECARD"
    metadata={
        "original_classifier_fields_verified_asof":True,
        "line_group":{"1":1,"2":1,"3":2,"4":2,"5":3,"6":3,"7":0},
        "line_pos":{"1":1,"2":2,"3":1,"4":2,"5":1,"6":2,"7":0},
        "style":{str(i):"逃" for i in range(1,8)},
        "race_point":{str(i):100. for i in range(1,8)},
        "behind_pct":{str(i):5. for i in range(1,8)},
        "day_index":2}
    yes=classify_row(row,metadata)
    assert yes["arare"]==2 and yes["type_label"]=="F"
    metadata["behind_pct"]["1"]=12
    no=classify_row(row,metadata)
    assert no["arare"]==0 and no["type_label"]=="E"
    print("ORIGINAL_TYPE_LAB_A_F_STRICT_SELFTEST_OK",flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--selftest",action="store_true")
    p.add_argument("--input")
    p.add_argument("--output")
    p.add_argument("--asof-metadata",help="optional owner-sealed certified as-of line/rider metadata")
    args=p.parse_args()
    if args.selftest:selftest()
    else:
        if not args.input or not args.output:p.error("require --input, --output")
        run(Path(args.input),Path(args.output),args.asof_metadata)
