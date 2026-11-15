#!/usr/bin/env python3
"""Chronological, no-hindsight rider ability feature reconstruction from all sealed years.

Historical odds/labels are only used for race IDs/settled past outcomes.
Never consume archived retrospective score/win_rate/age as prediction features.
Never include same-calendar-day results when constructing current-day features.
Selection: 7/9-car full independent roster with stable profile identity.
All result labels remain in a separately sealed training-label file.
This does NOT assert p1/p3 probability calibration or permission to place bets.
"""
from __future__ import annotations
import argparse, collections, contextlib, datetime as dt, gzip, hashlib
import json, math, os, re, subprocess, tempfile, unicodedata
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

BASE=Path(__file__).resolve().parents[2]
CERT=BASE/"odds_market/vault_public_cert.pem"
YEARS=tuple(range(2021,2027))
EXPECTED_FULL={2021:23797,2022:24445,2023:25085,2024:24700,2025:24372,2026:11937}
EXPECTED_TIES={2021:1,2022:6,2023:43,2024:58,2025:60,2026:20}
EXPECTED_RESULT={2021:23669,2022:24331,2023:24966,2024:24598,2025:24234,2026:11875}
MARKER="MULTIPLE_OFFICIAL_TRIFECTA_WINNERS"
PROFILE_RE=re.compile(r"^[a-zA-Z][\w+.-]*://")
MIN_HIST=5

def file_hash(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as fh:
        for block in iter(lambda: fh.read(1024*1024),b""):h.update(block)
    return h.hexdigest()

@contextlib.contextmanager
def private_key():
    secret=os.environ.pop("KEIRIN_SEVEN_BET_PRIVATE_KEY_PEM","")
    if not secret:raise RuntimeError("OWNER_SECRET_NOT_AVAILABLE")
    with tempfile.TemporaryDirectory(prefix="keirin_chronological_key_") as temp:
        file=Path(temp)/"key.pem"
        file.write_bytes(secret.encode("utf-8").replace(b"\r\n",b"\n"))
        file.chmod(0o600)
        secret=""
        yield file

def verified_stream(path,key):
    command=["openssl","cms","-decrypt","-inform","DER","-in",str(path),
             "-recip",str(CERT),"-inkey",str(key)]
    process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL)
    try:
        with gzip.GzipFile(fileobj=process.stdout,mode="rb") as fh:
            for ln in fh:
                if ln.strip():yield json.loads(ln)
        if process.wait(timeout=120):raise RuntimeError("INPUT_CMS_AUTHENTICATION_FAILED")
    finally:
        if process.poll() is None:process.kill()
        process.wait()

class SealedWriter:
    def __init__(self,path):self.path=Path(path)
    def __enter__(self):
        self.path.parent.mkdir(parents=True,exist_ok=True)
        self.p=subprocess.Popen(["openssl","cms","-encrypt","-binary","-aes-256-gcm",
           "-stream","-outform","DER","-out",str(self.path),str(CERT)],
           stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        self.gzip=gzip.GzipFile(fileobj=self.p.stdin,mode="wb",mtime=0)
        self.count=0
        return self
    def write(self,row):
        self.gzip.write((json.dumps(row,separators=(",",":"),ensure_ascii=False)+"\n").encode())
        self.count+=1
    def __exit__(self,kind,val,tb):
        self.gzip.close();self.p.stdin.close()
        ret=self.p.wait(timeout=120)
        if kind is not None or ret:
            self.path.unlink(missing_ok=True)
            if kind is None:raise RuntimeError("OUTPUT_CMS_ENCRYPTION_FAILED")
        return False

def stable_player_id(value, attrs=None):
    """Exact stable profile URL, or explicit numeric ID. No weak name-only join."""
    s=str(value or "").strip()
    if not s:return None
    if PROFILE_RE.match(s):
        parts=urlsplit(s)
        if not parts.netloc or not parts.path.strip("/"):return None
        # Preserve query-based rider IDs; dropping a player_id query could merge riders.
        return urlunsplit((parts.scheme.lower(),parts.netloc.lower(),parts.path.rstrip("/"),parts.query,""))
    if s.isascii() and s.isdigit() and len(s)>=5:
        return "numeric-rider-id:"+s
    # Archive player_profile is NOT a player identifier: it encodes region/age/term,
    # e.g. "千 葉/27/119". Use verified static name+term+prefecture surrogate.
    # Do not merge names alone. Independently quarantine temporal ID conflicts.
    attrs=attrs or {}
    clean=lambda value: "".join(unicodedata.normalize("NFKC",str(value or "")).split())
    name=clean(attrs.get("player_name"))
    pref=clean(attrs.get("prefecture"))
    term=clean(attrs.get("term"))
    if name and pref and term.isascii() and term.isdigit() and len(name)>=2:
        seed="type-lab-surrogate-v1|"+name+"|"+pref+"|"+term
        return "surrogate:"+hashlib.sha256(seed.encode("utf-8")).hexdigest()
    return None

def read_full(incoming,key,counts):
    """Use EVERY FULL-market source artifact. Payload discarded after parsing race identity."""
    records=[];seen=set()
    ties=collections.Counter()
    identity_profiles=collections.Counter()
    age_consistency=collections.defaultdict(list)
    for year in YEARS:
        root=incoming/f"type-lab-134336-full-market-rider-join-{year}"
        name=root/f"full_seven_market_with_riders_{year}.jsonl.gz.cms"
        summary=root/"full_join_summary.json"
        if not name.exists() or not summary.exists():
            raise RuntimeError(f"MISSING_FULL_MARKET_ARTIFACT_{year}")
        report=json.loads(summary.read_text(encoding="utf-8"))
        if report.get("year")!=year or file_hash(name)!=report["encrypted_output_sha256"]:
            raise RuntimeError("SOURCE_JOIN_SHA256_INTEGRITY_ERROR")
        n=0;nr=0
        for data in verified_stream(name,key):
            n+=1
            rid=str(data["race_id"])
            if rid in seen:raise RuntimeError("DUPLICATE_FULL_MARKET_RACE_ID")
            seen.add(rid)
            label=data.get("actual_order_label_not_feature")
            if label is not None:nr+=1
            flagged=set(data.get("settlement_quarantine_evidence") or ())
            if MARKER in flagged and label is not None:ties[year]+=1
            when=dt.date.fromisoformat(str(data["race_date"]))
            if when.year!=year or (year==2026 and when>dt.date(2026,6,30)):
                raise RuntimeError("RACE_DATE_OUTSIDE_SOURCE_SCOPE")
            cars=data.get("rider_entries") or []
            if not isinstance(cars,list):cars=[]
            if cars:counts["any_rider_rows_in_full_archive"]+=1
            if len(cars)==int(data["n_cars"]):counts["roster_length_matches_field"]+=1
            players={}
            ids=set()
            conflict=False
            for row in cars:
                car=row.get("car")
                attrs=row.get("values") or {}
                profile=str(attrs.get("player_profile") or "").strip()
                if not profile: identity_profiles["blank"]+=1
                elif PROFILE_RE.match(profile):identity_profiles["absolute_url"]+=1
                elif profile.startswith("/"): identity_profiles["relative_path"]+=1
                elif profile.isascii() and profile.isdigit(): identity_profiles["digits"]+=1
                else: identity_profiles["other_non_url"]+=1
                keyplayer=stable_player_id(profile,attrs)
                if keyplayer and keyplayer.startswith("surrogate:"):
                    counts["individual_rider_rows_using_surrogate"]+=1
                    age=str(attrs.get("age") or "").strip()
                    if age.isascii() and age.isdigit() and 17<=int(age)<=90:
                        age_consistency[keyplayer].append((when,int(age)))
                if keyplayer:counts["individual_rider_rows_with_stable_id"]+=1
                else:counts["individual_rider_rows_without_stable_id"]+=1
                if (not isinstance(car,int) or not keyplayer or keyplayer in ids
                    or row.get("conflicting_fields")):
                    conflict=True
                ids.add(keyplayer)
                players[car]=keyplayer
            supported=int(data["n_cars"]) in (7,9)
            full_roster=(supported and not conflict and
                         set(players)==set(range(1,int(data["n_cars"])+1)))
            if supported and label is not None and full_roster:
                counts["full_roster_named_and_identified"]+=1
            if supported and label is not None and not full_roster:
                counts["no_stable_complete_identity"]+=1
            if MARKER in flagged:
                counts["verified_tie_excluded"]+=1
            elif flagged:
                counts["other_settlement_suspect"]+=1
            if label is None:counts["missing_result"]+=1
            if supported:counts["7_9_car_market_races"]+=1
            # Only data from matched complete observed races can become training examples.
            if supported and label is not None and full_roster and not flagged:
                podium=tuple(int(x) for x in label)
                if len(podium)!=3 or len(set(podium))!=3 or not set(podium)<=set(players):
                    raise RuntimeError("INVALID_LABEL_OR_ROSTER")
                records.append({"rid":rid,"day":when,"p":players,"podium":podium,
                                "n":int(data["n_cars"]),"year":year})
                counts["records_eligible_for_temporal_reconstruction"]+=1
            elif supported and label is not None and full_roster and flagged:
                counts["full_roster_with_settlement_quarantine"]+=1
        print("TEMPORAL_ID_DIAGNOSTIC",json.dumps({
            "year":year,"full_archive_markets":n,
            "rosters_length_matched_cumulative":counts["roster_length_matches_field"],
            "with_stable_player_ids":counts["individual_rider_rows_with_stable_id"],
            "without_stable_player_ids":counts["individual_rider_rows_without_stable_id"],
            "profile_shapes":dict(identity_profiles)
        }),flush=True)
        if n!=EXPECTED_FULL[year] or nr!=EXPECTED_RESULT[year]:
            raise RuntimeError("INCOMPLETE_YEAR_FULL_ARCHIVE_DATASET")
        if ties[year]!=EXPECTED_TIES[year]:
            raise RuntimeError(f"INDEPENDENT_188_TIE_IDS_NOT_REPRODUCED_{year}: {ties[year]} vs {EXPECTED_TIES[year]}")
        counts[f"year_{year}_all_market_races"]=n
        counts[f"year_{year}_ties"]=ties[year]
        print("TEMPORAL_FULL_SOURCE_CHECK",
              json.dumps({"year":year,"market_books":n,"matched_results":nr,
                          "verified_ties":ties[year]}),flush=True)
    if len(seen)!=134336:raise RuntimeError("FULL_SEVEN_MARKET_UNIVERSE_NOT_LOADED")
    if not records:
        raise RuntimeError("ZERO_ABILITIES_AFTER_SURROGATE_ID_MATCH")
    bad=set()
    for person,observed in age_consistency.items():
        if len(observed)<2:continue
        days=[x[0] for x in observed]
        ages=[x[1] for x in observed]
        if max(ages)-min(ages)>(max(days)-min(days)).days/365.25+2.0:
            bad.add(person)
    counts["surrogate_ids_with_implausible_age_span"]=len(bad)
    if bad:
        retained=[]
        for e in records:
            if any(p in bad for p in e["p"].values()):
                counts["races_quarantined_for_identity_age_conflict"]+=1
            else: retained.append(e)
        records=retained
    if not records:
        raise RuntimeError("NO_STABLE_ID_COMPLETE_ROSTER: refusing false-green 0-record study")
    return records

def _stats(history,today,window):
    cutoff=today-dt.timedelta(days=window)
    recent=[x for x in history if x[0]<today and x[0]>=cutoff]
    starts=len(recent)
    return {"starts":starts,"win_rate":sum(x[1]==1 for x in recent)/starts if starts else None,
            "top2_rate":sum(x[1]<=2 for x in recent)/starts if starts else None,
            "top3_rate":sum(x[1]<=3 for x in recent)/starts if starts else None}

def prior_features(rid,day,n,players,histories,ratings):
    riders=[]
    for car,p in sorted(players.items()):
        hist=histories[p]
        earlier=[x for x in hist if x[0]<day]
        if len(earlier)!=len(hist):
            raise RuntimeError("LOOKAHEAD_OR_SAME_DAY_LEAKAGE")
        total=len(earlier)
        wins=sum(x[1]==1 for x in earlier)
        top2=sum(x[1]<=2 for x in earlier)
        top3=sum(x[1]<=3 for x in earlier)
        riders.append({
            "car":car,"rider_id":p,"event_date":day.isoformat(),
            "history_observations":total,
            "history_left_censored_before_2021":True,
            "history_sufficient_5":total>=MIN_HIST,
            "days_since_last_observed_race":(day-earlier[-1][0]).days if earlier else None,
            "lifetime_observed_win_rate":wins/total if total else None,
            "lifetime_observed_top2_rate":top2/total if total else None,
            "lifetime_observed_top3_rate":top3/total if total else None,
            "recent_30d":_stats(earlier,day,30),
            "recent_90d":_stats(earlier,day,90),
            "recent_365d":_stats(earlier,day,365),
            "asof_elo_pairwise_observed":round(ratings[p],4),
            "elo_is_heuristic_not_calibrated_probability":True,
            "no_lookahead":True
        })
    return {"race_id":rid,"race_date":day.isoformat(),"n_cars":n,
            "prediction_features_only_prior_calendar_days":True,"riders":riders}

def update_race(day,players,podium,histories,ratings):
    """Ordered podium pairwise Elo vs lower finishers; no invented rank among outsiders."""
    known=[players[c] for c in podium]
    rest=[players[c] for c in sorted(players) if c not in podium]
    allids=known+rest
    current={p:ratings[p] for p in allids}
    delta=collections.defaultdict(float)
    pairs=[]
    for i in range(3):
        for j in range(i+1,3):pairs.append((known[i],known[j]))
        for p in rest:pairs.append((known[i],p))
    # Rating comparison against known strictly ranked pairs only.
    for win,loss in pairs:
        x=1/(1+10**((current[loss]-current[win])/400))
        amount=(12.0/max(1,len(allids)-1))*(1-x)
        delta[win]+=amount
        delta[loss]-=amount
    for p in allids:
        ratings[p]=max(100.0,min(2900.0,current[p]+delta[p]))
    for c,p in players.items():
        place=next((i+1 for i,x in enumerate(podium) if x==c),4)
        histories[p].append((day,place))

def run(args):
    dest=Path(args.output);dest.mkdir(parents=True,exist_ok=True)
    counts=collections.Counter()
    with private_key() as key:
        events=read_full(Path(args.incoming),key,counts)
        # A rider cannot race two different races on one calendar day.
        # A composite identity collision is quarantined, not silently merged.
        byday=collections.defaultdict(set)
        for e in events:
            for person in e["p"].values():
                byday[(e["day"],person)].add(e["rid"])
        ambiguous={(date,person) for (date,person),rids in byday.items() if len(rids)>1}
        if ambiguous:
            safe=[]
            for e in events:
                if any((e["day"],person) in ambiguous for person in e["p"].values()):
                    counts["same_day_identity_collision_races_quarantined"]+=1
                else:
                    safe.append(e)
            events=safe
        events.sort(key=lambda r:(r["day"],r["rid"]))
        histories=collections.defaultdict(list)
        ratings=collections.defaultdict(lambda:1500.0)
        prevday=None
        pending=[]
        yearwriters={}
        labelwriters={}
        try:
            for y in YEARS:
                yearwriters[y]=SealedWriter(dest/f"prior_ability_{y}.jsonl.gz.cms").__enter__()
                labelwriters[y]=SealedWriter(dest/f"supervised_targets_{y}.jsonl.gz.cms").__enter__()
            for event in events:
                day=event["day"]
                if prevday is not None and prevday!=day:
                    for old in pending:
                        update_race(prevday,old["p"],old["podium"],histories,ratings)
                        counts["prior_historical_races_applied_after_day"]+=1
                    pending=[]
                x=prior_features(event["rid"],day,event["n"],event["p"],histories,ratings)
                y=event["year"]
                yearwriters[y].write(x)
                labelwriters[y].write({"race_id":event["rid"],"race_date":day.isoformat(),
                                      "finish_order":list(event["podium"])})
                counts["ability_examples_emitted"]+=1
                if all(s["history_sufficient_5"] for s in x["riders"]):
                    counts["full_field_prior_five_races_all_riders"]+=1
                if any(s["history_observations"]==0 for s in x["riders"]):
                    counts["race_has_cold_start_rider"]+=1
                if counts["ability_examples_emitted"]%20000==0:
                    print("TEMPORAL_ABILITY_PROGRESS",json.dumps({
                        "events":counts["ability_examples_emitted"],
                        "known_riders":len(histories),
                        "year":y}),flush=True)
                pending.append(event)
                prevday=day
            for old in pending:
                update_race(prevday,old["p"],old["podium"],histories,ratings)
        finally:
            # Only end successfully when CMS-encrypted outputs are complete.
            exc=sys.exc_info()
            for o in list(yearwriters.values())+list(labelwriters.values()):
                o.__exit__(*exc)
        counts["distinct_stable_rider_ids"]=len(histories)
        counts["distinct_event_days"]=len({e["day"] for e in events})
        totals={"schema":"type_lab_no_hindsight_asof_ability_v1",
                "historical_scope":"2021-01-01 to 2026-06-30",
                "input_all_full_seven_market_races":134336,
                "input_valid_7_9_market_and_result":118527,
                "input_independently_audited_ties":188,
                "estimator":"past observed win/top2/top3 windows plus 12-K pairwise podium Elo",
                "calibrated_prediction_probability":False,
                "stable_official_rider_ids_verified":False,
                "identity_strategy":"name+term+prefecture surrogate, with same-day collision quarantine",
                "identity_collision_quarantine_required":True,
                "asof_policy":"strictly earlier calendar dates only; same day withheld",
                "retrospective_racecard_score_ever_used":False,
                "model_ready_no_hindsight_feature_rows":counts["ability_examples_emitted"],
                "eligible_for_full_A_F_betting_model":False,
                "archive_provenance":"owner-sealed, SHA256 checked full seven market",
                "counts":dict(sorted(counts.items())),
                "private_outputs": [
                    {"year":y,"prior_features_sha256":file_hash(dest/f"prior_ability_{y}.jsonl.gz.cms"),
                     "separate_targets_sha256":file_hash(dest/f"supervised_targets_{y}.jsonl.gz.cms"),
                     "features_records":yearwriters[y].count}
                    for y in YEARS],
                "safety_flags":[
                    "Rider ID: provider URLs if available, else name+term+prefecture surrogate, not verified official ID",
                    "Same-day surrogate collisions are quarantined, not merged",
                    "Only clean uniquely settled observed podiums contribute history updates",
                    "Markets and retrospective racecard score never enter ability features",
                    "Pre-2021 history absent; 2021/22 sparse rider coverage, so cold start is explicit",
                    "No model training, calibrated p1/p3, forward test or betting performed"
                ]}
        (dest/"ability_rebuild_summary.json").write_text(
            json.dumps(totals,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
        print("TYPE_LAB_ABILITY_REBUILD_DONE",json.dumps({
          "source_all_races":134336,"generated":counts["ability_examples_emitted"],
          "rider_ids":counts["distinct_stable_rider_ids"],
          "all_riders_min5":counts["full_field_prior_five_races_all_riders"],
          "with_cold_start":counts["race_has_cold_start_rider"]
        }),flush=True)

def selftest():
    day=dt.date(2024,2,2)
    hist=collections.defaultdict(list)
    ratings=collections.defaultdict(lambda:1500.0)
    p={1:"player:a",2:"player:b",3:"player:c",4:"player:d",5:"player:e"}
    f=prior_features("one",day,5,p,hist,ratings)
    assert all(x["history_observations"]==0 for x in f["riders"])
    update_race(day,p,(1,2,3),hist,ratings)
    # Same day NEVER sees its own result.
    try:prior_features("two",day,5,p,hist,ratings)
    except RuntimeError:pass
    else:raise AssertionError("same-day leak not blocked")
    nextday=prior_features("next",day+dt.timedelta(days=1),5,p,hist,ratings)
    assert nextday["riders"][0]["lifetime_observed_win_rate"]==1
    assert nextday["riders"][4]["lifetime_observed_top3_rate"]==0
    assert ratings["player:a"]>ratings["player:e"]
    assert stable_player_id("https://example.com/rider/100/?id=1")=="https://example.com/rider/100?id=1"
    assert stable_player_id("John Smith") is None
    s=stable_player_id("千 葉/27/119",{"player_name":"架空 選手","prefecture":"千葉","term":"119"})
    assert s and s.startswith("surrogate:")
    assert stable_player_id("千 葉/28/119",{"player_name":"架空 選手","prefecture":"千葉","term":"119"})==s
    assert stable_player_id("千 葉/28/119",{"player_name":"架空 選手","prefecture":"千葉","term":"121"})!=s
    print("TYPE_LAB_ASOF_SELFTEST_OK")

if __name__=="__main__":
    import sys
    p=argparse.ArgumentParser()
    p.add_argument("--selftest",action="store_true")
    p.add_argument("--incoming")
    p.add_argument("--output")
    args=p.parse_args()
    if args.selftest:selftest()
    else:
        if not args.incoming or not args.output:p.error("require --incoming and --output")
        run(args)
