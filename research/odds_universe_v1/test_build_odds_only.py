"""Executable tests for the odds + race-type only exporter."""
from __future__ import annotations
import csv
import itertools
import json
import sqlite3
import tempfile
from collections import defaultdict
from pathlib import Path

from build_odds_only import (
    LABEL_FIELDS, construct_race_type, consolidate_source, export,
    market_rows_to_record, normalize_combi, year_of
)


def make_market(rid, cars=(1, 2, 3, 4), rank=3):
    odds = {"3連単": {}, "3連複": {}}
    for perm in itertools.permutations(cars, 3):
        odds["3連単"]["-".join(map(str,perm))] = 100.0
    for group in itertools.combinations(cars,3):
        odds["3連複"]["=".join(map(str,group))] = 10.0
    return {"odds":odds, "bad":0}


def row(rid,day,kind,grade="F1"):
    return {"race_id":rid,"race_date":day,"race_type":kind,"meeting_grade":grade}


def database():
    db=sqlite3.connect(":memory:")
    db.execute("CREATE TABLE races (race_id TEXT PRIMARY KEY, year TEXT NOT NULL, "
               "race_type TEXT NOT NULL, payload TEXT NOT NULL, checksum TEXT NOT NULL, "
               "source TEXT NOT NULL)")
    return db


def test_normalization_and_exclusion():
    assert normalize_combi("4=1=2","3連複") == "1=2=4"
    assert normalize_combi("4-1-2","3連単") == "4-1-2"
    assert normalize_combi("1-1-2","3連単") is None
    assert construct_race_type(row("1","2025-01-01","Ｓ級予選"),["S1"])[0] == "F1/S級予選"
    assert construct_race_type(row("1","2025-01-01","ガールズ予選"),["L1"])[0] is None
    assert construct_race_type(row("1","2025-01-01","A級予選"),["L1"])[0] is None


def test_complete_market_and_no_lookahead():
    rid="1320250101010001"
    metadata=row(rid,"2025-01-01","Ａ級予選")
    markets=make_market(rid)
    out,status=market_rows_to_record(rid,markets,{rid:metadata},
                    {rid:set(range(1,5))},{rid:["A1"]*4})
    assert status=="ok"
    assert set(out)==LABEL_FIELDS
    assert out["race_type"]=="F1/A級予選"
    assert len(out["odds"]["3連単"])==24
    assert len(out["odds"]["3連複"])==4
    assert "result" not in str(out)
    assert "player" not in str(out)
    del markets["odds"]["3連単"]["1-2-3"]
    assert market_rows_to_record(rid,markets,{rid:metadata},
                 {rid:set(range(1,5))},{rid:["A1"]*4})[0] is None
    assert year_of(metadata,rid)=="2025"


def test_scope_2026_first_half_and_girls():
    rid="1320260701010001"
    excluded=row(rid,"2026-07-01","S級予選")
    prices=make_market(rid)
    cars={rid:set(range(1,5))}
    klasses={rid:["S1"]*4}
    assert market_rows_to_record(rid,prices,{rid:excluded},cars,klasses)[0] is None
    assert year_of(excluded,rid) is None
    included=row(rid,"2026-06-30","S級予選")
    assert market_rows_to_record(rid,prices,{rid:included},cars,klasses)[0] is not None
    assert year_of(included,rid)=="2026"
    women=row(rid,"2026-06-30","ガールズ決勝")
    assert market_rows_to_record(rid,prices,{rid:women},cars,
                                 {rid:["L1"]*4})[0] is None


def test_dedup_and_gzip_real_export():
    rid="1320250101010001"
    kind="3連単"
    r=row(rid,"2025-01-01","Ｓ級予選")
    market=make_market(rid)
    tri=[{"race_id":rid,"ticket_type":"3連単","odds_phase":"final",
          "odds_status":"available","combination":k,"odds":v}
          for k,v in market["odds"]["3連単"].items()]
    trio=[{"race_id":rid,"ticket_type":"3連複","odds_phase":"final",
          "odds_status":"available","combination":k,"odds":v}
          for k,v in market["odds"]["3連複"].items()]
    entries=[{"race_id":rid,"car_no":str(i),"class":"S1"} for i in range(1,5)]
    db=database()
    stats={"year_counts":defaultdict(int),"conflict_examples":[]}
    class Meter(dict):
        def __missing__(self,key):
            self[key]=0
            return 0
    m=Meter(stats)
    for _ in range(2):
        consolidate_source(db,"synthetic",(v for v in [r]),(v for v in entries),
                           (v for v in tri),(v for v in trio),m)
    assert m["unique_complete_races"]==1,m
    assert m["duplicate_race"]==1,m
    with tempfile.TemporaryDirectory() as tmp:
        import gzip
        result=export(db,m,Path(tmp))
        assert result["2025"]["complete_races"]==1
        with gzip.open(Path(tmp)/"odds_2025.jsonl.gz","rt",encoding="utf-8") as stream:
            payload=json.loads(stream.readline())
            assert stream.readline()==""
        assert set(payload)==LABEL_FIELDS
        assert len(payload["odds"]["3連単"])==24
        assert payload["race_type"]=="F1/S級予選"


if __name__=="__main__":
    test_normalization_and_exclusion()
    test_scope_2026_first_half_and_girls()
    test_complete_market_and_no_lookahead()
    test_dedup_and_gzip_real_export()
    print("ODDS_AND_RACE_TYPE_ONLY_EXPORT_TESTS_PASSED")
