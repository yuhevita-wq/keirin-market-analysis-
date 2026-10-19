"""Reproducibility and segregation checks for a fixed, frozen future-date benchmark."""
from __future__ import annotations
import hashlib,json
import numpy as np
from growth_audit import ROOT,OUT,WEB,FIRST,LAST,read_generations,complete_holdout,aggregate,paired_ci

def main():
    path=OUT/"holdout_summary.json"
    r=json.loads(path.read_text(encoding="utf-8"))
    rows=[json.loads(line) for line in (OUT/"per_race.jsonl").read_text(encoding="utf-8").splitlines()]
    assert r["status"]=="COMPLETE"
    assert r["holdout_start"]=="2023-06-01" and r["holdout_end"]=="2023-06-07"
    assert r["no_training_or_policy_tuning_on_holdout"] is True
    n=r["race_count"]
    assert n>=100
    assert r["generation_count"]>=5
    gens=[x["generation"] for x in r["metrics"]]
    assert gens[:5]==[f"GEN-{k:03}" for k in range(1,6)]
    assert [x["generation"] for x in r["metrics"]]==gens
    assert len(rows)==n*len(gens)
    reference=[x["race_id"] for x in rows[:n]]
    assert len(set(reference))==n
    assert r["race_id_set_sha256"]==hashlib.sha256("\n".join(reference).encode()).hexdigest()
    for i,gen in enumerate(gens):
        rr=rows[i*n:(i+1)*n]
        assert all(x["generation"]==gen for x in rr)
        assert [x["race_id"] for x in rr]==reference
        saved=r["metrics"][i]
        assert saved["race_count"]==len(rr)
        assert saved["frozen_checkpoint_sha256"]==read_generations()[i][1]["weights_sha256"]
        if int(gen.split("-")[1])<=5:
            assert saved["learned_stop_or_skip_policy"] is False
            assert saved["skipped_races"]==0
        else:
            assert saved["learned_stop_or_skip_policy"] is True
        assert saved["mark_count"]==sum(x["marks"] for x in rr)
        assert saved["sheets"]==sum(x["sheets"] for x in rr)
        assert saved["stakes_yen"]==sum(x["stake_yen"] for x in rr)
        assert saved["payout_proxy_yen"]==sum(x["payout_proxy_yen"] for x in rr)
        assert saved["net_proxy_yen"]==saved["payout_proxy_yen"]-saved["stakes_yen"]
        for row in rr:
            assert row["stake_yen"]==row["marks"]*100
            assert row["net_proxy_yen"]==row["payout_proxy_yen"]-row["stake_yen"]
            for market in ("tan","fuku"):
                assert row["logloss_"+market]>=0
                assert row["market_logloss_"+market]>=0
                assert 0<=row["brier_"+market]<=2.00001
                assert 0<=row["market_brier_"+market]<=2.00001
        if gen=="GEN-001":
            assert all(x["marks"]==x["sheets"]==1 for x in rr)
        else:
            assert saved["sheets_per_race"]>=1
        cp=read_generations()[i][1]
        assert cp["training_span"][-1]<r["holdout_start"]
    report=json.loads((WEB/"nursery-growth-audit.json").read_text(encoding="utf-8"))
    assert report["race_id_set_sha256"]==r["race_id_set_sha256"]
    assert set(r["paired_comparisons"])==set(gens[1:])
    for g,entries in r["paired_comparisons"].items():
        for v in entries.values():
            assert len(v["paired_bootstrap_95_percentile_interval"])==2
            assert v["paired_bootstrap_95_percentile_interval"][0]<=v["paired_bootstrap_95_percentile_interval"][1]
    print("GROWTH_AUDIT_TEST_PASS",n,"holdout races",len(gens),"frozen generations")

if __name__=="__main__":
    main()
