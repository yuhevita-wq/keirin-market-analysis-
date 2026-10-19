"""Check first-generation freeze, original Git blobs, and safe paper inference."""
from __future__ import annotations
import hashlib
import itertools
import json
import math
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research" / "market_probability_v1"))
from t15_decision import FEATURES
from predict import MODEL, predict

def git_sha(data):
    return hashlib.sha1(("blob "+str(len(data))+"\0").encode()+data).hexdigest()

def test():
    path = MODEL.parent
    m = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    checkpoint = json.loads(MODEL.read_text(encoding="utf-8"))
    audit = json.loads((path/"audit.json").read_text(encoding="utf-8"))
    assert m["id"] == "GEN-001" and m["parent"] is None
    assert git_sha(MODEL.read_bytes()) == m["parent_model_blob"]
    assert git_sha((path/"audit.json").read_bytes()) == m["parent_audit_blob"]
    assert checkpoint["features"] == list(FEATURES)
    assert len(checkpoint["coefficients"]) == len(checkpoint["features"]) == 45
    assert all(math.isfinite(x) for x in checkpoint["coefficients"])
    assert checkpoint["train_through"] == "2025-12-31"
    values = audit["evaluations"][m["source_audit_key"]]
    assert values["market"]["races"] == values["corrected"]["races"] == m["metrics"]["races"]
    for key in ("market","corrected"):
        for metric in ("logloss","brier","top1_pct"):
            assert math.isclose(values[key][metric],m["metrics"][key][metric],rel_tol=1e-12)
    deadline = datetime(2026,10,19,22,0,tzinfo=timezone(timedelta(hours=9)))
    cutoff = deadline-timedelta(minutes=15)
    stamp = (cutoff-timedelta(seconds=30)).isoformat()
    cars=list(range(1,8))
    snapshot={
        "odds_phase":"live",
        "race":{"race_id":"synthetic","track":"TEST",
            "start_jst":(deadline+timedelta(minutes=5)).isoformat(),
            "deadline_jst":deadline.isoformat(),"t15_reference":"sales_deadline",
            "card_as_of_jst":(cutoff-timedelta(minutes=4)).isoformat(),
            "predicted_line_formation":"1-2-3/4-5/6-7"},
        "entries":[{"car_no":i,"score":90+i,"win_rate":13+i,"top3_rate":28+i,
            "b_count":2,"nige_count":2,"makuri_count":2,"sashi_count":2,
            "mark_count":2,"age":27+i,"style":"両"} for i in cars],
        "captured_at_jst":stamp,"market_as_of_jst":{"3連単":stamp,"3連複":stamp},
        "source_url":"https://example.org/test",
        "odds":{"3連単":{"-".join(map(str,p)):210.0 for p in itertools.permutations(cars,3)},
                "3連複":{"=".join(map(str,p)):35.0 for p in itertools.combinations(cars,3)}}
    }
    result=predict(snapshot,1.0)
    assert result["screened_trifecta_outcomes"]==210
    assert math.isclose(result["sum_probabilities"],1,abs_tol=1e-9)
    assert len(result["paper_decisions"])==result["candidate_count_above_sensitivity_threshold"]
    assert all(t["paper_stake_yen"]==0 for t in result["paper_decisions"])
    assert not result["real_money_automated_purchases"]
    snapshot["odds_phase"]="final"
    try:
        predict(snapshot)
        raise AssertionError("Final odds were accepted as live")
    except ValueError:
        pass
    reg=json.loads((ROOT/"docs/ai-lab/generations.json").read_text(encoding="utf-8"))
    assert reg["current_generation"]=="GEN-001"
    assert reg["generations"][0]["gain"]>0
    print("PASS: GEN-001 frozen checkpoint, paper inference, leakage gate, dashboard registry")

if __name__=="__main__":
    test()
