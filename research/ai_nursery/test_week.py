"""Audit that the baby wrote marks before seeing historical race outcomes."""
from pathlib import Path
import hashlib
import json
from datetime import date
import math
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/"research/ai_nursery/outputs/GEN-001"
def main():
    j=lambda name:json.loads((OUT/name).read_text(encoding="utf-8"))
    summary=j("summary.json");checkpoint=j("checkpoint.json")
    marked=[json.loads(x) for x in (OUT/"mark_sheets.jsonl").read_text(encoding="utf-8").splitlines()]
    trained=[json.loads(x) for x in (OUT/"learning_history.jsonl").read_text(encoding="utf-8").splitlines()]
    assert marked and len(marked)==len(trained)==summary["races"]
    assert summary["date_first"]<=summary["date_last"]
    assert (date.fromisoformat(summary["date_last"])-date.fromisoformat(summary["date_first"])).days==6
    assert checkpoint["after_step"]==len(trained) and checkpoint["dimension"]==256
    assert len(checkpoint["weights"])==256 and all(math.isfinite(x) for x in checkpoint["weights"])
    assert checkpoint["weights_sha256"]==trained[-1]["checkpoint_after_sha256"]
    assert not all(x==0 for x in checkpoint["weights"])
    zeros=hashlib.sha256(bytes(256*8)).hexdigest()
    assert marked[0]["checkpoint_before_sha256"]==zeros
    assert len({m["race_id"] for m in marked})==len(marked)
    for i,(a,b) in enumerate(zip(marked,trained)):
        assert a["step"]==b["step"]==i+1
        assert a["race_id"]==b["race_id"]
        assert a["checkpoint_before_sha256"]==b["checkpoint_before_sha256"]
        assert "finish_order" not in a and "hit" not in a and "reward_yen_proxy" not in a
        if i:assert a["checkpoint_before_sha256"]==trained[i-1]["checkpoint_after_sha256"]
        assert a["yen"]==100 and a["type"] in ("3連単","3連複")
        assert summary["date_first"]<=a["date"]<=summary["date_last"]
    assert len(summary["daily"])==7
    assert sum(x["races"] for x in summary["daily"])==len(marked)
    assert sum(x["hits"] for x in summary["daily"])==summary["hits"]
    assert sum(x["hit"] for x in trained)==summary["hits"]
    assert sum(x["paper_stake_yen"] for x in summary["daily"])==summary["paper_stake_yen"]
    print("BABY GENERATION VERIFIED",summary["date_first"],summary["date_last"],len(marked),summary["hits"])
if __name__=="__main__":main()
