"""Verify the formation week is an extension, not a new model or contaminated foresight."""
from __future__ import annotations
import hashlib
import itertools
import json
from pathlib import Path
import numpy as np
from train_week2 import ROOT, OUT, W1, form_expand

def load(p):
    return json.loads(p.read_text(encoding="utf-8"))

def main():
    parent=load(W1/"checkpoint.json")
    state=load(OUT/"checkpoint.json")
    summary=load(OUT/"summary.json")
    forms=[json.loads(x) for x in (OUT/"formation_mark_sheets.jsonl").read_text(encoding="utf-8").splitlines()]
    records=[json.loads(x) for x in (OUT/"learning_history.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(forms)==len(records)==summary["races"]>0
    assert summary["dates"]==["2023-01-08","2023-01-14"]
    assert summary["first_week_steps"]==parent["after_step"]==547
    assert summary["parent_checkpoint_sha256"]==parent["weights_sha256"]
    assert state["parent_weights_sha256"]==parent["weights_sha256"]
    assert forms[0]["checkpoint_before_sha256"]==parent["weights_sha256"]
    assert state["after_step"]==547+len(forms)
    weights=np.asarray(state["weights"],dtype=np.float64)
    assert weights.shape==(256,)
    assert hashlib.sha256(weights.tobytes()).hexdigest()==state["weights_sha256"]
    assert summary["multiple_sheet_races"]>0 and summary["multiple_point_races"]>0
    assert summary["sheets"]>=summary["races"] and summary["expanded_marks"]>=summary["sheets"]
    assert summary["max_sheets_observed"]>1
    assert len({x["race_id"] for x in forms})==len(forms)
    sheet_count=0
    mark_count=0
    for i,(a,b) in enumerate(zip(forms,records)):
        assert a["race_id"]==b["race_id"] and a["week_step"]==b["week_step"]==i+1
        assert a["step"]==b["step"]==548+i
        assert a["checkpoint_before_sha256"]==b["checkpoint_before_sha256"]
        assert "finish_order" not in a and "hit" not in a and "payout_proxy_yen" not in a
        assert a["sheets"] and len(a["sheets"])==a["sheet_count"]==b["sheet_count"]
        assert a["expanded_marks"]==b["expanded_marks"]
        assert a["total_yen"]==b["total_yen"]==100*a["expanded_marks"]
        assert "2023-01-08"<=a["date"]<="2023-01-14"
        for page in a["sheets"]:
            assert page["sheet_number"]>=1
            assert page["type"] in ("3連単","3連複")
            assert len(page["columns"])==3
            assert all(column for column in page["columns"])
            assert all(1<=car<=9 for column in page["columns"] for car in column)
            assert page["point_count"]==len(page["picks"])>=1
            assert page["total_yen"]==100*page["point_count"]
            sep="-" if page["type"]=="3連単" else "="
            expanded=set()
            for combo in itertools.product(*page["columns"]):
                if len(set(combo))==3:
                    valid=combo if page["type"]=="3連単" else tuple(sorted(combo))
                    expanded.add(sep.join(map(str,valid)))
            assert expanded==set(page["picks"])
        sheet_count+=a["sheet_count"]
        mark_count+=a["expanded_marks"]
        if i:assert a["checkpoint_before_sha256"]==records[i-1]["checkpoint_after_sha256"]
    assert summary["sheets"]==sheet_count and summary["expanded_marks"]==mark_count
    assert summary["hit_races"]==sum(int(x["hit"]) for x in records)
    assert len(summary["daily"])==7
    assert sum(d["races"] for d in summary["daily"])==len(records)
    assert sum(d["sheets"] for d in summary["daily"])==sheet_count
    assert sum(d["marks"] for d in summary["daily"])==mark_count
    assert state["weights_sha256"]==records[-1]["checkpoint_after_sha256"]
    assert state["weights_sha256"]!=parent["weights_sha256"]
    odds={"3連単":{"1-2-3":15.0,"1-3-2":18.0},"3連複":{"1=2=3":5.0}}
    assert set(form_expand("3連単",[[1],[2,3],[2,3]],odds))=={"1-2-3","1-3-2"}
    assert set(form_expand("3連複",[[1,2],[2,3],[3]],odds))=={"1=2=3"}
    dashboard=load(ROOT/"docs/ai-lab/nursery-week2-summary.json")
    assert dashboard["races"]==summary["races"]
    print("GEN-001 WEEK2 VERIFIED",summary["races"],"races",sheet_count,"sheets",mark_count,"marks")


if __name__=="__main__":
    main()
