"""GEN-003 must inherit GEN-002, spend only on marks, and refill only on empty wallet."""
from __future__ import annotations
import hashlib
import json

import numpy as np
from train_week3 import ROOT, OUT, PARENT_DIR, COST_PER_MARK, INITIAL_YEN, REPLENISH_YEN, spend_marks


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def run():
    parent=read(PARENT_DIR/"checkpoint.json")
    summary=read(OUT/"summary.json")
    checkpoint=read(OUT/"checkpoint.json")
    marks=[json.loads(x) for x in (OUT/"formation_mark_sheets.jsonl").read_text(encoding="utf-8").splitlines()]
    records=[json.loads(x) for x in (OUT/"learning_history.jsonl").read_text(encoding="utf-8").splitlines()]
    assert summary["id"]==checkpoint["id"]=="GEN-003"
    assert summary["parent_id"]==checkpoint["parent_id"]=="GEN-002"
    assert summary["dates"]==["2023-01-15","2023-01-21"]
    assert parent["id"]=="GEN-002" and parent["after_step"]==1029
    assert summary["prior_checkpoint_sha256"]==parent["weights_sha256"]
    assert checkpoint["parent_weights_sha256"]==parent["weights_sha256"]
    assert marks[0]["checkpoint_before_sha256"]==parent["weights_sha256"]
    assert len(marks)==len(records)==summary["races"]>0
    assert checkpoint["after_step"]==1029+len(marks)
    weights=np.asarray(checkpoint["weights"],dtype=np.float64)
    assert weights.shape==(256,)
    assert hashlib.sha256(weights.tobytes()).hexdigest()==checkpoint["weights_sha256"]
    assert checkpoint["weights_sha256"]!=parent["weights_sha256"]
    assert len({m["race_id"] for m in marks})==len(marks)
    # Ledger: virtual deposits are permitted only when < 100 yen is left.
    balance=INITIAL_YEN
    spent=paid=refill_events=sheets=mark_count=hits=0
    for i,(a,b) in enumerate(zip(marks,records),1):
        assert a["generation"]=="GEN-003" and a["parent"]=="GEN-002"
        assert a["generation_step"]==b["generation_step"]==i
        assert a["step"]==b["step"]==1029+i
        assert a["race_id"]==b["race_id"]
        assert a["checkpoint_before_sha256"]==b["checkpoint_before_sha256"]
        assert "finish_order" not in a and "hit" not in a and "payout_proxy_yen" not in a
        assert a["balance_before_yen"]==balance
        assert a["sheet_count"]==len(a["sheets"])>0
        assert a["total_marks"]==sum(page["point_count"] for page in a["sheets"])
        assert a["total_staked_yen"]==a["total_marks"]*COST_PER_MARK
        after_purchase,cost,events=spend_marks(a["sheets"],balance)
        assert cost==a["total_staked_yen"]
        assert events==a["replenishments_pre_result"]
        assert all(0<=e["balance_before_yen"]<100 and e["amount_yen"]==REPLENISH_YEN for e in events)
        assert len(events)==b["replenishments"]
        assert after_purchase==a["balance_after_marks_yen"]==b["balance_after_marks_yen"]
        assert b["spent_yen"]==cost
        assert b["balance_before_yen"]==balance
        balance=after_purchase+b["payout_proxy_yen"]
        assert balance==b["balance_after_result_yen"]
        spent+=cost
        paid+=b["payout_proxy_yen"]
        refill_events+=len(events)
        sheets+=len(a["sheets"])
        mark_count+=a["total_marks"]
        hits+=int(b["hit"])
        if i>1:
            assert a["checkpoint_before_sha256"]==records[i-2]["checkpoint_after_sha256"]
        for page in a["sheets"]:
            assert page["point_count"]==len(page["picks"])>0
            assert page["total_yen"]==100*page["point_count"]
    assert balance==checkpoint["bankroll_balance_yen"]==summary["final_balance_yen"]
    assert summary["total_contributed_capital_yen"]==INITIAL_YEN+refill_events*REPLENISH_YEN
    assert summary["replenishment_events"]==checkpoint["replenishment_events"]==refill_events
    assert summary["replenishment_amount_yen"]==refill_events*REPLENISH_YEN
    assert spent==summary["total_spent_yen"] and paid==summary["total_payout_proxy_yen"]
    assert balance==INITIAL_YEN+refill_events*REPLENISH_YEN+paid-spent
    assert summary["net_betting_proxy_yen"]==paid-spent
    assert sheets==summary["sheets"] and mark_count==summary["marks"]
    assert hits==summary["hit_races"]
    assert checkpoint["weights_sha256"]==records[-1]["checkpoint_after_sha256"]
    assert len(summary["daily"])==7 and sum(x["races"] for x in summary["daily"])==len(marks)
    assert sum(x["replenishments"] for x in summary["daily"])==refill_events
    assert summary["raw_racecards_in_week"]>=len(marks)
    pages=[{"sheet_number":1,"picks":["a"]*104}]
    after,cost,events=spend_marks(pages,0)
    assert len(events)==2 and after==9600 and cost==10400
    after2,cost2,events2=spend_marks([{"sheet_number":2,"picks":["a"]}],50)
    assert len(events2)==1 and after2==9950 and cost2==100
    dashboard=read(ROOT/"docs/ai-lab/nursery-gen003-summary.json")
    assert dashboard["races"]==summary["races"]
    registry=read(ROOT/"docs/ai-lab/generations.json")
    assert registry["current_generation"]=="GEN-003"
    assert registry["generations"][-1]["parent_id"]=="GEN-002"
    print("GEN003 BANKROLL AND GENERATION INTEGRITY PASS",len(records),"races",refill_events,"refills","final",balance)


if __name__=="__main__":
    run()
