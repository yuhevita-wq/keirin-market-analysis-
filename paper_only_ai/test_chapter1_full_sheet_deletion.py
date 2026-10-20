"""Tests of delete-from-FULL-sheet invariants and revenue accounting."""
import importlib.util
import sys
import types
from pathlib import Path
import numpy as np

for name in ("chapter1_accuracy_calibration_duel","chapter1_market_error",
             "backtest_line_rf_2024_q1","backtest_roi_2024_q1"):
    sys.modules[name]=types.ModuleType(name)
sp=importlib.util.spec_from_file_location(
    "delete_ai",Path(__file__).with_name("chapter1_full_sheet_deletion.py"))
lab=importlib.util.module_from_spec(sp)
sp.loader.exec_module(lab)

class FakeRemovalModel:
    def predict_proba(self,X):
        scores=np.asarray(X[:,0],dtype=float)
        return np.column_stack((1-scores,scores))

sheet={
    "race_id":"race1","date":"2023-11-01","kind":"3連単",
    "role":"accuracy","tickets":(("1","2","3"),("2","1","3"),("3","2","1")),
    "x":np.array([[.9],[.5],[.1]],dtype=np.float32),
    "model_p":np.array([.5,.3,.2],dtype=float),
    "prices":np.array([5.,9.,12.],dtype=float),
    "winner_index":0,
}
model=FakeRemovalModel()
full=np.ones(3,dtype=bool)
remain=lab.delete_from_full(model,sheet,.50)
assert remain.dtype==bool
assert remain.shape==full.shape
assert remain.tolist()==[True,False,False]
assert lab.delete_from_full(model,sheet,.80).sum()==2
assert lab.delete_from_full(model,sheet,1.0).all()

# A discarded winning ticket forfeits its ACTUAL payout, not a simulated
# probability-weighted payout. Full-stake saving minus forfeited return must
# equal post-deletion net minus the full-sheet net.
data=[]
for role in lab.ROLES:
    for kind in lab.KINDS:
        data.append({**sheet,"role":role,"kind":kind})
models={role:{kind:model for kind in lab.KINDS} for role in lab.ROLES}
stats=lab.score_rows(data,models,{r:.50 for r in lab.ROLES})
for role in lab.ROLES:
    s=stats[role]
    assert s["eligible_full_marks"]==6
    assert s["marked_tickets_remaining"]==2
    assert s["deleted_marks"]==4
    assert s["winning_tickets_retained"]==2
    assert s["winning_tickets_deleted"]==0
    assert s["full_sheet_investment_yen"]==600
    assert s["remaining_investment_yen"]==200
    assert s["full_sheet_payout_yen"]==1000
    assert s["remaining_payout_yen"]==1000
    assert s["net_change_from_full_sheet_yen"]==400
    assert s["remaining_return_percent"]==500

# Negative case: the winning mark was removed, so the genuine payout loss
# must be accounted for and can overwhelm the stake savings.
lost={**sheet,"winner_index":2}
lost_data=[{**lost,"role":role,"kind":kind}
           for role in lab.ROLES for kind in lab.KINDS]
miss=lab.score_rows(lost_data,models,{r:.5 for r in lab.ROLES})
assert miss["accuracy"]["forfeited_winning_payout_yen"]==2400
assert miss["accuracy"]["winning_tickets_deleted"]==2
assert miss["accuracy"]["net_change_from_full_sheet_yen"]==400-2400
assert miss["accuracy"]["remaining_return_percent"]==0

# No projection may touch labels: changing the winner cannot change the
# deletion mask or its feature input.
assert np.array_equal(lab.delete_from_full(model,sheet,.5),
                      lab.delete_from_full(model,lost,.5))
print("FULL_SHEET_DELETE_TESTS_PASSED")
