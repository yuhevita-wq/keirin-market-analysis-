"""Deterministic assertions for full-ticket sequential deletion learning."""
import importlib.util
import sys
import types
from pathlib import Path
import numpy as np

for name in ("chapter1_accuracy_calibration_duel","chapter1_blind_exam",
             "chapter1_full_sheet_deletion","backtest_line_rf_2024_q1",
             "backtest_roi_2024_q1"):
    sys.modules[name]=types.ModuleType(name)
spec=importlib.util.spec_from_file_location(
 "learner_under_test",Path(__file__).with_name("chapter1_iterative_deletion.py"))
d=importlib.util.module_from_spec(spec)
spec.loader.exec_module(d)

class Frozen:
    def predict_proba(self,x):
        q=np.array([0.4,0.6,0.7],dtype=float)
        return np.column_stack((1-q,q))

fake={"x":np.vstack((np.arange(20),np.zeros(20),np.ones(20))).astype(float),
      "prices":np.array([2.5,4.5,8.0]),"role":"accuracy","kind":"3連単",
      "model_p":np.array([.6,.3,.1]),"winner_index":0}
original=d.ticket_probabilities(fake,Frozen(),np.zeros(d.N_FEATURES))
assert np.isclose(original.sum(),1)
assert np.all(original>0)
mask=d.remove_from_all(original,fake["prices"],.35)
assert len(mask)==3 and mask.any() and (~mask).any()
assert d.remove_from_all(original,fake["prices"],1.).all()
assert np.array_equal(mask,d.remove_from_all(original,fake["prices"],.35))

# The result label cannot alter pre-deletion probabilities or masks.
wrong_winner={**fake,"winner_index":2}
assert np.allclose(original,d.ticket_probabilities(
    wrong_winner,Frozen(),np.zeros(d.N_FEATURES)))
assert np.array_equal(mask,d.remove_from_all(original,
    wrong_winner["prices"],.35))
weights=d.update_after_result(np.zeros(d.N_FEATURES),
    fake["x"],original,0,2.5)
assert np.isfinite(weights).all() and np.linalg.norm(weights)>0
weights2=d.update_after_result(np.zeros(d.N_FEATURES),
    fake["x"],original,2,8)
assert not np.allclose(weights,weights2)

st=d.base_checkpoint()
d.ST=100
assert st["processed_quarters"]==[]
assert st["frozen_2023_coverage"]=={"accuracy":.35,"calibration":.5}
assert d.load_state.__name__=="load_state"
assert d.model_digest({r:{k:np.zeros(d.N_FEATURES)
          for k in d.KINDS} for r in d.ROLES})

s=d.empty_stats()
d.mark_score(s,np.array([2.5,4.5,8.]),0,np.array([True,False,False]),
             np.array([True,True,True]))
v=d.summarize(s)
assert v["stake"]==100 and v["payout"]==250
assert v["full_stake"]==300 and v["full_payout"]==250
assert v["roi_percent"]==250
assert v["value_of_deletion_vs_all_yen"]==200
assert v["uniform_random_same_count_roi_percent"]>0
assert v["original_frozen_delete_roi_percent"]==250/3
s2=d.empty_stats()
d.mark_score(s2,np.array([2.5,4.5,8.]),0,np.array([False,True,False]),
             np.array([True,True,True]))
assert d.summarize(s2)["lost_payout"]==250
assert d.summarize(s2)["payout"]==0
print("CHAPTER1_ITERATIVE_DELETION_TESTS_PASSED")
