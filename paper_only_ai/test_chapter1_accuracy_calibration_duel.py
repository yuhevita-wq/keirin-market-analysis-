"""Deterministic duel safeguards: one lineup, same odds, proper distributions."""
import importlib.util
import sys
import types
from pathlib import Path
import numpy as np

acc=types.ModuleType("chapter1_accuracy_ai")
acc.race_relative_features=lambda races:np.zeros((sum(len(r["cars"]) for r in races.values()),38))
acc.candidate_inputs=lambda X,r,c:np.concatenate([X,r],axis=1)
acc.selected_columns=lambda config:np.arange(60)
sys.modules["chapter1_accuracy_ai"]=acc
prev=types.ModuleType("chapter1_blind_exam")
sys.modules["chapter1_blind_exam"]=prev
bet=types.ModuleType("backtest_line_rf_2024_q1")
sys.modules["backtest_line_rf_2024_q1"]=bet
prices=types.ModuleType("backtest_roi_2024_q1")
sys.modules["backtest_roi_2024_q1"]=prices
cmp=types.ModuleType("compare_lines_2023")
cmp.feature=lambda race,c,**kwargs:np.zeros(22,dtype=np.float32)
sys.modules["compare_lines_2023"]=cmp

spec=importlib.util.spec_from_file_location("duel",
    Path(__file__).with_name("chapter1_accuracy_calibration_duel.py"))
d=importlib.util.module_from_spec(spec)
spec.loader.exec_module(d)
race={"cars":{"1":0,"2":0,"3":0},
      "order":("1","2","3")}
metrics=d.CalibEce()
prob={
    "3連単":{("1","2","3"):0.6,("2","1","3"):0.3,("3","2","1"):0.1},
    "3連複":{("1","2","3"):1.0},
}
metrics.update({"1":0.7,"2":0.2,"3":0.1},race,prob)
res=metrics.summary()
assert res["first_place_correct"]==1
assert res["single_exact_trifecta_correct"]==1
assert res["single_unordered_trio_correct"]==1
assert res["races"]==1
paired={"accuracy":[(100,200),(200,0)]*100,
        "calibration":[(100,150),(100,0)]*100}
bootstrap=d.bootstrap_paired_annual(paired,iterations=150)
assert np.isfinite(bootstrap["point_difference_accuracy_minus_calibration_pp"])
assert bootstrap["p025"] <= bootstrap["p975"]
print("ACCURACY_CALIBRATION_DUEL_UNIT_TESTS_PASSED")
