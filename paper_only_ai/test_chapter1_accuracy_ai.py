"""Offline tests for AI accuracy milestones and no-outcome leakage."""
import importlib.util
import sys
import types
import numpy as np
from pathlib import Path

cmp = types.ModuleType("compare_lines_2023")
cmp.load = lambda: None
cmp.matrix = lambda *args, **kwargs: None
cmp.feature = lambda race, car, **kwargs: np.zeros(22, dtype=np.float32)
sys.modules["compare_lines_2023"] = cmp
a = types.ModuleType("chapter1_autonomous")
a.keep_columns = lambda stage, mask: np.arange(22,dtype=np.int32)
a.SEED = 230306021
sys.modules["chapter1_autonomous"] = a
bl = types.ModuleType("chapter1_blind_exam")
bl.load_frozen_champion = lambda *args,**kwargs: None
sys.modules["chapter1_blind_exam"] = bl
bt = types.ModuleType("backtest_line_rf_2024_q1")
bt.infer_new_chunk = lambda *args,**kwargs: None
sys.modules["backtest_line_rf_2024_q1"] = bt

spec = importlib.util.spec_from_file_location(
    "acc_lab", Path(__file__).with_name("chapter1_accuracy_ai.py"))
lab = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lab)

st = lab.create_state()
lab.validate_state(st)
assert st["goal"]["winner_top1"] == 0.45
assert st["goal"]["exact_trifecta_top1"] == 0.115
assert lab.generate_config(1,1,None)["family"] == "extra_trees"
assert lab.generate_config(1,2,None)["family"] == "hgb"
assert lab.generate_config(1,3,None)["family"] == "rf"
assert lab.generate_config(1,4,None)["family"] == "lr"
assert lab.generate_config(5,2,None) == lab.generate_config(5,2,None)
try:
    wrong = dict(st)
    wrong["goal"] = dict(wrong["goal"])
    wrong["goal"]["winner_top1"] = 0.40
    lab.validate_state(wrong)
    raise AssertionError("Changing the goal after outcomes were seen must fail")
except RuntimeError:
    pass

sample = {
    "r1":{"cars":{"1":None,"2":None,"3":None},
          "order":("1","2","3")},
    "r2":{"cars":{"1":None,"2":None,"3":None},
          "order":("3","2","1")},
}
cache = {
    "r1":{"cars":["1","2","3"],"real_winner":"1",
          "real_order":("1","2","3"),"baseline_win":"2",
          "baseline_trifecta":("2","3","1"),
          "conditional":{
              "1":(0.7,("1","2","3")),
              "2":(0.2,("2","3","1")),
              "3":(0.3,("3","1","2"))}},
    "r2":{"cars":["1","2","3"],"real_winner":"3",
          "real_order":("3","2","1"),"baseline_win":"2",
          "baseline_trifecta":("1","2","3"),
          "conditional":{
              "1":(0.4,("1","2","3")),
              "2":(0.35,("2","1","3")),
              "3":(0.9,("3","2","1"))}}
}
scores = np.asarray([0.8,0.15,0.05,0.1,0.2,0.7])
result = lab.evaluate(scores,sample,cache,1.0)
assert result["races"] == 2
assert result["winner_hits"] == 2
assert result["trifecta_hits"] == 2
assert result["winner_top1"]==1 and result["trifecta_top1"]==1
baseline = lab.baseline_metrics(sample,cache)
assert baseline["winner_hits"]==0 and baseline["trifecta_hits"]==0
goal = lab.objectives({"winner_top1":0.422,
                       "trifecta_top1":0.105})
assert goal == {"winner":0.45, "trifecta":0.115}
goal2 = lab.objectives({"winner_top1":0.48,
                        "trifecta_top1":0.125})
assert np.isclose(goal2["winner"],0.49)
assert np.isclose(goal2["trifecta"],0.13)
assert not lab.hypothesis_eligible(result,goal)  # only 2 races
assert lab.rank_candidate({"winner_top1":.44,"trifecta_top1":.3}) > lab.rank_candidate({"winner_top1":.43,"trifecta_top1":.9})
print("CHAPTER1_ACCURACY_AI_TESTS_PASSED")
