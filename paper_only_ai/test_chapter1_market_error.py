"""Synthetic test of the data-driven conditional-market residual learner."""
import sys
import types
import importlib.util
from pathlib import Path
import numpy as np
sys.modules["backtest_line_rf_2024_q1"] = types.ModuleType("backtest_line_rf_2024_q1")
sys.modules["backtest_roi_2024_q1"] = types.ModuleType("backtest_roi_2024_q1")
source = types.ModuleType("train_2023")
source.sources=lambda: []
sys.modules["train_2023"] = source
spec = importlib.util.spec_from_file_location(
    "learner", Path(__file__).with_name("chapter1_market_error.py"))
learner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(learner)
features = np.zeros((3,learner.N_FEATURES),dtype=np.float32)
features[:,0] = [1.,0.,-1.]
market = np.full(3, 1/3)
sample=("2023-01-11","r1",features,np.log(market),
        np.array([3.0,3.0,3.0]),0)
assert np.allclose(learner.probability_from_weights(
    sample,np.zeros(learner.N_FEATURES)),market)
train = [sample]*32
weight = learner.fit_one_kind(train,0.05)
assert weight[0] > 0
after=learner.probability_from_weights(sample,weight)
assert after[0]>market[0]
assert np.isclose(after.sum(),1.0)
sets={"3連単":[sample],"3連複":[sample]}
profit=learner.combined_backtest(sets,{"3連単":weight,"3連複":weight},0)
assert profit["purchased_tickets"]>=2
assert profit["winning_tickets"]==2
assert profit["return_percent"] is not None
print("CHAPTER1_MARKET_ERROR_LEARNER_TESTS_PASSED")
