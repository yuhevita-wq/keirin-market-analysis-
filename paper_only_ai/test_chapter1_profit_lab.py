"""Static tests for market fusion and positive-EV historical profit decisions."""
import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
for name in ("backtest_line_rf_2024_q1", "backtest_roi_2024_q1",
             "chapter1_blind_exam"):
    sys.modules[name] = types.ModuleType(name)
spec = importlib.util.spec_from_file_location(
    "lab_under_test", Path(__file__).with_name("chapter1_profit_lab.py"))
lab = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lab)

model = np.asarray([0.55, 0.30, 0.15])
market = np.asarray([0.4, 0.4, 0.2])
for family in ("linear", "log"):
    for alpha in (0.0, 0.5, 1.0):
        parms = {"family": family, "alpha": alpha, "gamma": 1.0}
        p = lab.coherent_probability("3連単", model, market, parms)
        assert np.isfinite(p).all() and np.all(p > 0)
        assert np.isclose(p.sum(), 1.0)
        if alpha == 0:
            assert np.allclose(p, market)
        if alpha == 1:
            assert np.allclose(p, model)

obs = [{
    "date": "2023-10-09", "race": "demo_1", "kind": "3連単",
    "ai": model, "market": market,
    "prices": np.asarray([2.2, 4.0, 8.0]),
    "actual_index": 1,
}]
selected = {"3連単": {"family": "linear", "alpha": 1.0, "gamma": 1.0}}
result = lab.realized_book_profit(obs, selected, 0.0)
assert result["purchased_tickets"] == 3
assert result["winning_tickets"] == 1
assert result["investment_yen"] == 300
assert result["payout_yen"] == 400
assert np.isclose(result["return_percent"], 400/3)
none = lab.realized_book_profit(obs, selected, 10.0)
assert none["purchased_tickets"] == 0
assert none["return_percent"] is None

# October outcome is allowed for development policy selection but November/
# December can only score a policy chosen before their results are inspected.
options = {"solo": selected}
choice, rows = lab.choose_development_policy(obs, options)
assert choice is None  # sample floors block cherry-picking a single race
assert all(not r["sample_gate"] for r in rows)
print("CHAPTER1_PROFIT_LAB_SAFETY_TESTS_PASSED")
