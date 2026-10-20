"""Synthetic safety checks; never fetch 2025 odds in testing."""
import importlib.util
import sys
import types
from pathlib import Path

import numpy as np

bt = types.ModuleType("backtest_line_rf_2024_q1")
bt.Ledger = lambda: None
sys.modules["backtest_line_rf_2024_q1"] = bt
mkt = types.ModuleType("backtest_roi_2024_q1")
sys.modules["backtest_roi_2024_q1"] = mkt
spec = importlib.util.spec_from_file_location(
    "exam_under_test", Path(__file__).with_name("chapter1_blind_exam.py"))
exam = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exam)

class FakeModel:
    def predict_proba(self, X):
        p = np.clip(X[:, 0], 0.001, 0.999)
        return np.column_stack((1 - p, p))

adapt = exam.StageAdapter({
    "estimator": FakeModel(), "columns": np.asarray([1]),
    "temperature": 1.25,
})
result = adapt.predict_proba(np.asarray([[0.9, 0.2], [0.1, 0.8]]))
assert result.shape == (2, 2)
assert np.allclose(result.sum(axis=1), 1.0)
assert result[0, 1] < result[1, 1]

def state(gen, left, right, exams=0):
    return {"generation": gen,
            "attempts": [{"generation": x + 1} for x in range(gen)],
            "champions": {"accuracy": {"id": left},
                          "calibration": {"id": right}},
            "external_exams": [{"quarter": f"2025_q{i+1}"} for i in range(exams)]}

assert exam.readiness(state(5, "g01-c01", "g02-c01")) is None
assert exam.readiness(state(6, "g01-c01", "g01-c01")) is None
assert exam.readiness(state(6, "g01-c01", "g02-c01")) == 1
assert exam.readiness(state(10, "g01-c01", "g02-c01", exams=1)) is None
assert exam.readiness(state(12, "g01-c01", "g02-c01", exams=1)) == 2
print("CHAPTER1_BLIND_EXAM_GUARDS_PASSED")


# Previously completed race returns gate blocks 2025 unless the calibrated
# model generates real positive historical P&L at a predeclared stake.
def historical_result(percent, races=300, investment=100000):
    return {"qualified_races": races, "by_role": {
        "calibration": {"combined": {
            "return_rate_percent": percent, "investment_yen": investment
        }}
    }}
assert exam.past_profit_gate(historical_result(100.01))
assert not exam.past_profit_gate(historical_result(99.99))
assert not exam.past_profit_gate(historical_result(100.0))
assert not exam.past_profit_gate(historical_result(105.0, races=199))
assert not exam.past_profit_gate(historical_result(None))
assert not exam.past_profit_gate(historical_result(110.0, investment=0))
print("CHAPTER1_HISTORICAL_PROFIT_GATE_GUARDS_PASSED")
