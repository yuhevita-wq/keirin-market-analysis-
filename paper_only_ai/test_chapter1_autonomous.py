"""Fast deterministic smoke tests for autonomous Chapter 1 model search."""
import importlib.util
import sys
import types
from pathlib import Path
import numpy as np

mock = types.ModuleType("compare_lines_2023")
mock.load = lambda: None
mock.matrix = lambda *a, **k: None
def normalise(x, g):
    x = np.maximum(np.asarray(x, dtype=float), 1e-12)
    g = np.asarray(g, dtype=int)
    return x / np.bincount(g, weights=x)[g]
mock.normalise = normalise
sys.modules["compare_lines_2023"] = mock
base = types.ModuleType("train_2023")
base.FEATURES = list(range(18))
sys.modules["train_2023"] = base

spec = importlib.util.spec_from_file_location(
    "chapter1_test_subject", Path(__file__).with_name("chapter1_autonomous.py"))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

state = mod.initial_state()
assert mod.configs_for_generation(1, state) == mod.configs_for_generation(1, state)
assert len(mod.configs_for_generation(1, state)) >= 2
for s, k in [(1, 4), (2, 11), (3, 20)]:
    for mask in ("all", "drop_age", "drop_term", "drop_gear",
                 "drop_rates", "drop_style"):
        cols = mod.keep_columns(s, mask)
        assert len(set(cols) & set(range({1:22, 2:65, 3:110}[s] - k,
                                         {1:22, 2:65, 3:110}[s]))) == k
group = np.array([0, 0, 0, 1, 1])
y = np.array([0, 0, 1, 0, 1])
for t in (0.5, 1.0, 2.0):
    p = mod.conditional_prob(np.log([0.2, 0.3, 0.5, 0.1, 0.9]), group, t)
    assert np.allclose(np.bincount(group, weights=p), [1, 1])
    assert mod.score_stage(p, y, group)["conditional_top1_accuracy"] == 1.0
a = {"development": {"stagewise_mean_accuracy_proxy": 0.3,
                     "mean_conditional_positive_ece20": 0.02,
                     "all_three_80pct_occupancy_proxies_passed": False}}
b = {"development": {"stagewise_mean_accuracy_proxy": 0.4,
                     "mean_conditional_positive_ece20": 0.04,
                     "all_three_80pct_occupancy_proxies_passed": False}}
assert mod.is_improvement(b, a, "accuracy")
assert not mod.is_improvement(b, a, "calibration")
b["development"]["all_three_80pct_occupancy_proxies_passed"] = True
assert mod.is_improvement(b, a, "calibration")
state["generation"] = 1
try:
    mod.check_state(state)
    raise AssertionError("Ledger mismatch must fail")
except ValueError:
    pass
print("CHAPTER1_AUTONOMY_SMOKE_PASSED")
