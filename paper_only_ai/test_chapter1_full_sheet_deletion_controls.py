"""Mathematical equal-mark-count deletion controls are honest comparators."""
import importlib.util
import sys
import types
from pathlib import Path
import numpy as np

fake=types.ModuleType("chapter1_full_sheet_deletion")
fake.ROLES=("accuracy","calibration")
fake.KINDS=("3連単","3連複")
fake.STAKE=100
fake.delete_from_full=lambda model,row,coverage:np.array([True,False,False])
fake.OUT=Path("unused")
fake.REPORT=Path("unused.json")
sys.modules["chapter1_full_sheet_deletion"]=fake
spec=importlib.util.spec_from_file_location(
 "control_test",Path(__file__).with_name("chapter1_full_sheet_deletion_controls.py"))
mod=importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
rows=[]
for role in fake.ROLES:
 for kind in fake.KINDS:
  rows.append({"role":role,"kind":kind,"prices":np.array([2.,3.,10.]),
               "winner_index":0,"race_id":"r1"})
models={role:{kind:object() for kind in fake.KINDS} for role in fake.ROLES}
results=mod.sum_stats(rows,models,{role:.5 for role in fake.ROLES})
a=results["accuracy"]["combined"]
assert a["stake"]==200
assert a["retained_marks"]==2
assert a["learned_paid"]==400
assert np.isclose(a["random_expected_paid"],400/3)
assert a["favorite_paid"]==400
assert np.isclose(a["learned_roi"],200)
assert np.isclose(a["random_matched_count_expected_roi"],200/3)
print("DELETION_SAME_MARK_COUNT_CONTROL_TESTS_PASSED")
