import ast,unittest,itertools
from pathlib import Path
from collections import defaultdict
import numpy as np
SOURCE=Path(__file__).with_name("replay.py").read_text()
ns={"np":np,"itertools":itertools,"defaultdict":defaultdict}
tree=ast.parse(SOURCE)
exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ("roles","score","evaluate")],type_ignores=[]),"isolated-replay-test","exec"),ns)
class ReplayTest(unittest.TestCase):
    def test_roles_and_undefined_structures(self):
        card=(((1,110.),(2,108.),(3,90.)),((4,105.),(5,100.)),((6,85.),(7,80.)))
        r,why=ns["roles"](card)
        self.assertEqual(r,{"A":1,"B":2,"W1L":6,"W1B":7})
        self.assertIsNone(why)
        self.assertEqual(ns["roles"](card[:2])[1],"fewer_than_three_two_plus_lines")
    def test_all_days_kept_in_cscv_and_rank(self):
        ns["STAKE"]=np.ones((11,2))*100
        ns["PAY"]=np.column_stack([np.ones(11)*120,np.ones(11)*70])
        ns["NAMES"]=["better","worse"]
        r=ns["evaluate"]([0,1],blocks=4)
        self.assertEqual(r["padding_zero_exposure_rows"],1)
        self.assertEqual(r["valid_splits"],6)
        self.assertEqual(r["pbo_strict_below_median"],0)
        self.assertEqual(r["loss_frequency_separate_from_pbo"],0)
        for s in r["splits"]:self.assertAlmostEqual(s["test_roi"],1.2)
    def test_tie_is_not_below_median(self):
        ns["STAKE"]=np.ones((12,2))*100
        ns["PAY"]=np.ones((12,2))*80
        ns["NAMES"]=["a","b"]
        r=ns["evaluate"]([0,1],blocks=4)
        self.assertEqual(r["pbo_strict_below_median"],0)
        self.assertEqual(r["loss_frequency_separate_from_pbo"],1)
    def test_no_exposure_is_not_profit_or_failure(self):
        ns["STAKE"]=np.zeros((12,2));ns["PAY"]=np.zeros((12,2));ns["NAMES"]=["a","b"]
        r=ns["evaluate"]([0,1],blocks=4)
        self.assertEqual(r["valid_splits"],0)
        self.assertIsNone(r["pbo_strict_below_median"])
if __name__=="__main__":unittest.main()
