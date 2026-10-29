"""Regression tests for independent, outcome-free V2 probability scoring."""
from __future__ import annotations
import itertools
import json
import math
import tempfile
import unittest
from pathlib import Path
from scoring import apply_v2,load_coefficients

class NoOddsScoringTest(unittest.TestCase):
    def make(self,n):
        cars=list(map(str,range(1,n+1)))
        combinations=["-".join(trio) for trio in itertools.permutations(cars,3)]
        weights=[(1+int(ticket.split("-")[0])) for ticket in combinations]
        den=sum(weights)
        return ({ticket:a/den for ticket,a in zip(combinations,weights)},
                {str(i):100.0-i for i in range(1,n+1)})
    def test_identity_7(self):
        p,s=self.make(7)
        y=apply_v2(p,"1-2-3/4-5/6-7",s,[1.0]+[0.0]*7)
        self.assertEqual(len(y),210)
        self.assertAlmostEqual(sum(y.values()),1,places=12)
        for t,original in p.items():self.assertAlmostEqual(y[t],original,places=12)
    def test_identity_9(self):
        p,s=self.make(9)
        y=apply_v2(p,"1-2-3/4-5-6/7-8-9",s,[1.0]+[0.0]*7)
        self.assertEqual(len(y),504)
        self.assertAlmostEqual(sum(y.values()),1,places=12)
    def test_same_line_adjustment_effect(self):
        p,s=self.make(7)
        w=[1,0,0,0,1.0,0,0,0]
        y=apply_v2(p,"1-2-3/4-5/6-7",s,w)
        self.assertGreater(y["1-2-3"],p["1-2-3"])
        self.assertLess(y["1-4-6"],p["1-4-6"])
        self.assertAlmostEqual(sum(y.values()),1,places=12)
    def test_incomplete_grid_rejected(self):
        p,s=self.make(7)
        del p["1-2-3"]
        with self.assertRaises(ValueError):apply_v2(p,"1-2-3/4-5/6-7",s,[1.0]+[0.0]*7)
    def test_odds_does_not_enter_interface(self):
        p,s=self.make(7)
        # Signature includes ONLY model probabilities, lineup, pre-race scores, weights.
        got=apply_v2(p,"1-2-3/4-5/6-7",s,[1.042,.03,.09,.05,.09,.02,.05,-.01])
        self.assertEqual(len(got),210)
        self.assertTrue(all(math.isfinite(x) and x>0 for x in got.values()))
        self.assertAlmostEqual(sum(got.values()),1,places=12)
    def test_weight_manifest_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"weights.json"
            path.write_text(json.dumps({"schema":"frozen_keirin_3trifecta_no_odds_postcalibrator_v2",
                 "holdout_2025_not_used":True,"coefficients":[1.04]+[0]*7}),encoding="utf-8")
            self.assertEqual(len(load_coefficients(path)),8)
            path.write_text(json.dumps({"schema":"unverified","coefficients":[1.04]+[0]*7}),encoding="utf-8")
            with self.assertRaises(ValueError):load_coefficients(path)

if __name__=="__main__":
    unittest.main(verbosity=2)
