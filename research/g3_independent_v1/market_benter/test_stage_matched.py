"""Synthetic 5-car G3 outcome/odds-to-training-rows staging tests."""
import json
import sys
import unittest
from pathlib import Path

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from stage_matched import convert
from train_fusion import parse_record,SINGLE_WIN_PROXY
from engine import aggregate_probability
from predict_g3_ordered_triples import all_ordered_triples

CARD=json.loads((HERE/"fixtures/synthetic_g3_card.json").read_text(encoding="utf-8"))
ODDS=json.loads((HERE/"fixtures/synthetic_final_odds.json").read_text(encoding="utf-8"))
SECOND=json.loads((HERE.parent/"models/g3_second_conditional_v1.json").read_text(encoding="utf-8"))
THIRD=json.loads((HERE.parent/"models/g3_third_conditional_v1.json").read_text(encoding="utf-8"))

def sample():
    return {"scope":"G3 S級","race_id":"SYNTHETIC-G3-ID-ONLY",
            "race_date":"2024-01-01",
            "fundamental_provenance":"G3_only_prior_period_independent",
            "price_phase":"final",
            "original_asof_card_certified":False,
            "card":CARD,"finish":{"first":1,"second":2,"third":3},
            "markets":ODDS["markets"]}

class StageMarket(unittest.TestCase):
    def test_stages_triple_trio_and_win_proxy(self):
        rr=convert(sample(),SECOND,THIRD)
        self.assertEqual(set(rr),{"3連単","3連複",SINGLE_WIN_PROXY})
        for name,item in rr.items():
            validated=parse_record(item,name)
            self.assertAlmostEqual(sum(validated["f"].values()),1,places=10)
            self.assertAlmostEqual(sum(validated["p"].values()),1,places=10)
        self.assertEqual(rr["3連単"]["actual_winning_selection"],"1-2-3")
        self.assertEqual(rr["3連複"]["actual_winning_selection"],"1-2-3")
        self.assertEqual(rr[SINGLE_WIN_PROXY]["actual_winning_selection"],"1")
        self.assertEqual(len(rr["3連単"]["fundamental_probabilities"]),60)
        self.assertEqual(len(rr["3連複"]["fundamental_probabilities"]),10)

    def test_missing_price_or_podium_refused(self):
        s=sample()
        s["markets"]={k:dict(v) for k,v in ODDS["markets"].items()}
        s["markets"]["3連単"].pop(next(iter(s["markets"]["3連単"])))
        with self.assertRaises(ValueError):convert(s,SECOND,THIRD)
        s=sample();s["finish"]["third"]=s["finish"]["second"]
        with self.assertRaises(ValueError):convert(s,SECOND,THIRD)

    def test_scope_and_provenance_enforced(self):
        s=sample();s["scope"]="F1 S級"
        with self.assertRaises(ValueError):convert(s,SECOND,THIRD)
        s=sample();s["fundamental_provenance"]="training_set_same_day"
        with self.assertRaises(ValueError):convert(s,SECOND,THIRD)
        s=sample();s["card"]["results"]={"first":1}
        with self.assertRaises(ValueError):convert(s,SECOND,THIRD)

if __name__=="__main__":
    unittest.main()
