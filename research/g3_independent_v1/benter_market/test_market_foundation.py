"""Unit-only synthetic tickets. No actual match, venue, odds or betting result.

Covers Benter 2nd-stage G3 proxy, 7 types from 504 ordered triples,
strict T-15 snapshot times, no invented prices, and hard calibration gate.
"""
from __future__ import annotations
from datetime import datetime
from itertools import permutations
import unittest,sys
from pathlib import Path

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from ticket_space import projection,BET_TYPES,FRAME_BY_CAR
from combination import public_first_from_complete_exacta,blended_first,fit_combination
from evaluate import evaluate_race,audit_snapshot
sys.path.insert(0,str(ROOT.parent))
from test_g3_third import card9,SECOND,THIRD
from predict_g3_ordered_triples import all_ordered_triples

def complete_synthetic():
    c=card9()
    c["race_id"]="SYNTHETIC-G3-TEST-NOT-A-REAL-RACE"
    res=all_ordered_triples(c,SECOND,THIRD)
    triples=[{"first_car_no":t["first_car_no"],
              "second_car_no":t["second_car_no"],
              "third_car_no":t["third_car_no"],
              "probability":t["joint_first_second_third_probability"]}
             for t in res["ordered_triples"]]
    probs=projection(triples)
    quotes=[]
    for typ in BET_TYPES:
        for sel in sorted(probs[typ]):
            # A positive, artificially generated price solely for unit tests.
            quotes.append({"bet_type":typ,"selection":list(sel),
                           "decimal_odds":12.0+sum(sel)/10.0})
    m={
      "race_id":c["race_id"],
      "source_url":"https://example.test/synthetic-not-actual-odds",
      "capture_provenance":"GENERATED_TEST_FIXTURE_DO_NOT_USE_TO_BET",
      "captured_at_utc":"2026-10-18T00:45:00+00:00",
      "scheduled_start_at_utc":"2026-10-18T01:00:00+00:00",
      "odds_phase":"T-15",
      "sold_bet_types":list(BET_TYPES),
      "not_offered_tickets":[],
      "quotes":quotes
    }
    return c,m,probs

class TestBenterG3Foundation(unittest.TestCase):
    def test_seven_bet_types_and_exact_probability_marginalization(self):
        card,odds,prob=complete_synthetic()
        self.assertEqual(len(prob["3連単"]),504)
        self.assertEqual(len(prob["3連複"]),84)
        self.assertEqual(len(prob["2車単"]),72)
        self.assertEqual(len(prob["2車複"]),36)
        self.assertEqual(len(prob["ワイド"]),36)
        self.assertEqual(len(prob["2枠単"]),33)
        self.assertEqual(len(prob["2枠複"]),18)
        for typ,book in prob.items():
            self.assertAlmostEqual(sum(book.values()),3 if typ=="ワイド" else 1,places=9)
        self.assertEqual(FRAME_BY_CAR[4],FRAME_BY_CAR[5])
        self.assertEqual(FRAME_BY_CAR[8],FRAME_BY_CAR[9])

    def test_market_first_proxy_requires_full_exacta_book(self):
        card,market,prob=complete_synthetic()
        first={x["car_no"]:x["probability"] for x in card["first_win"]}
        exacta=[{"selection":r["selection"],"decimal_odds":r["decimal_odds"]}
                for r in market["quotes"] if r["bet_type"]=="2車単"]
        q,meta=public_first_from_complete_exacta(first,exacta)
        self.assertAlmostEqual(sum(q.values()),1.,places=10)
        self.assertFalse(meta["benter_original_method"])
        with self.assertRaises(ValueError):
            public_first_from_complete_exacta(first,exacta[1:])
        with self.assertRaises(ValueError):
            public_first_from_complete_exacta(first,exacta+[exacta[0]])

    def test_missing_calibration_refuses_expected_value(self):
        card,market,prob=complete_synthetic()
        result=evaluate_race(card,market,mode="live")
        self.assertEqual(result["status"],"CALIBRATION_REQUIRED_NOT_EV_READY")
        self.assertIsNone(result["EV_positive_tickets"])
        self.assertEqual(len(result["fundamental_candidate_distribution"]),504)
        self.assertEqual(len(result["bet_type_book_audit"]),7)
        self.assertIn("NOT_CALCULATED",result["stake_sizing"])
        self.assertEqual(result["odds_audit"]["minutes_before_scheduled_start"],15.)

    def test_t15_snapshot_validation_refuses_timing_and_final_confusion(self):
        card,market,prob=complete_synthetic()
        market["odds_phase"]="final"
        with self.assertRaises(ValueError):
            evaluate_race(card,market,mode="live")
        market["odds_phase"]="T-15"
        market["captured_at_utc"]="2026-10-18T01:02:00+00:00"
        with self.assertRaises(ValueError):
            evaluate_race(card,market,mode="live")

    def test_no_automatically_filled_missing_ticket_quotes(self):
        card,market,prob=complete_synthetic()
        market["quotes"]=market["quotes"][1:]
        with self.assertRaises(ValueError):
            evaluate_race(card,market,mode="live")
        card,market,prob=complete_synthetic()
        market["quotes"]=[x for x in market["quotes"] if x["bet_type"]!="2車単"]
        with self.assertRaises(ValueError):
            evaluate_race(card,market,mode="live")

    def test_calibration_benchmark_is_not_assumed(self):
        f={x["car_no"]:x["probability"] for x in card9()["first_win"]}
        p={i:1./len(f) for i in f}
        c=blended_first(f,p,0.5,1.1)
        self.assertAlmostEqual(sum(c.values()),1.,places=10)
        self.assertNotEqual(c,f)

    def test_bad_calibration_state_must_not_generate_ev_list(self):
        card,market,prob=complete_synthetic()
        with self.assertRaises(ValueError):
            evaluate_race(card,market,calibration={
                "status":"G3_BENTER_PUBLIC_COMBINATION_NOT_YET_VALIDATED",
                "alpha":1.0,"beta":1.0},mode="live")

if __name__=="__main__":unittest.main()
