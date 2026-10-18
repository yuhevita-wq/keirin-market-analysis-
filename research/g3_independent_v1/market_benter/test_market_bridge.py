"""Synthetic-only regression checks of G3 market bridge and Benter logit."""
import json
import math
import sys
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
G3ROOT=ROOT.parent
from engine import (analyze,aggregate_probability,frame_mapping,fuse,
                    inverse_price_public_probability,validate_quotes,_price_phase_gate)
from train_fusion import first_proxy_from_3rentan,parse_record,fit,evaluation
from predict_g3_ordered_triples import all_ordered_triples

SECOND=json.loads((G3ROOT/"models/g3_second_conditional_v1.json").read_text(encoding="utf-8"))
THIRD=json.loads((G3ROOT/"models/g3_third_conditional_v1.json").read_text(encoding="utf-8"))

def card():
    return {"meeting_grade":"G3","race_type":"Ｓ級予選",
            "probability_model":"G3 independent 1st v1",
            "predicted_line_formation":"1-2/3-4/5",
            "first_win":[{"car_no":i+1,"probability":p}
                         for i,p in enumerate((.31,.25,.20,.14,.10))],
            "frame_by_car":{"1":1,"2":2,"3":3,"4":4,"5":5}}

def final_quote():
    return {"price_phase":"final"}

def predeadline_quote(mins=15):
    return {"price_phase":"predeadline",
            "captured_at":"2026-10-18T10:00:00+09:00",
            "bet_close_at":f"2026-10-18T10:{mins:02d}:00+09:00"}

def odds_books():
    combos=all_ordered_triples(card(),SECOND,THIRD)["ordered_triples"]
    books={}
    for name in ("3連単","3連複","2車単","2車複","2枠単","2枠複","ワイド"):
        outcomes=aggregate_probability(combos,name,frame_mapping(card(),set(range(1,6))))
        books[name]={k:float(12+len(outcomes)*2+(sum(map(int,k.split("-")))%7))
                     for k in outcomes}
    return books

class MarketBridge(unittest.TestCase):
    def test_all_exclusive_wagers_aggregate_to_one_wide_is_three(self):
        triples=all_ordered_triples(card(),SECOND,THIRD)["ordered_triples"]
        self.assertEqual(len(triples),60)
        fm=frame_mapping(card(),set(range(1,6)))
        expected={"3連単":60,"3連複":10,"2車単":20,"2車複":10,
                  "2枠単":20,"2枠複":10,"ワイド":10}
        for market,size in expected.items():
            agg=aggregate_probability(triples,market,fm)
            self.assertEqual(len(agg),size)
            self.assertAlmostEqual(sum(agg.values()),3. if market=="ワイド" else 1.,places=9)

    def test_missing_quote_causes_hard_failure(self):
        quotes=odds_books()
        missing=quotes["3連単"].copy()
        missing.pop(next(iter(missing)))
        with self.assertRaises(ValueError):
            analyze(card(),SECOND,THIRD,{"3連単":missing},final_quote(),allow_final_research=True)

    def test_no_fitted_parameters_no_positive_ev_claim(self):
        a=analyze(card(),SECOND,THIRD,{"3連単":odds_books()["3連単"]},
                  final_quote(),allow_final_research=True)
        self.assertEqual(a["status"],"MARKET_COMPARISON_REQUIRES_FITTED_COEFFICIENTS")
        self.assertEqual(a["calibration_missing_markets"],["3連単"])
        for x in a["markets"]["3連単"]["rows"]:
            self.assertIsNone(x["combined_probability"])
            self.assertIsNone(x["expected_return_multiple"])
            self.assertIsNone(x["positive_expected_value"])

    def test_timestamp_and_quote_phase_restrictions(self):
        with self.assertRaises(ValueError):_price_phase_gate(final_quote())
        _price_phase_gate(predeadline_quote())
        with self.assertRaises(ValueError):_price_phase_gate(predeadline_quote(7))
        with self.assertRaises(ValueError):_price_phase_gate({
            "price_phase":"predeadline","captured_at":"2026-10-18T10:00:00",
            "bet_close_at":"2026-10-18T10:15:00"})
        old={"status":"CHRONOLOGICAL_MARKET_FIT_COMPLETED",
             "market":"3連単","price_phase":"final",
             "approved_for_15min_runtime":False,
             "coefficients":{"alpha_fundamental":1.1,"beta_public":.6}}
        with self.assertRaises(ValueError):
            analyze(card(),SECOND,THIRD,{"3連単":odds_books()["3連単"]},
                    predeadline_quote(),{"3連単":old})

    def test_final_price_calibration_can_run_research_only(self):
        cal={"status":"CHRONOLOGICAL_MARKET_FIT_COMPLETED",
             "market":"3連単","price_phase":"final",
             "approved_for_15min_runtime":False,
             "coefficients":{"alpha_fundamental":1.2,"beta_public":-.05}}
        a=analyze(card(),SECOND,THIRD,{"3連単":odds_books()["3連単"]},
                  final_quote(),{"3連単":cal},allow_final_research=True)
        self.assertEqual(a["status"],"RESEARCH_EXPECTED_RETURN_ESTIMATES_READY")
        rr=a["markets"]["3連単"]["rows"]
        self.assertAlmostEqual(sum(r["combined_probability"] for r in rr),1.,places=9)
        self.assertTrue(all(r["expected_return_multiple"]>0 for r in rr))
        self.assertEqual(sum("positive_expected_value" in r for r in rr),60)

    def test_wide_is_never_market_normalized(self):
        with self.assertRaises(ValueError):
            inverse_price_public_probability(odds_books()["ワイド"],"ワイド")
        a=analyze(card(),SECOND,THIRD,{"ワイド":odds_books()["ワイド"]},
                  final_quote(),allow_final_research=True)
        self.assertAlmostEqual(a["markets"]["ワイド"]["sum_fundamental_probabilities"],3.,places=9)
        self.assertIsNone(a["markets"]["ワイド"]["market_inverse_odds_mass"])

    def test_first_win_market_proxy_recover_probability(self):
        proxy,mass=first_proxy_from_3rentan(odds_books()["3連単"],range(1,6))
        self.assertEqual(len(proxy),5)
        self.assertAlmostEqual(sum(proxy.values()),1.,places=10)
        first={str(i):v for i,v in ((1,.31),(2,.25),(3,.2),(4,.14),(5,.1))}
        fused=fuse(first,proxy,1.2,.5)
        self.assertAlmostEqual(sum(fused.values()),1.,places=9)

    def test_train_record_excludes_other_grades_and_unknown_first_source(self):
        triple=all_ordered_triples(card(),SECOND,THIRD)["ordered_triples"]
        fundamental=aggregate_probability(triple,"3連単")
        record={"scope":"G3 S級","market":"3連単","race_id":"synthetic",
                "race_date":"2024-11-01","fundamental_provenance":"G3_only_prior_period_independent",
                "price_phase":"final","fundamental_probabilities":fundamental,
                "odds":odds_books()["3連単"],"actual_winning_selection":"1-2-3"}
        parsed=parse_record(record,"3連単")
        self.assertAlmostEqual(sum(parsed["p"].values()),1.,places=9)
        record["scope"]="F1 S級"
        with self.assertRaises(ValueError):parse_record(record,"3連単")
        record["scope"]="G3 S級";record["fundamental_provenance"]="F1"
        with self.assertRaises(ValueError):parse_record(record,"3連単")

    def test_optim_produces_finite_probabilities(self):
        # A deliberately small SYNTHETIC numerical fixture; not betting evidence.
        data=[]
        for i in range(80):
            fundamental={"1":.15+(.01 if i%2==0 else -.01),"2":.35,"3":.5-(.01 if i%2==0 else -.01)}
            public={"1":.25,"2":.3,"3":.45}
            winner=("3" if i%4 in (0,1) else "2" if i%4==2 else "1")
            data.append({"id":str(i),"date":"2024-02-01","f":fundamental,"p":public,"winner":winner,
                         "model_source_certified_predeadline":False})
        c,info=fit(data)
        self.assertTrue(all(math.isfinite(v) for v in c))
        score=evaluation(data,c)
        self.assertTrue(math.isfinite(score["mean_log_loss_combined"]))
        self.assertLessEqual(score["mean_log_loss_combined"],score["mean_log_loss_public"]+.2)

if __name__=="__main__":
    unittest.main()
