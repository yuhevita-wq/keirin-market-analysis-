"""Test self-contained G3 second independent probability; no historical event involved."""
import json,sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from predict_g3_first_second import infer,conditional_position

MODEL=json.loads((ROOT/"models/g3_second_conditional_v1.json").read_text(encoding="utf-8"))
def card():
    q=[.23,.18,.16,.14,.11,.08,.06,.04]
    return {
        "meeting_grade":"G3","race_type":"Ｓ級一予選",
        "probability_model":"G3 independent 1st v1",
        "predicted_line_formation":"1-2-3/4-5/6-7-8",
        "first_win":[{"car_no":i+1,"probability":p} for i,p in enumerate(q)]
    }

class TestG3Second(unittest.TestCase):
    def test_56_distinct_ordered_outcomes(self):
        output=infer(card(),MODEL)
        self.assertEqual(output["ordered_first_second_pair_count"],56)
        self.assertEqual(len(output["pairs"]),56)
        self.assertEqual(len({(p["first_car_no"],p["second_car_no"]) for p in output["pairs"]}),56)
        self.assertTrue(all(p["first_car_no"]!=p["second_car_no"] for p in output["pairs"]))
        self.assertAlmostEqual(sum(p["joint_first_second_probability"] for p in output["pairs"]),1,places=10)
        for first in output["first_marginal"]:
            joint=sum(p["joint_first_second_probability"] for p in output["pairs"]
                      if p["first_car_no"]==first["car_no"])
            self.assertAlmostEqual(first["probability"],joint,places=10)
        self.assertAlmostEqual(sum(p["probability"] for p in output["second_marginal"]),1,places=10)
        self.assertEqual(output,infer(card(),MODEL))

    def test_roles_and_line_pos_are_reproducible(self):
        spots={1:(0,0),2:(0,1),3:(0,2),4:(1,0),5:(1,1),6:(2,0)}
        self.assertEqual(conditional_position(1,2,spots),"directly_behind_first")
        self.assertEqual(conditional_position(1,3,spots),"further_behind_first")
        self.assertEqual(conditional_position(2,1,spots),"ahead_of_first")
        self.assertEqual(conditional_position(1,4,spots),"other_line")

    def test_results_odds_and_other_grades_rejected(self):
        for field,value in (("odds",{}),("result",{}),("second",2)):
            c=card();c[field]=value
            with self.assertRaises(ValueError):infer(c,MODEL)
        for grade,typ in (("F1","Ｓ級予選"),("G3","Ａ級予選"),("G2","Ｓ級予選")):
            c=card();c["meeting_grade"]=grade;c["race_type"]=typ
            with self.assertRaises(ValueError):infer(c,MODEL)
        c=card();c["probability_model"]="F1 adopted model"
        with self.assertRaises(ValueError):infer(c,MODEL)

    def test_bad_probabilities_lineup_and_duplicate_car_blocked(self):
        c=card();c["first_win"][0]["probability"]=.24
        with self.assertRaises(ValueError):infer(c,MODEL)
        c=card();c["predicted_line_formation"]="1-2-3/4-5/6-7-7"
        with self.assertRaises(ValueError):infer(c,MODEL)

    def test_research_model_is_g3_and_conditioned_result_only(self):
        self.assertFalse(MODEL["first_f1_coefficients_imported"])
        self.assertFalse(MODEL["second_f1_coefficients_imported"])
        self.assertEqual(MODEL["all_G3_second_eligible_races"],8967)
        self.assertEqual(MODEL["all_G3_second_candidate_rows"],69620)
        self.assertEqual(MODEL["final_second_coefficient_training_2023_2025"],5930)
        self.assertEqual(MODEL["evaluation_2026h1"],1083)
        self.assertEqual(MODEL["coefficients"]["gamma"],0.6389831175236712)

if __name__=="__main__":
    unittest.main()
