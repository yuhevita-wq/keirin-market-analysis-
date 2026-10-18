"""Independent G3 third conditional probability inference, all triple permutations."""
import json,sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from predict_g3_ordered_triples import all_ordered_triples,third_position,third_role
from test_g3_second import card as card8

SECOND=json.loads((ROOT/"models/g3_second_conditional_v1.json").read_text(encoding="utf-8"))
THIRD=json.loads((ROOT/"models/g3_third_conditional_v1.json").read_text(encoding="utf-8"))

def card9():
    q=(.21,.17,.14,.13,.105,.085,.07,.05,.04)
    return {
        "meeting_grade":"G3","race_type":"Ｓ級選抜",
        "probability_model":"G3 independent 1st v1",
        "predicted_line_formation":"1-2-3/4-5-6/7-8-9",
        "first_win":[{"car_no":i+1,"probability":v} for i,v in enumerate(q)]
    }

class G3IndependentThirdTest(unittest.TestCase):
    def test_9_rider_all_504_triples(self):
        result=all_ordered_triples(card9(),SECOND,THIRD)
        combos=result["ordered_triples"]
        self.assertEqual(len(combos),9*8*7)
        self.assertEqual(result["all_distinct_triples_count"],504)
        self.assertEqual(len({(p["first_car_no"],p["second_car_no"],p["third_car_no"])
                              for p in combos}),504)
        self.assertTrue(all(len({p["first_car_no"],p["second_car_no"],p["third_car_no"]})==3
                            for p in combos))
        self.assertAlmostEqual(sum(p["joint_first_second_third_probability"]
                                   for p in combos),1.0,places=10)
        for k in ("first_marginal","second_marginal","third_marginal"):
            self.assertAlmostEqual(sum(x["probability"] for x in result[k]),1.0,places=10)
        for x in result["first_marginal"]:
            p=sum(t["joint_first_second_third_probability"] for t in combos
                  if t["first_car_no"]==x["car_no"])
            self.assertAlmostEqual(x["probability"],p,places=10)
        self.assertEqual(result,all_ordered_triples(card9(),SECOND,THIRD))

    def test_8_rider_336_triples(self):
        result=all_ordered_triples(card8(),SECOND,THIRD)
        self.assertEqual(result["all_distinct_triples_count"],8*7*6)

    def test_third_position_exactly_one_category(self):
        group=[[1,2,3],[4,5,6],[7,8,9]]
        pos={car:(li,i) for li,line in enumerate(group) for i,car in enumerate(line)}
        self.assertEqual(third_position(1,2,3,pos),"direction_behind_both")
        self.assertEqual(third_position(2,3,1,pos),"direction_ahead_of_both")
        self.assertEqual(third_position(1,3,2,pos),"direction_between_first_second")
        self.assertEqual(third_position(1,4,2,pos),"direction_directly_behind_first")
        self.assertEqual(third_position(1,4,3,pos),"direction_further_behind_first")
        self.assertEqual(third_position(2,4,1,pos),"direction_ahead_of_first")
        self.assertEqual(third_position(4,1,2,pos),"direction_directly_behind_second")
        self.assertEqual(third_position(4,1,3,pos),"direction_further_behind_second")
        self.assertEqual(third_position(4,2,1,pos),"direction_ahead_of_second")
        self.assertIsNone(third_position(1,4,7,pos))
        self.assertEqual(third_role(1,pos,group),None)
        self.assertEqual(third_role(2,pos,group),"role_second")
        self.assertEqual(third_role(3,pos,group),"role_third_or_later")
        solo=[[1,2,3],[4,5],[6]]
        positions={car:(li,k) for li,line in enumerate(solo) for k,car in enumerate(line)}
        self.assertEqual(third_role(6,positions,solo),"role_solo")

    def test_no_results_odds_or_other_grades(self):
        for k,v in (("results",{}),("odds",{}),("first",1),("third",3)):
            c=card9();c[k]=v
            with self.assertRaises(ValueError):
                all_ordered_triples(c,SECOND,THIRD)
        for grade,typ in (("F1","Ｓ級予選"),("G1","Ｓ級選抜"),("G3","Ａ級予選")):
            c=card9();c["meeting_grade"]=grade;c["race_type"]=typ
            with self.assertRaises(ValueError):
                all_ordered_triples(c,SECOND,THIRD)

    def test_frozen_model_scope_and_counts(self):
        self.assertEqual(THIRD["empirical_population"],8945)
        self.assertEqual(THIRD["empirical_candidate_rows"],60506)
        self.assertEqual(THIRD["selected_family"],"g3_delta_plus_direction_and_role")
        self.assertEqual(THIRD["refit_2023_25"],5916)
        self.assertEqual(THIRD["chronological_2026h1"],1081)
        self.assertEqual(len(THIRD["parameter_vector"]),13)
        self.assertFalse(THIRD["old_f1_or_g3_third_coefficients_reused"])
        self.assertTrue(THIRD["first_and_second_frozen_models_unchanged"])

if __name__=="__main__":
    unittest.main()
