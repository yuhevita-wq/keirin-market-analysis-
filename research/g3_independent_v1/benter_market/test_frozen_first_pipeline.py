"""Unit-only synthetic raw G3 cards exercise frozen first->market pipeline."""
import json,sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from predict_g3_first import independent_first,DEFAULT_MODEL
from evaluate import evaluate_race
from test_market_foundation import complete_synthetic

MODEL=json.loads(DEFAULT_MODEL.read_text(encoding="utf-8"))

def raw_card():
    entries=[]
    for i in range(1,10):
        entries.append({
            "car_no":i,"score":112.0-i,"age":25+i,"term":999 if i==5 else 80+i,
            "b_count":10 if i in (1,4,7) else 1,"s_count":i%3,
            "nige_count":5 if i in (1,4,7) else 0,
            "makuri_count":i%4,"sashi_count":i%3,
            "mark_count":i%5,"first_count":8+i,
            "second_count":4+i%3,"third_count":6+i%2,
            "outside_count":10,
            "win_rate":10+i,"top2_rate":20+i,"top3_rate":30+i,
            "gear":3.92
        })
    return {
        "race_id":"SYNTHETIC-G3-TEST-NOT-A-REAL-RACE",
        "meeting_grade":"G3",
        "race_type":"Ｓ級二次予選",
        "source_url":"https://example.test/synthetic-fixture",
        "snapshot_captured_at_utc":"2026-10-18T00:45:00+00:00",
        "predicted_line_formation":"1-2-3/4-5-6/7-8-9",
        "riders":entries
    }

class FrozenG3FirstPipelineTest(unittest.TestCase):
    def test_infer_frozen_G3_first_from_raw_entries(self):
        q=independent_first(raw_card(),MODEL)
        self.assertEqual(len(q["first_win"]),9)
        self.assertAlmostEqual(sum(x["probability"] for x in q["first_win"]),1,places=10)
        self.assertEqual(q["probability_model"],"G3 independent 1st v1")
        self.assertFalse(q["used_market_odds"])
        self.assertFalse(q["used_future_result_features"])
        self.assertEqual(q,independent_first(raw_card(),MODEL))

    def test_first_to_seven_market_without_guessed_calibration(self):
        q=independent_first(raw_card(),MODEL)
        _,market,_=complete_synthetic()
        output=evaluate_race(q,market,calibration=None,mode="live")
        self.assertEqual(output["status"],"CALIBRATION_REQUIRED_NOT_EV_READY")
        self.assertEqual(len(output["bet_type_book_audit"]),7)
        self.assertIsNone(output["EV_positive_tickets"])

    def test_source_required_and_no_outcome_features(self):
        card=raw_card()
        del card["source_url"]
        with self.assertRaises(ValueError):
            independent_first(card,MODEL)
        card=raw_card()
        card["actual_first"]=2
        with self.assertRaises(ValueError):
            independent_first(card,MODEL)
        card=raw_card()
        card["riders"][0]["finish_position"]=1
        with self.assertRaises(ValueError):
            independent_first(card,MODEL)

    def test_no_other_grade_or_unproved_lineups(self):
        card=raw_card()
        card["meeting_grade"]="F1"
        with self.assertRaises(ValueError):
            independent_first(card,MODEL)
        card=raw_card()
        card["predicted_line_formation"]="1-2-3/4-5-6/7-8"
        with self.assertRaises(ValueError):
            independent_first(card,MODEL)

if __name__=="__main__":
    unittest.main()
