"""No invented podiums: test real site-shaped published summary fixtures."""
import importlib.util
import unittest
from datetime import date
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location("collect_result_days",ROOT/"research/result_alignment_v2/collect_result_days.py")
C=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(C)
DAY=date(2023,7,21)
VENUE="https://keirin.kdreams.jp/hakodate/raceresult/11202307210100/"

HTML='''
<p class="race"><a href="/hakodate/racedetail/1120230721010001/">1R チャレンジ予選</a></p>
<table class="order_table"><tr><th>1着</th><th>2着</th><th>3着</th><th>4</th></tr>
<tr><td>7 木村 元信</td><td>1 小原 伸哉</td><td>3 助川 翔太郎</td><td>6 その他</td></tr></table>
<p class="race"><a href="/hakodate/racedetail/1120230721010002/">2R ガールズ予選</a></p>
<table class="order_table"><tr><th>1着</th><th>2着</th><th>3着</th></tr><tr><td>4 A</td><td>2 B</td><td>1 C</td></tr></table>
<p class="race"><a href="/hakodate/racedetail/1120230721010003/">3R A級予選</a></p>
<table class="order_table"><tr><th>1着</th><th>2着</th><th>3着</th></tr><tr><td>2 X</td><td>5 Y</td><td>6 Z</td></tr></table>
'''

class ResultDayTests(unittest.TestCase):
    def test_index_meeting_discovery_exact_date(self):
        html='''<a href="/hakodate/raceresult/11202307210100/">一覧</a>
        <a href="/hakodate/raceresult/11202307210200/">一覧</a>'''
        self.assertEqual(C.daily_meetings(html,DAY),{"11202307210100":VENUE})

    def test_published_podium_and_skip_girls(self):
        got,meta=C.venue_podiums(HTML,VENUE,DAY)
        self.assertEqual(got["1120230721010001"]["finish_order"],[7,1,3])
        self.assertEqual(got["1120230721010003"]["finish_order"],[2,5,6])
        self.assertNotIn("1120230721010002",got)
        self.assertEqual(meta,{"cards":3,"girls_excluded":1,"men_podiums":2,"unresolved_podiums":[]})

    def test_missing_podium_cannot_borrow_next_race_table(self):
        bad=HTML.replace('<table class="order_table">', '<div class="hidden">',1).replace('</table>','</div>',1)
        good,meta=C.venue_podiums(bad,VENUE,DAY)
        self.assertNotIn("1120230721010001",good)
        self.assertEqual(len(meta["unresolved_podiums"]),1)

    def test_reject_duplicate_cars(self):
        bad=HTML.replace('3 助川 翔太郎','7 助川 翔太郎')
        good,meta=C.venue_podiums(bad,VENUE,DAY)
        self.assertNotIn("1120230721010001",good)
        self.assertEqual(len(meta["unresolved_podiums"]),1)

    def test_unverifiable_podium_is_excluded_not_assumed_cancelled(self):
        item=[["1120230721010002","published first-second-third headers not present"]]
        records=C.excludable_entries(item,VENUE,DAY)
        self.assertEqual(records["1120230721010002"]["status"],
                         "excluded_unverified_standard_podium")
        self.assertNotIn("cancelled",records["1120230721010002"]["reason"])

    def test_bad_identity_or_unknown_parser_error_never_excluded(self):
        wrong=[["1120230721020002","published first-second-third headers not present"]]
        self.assertIsNone(C.excludable_entries(wrong,VENUE,DAY))
        conflict=[["1120230721010002","conflicting podium in same meeting"]]
        self.assertIsNone(C.excludable_entries(conflict,VENUE,DAY))

    def test_prior_parsed_failure_reconciles_without_refetch(self):
        state={
          "index_verified":True,
          "published_venue_count":2,
          "venues_completed":["11202307210100"],
          "recent_errors":[{
              "stage":"race","venue":"23202307200200",
              "url":"https://keirin.kdreams.jp/toride/raceresult/23202307200200/",
              "error":"Unresolved published podiums "+
                 '[["2320230720020002", "no results order_table"]]'
          }]
        }
        self.assertTrue(C.accept_previous_exclusions(state,DAY))
        self.assertTrue(state["complete"])
        self.assertEqual(len(state["excluded_nonstandard_races"]),1)
        self.assertEqual(len(state["venues_completed"]),2)

    def test_network_error_must_remain_unresolved(self):
        state={
          "index_verified":True,
          "published_venue_count":2,
          "venues_completed":["11202307210100"],
          "recent_errors":[{"stage":"venue","venue":"23202307200200",
                            "error":"Timeout: HTTP request failed"}]
        }
        self.assertFalse(C.accept_previous_exclusions(state,DAY))
        self.assertNotIn("complete",state)

    def test_race_id_date_identity(self):
        self.assertEqual(C.race_date_from_id("1120230721010001"),DAY)
        self.assertNotEqual(C.race_date_from_id("1120230721020001"),DAY)

if __name__=="__main__":
    unittest.main()
