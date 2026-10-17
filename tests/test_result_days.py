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
        self.assertEqual(meta,{"cards":3,"girls_excluded":1,"men_podiums":2})

    def test_missing_podium_cannot_borrow_next_race_table(self):
        bad=HTML.replace('<table class="order_table">', '<div class="hidden">',1).replace('</table>','</div>',1)
        with self.assertRaises(ValueError):
            C.venue_podiums(bad,VENUE,DAY)

    def test_reject_duplicate_cars(self):
        bad=HTML.replace('3 助川 翔太郎','7 助川 翔太郎')
        with self.assertRaises(ValueError):
            C.venue_podiums(bad,VENUE,DAY)

    def test_race_id_date_identity(self):
        self.assertEqual(C.race_date_from_id("1120230721010001"),DAY)
        self.assertNotEqual(C.race_date_from_id("1120230721020001"),DAY)

if __name__=="__main__":
    unittest.main()
