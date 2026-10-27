"""Regression: isolate car markets from frame markets and reject double swaps."""
import itertools, unittest
from research.snowberg_observation_corrected_20261027.observe import verify_page, swap_key
class OrientationTest(unittest.TestCase):
    def fixture(self):
        books={};parts=['<table class="result_table">']
        for rank,car in enumerate((1,2,3),1):
            parts.append(f'<tr><td></td><td>{rank}</td><td>{car}</td></tr>')
        parts.append('</table>')
        for kind,n in (("2shatan",2),("3rentan",3)):
            canonical={t:float(i+2) for i,t in enumerate(itertools.permutations(range(1,6),n))}
            books[kind]={swap_key(t):v for t,v in canonical.items()}
            parts.append(f'<div id="JS_ODDSCONTENTS_{kind}">')
            for t,v in canonical.items():
                parts.append('<span><span class="num">'+'-'.join(map(str,t))+
                             f'</span><span class="odds">{v}</span></span>')
            parts.append('</div>')
        parts.append('<div id="JS_ODDSCONTENTS_2wakutan"><span><span class="num">1-2</span><span class="odds">9999</span></span></div>')
        return ''.join(parts),books
    def test_frame_market_does_not_overwrite_car_market(self):
        html,books=self.fixture()
        result=verify_page(html,"synthetic",books,(1,2,3))
        self.assertEqual(result["2shatan"]["corrected_matches"],20)
        self.assertEqual(result["3rentan"]["corrected_matches"],60)
    def test_wrong_result_is_rejected(self):
        html,books=self.fixture()
        with self.assertRaises(RuntimeError):verify_page(html,"synthetic",books,(2,1,3))
    def test_double_swap_is_rejected(self):
        html,books=self.fixture()
        corrected={k:{swap_key(t):v for t,v in b.items()} for k,b in books.items()}
        with self.assertRaises(RuntimeError):verify_page(html,"synthetic",corrected,(1,2,3))
if __name__=="__main__":unittest.main()
