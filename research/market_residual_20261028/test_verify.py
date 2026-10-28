import unittest
from research.market_residual_20261028.verify import calculate

class FormulaTests(unittest.TestCase):
    def test_fair_market(self):
        self.assertEqual(calculate(1000, 100, 100, 750)['D'], 0)
    def test_underpriced_but_losing(self):
        r = calculate(1000, 120, 100, 900)
        self.assertEqual(r['D'], 20)
        self.assertEqual(r['roi_pct_final_odds_proxy'], 90)
    def test_opposite_errors_cancel(self):
        self.assertEqual(calculate(100, 20, 10, 100)['D'] + calculate(100, 80, 90, 100)['D'], 0)
    def test_invalid_statistics(self):
        with self.assertRaises(ValueError):
            calculate(10, 11, 1, 0)

if __name__ == '__main__':
    unittest.main()
