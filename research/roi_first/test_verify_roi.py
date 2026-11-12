import csv
import tempfile
import unittest
from pathlib import Path
from verify_roi import audit

HEADER = ['race_id', 'race_date', 'bet_type', 'combination', 'stake_yen',
          'returned_yen', 'settlement_verified', 'price_phase']
BASE = {'goal': 'maximize_realized_official_roi', 'strategy_sha256': 'abc123',
        'test_start': '2025-04-20', 'test_end': '2025-08-09',
        'claimed_new_untouched_test': True,
        'policy_tuned_using_test_results': False}

class TestROI(unittest.TestCase):
    def make(self, rows, cfg=None):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        path = Path(tmp.name)/'ledger.csv'
        with path.open('w', newline='') as f:
            w=csv.DictWriter(f, fieldnames=HEADER)
            w.writeheader(); w.writerows(rows)
        return audit(path, cfg or BASE)
    def test_zero_bets_never_profitable(self):
        r=self.make([])
        self.assertIsNone(r['realized_roi'])
        self.assertEqual(r['status'],'NO_BETS_ROI_UNDEFINED')
    def test_two_markets_not_fixed_point_count(self):
        a={'race_id':'x','race_date':'2025-05-01', 'bet_type':'trifecta',
           'combination':'1-2-3','stake_yen':'100','returned_yen':'0',
           'settlement_verified':'true','price_phase':'preclose_timestamp_verified'}
        b={**a,'combination':'1=2=3','bet_type':'trio','stake_yen':'300',
           'returned_yen':'630'}
        r=self.make([a,b])
        self.assertEqual(r['realized_roi'],1.575)
        self.assertEqual(r['net_yen'],'230')
        self.assertEqual(len(r['market_breakdown']),2)
        self.assertEqual(r['status'],'OBSERVED_POSITIVE_ROI_NOT_PROVEN')
    def test_final_odds_not_deployable(self):
        a={'race_id':'x','race_date':'2025-05-01','bet_type':'trifecta',
           'combination':'1-2-3','stake_yen':'100','returned_yen':'200',
           'settlement_verified':'true','price_phase':'retrospective_final'}
        self.assertTrue(self.make([a])['status'].startswith('RETROSPECTIVE_RESEARCH_ONLY'))
    def test_no_official_settlement_rejected(self):
        a={'race_id':'x','race_date':'2025-05-01','bet_type':'trifecta',
           'combination':'1-2-3','stake_yen':'100','returned_yen':'120',
           'settlement_verified':'false','price_phase':'retrospective_final'}
        with self.assertRaisesRegex(ValueError,'UNVERIFIED'):
            self.make([a])
    def test_looked_at_april_not_fresh(self):
        cfg=dict(BASE,test_start='2025-04-01',test_end='2025-04-19')
        with self.assertRaisesRegex(ValueError,'ALREADY_EXAMINED'):
            self.make([],cfg)
    def test_hindsight_strategy_selection(self):
        cfg=dict(BASE,policy_tuned_using_test_results=True)
        with self.assertRaisesRegex(ValueError,'HINDSIGHT'):
            self.make([],cfg)

if __name__ == '__main__': unittest.main()
