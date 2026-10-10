from __future__ import annotations

"""v8.25-F26: FINAL SET LOCK x ORDER SPLIT promoted; HEAD LOCK diagnostic-only.

All non-FINAL branches inherit v8.23-F24 unchanged.

Development decision after the predefined v8.24 comparison:
- HEAD LOCK x TAIL SPLIT: Q1 ROI < 100%, therefore not eligible for adoption.
- SET LOCK x ORDER SPLIT: positive in both Q1 and Q2, therefore retained as the
  only active FINAL purchase branch.

No new fitted cutoff is introduced here. The active FINAL rule is the exact
v8.24 semantic SET LOCK x ORDER SPLIT rule.
"""

from v7_0_f01_market_hierarchy import implied_probabilities
from v8_23_f24_selection_ultra_roi import build_v8_23_f24
from v8_24_f25_final_order_hierarchy import _head_support, _set_lock_order_split

SCHEME_VERSION='v8.25-F26'
BASE_SCHEME='v8.23-F24'
STATUS='DEVELOPMENT_Q1Q2_FINAL_SET_LOCK_ONLY'


def _build_final(trio_odds,trifecta_odds,predicted_line_formation,race_type):
    q=implied_probabilities(trifecta_odds);h=_head_support(q)
    hr=tuple(sorted(h,key=lambda x:(-h[x],x)))
    if len(hr)<2:
        return {'scheme_version':SCHEME_VERSION,'buy':False,'reason':'FINAL_MISSING_H_TOP2','race_type':race_type,'race_type_group':'FINAL'}
    if h[hr[0]] >= 2*h[hr[1]]:
        return {
            'scheme_version':SCHEME_VERSION,'buy':False,
            'reason':'FINAL_HEAD_LOCK_DIAGNOSTIC_ONLY_Q1_FAIL',
            'race_type':race_type,'race_type_group':'FINAL',
            'global_H1':hr[0],'global_H2':hr[1],
            'H1_top':h[hr[0]],'H1_second':h[hr[1]],
            'branch_policy':'HEAD_LOCK x TAIL_SPLIT retained as diagnostic only after Q1 ROI below 100',
            'development_status':STATUS,
        }
    out=_set_lock_order_split(trio_odds,trifecta_odds,predicted_line_formation,race_type,q,h,hr)
    out=dict(out);out['scheme_version']=SCHEME_VERSION
    if out.get('buy'):
        out['branch_policy']='FINAL ACTIVE: SET_LOCK x ORDER_SPLIT only; top trio set equals top collapsed-TF set; H1/H2 heads; longer tail order; clean rectangle only'
        out['development_status']=STATUS
    return out


def build_v8_25_f26(trio_odds,trifecta_odds,predicted_line_formation: str,race_type: str):
    if (race_type or '').strip()=='Ｓ級決勝':
        return _build_final(trio_odds,trifecta_odds,predicted_line_formation,race_type)
    out=build_v8_23_f24(trio_odds,trifecta_odds,predicted_line_formation,race_type)
    out=dict(out);out['scheme_version']=SCHEME_VERSION
    return out
