from __future__ import annotations

"""v8.13-F14: preserve Q/semifinal branches and rebuild GENERAL price stage.

GENERAL design:
- entry remains PS_AB + H_CONCENTRATED + H_AB;
- build the full clean market-cliff formation first;
- only then inspect prices;
- walk the existing whole-rider structural compression path;
- select the earliest clean rectangle satisfying both natural price-viability conditions:
    gm_return_multiple >= 1.0
    profitable_q_share >= 0.5
- if no state satisfies both, skip the race.

The boundaries are semantic, not Q1-fitted: 1.0 is formation-level break-even and
0.5 means a majority of retained market support lies on tickets individually priced
above the formation's total stake multiple.
"""

from v8_11_f12_race_type_adaptive import classify_race_type
from v8_12_f13_branch_rebuild import build_v8_12_f13
from v8_8_f09_market_cliff import build_v8_8_f09, _state
from v8_10_f11_structural_price_compression import _price_state, _compression_path
from v7_0_f01_market_hierarchy import implied_probabilities

SCHEME_VERSION='v8.13-F14'


def _general_entry_ok(base):
    eg=base.get('entry_gate') or {}
    return (not bool(eg.get('H_RATIO'))) and bool(eg.get('H_AB'))


def _general_price_select(base,trifecta_odds):
    q=implied_probabilities(trifecta_odds)
    s=_state(base['first'],base['second'],base['third'],q)
    if s is None:return None,[]
    p0=_price_state(s,s.q_mass,q,trifecta_odds)
    states,moves=_compression_path(p0,q,trifecta_odds)
    for i,ps in enumerate(states):
        if ps.gm_return_multiple >= 1.0 and ps.profitable_q_share >= 0.5:
            return (i,ps),states
    return None,states


def build_v8_13_f14(trio_odds,trifecta_odds,predicted_line_formation,race_type:str):
    group=classify_race_type(race_type)
    if group!='GENERAL':
        out=build_v8_12_f13(trio_odds,trifecta_odds,predicted_line_formation,race_type)
        return {**out,'scheme_version':SCHEME_VERSION}

    base=build_v8_8_f09(trio_odds,trifecta_odds,predicted_line_formation)
    if not base.get('buy'):
        return {**base,'scheme_version':SCHEME_VERSION,'race_type':race_type,'race_type_group':'GENERAL'}
    if not _general_entry_ok(base):
        return {
            'scheme_version':SCHEME_VERSION,'buy':False,'reason':'GENERAL_ENTRY_FAIL_H_STRUCTURE',
            'race_type':race_type,'race_type_group':'GENERAL','entry_gate':base.get('entry_gate'),
            'formation_before_price':base.get('formation'),'ticket_count_before_price':base.get('ticket_count')
        }
    selected,states=_general_price_select(base,trifecta_odds)
    if selected is None:
        return {
            'scheme_version':SCHEME_VERSION,'buy':False,'reason':'GENERAL_NO_PRICE_VIABLE_RECTANGLE',
            'race_type':race_type,'race_type_group':'GENERAL','entry_gate':base.get('entry_gate'),
            'formation_before_price':base.get('formation'),'ticket_count_before_price':base.get('ticket_count')
        }
    i,chosen=selected
    return {
        **base,'scheme_version':SCHEME_VERSION,'race_type':race_type,'race_type_group':'GENERAL',
        'branch_policy':'GENERAL_REBUILT_PRICE_VIABILITY',
        'formation_before_price':base['formation'],'ticket_count_before_price':base['ticket_count'],
        'formation':chosen.display,'first':chosen.state.first,'second':chosen.state.second,'third':chosen.state.third,
        'tickets':chosen.state.tickets,'ticket_count':chosen.state.ticket_count,'q_mass':chosen.state.q_mass,
        'price_compression_steps':i,'price_q_retention':chosen.q_retention,
        'price_weighted_gm_odds':chosen.weighted_gm_odds,'price_gm_return_multiple':chosen.gm_return_multiple,
        'price_profitable_q_share':chosen.profitable_q_share,
        'general_price_rule':'earliest clean rectangle with GM_return>=1.0 and profitable_q_share>=0.5',
        'individual_ticket_pruning':False,
    }
