from __future__ import annotations

"""v8.14-F15: GENERAL branch preserves the H1 anchor identified at entry.

This is NOT a global fixed-one-rider rule.  GENERAL enters only when the first-place
market has a genuine concentration cliff: H1_top >= 2*H1_second and H_AB is aligned.
Once that market fact is used to enter the race, formation construction is not allowed
to later contradict it by crossing that same H1 cliff.

Order:
1. PS_AB + H_CONCENTRATED + H_AB GENERAL entry.
2. Build the normal F09 market-cliff rectangle without price pruning.
3. Apply a market-structure correction: retain the H1 dominant block in first place.
   Under the current GENERAL entry the dominant block ends at the top rider because
   the top/second support ratio is already >=2.
4. Only then run the existing structural price-compression knee.  Whole-rider removal
   only; no individual ticket pruning.
"""

from v8_11_f12_race_type_adaptive import classify_race_type
from v8_12_f13_branch_rebuild import build_v8_12_f13
from v8_8_f09_market_cliff import build_v8_8_f09, _state
from v8_10_f11_structural_price_compression import _price_state, _compression_path, _select_structural_knee
from v7_0_f01_market_hierarchy import implied_probabilities

SCHEME_VERSION='v8.14-F15'


def build_v8_14_f15(trio_odds,trifecta_odds,predicted_line_formation,race_type:str):
    group=classify_race_type(race_type)
    if group!='GENERAL':
        out=build_v8_12_f13(trio_odds,trifecta_odds,predicted_line_formation,race_type)
        return {**out,'scheme_version':SCHEME_VERSION}

    base=build_v8_8_f09(trio_odds,trifecta_odds,predicted_line_formation)
    if not base.get('buy'):
        return {**base,'scheme_version':SCHEME_VERSION,'race_type':race_type,'race_type_group':'GENERAL'}
    eg=base.get('entry_gate') or {}
    if bool(eg.get('H_RATIO')) or not bool(eg.get('H_AB')):
        return {
            'scheme_version':SCHEME_VERSION,'buy':False,'reason':'GENERAL_ENTRY_FAIL_H_STRUCTURE',
            'race_type':race_type,'race_type_group':'GENERAL','entry_gate':eg,
            'formation_before_anchor':base.get('formation'),'ticket_count_before_anchor':base.get('ticket_count')
        }

    anchor=int(eg['top2_H'][0])
    q=implied_probabilities(trifecta_odds)
    anchored=_state((anchor,),base['second'],base['third'],q)
    if anchored is None:
        return {
            'scheme_version':SCHEME_VERSION,'buy':False,'reason':'GENERAL_INVALID_ANCHORED_RECTANGLE',
            'race_type':race_type,'race_type_group':'GENERAL','entry_gate':eg,
            'formation_before_anchor':base.get('formation')
        }

    # Price phase begins only after the clean anchored formation exists.
    p0=_price_state(anchored,anchored.q_mass,q,trifecta_odds)
    states,moves=_compression_path(p0,q,trifecta_odds)
    k=_select_structural_knee(states)
    chosen=states[k]
    return {
        **base,'scheme_version':SCHEME_VERSION,'race_type':race_type,'race_type_group':'GENERAL',
        'branch_policy':'GENERAL_H1_ANCHOR_PRESERVED',
        'formation_before_anchor':base['formation'],'ticket_count_before_anchor':base['ticket_count'],
        'H1_anchor':anchor,'formation_before_price':anchored.display,'ticket_count_before_price':anchored.ticket_count,
        'formation':chosen.display,'first':chosen.state.first,'second':chosen.state.second,'third':chosen.state.third,
        'tickets':chosen.state.tickets,'ticket_count':chosen.state.ticket_count,'q_mass':chosen.state.q_mass,
        'price_compression_steps':k,'price_q_retention':chosen.q_retention,
        'price_weighted_gm_odds':chosen.weighted_gm_odds,'price_gm_return_multiple':chosen.gm_return_multiple,
        'price_profitable_q_share':chosen.profitable_q_share,
        'general_anchor_rule':'do not cross the H1 top/second concentration cliff used by GENERAL entry',
        'individual_ticket_pruning':False,'fixed_first_count_rule':False,
    }
