from __future__ import annotations

"""v8.23-F24 candidate: Selection ultra-ROI one-ticket disagreement branch.

All non-SELECTION branches inherit v8.22-F23 unchanged.

SELECTION objective is deliberately different from qualifying:
- low hit rate is acceptable;
- maximize return multiple with minimal stake;
- use cross-market disagreement rather than consensus strength.

Rule (Q1+Q2 development semantics; no fitted numeric cutoff):
1. Race type must be exact `Ｓ級選抜`.
2. H market must be BALANCED: H1 < 2*H2.
3. Rank the 35 trio sets by P_trio. Among trio top3, keep sets with D_log < 0,
   where D_log = log(Q_tf_set / P_trio). Choose the most negative D set.
4. Target set must include global H1.
5. Target set must span exactly two predicted lines.
6. Within target, first place = highest-H rider.
7. The other two riders have two possible tail orders. Buy ONLY the higher-odds
   (less-supported) of those two orders: exactly one trifecta ticket.

This is a sparse contrarian order translation of a trio-supported but
collapsed-trifecta-discounted set. No result/payout input, no individual-ticket
pruning after formation, and no fitted odds threshold.
"""

from math import log
from collections import defaultdict

from v7_0_f01_market_hierarchy import implied_probabilities, trio_implied_probabilities
from v8_11_f12_race_type_adaptive import classify_race_type
from v8_17_f18_market_semantics import selection_divergence_diagnostics
from v8_22_f23_qualifying_single_pole import build_v8_22_f23

SCHEME_VERSION='v8.23-F24'
BASE_SCHEME='v8.22-F23'
STATUS='DEVELOPMENT_Q1Q2_SELECTION_ULTRA_ROI'


def _lines(predicted_line_formation: str):
    raw=(predicted_line_formation or '').strip()
    if not raw:return None
    out=[]
    try:
        for block in raw.split('/'):
            vals=tuple(int(ch) for ch in block if ch.isdigit())
            if vals:out.append(vals)
    except Exception:
        return None
    return out or None


def _h_support(tf_odds):
    q=implied_probabilities(tf_odds); h=defaultdict(float)
    for t,p in q.items():h[t[0]]+=p
    return dict(h)


def _build_selection(trio_odds,trifecta_odds,predicted_line_formation,race_type):
    p=trio_implied_probabilities(trio_odds)
    q=implied_probabilities(trifecta_odds)
    h=_h_support(trifecta_odds)
    hr=tuple(sorted(h,key=lambda x:(-h[x],x)))
    if len(hr)<2:
        return {'scheme_version':SCHEME_VERSION,'buy':False,'reason':'SELECTION_MISSING_H_TOP2','race_type':race_type,'race_type_group':'SPECIAL','special_subtype':'SELECTION'}
    h1,h2=hr[:2]
    if h[h1] >= 2*h[h2]:
        return {'scheme_version':SCHEME_VERSION,'buy':False,'reason':'SELECTION_H_NOT_BALANCED','race_type':race_type,'race_type_group':'SPECIAL','special_subtype':'SELECTION','branch_policy':'ULTRA_ROI_H_BALANCED_DISCOUNTED_SET_LONGER_ORDER'}

    diag=selection_divergence_diagnostics(trio_odds,trifecta_odds)
    rows={tuple(r['set']):r for r in diag['set_rows']}
    p_rank=tuple(sorted(p,key=lambda c:(-p[c],c)))
    neg=[c for c in p_rank[:3] if rows[c]['D_log'] is not None and rows[c]['D_log']<0]
    if not neg:
        return {'scheme_version':SCHEME_VERSION,'buy':False,'reason':'SELECTION_NO_NEGATIVE_D_IN_TRIO_TOP3','race_type':race_type,'race_type_group':'SPECIAL','special_subtype':'SELECTION','selection_diagnostics':diag}
    target=min(neg,key=lambda c:(rows[c]['D_log'],p_rank.index(c),c))
    if h1 not in target:
        return {'scheme_version':SCHEME_VERSION,'buy':False,'reason':'SELECTION_TARGET_EXCLUDES_GLOBAL_H1','race_type':race_type,'race_type_group':'SPECIAL','special_subtype':'SELECTION','target_set':target,'target_D_log':rows[target]['D_log']}

    lines=_lines(predicted_line_formation)
    if lines is None:
        return {'scheme_version':SCHEME_VERSION,'buy':False,'reason':'SELECTION_INVALID_LINES','race_type':race_type,'race_type_group':'SPECIAL','special_subtype':'SELECTION'}
    rider_line={v:i for i,line in enumerate(lines) for v in line}
    if any(v not in rider_line for v in target):
        return {'scheme_version':SCHEME_VERSION,'buy':False,'reason':'SELECTION_TARGET_LINE_MAP_INCOMPLETE','race_type':race_type,'race_type_group':'SPECIAL','special_subtype':'SELECTION'}
    if len({rider_line[v] for v in target}) != 2:
        return {'scheme_version':SCHEME_VERSION,'buy':False,'reason':'SELECTION_TARGET_NOT_TWO_LINES','race_type':race_type,'race_type_group':'SPECIAL','special_subtype':'SELECTION','target_set':target}

    head=max(target,key=lambda x:(h.get(x,0.0),-x))
    tails=tuple(sorted(v for v in target if v!=head))
    a=(head,tails[0],tails[1]); b=(head,tails[1],tails[0])
    oa=float(trifecta_odds[a]); ob=float(trifecta_odds[b])
    ticket=a if oa>=ob else b
    formation=f'{head}-{ticket[1]}-{ticket[2]}'
    return {
        'scheme_version':SCHEME_VERSION,'base_scheme':BASE_SCHEME,'buy':True,
        'race_type':race_type,'race_type_group':'SPECIAL','special_subtype':'SELECTION',
        'formation':formation,'first':(ticket[0],),'second':(ticket[1],),'third':(ticket[2],),
        'tickets':(ticket,),'ticket_count':1,
        'H_BALANCED':True,'global_H1':h1,'target_set':target,'target_D_log':rows[target]['D_log'],
        'target_trio_rank':p_rank.index(target)+1,'target_line_span':2,
        'tail_order_policy':'BUY_HIGHER_ODDS_OF_TWO_TAIL_ORDERS',
        'individual_ticket_pruning':False,
        'branch_policy':'SELECTION ultra-ROI: H balanced; most negative-D set among trio top3; target includes global H1 and spans 2 lines; head=target H max; buy longer tail order only',
        'development_status':STATUS,
    }


def build_v8_23_f24(trio_odds,trifecta_odds,predicted_line_formation: str,race_type: str):
    if (race_type or '').strip()=='Ｓ級選抜':
        return _build_selection(trio_odds,trifecta_odds,predicted_line_formation,race_type)
    out=build_v8_22_f23(trio_odds,trifecta_odds,predicted_line_formation,race_type)
    out=dict(out);out['scheme_version']=SCHEME_VERSION
    return out
