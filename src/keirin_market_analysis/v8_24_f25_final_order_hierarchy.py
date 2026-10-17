from __future__ import annotations

"""v8.24-F25: FINAL order-hierarchy rebuild.

All non-FINAL branches inherit v8.23-F24 unchanged.

FINAL is rebuilt from 210-way trifecta order information instead of broad
market heat. Two mutually exclusive semantic branches are allowed:

1) HEAD LOCK x TAIL SPLIT
   - H1 >= 2*H2 (the pre-existing semantic head-concentration boundary).
   - first place is H1 only.
   - conditional second-place support given H1 is ranked and cut at its
     largest adjacent support cliff.
   - conditional third-place support, aggregated over the selected second
     pool, is ranked and cut at its largest adjacent support cliff.
   - the result must be a valid clean rectangular formation.
   - at least one tail pool must contain more than one rider; a completely
     locked 1-1-1 market is not a TAIL SPLIT branch.

2) SET LOCK x ORDER SPLIT
   - H1 < 2*H2 (head market balanced).
   - the top trio set by P_trio must exactly equal the top unordered set from
     collapsed 210-way trifecta support Q_tf_set.
   - that locked set must contain global H1 and H2.
   - under each of H1 and H2 as head, choose the less-supported / higher-odds
     of the two tail orders.
   - those two chosen orders are bought only if they themselves form one clean
     rectangular 2-ticket formation. Otherwise skip rather than prune tickets.

No fitted numeric cutoff, result input, payout input, or individual-ticket
pruning is used.
"""

from collections import defaultdict
from itertools import product

from v7_0_f01_market_hierarchy import implied_probabilities, trio_implied_probabilities
from v8_23_f24_selection_ultra_roi import build_v8_23_f24

SCHEME_VERSION = 'v8.24-F25'
BASE_SCHEME = 'v8.23-F24'
STATUS = 'DEVELOPMENT_Q1Q2_FINAL_ORDER_HIERARCHY'


def _tickets(first, second, third):
    return tuple(sorted({
        (a, b, c)
        for a, b, c in product(first, second, third)
        if len({a, b, c}) == 3
    }))


def _display(first, second, third):
    show=lambda xs: ''.join(str(x) for x in sorted(xs))
    return f'{show(first)}-{show(second)}-{show(third)}'


def _valid_rectangle(first, second, third, intended=None):
    f=tuple(sorted(set(first)));s=tuple(sorted(set(second)));t=tuple(sorted(set(third)))
    ts=_tickets(f,s,t)
    if not ts:return None
    if {x[0] for x in ts}!=set(f) or {x[1] for x in ts}!=set(s) or {x[2] for x in ts}!=set(t):return None
    if intended is not None and set(ts)!=set(intended):return None
    return f,s,t,ts


def _head_support(q):
    h=defaultdict(float)
    for (a,b,c),p in q.items():h[a]+=p
    return dict(h)


def _cliff_prefix(scores):
    """Parameter-free prefix ending immediately before largest adjacent support drop."""
    ranked=tuple(sorted(scores,key=lambda x:(-scores[x],x)))
    if not ranked:return tuple(),None,tuple()
    if len(ranked)==1:return ranked,None,tuple()
    drops=[]
    for i in range(len(ranked)-1):
        a=scores[ranked[i]];b=scores[ranked[i+1]]
        drops.append((a/b) if b>0 else float('inf'))
    m=max(drops);idx=next(i for i,d in enumerate(drops) if d==m)
    return ranked[:idx+1],m,tuple(drops)


def _collapsed_set_support(q):
    out=defaultdict(float)
    for t,p in q.items():out[tuple(sorted(t))]+=p
    return dict(out)


def _line_span(target,predicted_line_formation):
    raw=(predicted_line_formation or '').strip()
    if not raw:return None
    lines=[]
    for block in raw.split('/'):
        vals=tuple(int(ch) for ch in block if ch.isdigit())
        if vals:lines.append(vals)
    rider_line={v:i for i,line in enumerate(lines) for v in line}
    if any(v not in rider_line for v in target):return None
    return len({rider_line[v] for v in target})


def _head_lock_tail_split(trio_odds,trifecta_odds,predicted_line_formation,race_type,q,h,hr):
    h1,h2=hr[:2]
    if h[h1] < 2*h[h2]:return None

    c2=defaultdict(float)
    for (a,b,c),p in q.items():
        if a==h1:c2[b]+=p
    c2.pop(h1,None)
    second,second_cliff,second_drops=_cliff_prefix(c2)
    if not second:
        return {'scheme_version':SCHEME_VERSION,'buy':False,'reason':'FINAL_HEAD_LOCK_NO_SECOND_POOL','race_type':race_type,'race_type_group':'FINAL'}

    c3=defaultdict(float)
    for (a,b,c),p in q.items():
        if a==h1 and b in second and c!=h1:c3[c]+=p
    c3.pop(h1,None)
    third,third_cliff,third_drops=_cliff_prefix(c3)
    if not third:
        return {'scheme_version':SCHEME_VERSION,'buy':False,'reason':'FINAL_HEAD_LOCK_NO_THIRD_POOL','race_type':race_type,'race_type_group':'FINAL'}
    if len(second)==1 and len(third)==1:
        return {'scheme_version':SCHEME_VERSION,'buy':False,'reason':'FINAL_HEAD_LOCK_TAIL_NOT_SPLIT','race_type':race_type,'race_type_group':'FINAL','global_H1':h1}

    rect=_valid_rectangle((h1,),second,third)
    if rect is None:
        return {'scheme_version':SCHEME_VERSION,'buy':False,'reason':'FINAL_HEAD_LOCK_INVALID_RECTANGLE','race_type':race_type,'race_type_group':'FINAL','global_H1':h1}
    first,second,third,tickets=rect
    return {
        'scheme_version':SCHEME_VERSION,'base_scheme':BASE_SCHEME,'buy':True,'reason':'PASS',
        'race_type':race_type,'race_type_group':'FINAL','final_branch':'HEAD_LOCK_TAIL_SPLIT',
        'formation':_display(first,second,third),'first':first,'second':second,'third':third,
        'tickets':tickets,'ticket_count':len(tickets),'global_H1':h1,'global_H2':h2,
        'H1_top':h[h1],'H1_second':h[h2],'H_CONCENTRATED':True,
        'conditional_second_pool':second,'conditional_second_cliff':second_cliff,'conditional_second_drops':second_drops,
        'conditional_third_pool':third,'conditional_third_cliff':third_cliff,'conditional_third_drops':third_drops,
        'formation_line_span':_line_span(set(first)|set(second)|set(third),predicted_line_formation),
        'individual_ticket_pruning':False,'fixed_place_counts':False,
        'branch_policy':'FINAL HEAD_LOCK x TAIL_SPLIT: H1 anchor; conditional C2 and C3 largest-cliff rectangular formation',
        'development_status':STATUS,
    }


def _set_lock_order_split(trio_odds,trifecta_odds,predicted_line_formation,race_type,q,h,hr):
    h1,h2=hr[:2]
    if h[h1] >= 2*h[h2]:return None
    p=trio_implied_probabilities(trio_odds);qs=_collapsed_set_support(q)
    top_p=max(p,key=lambda c:(p[c],tuple(-x for x in c)))
    top_q=max(qs,key=lambda c:(qs[c],tuple(-x for x in c)))
    if top_p!=top_q:
        return {'scheme_version':SCHEME_VERSION,'buy':False,'reason':'FINAL_SET_LOCK_MARKETS_DISAGREE','race_type':race_type,'race_type_group':'FINAL','top_trio_set':top_p,'top_tf_set':top_q}
    target=tuple(sorted(top_p))
    if h1 not in target or h2 not in target:
        return {'scheme_version':SCHEME_VERSION,'buy':False,'reason':'FINAL_SET_LOCK_MISSING_H_TOP2','race_type':race_type,'race_type_group':'FINAL','target_set':target}

    chosen=[]
    for head in (h1,h2):
        tails=tuple(v for v in target if v!=head)
        if len(tails)!=2:
            return {'scheme_version':SCHEME_VERSION,'buy':False,'reason':'FINAL_SET_LOCK_INVALID_TARGET','race_type':race_type,'race_type_group':'FINAL'}
        a=(head,tails[0],tails[1]);b=(head,tails[1],tails[0])
        oa=float(trifecta_odds[a]);ob=float(trifecta_odds[b])
        chosen.append(a if oa>=ob else b)
    intended=tuple(sorted(set(chosen)))
    second=tuple(t[1] for t in intended);third=tuple(t[2] for t in intended)
    rect=_valid_rectangle((h1,h2),second,third,intended=intended)
    if rect is None:
        return {'scheme_version':SCHEME_VERSION,'buy':False,'reason':'FINAL_SET_LOCK_NON_RECTANGULAR_ORDER_SPLIT','race_type':race_type,'race_type_group':'FINAL','target_set':target,'candidate_orders':intended}
    first,second,third,tickets=rect
    return {
        'scheme_version':SCHEME_VERSION,'base_scheme':BASE_SCHEME,'buy':True,'reason':'PASS',
        'race_type':race_type,'race_type_group':'FINAL','final_branch':'SET_LOCK_ORDER_SPLIT',
        'formation':_display(first,second,third),'first':first,'second':second,'third':third,
        'tickets':tickets,'ticket_count':len(tickets),'global_H1':h1,'global_H2':h2,
        'H_BALANCED':True,'target_set':target,'target_trio_support':p[target],'target_tf_set_support':qs[target],
        'target_line_span':_line_span(target,predicted_line_formation),
        'tail_order_policy':'FOR_EACH_H1_H2_HEAD_BUY_HIGHER_ODDS_TAIL_ORDER_IF_RECTANGULAR',
        'individual_ticket_pruning':False,'fixed_place_counts':False,
        'branch_policy':'FINAL SET_LOCK x ORDER_SPLIT: top trio set equals top collapsed-TF set; H1/H2 heads; longer tail order; clean rectangle only',
        'development_status':STATUS,
    }


def _build_final(trio_odds,trifecta_odds,predicted_line_formation,race_type):
    q=implied_probabilities(trifecta_odds);h=_head_support(q)
    hr=tuple(sorted(h,key=lambda x:(-h[x],x)))
    if len(hr)<2:
        return {'scheme_version':SCHEME_VERSION,'buy':False,'reason':'FINAL_MISSING_H_TOP2','race_type':race_type,'race_type_group':'FINAL'}
    if h[hr[0]] >= 2*h[hr[1]]:
        return _head_lock_tail_split(trio_odds,trifecta_odds,predicted_line_formation,race_type,q,h,hr)
    return _set_lock_order_split(trio_odds,trifecta_odds,predicted_line_formation,race_type,q,h,hr)


def build_v8_24_f25(trio_odds,trifecta_odds,predicted_line_formation: str,race_type: str):
    if (race_type or '').strip()=='Ｓ級決勝':
        return _build_final(trio_odds,trifecta_odds,predicted_line_formation,race_type)
    out=build_v8_23_f24(trio_odds,trifecta_odds,predicted_line_formation,race_type)
    out=dict(out);out['scheme_version']=SCHEME_VERSION
    return out
