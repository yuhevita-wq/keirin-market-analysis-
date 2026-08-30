from __future__ import annotations

import importlib.util, itertools, json, math
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
SRC=ROOT/'src'/'keirin_market_analysis'/'search_dynamic_two_point_laws_2023.py'
OUT=ROOT/'data'/'audits'/'pair_flow_two_point_2023.json'
spec=importlib.util.spec_from_file_location('dyn',SRC); dyn=importlib.util.module_from_spec(spec); assert spec and spec.loader; spec.loader.exec_module(dyn)


def pair_stats(r):
    rows=[]
    for pair in itertools.combinations(r['riders'],2):
        cs=[c for c in r['combos'] if pair[0] in c and pair[1] in c]
        pm=sum(r['m'][c] for c in cs); pq=sum(r['q'][c] for c in cs); d=pm-pq
        rows.append((pair,d))
    pos=sum(max(d,0.0) for _,d in rows)
    top_pair,top_delta=max(rows,key=lambda x:(x[1],tuple(-z for z in x[0])))
    concentration=top_delta/pos if pos>0 else 0.0
    return top_pair,concentration


def tickets(r,pair):
    cs=[c for c in r['combos'] if pair[0] in c and pair[1] in c]
    d={c:r['m'][c]-r['q'][c] for c in cs}
    return tuple(sorted(sorted(cs,key=lambda c:(-abs(d[c]),-r['m'][c],c))[:2]))


def score(rows):
    stake=pay=hits=0; cur=maxloss=0
    for r in rows:
        pair,_=pair_stats(r); ts=tickets(r,pair); stake+=200
        if r['win'] in ts: hits+=1; pay+=r['payout']; cur=0
        else: cur+=1; maxloss=max(maxloss,cur)
    return {'races':len(rows),'tickets':2*len(rows),'stake_yen':stake,'payout_yen':pay,'profit_yen':pay-stake,'roi_pct':100*pay/stake if stake else None,'hit_races':hits,'hit_rate_pct':100*hits/len(rows) if rows else None,'max_losing_streak_races':maxloss}


def nearest_rank_q75(xs):
    ys=sorted(xs); idx=max(0,math.ceil(.75*len(ys))-1); return ys[idx]


def main():
    # Load all complete 7-rider races, not the old P1-TV MUST_ENTER set.
    old=dyn.MUST_TV; dyn.MUST_TV=-1.0
    races,acc=dyn.load_races(); dyn.MUST_TV=old
    cs=[pair_stats(r)[1] for r in races]
    threshold=nearest_rank_q75(cs)
    selected=[r for r in races if pair_stats(r)[1]>=threshold]
    quarters={f'Q{i}':[r for r in selected if (r['date'].month-1)//3+1==i] for i in range(1,5)}
    out={
      'status':'PAIR_FLOW_TWO_POINT_2023_DEVELOPMENT',
      'scope':'2023 development; model-specific entry threshold defined from market structure only',
      'law':{
        'core':'pair with maximum pair residual delta = sum(M-Q) over its five trios',
        'entry_metric':'top positive pair delta / sum of all positive pair deltas',
        'entry_threshold_rule':'nearest-rank 75th percentile of 2023 complete-race market-only entry metric',
        'entry_threshold':threshold,
        'wings':'among five trios containing selected pair, choose two largest absolute trio residual |M-Q|',
        'stake':'100 yen each, exactly two tickets',
        'outcome_used_for_entry_or_ticket_selection':False,
      },
      'accounting':acc,
      'eligible_complete_races':len(races),
      'must_enter_races':len(selected),
      'must_enter_rate_pct':100*len(selected)/len(races),
      'full_year':score(selected),
      'quarters':{k:score(v) for k,v in quarters.items()},
      'warning':'2023 outcome is development scoring, not validation. Freeze threshold and law for 2024.'
    }
    OUT.parent.mkdir(parents=True,exist_ok=True); OUT.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8'); print(json.dumps(out,ensure_ascii=False,indent=2))

if __name__=='__main__': main()
