from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

DATA = Path('data/2025/s_class_yosen')
SIM = DATA / 'simulations' / 'mainline_v1'
OUT = SIM / 'third2_exploration.json'


def read_csv(path: Path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def score(e):
    try:
        return float(e.get('score',''))
    except Exception:
        return -10**9


def car(e):
    return int(e['car_no'])


def choose_two(rule, es, a, b, main_line_id):
    remaining=[e for e in es if car(e) not in {a,b}]
    by_score=sorted(remaining, key=lambda e:(-score(e), car(e)))
    leaders=sorted([e for e in remaining if e.get('line_position')=='1' and e.get('line_id')!=str(main_line_id)], key=lambda e:(-score(e),car(e)))
    main3=next((e for e in remaining if e.get('line_id')==str(main_line_id) and e.get('line_position')=='3'),None)

    broad=[]
    if main3: broad.append(main3)
    broad += leaders
    broad += by_score[:4]
    uniq=[]; seen=set()
    for e in broad:
        if car(e) not in seen:
            seen.add(car(e)); uniq.append(e)

    if rule=='top2_score_remaining':
        picks=by_score[:2]
    elif rule=='top2_score_v1_pool':
        picks=sorted(uniq,key=lambda e:(-score(e),car(e)))[:2]
    elif rule=='main3_plus_best_other_leader':
        picks=[]
        if main3: picks.append(main3)
        for e in leaders:
            if car(e) not in {car(x) for x in picks}:
                picks.append(e)
            if len(picks)==2: break
        for e in by_score:
            if car(e) not in {car(x) for x in picks}:
                picks.append(e)
            if len(picks)==2: break
    elif rule=='top2_other_leaders':
        picks=leaders[:2]
        for e in by_score:
            if car(e) not in {car(x) for x in picks}:
                picks.append(e)
            if len(picks)==2: break
    elif rule=='main3_plus_best_score':
        picks=[]
        if main3: picks.append(main3)
        for e in by_score:
            if car(e) not in {car(x) for x in picks}:
                picks.append(e)
            if len(picks)==2: break
    else:
        raise ValueError(rule)
    return [car(e) for e in picks[:2]]


def main():
    race_rows=read_csv(SIM/'race_simulation.csv')
    entries=read_csv(DATA/'entries.csv')
    payouts=read_csv(DATA/'payouts.csv')
    entries_by=defaultdict(list)
    for e in entries: entries_by[e['race_id']].append(e)
    trifecta=defaultdict(dict)
    for p in payouts:
        if p.get('ticket_type')=='3連単' and p.get('status')=='paid' and p.get('combination'):
            trifecta[p['race_id']][p['combination']]=int(p['payout_yen'])

    later=[r for r in race_rows if r['segment']=='後半']
    strict=[r for r in later if r['mainline_top2']=='1']

    position=Counter(); overall_rank=Counter(); remaining_rank=Counter(); feature=Counter(); no_third=0
    detail=[]
    for r in strict:
        es=entries_by[r['race_id']]
        a=int(r['a_car']); b=int(r['b_car']); lid=str(r['main_line_id'])
        thirds=[int(x) for x in r['actual_third'].split('-') if x]
        if not thirds:
            no_third += 1; continue
        for t in thirds:
            e=next((x for x in es if car(x)==t),None)
            if not e: continue
            lp=e.get('line_position',''); eid=e.get('line_id',''); ls=e.get('line_size','')
            if eid==lid and lp=='3': pos='本線3番手'
            elif lp=='1': pos='他ライン先頭'
            elif lp=='2': pos='他ライン番手'
            else: pos='その他ライン位置'
            position[pos]+=1
            ranked=sorted(es,key=lambda x:(-score(x),car(x)))
            orank=next(i+1 for i,x in enumerate(ranked) if car(x)==t)
            rem=[x for x in ranked if car(x) not in {a,b}]
            rrank=next(i+1 for i,x in enumerate(rem) if car(x)==t)
            overall_rank[orank]+=1; remaining_rank[rrank]+=1
            if orank<=4: feature['得点全体4位以内']+=1
            if lp=='1': feature['ライン先頭']+=1
            if eid==lid and lp=='3': feature['本線3番手']+=1
            if (lp=='1') or (eid==lid and lp=='3'): feature['本線3番手or他ライン先頭']+=1
            detail.append({'race_id':r['race_id'],'third':t,'position':pos,'overall_score_rank':orank,'remaining_score_rank':rrank})

    rules=['top2_score_remaining','top2_score_v1_pool','main3_plus_best_other_leader','top2_other_leaders','main3_plus_best_score']
    rule_stats={}
    for rule in rules:
        agg={k:{'races':0,'capture':0,'stake':0,'payout':0,'hits':0} for k in ['all','H1','H2']}
        for r in later:
            es=entries_by[r['race_id']]; a=int(r['a_car']); b=int(r['b_car']); lid=int(r['main_line_id'])
            picks=choose_two(rule,es,a,b,lid)
            half='H1' if r['race_date']<='2025-06-30' else 'H2'
            for bucket in ['all',half]:
                agg[bucket]['races']+=1; agg[bucket]['stake']+=400
            if r['mainline_top2']=='1':
                actual={int(x) for x in r['actual_third'].split('-') if x}
                captured=bool(actual & set(picks))
                if captured:
                    for bucket in ['all',half]: agg[bucket]['capture']+=1
            returned=0
            for x in picks:
                returned += trifecta[r['race_id']].get(f'{a}-{b}-{x}',0)
                returned += trifecta[r['race_id']].get(f'{b}-{a}-{x}',0)
            if returned:
                for bucket in ['all',half]: agg[bucket]['hits']+=1
            for bucket in ['all',half]: agg[bucket]['payout']+=returned
        # denominator for capture is strict top2 races in bucket
        denom_all=sum(1 for r in later if r['mainline_top2']=='1')
        denom_h1=sum(1 for r in later if r['mainline_top2']=='1' and r['race_date']<='2025-06-30')
        denom_h2=sum(1 for r in later if r['mainline_top2']=='1' and r['race_date']>'2025-06-30')
        denoms={'all':denom_all,'H1':denom_h1,'H2':denom_h2}
        for k,v in agg.items():
            v['capture_denominator']=denoms[k]
            v['capture_rate_given_top2']=v['capture']/denoms[k] if denoms[k] else 0
            v['hit_rate_all_later']=v['hits']/v['races'] if v['races'] else 0
            v['profit']=v['payout']-v['stake']
            v['roi']=v['payout']/v['stake'] if v['stake'] else 0
        rule_stats[rule]=agg

    summary={
        'scope':'2025 exact S級予選, 後半 only; descriptive third-place analysis uses races where mainline A/B occupied 1st-2nd under mainline_v1 definition',
        'later_races':len(later),
        'mainline_top2_races':len(strict),
        'mainline_top2_rate':len(strict)/len(later),
        'third_rows_analyzed':len(detail),
        'no_numeric_third_rows':no_third,
        'actual_third_position_counts':dict(position),
        'actual_third_overall_score_rank_counts':dict(sorted(overall_rank.items())),
        'actual_third_remaining_score_rank_counts':dict(sorted(remaining_rank.items())),
        'actual_third_feature_counts':dict(feature),
        'two_candidate_rule_exploration':rule_stats,
        'warning':'Exploratory on 2025 data. Do not treat the best 2025 rule as validated without holdout/out-of-sample confirmation.'
    }
    OUT.write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__': main()
