from __future__ import annotations

import csv, json
from collections import defaultdict
from pathlib import Path

from .simulate_mainline_v1 import choose_main_line, segment_for

DATA_DIR = Path('data/2025/s_class_yosen')
OUT_DIR = DATA_DIR / 'simulations' / 'middle_candidate_v0'


def read_csv(path: Path):
    with path.open('r', encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def fnum(v):
    try: return float(v)
    except (TypeError, ValueError): return float('-inf')


def strongest_rival(entries, main_id):
    by = defaultdict(list)
    for e in entries:
        if e.get('line_id','').isdigit() and e.get('line_position','').isdigit():
            by[int(e['line_id'])].append(e)
    candidates=[]
    for lid,m in by.items():
        m.sort(key=lambda e:int(e['line_position']))
        if lid == main_id or len(m) < 2: continue
        strength=fnum(m[0].get('score'))+fnum(m[1].get('score'))
        candidates.append((strength, -lid, lid, m))
    if not candidates: return None
    _,_,lid,m=max(candidates,key=lambda x:(x[0],x[1]))
    return lid,m


def summarize(rows):
    n=len(rows); stake=sum(r['stake_yen'] for r in rows); payout=sum(r['payout_yen'] for r in rows)
    hits=sum(r['hit'] for r in rows)
    return {'races':n,'hits':hits,'hit_rate':hits/n if n else 0.0,'stake_yen':stake,'payout_yen':payout,'profit_yen':payout-stake,'roi':payout/stake if stake else 0.0}


def main():
    races=read_csv(DATA_DIR/'races.csv'); entries=read_csv(DATA_DIR/'entries.csv'); payouts=read_csv(DATA_DIR/'payouts.csv')
    eb=defaultdict(list); groups=defaultdict(list); tri=defaultdict(dict)
    for e in entries: eb[e['race_id']].append(e)
    for r in races: groups[(r['race_date'],r['track'])].append(r)
    for p in payouts:
        if p.get('ticket_type')=='3連単' and p.get('status')=='paid' and p.get('combination'):
            try: tri[p['race_id']][p['combination']]=int(float(p['payout_yen']))
            except (TypeError,ValueError): pass
    seg={}
    for g in groups.values():
        g.sort(key=lambda r:int(r['race_no'])); n=len(g)
        for pos,r in enumerate(g,1): seg[r['race_id']]=segment_for(pos,n)

    rows=[]
    for race in sorted(races,key=lambda r:(r['race_date'],r['track'],int(r['race_no']))):
        rid=race['race_id']
        if seg.get(rid)!='中盤': continue
        chosen=choose_main_line(eb[rid])
        if not chosen: continue
        main_id,main=chosen
        if len(main)<3: continue
        rival=strongest_rival(eb[rid],main_id)
        if not rival: continue
        _,rm=rival
        a=int(main[0]['car_no']); b=int(main[1]['car_no']); r1l=int(rm[0]['car_no']); r1b=int(rm[1]['car_no'])
        b_score=fnum(main[1].get('score'))
        if not (b_score > 105.0): continue
        bets=[f'{a}-{b}-{r1b}', f'{r1l}-{r1b}-{b}']
        ret=sum(tri[rid].get(x,0) for x in bets)
        rows.append({'race_id':rid,'race_date':race['race_date'],'half':'H1' if race['race_date']<='2025-06-30' else 'H2','track':race['track'],'race_no':int(race['race_no']),'b_score':b_score,'bet_1':bets[0],'bet_2':bets[1],'stake_yen':200,'payout_yen':ret,'profit_yen':ret-200,'hit':1 if ret else 0})

    out={'scope':'2025 exact S級予選, 中盤, main line size >=3, strongest rival 2+ exists','status':'exploratory candidate only; not frozen V1','pre_race_rule':'main-line second rider score > 105.0','bets':['A-B-R1B','R1L-R1B-B'],'stake_per_race_yen':200,'all':summarize(rows),'H1':summarize([r for r in rows if r['half']=='H1']),'H2':summarize([r for r in rows if r['half']=='H2']),'warning':'Rule and formation were discovered within 2025. H1/H2 stability is reported, but another year is required for true out-of-sample validation.'}
    OUT_DIR.mkdir(parents=True,exist_ok=True)
    (OUT_DIR/'summary.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    with (OUT_DIR/'races.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(json.dumps(out,ensure_ascii=False,indent=2))

if __name__=='__main__': main()
