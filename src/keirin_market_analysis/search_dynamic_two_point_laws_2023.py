from __future__ import annotations

import csv, itertools, json, math, re, statistics
from collections import defaultdict
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'data' / '2023' / 's_class_yosen'
OUT = ROOT / 'data' / 'audits' / 'dynamic_two_point_laws_2023.json'
BETA = 0.02219612332210088
MUST_TV = 0.004982810992042711
CORE_RULES = ('PAIR_DELTA', 'PAIR_RATIO_MASS', 'PAIR_KL')
WING_RULES = ('MARKET', 'PREMIUM', 'BARBELL', 'FLOW')


def read_csv(p):
    with p.open('r', encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def combo(s):
    s = str(s).strip().replace('=', '-').replace(',', '-')
    try: return tuple(sorted(int(x) for x in s.split('-') if x.strip()))
    except Exception: return ()


def parse_date(row):
    for k in ('race_date','date','held_date','event_date'):
        v = str(row.get(k,'') or '')
        m = re.search(r'(20\d{2})[-/]?(\d{2})[-/]?(\d{2})', v)
        if m:
            try: return date(int(m.group(1)),int(m.group(2)),int(m.group(3)))
            except ValueError: pass
    rid = str(row.get('race_id','') or '')
    m = re.search(r'(20\d{2})[-_/]?(\d{2})[-_/]?(\d{2})', rid)
    if m:
        try: return date(int(m.group(1)),int(m.group(2)),int(m.group(3)))
        except ValueError: pass
    return None


def q_from_weights(combos, weights):
    raw={c:math.prod(weights[x] for x in c) for c in combos}; z=sum(raw.values())
    return {c:v/z for c,v in raw.items()}


def fit_maxent(riders, combos, target, tol=1e-12, max_iter=20000):
    w={r:1.0 for r in riders}
    for _ in range(max_iter):
        for r in riders:
            q=q_from_weights(combos,w); cur=sum(v for c,v in q.items() if r in c); t=target[r]
            w[r] *= (t*(1-cur))/(cur*(1-t))
        g=math.exp(sum(math.log(max(w[r],1e-300)) for r in riders)/len(riders))
        for r in riders: w[r]/=g
        q=q_from_weights(combos,w)
        err=max(abs(sum(v for c,v in q.items() if r in c)-target[r]) for r in riders)
        if err < tol: return q,err
    return q,err


def p1_probs(m,R):
    vals={c:m[c]*(R[c]**BETA) for c in m}; z=sum(vals.values())
    return {c:v/z for c,v in vals.items()}


def tv(a,b): return .5*sum(abs(a[c]-b[c]) for c in a)


def payout_value(row):
    for k in ('payout_yen','payout','amount_yen','amount'):
        x=row.get(k)
        if x not in (None,''):
            try:return int(float(str(x).replace(',','')))
            except Exception:pass
    return None


def load_races():
    by=defaultdict(list); dates={}
    for r in read_csv(DATA/'trio_final_odds.csv'):
        if r.get('odds_status')!='available': continue
        try:o=float(r['odds'])
        except Exception:continue
        c=combo(r.get('combination',''))
        if len(c)==3 and o>0:
            rid=str(r['race_id']); by[rid].append((c,o)); d=parse_date(r)
            if d: dates[rid]=d
    paid=defaultdict(list)
    for r in read_csv(DATA/'payouts.csv'):
        if r.get('ticket_type')=='3連複' and r.get('status')=='paid':
            c=combo(r.get('combination','')); pv=payout_value(r); rid=str(r.get('race_id',''))
            if len(c)==3 and pv is not None: paid[rid].append((c,pv))
            if rid not in dates:
                d=parse_date(r)
                if d: dates[rid]=d
    acc={'market_races':len(by),'excluded_non7_or_incomplete':0,'no_unique_paid_trio':0,'solver_failures':0,'non_must_enter':0,'date_parse_failures':0}
    races=[]
    for rid,rows in sorted(by.items()):
        riders=sorted({x for c,_ in rows for x in c})
        if len(riders)!=7: acc['excluded_non7_or_incomplete']+=1; continue
        combos=list(itertools.combinations(riders,3)); odds={c:o for c,o in rows}
        if set(odds)!=set(combos): acc['excluded_non7_or_incomplete']+=1; continue
        ps=paid.get(rid,[])
        if len(ps)!=1: acc['no_unique_paid_trio']+=1; continue
        d=dates.get(rid)
        if d is None: acc['date_parse_failures']+=1; continue
        inv={c:1/o for c,o in odds.items()}; z=sum(inv.values()); m={c:v/z for c,v in inv.items()}
        support={r:sum(v for c,v in m.items() if r in c) for r in riders}
        q,err=fit_maxent(riders,combos,support)
        if err>=1e-9: acc['solver_failures']+=1; continue
        R={c:m[c]/q[c] for c in combos}; p1=p1_probs(m,R)
        if tv(p1,m)<MUST_TV: acc['non_must_enter']+=1; continue
        win,payout=ps[0]
        races.append({'race_id':rid,'date':d,'riders':riders,'combos':combos,'m':m,'q':q,'win':win,'payout':payout})
    acc['must_enter_evaluated']=len(races)
    return races,acc


def choose_pair(r, core_rule):
    rows=[]
    for pair in itertools.combinations(r['riders'],2):
        cs=[c for c in r['combos'] if pair[0] in c and pair[1] in c]
        pm=sum(r['m'][c] for c in cs); pq=sum(r['q'][c] for c in cs)
        delta=pm-pq; ratio=pm/pq
        kl=sum(r['m'][c]*math.log(r['m'][c]/r['q'][c]) for c in cs)
        if core_rule=='PAIR_DELTA': score=delta
        elif core_rule=='PAIR_RATIO_MASS': score=pm*math.log(ratio)
        elif core_rule=='PAIR_KL': score=kl
        else: raise ValueError(core_rule)
        rows.append((score,pm,delta,tuple(-x for x in pair),pair))
    return max(rows)[-1]


def choose_wings(r,pair,wing_rule):
    cs=[c for c in r['combos'] if pair[0] in c and pair[1] in c]
    delta={c:r['m'][c]-r['q'][c] for c in cs}
    if wing_rule=='MARKET':
        return tuple(sorted(sorted(cs,key=lambda c:(-r['m'][c],c))[:2]))
    if wing_rule=='PREMIUM':
        return tuple(sorted(sorted(cs,key=lambda c:(-delta[c],-r['m'][c],c))[:2]))
    if wing_rule=='FLOW':
        return tuple(sorted(sorted(cs,key=lambda c:(-abs(delta[c]),-r['m'][c],c))[:2]))
    if wing_rule=='BARBELL':
        hi=max(cs,key=lambda c:(delta[c],r['m'][c],tuple(-x for x in c)))
        lo=min((c for c in cs if c!=hi),key=lambda c:(delta[c],-r['m'][c],c))
        return tuple(sorted((hi,lo)))
    raise ValueError(wing_rule)


def score(rows,core,wing):
    stake=200*len(rows); pay=hits=0
    for r in rows:
        pair=choose_pair(r,core); tickets=choose_wings(r,pair,wing)
        if r['win'] in tickets: hits+=1; pay+=r['payout']
    return {'races':len(rows),'stake_yen':stake,'payout_yen':pay,'profit_yen':pay-stake,'roi_pct':100*pay/stake if stake else None,'hit_races':hits,'hit_rate_pct':100*hits/len(rows) if rows else None}


def qbucket(d): return f'Q{(d.month-1)//3+1}'


def main():
    races,acc=load_races(); quarters={f'Q{i}':[r for r in races if qbucket(r['date'])==f'Q{i}'] for i in range(1,5)}
    halves={'H1':[r for r in races if r['date'].month<=6],'H2':[r for r in races if r['date'].month>=7]}
    laws=[]
    for core in CORE_RULES:
        for wing in WING_RULES:
            full=score(races,core,wing); qs={k:score(v,core,wing) for k,v in quarters.items()}; hs={k:score(v,core,wing) for k,v in halves.items()}
            qrois=[qs[f'Q{i}']['roi_pct'] for i in range(1,5)]; hrois=[hs['H1']['roi_pct'],hs['H2']['roi_pct']]
            laws.append({'law':f'{core}__{wing}','core_rule':core,'wing_rule':wing,'full_year':full,'quarters':qs,'halves':hs,'robustness':{'worst_quarter_roi_pct':min(qrois),'median_quarter_roi_pct':statistics.median(qrois),'profitable_quarters':sum(x>=100 for x in qrois),'quarters_ge_90':sum(x>=90 for x in qrois),'worst_half_roi_pct':min(hrois),'profitable_halves':sum(x>=100 for x in hrois)}})
    ranked=sorted(laws,key=lambda x:(-x['robustness']['profitable_quarters'],-x['robustness']['quarters_ge_90'],-x['robustness']['worst_quarter_roi_pct'],-x['full_year']['roi_pct']))
    out={'status':'DYNAMIC_TWO_POINT_LAWS_2023','scope':'2023 development only; 12 predefined market-structure laws; no 2024 used','must_enter_rule':'Frozen TV(P1,M) threshold','two_point_rule':'Exactly two trio tickets sharing dynamically selected pair core','accounting':acc,'quarter_counts':{k:len(v) for k,v in quarters.items()},'law_count':len(laws),'ranked_by_temporal_robustness':ranked,'selected_for_forward_test':ranked[0],'warning':'Selection is development on 2023. Forward test must freeze the selected law exactly.'}
    OUT.parent.mkdir(parents=True,exist_ok=True); OUT.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'accounting':acc,'top':ranked[:5]},ensure_ascii=False,indent=2))

if __name__=='__main__': main()
