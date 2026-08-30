from __future__ import annotations

import csv
import itertools
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "2024" / "s_class_yosen"
AUDITS = ROOT / "data" / "audits"
GATE_JSON = AUDITS / "fake_favorite_true_middle_2023_2024.json"
FROZEN_JSON = AUDITS / "frozen_axis_drop_rule_v1.json"
OUT_JSON = AUDITS / "axis_drop_rule_v1_validation_2024.json"
OUT_CSV = AUDITS / "axis_drop_rule_v1_validation_2024_decisions.csv"
STAKE = 100


def norm(*xs): return tuple(sorted(int(x) for x in xs))
def read_csv(p):
    with p.open('r', encoding='utf-8-sig', newline='') as f: return list(csv.DictReader(f))
def pick(fields, names):
    m={x.lower():x for x in fields}
    for n in names:
        if n.lower() in m: return m[n.lower()]
    return None

def parse_combo(v):
    s=v.strip().replace('=', '-').replace(',', '-')
    return tuple(sorted(int(x) for x in s.split('-') if x.strip()))

def load_odds(path, ordered=False):
    rows=read_csv(path); f=rows[0].keys(); rid=pick(f,['race_id']); od=pick(f,['odds','final_odds']); combo=pick(f,['combo','combination','numbers','selection']); out=defaultdict(dict)
    for r in rows:
        if combo:
            s=r[combo].strip().replace('=', '-').replace(',', '-')
            vals=tuple(int(x) for x in s.split('-') if x.strip())
            key=vals if ordered else tuple(sorted(vals))
        else:
            cols=[pick(f,[x]) for x in ['first','second','third']]
            if not all(cols): cols=[pick(f,[x]) for x in ['n1','n2','n3']]
            vals=tuple(int(r[x]) for x in cols)
            key=vals if ordered else tuple(sorted(vals))
        out[r[rid]][key]=float(r[od])
    return out

def load_results():
    rows=read_csv(DATA/'results.csv'); f=rows[0].keys(); rid=pick(f,['race_id']); rank=pick(f,['finish_position','finish','rank','placing']); car=pick(f,['car_no','number','bike_no']); g=defaultdict(list)
    for r in rows:
        try: g[r[rid]].append((int(float(r[rank])), int(float(r[car]))))
        except: pass
    return {k: tuple(x[1] for x in sorted(v)[:3]) for k,v in g.items() if len(v)>=3}

def implied(m):
    inv={k:1/v for k,v in m.items() if v and v>0}; z=sum(inv.values()); return {k:v/z for k,v in inv.items()} if z else {}
def delta_map(trio, tf):
    pt=implied(trio); po=implied(tf); ps=defaultdict(float)
    for order,p in po.items(): ps[norm(*order)] += p
    return {c: ps[c]-pt[c] for c in set(pt)&set(ps)}
def entrants(trio): return tuple(sorted({x for c in trio for x in c}))

def load_gate():
    p=json.loads(GATE_JSON.read_text(encoding='utf-8')); rows=p['years']['2024']['races']; return {str(r['race_id']):r for r in rows}

def choose_axis(favorite, d, ents):
    fav=tuple(favorite); outs=[x for x in ents if x not in fav]; scored=[]
    for pair in itertools.combinations(fav,2):
        vals=[d.get(norm(pair[0],pair[1],x), float('-inf')) for x in outs]
        pos=[v for v in vals if v>0]
        scored.append((len(pos), sum(pos), sum(v for v in vals if v!=float('-inf')), tuple(-x for x in pair), tuple(sorted(pair))))
    return max(scored)[-1]

def payout(result, ticket, trio_odds):
    return int(round(STAKE*trio_odds[ticket])) if result==ticket and ticket in trio_odds else 0

def agg(): return {'races':0,'tickets':0,'hit_races':0,'stake_yen':0,'payout_yen':0}

def main():
    frozen=json.loads(FROZEN_JSON.read_text(encoding='utf-8'))
    if frozen['rule']['name']!='positive_count_then_sum': raise RuntimeError('Frozen rule mismatch')
    gate=load_gate(); trio=load_odds(DATA/'trio_final_odds.csv'); tf=load_odds(DATA/'trifecta_final_odds.csv', ordered=True); results=load_results()
    methods={'two_head_flow':agg(),'four_car_box':agg()}; rows=[]
    for rid in sorted(gate):
        if rid not in trio or rid not in tf or rid not in results: raise RuntimeError(f'missing {rid}')
        fav=norm(*gate[rid]['favorite']); d=delta_map(trio[rid], tf[rid]); ents=entrants(trio[rid]); axis=choose_axis(fav,d,ents); drop=next(x for x in fav if x not in axis); outs=[x for x in ents if x not in fav]
        t1=sorted({norm(axis[0],axis[1],x) for x in outs})
        ranked=sorted(outs, key=lambda x:(d.get(norm(axis[0],axis[1],x), float('-inf')),-x), reverse=True)
        four=tuple(sorted((*axis,*ranked[:2]))) if len(ranked)>=2 else tuple()
        t2=sorted({norm(*c) for c in itertools.combinations(four,3)}) if len(four)==4 else []
        result=norm(*results[rid])
        row={'race_id':rid,'favorite':'-'.join(map(str,fav)),'axis':'-'.join(map(str,axis)),'drop':drop,'result':'-'.join(map(str,result))}
        for name,tickets in [('two_head_flow',t1),('four_car_box',t2)]:
            s=methods[name]; s['races']+=1; s['tickets']+=len(tickets); s['stake_yen']+=STAKE*len(tickets); pay=sum(payout(result,t,trio[rid]) for t in tickets); hit=int(result in tickets); s['payout_yen']+=pay; s['hit_races']+=hit
            row[name+'_tickets']='|'.join('-'.join(map(str,t)) for t in tickets); row[name+'_hit']=hit; row[name+'_payout_yen']=pay
        rows.append(row)
    for s in methods.values():
        s['profit_yen']=s['payout_yen']-s['stake_yen']; s['roi_pct']=100*s['payout_yen']/s['stake_yen']; s['hit_rate_pct']=100*s['hit_races']/s['races']
    out={'status':'AXIS_DROP_RULE_V1_VALIDATION_2024','development_year':2023,'validation_year':2024,'rule_source':'data/audits/frozen_axis_drop_rule_v1.json','rule_changed':False,'fake_favorite_races':len(gate),'methods':methods}
    OUT_JSON.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
    with OUT_CSV.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(json.dumps(out,ensure_ascii=False,indent=2))
if __name__=='__main__': main()
