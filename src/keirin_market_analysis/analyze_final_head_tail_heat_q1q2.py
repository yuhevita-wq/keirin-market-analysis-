from __future__ import annotations
import csv,json,math
from collections import defaultdict
from pathlib import Path
from statistics import median

from simulate_v8_1_f02_2024q1 import load,pi,pl
from evaluate_v8_17_q2_oos import load_q2,DATA as Q2_DATA
from v7_0_f01_market_hierarchy import implied_probabilities,positional_support
from v8_8_f09_market_cliff import build_v8_8_f09
from v8_20_f21_initial_special_unified import build_v8_20_f21

ROOT=Path(__file__).resolve().parents[2]
Q1_DATA=ROOT/'data/2024/s_class_f1_all_parts/2024_q1'
STAKE=100

def entropy(vals):
    return -sum(x*math.log(x) for x in vals if x>0)

def metrics(trio_odds,tf_odds,line,cars):
    base=build_v8_8_f09(trio_odds,tf_odds,line)
    if not base.get('buy'):return None
    q=implied_probabilities(tf_odds)
    h1,_,_=positional_support(q,cars)
    ranked=sorted(cars,key=lambda c:(-h1[c],c)); head=ranked[0]
    H=[h1[c] for c in cars]
    h_ent=entropy(H)/math.log(7)
    tail=[v for t,v in q.items() if t[0]==head]
    s=sum(tail)
    tail_ent=entropy([v/s for v in tail])/math.log(30) if s>0 else None
    trio_p=implied_probabilities(trio_odds)
    trio_ent=entropy(list(trio_p.values()))/math.log(35)
    return {'head':head,'head_share':h1[head],'head_entropy':h_ent,'tail_entropy_given_head':tail_ent,'trio_entropy':trio_ent,'head_tail_contrast':(tail_ent-h_ent) if tail_ent is not None else None}

def valid(data):
    races,trio,tf,pay=data;out=[]
    for rid,r in races.items():
        if r.get('meeting_grade')!='F1' or not (r.get('race_type') or '').startswith('Ｓ級'):continue
        if pi(r.get('entry_count'))!=7 or len(trio.get(rid,{}))!=35 or len(tf.get(rid,{}))!=210:continue
        cars=sorted({v for c in trio[rid] for v in c});lines=pl(r.get('predicted_line_formation'))
        if len(cars)!=7 or lines is None or set(v for line in lines for v in line)!=set(cars) or rid not in pay or not pay[rid]:continue
        out.append((rid,r,cars))
    return out

def summary(rows):
    n=len(rows);h=sum(r['hit'] for r in rows);t=sum(r['tickets'] for r in rows);p=sum(r['payout'] for r in rows);s=t*STAKE
    return {'bet_races':n,'hits':h,'hit_rate_pct':100*h/n if n else None,'tickets':t,'avg_tickets':t/n if n else None,'stake_yen':s,'payout_yen':p,'profit_yen':p-s,'roi_pct':100*p/s if s else None}

def analyze(label,data):
    races,trio,tf,pay=data;vals=valid(data);mm={};byday=defaultdict(list)
    for rid,r,cars in vals:
        m=metrics(trio[rid],tf[rid],r.get('predicted_line_formation') or '',cars)
        if m:
            mm[rid]=m;byday[(r.get('race_date'),r.get('track'))].append(rid)
    rows=[]
    for rid,r,cars in vals:
        if (r.get('race_type') or '')!='Ｓ級決勝':continue
        d=build_v8_20_f21(trio[rid],tf[rid],r.get('predicted_line_formation') or '','Ｓ級決勝')
        if not d.get('buy') or rid not in mm:continue
        peers=[x for x in byday[(r.get('race_date'),r.get('track'))] if x!=rid and x in mm]
        if not peers:continue
        def med(k):
            xs=[mm[x][k] for x in peers if mm[x].get(k) is not None]
            return median(xs) if xs else None
        m=mm[rid]; he=med('head_entropy');te=med('tail_entropy_given_head');ce=med('head_tail_contrast');trie=med('trio_entropy')
        wins=[t for t in d['tickets'] if t in pay[rid]];payout=sum(pay[rid][t] for t in wins)
        row={'race_id':rid,'race_date':r.get('race_date'),'track':r.get('track'),'hit':int(bool(wins)),'payout':payout,'tickets':int(d['ticket_count']),'formation':d.get('formation'),'head_entropy':m['head_entropy'],'tail_entropy':m['tail_entropy_given_head'],'trio_entropy':m['trio_entropy'],'contrast':m['head_tail_contrast'],'head_entropy_vs_peer':m['head_entropy']/he if he else None,'tail_entropy_vs_peer':m['tail_entropy_given_head']/te if te else None,'trio_entropy_vs_peer':m['trio_entropy']/trie if trie else None,'contrast_vs_peer':m['head_tail_contrast']/ce if ce and ce>0 else None}
        row['head_sharper_than_peer']=int(he is not None and m['head_entropy']<=he)
        row['tail_more_diffuse_than_peer']=int(te is not None and m['tail_entropy_given_head']>=te)
        row['trio_more_diffuse_than_peer']=int(trie is not None and m['trio_entropy']>=trie)
        row['sharp_head_diffuse_tail']=int(row['head_sharper_than_peer'] and row['tail_more_diffuse_than_peer'])
        row['sharp_head_diffuse_trio']=int(row['head_sharper_than_peer'] and row['trio_more_diffuse_than_peer'])
        row['tail_and_trio_diffuse']=int(row['tail_more_diffuse_than_peer'] and row['trio_more_diffuse_than_peer'])
        rows.append(row)
    states={'ALL':summary(rows)}
    for f in ('head_sharper_than_peer','tail_more_diffuse_than_peer','trio_more_diffuse_than_peer','sharp_head_diffuse_tail','sharp_head_diffuse_trio','tail_and_trio_diffuse'):
        states[f.upper()]=summary([r for r in rows if r[f]])
    return {'dataset':label,'states':states,'rows':rows}

def main():
    result={'analysis':'FINAL_HEAD_TAIL_HEAT_Q1Q2','status':'DEVELOPMENT_DIAGNOSTIC','concept':'Current FINAL already requires a concentrated head. Test whether value exists when head certainty is sharper than same-day PS_AB peers while conditional 2nd/3rd order distribution remains more diffuse.','semantic_boundaries':'same-track/date PS_AB peer median only; no outcome-fitted numeric cutoff','Q1':analyze('2024Q1_DEVELOPMENT',load()),'Q2':analyze('2024Q2_DEVELOPMENT_AFTER_OBSERVED',load_q2())}
    print('FINAL_HEAD_TAIL_HEAT_BEGIN');print(json.dumps(result,ensure_ascii=False,indent=2));print('FINAL_HEAD_TAIL_HEAT_END')
if __name__=='__main__':main()
