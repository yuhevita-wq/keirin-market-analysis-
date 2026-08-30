from __future__ import annotations
import csv,itertools,json
from collections import defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/'data'/'2023'/'s_class_yosen'
AUDITS=ROOT/'data'/'audits'
GATE=AUDITS/'fake_favorite_true_middle_2023_2024.json'
OUT=AUDITS/'market_support_134_quinella_box_2023.json'
STAKE=100

def read_csv(p):
    with p.open('r',encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def norm3(s):
    return tuple(sorted(int(x) for x in s.replace('=','-').replace(',','-').split('-') if x.strip()))
def load_gate():
    d=json.loads(GATE.read_text(encoding='utf-8')); rows=d['years']['2023']['races']; g={str(r['race_id']):r for r in rows}; assert len(g)==208; return g
def load_trio():
    rows=read_csv(DATA/'trio_final_odds.csv'); out=defaultdict(dict)
    for r in rows: out[str(r['race_id'])][norm3(r['combination'])]=float(r['odds'])
    return out
def load_quinella_payouts():
    rows=read_csv(DATA/'payouts.csv'); out=defaultdict(list)
    for r in rows:
        if r.get('ticket_type')=='2車複' and r.get('status')=='paid':
            combo=tuple(sorted(int(x) for x in r['combination'].replace('=','-').split('-')))
            out[str(r['race_id'])].append((combo,int(r['payout_yen'])))
    return out
def market_ranks(odds):
    cars=sorted({c for t in odds for c in t}); support={c:sum(1/o for t,o in odds.items() if c in t and o>0) for c in cars}
    return sorted(cars,key=lambda c:(-support[c],c)),support
def main():
    gate=load_gate(); trio=load_trio(); qp=load_quinella_payouts();
    stake=payout=hits=0; loss=mx=0; hit_payouts=[]; examples=[]
    for rid in sorted(gate):
        ranks,support=market_ranks(trio[rid]); r1,r3,r4=ranks[0],ranks[2],ranks[3]
        tickets={tuple(sorted(x)) for x in itertools.combinations((r1,r3,r4),2)}
        race_payout=sum(pay for combo,pay in qp.get(rid,[]) if combo in tickets)
        stake += STAKE*len(tickets); payout += race_payout
        hit=race_payout>0
        if hit:
            hits+=1; loss=0; hit_payouts.append(race_payout)
        else:
            loss+=1; mx=max(mx,loss)
        if len(examples)<10: examples.append({'race_id':rid,'market_rank_1':r1,'market_rank_3':r3,'market_rank_4':r4,'tickets':[list(x) for x in sorted(tickets)],'winning_quinella_rows':[{'combo':list(c),'payout_yen':p} for c,p in qp.get(rid,[])],'hit':hit,'race_payout_yen':race_payout})
    hit_payouts_sorted=sorted(hit_payouts)
    med=None
    if hit_payouts_sorted:
        n=len(hit_payouts_sorted); med=hit_payouts_sorted[n//2] if n%2 else (hit_payouts_sorted[n//2-1]+hit_payouts_sorted[n//2])/2
    result={'bet_races':len(gate),'tickets':len(gate)*3,'avg_tickets_per_race':3.0,'hit_races':hits,'race_hit_rate_pct':100*hits/len(gate),'stake_yen':stake,'payout_yen':payout,'profit_yen':payout-stake,'roi_pct':100*payout/stake,'max_losing_streak':mx,'mean_hit_payout_yen':sum(hit_payouts)/len(hit_payouts) if hit_payouts else None,'median_hit_payout_yen':med,'min_hit_payout_yen':min(hit_payouts) if hit_payouts else None,'max_hit_payout_yen':max(hit_payouts) if hit_payouts else None}
    out={'status':'MARKET_SUPPORT_134_QUINELLA_BOX_2023','year':2023,'years_read':[2023],'population':'Audited 208 fake-favorite races only.','market_support_definition':'For each rider, sum 1/trio_final_odds across every valid trio containing that rider; rank descending, lower car number only breaks exact ties.','strategy':'Use market support ranks 1,3,4 as a 3-car quinella BOX: pairs 1-3, 1-4, 3-4; exactly 3 tickets/race.','payout_method':'Actual published 2車複 payout_yen from payouts.csv; 100 yen per ticket. Multiple paid quinella rows in dead-heat cases are all credited if selected.','result':result,'examples_first_10':examples,'note':'2023 development simulation only. No 2024/2025/2026 data read.'}
    OUT.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
if __name__=='__main__': main()
