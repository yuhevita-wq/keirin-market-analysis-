"""Frozen 2023 LINE-aware RF -> genuine 2024 Q1 final-odds ticket P&L.

Primary: paper-style P_model>1/decimal_odds for every trifecta/trio,
fixed JPY 100 / ticket (unlimited theoretical working capital).
Secondary: paper's 1/8 Kelly *single-bet formula*, with bankroll feasibility
normalization. This is only an illustrative extension to correlated keirin
portfolios, NOT the original NBA paper's portfolio model.

2024 labels are strictly evaluation only; NO model fitting or tuning.
"""
from __future__ import annotations
import csv
import gzip
import hashlib
import io
import json
import math
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np

from compare_lines_2023 import parse_line,lines_lookup,feature,normalise
from evaluate_2024_q1 import sources, START,END
from backtest_roi_2024_q1 import decode_odds,collect_2024_books
from train_2023 import FEATURES,fetch
from validate_tickets_2024_q1 import load_models as old_load_models
import validate_tickets_2024_q1 as old_inference

OUT=Path('docs/paper-only-ai-2023/line_roi_2024_q1')
LINE_DIR=Path(os.environ.get('FROZEN_LINE_FULL_DIR','frozen_line_full'))
NO_LINE_DIR=Path(os.environ.get('FROZEN_TICKET_MODEL_DIR','frozen_conditional_bundle'))
LABELS='data/result_universe_v2/combined/results_2024.jsonl.gz'
KINDS=('3連単','3連複')
VARIANTS=('line_rf_2023','old_no_line_2023')
STAKE=100
KELLY_INITIAL_BANKROLL_YEN=1_000_000
KELLY_FRACTION=0.125
BATCH=30
EPS=1e-12

def load_2024_line_cards():
    orders={}
    with gzip.open(io.BytesIO(fetch(LABELS)),'rt',encoding='utf-8') as f:
        for line in f:
            row=json.loads(line)
            top=row.get('finish_order')
            if isinstance(top,list) and len(top)>=3:
                tup=tuple(str(x) for x in top[:3])
                if len(set(tup))==3 and all(k.isdigit() for k in tup):
                    orders[str(row['race_id'])]=tup
    metadata={}
    entries={}
    files=[]
    audit=Counter()
    for path in sources():
        mp=path.replace('/entries.csv','/races.csv')
        lines=0; card_rows=0
        data=fetch(mp).decode('utf-8-sig')
        for row in csv.DictReader(io.StringIO(data)):
            day=str(row.get('race_date','')).strip()
            rid=str(row.get('race_id','')).strip()
            if not START<=day<=END or rid not in orders:
                continue
            if row.get('line_status')!='published' or not str(row.get('predicted_line_formation') or '').strip():
                audit['race_manifest_line_missing']+=1
                continue
            if rid in metadata:
                audit['duplicate_manifest_rows']+=1
                continue
            metadata[rid]={'date':day,
                'line':str(row['predicted_line_formation']),
                'source':('A1_A2' if '/a_class_12/' in path else 'A3' if '/a3_challenge/' in path else 'S')}
            lines+=1
        for row in csv.DictReader(io.StringIO(fetch(path).decode('utf-8-sig'))):
            day=str(row.get('race_date','')).strip()
            rid=str(row.get('race_id','')).strip()
            if not START<=day<=END or rid not in metadata:continue
            car=str(row.get('car_no','')).strip()
            try:
                arr=np.array([float(row[k]) for k in FEATURES],dtype=np.float32)
                if not car.isdigit() or not np.isfinite(arr).all():
                    raise ValueError('missing nonfinite or malformed')
            except (ValueError,KeyError,TypeError):
                audit['entry_rows_missing_numeric']+=1
                continue
            card=entries.setdefault(rid,{})
            if car in card:
                audit['duplicate_entry_rows']+=1
                continue
            card[car]=arr
            card_rows+=1
        files.append({'path':path,'race_manifests_with_line':lines,'entry_rows':card_rows})
        print('INPUT',path,lines,card_rows,flush=True)
    valid={}
    for rid,m in metadata.items():
        cars=entries.get(rid,{})
        if len(cars)<4:
            audit['exclude_card_too_small']+=1
            continue
        if not all(c in cars for c in orders[rid]):
            audit['exclude_podium_not_in_card']+=1
            continue
        groups=parse_line(m['line'],cars.keys())
        if groups is None:
            audit['exclude_line_coverage_mismatch']+=1
            continue
        valid[rid]={
            'date':m['date'],'source':m['source'],
            'cars':cars,'riders':cars,'first3':orders[rid],
            'order':orders[rid],'groups':groups,
            'positions':lines_lookup(groups)}
    valid=dict(sorted(valid.items(),key=lambda x:(x[1]['date'],x[0])))
    if not valid:
        raise RuntimeError('No 2024 Q1 line cards available')
    return valid,{'files':files,'exclusions':dict(audit),
                 'complete_podium_with_line_races':len(valid),
                 'months':dict(Counter(v['date'][:7] for v in valid.values())),
                 'categories':dict(Counter(v['source'] for v in valid.values()))}

def load_line_models():
    directory=LINE_DIR/'docs/paper-only-ai-2023/line_full_2023'
    fallback=LINE_DIR
    models={}
    sha={}
    for stage in (1,2,3):
        path=directory/f'stage_{stage}.joblib'
        if not path.is_file():path=fallback/f'stage_{stage}.joblib'
        if not path.is_file():raise RuntimeError('Frozen LINE model missing: '+str(path))
        data=path.read_bytes()
        sha[str(stage)]=hashlib.sha256(data).hexdigest()
        model=joblib.load(path)
        if model.n_features_in_!={1:22,2:65,3:110}[stage]:
            raise RuntimeError('Unexpected frozen LINE model feature count')
        models[stage]=model
    manifest=directory/'manifest.json'
    if not manifest.is_file():manifest=fallback/'manifest.json'
    if not manifest.is_file():raise RuntimeError('Frozen model manifest missing')
    info=json.loads(manifest.read_text(encoding='utf-8'))
    if info.get('training_date_range')!=['2023-01-01','2023-12-31']:
        raise RuntimeError('Unexpected training period')
    for stage in (1,2,3):
        expected=info['stages'][str(stage)]['sha256']
        if sha[str(stage)]!=expected:raise RuntimeError('Frozen model SHA mismatch')
    print('MODEL_VERIFIED',sha,'train races',info['trained_races'],flush=True)
    return models,info,sha

def infer_new_chunk(chunk, models, temperatures=None):
    groups={1:[],2:[],3:[]}
    keys={1:[],2:[],3:[]}
    vectors={1:[],2:[],3:[]}
    for rid,race in chunk:
        cars=sorted(race['cars'],key=int)
        for a in cars:
            keys[1].append((rid,a))
            vectors[1].append(feature(race,a,use_line=True))
            groups[1].append(rid)
        for a in cars:
            for b in cars:
                if b==a:continue
                keys[2].append((rid,a,b))
                vectors[2].append(feature(race,b,a,use_line=True))
                groups[2].append((rid,a))
                for c in cars:
                    if c==a or c==b:continue
                    keys[3].append((rid,a,b,c))
                    vectors[3].append(feature(race,c,a,b,use_line=True))
                    groups[3].append((rid,a,b))
    distributions={}
    for stage in (1,2,3):
        scores=models[stage].predict_proba(np.asarray(vectors[stage],dtype=np.float32))[:,1]
        if temperatures is not None:
            temp=float(temperatures[str(stage)])
            if not math.isfinite(temp) or not 0.5<=temp<=2.5:
                raise ValueError('Invalid frozen calibration temperature')
            scores=np.exp(np.log(np.clip(scores,EPS,1.0))/temp)
        dictionary={}
        gids=[]
        for gid in groups[stage]:
            if gid not in dictionary:dictionary[gid]=len(dictionary)
            gids.append(dictionary[gid])
        p=normalise(scores,gids)
        distributions[stage]=p
    pp1=dict(zip(keys[1],distributions[1]))
    pp2=dict(zip(keys[2],distributions[2]))
    output=defaultdict(lambda:{'3連単':{},'3連複':defaultdict(float)})
    for (rid,a,b,c),p3 in zip(keys[3],distributions[3]):
        p=float(pp1[(rid,a)]*pp2[(rid,a,b)]*p3)
        output[rid]['3連単'][(a,b,c)]=p
        trio=tuple(sorted((a,b,c),key=int))
        output[rid]['3連複'][trio]+=p
    for rid,_ in chunk:
        for kind in KINDS:
            if abs(math.fsum(output[rid][kind].values())-1)>1e-8:
                raise RuntimeError('Normalization integrity failure')
    return output

def actual_tickets(race):
    a,b,c=race['order']
    return {'3連単':(a,b,c),'3連複':tuple(sorted((a,b,c),key=int))}

class Ledger:
    def __init__(self):
        self.races=0
        self.bet_races=0
        self.win_races=0
        self.tickets=0
        self.hits=0
        self.invest=0.0
        self.paid=0.0
        self.expected=0.0
        self.nll=0.0
        self.market_nll=0.0
        self.sum_brier=0.0
        self.top1=0
        self.top3=0
        self.race_loss_count=0
        self.balance=0.0
        self.peak=0.0
        self.max_drawdown=0.0
    def update(self,bets,actual,price_book,all_prob):
        cost=STAKE*len(bets)
        hit=1 if actual in bets else 0
        return_yen=STAKE*price_book[actual] if hit else 0.0
        expect=sum(STAKE*p*odd for _,p,odd in bets.values())
        self.races+=1
        self.bet_races+=int(cost>0)
        self.win_races+=int(hit>0)
        self.tickets+=len(bets)
        self.hits+=hit
        self.invest+=cost
        self.paid+=return_yen
        self.expected+=expect
        self.balance+=return_yen-cost
        self.peak=max(self.peak,self.balance)
        self.max_drawdown=max(self.max_drawdown,self.peak-self.balance)
        self.race_loss_count+=int(return_yen<cost)
        if all_prob is not None:
            prices=np.array(list(all_prob.values()),dtype=float)
            true=float(all_prob[actual])
            self.nll-=math.log(max(true,EPS))
            market_norm=math.fsum(1/p for p in price_book.values())
            mk=(1/price_book[actual])/market_norm
            self.market_nll-=math.log(max(mk,EPS))
            self.sum_brier+=float(1-2*true+np.sum(np.square(prices)))
            ranked=sorted(all_prob,key=lambda k:(-all_prob[k],tuple(int(x) for x in k)))
            self.top1+=int(actual==ranked[0])
            self.top3+=int(actual in ranked[:3])
        return cost,return_yen
    def summary(self,prob_metrics=True):
        n=self.races
        s=self.invest
        return {
         'eligible_races':n,'betting_races':self.bet_races,'winning_races':self.win_races,
         'purchased_tickets':self.tickets,'winning_tickets':self.hits,
         'investment_yen':round(s,2),'payout_yen':round(self.paid,2),
         'net_profit_yen':round(self.paid-s,2),
         'return_rate_percent':100*self.paid/s if s else None,
         'roi_percent':100*(self.paid-s)/s if s else None,
         'model_expected_roi_percent':100*(self.expected-s)/s if s else None,
         'max_drawdown_yen_unlimited_flat_stake':round(self.max_drawdown,2),
         'losing_races':self.race_loss_count,
         **({'nll':self.nll/n,'normalized_market_nll':self.market_nll/n,
             'proper_multiclass_brier':self.sum_brier/n,
             'most_probable_ticket_hit_percent':100*self.top1/n,
             'top3_tickets_hit_percent':100*self.top3/n} if prob_metrics and n else {})}

class KellyLedger:
    """Cash-limited separate bankroll (both/each kind); naive fractional Kelly per ticket.

    For mutually exclusive outcomes WITHIN one market fractions are still from
    isolated binary bets. Across 3連単 and 3連複 ticket returns are correlated.
    A within-race cash constraint is applied; NOT an optimal joint Kelly solution.
    """
    def __init__(self):
        self.bank=float(KELLY_INITIAL_BANKROLL_YEN)
        self.start=float(KELLY_INITIAL_BANKROLL_YEN)
        self.invest=0.0
        self.paid=0.0
        self.bets=0
        self.hits=0
        self.active_races=0
        self.races=0
        self.peak=self.bank
        self.drawdown=0.0
        self.cap_races=0
        self.below_min_stake=0
    def step(self, raw):
        self.races+=1
        if self.bank<STAKE:
            self.below_min_stake+=1
            return
        desires=[]
        for ticket,price,p,win in raw:
            fraction=max(0.0, (p*price-1.0)/(price-1.0))*KELLY_FRACTION
            if fraction>0:
                desires.append((ticket,price,p,win,fraction))
        frac_total=sum(x[4] for x in desires)
        if frac_total>KELLY_FRACTION*8:
            self.cap_races+=1
        # Never stake more than currently owned capital.
        scale=min(1.0,1.0/frac_total) if frac_total else 1.0
        cost=0.0
        paid=0.0
        hit=0
        n=0
        for ticket,price,p,win,fraction in desires:
            stake=STAKE*math.floor(self.bank*fraction*scale/STAKE+1e-10)
            if stake<STAKE:continue
            cost+=stake
            n+=1
            if win:
                hit+=1
                paid+=stake*price
        if cost>self.bank+1e-4:
            raise RuntimeError('Simulated Kelly bet exposure exceeds bankroll')
        self.bank+=paid-cost
        if self.bank<0 and self.bank>-1e-5:self.bank=0.0
        if self.bank<-1e-5:
            raise RuntimeError('Negative capital after payout settlement')
        self.invest+=cost
        self.paid+=paid
        self.bets+=n
        self.hits+=hit
        self.active_races+=int(cost>0)
        self.peak=max(self.peak,self.bank)
        self.drawdown=max(self.drawdown,self.peak-self.bank)
    def summary(self):
        return {'initial_bankroll_yen':self.start,
                'ending_bankroll_yen':round(self.bank,2),
                'net_profit_yen':round(self.bank-self.start,2),
                'roi_on_staked_yen_percent':100*(self.paid-self.invest)/self.invest if self.invest else None,
                'return_on_initial_capital_percent':100*(self.bank-self.start)/self.start,
                'investment_yen':round(self.invest,2),
                'payout_yen':round(self.paid,2),
                'purchased_tickets':self.bets,'winning_tickets':self.hits,
                'betting_races':self.active_races,'eligible_races':self.races,
                'exposure_scaled_to_cash_races':self.cap_races,
                'races_with_bankroll_below_min_bet':self.below_min_stake,
                'max_drawdown_yen':round(self.drawdown,2)}

def safe_records(distributions,books):
    for kind in KINDS:
        p=distributions[kind]
        o=books[kind]
        if set(p)!=set(o):
            raise RuntimeError('Model and price combinations differ')
        if abs(math.fsum(p.values())-1.0)>1e-8:
            raise RuntimeError('Probability book does not sum to one')
    return True

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    began=datetime.now(timezone.utc).isoformat()
    (OUT/'progress.json').write_text(json.dumps({
        'status':'RUNNING','started_at':began,'phase':'load frozen 2023 models'
        },ensure_ascii=False)+'\n')
    new_models,manifest,hashes=load_line_models()
    old_inference.BUNDLE=NO_LINE_DIR
    old_models=old_load_models()
    cards,entry_audit=load_2024_line_cards()
    odds,odds_audit=collect_2024_books()
    common=set(cards)&set(odds)
    ordered=[(rid,cards[rid]) for rid in sorted(common,key=lambda i:(cards[i]['date'],i))]
    audit=Counter()
    example=[]
    ledgers={b:{k:Ledger() for k in KINDS+('combined',)} for b in VARIANTS}
    class_ledgers={b:{c:{k:Ledger() for k in KINDS} for c in ('A1_A2','A3','S')} for b in VARIANTS}
    month_ledgers={b:{m:{k:Ledger() for k in KINDS} for m in ('2024-01','2024-02','2024-03')} for b in VARIANTS}
    kelly={b:{t:KellyLedger() for t in ('3連単','3連複','combined')} for b in VARIANTS}
    qualified=0
    sample_details=[]
    for offset in range(0,len(ordered),BATCH):
        candidates=[]
        pricebooks={}
        for rid,race in ordered[offset:offset+BATCH]:
            try:
                pricebooks[rid]=decode_odds(odds[rid],rid,set(race['cars']))
            except ValueError as e:
                audit['excluded_odds_'+str(e)]+=1
                if len(example)<15:example.append({'race_id':rid,'reason':str(e)})
                continue
            candidates.append((rid,race))
        if not candidates:continue
        predictions={
            'line_rf_2023':infer_new_chunk(candidates,new_models),
        }
        old_inference.set_group2=[]
        old_inference.set_group3=[]
        predictions['old_no_line_2023']=old_inference.infer_batch(candidates,old_models)
        for rid,race in candidates:
            books=pricebooks[rid]
            results=actual_tickets(race)
            qualified+=1
            for variant in VARIANTS:
                safe_records(predictions[variant][rid],books)
                pool=[]
                for kind in KINDS:
                    dist=predictions[variant][rid][kind]
                    prices=books[kind]
                    true=results[kind]
                    if true not in dist:raise RuntimeError('True finish outcome not covered')
                    bets={}
                    kraw=[]
                    for ticket,p in dist.items():
                        price=prices[ticket]
                        if p*price>1.0:
                            bets[ticket]=(kind,p,price)
                            kraw.append(((kind,ticket),price,p,ticket==true))
                    ledgers[variant][kind].update(bets,true,prices,dist)
                    class_ledgers[variant][race['source']][kind].update(bets,true,prices,dist)
                    month_ledgers[variant][race['date'][:7]][kind].update(bets,true,prices,dist)
                    kelly[variant][kind].step(kraw)
                    pool.extend(kraw)
                # Combine across correlated markets, preserve same-race bankroll settlement.
                # Combined flat-stake ledger is formed from the two market snapshots above.
                # For total net only, use synthetic keyed outcomes to prevent key collisions.
                combined_bets={}
                combined_prices={}
                for kind in KINDS:
                    true=results[kind]
                    dist=predictions[variant][rid][kind]
                    price=books[kind]
                    for ticket,p in dist.items():
                        combined_prices[(kind,ticket)]=price[ticket]
                        if p*price[ticket]>1:
                            combined_bets[(kind,ticket)]=(kind,p,price[ticket])
                # Two tickets can win on the same race. Pass combined payouts separately.
                current=ledgers[variant]['combined']
                cost=STAKE*len(combined_bets)
                winners=[item for item in combined_bets if item[1]==results[item[0]]]
                payout=STAKE*sum(combined_prices[item] for item in winners)
                expected=STAKE*sum(p*odd for _,p,odd in combined_bets.values())
                current.races+=1
                current.bet_races+=int(cost>0)
                current.win_races+=int(bool(winners))
                current.tickets+=len(combined_bets)
                current.hits+=len(winners)
                current.invest+=cost
                current.paid+=payout
                current.expected+=expected
                current.balance+=payout-cost
                current.peak=max(current.peak,current.balance)
                current.max_drawdown=max(current.max_drawdown,current.peak-current.balance)
                current.race_loss_count+=int(payout<cost)
                kelly[variant]['combined'].step(pool)
            if len(sample_details)<8:
                sample_details.append({'race_id':rid,'date':race['date'],'class':race['source'],
                    'line':[[*group] for group in race['groups']],
                    'results':list(race['order']),
                    'odds_matched':True})
        if qualified%300<BATCH or offset+BATCH>=len(ordered):
            print('VALIDATED',qualified,'/',len(ordered),'line-qualified odds-complete',flush=True)
            (OUT/'progress.json').write_text(json.dumps({
                'status':'RUNNING','started_at':began,
                'processed_aligned_races':qualified,
                'intersection_before_book_checks':len(ordered),
                '2024_fit_calls':0},ensure_ascii=False)+'\n')
    if qualified==0:raise RuntimeError('No 2024 Q1 result + line + complete-odds matched races')
    summary={v:{k:ledgers[v][k].summary(prob_metrics=(k!='combined')) for k in KINDS+('combined',)} for v in VARIANTS}
    breakdown_class={v:{cat:{k:class_ledgers[v][cat][k].summary() for k in KINDS}
                  for cat in ('A1_A2','A3','S')} for v in VARIANTS}
    breakdown_month={v:{month:{k:month_ledgers[v][month][k].summary() for k in KINDS}
                  for month in ('2024-01','2024-02','2024-03')} for v in VARIANTS}
    paired={}
    for kind in KINDS+('combined',):
        line=summary['line_rf_2023'][kind]
        old=summary['old_no_line_2023'][kind]
        paired[kind]={
            'same_race_set':True,
            'line_minus_no_line_roi_percentage_points':
              line['roi_percent']-old['roi_percent'] if line['roi_percent'] is not None and old['roi_percent'] is not None else None,
            'line_minus_no_line_net_yen':line['net_profit_yen']-old['net_profit_yen'],
            **({'line_minus_no_line_nll':line['nll']-old['nll'],
                'line_minus_no_line_top1_percentage_points':
                  line['most_probable_ticket_hit_percent']-old['most_probable_ticket_hit_percent']}
                if kind!='combined' else {})}
    complete=datetime.now(timezone.utc).isoformat()
    report={
       'status':'COMPLETE_FROZEN_LINE_RF_FINAL_ODDS_2024_Q1',
       'started_at':began,
       'finished_at':complete,
       'train_dates':['2023-01-01','2023-12-31'],
       'test_dates':[START,END],
       '2023_line_full_training_run':39376331186,
       '2023_line_model_sha256':hashes,
       'trained_line_races':manifest['trained_races'],
       '2024_fit_calls':0,
       '2024_model_selection_calls':0,
       'training_model_is_unchanged':True,
       'selection_rule':'p_model*decimal_final_odds > 1, all eligible tickets',
       'odds_source':'data/odds_only/odds_2024.jsonl.gz',
       'odds_are_final':True,
       'real_time_executable_claim':False,
       'ticket_markets':['3連単','3連複'],
       'primary_fixed_stake_yen_per_bet':STAKE,
       'primary_unlimited_working_capital':True,
       'odds_join':{
            'all_2024_q1_line_complete_card_races':len(cards),
            'all_2024_full_year_odds_records':len(odds),
            'q1_line_entry_and_odds_joined_ids_before_price_checks':len(common),
            'q1_no_complete_odds_races':len(cards)-len(common),
            'price_exclusions':dict(audit),
            'aligned_evaluation_races':qualified,
            'excluded_examples':example,
            'card_source_audit':entry_audit,
            'market_source_audit':odds_audit},
       'primary_fixed_100yen':summary,
       'paired_same_races':paired,
       'by_category':breakdown_class,
       'by_month':breakdown_month,
       'one_eighth_kelly_simulation':{
          'status':'illustrative extension, NOT original paper optimal overlapping ticket portfolio',
          'initial_bankroll_yen':KELLY_INITIAL_BANKROLL_YEN,
          'individual_bet_formula':'max(0, (p*odds-1)/(odds-1))/8 times current bankroll; floor to 100 yen',
          'portfolio_feasibility':'same-race sum of fractions rescaled to <=100% of available bankroll, then 100yen floor; for each market and both combined separately',
          'capital_reinvested_after_each_race':True,
          'results':{v:{k:kelly[v][k].summary() for k in KINDS+('combined',)} for v in VARIANTS}
       },
       'sample_validated_races':sample_details,
       'interpretation_limits':[
          'Historical confirmed *final* odds, NOT 15-minutes-before-closure odds; this is a retrospective hypothetical backtest.',
          '2024 labels were used only to compute payoffs and predictive scores; not as features, for training, for threshold selection, or any post-hoc modifications.',
          'Not every 2024 race has completely matched line, rider cards, top-3 result, and two full 3-combination markets. Missing races are excluded and counts reported.',
          'LINE published formation is KDreams source prediction, not known realized in-race order.',
          'The old no-LINE baseline uses another 2023 training design/hyperparameters and is compared on IDENTICAL 2024 races, but is not a pure-line-feature controlled causal ablation.',
          'The LINE RF model was chosen based on 2023 November-December and refitted on ALL 2023; the 2024 period remains new for the LINE model.',
          'Kelly per-bet fractions are not jointly optimal for correlated and mutually exclusive multi-ticket outcomes; the same-race cash constraint is a transparent engineering adaptation, not found in the NBA paper.',
          'Minimum 100yen stake and no transaction/price impact limit; these outcomes do not prove live bookable profitability.',
          'Neither previous no-LINE model nor selected LINE models are certified by original Walsh-Joshi strict 20-bin classwise-ECE gate for every ticket.'
       ]}
    (OUT/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (OUT/'progress.json').write_text(json.dumps({'status':report['status'],'completed_at':complete,
        'races':qualified,'2024_fit_calls':0},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('MAIN_FIXED_STAKE_RESULTS',json.dumps(summary,ensure_ascii=False),flush=True)
    print('PAIRED',json.dumps(paired,ensure_ascii=False),flush=True)
    print('KELLY',json.dumps(report['one_eighth_kelly_simulation']['results'],ensure_ascii=False),flush=True)
    print('COMPLETE',qualified,flush=True)

if __name__=='__main__':
    main()
