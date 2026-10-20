"""2024 Q1 genuine historical odds / 2023 frozen AI financial experiment.

Methodology: paper's value-betting criterion p_model > 1/decimal_odds
as EV = p_model*final_odds-1 > 0, applied to *all* trifecta and trio
combinations without an arbitrary pick count limit, with fixed JPY100 stakes.
Final odds are retrospective: this is NOT a live-executable pre-close result.

All 2024 results are used only to score selections; no fitting or tuning.
"""
from __future__ import annotations

import gzip
import io
import json
import math
from collections import Counter, defaultdict
from datetime import datetime,timezone
from pathlib import Path

import numpy as np

import validate_tickets_2024_q1 as validation
from train_2023 import fetch

OUTPUT=Path('docs/paper-only-ai-2023/roi_2024_q1')
ODDS='data/odds_only/odds_2024.jsonl.gz'
KINDS=('3連単','3連複')
START,END='2024-01-01','2024-03-31'
STAKE=100
SPLIT=validation.SPLIT
# 0.0 is THE preregistered main rule; further cutoffs solely display sensitivity.
CUTS=(0.0,0.05,0.1,0.2)


def decode_odds(data,rid,cars):
    """Parse odds book strictly, enforce complete identical runner combinations."""
    expected = {
      '3連単':len(cars)*(len(cars)-1)*(len(cars)-2),
      '3連複':math.comb(len(cars),3)}
    books={}
    for kind in KINDS:
        if not isinstance(data,dict) or not isinstance(data.get(kind),dict):
            raise ValueError('market_missing')
        bk={}
        for raw,value in data[kind].items():
            fields=raw.split('-' if kind=='3連単' else '=')
            if len(fields)!=3 or any(not x.isdigit() for x in fields):
                raise ValueError('invalid_combination_key')
            parts=tuple(fields)
            if len(set(parts))!=3:
                raise ValueError('repeated_car_in_ticket')
            if kind=='3連複':
                if parts!=tuple(sorted(parts,key=int)):
                    raise ValueError('unordered_ticket_not_canonical')
            if any(car not in cars for car in parts):
                raise ValueError('odds_car_not_in_source_entry')
            try:
                price=float(value)
            except (TypeError,ValueError):
                raise ValueError('invalid_odd')
            if not math.isfinite(price) or price<=1:
                raise ValueError('invalid_odd')
            if parts in bk:
                raise ValueError('duplicate_price')
            bk[parts]=price
        if len(bk)!=expected[kind]:
            raise ValueError('incomplete_odds_book')
        books[kind]=bk
    return books


def collect_2024_books():
    books={}
    audit=Counter()
    with gzip.open(io.BytesIO(fetch(ODDS)),'rt',encoding='utf-8') as handle:
        for line in handle:
            if not line.strip():
                continue
            row=json.loads(line)
            audit['full_year_odds_records']+=1
            rid=str(row['race_id'])
            if rid in books:
                raise RuntimeError('Duplicate race id in odds archive: '+rid)
            books[rid]=row.get('odds')
    return books,dict(audit)


def empty_acc():
    return {'races':set(), 'bets':0,'hit_tickets':0,
            'stake_yen':0, 'return_yen':0.0, 'expected_return_yen':0.0,
            'race_count_with_positive_ev':set(),
            'winning_races':set(),'per_race':defaultdict(lambda:[0.0,0.0])}


def add(acc,rid, bets,winning,return_yen,expected_yen):
    acc['races'].add(rid)
    acc['bets']+=bets
    acc['hit_tickets']+=winning
    stake=STAKE*bets
    acc['stake_yen']+=stake
    acc['return_yen']+=return_yen
    acc['expected_return_yen']+=expected_yen
    acc['per_race'][rid][0]+=stake
    acc['per_race'][rid][1]+=return_yen
    if bets>0:acc['race_count_with_positive_ev'].add(rid)
    if winning:acc['winning_races'].add(rid)


def stats(acc):
    invested=acc['stake_yen']
    ret=acc['return_yen']
    expected=acc['expected_return_yen']
    initial=0.0
    highwater=0.0
    maxdraw=0.0
    neg_races=0
    for rid in sorted(acc['per_race'],key=lambda rid:(race_dates[rid],rid)):
        cost,pay=acc['per_race'][rid]
        initial+=pay-cost
        highwater=max(highwater,initial)
        maxdraw=max(maxdraw,highwater-initial)
        if pay<cost:neg_races+=1
    return {
        'eligible_races':len(acc['races']),
        'betting_races':len(acc['race_count_with_positive_ev']),
        'winning_races':len(acc['winning_races']),
        'tickets_purchased':acc['bets'],
        'winning_tickets':acc['hit_tickets'],
        'hit_rate_per_bet':acc['hit_tickets']/acc['bets'] if acc['bets'] else None,
        'stake_yen':invested,
        'return_yen':round(ret,2),
        'net_yen':round(ret-invested,2),
        'roi':(ret-invested)/invested if invested else None,
        'return_rate':ret/invested if invested else None,
        'model_expected_net_yen':round(expected-invested,2),
        'model_expected_roi':(expected-invested)/invested if invested else None,
        'max_drawdown_yen_unconstrained_flat_stake':round(maxdraw,2),
        'losing_races':neg_races,
        'assumed_staking':'fixed JPY100 on each ticket with EV above threshold; unrestricted stake budget',
        'negative_bankroll_protection':False,
    }


race_dates={}


def main():
    OUTPUT.mkdir(parents=True,exist_ok=True)
    started=datetime.now(timezone.utc).isoformat()
    (OUTPUT/'progress.json').write_text(json.dumps({'status':'RUNNING','started_at':started},
                                                   ensure_ascii=False)+'\n')
    models=validation.load_models()
    races,source_audit=validation.cards_and_results()
    books,book_audit=collect_2024_books()
    all_race_ids=set(races)
    intersection=all_race_ids.intersection(books)
    missing_count=len(all_race_ids-intersection)
    ready={rid:races[rid] for rid in sorted(intersection,key=lambda rid:(races[rid]['date'],rid))}
    global race_dates
    race_dates={rid:r['date'] for rid,r in ready.items()}
    print('ODDS_JOIN',len(all_race_ids),len(books),len(ready),
          'missing',missing_count,flush=True)
    counters=Counter()
    examples=[]
    metrics={c:{k:empty_acc() for k in KINDS} for c in CUTS}
    category={k:{cat:empty_acc() for cat in ('A1_A2','A3','S')} for k in KINDS}
    month={k:{m:empty_acc() for m in ('2024-01','2024-02','2024-03')} for k in KINDS}
    model_nll={k:[] for k in KINDS}
    market_nll={k:[] for k in KINDS}
    overround={k:[] for k in KINDS}
    valid_races=0
    batches=list(ready.items())
    for idx in range(0,len(batches),SPLIT):
        chunk=[]
        decoded={}
        for rid,r in batches[idx:idx+SPLIT]:
            try:
                bk=decode_odds(books[rid],rid,set(r['riders']))
                decoded[rid]=bk
                chunk.append((rid,r))
            except ValueError as e:
                counters['excluded_'+str(e)]+=1
                if len(examples)<15:examples.append({'race_id':rid,'reason':str(e)})
        if not chunk:
            continue
        validation.set_group2=[]
        validation.set_group3=[]
        probs=validation.infer_batch(chunk,models)
        for rid,r in chunk:
            correct=validation.truth_for(r)
            valid_races+=1
            for kind in KINDS:
                pred=probs[rid][kind]
                final_odds=decoded[rid][kind]
                if set(pred)!=set(final_odds):
                    # Audit carefully: market mismatches should never become zero odds.
                    raise RuntimeError('Mismatch of full odds and model ticket outcome universe: '+rid+' '+kind)
                if correct[kind] not in final_odds:
                    raise RuntimeError('Real outcome missing from market: '+rid+' '+kind)
                true=correct[kind]
                pt=float(pred[true])
                market_inverse_sum=sum(1/x for x in final_odds.values())
                market_prob=(1/final_odds[true])/market_inverse_sum
                model_nll[kind].append(-math.log(max(pt,1e-12)))
                market_nll[kind].append(-math.log(max(market_prob,1e-12)))
                overround[kind].append(market_inverse_sum)
                cut_bets={c:0 for c in CUTS}
                cut_hits={c:0 for c in CUTS}
                cut_return={c:0.0 for c in CUTS}
                cut_expected={c:0.0 for c in CUTS}
                for ticket,p in pred.items():
                    odd=final_odds[ticket]
                    edge=p*odd-1
                    if not math.isfinite(edge):
                        raise RuntimeError('nonfinite edge')
                    for c in CUTS:
                        if edge>c:
                            cut_bets[c]+=1
                            cut_expected[c]+=STAKE*p*odd
                            if ticket==true:
                                cut_hits[c]+=1
                                cut_return[c]+=STAKE*odd
                for c in CUTS:
                    add(metrics[c][kind],rid,cut_bets[c],cut_hits[c],cut_return[c],cut_expected[c])
                # All ticket markets' main-rule outcomes separately by race category and month
                c=0.0
                add(category[kind][r['source']],rid,cut_bets[c],cut_hits[c],cut_return[c],cut_expected[c])
                add(month[kind][r['date'][:7]],rid,cut_bets[c],cut_hits[c],cut_return[c],cut_expected[c])
            if valid_races%200==0:
                print('PROCESSED',valid_races,'/',len(ready),flush=True)
                (OUTPUT/'progress.json').write_text(json.dumps({
                    'status':'RUNNING','started_at':started,
                    'processed_races':valid_races,'odds_join_races':len(ready)},
                    ensure_ascii=False)+'\n')
    if not valid_races:
        raise RuntimeError('No fully aligned odds/probability/result samples')
    main_stats={kind:stats(metrics[0.0][kind]) for kind in KINDS}
    both=empty_acc()
    for kind in KINDS:
        acc=metrics[0.0][kind]
        both['races'].update(acc['races'])
        both['bets']+=acc['bets']
        both['hit_tickets']+=acc['hit_tickets']
        both['stake_yen']+=acc['stake_yen']
        both['return_yen']+=acc['return_yen']
        both['expected_return_yen']+=acc['expected_return_yen']
        both['race_count_with_positive_ev'].update(acc['race_count_with_positive_ev'])
        both['winning_races'].update(acc['winning_races'])
        for rid,vr in acc['per_race'].items():
            both['per_race'][rid][0]+=vr[0]
            both['per_race'][rid][1]+=vr[1]
    nll={kind:{
            'avg_model_nll_realized_ticket':float(np.mean(model_nll[kind])),
            'avg_normalized_market_nll_realized_ticket':float(np.mean(market_nll[kind])),
            'mean_final_odds_market_overround':float(np.mean(overround[kind])),
            'market_normalization_method':'(1/decimal_odd)/sum_all_tickets(1/decimal_odd)'
         } for kind in KINDS}
    report={
        'status':'COMPLETED_HISTORICAL_FINAL_ODDS_BACKTEST',
        'completed_at':datetime.now(timezone.utc).isoformat(),
        'started_at':started,
        'period':[START,END],
        'training_data_end':'2023-12-31',
        'frozen_model_bundle_run':39362270286,
        '2024_model_fit_calls':0,
        '2024_threshold_fit_calls':0,
        'cutoff_selected_in_advance':0.0,
        'fixed_stake_yen':STAKE,
        'selection_rule':'For EVERY ticket in 3連単 and 3連複, include if (P_model*decimal_final_odds-1)>0. Strict greater-than.',
        'invested_unlimited_100yen_bets':True,
        'no_bet_if_all_tickets_nonpositive_ev':True,
        'odds_source':ODDS,
        'odds_phase':'final (retrospective, NOT 15-minute pre-close)',
        'pricing_and_execution_not_guaranteed':True,
        'included_ticket_types':list(KINDS),
        'excluded_ticket_types_without_verified_odds':['2車単','2車複'],
        'data_join':{'q1_full_podium_with_entries_races':len(races),
                     '2024_full_year_complete_odds_races':len(books),
                     'q1_races_with_both_sources':len(intersection),
                     'q1_no_odds_available_count':missing_count,
                     'full_odds_book_aligned_races':valid_races,
                     'q1_book_mismatch_reasons':dict(counters),
                     'mismatch_examples':examples,
                     '2024_card_audit':source_audit,
                     'odds_archive_audit':book_audit},
        'by_ticket_type':main_stats,
        'both_types_combined':stats(both),
        'sensitivity_only_not_tuned':{
             str(c):{k:stats(metrics[c][k]) for k in KINDS} for c in CUTS if c!=0.0},
        'by_class_main_cutoff':{k:{s:stats(v) for s,v in category[k].items()} for k in KINDS},
        'by_month_main_cutoff':{k:{m:stats(v) for m,v in month[k].items()} for k in KINDS},
        'probability_logloss_market_comparison':nll,
        'caveats':['The model has NOT passed the strict calibration gate for 2nd and 3rd conditional models or final-ticket odds in the 2023 holdout/2024 validation.',
                   'Even a historical positive net does not establish deployable positive EV. Final odds are unknowable 15 minutes before start and settlement may differ if scratched.',
                   'Flat-stake all-positive-EV strategy is NOT capital constrained: it can exceed available bankroll and stake many correlated bets in same race.',
                   'Only races with completely matching 2024 Q1 card, 1st-3rd result and two complete final-odds books enter. Missing price is never guessed.',
                   'The upper EV threshold tables are diagnostic only, not a strategy fitted or chosen from 2024 results.',
                   'Odds do not include transaction timing, cancellations, takeout changes, or slippage; this experiment should not be labeled real live performance.']
    }
    (OUTPUT/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (OUTPUT/'progress.json').write_text(json.dumps(
        {'status':report['status'],'completed_at':report['completed_at'],
         'races':valid_races,'2024_fit_calls':0},ensure_ascii=False,indent=2)+'\n')
    for kind in KINDS:
        print('ROI_RESULT',kind,json.dumps(main_stats[kind],ensure_ascii=False),flush=True)
    print('ROI_RESULT_COMBINED',json.dumps(report['both_types_combined'],ensure_ascii=False),flush=True)
    print('COMPLETED',len(races),len(intersection),valid_races,flush=True)


if __name__=='__main__': main()
